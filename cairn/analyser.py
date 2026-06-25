#!/usr/bin/env python3
"""Calibration session analyser — Phase 2.

Single LLM pass over a cleaned session transcript produces a sectioned
JSON output with 13 bounded dimensions (see docs/spec-calibration-system.md
Amendment 1). 8 dimensions write to `calibration_rows`; 5 write to the
existing `memories` table with `source_ref="analyser-session-arc"`. A
post-pass scores effectiveness on prior `calibration_deliveries`.

The LLM is invoked via `claude -p` subprocess with `CAIRN_MODE=read-only`
to bypass the Stop hook capture path (per memory 4002 + 885). Cron / idle
detection walks `~/.claude/projects/*/*.jsonl` and triggers analysis on
sessions idle ≥ N minutes that haven't been analysed yet.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err

from cairn import session_extract


DB_PATH = os.environ.get(
    "CAIRN_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db"),
)
EPH_DB_PATH = os.environ.get(
    "CAIRN_EPHEMERAL_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db"),
)

TRANSCRIPT_ROOTS = [os.path.expanduser("~/.claude/projects")]
DEFAULT_IDLE_MINUTES = 15
DEFAULT_TIMEOUT_S = 1800
DEFAULT_MODEL = os.environ.get("CAIRN_ANALYSER_MODEL", "claude-sonnet-4-6")

# Subagent / triviality filter. A session is worth analysing only if it has
# at least this many substantive turns (after signal-only cleaning) AND at
# least this much cleaned-text content. Most subagent and heartbeat
# transcripts fall well below both thresholds.
MIN_SUBSTANTIVE_TURNS = 4
MIN_CLEANED_CHARS = 500

# Incremental analysis. A session that has already been analysed becomes
# eligible again only when its turn count has grown by at least this many
# turns since the last analysis. Long-running sessions thus get re-analysed
# periodically without paying for every appended turn.
INCREMENTAL_TURN_THRESHOLD = 10
ANALYSER_STATE_KEY = "calibration_analyser_state"

ANALYSER_SOURCE_REF = "analyser-session-arc"

# Source-tier initial confidences (spec §"Source tiers"). Match the
# CLI defaults in cairn/calibration.py SOURCE_INITIAL_CONFIDENCE.
_SOURCE_INITIAL_CONFIDENCE = {
    "explicit": 0.90,
    "correction": 0.60,
    "observation": 0.15,
    "meta-assessment": 0.20,
}

# Output dimensions. Per cairn entry 623 ("hook enforces format
# mechanically, LLM enforces content quality") and Amendment 1, content
# quantity per dimension is editorial work — left to the LLM via prompt
# instruction rather than enforced as a numeric cap. The only structural
# guard is the envelope budget on total parsed output size, which exists
# because claude -p truncates above an empirically observed ceiling
# (cairn entry 1734).
DIMENSIONS = {
    # calibration_rows-bound
    "user_observations",
    "explicit_instructions",
    "approach_assessment",
    "contradictions",
    "drift_signals",
    "row_effectiveness",
    "tool_redirect_signals",
    "misalignment_reconvergence",
    # memories-bound
    "session_arc_memories",
    "decision_memories_with_alternatives",
    "tool_brittleness_patterns",
    "loose_ends",
    "confidence_calibration_audit",
}

# Envelope budget — hard upper bound on parsed output JSON length in
# characters. Sized below the ~25K-token Isosync envelope (cairn entry
# 1948) at ~4 chars/token. Output that exceeds this is logged via the
# analyser_envelope_exceeded metric; the analyser still processes what
# parsed successfully — claude -p truncates upstream of us anyway.
ENVELOPE_CHARS_MAX = 60000

CAL_DIMS = {
    "user_observations", "explicit_instructions", "approach_assessment",
    "contradictions", "drift_signals", "row_effectiveness",
    "tool_redirect_signals", "misalignment_reconvergence",
}
MEM_DIMS = {
    "session_arc_memories", "decision_memories_with_alternatives",
    "tool_brittleness_patterns", "loose_ends", "confidence_calibration_audit",
}

DIM_TO_SOURCE = {
    "user_observations": "observation",
    "explicit_instructions": "explicit",
    "approach_assessment": "meta-assessment",
    "contradictions": "meta-assessment",
    "drift_signals": "meta-assessment",
    "tool_redirect_signals": "correction",
    "misalignment_reconvergence": "correction",
}

DIM_TO_MEM_TYPE = {
    "session_arc_memories": "project",
    "decision_memories_with_alternatives": "decision",
    "tool_brittleness_patterns": "correction",
    "loose_ends": "project",
    "confidence_calibration_audit": "correction",
}


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are the Cairn calibration session analyser. You have just been given a cleaned transcript of one Claude Code session. Your job: distill the session into structured signal across 13 bounded dimensions in a single pass.

Output STRICT JSON only — no prose, no markdown fence, no commentary. The JSON object MUST have exactly these top-level keys, each an array (use [] if nothing applies):

CALIBRATION DIMENSIONS (shape *how* the agent interacts with this user):
  "user_observations"           : inferred from terminology / level fit / style mirroring
  "explicit_instructions"       : declared by user ("I prefer...", "always...", "stop X")
  "approach_assessment"         : meta self-audit of the agent's session-level approach
  "contradictions"              : current session evidence contradicting a prior calibration row
  "drift_signals"               : calibration rows worth surfacing for user review
  "row_effectiveness"           : per-delivery — score each delivered row in this session
  "tool_redirect_signals"       : user overrode the agent's proposed tool action (correction-grade)
  "misalignment_reconvergence"  : agent diverged from user intent then reconverged after N turns

MEMORY DIMENSIONS (write to the *what* knowledge store; require the arc):
  "session_arc_memories"              : narrative arc: set-out-to-do X, hit Y, pivoted to Z, left W
  "decision_memories_with_alternatives": decisions with rejected alternatives reconstructed across turns
  "tool_brittleness_patterns"         : sequences of failed/retried tool calls signalling approach error
  "loose_ends"                        : open questions / unfinished threads for next-session bootstrap
  "confidence_calibration_audit"      : where was the agent overconfident in this session

DIMENSION SIZING — your editorial judgment, not a numeric cap:
- Emit as many items per dimension as the session genuinely warrants — no more, no less.
- A short session may produce 1 item for a dimension; a deep session may produce 15. Both are correct if the content earns its place.
- Return [] freely for dimensions the session doesn't support. There is no expectation that any given dimension produce content.
- NEVER pad to look productive. Quality > quantity.
- A row that adds no retrievable signal a future session could act on is worse than no row.

ITEM SHAPES:

Calibration items (all calibration dimensions except row_effectiveness):
  {{
    "content": "<single information-dense line — what, why, context>",
    "kw": ["keyword1", "keyword2", ...],
    "qf": ["hypothetical user prompt 1", "...2", "...3"]
  }}
  - kw: 2-6 content keywords (lowercase, no hashes/ids)
  - qf: 3-6 hypothetical user-prompt phrasings this calibration row should fire before. THIS IS CRITICAL — the qf field is the symmetric-intent retrieval signal. Each phrasing is a complete short user prompt as the user might write it.
  - qf MUST be written for a *different future session* in a *different topical context*, not as a recap of what triggered the row in the source session. The user opening a new conversation on an unrelated topic should still issue a qf phrasing naturally if this row would help them.
  - BAD qf (source-session-bound; only matches if the same failure recurs): ["why is dnsmasq failing", "should I worry about swap", "did you add the import for StopHookResult", "why did you not run the command"]. These are recaps of the original transcript, not portable retrieval triggers.
  - GOOD qf (portable across contexts): ["help me debug a flaky service", "ignore expected health failures", "set up a Python import", "actually run the command don't guess"]. These match the *kind* of situation where the row's claim applies, regardless of which specific tool/service/file/error is involved. Strip project-specific names, file paths, error strings, and tool brands from qf phrasings.
  - Litmus test: would a user in an unrelated project still type this phrasing? If no, rewrite or drop the row. A row whose qf is source-session-bound is dead weight in the database.

Memory items (all memory dimensions):
  {{
    "topic": "<short topic phrase>",
    "content": "<single information-dense line>",
    "kw": ["keyword1", ...]
  }}

row_effectiveness items (one per delivered row in this session — list of deliveries supplied below):
  {{
    "row_id": <int>,
    "outcome": "followed" | "ignored" | "corrected",
    "evidence": "<short citation of why>"
  }}

VOCABULARY DISCIPLINE — every emitted row MUST be definitive and assert something concrete:
- NO hedge phrasing. Forbidden: "may or may not", "possibly", "unclear if", "could be", "might want to", "user did not respond". If you can't assert it, do not emit the row.
- NO meta-observations about the session that don't translate to retrievable signal. ("Session was about X" is not actionable; the row needs a claim a future session can act on.)
- Drop noise tokens — IDs, hashes, raw addresses, specs already captured. These are never search terms.
- Every clause must earn its place — if removing it wouldn't make the item less findable or useful, cut it.
- Never fabricate. If the session doesn't support a dimension, return [].
- One line per content. No multi-paragraph values.
- A no-op (empty array) for any dimension is always better than a hedged row.

PRIOR DELIVERIES TO SCORE (row_effectiveness):
{deliveries_json}

PRIOR CALIBRATION ROWS FROM THIS SESSION (do NOT re-emit semantically equivalent rows; only emit rows for genuinely new signal):
{prior_rows_json}

CLEANED SESSION TRANSCRIPT:
{transcript}

If you cannot tell from the transcript whether a row is warranted, do NOT emit it. A no-op block is always better than a hedged row.

RESPOND WITH JSON ONLY."""


# ---------------------------------------------------------------------------
# Extraction & LLM call
# ---------------------------------------------------------------------------

def _load_deliveries_for_session(session_id: str) -> list[dict]:
    if not os.path.exists(EPH_DB_PATH):
        return []
    conn = sqlite3.connect(EPH_DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT id, row_id, turn_index FROM calibration_deliveries "
            "WHERE session_id = ? AND outcome IS NULL",
            (session_id,),
        ).fetchall()
        return [{"id": r[0], "row_id": r[1], "turn_index": r[2]} for r in rows]
    finally:
        conn.close()


def build_prompt(transcript: str, deliveries: list[dict],
                 prior_rows: Optional[list[dict]] = None) -> str:
    prior = prior_rows or []
    return PROMPT_TEMPLATE.format(
        deliveries_json=json.dumps(deliveries, indent=2) if deliveries else "[]",
        prior_rows_json=json.dumps(prior, indent=2) if prior else "[]",
        transcript=transcript,
    )


# ---------------------------------------------------------------------------
# Subagent / triviality filter + incremental state
# ---------------------------------------------------------------------------

def _is_worth_analysing(turns: list[dict],
                        min_turns: int = MIN_SUBSTANTIVE_TURNS,
                        min_chars: int = MIN_CLEANED_CHARS) -> bool:
    """Return True if the session has enough substantive content to justify
    a Sonnet-class analyser run.

    Subagent / heartbeat / monitoring transcripts are typically far below
    both thresholds. Filtering them at this layer means we never spend a
    Sonnet call on a 2-turn 'hello / ok' exchange.
    """
    substantive = [t for t in turns if t.get("text", "").strip()]
    if len(substantive) < min_turns:
        return False
    total_chars = sum(len(t["text"]) for t in substantive)
    return total_chars >= min_chars


def _get_analyser_state(session_id: str,
                        eph_db_path: Optional[str] = None) -> Optional[dict]:
    """Read the per-session analyser state from `hook_state`. Returns None
    if the session has never been analysed."""
    path = eph_db_path or EPH_DB_PATH
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        row = conn.execute(
            "SELECT value FROM hook_state WHERE session_id = ? AND key = ?",
            (session_id, ANALYSER_STATE_KEY),
        ).fetchone()
        if not row or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
    finally:
        conn.close()


def _set_analyser_state(session_id: str, state: dict,
                        eph_db_path: Optional[str] = None) -> None:
    path = eph_db_path or EPH_DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        conn.execute(
            "INSERT INTO hook_state (session_id, key, value, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(session_id, key) DO UPDATE SET "
            "value = excluded.value, updated_at = CURRENT_TIMESTAMP",
            (session_id, ANALYSER_STATE_KEY, json.dumps(state)),
        )
        conn.commit()
    finally:
        conn.close()


def _load_prior_rows_for_session(session_id: str,
                                  db_path: Optional[str] = None) -> list[dict]:
    """Load calibration rows previously written by the analyser for this
    session, so the LLM can avoid re-emitting them. We can't filter by
    session_id directly (calibration_rows don't carry one) so we lean on
    the analyser_session_processed metric detail. As a pragmatic
    approximation, we return rows whose source is in the analyser-written
    set and that have been written since the session's first analysis.
    """
    state = _get_analyser_state(session_id)
    if not state or not state.get("last_analysed_at"):
        return []
    path = db_path or DB_PATH
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        first_seen = state.get("first_analysed_at") or state["last_analysed_at"]
        rows = conn.execute(
            "SELECT content, kw, qf, source FROM calibration_rows "
            "WHERE created_at >= ? AND source IN "
            "('observation','explicit','meta-assessment','correction') "
            "ORDER BY id DESC LIMIT 50",
            (first_seen,),
        ).fetchall()
        return [{"content": r[0], "kw": r[1], "qf": r[2], "source": r[3]}
                for r in rows]
    finally:
        conn.close()


def call_llm(prompt: str, timeout: int = DEFAULT_TIMEOUT_S,
             model: Optional[str] = None) -> str:
    """Invoke `claude -p` with CAIRN_MODE=read-only so the analyser doesn't
    itself trigger the Stop hook (which would consume the output).

    Model defaults to Sonnet 4.6 (override via CAIRN_ANALYSER_MODEL env var
    or the `model` argument). Per cairn entry 2087, the 13-dimension
    sectioned output benefits from a mode-switching-capable model — Haiku
    is the cost-floor option, not the default. Entry 1734 noted output
    truncation under `--output-format json` with Sonnet 4.6 on large
    outputs; we use plain text capture to avoid that path.
    """
    env = os.environ.copy()
    env["CAIRN_MODE"] = "read-only"
    chosen_model = model or DEFAULT_MODEL
    # Pass the prompt via stdin — large session transcripts blow past
    # Linux ARG_MAX (~128KB) when passed as a positional argument.
    # `claude -p` reads from stdin when no prompt arg is supplied.
    cmd = ["claude", "-p",
           "--allowedTools", "",
           "--output-format", "text",
           "--input-format", "text"]
    if chosen_model:
        cmd.extend(["--model", chosen_model])
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p exited {result.returncode}: {result.stderr[:400]}"
        )
    return result.stdout


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(text: str) -> dict:
    """Extract the JSON object from an LLM response.

    Tolerates a leading prose preamble or a markdown fence — finds the
    first { and the last } and json.loads the span. Raises ValueError if
    no valid JSON object is present.
    """
    if not text or not text.strip():
        raise ValueError("empty LLM output")
    # Strip markdown fence if present
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    # Find first { — handles prose preamble. Use raw_decode so we
    # accept the first valid JSON object even if the LLM trailed extra
    # objects or commentary afterwards.
    first = stripped.find("{")
    if first < 0:
        raise ValueError(f"no JSON object found in LLM output: {text[:200]!r}")
    candidate = stripped[first:]
    try:
        obj, _end = json.JSONDecoder().raw_decode(candidate)
        return obj
    except json.JSONDecodeError as e:
        # Fall back to greedy bracket span (handles minor noise inside)
        last = stripped.rfind("}")
        if last > first:
            try:
                return json.loads(stripped[first:last + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"could not parse JSON object from LLM output: "
            f"{e.msg} at {e.pos}; preview={text[:200]!r}"
        ) from e


def normalise_dimensions(parsed: dict) -> dict:
    """Normalise parsed LLM output to the canonical dimension schema:
    every known dimension present (empty list if absent), every value
    coerced to a list, unknown keys dropped. No numeric truncation —
    content quantity is editorial, governed by the prompt."""
    out = {}
    for dim in DIMENSIONS:
        items = parsed.get(dim, [])
        if not isinstance(items, list):
            items = []
        out[dim] = items
    return out


def envelope_exceeded(raw: str,
                      limit: int = ENVELOPE_CHARS_MAX) -> bool:
    """True if raw LLM output exceeds the envelope budget. Used for
    metric logging only — we still parse what we can. claude -p
    truncates upstream of us (cairn entry 1734) so this is mostly a
    detection signal for the rare case where it does not."""
    return len(raw or "") > limit


# Backwards-compatible alias for callers/tests written against the old
# name. The new name is normalise_dimensions; this shim is intentionally
# kept lightweight and will be removed once Phase 2E lands.
def enforce_caps(parsed: dict) -> dict:
    return normalise_dimensions(parsed)


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------

DEDUP_THRESHOLD = 0.85


def _embed_text(text: str):
    """Generate an embedding blob via the cairn embedding daemon.

    Returns (blob, vec) or (None, None) if embedding unavailable. The
    write path is fail-open: rows without embeddings still land, dedup
    and semantic retrieval degrade only for those rows.
    """
    try:
        from cairn import embeddings as emb
        vec = emb.embed(text)
        if vec is None:
            return None, None
        return emb.to_blob(vec), vec
    except Exception:
        return None, None


def _calibration_row_is_duplicate(conn, vec, threshold: float = DEDUP_THRESHOLD
                                  ) -> Optional[int]:
    """Return the id of the nearest existing calibration_row if its
    cosine similarity to `vec` exceeds `threshold`, else None. Uses
    brute-force scan — calibration_rows is small (hundreds to low
    thousands at scale, not millions) so this is cheap."""
    if vec is None:
        return None
    try:
        from cairn import embeddings as emb
    except Exception:
        return None
    rows = conn.execute(
        "SELECT id, embedding FROM calibration_rows "
        "WHERE embedding IS NOT NULL AND archived_at IS NULL"
    ).fetchall()
    best_id, best_sim = None, 0.0
    for rid, blob in rows:
        try:
            other = emb.from_blob(blob)
            sim = float(emb.cosine_similarity(vec, other))
        except Exception:
            continue
        if sim > best_sim:
            best_sim = sim
            best_id = rid
    if best_sim >= threshold:
        return best_id
    return None


def _memory_is_duplicate(conn, embedding_text: str,
                         threshold: float = DEDUP_THRESHOLD) -> Optional[int]:
    """Return the id of an existing analyser-written memory that is
    semantically equivalent to `embedding_text`, else None. Restricts
    the search to rows with source_ref=analyser-session-arc so that
    per-turn writes remain authoritative for the same content (they have
    higher write-time quality per cairn entry 3302).
    """
    try:
        from cairn import embeddings as emb
    except Exception:
        return None
    vec = emb.embed(embedding_text)
    if vec is None:
        return None
    rows = conn.execute(
        "SELECT id, embedding FROM memories WHERE source_ref = ? "
        "AND embedding IS NOT NULL AND deleted_at IS NULL",
        (ANALYSER_SOURCE_REF,),
    ).fetchall()
    best_id, best_sim = None, 0.0
    for rid, blob in rows:
        try:
            other = emb.from_blob(blob)
            sim = float(emb.cosine_similarity(vec, other))
        except Exception:
            continue
        if sim > best_sim:
            best_sim = sim
            best_id = rid
    if best_sim >= threshold:
        return best_id
    return None


def write_calibration_rows(parsed: dict, *,
                            session_id: Optional[str] = None,
                            db_path: Optional[str] = None) -> list[int]:
    """Insert calibration-bound dimensions into `calibration_rows`. Returns
    the list of new row IDs (in dimension-then-item order)."""
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    new_ids: list[int] = []
    try:
        for dim in CAL_DIMS:
            if dim == "row_effectiveness":
                continue  # not a row-write — handled separately
            source = DIM_TO_SOURCE[dim]
            for item in parsed.get(dim, []):
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                kw = item.get("kw") or []
                qf = item.get("qf") or []
                kw_csv = ",".join(k for k in kw if isinstance(k, str))
                qf_json = json.dumps([q for q in qf if isinstance(q, str)])
                embedding_text = " ".join([content] + [k for k in kw if isinstance(k, str)] + [q for q in qf if isinstance(q, str)])
                emb_blob, vec = _embed_text(embedding_text)
                # Dedup: skip if a non-archived calibration_row already
                # carries equivalent signal. The LLM was also told (via
                # prior_rows in the prompt) not to re-emit, but this is
                # the mechanical safety net.
                dup = _calibration_row_is_duplicate(conn, vec)
                if dup is not None:
                    _record_metric("calibration_dedup_filtered", None,
                                   '{"existing_row_id": ' + str(dup) +
                                   ', "source": "' + source + '"}')
                    continue
                conn.execute(
                    "INSERT INTO calibration_rows (content, kw, qf, source, "
                    "confidence, layer, embedding, origin_session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (content, kw_csv, qf_json, source,
                     _SOURCE_INITIAL_CONFIDENCE.get(source, 0.5),
                     "subject", emb_blob, session_id),
                )
                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                new_ids.append(new_id)
                # Per-qf sidecar embeddings — embed each qf string
                # individually so retrieval can score row as
                # max_i cos(prompt, qf_i). Falls back to row.embedding
                # only when no qf embeddings exist.
                for qf_index, qf_text in enumerate(q for q in qf if isinstance(q, str) and q.strip()):
                    qf_blob, _ = _embed_text(qf_text)
                    conn.execute(
                        "INSERT OR REPLACE INTO calibration_qf_embeddings "
                        "(row_id, qf_index, qf_text, embedding) VALUES (?, ?, ?, ?)",
                        (new_id, qf_index, qf_text, qf_blob),
                    )
                _record_metric("calibration_row_written", None,
                               '{"row_id": ' + str(new_id) +
                               ', "source": "' + source + '"}')
        conn.commit()
    finally:
        conn.close()
    return new_ids


def write_session_memories(parsed: dict, *, session_id: Optional[str],
                            project: Optional[str],
                            db_path: Optional[str] = None) -> list[int]:
    """Insert memory-bound dimensions into the existing `memories` table
    with `source_ref="analyser-session-arc"` so they are distinguishable
    from per-turn writes for confidence/audit."""
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    new_ids: list[int] = []
    try:
        for dim in MEM_DIMS:
            mem_type = DIM_TO_MEM_TYPE[dim]
            for item in parsed.get(dim, []):
                topic = (item.get("topic") or dim).strip()
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                kw = item.get("kw") or []
                kw_csv = ",".join(k for k in kw if isinstance(k, str))
                embedding_text = f"{project or ''} {mem_type} {topic} {content}".strip()
                # Dedup against prior analyser-written memories. Per cairn
                # entry 3302 per-turn writes get write-time quality
                # priority, so we don't dedup against them — they remain
                # authoritative for the same content.
                if _memory_is_duplicate(conn, embedding_text) is not None:
                    continue
                emb_blob, _ = _embed_text(embedding_text)
                # Dual embedding (schema v8): embed the bare topic separately so
                # symmetric topic retrieval cos(prompt, topic_embedding) can match
                # these rows — mirrors hooks/storage.py:_insert_memory. Without this
                # the analyser path leaves topic_embedding NULL and the gap only
                # clears on the next install-time backfill. Fail-open (None blob).
                topic_emb_blob, _ = _embed_text(topic)
                origin = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO memories (type, topic, content, embedding, topic_embedding, "
                    "session_id, project, origin_id, keywords, source_ref, "
                    "confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (mem_type, topic, content, emb_blob, topic_emb_blob, session_id, project,
                     origin, kw_csv, ANALYSER_SOURCE_REF, 0.6),
                )
                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                new_ids.append(new_id)
                if emb_blob:
                    try:
                        from cairn import embeddings as emb
                        emb.upsert_vec_index(conn, new_id, emb_blob)
                    except Exception:
                        pass
        conn.commit()
    finally:
        conn.close()
    return new_ids


def score_effectiveness(parsed: dict, *, db_path: Optional[str] = None) -> int:
    """Apply `row_effectiveness` outcomes back to `calibration_deliveries`
    and bump the corresponding counter on `calibration_rows`."""
    db_path = db_path or EPH_DB_PATH
    items = parsed.get("row_effectiveness", [])
    if not items:
        return 0
    eph = sqlite3.connect(db_path)
    eph.execute("PRAGMA busy_timeout=5000")
    dur = sqlite3.connect(DB_PATH)
    dur.execute("PRAGMA busy_timeout=5000")
    updated = 0
    try:
        for item in items:
            row_id = item.get("row_id")
            outcome = item.get("outcome")
            evidence = (item.get("evidence") or "")[:500]
            if outcome not in ("followed", "ignored", "corrected"):
                continue
            if not isinstance(row_id, int):
                continue
            cur = eph.execute(
                "UPDATE calibration_deliveries SET outcome = ?, "
                "outcome_evidence = ? WHERE row_id = ? AND outcome IS NULL",
                (outcome, evidence, row_id),
            )
            if cur.rowcount == 0:
                continue
            updated += cur.rowcount
            col = {
                "followed": "followed_count",
                "ignored": "ignored_count",
                "corrected": "corrected_count",
            }[outcome]
            _record_metric(f"calibration_row_{outcome}", None,
                           '{"row_id": ' + str(row_id) +
                           ', "count": ' + str(cur.rowcount) + '}')
            # delivered_count is incremented by log_deliveries at injection
            # time — only bump the outcome column here.
            dur.execute(
                f"UPDATE calibration_rows SET {col} = {col} + ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (cur.rowcount, row_id),
            )
        eph.commit()
        dur.commit()
    finally:
        eph.close()
        dur.close()
    return updated


# ---------------------------------------------------------------------------
# Orchestrator + cron
# ---------------------------------------------------------------------------

def _session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _project_from_path(path: str) -> Optional[str]:
    """Recover the project slug from a transcript path
    `~/.claude/projects/<project>/<session>.jsonl`."""
    try:
        parts = os.path.abspath(path).split(os.sep)
        i = parts.index("projects")
        return parts[i + 1] if i + 1 < len(parts) else None
    except (ValueError, IndexError):
        return None


def analyse_session(jsonl_path: str, *, dry_run: bool = False,
                    timeout: int = DEFAULT_TIMEOUT_S,
                    model: Optional[str] = None,
                    force: bool = False,
                    min_turns: int = MIN_SUBSTANTIVE_TURNS,
                    min_chars: int = MIN_CLEANED_CHARS,
                    incremental_threshold: int = INCREMENTAL_TURN_THRESHOLD,
                    llm_caller=None) -> dict:
    """End-to-end: extract → filter → LLM call → parse → cap → write.

    Skips sessions that fail the substantive-content filter (subagent /
    heartbeat detection). For sessions previously analysed, skips unless
    the turn count has grown by `incremental_threshold` turns. `force`
    bypasses both filters."""
    session_id = _session_id_from_path(jsonl_path)
    project = _project_from_path(jsonl_path)

    turns = session_extract.load_turns(jsonl_path)
    turns = session_extract.filter_turns(turns, signal_only=True)

    # Subagent / triviality filter — also write state so find_idle_sessions
    # excludes this session until it grows by incremental_threshold turns.
    if not force and not _is_worth_analysing(turns, min_turns=min_turns,
                                              min_chars=min_chars):
        try:
            file_line_count = sum(1 for _ in open(jsonl_path, "rb"))
        except OSError:
            file_line_count = len(turns)
        now_iso = datetime.now(timezone.utc).isoformat()
        _set_analyser_state(session_id, {
            "last_turn_count": file_line_count,
            "last_analysed_at": now_iso,
            "first_analysed_at": now_iso,
            "skipped_reason": "below-substance-threshold",
        })
        return {
            "session_id": session_id,
            "project": project,
            "skipped": "below-substance-threshold",
            "turns": len(turns),
            "chars": sum(len(t.get("text", "")) for t in turns),
        }

    # Incremental gate: re-analyse only if enough new turns have arrived
    current_turn_count = len(turns)
    prior_state = _get_analyser_state(session_id)
    if not force and prior_state:
        last_count = prior_state.get("last_turn_count", 0)
        if current_turn_count - last_count < incremental_threshold:
            return {
                "session_id": session_id,
                "project": project,
                "skipped": "below-incremental-threshold",
                "current_turns": current_turn_count,
                "last_analysed_turns": last_count,
                "needed": incremental_threshold,
            }

    transcript = session_extract.render(turns, with_tools=False)
    deliveries = _load_deliveries_for_session(session_id)
    prior_rows = _load_prior_rows_for_session(session_id) if prior_state else []
    prompt = build_prompt(transcript, deliveries, prior_rows=prior_rows)

    if llm_caller is None:
        raw = call_llm(prompt, timeout=timeout, model=model)
    else:
        raw = llm_caller(prompt)
    envelope_breach = envelope_exceeded(raw)
    parsed = normalise_dimensions(parse_output(raw))

    report = {
        "session_id": session_id,
        "project": project,
        "deliveries_to_score": len(deliveries),
        "dims": {k: len(parsed.get(k, [])) for k in DIMENSIONS},
        "calibration_row_ids": [],
        "memory_ids": [],
        "deliveries_scored": 0,
        "dry_run": dry_run,
        "envelope_chars": len(raw or ""),
        "envelope_breached": envelope_breach,
    }

    if not dry_run:
        report["calibration_row_ids"] = write_calibration_rows(parsed, session_id=session_id)
        report["memory_ids"] = write_session_memories(
            parsed, session_id=session_id, project=project)
        report["deliveries_scored"] = score_effectiveness(parsed)
        # Record incremental state so the next pass knows where we got to.
        now_iso = datetime.now(timezone.utc).isoformat()
        new_state = {
            "last_turn_count": current_turn_count,
            "last_analysed_at": now_iso,
            "first_analysed_at": (prior_state or {}).get("first_analysed_at", now_iso),
        }
        _set_analyser_state(session_id, new_state)
        _record_metric("analyser_session_processed", session_id, json.dumps({
            "rows": len(report["calibration_row_ids"]),
            "mems": len(report["memory_ids"]),
            "scored": report["deliveries_scored"],
            "turn_count": current_turn_count,
            "is_incremental": bool(prior_state),
            "envelope_chars": report["envelope_chars"],
        }))
        if envelope_breach:
            _record_metric("analyser_envelope_exceeded", session_id,
                           json.dumps({"chars": report["envelope_chars"],
                                       "limit": ENVELOPE_CHARS_MAX}))
    report["is_incremental"] = bool(prior_state)
    report["current_turn_count"] = current_turn_count
    return report


def _record_metric(event: str, session_id: str, detail: str) -> None:
    try:
        conn = sqlite3.connect(EPH_DB_PATH)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "INSERT INTO metrics (event, session_id, detail) VALUES (?, ?, ?)",
            (event, session_id, detail),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


def find_idle_sessions(idle_minutes: int = DEFAULT_IDLE_MINUTES,
                       roots: Optional[list[str]] = None,
                       now_ts: Optional[float] = None,
                       incremental_threshold: int = INCREMENTAL_TURN_THRESHOLD
                       ) -> list[str]:
    """Walk transcript roots; return paths whose mtime is older than
    `idle_minutes` AND are eligible for analysis.

    Eligible means either (a) never analysed, or (b) analysed previously
    but the JSONL has grown by at least `incremental_threshold` turns
    since the last run (incremental re-analysis for long-running
    sessions). Cheap per-session check: line-count > last_turn_count is
    a strict upper bound on substantive-turn growth and avoids parsing
    the full JSONL just to decide eligibility.
    """
    roots = roots or TRANSCRIPT_ROOTS
    now_ts = now_ts if now_ts is not None else time.time()
    cutoff = now_ts - idle_minutes * 60
    candidate_paths = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "**/*.jsonl"), recursive=True):
            try:
                if os.path.getmtime(p) < cutoff:
                    candidate_paths.append(p)
            except OSError:
                continue
    if not candidate_paths or not os.path.exists(EPH_DB_PATH):
        return candidate_paths

    # Load prior state for all candidate sessions in one query.
    conn = sqlite3.connect(EPH_DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT session_id, value FROM hook_state WHERE key = ?",
            (ANALYSER_STATE_KEY,),
        ).fetchall()
    finally:
        conn.close()
    prior_state = {}
    for sid, val in rows:
        if not val:
            continue
        try:
            prior_state[sid] = json.loads(val)
        except (TypeError, json.JSONDecodeError):
            continue

    eligible = []
    for path in candidate_paths:
        sid = _session_id_from_path(path)
        state = prior_state.get(sid)
        if state is None:
            eligible.append(path)
            continue
        # Incremental: include only if file has grown enough since last
        # analysis. Line count is a strict upper bound on turn count.
        last_count = state.get("last_turn_count", 0)
        try:
            line_count = sum(1 for _ in open(path, "rb"))
        except OSError:
            continue
        if line_count - last_count >= incremental_threshold:
            eligible.append(path)
    return eligible


def mark_insubstantive_all(idle_minutes: int = 0,
                              min_turns: int = MIN_SUBSTANTIVE_TURNS,
                              min_chars: int = MIN_CLEANED_CHARS) -> dict:
    """One-shot pre-flight: scan every idle candidate, run the substance
    filter, and write hook_state for sessions that fail. Excludes them
    from future find_idle_sessions passes until they grow by
    incremental_threshold turns. Pure I/O — no LLM cost.

    Returns counts and the list of session_ids marked.
    """
    paths = find_idle_sessions(idle_minutes=idle_minutes,
                                incremental_threshold=1)
    marked = []
    kept = []
    errors = 0
    for path in paths:
        try:
            turns = session_extract.load_turns(path)
            turns = session_extract.filter_turns(turns, signal_only=True)
            if _is_worth_analysing(turns, min_turns=min_turns,
                                    min_chars=min_chars):
                kept.append(path)
                continue
            sid = _session_id_from_path(path)
            try:
                line_count = sum(1 for _ in open(path, "rb"))
            except OSError:
                line_count = len(turns)
            now_iso = datetime.now(timezone.utc).isoformat()
            _set_analyser_state(sid, {
                "last_turn_count": line_count,
                "last_analysed_at": now_iso,
                "first_analysed_at": now_iso,
                "skipped_reason": "below-substance-threshold",
            })
            marked.append(sid)
        except Exception:
            errors += 1
    return {
        "scanned": len(paths),
        "marked_insubstantive": len(marked),
        "kept_substantive": len(kept),
        "errors": errors,
    }


# Safety cap: max sessions to *examine* per cron run when most
# candidates are likely to skip at the inner substantive-turn gate.
# `find_idle_sessions` pre-filters by raw line-count growth (a strict
# upper bound on turns); the real test only runs inside analyse_session
# after session_extract.filter_turns. With a deep backlog, scanning
# 10x the limit is enough to surface enough eligible sessions while
# bounding per-run JSONL parse cost.
DEFAULT_MAX_SCANNED_MULTIPLIER = 10


def run_cron(idle_minutes: int = DEFAULT_IDLE_MINUTES, limit: int = 5,
             dry_run: bool = False,
             model: Optional[str] = None,
             incremental_threshold: int = INCREMENTAL_TURN_THRESHOLD,
             max_scanned: Optional[int] = None,
             ) -> list[dict]:
    """One pass of the analyser cron — walk idle eligible sessions
    (never-analysed OR analysed previously and grown by
    `incremental_threshold` turns) and process until either `limit`
    sessions have been *processed* (eligible, LLM call made) OR the
    candidate pool / scan cap is exhausted, whichever comes first.

    Sessions skipped at the inner substantive-turn / incremental gate
    do NOT consume the limit — the loop keeps pulling from the queue
    until the target eligible count is reached. This addresses the
    failure mode where find_idle_sessions's cheap line-count upper bound
    surfaces a candidate that the inner gate rejects, wasting a slot.

    `max_scanned` bounds total JSONL parses to protect against a deep
    backlog of stale-but-not-eligible sessions; defaults to
    `limit * DEFAULT_MAX_SCANNED_MULTIPLIER`. Each session is wrapped in
    try/except so one failure doesn't block the rest.
    """
    paths = find_idle_sessions(idle_minutes=idle_minutes,
                                incremental_threshold=incremental_threshold)
    paths.sort(key=lambda p: os.path.getmtime(p))
    if max_scanned is None:
        max_scanned = max(limit * DEFAULT_MAX_SCANNED_MULTIPLIER, limit)

    reports = []
    processed_count = 0
    for path in paths[:max_scanned]:
        if processed_count >= limit:
            break
        try:
            report = analyse_session(
                path, dry_run=dry_run, model=model,
                incremental_threshold=incremental_threshold)
        except Exception as exc:
            session_id = _session_id_from_path(path)
            _record_metric("analyser_session_failed", session_id, str(exc)[:400])
            reports.append({"session_id": session_id, "error": str(exc)[:400]})
            processed_count += 1  # failures consume the slot — they got an LLM call
            continue
        reports.append(report)
        # A "processed" session is one where analyse_session actually
        # invoked the LLM. Skips at the substantive / incremental gate
        # return a `skipped` key and do NOT count toward the limit.
        if "skipped" not in report:
            processed_count += 1
    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cairn-calibration-analyser",
                                description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_a = sub.add_parser("analyse", help="Analyse a single JSONL session")
    sp_a.add_argument("jsonl")
    sp_a.add_argument("--dry-run", action="store_true")
    sp_a.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    sp_a.add_argument("--model", default=None,
                      help=f"Override LLM model (default {DEFAULT_MODEL}, "
                           "env CAIRN_ANALYSER_MODEL)")
    sp_a.add_argument("--force", action="store_true",
                      help="Bypass subagent + incremental filters")
    sp_a.add_argument("--min-turns", type=int, default=MIN_SUBSTANTIVE_TURNS)
    sp_a.add_argument("--min-chars", type=int, default=MIN_CLEANED_CHARS)
    sp_a.add_argument("--incremental-threshold", type=int,
                      default=INCREMENTAL_TURN_THRESHOLD)

    sp_c = sub.add_parser("cron", help="One pass of the idle-session cron")
    sp_c.add_argument("--idle-minutes", type=int, default=DEFAULT_IDLE_MINUTES)
    sp_c.add_argument("--limit", type=int, default=5,
                      help="Target number of eligible sessions to PROCESS "
                           "(LLM call made). Skips at the substantive / "
                           "incremental gate do not consume the limit.")
    sp_c.add_argument("--max-scanned", type=int, default=None,
                      help=f"Cap on sessions examined per run (defaults to "
                           f"limit * {DEFAULT_MAX_SCANNED_MULTIPLIER}). Bounds "
                           "JSONL parse cost when most candidates skip.")
    sp_c.add_argument("--dry-run", action="store_true")
    sp_c.add_argument("--model", default=None,
                      help=f"Override LLM model (default {DEFAULT_MODEL}, "
                           "env CAIRN_ANALYSER_MODEL)")
    sp_c.add_argument("--incremental-threshold", type=int,
                      default=INCREMENTAL_TURN_THRESHOLD)

    sp_m = sub.add_parser("mark-insubstantive",
                              help="Pre-flight: scan all idle candidates, mark "
                                   "below-substance ones in hook_state to "
                                   "exclude from future cron runs. No LLM cost.")
    sp_m.add_argument("--idle-minutes", type=int, default=0)
    sp_m.add_argument("--min-turns", type=int, default=MIN_SUBSTANTIVE_TURNS)
    sp_m.add_argument("--min-chars", type=int, default=MIN_CLEANED_CHARS)

    sp_l = sub.add_parser("list-idle", help="List idle un-analysed session paths")
    sp_l.add_argument("--idle-minutes", type=int, default=DEFAULT_IDLE_MINUTES)
    sp_l.add_argument("--incremental-threshold", type=int,
                      default=INCREMENTAL_TURN_THRESHOLD)

    args = p.parse_args(argv)

    if args.cmd == "analyse":
        report = analyse_session(args.jsonl, dry_run=args.dry_run,
                                  timeout=args.timeout, model=args.model,
                                  force=args.force,
                                  min_turns=args.min_turns,
                                  min_chars=args.min_chars,
                                  incremental_threshold=args.incremental_threshold)
        print(json.dumps(report, indent=2))
        return 0
    if args.cmd == "cron":
        reports = run_cron(idle_minutes=args.idle_minutes, limit=args.limit,
                           dry_run=args.dry_run, model=args.model,
                           incremental_threshold=args.incremental_threshold,
                           max_scanned=args.max_scanned)
        print(json.dumps(reports, indent=2))
        return 0
    if args.cmd == "mark-insubstantive":
        report = mark_insubstantive_all(
            idle_minutes=args.idle_minutes,
            min_turns=args.min_turns,
            min_chars=args.min_chars)
        print(json.dumps(report, indent=2))
        return 0
    if args.cmd == "list-idle":
        for path in find_idle_sessions(idle_minutes=args.idle_minutes):
            print(path)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
