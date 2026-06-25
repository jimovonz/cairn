"""Read-side memory relevance grading — Phase 1 (instrument) + Phase 2 (label).

The portable (T0) core of docs/spec-memory-relevance-grading.md:

  * build_context_window  — the cleaned recent-context representation (current
      prompt + the last turn) that keys a delivery and feeds the cross-encoder
      student. Reuses session_extract cleaning so write-time (logging) and
      read-time (student inference) reps are byte-identical — parity is
      load-bearing (spec A.3). The prior assistant response is capped so a long
      response can't swamp the short current prompt in the embedding.
  * is_self_referential_meta — the mechanical bucket-4 prefilter (drops
      cairn-about-cairn meta-memories). High-precision by design; gated +
      audited by the caller so it can't silently drop useful domain memories.
  * log_memory_deliveries — one memory_deliveries row per injected memory
      (ephemeral DB), mirroring calibration_inject.log_deliveries.
  * parse_relevance_grades / apply_relevance_grades — agent-as-teacher 0-3
      grades + hard-negative flag, written back to the matching delivery rows.

All heavy/generative work (T1: synthesised intent, label densification) lives in
async crons over this log, never here on the hot path.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Optional

# The prior assistant response only supplies referents for anaphora, so cap it:
# a long response must not dominate the short current prompt in the embedding
# (the prompt-vs-response length asymmetry). Chars, not tokens — cheap + good enough.
PRIOR_RESPONSE_CAP = 600


def build_context_window(current_prompt: str, transcript_path: Optional[str] = None,
                         *, prior_response_cap: int = PRIOR_RESPONSE_CAP) -> str:
    """Cleaned (prev-user + capped prev-assistant + current-prompt) window.

    Pulls the prior exchange from the transcript via session_extract.load_turns
    (which already strips tool blocks, thinking, <cairn_context>, system reminders
    and [cm] defs). Fails soft to just the current prompt if the transcript is
    unavailable. Deterministic — same inputs give the same string on both sides.
    """
    cur = (current_prompt or "").strip()
    prior_user = ""
    prior_asst = ""
    if transcript_path:
        try:
            from cairn.session_extract import load_turns
            turns = [t for t in load_turns(transcript_path) if t.get("text")]
            prior_asst = next(
                (t["text"] for t in reversed(turns) if t["role"] == "assistant"), "")
            users = [t["text"] for t in turns if t["role"] == "user"]
            # If the current prompt is already the tail user turn, drop it so we
            # take the one *before* it as the prior-user referent.
            if users and cur and users[-1].strip() == cur:
                users = users[:-1]
            prior_user = users[-1] if users else ""
        except Exception:
            pass
    parts = []
    if prior_user:
        parts.append(f"[prev user] {prior_user}")
    if prior_asst:
        capped = (prior_asst if len(prior_asst) <= prior_response_cap
                  else prior_asst[:prior_response_cap].rstrip() + " …")
        parts.append(f"[prev assistant] {capped}")
    parts.append(f"[user] {cur}")
    return "\n".join(parts).strip()


# --- Bucket-4: self-referential meta ("cairn-about-cairn") ---------------------
# HIGH-PRECISION ONLY. In the cairn repo itself, legitimate domain memories mention
# "cairn" constantly, so we match meta-statements *about memory existence / coverage
# / gaps*, never the bare token "cairn". Tune additively; audit drops via metric.
_META_PATTERNS = [
    r"\bno (?:prior )?(?:memory|memories|record|entry|entries) (?:of|about|for|exist)",
    r"\bcairn (?:has no|contains no|has limited|lacks|knows nothing)",
    r"\bcairn contains a (?:profile|memory|record)",
    r"\b(?:should be|to be|will be|not yet|never) captured\b",
    r"\bcaptured when (?:shared|provided|mentioned)",
    r"\bcairn-about-cairn\b",
    r"\bself-referential meta",
    r"\bmemor(?:y|ies) (?:of [^.]{0,40} )?exists?\b",
    r"\b(?:limited|no) info(?:rmation)? (?:on|about) [^.]{0,40} in cairn",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)


def is_self_referential_meta(entry: dict[str, Any]) -> bool:
    """True if a memory is a bucket-4 self-referential meta-memory.

    Conservative: matches statements about what cairn does/doesn't remember, not
    domain content that merely mentions cairn. Never call on corrections/bootstrap
    (the spec keeps those ungated) — that gating is the caller's responsibility.
    """
    text = entry.get("content") or entry.get("c") or ""
    return bool(text) and _META_RE.search(text) is not None


# --- Delivery log -------------------------------------------------------------
def _eph_path(eph_path: Optional[str]) -> str:
    if eph_path:
        return eph_path
    from cairn.config import EPHEMERAL_DB_PATH
    return EPHEMERAL_DB_PATH


def _next_turn_index(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM memory_deliveries WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _score_components(r: dict[str, Any]) -> Optional[str]:
    """JSON of the heterogeneous score signals on a delivered row, so a label can
    be attributed to the exact scoring that produced it (ce_score is heterogeneous
    across the ms-marco->bge reranker transition; the composite blends CE + RRF +
    similarity). Only present keys are stored; returns None if nothing is known."""
    comp = {}
    for src, dst in (("ce_score", "ce"), ("score", "composite"), ("rrf_score", "rrf"),
                     ("similarity", "sim"), ("confidence", "conf")):
        v = r.get(src)
        if v is not None:
            try:
                comp[dst] = round(float(v), 6)
            except (TypeError, ValueError):
                pass
    if not comp:
        return None
    import json
    return json.dumps(comp, separators=(",", ":"))


def log_memory_deliveries(delivered: list[dict[str, Any]], *, session_id: str,
                          context_text: str = "", context_vec: Optional[bytes] = None,
                          turn_index: Optional[int] = None,
                          layer: Optional[str] = None, project: Optional[str] = None,
                          eph_path: Optional[str] = None) -> int:
    """Insert one memory_deliveries row per injected memory. Fail-soft: returns
    the count written (0 on any error) — instrumentation must never break delivery.

    Ranking provenance (step 1a) is stamped per row: reranker_model (the model that
    produced ce_score, from the daemon), score_components (JSON of all score signals),
    layer (the retrieval layer), and scope (project vs global, computed against
    `project` exactly as split_by_scope does)."""
    if not delivered or not session_id:
        return 0
    try:
        conn = sqlite3.connect(_eph_path(eph_path))
    except sqlite3.Error:
        return 0
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        if turn_index is None:
            turn_index = _next_turn_index(conn, session_id)
        n = 0
        for rank, r in enumerate(delivered):
            mid = r.get("id")
            if mid is None:
                continue
            ce = r.get("ce_score")
            if ce is None:
                ce = r.get("score")
            scope = "project" if (project and r.get("project") == project) else "global"
            conn.execute(
                "INSERT INTO memory_deliveries "
                "(session_id, turn_index, memory_id, context_text, context_vec, "
                " ce_score, served_rank, reranker_model, score_components, layer, scope) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, turn_index, int(mid), context_text, context_vec,
                 ce, r.get("served_rank", rank), r.get("reranker_model"),
                 _score_components(r), layer, scope),
            )
            n += 1
        conn.commit()
        return n
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# --- Phase 2: agent-as-teacher labels -----------------------------------------
_GRADE_RE = re.compile(r"^\s*(\d+)\s*:\s*([0-3])\s*(!)?\s*$")


def parse_relevance_grades(raw: Any) -> list[tuple[int, int, bool]]:
    """Parse ["42:3", "17:0!"] -> [(42,3,False),(17,0,True)]. memory_id:grade,
    trailing '!' = hard-negative (actively wrong/misleading; a distinct axis)."""
    out: list[tuple[int, int, bool]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        if not isinstance(item, str):
            continue
        m = _GRADE_RE.match(item)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), bool(m.group(3))))
    return out


def apply_relevance_grades(grades: list[tuple[int, int, bool]], *, session_id: str,
                           eph_path: Optional[str] = None) -> int:
    """Write 0-3 grade + hard_negative onto the most-recent delivery of each graded
    memory in this session (the one the agent just judged). Fail-soft."""
    if not grades or not session_id:
        return 0
    try:
        conn = sqlite3.connect(_eph_path(eph_path))
    except sqlite3.Error:
        return 0
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        n = 0
        for mid, grade, hard in grades:
            cur = conn.execute(
                "UPDATE memory_deliveries SET grade = ?, hard_negative = ? WHERE id = ("
                "  SELECT id FROM memory_deliveries WHERE session_id = ? AND memory_id = ? "
                "  ORDER BY id DESC LIMIT 1)",
                (int(grade), 1 if hard else 0, session_id, int(mid)),
            )
            n += cur.rowcount
        conn.commit()
        return n
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
