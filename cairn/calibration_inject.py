"""Phase 3 — UserPromptSubmit injector for calibration rows.

Reads the user prompt, retrieves matching `calibration_rows` via symmetric
similarity (current prompt vs row embedding, which encodes content+kw+qf),
filters out rows already delivered in this session, renders a
`<calibration_profile>` block distinct from `<cairn_context>`, and logs
delivery records to `calibration_deliveries` for the analyser's
effectiveness feedback pass.

Per spec §3 (docs/spec-calibration-system.md). Surface is intentionally
distinct from `<cairn_context>` — calibration is *priming* (shape
responses), not *facts* (cite in output).
"""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3


DB_PATH = os.environ.get(
    "CAIRN_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db"),
)
EPH_DB_PATH = os.environ.get(
    "CAIRN_EPHEMERAL_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db"),
)

# Tuning knobs — kept small / overridable. The composite score combines
# semantic similarity (row.embedding vs prompt embedding), keyword
# overlap (Jaccard on `kw` tokens), confidence, and a pinned bonus.
DEFAULT_TOP_K = 5
# Raw cosine-similarity floor — used for filter (drop noise). Composite
# score (W_SIMILARITY etc) is used only for ranking matched rows.
# 0.40 catches strong+moderate topical matches and excludes tangential noise.
# Per-qf sidecar embedding (calibration_qf_embeddings) is what actually unlocks
# delivery — see _retrieve_candidates max-cos path. Single-vector cosine on
# row.embedding is only the fallback for rows without qf embeddings.
SIMILARITY_FLOOR = 0.40
W_SIMILARITY = 0.55
W_KW_OVERLAP = 0.20
W_CONFIDENCE = 0.15
W_PINNED = 0.10


def _tokenise(text: str) -> set[str]:
    """Phrase / identifier extractor for kw overlap.

    Delegates to `cairn.keywords.prompt_keywords` — YAKE phrases plus
    compound identifiers (cairn-calibration-analyser, FOO_BAR_BAZ)
    preserved verbatim. Falls back internally if yake unavailable.
    """
    from cairn.keywords import prompt_keywords
    return prompt_keywords(text)


def _kw_overlap(prompt_tokens: set[str], kw_csv: Optional[str]) -> float:
    if not kw_csv or not prompt_tokens:
        return 0.0
    row_tokens = {k.strip().lower() for k in kw_csv.split(",") if k.strip()}
    if not row_tokens:
        return 0.0
    inter = prompt_tokens & row_tokens
    union = prompt_tokens | row_tokens
    return len(inter) / len(union)  # Jaccard


def _delivered_row_ids(session_id: str,
                        eph_path: Optional[str] = None) -> set[int]:
    path = eph_path or EPH_DB_PATH
    if not session_id or not os.path.exists(path):
        return set()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT DISTINCT row_id FROM calibration_deliveries "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def _score_row(prompt_vec, row: dict, prompt_tokens: set[str]) -> float:
    """Composite score in [0, ~1.2]. Pinned rows can exceed 1.0 via the
    pinned bonus."""
    try:
        from cairn import embeddings as emb
    except Exception:
        return 0.0
    sim = 0.0
    if prompt_vec is not None and row.get("embedding") is not None:
        try:
            other = emb.from_blob(row["embedding"])
            sim = max(0.0, float(emb.cosine_similarity(prompt_vec, other)))
        except Exception:
            sim = 0.0
    overlap = _kw_overlap(prompt_tokens, row.get("kw"))
    confidence = float(row.get("confidence") or 0.0)
    pinned = 1.0 if row.get("pinned") else 0.0
    return (W_SIMILARITY * sim
            + W_KW_OVERLAP * overlap
            + W_CONFIDENCE * confidence
            + W_PINNED * pinned)


def retrieve_calibration(user_message: str, session_id: str,
                         project: Optional[str] = None,
                         top_k: int = DEFAULT_TOP_K,
                         similarity_floor: float = SIMILARITY_FLOOR,
                         db_path: Optional[str] = None,
                         eph_path: Optional[str] = None) -> list[dict]:
    """Retrieve top-K matching calibration rows for the user prompt.

    Returns rows ordered by composite score descending. Filters: not
    archived, not previously delivered in this session, score above
    `similarity_floor`. Session-scoped rows pinned to a different
    session_id are excluded.
    """
    path = db_path or DB_PATH
    if not user_message or not os.path.exists(path):
        return []

    try:
        from cairn import embeddings as emb
        prompt_vec = emb.embed(user_message)
    except Exception:
        prompt_vec = None
    prompt_tokens = _tokenise(user_message)

    already = _delivered_row_ids(session_id, eph_path=eph_path)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT id, content, kw, qf, source, confidence, pinned, "
            "layer, session_scope, embedding "
            "FROM calibration_rows "
            "WHERE archived_at IS NULL AND superseded_by IS NULL"
        ).fetchall()
        # Pre-fetch all qf embeddings in one query — avoids N+1 against
        # the sidecar. Group by row_id for max-cos lookup below.
        qf_rows = conn.execute(
            "SELECT row_id, embedding FROM calibration_qf_embeddings"
        ).fetchall()
    finally:
        conn.close()
    qf_embeds_by_row: dict[int, list[bytes]] = {}
    for rid, blob in qf_rows:
        qf_embeds_by_row.setdefault(rid, []).append(blob)

    scored = []
    for r in rows:
        row = {
            "id": r[0], "content": r[1], "kw": r[2], "qf": r[3],
            "source": r[4], "confidence": r[5], "pinned": r[6],
            "layer": r[7], "session_scope": r[8], "embedding": r[9],
        }
        if row["id"] in already:
            continue
        if row["session_scope"] and row["session_scope"] != session_id:
            continue
        # Compute raw similarity first — used for filtering. Composite
        # score is computed only for rows that pass the floor and is
        # used for ranking.
        # Symmetric per-qf retrieval (preferred): score row as
        # max_i cos(prompt, qf_i_embedding). qf strings are phrased as
        # user prompts so they live in the same embedding region as the
        # incoming prompt — far stronger signal than the third-person
        # row.embedding. Falls back to row.embedding cosine when no qf
        # embeddings exist (legacy rows pre-backfill, or rows the analyser
        # wrote without qf strings).
        sim = None
        if prompt_vec is not None:
            qf_blobs = qf_embeds_by_row.get(row["id"], [])
            if qf_blobs:
                try:
                    sims = []
                    for blob in qf_blobs:
                        qf_vec = emb.from_blob(blob)
                        sims.append(float(emb.cosine_similarity(prompt_vec, qf_vec)))
                    sim = max(sims) if sims else None
                except Exception:
                    sim = None
            elif row["embedding"] is not None:
                try:
                    other = emb.from_blob(row["embedding"])
                    sim = float(emb.cosine_similarity(prompt_vec, other))
                except Exception:
                    sim = None
        row["similarity"] = sim
        # Filter: raw similarity floor. Applies only when sim is
        # computable (both vectors present). If either side has no
        # embedding (degraded mode or unindexed row), we cannot filter
        # semantically — let the row through and let kw/conf/pinned in
        # the composite ranking carry the signal.
        # Pinned rows always bypass the floor.
        if sim is not None and sim < similarity_floor \
                and not row.get("pinned"):
            continue
        row["score"] = _score_row(prompt_vec, row, prompt_tokens)
        scored.append(row)

    # Override hierarchy: pinned > explicit > correction/meta-assessment >
    # observation (drives sort tiebreaker; primary sort remains score).
    source_priority = {
        "explicit": 4, "correction": 3, "meta-assessment": 2,
        "observation": 1,
    }
    scored.sort(
        key=lambda r: (r["pinned"], r["score"],
                       source_priority.get(r["source"], 0)),
        reverse=True,
    )
    return scored[:top_k]


def render_calibration_block(rows: list[dict]) -> str:
    """Render a `<calibration_profile>` XML block. Sections are grouped
    by `layer` (general / subject) with pinned rows promoted to an
    override section."""
    if not rows:
        return ""
    pinned = [r for r in rows if r.get("pinned")]
    general = [r for r in rows if not r.get("pinned") and r.get("layer") == "general"]
    subject = [r for r in rows if not r.get("pinned") and r.get("layer") != "general"]
    parts = ["<calibration_profile>"]
    if pinned:
        parts.append("  <override weight=\"pinned\">")
        for r in pinned:
            parts.append(f"    - {r['content']}")
        parts.append("  </override>")
    if general:
        parts.append("  <general weight=\"always\">")
        for r in general:
            parts.append(f"    - {r['content']}")
        parts.append("  </general>")
    if subject:
        parts.append("  <subject>")
        for r in subject:
            conf = r.get("confidence")
            tag = f" confidence=\"{conf:.2f}\"" if isinstance(conf, (int, float)) else ""
            parts.append(f"    - [{r['source']}{tag}] {r['content']}")
        parts.append("  </subject>")
    parts.append("</calibration_profile>")
    return "\n".join(parts)


def log_deliveries(rows: list[dict], session_id: str,
                   turn_index: Optional[int] = None,
                   eph_path: Optional[str] = None) -> int:
    """Insert delivery records for each surfaced row. `turn_index` is
    advisory — if not provided, increments per-session via `hook_state`."""
    if not rows or not session_id:
        return 0
    path = eph_path or EPH_DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        if turn_index is None:
            turn_index = _next_turn_index(conn, session_id)
        # Connect to durable DB to bump per-row delivered_count alongside
        # the ephemeral delivery record. Without this the stats panel
        # shows 0 deliveries until an analyser pass scores outcomes.
        from cairn import calibration_inject as _ci
        durable_path = _ci.DB_PATH if hasattr(_ci, "DB_PATH") else None
        durable = None
        try:
            if durable_path and os.path.exists(durable_path):
                durable = sqlite3.connect(durable_path)
                durable.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            durable = None
        n = 0
        for r in rows:
            conn.execute(
                "INSERT INTO calibration_deliveries "
                "(session_id, turn_index, row_id, similarity) "
                "VALUES (?, ?, ?, ?)",
                (session_id, turn_index, r["id"], r.get("similarity")),
            )
            n += 1
            conn.execute(
                "INSERT INTO metrics (event, session_id, detail) "
                "VALUES ('calibration_row_delivered', ?, ?)",
                (session_id, json.dumps({"row_id": r["id"],
                                          "turn": turn_index,
                                          "similarity": r.get("similarity")})),
            )
            if durable is not None:
                try:
                    durable.execute(
                        "UPDATE calibration_rows SET "
                        "delivered_count = delivered_count + 1, "
                        "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (r["id"],),
                    )
                except sqlite3.Error:
                    pass
        conn.commit()
        if durable is not None:
            try:
                durable.commit()
            finally:
                durable.close()
        return n
    finally:
        conn.close()


def _next_turn_index(conn, session_id: str) -> int:
    row = conn.execute(
        "SELECT value FROM hook_state "
        "WHERE session_id = ? AND key = 'calibration_turn_index'",
        (session_id,),
    ).fetchone()
    if row and row[0]:
        try:
            idx = int(row[0]) + 1
        except ValueError:
            idx = 0
    else:
        idx = 0
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value, updated_at) "
        "VALUES (?, 'calibration_turn_index', ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(session_id, key) DO UPDATE SET "
        "value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        (session_id, str(idx)),
    )
    return idx


GLOBAL_STATE_SESSION = "__global__"


def _read_state(session_id: str, key: str,
                eph_path: Optional[str] = None) -> Optional[str]:
    path = eph_path or EPH_DB_PATH
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        row = conn.execute(
            "SELECT value FROM hook_state WHERE session_id = ? AND key = ?",
            (session_id, key),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _is_disabled(session_id: str, eph_path: Optional[str] = None) -> bool:
    if _read_state(GLOBAL_STATE_SESSION, "calibration_disabled",
                    eph_path) == "1":
        return True
    return _read_state(session_id, "calibration_disabled",
                       eph_path) == "1"


def _session_muted_ids(session_id: str,
                        eph_path: Optional[str] = None) -> set[int]:
    path = eph_path or EPH_DB_PATH
    if not os.path.exists(path):
        return set()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT key FROM hook_state WHERE session_id = ? "
            "AND key LIKE 'calibration_mute_%'",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    out: set[int] = set()
    for (k,) in rows:
        try:
            out.add(int(k.split("_")[-1]))
        except (ValueError, IndexError):
            pass
    return out


def _mode_override_row(session_id: str,
                       eph_path: Optional[str] = None) -> Optional[dict]:
    """If a session or global interaction-level mode is set, return a
    synthetic pinned override row that the injector can prepend."""
    level = (_read_state(session_id, "calibration_mode", eph_path)
             or _read_state(GLOBAL_STATE_SESSION, "calibration_mode",
                            eph_path))
    if level not in ("novice", "expert"):
        return None
    content = (
        "User interaction level: NOVICE — explain reasoning, name file paths "
        "explicitly, avoid jargon without definition, surface assumptions"
        if level == "novice"
        else "User interaction level: EXPERT — terse / no preamble, skip "
             "obvious explanations, assume domain fluency, surface only "
             "non-obvious context"
    )
    return {
        "id": -1, "content": content, "kw": level, "qf": "[]",
        "source": "explicit", "confidence": 0.95, "pinned": 1,
        "layer": "general", "session_scope": None, "embedding": None,
        "score": 1.0, "similarity": None,
    }


def inject_for_prompt(user_message: str, session_id: str,
                      project: Optional[str] = None,
                      top_k: int = DEFAULT_TOP_K) -> str:
    """Convenience: retrieve + render + log in one call. Returns the
    rendered block, or empty string if no rows surfaced. Safe to call
    from a UserPromptSubmit hook — fails open on any error.

    Respects three hook_state flags (set via `cairn-calibration` CLI):
    - `calibration_disabled` → skip injection entirely
    - `calibration_mute_<row_id>` → exclude specific rows for this session
    - `calibration_mode` (novice|expert) → prepend a synthetic override row
    """
    try:
        if _is_disabled(session_id):
            return ""
        rows = retrieve_calibration(
            user_message, session_id, project=project, top_k=top_k)
        muted = _session_muted_ids(session_id)
        if muted:
            rows = [r for r in rows if r["id"] not in muted]
        mode_row = _mode_override_row(session_id)
        if mode_row is not None:
            rows = [mode_row] + rows
        if not rows:
            return ""
        block = render_calibration_block(rows)
        # Don't log delivery for the synthetic mode row (id=-1)
        real_rows = [r for r in rows if r.get("id", -1) >= 0]
        log_deliveries(real_rows, session_id)
        return block
    except Exception:
        return ""
