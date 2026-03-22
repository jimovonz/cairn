"""Memory storage, deduplication, confidence updates, and quality gates."""

from __future__ import annotations

from types import ModuleType
from typing import Optional

import hook_helpers
from hook_helpers import log, get_conn, get_session_project, record_metric
from config import (DEDUP_THRESHOLD, CONFIDENCE_BOOST, CONFIDENCE_PENALTY,
                     CONFIDENCE_MIN, CONFIDENCE_MAX, CONFIDENCE_DEFAULT,
                     DISTINCT_VARIANT_SIM_THRESHOLD, NEGATION_SIM_FLOOR)


EMPTY_MEMORY_PATTERNS: list[str] = [
    "no context available",
    "no relevant context",
    "no technical context",
    "no information available",
    "no data available",
    "nothing to report",
    "no memories found",
    "context not available",
    "unclear what this refers to",
    "insufficient context",
    "no prior context",
    "unable to determine",
    "not enough information",
    "no relevant information",
]

NEGATION_PATTERNS: set[str] = {"not", "never", "no longer", "isn't", "aren't", "shouldn't",
                     "don't", "doesn't", "won't", "can't", "cannot", "without",
                     "instead of", "rather than", "replaced", "removed", "deprecated"}

DIRECTIONAL_PAIRS: list[tuple[str, str]] = [
    ("increase", "decrease"), ("enable", "disable"), ("add", "remove"),
    ("use", "avoid"), ("prefer", "avoid"), ("include", "exclude"),
    ("allow", "block"), ("accept", "reject"), ("start", "stop"),
    ("upgrade", "downgrade"), ("required", "optional"),
]


def _is_empty_memory(content: Optional[str]) -> bool:
    """Check if a memory's content is essentially 'I don't know' — no retrievable knowledge.

    Uses substring matching (fast, deterministic) rather than embeddings to avoid
    interference with mocked embedders in tests and to keep the gate lightweight.
    """
    if not content or len(content.strip()) < 10:
        return True

    content_lower = content.lower()
    return any(p in content_lower for p in EMPTY_MEMORY_PATTERNS)


def _has_negation_mismatch(text_a: str, text_b: str) -> bool:
    """Lightweight heuristic: check if one text negates or directionally contradicts the other."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    # Check negation word mismatch
    neg_a = words_a & NEGATION_PATTERNS
    neg_b = words_b & NEGATION_PATTERNS
    if neg_a ^ neg_b:
        return True

    # Check directional opposition
    for pos, neg in DIRECTIONAL_PAIRS:
        if (pos in words_a and neg in words_b) or (neg in words_a and pos in words_b):
            return True

    return False


def apply_confidence_updates(updates: list[tuple[int, str]], session_id: Optional[str] = None) -> int:
    """Apply confidence adjustments from LLM feedback."""
    if not updates:
        return 0
    conn = get_conn()
    applied = 0
    for memory_id, direction in updates:
        row = conn.execute("SELECT confidence FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            log(f"Confidence update: memory {memory_id} not found")
            continue
        current = row[0] if row[0] is not None else CONFIDENCE_DEFAULT
        if direction == "+":
            new = min(current + CONFIDENCE_BOOST * (1 - current), CONFIDENCE_MAX)
        else:
            new = max(current - CONFIDENCE_PENALTY * (1 + current), CONFIDENCE_MIN)
        conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (new, memory_id))
        log(f"Confidence: memory {memory_id} {current:.2f} → {new:.2f} ({direction})")
        applied += 1
    conn.commit()
    conn.close()
    return applied


def insert_memories(entries: list[dict[str, str]], session_id: Optional[str] = None) -> int:
    """Insert memory entries, deduplicating via cosine similarity."""
    if not entries:
        return 0

    from config import MAX_MEMORIES_PER_RESPONSE

    # Write throttling: cap entries per response, keep highest-value ones
    if len(entries) > MAX_MEMORIES_PER_RESPONSE:
        type_priority: dict[str, int] = {"correction": 0, "decision": 1, "fact": 2, "preference": 3,
                         "person": 4, "skill": 5, "workflow": 6, "project": 7}
        entries.sort(key=lambda e: type_priority.get(e.get("type", ""), 99))
        dropped = entries[MAX_MEMORIES_PER_RESPONSE:]
        entries = entries[:MAX_MEMORIES_PER_RESPONSE]
        log(f"Write throttle: kept {len(entries)}, dropped {len(dropped)}")

    emb: Optional[ModuleType] = hook_helpers.get_embedder()
    conn = get_conn()
    project: Optional[str] = get_session_project(conn, session_id)
    inserted: int = 0

    for entry in entries:
        mem_type: str = entry.get("type", "fact")
        topic: str = entry.get("topic", "unknown")
        content: str = entry.get("content", "")
        source_start: Optional[str] = entry.get("source_start")
        source_end: Optional[str] = entry.get("source_end")

        # Content quality gate: reject memories with no retrievable knowledge
        if _is_empty_memory(content):
            log(f"Quality gate: rejected empty memory '{topic}': '{content[:60]}'")
            record_metric(session_id, "empty_memory_rejected", f"{mem_type}/{topic}")
            continue

        # Augment embedding text with project to push unrelated domains apart in vector space
        project_prefix: str = f"{project} " if project else ""
        search_text: str = f"{project_prefix}{mem_type} {topic} {content}"

        embedding_blob: Optional[bytes] = None
        if emb:
            try:
                vec = emb.embed(search_text, allow_slow=False)
                if vec is None:
                    log(f"No daemon — storing '{topic}' without embedding (will backfill)")
                    raise Exception("daemon_unavailable")
                embedding_blob = emb.to_blob(vec)

                # Check for semantic duplicates via vec index
                nearest = emb.find_nearest(conn, search_text, limit=1)
                if nearest and nearest[0]["similarity"] >= DEDUP_THRESHOLD:
                    match = nearest[0]
                    log(f"Dedup: '{content[:50]}' ~= '{match['content'][:50]}' (sim={match['similarity']:.3f})")
                    conn.execute(
                        "UPDATE memories SET content = ?, embedding = ?, session_id = ?, project = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (content, embedding_blob, session_id, project, match["id"])
                    )
                    emb.upsert_vec_index(conn, match["id"], embedding_blob)
                    inserted += 1
                    continue
                # Negation-based contradiction dampening for similar but non-duplicate entries
                if nearest and nearest[0]["similarity"] >= NEGATION_SIM_FLOOR and nearest[0]["similarity"] < DEDUP_THRESHOLD:
                    match = nearest[0]
                    if _has_negation_mismatch(content, match["content"]):
                        log(f"Negation mismatch: '{content[:40]}' vs '{match['content'][:40]}' — dampening both")
                        conn.execute(
                            "UPDATE memories SET confidence = MAX(confidence - 0.1, 0) WHERE id = ?",
                            (match["id"],)
                        )
                        record_metric(session_id, "negation_contradiction", f"{mem_type}/{topic}")

            except (ConnectionError, TimeoutError, OSError) as e:
                log(f"Embedding unavailable: {e}")
            except Exception as e:
                log(f"Embedding error ({type(e).__name__}): {e}")

        # No semantic match — fall back to exact type+topic check
        same_topic = conn.execute(
            "SELECT id, content FROM memories WHERE type = ? AND topic = ?",
            (mem_type, topic)
        ).fetchone()

        if same_topic:
            old_content: Optional[str] = same_topic[1]
            old_sim: float = 0.0
            if embedding_blob and emb and old_content:
                try:
                    old_vec = emb.embed(f"{project or ''} {mem_type} {topic} {old_content}".strip(), allow_slow=False)
                    new_vec = emb.embed(search_text, allow_slow=False)
                    if old_vec is None or new_vec is None:
                        raise Exception("daemon_unavailable")
                    old_sim = emb.cosine_similarity(old_vec, new_vec)
                except Exception:
                    old_sim = 0.0

            if old_content and old_content != content and old_sim < DISTINCT_VARIANT_SIM_THRESHOLD:
                log(f"Distinct variant: type={mem_type} topic={topic} (sim={old_sim:.2f}) — inserting as new")
                conn.execute(
                    "INSERT INTO memories (type, topic, content, embedding, session_id, project, source_start, source_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (mem_type, topic, content, embedding_blob, session_id, project, source_start, source_end)
                )
                if embedding_blob and emb:
                    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    emb.upsert_vec_index(conn, new_id, embedding_blob)
            else:
                if old_content and old_content != content:
                    conn.execute(
                        "UPDATE memories SET confidence = 0.2 WHERE id = ? AND confidence > 0.2",
                        (same_topic[0],)
                    )
                    log(f"Contradiction: type={mem_type} topic={topic} (sim={old_sim:.2f}) — old confidence dropped to 0.2")
                    record_metric(session_id, "contradiction_detected", f"{mem_type}/{topic}")
                conn.execute(
                    "UPDATE memories SET content = ?, embedding = ?, session_id = ?, project = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (content, embedding_blob, session_id, project, CONFIDENCE_DEFAULT, same_topic[0])
                )
                if embedding_blob and emb:
                    emb.upsert_vec_index(conn, same_topic[0], embedding_blob)
        else:
            conn.execute(
                "INSERT INTO memories (type, topic, content, embedding, session_id, project, source_start, source_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (mem_type, topic, content, embedding_blob, session_id, project, source_start, source_end)
            )
            if embedding_blob and emb:
                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                emb.upsert_vec_index(conn, new_id, embedding_blob)
        inserted += 1

    conn.commit()

    # Auto-backfill: if any memories lack embeddings, trigger background backfill
    missing = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL").fetchone()[0]
    conn.close()
    if missing > 0:
        _trigger_background_backfill(missing)

    return inserted


def _trigger_background_backfill(missing_count: int) -> None:
    """Start the daemon (if not running) and backfill memories without embeddings.

    Starting the daemon is preferred over a standalone backfill because:
    1. The daemon keeps the model resident — future embeds are instant
    2. Backfill uses the daemon for embedding, no duplicate model load
    """
    import subprocess
    import os as _os
    venv_python = _os.path.join(_os.path.dirname(__file__), "..", ".venv", "bin", "python3")
    daemon_py = _os.path.join(_os.path.dirname(__file__), "..", "cairn", "daemon.py")
    query_py = _os.path.join(_os.path.dirname(__file__), "..", "cairn", "query.py")
    if not _os.path.exists(venv_python):
        log(f"Auto-backfill: {missing_count} missing but venv not found")
        return

    # Start daemon first (if not already running) — it auto-checks
    if _os.path.exists(daemon_py):
        subprocess.Popen(
            [venv_python, daemon_py, "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Then backfill via the daemon
    if _os.path.exists(query_py):
        log(f"Auto-backfill: {missing_count} memories without embeddings — starting daemon + backfill")
        subprocess.Popen(
            [venv_python, query_py, "--backfill"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
