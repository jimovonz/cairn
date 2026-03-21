"""Memory storage, deduplication, confidence updates, and quality gates."""

import hook_helpers
from hook_helpers import log, get_conn, get_session_project, record_metric
from config import (DEDUP_THRESHOLD, CONFIDENCE_BOOST, CONFIDENCE_PENALTY,
                     CONFIDENCE_MIN, CONFIDENCE_MAX, CONFIDENCE_DEFAULT,
                     DISTINCT_VARIANT_SIM_THRESHOLD, NEGATION_SIM_FLOOR)


EMPTY_MEMORY_PATTERNS = [
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

NEGATION_PATTERNS = {"not", "never", "no longer", "isn't", "aren't", "shouldn't",
                     "don't", "doesn't", "won't", "can't", "cannot", "without",
                     "instead of", "rather than", "replaced", "removed", "deprecated"}

DIRECTIONAL_PAIRS = [
    ("increase", "decrease"), ("enable", "disable"), ("add", "remove"),
    ("use", "avoid"), ("prefer", "avoid"), ("include", "exclude"),
    ("allow", "block"), ("accept", "reject"), ("start", "stop"),
    ("upgrade", "downgrade"), ("required", "optional"),
]


def _is_empty_memory(content):
    """Check if a memory's content is essentially 'I don't know' — no retrievable knowledge.

    Uses substring matching (fast, deterministic) rather than embeddings to avoid
    interference with mocked embedders in tests and to keep the gate lightweight.
    """
    if not content or len(content.strip()) < 10:
        return True

    content_lower = content.lower()
    return any(p in content_lower for p in EMPTY_MEMORY_PATTERNS)


def _has_negation_mismatch(text_a, text_b):
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


def apply_confidence_updates(updates, session_id=None):
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


def insert_memories(entries, session_id=None):
    """Insert memory entries, deduplicating via cosine similarity."""
    if not entries:
        return 0

    from config import MAX_MEMORIES_PER_RESPONSE

    # Write throttling: cap entries per response, keep highest-value ones
    if len(entries) > MAX_MEMORIES_PER_RESPONSE:
        type_priority = {"correction": 0, "decision": 1, "fact": 2, "preference": 3,
                         "person": 4, "skill": 5, "workflow": 6, "project": 7}
        entries.sort(key=lambda e: type_priority.get(e.get("type", ""), 99))
        dropped = entries[MAX_MEMORIES_PER_RESPONSE:]
        entries = entries[:MAX_MEMORIES_PER_RESPONSE]
        log(f"Write throttle: kept {len(entries)}, dropped {len(dropped)}")

    emb = hook_helpers.get_embedder()
    conn = get_conn()
    project = get_session_project(conn, session_id)
    inserted = 0

    for entry in entries:
        mem_type = entry.get("type", "fact")
        topic = entry.get("topic", "unknown")
        content = entry.get("content", "")
        source_start = entry.get("source_start")
        source_end = entry.get("source_end")

        # Content quality gate: reject memories with no retrievable knowledge
        if _is_empty_memory(content):
            log(f"Quality gate: rejected empty memory '{topic}': '{content[:60]}'")
            record_metric(session_id, "empty_memory_rejected", f"{mem_type}/{topic}")
            continue

        # Augment embedding text with project to push unrelated domains apart in vector space
        project_prefix = f"{project} " if project else ""
        search_text = f"{project_prefix}{mem_type} {topic} {content}"

        embedding_blob = None
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
            old_content = same_topic[1]
            old_sim = 0.0
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
    conn.close()
    return inserted
