"""Memory storage, deduplication, confidence updates, and quality gates."""

from __future__ import annotations

import json
import re
from types import ModuleType
from typing import Optional

import hooks.hook_helpers as hook_helpers
from hooks.hook_helpers import log, get_conn, get_session_project, record_metric
from hooks import hook_helpers
from cairn.config import (DEDUP_THRESHOLD, CONFIDENCE_BOOST,
                     CONFIDENCE_MIN, CONFIDENCE_MAX, CONFIDENCE_DEFAULT,
                     DISTINCT_VARIANT_SIM_THRESHOLD, NEGATION_SIM_FLOOR)


def extract_associated_files(transcript_path: str, lookback: int = 30) -> list[str]:
    """Extract file paths from recent tool calls in the transcript.

    Scans the last `lookback` entries for Read, Edit, Write, and MultiEdit
    tool uses and returns unique file paths found.
    """
    if not transcript_path:
        return []
    try:
        # Read all lines and take the last N
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        recent = lines[-lookback:] if len(lines) > lookback else lines

        files: list[str] = []
        seen: set[str] = set()
        for line in recent:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Tool use entries have tool_name and parameters
            tool_name = entry.get("tool_name") or ""
            if tool_name in ("Read", "Edit", "Write", "MultiEdit"):
                params = entry.get("parameters") or entry.get("input") or {}
                fp = params.get("file_path") or params.get("filePath") or ""
                if fp and fp not in seen:
                    files.append(fp)
                    seen.add(fp)
                # MultiEdit may have multiple file edits
                edits = params.get("edits") or []
                for edit in edits:
                    efp = edit.get("file_path") or edit.get("filePath") or ""
                    if efp and efp not in seen:
                        files.append(efp)
                        seen.add(efp)

            # Also check Bash tool calls for file paths in commands
            if tool_name == "Bash":
                bash_params = entry.get("parameters") or entry.get("input") or {}
                cmd = bash_params.get("command", "") if isinstance(bash_params, dict) else ""
                # Extract paths that look like file references
                for match in re.findall(r'(?:^|\s)(/[^\s;|&>]+\.[a-zA-Z0-9]+)', str(cmd)):
                    if match not in seen:
                        files.append(match)
                        seen.add(match)

        return files
    except (FileNotFoundError, PermissionError, OSError):
        return []


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


def apply_confidence_updates(updates: list[tuple[int, str, Optional[str]]], session_id: Optional[str] = None) -> int:
    """Apply confidence adjustments from LLM feedback.

    Directions:
      +   — corroboration: boost confidence (saturating) — memory is consistent with observations
      -   — irrelevant: no confidence change — irrelevance is not evidence against truth
      -!  — contradiction: annotate memory with reason, keep retrievable
    """
    if not updates:
        return 0
    conn = get_conn()
    applied = 0
    for update in updates:
        memory_id, direction = update[0], update[1]
        reason = update[2] if len(update) > 2 else None
        row = conn.execute("SELECT confidence, content FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            log(f"Confidence update: memory {memory_id} not found")
            continue

        if direction == "-!":
            # Contradiction annotation — mark memory as superseded with reason
            annotation = reason or "contradicted by later session"
            conn.execute(
                "UPDATE memories SET archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (annotation, memory_id)
            )
            log(f"Contradicted: memory {memory_id} — {annotation}")
            record_metric(session_id, "contradiction_annotated", f"{memory_id}: {annotation[:80]}")
            applied += 1
            continue

        if direction == "+":
            # Corroboration — boost veracity
            current = row[0] if row[0] is not None else CONFIDENCE_DEFAULT
            new = min(current + CONFIDENCE_BOOST * (1 - current), CONFIDENCE_MAX)
            conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (new, memory_id))
            log(f"Corroborated: memory {memory_id} {current:.2f} → {new:.2f}")
            applied += 1
        else:
            # Irrelevant (-) — log but don't adjust confidence
            log(f"Irrelevant: memory {memory_id} — no confidence change")
            record_metric(session_id, "confidence_irrelevant", f"{memory_id}")
            applied += 1
    conn.commit()
    conn.close()
    return applied


def insert_memories(entries: list[dict[str, str]], session_id: Optional[str] = None,
                    transcript_path: Optional[str] = None) -> int:
    """Insert memory entries, deduplicating via cosine similarity."""
    if not entries:
        return 0

    from cairn.config import MAX_MEMORIES_PER_RESPONSE

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

    # Lazy file extraction — associate touched files with all memory types
    _associated_files: Optional[list[str]] = None
    if transcript_path:
        _associated_files = extract_associated_files(transcript_path)
        if _associated_files:
            log(f"File association: {len(_associated_files)} files from transcript")

    for entry in entries:
        mem_type: str = entry.get("type", "fact")
        topic: str = entry.get("topic", "unknown")
        content: str = entry.get("content", "")
        depth: Optional[int] = entry.get("depth")

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
            except (ConnectionError, TimeoutError, OSError) as e:
                log(f"Embedding unavailable: {e}")
            except Exception as e:
                log(f"Embedding error ({type(e).__name__}): {e}")

        # Step 1: Check type+topic match first — catches same-topic contradictions
        # that embedding nearest-neighbour might miss (if a closer global match exists)
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
                # Different enough to be a distinct variant — check for negation
                if _has_negation_mismatch(content, old_content):
                    log(f"Negation on same topic: '{content[:40]}' vs '{old_content[:40]}' — annotating old as superseded")
                    conn.execute(
                        "UPDATE memories SET archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (f"superseded: {content[:200]}", same_topic[0])
                    )
                    record_metric(session_id, "contradiction_detected", f"{mem_type}/{topic}")
                log(f"Distinct variant: type={mem_type} topic={topic} (sim={old_sim:.2f}) — inserting as new")
                conn.execute(
                    "INSERT INTO memories (type, topic, content, embedding, session_id, project, depth) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (mem_type, topic, content, embedding_blob, session_id, project, depth)
                )
                if embedding_blob and emb:
                    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    emb.upsert_vec_index(conn, new_id, embedding_blob)
            else:
                if old_content and old_content != content:
                    # Annotate old content as superseded
                    conn.execute(
                        "UPDATE memories SET archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (f"superseded: {content[:200]}", same_topic[0])
                    )
                    log(f"Contradiction: type={mem_type} topic={topic} (sim={old_sim:.2f}) — old annotated as superseded")
                    record_metric(session_id, "contradiction_detected", f"{mem_type}/{topic}")
                conn.execute(
                    "UPDATE memories SET content = ?, embedding = ?, session_id = ?, project = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (content, embedding_blob, session_id, project, CONFIDENCE_DEFAULT, same_topic[0])
                )
                if embedding_blob and emb:
                    emb.upsert_vec_index(conn, same_topic[0], embedding_blob)
        else:
            # Step 2: No type+topic match — check for semantic near-duplicates and
            # cross-topic negation via embedding similarity
            deduped = False
            if embedding_blob and emb:
                try:
                    nearest = emb.find_nearest(conn, search_text, limit=1)
                    if nearest and nearest[0]["similarity"] >= DEDUP_THRESHOLD:
                        match = nearest[0]
                        log(f"Dedup: '{content[:50]}' ~= '{match['content'][:50]}' (sim={match['similarity']:.3f})")
                        conn.execute(
                            "UPDATE memories SET content = ?, embedding = ?, session_id = ?, project = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                            (content, embedding_blob, session_id, project, match["id"])
                        )
                        emb.upsert_vec_index(conn, match["id"], embedding_blob)
                        deduped = True
                    elif nearest and nearest[0]["similarity"] >= NEGATION_SIM_FLOOR:
                        match = nearest[0]
                        if _has_negation_mismatch(content, match["content"]):
                            log(f"Negation mismatch: '{content[:40]}' vs '{match['content'][:40]}' — annotating old as superseded")
                            conn.execute(
                                "UPDATE memories SET archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                                (f"superseded: {content[:200]}", match["id"])
                            )
                            record_metric(session_id, "contradiction_detected", f"{mem_type}/{topic}")
                except Exception:
                    pass  # Embedding issues don't block insertion

            if not deduped:
                conn.execute(
                    "INSERT INTO memories (type, topic, content, embedding, session_id, project, depth) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (mem_type, topic, content, embedding_blob, session_id, project, depth)
                )
                if embedding_blob and emb:
                    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    emb.upsert_vec_index(conn, new_id, embedding_blob)
        inserted += 1

        # Associate file paths with all memories
        if _associated_files:
            files_json = json.dumps(_associated_files)
            # Get the ID of the memory we just inserted/updated
            last_id = conn.execute(
                "SELECT id FROM memories WHERE type = ? AND topic = ? ORDER BY updated_at DESC LIMIT 1",
                (mem_type, topic)
            ).fetchone()
            if last_id:
                conn.execute(
                    "UPDATE memories SET associated_files = ? WHERE id = ?",
                    (files_json, last_id[0])
                )
                log(f"Associated {len(_associated_files)} files with {mem_type} {last_id[0]}: {_associated_files[:3]}")

    conn.commit()

    # Inline backfill: fill up to BACKFILL_INLINE_MAX missing embeddings synchronously
    # via the daemon socket. Previously this spawned a detached subprocess that held
    # a write lock after the parent stop_hook exited — a classic concurrent-writer
    # pattern that contributed to DB corruption. Inline keeps all writes within the
    # current transaction boundary; remaining missing embeddings get picked up on
    # subsequent responses. Brute-force fallback handles retrieval for memories
    # still missing embeddings.
    _inline_backfill(conn)
    conn.close()

    return inserted


def _inline_backfill(conn) -> None:
    """Fill a bounded number of missing embeddings inline via the daemon socket.
    Bounded so a single response can't turn into a long-running transaction."""
    BACKFILL_INLINE_MAX = 5
    try:
        rows = conn.execute(
            "SELECT id, type, topic, content, project FROM memories "
            "WHERE embedding IS NULL LIMIT ?",
            (BACKFILL_INLINE_MAX,)
        ).fetchall()
        if not rows:
            return
        emb = hook_helpers.get_embedder()
        if emb is None:
            return
        filled = 0
        for row in rows:
            mem_id, mem_type, topic, content, project = row
            project_prefix = f"{project} " if project else ""
            search_text = f"{project_prefix}{mem_type} {topic} {content}"
            try:
                vec = emb.embed(search_text, allow_slow=False)
            except Exception:
                vec = None
            if vec is None:
                # Daemon unavailable; skip, let brute-force handle retrieval and
                # try again on the next response
                continue
            blob = emb.to_blob(vec)
            conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (blob, mem_id))
            try:
                emb.upsert_vec_index(conn, mem_id, blob)
            except Exception:
                pass
            filled += 1
        if filled:
            conn.commit()
            log(f"Inline backfill: filled {filled} missing embeddings")
    except Exception as exc:
        log(f"Inline backfill error: {exc}")
