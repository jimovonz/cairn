#!/usr/bin/env python3
"""
Claude Code Stop Hook for Cairn.

Reads the hook input from stdin, parses <memory> blocks from the transcript,
inserts new memories into the database, and blocks stopping if complete: false.

Exit codes:
  0 = allow stop
  2 = block stop (force continuation)
"""

import json
import re
import sqlite3
import sys
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", "cairn.db")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", "hook.log")
CONTEXT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", ".context_cache")
CONTINUATION_COUNT_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", ".continuation_count")
BRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "cairn")
sys.path.insert(0, BRAIN_DIR)


def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"{msg}\n")


def record_metric(session_id, event, detail=None, value=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO metrics (event, session_id, detail, value) VALUES (?, ?, ?, ?)",
            (event, session_id, detail, value)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def parse_memory_block(text):
    """Extract memory entries, completeness, and context needs from a <memory> block.

    Robust parser that handles:
    - Malformed/unclosed tags (tries to find content after last <memory>)
    - Unknown fields (ignored gracefully)
    - Unknown types (accepted as-is)
    - Extra whitespace, missing dashes, inconsistent formatting
    """
    # Try closed tags first, fall back to unclosed
    pattern = r"<memory>(.*?)</memory>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        # Try unclosed tag — grab everything after last <memory>
        unclosed = re.search(r"<memory>(.*?)$", text, re.DOTALL)
        if unclosed:
            matches = [unclosed.group(1)]
            log("Warning: unclosed <memory> tag — parsed anyway")
        else:
            return None, None, None, None, None, [], None, []

    block = matches[-1].strip()

    # Check for no-op block
    if block in ("complete: true", "- complete: true"):
        return [], True, None, "sufficient", None, [], None, []

    # Parse entries
    entries = []
    current = {}
    complete = True
    remaining = None
    context = "sufficient"
    context_need = None
    retrieval_outcome = None  # useful | neutral | harmful
    keywords = []  # topic keywords for Layer 2 cross-project search
    confidence_updates = []  # list of (memory_id, direction)

    for line in block.split("\n"):
        line = line.strip()
        if not line or line == "-":
            continue

        # Parse confidence_update: <id>:+ or <id>:-
        conf_match = re.match(r"^-?\s*confidence_update:\s*(\d+)\s*:\s*([+-])", line)
        if conf_match:
            confidence_updates.append((int(conf_match.group(1)), conf_match.group(2)))
            continue

        # Parse any "- key: value" or "key: value" pattern
        match = re.match(r"^-?\s*(\w+):\s*(.+)$", line)
        if not match:
            continue

        key, value = match.group(1).strip(), match.group(2).strip()

        if key == "complete":
            complete = value.lower() == "true"
        elif key == "remaining":
            remaining = value
        elif key == "context":
            context = value.lower()
        elif key == "context_need":
            context_need = value
        elif key == "retrieval_outcome":
            retrieval_outcome = value.lower()
        elif key == "keywords":
            keywords = [k.strip() for k in value.split(",") if k.strip()]
        elif key == "source_messages":
            # Parse "12-18" or "5" into start, end
            try:
                if "-" in value:
                    parts = value.split("-")
                    current["source_start"] = int(parts[0].strip())
                    current["source_end"] = int(parts[1].strip())
                else:
                    current["source_start"] = int(value.strip())
                    current["source_end"] = int(value.strip())
            except (ValueError, IndexError):
                pass
        elif key in ("type", "topic", "content"):
            # If starting a new entry (type seen) and current entry is complete, commit it
            if key == "type" and "type" in current and "topic" in current and "content" in current:
                entries.append(current.copy())
                current = {}
            current[key] = value
        # Unknown fields are silently ignored

    # Handle partial entry (has type+topic but missing content)
    if current and "type" in current and "topic" in current:
        current.setdefault("content", current.get("topic", ""))
        entries.append(current.copy())

    return entries, complete, remaining, context, context_need, confidence_updates, retrieval_outcome, keywords


from config import (DEDUP_THRESHOLD, MAX_CONTINUATIONS, CONFIDENCE_BOOST, CONFIDENCE_PENALTY,
                     CONFIDENCE_MIN, CONFIDENCE_MAX, CONFIDENCE_DEFAULT,
                     L3_PROJECT_SIM_THRESHOLD, L3_GLOBAL_SIM_WITH_PROJECT,
                     L3_GLOBAL_SIM_WITHOUT_PROJECT, L3_MAX_PROJECT_RESULTS,
                     L3_MAX_GLOBAL_RESULTS, MIN_INJECTION_SIMILARITY)


def apply_confidence_updates(updates, session_id=None):
    """Apply confidence adjustments from LLM feedback."""
    if not updates:
        return 0
    conn = sqlite3.connect(DB_PATH)
    applied = 0
    for memory_id, direction in updates:
        row = conn.execute("SELECT confidence FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            log(f"Confidence update: memory {memory_id} not found")
            continue
        current = row[0] if row[0] is not None else CONFIDENCE_DEFAULT
        if direction == "+":
            # Saturating boost: diminishing returns as confidence approaches 1.0
            new = min(current + CONFIDENCE_BOOST * (1 - current), CONFIDENCE_MAX)
        else:
            # Scaled penalty: overconfident entries fall harder
            new = max(current - CONFIDENCE_PENALTY * (1 + current), CONFIDENCE_MIN)
        conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (new, memory_id))
        log(f"Confidence: memory {memory_id} {current:.2f} → {new:.2f} ({direction})")
        applied += 1
    conn.commit()
    conn.close()
    return applied


def get_embedder():
    """Lazy-load the embeddings module."""
    sys.path.insert(0, BRAIN_DIR)
    try:
        import embeddings
        return embeddings
    except ImportError:
        return None


def get_session_project(conn, session_id):
    """Look up the project label for a session."""
    if not session_id:
        return None
    row = conn.execute("SELECT project FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return row[0] if row else None


NEGATION_PATTERNS = {"not", "never", "no longer", "isn't", "aren't", "shouldn't",
                     "don't", "doesn't", "won't", "can't", "cannot", "without",
                     "instead of", "rather than", "replaced", "removed", "deprecated"}

DIRECTIONAL_PAIRS = [
    ("increase", "decrease"), ("enable", "disable"), ("add", "remove"),
    ("use", "avoid"), ("prefer", "avoid"), ("include", "exclude"),
    ("allow", "block"), ("accept", "reject"), ("start", "stop"),
    ("upgrade", "downgrade"), ("required", "optional"),
]


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


def insert_memories(entries, session_id=None):
    """Insert memory entries, deduplicating via cosine similarity."""
    if not entries:
        return 0

    from config import MAX_MEMORIES_PER_RESPONSE

    # Write throttling: cap entries per response, keep highest-value ones
    if len(entries) > MAX_MEMORIES_PER_RESPONSE:
        # Prioritise: corrections > decisions > facts > preferences > project > others
        type_priority = {"correction": 0, "decision": 1, "fact": 2, "preference": 3,
                         "person": 4, "skill": 5, "workflow": 6, "project": 7}
        entries.sort(key=lambda e: type_priority.get(e.get("type", ""), 99))
        dropped = entries[MAX_MEMORIES_PER_RESPONSE:]
        entries = entries[:MAX_MEMORIES_PER_RESPONSE]
        log(f"Write throttle: kept {len(entries)}, dropped {len(dropped)}")

    emb = get_embedder()
    conn = sqlite3.connect(DB_PATH)
    project = get_session_project(conn, session_id)
    inserted = 0

    for entry in entries:
        mem_type = entry.get("type", "fact")
        topic = entry.get("topic", "unknown")
        content = entry.get("content", "")
        source_start = entry.get("source_start")
        source_end = entry.get("source_end")
        # Augment embedding text with project to push unrelated domains apart in vector space
        project_prefix = f"{project} " if project else ""
        search_text = f"{project_prefix}{mem_type} {topic} {content}"

        embedding_blob = None
        if emb:
            try:
                vec = emb.embed(search_text, allow_slow=False)
                if vec is None:
                    # Daemon not available — store without embedding, skip dedup
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
                if nearest and nearest[0]["similarity"] >= 0.6 and nearest[0]["similarity"] < DEDUP_THRESHOLD:
                    match = nearest[0]
                    if _has_negation_mismatch(content, match["content"]):
                        log(f"Negation mismatch: '{content[:40]}' vs '{match['content'][:40]}' — dampening both")
                        conn.execute(
                            "UPDATE memories SET confidence = MAX(confidence - 0.1, 0) WHERE id = ?",
                            (match["id"],)
                        )
                        record_metric(session_id, "negation_contradiction", f"{mem_type}/{topic}")

            except Exception as e:
                log(f"Embedding error: {e}")

        # No semantic match — fall back to exact type+topic check
        same_topic = conn.execute(
            "SELECT id, content FROM memories WHERE type = ? AND topic = ?",
            (mem_type, topic)
        ).fetchone()

        if same_topic:
            old_content = same_topic[1]
            # Check if this is a true update or a distinct variant
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

            if old_content and old_content != content and old_sim < 0.8:
                # Low similarity despite same type+topic — treat as distinct variant, not contradiction
                log(f"Distinct variant: type={mem_type} topic={topic} (sim={old_sim:.2f}) — inserting as new")
                conn.execute(
                    "INSERT INTO memories (type, topic, content, embedding, session_id, project, source_start, source_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (mem_type, topic, content, embedding_blob, session_id, project, source_start, source_end)
                )
                if embedding_blob and emb:
                    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    emb.upsert_vec_index(conn, new_id, embedding_blob)
            else:
                # High similarity or identical — true update
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


def register_session(session_id, transcript_path):
    """Register this session in the sessions table, extracting parent if available."""
    if not session_id:
        return
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if existing:
        conn.close()
        return

    # Extract parent session from first user message in transcript
    parent_session_id = None
    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") in ("user", "assistant"):
                    # First message's parentUuid being None means root session
                    # If sessionId differs from our session_id, it's a parent
                    entry_session = entry.get("sessionId", "")
                    if entry_session and entry_session != session_id:
                        parent_session_id = entry_session
                    break
    except (FileNotFoundError, PermissionError):
        pass

    # Inherit project label from parent session
    project = None
    if parent_session_id:
        row = conn.execute(
            "SELECT project FROM sessions WHERE session_id = ?", (parent_session_id,)
        ).fetchone()
        if row:
            project = row[0]

    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, parent_session_id, project, transcript_path) VALUES (?, ?, ?, ?)",
        (session_id, parent_session_id, project, transcript_path)
    )
    conn.commit()
    conn.close()
    if parent_session_id:
        log(f"Session {session_id[:8]}... parent: {parent_session_id[:8]}... project: {project}")
    else:
        log(f"Session {session_id[:8]}... (root)")


def get_adaptive_threshold_boost():
    """Check recent retrieval outcomes. If harmful/neutral rate is high, boost the similarity floor."""
    try:
        conn = sqlite3.connect(DB_PATH)
        recent = conn.execute("""
            SELECT event, COUNT(*) FROM metrics
            WHERE event IN ('retrieval_useful', 'retrieval_neutral', 'retrieval_harmful')
            AND created_at > datetime('now', '-7 days')
            GROUP BY event
        """).fetchall()
        conn.close()

        counts = {r[0]: r[1] for r in recent}
        total = sum(counts.values())
        if total < 5:
            return 0.0  # Not enough data

        harmful_rate = (counts.get("retrieval_harmful", 0) + counts.get("retrieval_neutral", 0)) / total
        if harmful_rate > 0.5:
            return 0.10  # Significant noise — tighten substantially
        elif harmful_rate > 0.3:
            return 0.05  # Some noise — tighten slightly
        return 0.0
    except Exception:
        return 0.0


def retrieve_context(context_need, session_id=None):
    """Search the brain for memories matching the context need. Returns structured XML context.

    Uses composite scoring, quality gates, adaptive thresholds, and epistemic qualifiers."""
    import time
    from datetime import datetime as dt
    start = time.time()
    conn = sqlite3.connect(DB_PATH)
    project = get_session_project(conn, session_id)

    emb = get_embedder()
    project_results = []
    global_results = []

    # Adaptive threshold: boost floors if recent retrieval outcomes are poor
    threshold_boost = get_adaptive_threshold_boost()
    if threshold_boost > 0:
        log(f"Adaptive threshold boost: +{threshold_boost:.2f}")

    if emb:
        try:
            all_results = emb.find_similar(conn, context_need, current_project=project)
            global_threshold = (L3_GLOBAL_SIM_WITH_PROJECT if project else L3_GLOBAL_SIM_WITHOUT_PROJECT) + threshold_boost
            project_threshold = L3_PROJECT_SIM_THRESHOLD + threshold_boost

            for r in all_results:
                if project and r.get("project") == project and r["similarity"] >= project_threshold:
                    project_results.append(r)
                elif r["similarity"] >= global_threshold:
                    global_results.append(r)
        except Exception as e:
            log(f"Context retrieval embedding error: {e}")

    # Fall back to FTS if no embedding results
    if not project_results and not global_results:
        try:
            rows = conn.execute("""
                SELECT m.id, m.type, m.topic, m.content, m.updated_at, m.project, m.session_id, m.confidence
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ?
                ORDER BY rank LIMIT 15
            """, (context_need,)).fetchall()
            for r in rows:
                entry = {"id": r[0], "type": r[1], "topic": r[2], "content": r[3],
                         "updated_at": r[4], "project": r[5], "session_id": r[6],
                         "confidence": r[7] or 0.7, "similarity": 0.5, "score": 0.5}
                if project and r[5] == project:
                    project_results.append(entry)
                else:
                    global_results.append(entry)
        except Exception:
            pass

    conn.close()

    elapsed_ms = (time.time() - start) * 1000
    total = len(project_results) + len(global_results)
    record_metric(session_id, "context_retrieval", context_need[:100], total)
    record_metric(session_id, "retrieval_latency_ms", context_need[:50], elapsed_ms)
    log(f"Retrieval: {len(project_results)} project + {len(global_results)} global in {elapsed_ms:.0f}ms")

    if not project_results and not global_results:
        record_metric(session_id, "context_empty", context_need[:100])
        return None

    def recency_days(updated_at_str):
        try:
            updated = dt.strptime(updated_at_str[:19], "%Y-%m-%d %H:%M:%S")
            return max(0, (dt.now() - updated).days)
        except Exception:
            return -1

    def reliability(r):
        """Epistemic qualifier: combines confidence and score into a single reliability signal."""
        return r.get("score", r.get("confidence", 0.7))

    def format_entry(r):
        rel = reliability(r)
        rel_label = "strong" if rel >= 0.6 else "moderate" if rel >= 0.4 else "weak"
        days = recency_days(r.get("updated_at", ""))
        has_source = r.get("source_start") is not None
        source_attr = ' ctx="y"' if has_source else ""
        return (
            f'  <entry id="{r["id"]}" reliability="{rel_label}" days="{days}"{source_attr}>'
            f'{r["content"]}</entry>'
        )

    lines = ['<brain_context query="{}" current_project="{}">'.format(
        context_need.replace('"', '&quot;'),
        project or "none"
    )]

    if project_results:
        project_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        lines.append(f'  <scope level="project" name="{project}" weight="high">')
        for r in project_results[:L3_MAX_PROJECT_RESULTS]:
            lines.append("  " + format_entry(r))
        lines.append("  </scope>")

    if global_results:
        global_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        lines.append('  <scope level="global" weight="low">')
        for r in global_results[:L3_MAX_GLOBAL_RESULTS]:
            lines.append("  " + format_entry(r))
        lines.append("  </scope>")

    lines.append("</brain_context>")

    return "\n".join(lines)


STAGED_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", ".staged_context")


def layer2_cross_project_search(keywords_list, session_id=None):
    """Layer 2: Search global memories for cross-project relevance using keywords.
    Stages results for the next UserPromptSubmit hook injection."""
    from config import L2_SIM_THRESHOLD, L2_MAX_RESULTS

    if not keywords_list:
        return

    emb = get_embedder()
    if not emb:
        return

    conn = sqlite3.connect(DB_PATH)
    project = get_session_project(conn, session_id)
    conn.close()

    query = " ".join(keywords_list)
    try:
        conn = sqlite3.connect(DB_PATH)
        results = emb.find_similar(conn, query, threshold=L2_SIM_THRESHOLD,
                                   limit=L2_MAX_RESULTS * 2, current_project=project)
        conn.close()
    except Exception as e:
        log(f"Layer 2 search error: {e}")
        return

    # Filter: cross-project only, strong match, not current project
    cross_project = [r for r in results
                     if r.get("project") != project
                     and r["similarity"] >= L2_SIM_THRESHOLD][:L2_MAX_RESULTS]

    if not cross_project:
        log(f"Layer 2: no cross-project matches for keywords: {query[:50]}")
        return

    # Format as brain_context XML
    from datetime import datetime as dt
    lines = [f'<brain_context query="cross-project keywords: {query[:60]}" current_project="{project or "none"}" layer="cross-project">']
    lines.append('  <scope level="global" weight="low">')
    for r in cross_project:
        proj = r.get("project") or "global"
        conf = r.get("confidence", 0.7)
        score = r.get("score", conf)
        days = 0
        try:
            updated = dt.strptime(r["updated_at"][:19], "%Y-%m-%d %H:%M:%S")
            days = max(0, (dt.now() - updated).days)
        except Exception:
            pass
        rel = "strong" if score >= 0.6 else "moderate" if score >= 0.4 else "weak"
        lines.append(
            f'    <entry id="{r["id"]}" type="{r["type"]}" topic="{r["topic"]}" '
            f'project="{proj}" date="{r["updated_at"]}" confidence="{conf:.2f}" '
            f'score="{score:.2f}" recency_days="{days}" reliability="{rel}" similarity="{r["similarity"]:.2f}">'
            f'{r["content"]}</entry>'
        )
    lines.append('  </scope>')
    lines.append('</brain_context>')

    staged_xml = "\n".join(lines)

    # Stage for next prompt
    try:
        with open(STAGED_PATH, "r") as f:
            staged = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        staged = {}
    staged[session_id] = staged_xml
    with open(STAGED_PATH, "w") as f:
        json.dump(staged, f)

    log(f"Layer 2: staged {len(cross_project)} cross-project entries for next prompt")
    record_metric(session_id, "layer2_staged", query[:100], len(cross_project))


CONTEXT_CACHE_SIM_THRESHOLD = 0.9  # Semantic similarity threshold for cache hit


def load_context_cache(session_id):
    """Load cached context_need embeddings for this session."""
    try:
        with open(CONTEXT_CACHE_PATH, "r") as f:
            cache = json.load(f)
        return cache.get(session_id, [])  # List of {"text": str, "embedding_hex": str}
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_context_cache(session_id, served_needs):
    """Save context_need embeddings for this session."""
    try:
        with open(CONTEXT_CACHE_PATH, "r") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    cache[session_id] = served_needs
    with open(CONTEXT_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def is_context_cached(context_need, served_needs, emb):
    """Check if a semantically similar context_need has already been served."""
    if not emb or not served_needs:
        # Fallback to exact string match
        return any(s.get("text") == context_need for s in served_needs)
    try:
        import numpy as np
        query_vec = emb.embed(context_need)
        for cached in served_needs:
            cached_vec = np.frombuffer(bytes.fromhex(cached["embedding_hex"]), dtype=np.float32)
            sim = emb.cosine_similarity(query_vec, cached_vec)
            if sim >= CONTEXT_CACHE_SIM_THRESHOLD:
                return True
    except Exception:
        # Fallback to exact string match
        return any(s.get("text") == context_need for s in served_needs)
    return False


def add_to_context_cache(context_need, served_needs, emb):
    """Add a context_need to the cache with its embedding."""
    entry = {"text": context_need}
    if emb:
        try:
            vec = emb.embed(context_need)
            entry["embedding_hex"] = emb.to_blob(vec).hex()
        except Exception:
            pass
    served_needs.append(entry)
    return served_needs


def get_continuation_count(session_id):
    """Get how many times we've re-prompted this session."""
    try:
        with open(CONTINUATION_COUNT_PATH, "r") as f:
            counts = json.load(f)
        return counts.get(session_id, 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def increment_continuation(session_id):
    """Increment and return the continuation count."""
    try:
        with open(CONTINUATION_COUNT_PATH, "r") as f:
            counts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        counts = {}
    counts[session_id] = counts.get(session_id, 0) + 1
    with open(CONTINUATION_COUNT_PATH, "w") as f:
        json.dump(counts, f)
    return counts[session_id]


def reset_continuation(session_id):
    """Reset continuation count (called when a response completes normally)."""
    try:
        with open(CONTINUATION_COUNT_PATH, "r") as f:
            counts = json.load(f)
        if session_id in counts:
            del counts[session_id]
            with open(CONTINUATION_COUNT_PATH, "w") as f:
                json.dump(counts, f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def auto_label_project(session_id, cwd):
    """Heuristically label a session's project based on the working directory."""
    if not session_id or not cwd:
        return
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT project FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row and row[0]:
        conn.close()
        return  # Already labelled

    # Derive project name from directory
    project_name = os.path.basename(cwd.rstrip("/"))
    if not project_name or project_name in (".", "/", "home"):
        conn.close()
        return

    conn.execute("UPDATE sessions SET project = ? WHERE session_id = ?", (project_name, session_id))
    conn.commit()
    conn.close()
    log(f"Auto-labelled project: {project_name} (from cwd: {cwd})")


def get_last_assistant_text(transcript_path):
    """Read the last assistant message from the transcript JSONL."""
    last_text = ""
    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Transcript format: entry["message"]["role"]
                msg = entry.get("message", entry)
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        if text_parts:
                            last_text = "\n".join(text_parts)
                    elif isinstance(content, str):
                        last_text = content
    except (FileNotFoundError, PermissionError):
        pass
    return last_text


def main():
    raw = sys.stdin.read()
    log(f"--- Hook fired ---")
    hook_input = json.loads(raw)

    is_continuation = hook_input.get("stop_hook_active", False)
    transcript_path = hook_input.get("transcript_path", "")
    session_id = hook_input.get("session_id", "")
    cwd = hook_input.get("cwd", "")

    # Register session and track parent chain
    register_session(session_id, transcript_path)

    # Auto-label project from working directory
    auto_label_project(session_id, cwd)

    # Check continuation cap
    if is_continuation:
        count = get_continuation_count(session_id)
        if count >= MAX_CONTINUATIONS:
            log(f"Continuation cap reached ({count}/{MAX_CONTINUATIONS}) — forcing stop")
            record_metric(session_id, "continuation_cap_hit", None, count)
            reset_continuation(session_id)
            sys.exit(0)

    # Use last_assistant_message — this is the current response
    text = hook_input.get("last_assistant_message", "")

    if not text:
        log("No text found, allowing stop")
        sys.exit(0)

    log(f"Text length: {len(text)}, has <memory>: {'<memory>' in text}, continuation: {is_continuation}")

    # Parse memory block
    entries, complete, remaining, context, context_need, confidence_updates, retrieval_outcome, keywords = parse_memory_block(text)

    # No memory block found
    if entries is None and complete is None:
        record_metric(session_id, "missing_memory_block", None, 1 if is_continuation else 0)
        if is_continuation:
            log("Missing memory block on continuation — allowing stop to prevent loop")
            reset_continuation(session_id)
            sys.exit(0)
        increment_continuation(session_id)

        # Check if there was an attempted but malformed block
        has_open_tag = "<memory>" in text
        has_close_tag = "</memory>" in text
        if has_open_tag:
            record_metric(session_id, "malformed_memory_block")
            hint = "Your <memory> block could not be parsed. "
            if not has_close_tag:
                hint += "Missing closing </memory> tag. "
            hint += "Use this exact format:\n<memory>\n- type: fact\n- topic: example\n- content: one line description\n- complete: true\n</memory>"
            result = {"decision": "block", "reason": hint}
        else:
            result = {
                "decision": "block",
                "reason": "Response missing required <memory> block. Add a <memory> block with at least complete: true before finishing."
            }
        print(json.dumps(result))
        sys.exit(0)

    log(f"Parsed: entries={len(entries) if entries else 0}, complete={complete}, remaining={remaining}, context={context}, context_need={context_need}, conf_updates={len(confidence_updates)}")

    # Apply confidence updates
    if confidence_updates:
        applied = apply_confidence_updates(confidence_updates, session_id=session_id)
        record_metric(session_id, "confidence_updates", None, applied)

    # Record retrieval outcome (system-level learning signal)
    if retrieval_outcome:
        record_metric(session_id, f"retrieval_{retrieval_outcome}", context_need[:100] if context_need else None)
        log(f"Retrieval outcome: {retrieval_outcome}")

    # Insert memories into DB
    if entries:
        count = insert_memories(entries, session_id=session_id)
        record_metric(session_id, "memories_stored", None, count)
        log(f"Stored {count} memories (session: {session_id[:8]}...)" if session_id else f"Stored {count} memories")

    # Record dedup stats
    record_metric(session_id, "hook_fired", f"entries={len(entries) if entries else 0}")

    # Layer 2: cross-project keyword search (stages for next prompt, doesn't block)
    if keywords and not is_continuation:
        layer2_cross_project_search(keywords, session_id=session_id)

    # Check context sufficiency — retrieve and inject if insufficient
    LOW_INFO_STOPLIST = {"help", "continue", "more", "yes", "no", "ok", "thanks", "done", "info", "more info"}
    if context == "insufficient" and context_need and not is_continuation:
        # Pre-filter: skip low-information queries
        need_words = set(context_need.lower().split())
        if len(context_need) < 8 or need_words <= LOW_INFO_STOPLIST:
            log(f"Pre-filter: skipping low-info context_need: {context_need}")
            record_metric(session_id, "context_prefiltered", context_need[:100])
        else:
            record_metric(session_id, "context_requested", context_need[:100])
            emb = get_embedder()
            served = load_context_cache(session_id)
            if not is_context_cached(context_need, served, emb):
                retrieved = retrieve_context(context_need, session_id=session_id)
                if retrieved:
                    # Weak-entry suppression: don't inject if top result is unreliable
                    # Parse the first score from the XML to check reliability
                    import re as _re
                    score_match = _re.search(r'score="([0-9.]+)"', retrieved)
                    top_score = float(score_match.group(1)) if score_match else 1.0
                    if top_score < 0.4:
                        log(f"Weak-entry suppression: top score {top_score:.2f} — skipping injection")
                        record_metric(session_id, "context_weak_suppressed", context_need[:100])
                    else:
                        served = add_to_context_cache(context_need, served, emb)
                        save_context_cache(session_id, served)
                        record_metric(session_id, "context_served", context_need[:100])
                        log(f"Context retrieval for: {context_need[:50]}...")
                        increment_continuation(session_id)
                        result = {
                            "decision": "block",
                            "reason": f"BRAIN CONTEXT:\n{retrieved}"
                        }
                        print(json.dumps(result))
                        sys.exit(0)
                else:
                    log(f"No context found for: {context_need}")
            else:
                record_metric(session_id, "context_cache_hit", context_need[:100])
                log(f"Context already served (semantic match) for: {context_need[:50]}... — skipping")

    # Check completeness
    if complete is False:
        count = get_continuation_count(session_id)
        if count >= MAX_CONTINUATIONS:
            log(f"Completeness re-prompt cap reached ({count}/{MAX_CONTINUATIONS}) — forcing stop")
            record_metric(session_id, "completeness_cap_hit", remaining, count)
            reset_continuation(session_id)
            sys.exit(0)
        increment_continuation(session_id)
        llm_reason = f"Response marked incomplete. Continue with: {remaining}" if remaining else "Response marked incomplete. Continue."
        result = {
            "decision": "block",
            "reason": llm_reason
        }
        print(json.dumps(result))
        sys.exit(0)

    # All good — reset continuation counter and allow stop
    reset_continuation(session_id)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail open — never block the user due to a hook bug
        try:
            log(f"HOOK CRASH: {e}")
            record_metric("", "hook_crash", str(e))
        except Exception:
            pass
        sys.exit(0)
