#!/usr/bin/env python3
"""Query the Cairn database. Used by Claude Code to retrieve context."""

import sqlite3
import sys
import os

# Re-exec under venv python if not already in the venv
_venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "python3")
if os.path.exists(_venv_python) and sys.prefix == sys.base_prefix:
    os.execv(_venv_python, [_venv_python] + sys.argv)

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")


def search(query, limit=10):
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT m.id, m.type, m.topic, m.content, m.updated_at
        FROM memories_fts f
        JOIN memories m ON f.rowid = m.id
        WHERE memories_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (query, limit)).fetchall()
    conn.close()
    return rows


def list_by_type(memory_type, limit=20):
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, type, topic, content, updated_at
        FROM memories WHERE type = ?
        ORDER BY updated_at DESC LIMIT ?
    """, (memory_type, limit)).fetchall()
    conn.close()
    return rows


def list_recent(limit=20):
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, type, topic, content, updated_at
        FROM memories ORDER BY updated_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return rows


def semantic_search(query, limit=10, threshold=0.5):
    try:
        import embeddings as emb
        conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
        results = emb.find_similar(conn, query, threshold=threshold, limit=limit)
        conn.close()
        return results
    except ImportError:
        print("sentence-transformers not installed, falling back to FTS")
        return None


def _parse_date(date_str):
    """Parse a date string into ISO format. Supports:
    - ISO dates: 2026-03-22, 2026-03-22T14:00
    - Relative: today, yesterday, 3d (days ago), 2w (weeks ago), 1m (months ago)
    """
    from datetime import datetime, timedelta
    date_str = date_str.strip().lower()
    if date_str == "today":
        return datetime.now().strftime("%Y-%m-%d")
    if date_str == "yesterday":
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # Relative: 3d, 2w, 1m
    import re
    rel = re.match(r"^(\d+)([dwm])$", date_str)
    if rel:
        n, unit = int(rel.group(1)), rel.group(2)
        if unit == "d":
            return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")
        elif unit == "w":
            return (datetime.now() - timedelta(weeks=n)).strftime("%Y-%m-%d")
        elif unit == "m":
            return (datetime.now() - timedelta(days=n * 30)).strftime("%Y-%m-%d")
    # Assume ISO format
    return date_str


def list_by_date(since=None, until=None, limit=50):
    """List memories filtered by date range."""
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    conditions = []
    params = []
    if since:
        conditions.append("updated_at >= ?")
        params.append(_parse_date(since))
    if until:
        # Include the full day
        parsed = _parse_date(until)
        if len(parsed) == 10:  # date only, no time
            parsed += " 23:59:59"
        conditions.append("updated_at <= ?")
        params.append(parsed)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)
    rows = conn.execute(f"""
        SELECT id, type, topic, content, updated_at
        FROM memories {where}
        ORDER BY updated_at DESC LIMIT ?
    """, params).fetchall()
    conn.close()
    return rows


def list_by_session(session_id, limit=50):
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, type, topic, content, updated_at
        FROM memories WHERE session_id LIKE ?
        ORDER BY updated_at DESC LIMIT ?
    """, (f"{session_id}%", limit)).fetchall()
    conn.close()
    return rows


def show_context(memory_id, margin=3):
    """Show the conversation context around where a memory was recorded."""
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    row = conn.execute(
        "SELECT id, type, topic, content, session_id, source_start, source_end FROM memories WHERE id = ?",
        (memory_id,)
    ).fetchone()
    if not row:
        print(f"No memory with id {memory_id}")
        conn.close()
        return

    print(f"=== Memory [{row[0]}] {row[1]}/{row[2]} ===")
    print(f"  {row[3]}")

    session_id = row[4]
    source_start = row[5]
    source_end = row[6]

    if not session_id:
        print("  No session ID — cannot locate transcript.")
        conn.close()
        return

    session = conn.execute(
        "SELECT transcript_path FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()

    if not session or not session[0]:
        print(f"  Session {session_id[:12]}... — no transcript path recorded.")
        return

    transcript_path = session[0]
    if not os.path.exists(transcript_path):
        print(f"  Transcript not found: {transcript_path}")
        return

    if source_start is None:
        print(f"  No source_messages recorded — showing transcript search instead.")
        # Fallback: grep transcript for content keywords
        import json as _json
        keywords = row[3].lower().split()[:3]
        msg_num = 0
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = _json.loads(line.strip())
                    msg = entry.get("message", entry)
                    if msg.get("role") in ("user", "assistant"):
                        msg_num += 1
                except Exception:
                    continue
        print(f"  Transcript: {transcript_path} ({msg_num} messages)")
        print(f"  Search for: {row[2]} / {' '.join(keywords)}")
        return

    print(f"  Source messages: {source_start}–{source_end} (LLM-estimated)")
    print(f"  Transcript: {transcript_path}")

    # Read transcript and extract message range
    # Count only text-bearing messages (not tool calls) to align with LLM's perception of "messages"
    import json as _json
    show_start = max(1, source_start - margin)
    show_end = source_end + margin
    turn_num = 0

    print(f"\n  --- Conversation context (turns {show_start}–{show_end}) ---")
    with open(transcript_path) as f:
        for line in f:
            try:
                entry = _json.loads(line.strip())
                msg = entry.get("message", entry)
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue

                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [b.get("text", "") for b in content
                                  if isinstance(b, dict) and b.get("type") == "text"]
                    content = " ".join(text_parts)
                elif not isinstance(content, str):
                    continue

                # Skip empty messages (tool calls with no text)
                if not content or not content.strip():
                    continue

                turn_num += 1
                if turn_num < show_start:
                    continue
                if turn_num > show_end:
                    break

                # Truncate long messages
                display = content[:200] + "..." if len(content) > 200 else content

                marker = " <<<" if source_start <= turn_num <= source_end else ""
                print(f"  [{turn_num}] {role}: {display}{marker}")
            except Exception:
                continue


def verify_sources():
    """Analyse accuracy of LLM source_messages estimates against actual transcript content."""
    import json as _json

    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    memories = conn.execute("""
        SELECT m.id, m.type, m.topic, m.content, m.session_id, m.source_start, m.source_end
        FROM memories m
        WHERE m.source_start IS NOT NULL AND m.source_end IS NOT NULL
    """).fetchall()

    if not memories:
        print("No memories with source_messages recorded yet.")
        conn.close()
        return

    # Cache transcripts
    transcripts = {}
    sessions = conn.execute("SELECT session_id, transcript_path FROM sessions").fetchall()
    session_paths = {s[0]: s[1] for s in sessions}
    conn.close()

    results = []
    for mem in memories:
        mem_id, mem_type, topic, content, session_id, est_start, est_end = mem
        transcript_path = session_paths.get(session_id)
        if not transcript_path or not os.path.exists(transcript_path):
            continue

        # Load transcript messages if not cached
        if transcript_path not in transcripts:
            messages = []
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = _json.loads(line.strip())
                        msg = entry.get("message", entry)
                        role = msg.get("role", "")
                        if role not in ("user", "assistant"):
                            continue
                        msg_content = msg.get("content", "")
                        if isinstance(msg_content, list):
                            text_parts = [b.get("text", "") for b in msg_content
                                          if isinstance(b, dict) and b.get("type") == "text"]
                            msg_content = " ".join(text_parts)
                        # Only count text-bearing messages (align with LLM turn perception)
                        if not msg_content or not msg_content.strip():
                            continue
                        messages.append({"num": len(messages) + 1, "role": role, "text": msg_content.lower()})
                    except Exception:
                        continue
            transcripts[transcript_path] = messages

        messages = transcripts[transcript_path]

        # Search for content keywords in transcript
        keywords = [w.lower() for w in content.split() if len(w) > 3][:5]
        if not keywords:
            continue

        # Find messages that match the most keywords
        matches = []
        for msg in messages:
            hits = sum(1 for k in keywords if k in msg["text"])
            if hits >= max(2, len(keywords) // 2):
                matches.append(msg["num"])

        if not matches:
            results.append({
                "id": mem_id, "topic": topic,
                "estimated": f"{est_start}-{est_end}",
                "actual": "not found",
                "drift": None
            })
            continue

        actual_start = min(matches)
        actual_end = max(matches)

        # Calculate drift
        drift_start = est_start - actual_start
        drift_end = est_end - actual_end
        avg_drift = (abs(drift_start) + abs(drift_end)) / 2

        results.append({
            "id": mem_id, "topic": topic,
            "estimated": f"{est_start}-{est_end}",
            "actual": f"{actual_start}-{actual_end}",
            "drift_start": drift_start,
            "drift_end": drift_end,
            "avg_drift": avg_drift
        })

    # Report
    print(f"=== Source Accuracy Report ({len(results)} memories with source ranges) ===\n")

    found = [r for r in results if r.get("avg_drift") is not None]
    not_found = [r for r in results if r.get("drift") is None and "avg_drift" not in r]

    if found:
        print(f"{'ID':>5} {'Topic':<30} {'Estimated':>12} {'Actual':>12} {'Drift':>8}")
        print("-" * 75)
        for r in found:
            print(f"{r['id']:>5} {r['topic']:<30} {r['estimated']:>12} {r['actual']:>12} {r['avg_drift']:>7.1f}")

        avg_total = sum(r["avg_drift"] for r in found) / len(found)
        print(f"\nAverage drift: {avg_total:.1f} messages")
        print(f"Memories within ±5 messages: {sum(1 for r in found if r['avg_drift'] <= 5)}/{len(found)}")
        print(f"Memories within ±10 messages: {sum(1 for r in found if r['avg_drift'] <= 10)}/{len(found)}")

    if not_found:
        print(f"\nContent not found in transcript: {len(not_found)} memories")
        for r in not_found:
            print(f"  [{r['id']}] {r['topic']}: {r['estimated']}")

    if not found and not not_found:
        print("No memories with source ranges found.")


def show_history(memory_id):
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    current = conn.execute(
        "SELECT id, type, topic, content, session_id, updated_at FROM memories WHERE id = ?",
        (memory_id,)
    ).fetchone()
    if not current:
        print(f"No memory with id {memory_id}")
        return
    print(f"=== Memory [{current['id']}] {current['type']}/{current['topic']} ===")
    print(f"  Current: {current['content']}")
    print(f"  Updated: {current['updated_at']}")
    print(f"  Session: {current['session_id'] or 'unknown'}")
    history = conn.execute("""
        SELECT content, session_id, changed_at FROM memory_history
        WHERE memory_id = ? ORDER BY changed_at DESC
    """, (memory_id,)).fetchall()
    if history:
        print(f"\n  --- History ({len(history)} prior versions) ---")
        for h in history:
            print(f"  [{h['changed_at']}] {h['content']}")
            print(f"    Session: {h['session_id'] or 'unknown'}")
    else:
        print("  No prior versions.")
    conn.close()


def delete_memory(memory_id):
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    existing = conn.execute("SELECT id, type, topic, content FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not existing:
        print(f"No memory with id {memory_id}")
        return
    conn.execute("DELETE FROM memory_history WHERE memory_id = ?", (memory_id,))
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    print(f"Deleted memory {memory_id}: {existing[1]}/{existing[2]}")


def update_memory(memory_id, new_content):
    """Update a memory's content in place, preserving history."""
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    existing = conn.execute("SELECT id, type, topic, content FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not existing:
        print(f"No memory with id {memory_id}")
        return
    old_content = existing[3]
    # The version trigger fires automatically on UPDATE of content
    conn.execute(
        "UPDATE memories SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (new_content, memory_id)
    )
    conn.commit()
    conn.close()
    print(f"Updated memory {memory_id}: {existing[1]}/{existing[2]}")
    print(f"  Old: {old_content}")
    print(f"  New: {new_content}")


def show_session_chain(session_id):
    """Show a session and all linked sessions (parent/child chain)."""
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row

    # Walk up to find root
    chain = []
    current = session_id
    while current:
        row = conn.execute(
            "SELECT session_id, parent_session_id, started_at FROM sessions WHERE session_id LIKE ?",
            (f"{current}%",)
        ).fetchone()
        if not row:
            break
        chain.insert(0, row)
        current = row["parent_session_id"]

    # Walk down from root to find children
    if chain:
        root_id = chain[0]["session_id"]
        children = conn.execute("""
            WITH RECURSIVE chain(sid) AS (
                SELECT session_id FROM sessions WHERE session_id = ?
                UNION ALL
                SELECT s.session_id FROM sessions s JOIN chain c ON s.parent_session_id = c.sid
            )
            SELECT s.session_id, s.parent_session_id, s.started_at
            FROM chain c JOIN sessions s ON s.session_id = c.sid
            ORDER BY s.started_at
        """, (root_id,)).fetchall()
        chain = children

    if not chain:
        print(f"No session found matching {session_id}")
        conn.close()
        return

    print("=== Session Chain ===")
    for i, s in enumerate(chain):
        sid = s["session_id"]
        mem_count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        prefix = "  " * i + ("-> " if i > 0 else "")
        marker = " <-- current" if sid.startswith(session_id) else ""
        print(f"{prefix}{sid[:12]}... ({s['started_at']}) [{mem_count} memories]{marker}")

    conn.close()


def stats():
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    with_emb = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL").fetchone()[0]
    history_count = conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0]
    types = conn.execute("SELECT type, COUNT(*) as c FROM memories GROUP BY type ORDER BY c DESC").fetchall()
    sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM memories WHERE session_id IS NOT NULL").fetchone()[0]
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    print("=== Memory Stats ===")
    print(f"Total memories: {total}")
    print(f"With embeddings: {with_emb}")
    print(f"History entries: {history_count}")
    print(f"Memory sessions: {sessions}")
    print(f"Tracked sessions: {session_count}")
    print(f"Types: {', '.join(f'{t[0]}={t[1]}' for t in types)}")

    # Performance metrics
    try:
        metrics = conn.execute("""
            SELECT event, COUNT(*) as c, AVG(value) as avg_val
            FROM metrics GROUP BY event ORDER BY c DESC
        """).fetchall()
        if metrics:
            print("\n=== Performance Metrics ===")
            for m in metrics:
                avg = f" (avg={m[2]:.1f})" if m[2] is not None else ""
                print(f"  {m[0]}: {m[1]}{avg}")

        # Retrieval latency stats
        latency = conn.execute("""
            SELECT MIN(value), AVG(value), MAX(value)
            FROM metrics WHERE event = 'retrieval_latency_ms'
        """).fetchone()
        if latency and latency[0] is not None:
            print(f"\n=== Retrieval Latency ===")
            print(f"  Min: {latency[0]:.0f}ms  Avg: {latency[1]:.0f}ms  Max: {latency[2]:.0f}ms")
    except Exception:
        pass  # metrics table may not exist yet

    # Drift detection: confidence distribution
    try:
        conf_dist = conn.execute("""
            SELECT
                COUNT(CASE WHEN confidence < 0.3 THEN 1 END) as suppressed,
                COUNT(CASE WHEN confidence >= 0.3 AND confidence < 0.5 THEN 1 END) as low,
                COUNT(CASE WHEN confidence >= 0.5 AND confidence < 0.7 THEN 1 END) as medium,
                COUNT(CASE WHEN confidence >= 0.7 AND confidence < 0.9 THEN 1 END) as high,
                COUNT(CASE WHEN confidence >= 0.9 THEN 1 END) as very_high,
                AVG(confidence) as avg_conf
            FROM memories
        """).fetchone()
        if conf_dist:
            print(f"\n=== Confidence Distribution ===")
            print(f"  Suppressed (<0.3): {conf_dist[0]}  Low (0.3-0.5): {conf_dist[1]}  Med (0.5-0.7): {conf_dist[2]}  High (0.7-0.9): {conf_dist[3]}  Very high (>0.9): {conf_dist[4]}")
            print(f"  Average confidence: {conf_dist[5]:.3f}")

        overwrite_count = conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0]
        contradiction_count = conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE event = 'contradiction_detected'"
        ).fetchone()[0] if conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE event LIKE 'contradiction%'"
        ).fetchone()[0] else 0
        print(f"\n=== Drift Indicators ===")
        print(f"  Memory overwrites (history entries): {overwrite_count}")
        print(f"  Contradictions detected: {contradiction_count}")
    except Exception:
        pass

    conn.close()


def review(threshold_low=0.3, threshold_high=0.6):
    """Surface memories with low or uncertain confidence for user inspection."""
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    # Below retrieval threshold — effectively suppressed
    suppressed = conn.execute("""
        SELECT id, type, topic, content, confidence, updated_at, project
        FROM memories WHERE confidence < ? ORDER BY confidence ASC
    """, (threshold_low,)).fetchall()

    # Uncertain range — retrieved but not strongly trusted
    uncertain = conn.execute("""
        SELECT id, type, topic, content, confidence, updated_at, project
        FROM memories WHERE confidence >= ? AND confidence < ? ORDER BY confidence ASC
    """, (threshold_low, threshold_high)).fetchall()

    if suppressed:
        print(f"=== Suppressed (confidence < {threshold_low}) — not returned in retrieval ===")
        for r in suppressed:
            proj = r[6] or "global"
            print(f"  [{r[0]}] {r[1]}/{r[2]} (conf={r[4]:.2f}, {r[5]}) [{proj}]")
            print(f"      {r[3]}")

    if uncertain:
        print(f"\n=== Uncertain (confidence {threshold_low}–{threshold_high}) ===")
        for r in uncertain:
            proj = r[6] or "global"
            print(f"  [{r[0]}] {r[1]}/{r[2]} (conf={r[4]:.2f}, {r[5]}) [{proj}]")
            print(f"      {r[3]}")

    if not suppressed and not uncertain:
        print("All memories have confidence >= 0.6")

    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"\nSummary: {len(suppressed)} suppressed, {len(uncertain)} uncertain, {total - len(suppressed) - len(uncertain)} healthy (of {total} total)")
    conn.close()


def compact(project_name=None, limit=100):
    """Produce a dense cairn dump suitable for LLM ingestion at session start."""
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    if project_name:
        rows = conn.execute("""
            SELECT type, topic, content, updated_at FROM memories
            WHERE project = ? ORDER BY type, updated_at DESC LIMIT ?
        """, (project_name, limit)).fetchall()
        print(f"# Cairn dump: {project_name}")
    else:
        rows = conn.execute("""
            SELECT type, topic, content, updated_at FROM memories
            ORDER BY type, updated_at DESC LIMIT ?
        """, (limit,)).fetchall()
        print("# Cairn dump: all memories")

    if not rows:
        print("(empty)")
        conn.close()
        return

    current_type = None
    for r in rows:
        if r[0] != current_type:
            current_type = r[0]
            print(f"\n## {current_type}s")
        print(f"- [{r[1]}] {r[2]} ({r[3][:10]})")

    conn.close()
    print(f"\n({len(rows)} memories)")


def backfill_embeddings():
    """Generate embeddings for memories that were stored without them (daemon was unavailable)."""
    try:
        import embeddings as emb
    except ImportError:
        print("sentence-transformers not available")
        return

    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    rows = conn.execute(
        "SELECT id, type, topic, content, project FROM memories WHERE embedding IS NULL"
    ).fetchall()

    if not rows:
        print("All memories have embeddings.")
        conn.close()
        return

    print(f"Backfilling {len(rows)} memories...")
    for row in rows:
        mem_id, mem_type, topic, content, project = row
        project_prefix = f"{project} " if project else ""
        search_text = f"{project_prefix}{mem_type} {topic} {content}"
        vec = emb.embed(search_text)
        blob = emb.to_blob(vec)
        conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (blob, mem_id))
        emb.upsert_vec_index(conn, mem_id, blob)

    conn.commit()
    conn.close()
    print(f"Done. {len(rows)} embeddings generated.")


def format_rows(rows):
    if not rows:
        print("No results.")
        return
    for r in rows:
        if isinstance(r, dict):
            print(f"[{r['id']}] {r['type']}/{r['topic']} (sim={r.get('similarity', 'N/A'):.3f}, {r['updated_at']})")
            print(f"    {r['content']}")
        else:
            print(f"[{r['id']}] {r['type']}/{r['topic']} ({r['updated_at']})")
            print(f"    {r['content']}")


USAGE = """Usage: query.py <command> [args]

Commands:
  <search_term>          Full-text search
  --recent               List recent memories
  --type <type>          Filter by type (decision|preference|fact|correction|person|project|skill|workflow)
  --since <date>         Memories updated on or after date (ISO, today, yesterday, 3d, 2w, 1m)
  --until <date>         Memories updated on or before date
  --today                Shorthand for --since today
  --semantic <query>     Semantic similarity search
  --session <id>         List memories from a session
  --chain <id>           Show session chain (parent/child links)
  --project <name>       List all memories for a project (across all sessions in chain)
  --projects             List all known projects
  --label <id> <name>    Label a session's chain as a project
  --context <id>         Show conversation context around where a memory was recorded
  --history <id>         Show version history for a memory
  --update <id> <text>   Update a memory's content (preserves history)
  --delete <id>          Delete a memory and its history
  --compact [project]    Dense cairn dump for LLM ingestion
  --review               Surface low-confidence memories for inspection
  --verify-sources       Analyse accuracy of LLM source_messages estimates
  --backfill             Generate embeddings for memories missing them
  --check                Validate system health (DB, hooks, daemon, embeddings)
  --audit <session_id>   Dump unaudited memories from session for review
  --stats                Show database statistics"""


def check():
    """Validate Cairn system health — DB, hooks, daemon, embeddings."""
    import glob as _glob

    cairn_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(cairn_dir)
    claude_dir = os.path.expanduser("~/.claude")
    passed = 0
    failed = 0
    warnings = 0

    def ok(msg):
        nonlocal passed
        passed += 1
        print(f"  \033[32m✓\033[0m {msg}")

    def fail(msg):
        nonlocal failed
        failed += 1
        print(f"  \033[31m✗\033[0m {msg}")

    def warn(msg):
        nonlocal warnings
        warnings += 1
        print(f"  \033[33m!\033[0m {msg}")

    print("=== Cairn Health Check ===\n")

    # 1. Database
    print("Database:")
    if os.path.exists(DB_PATH):
        ok(f"Found at {DB_PATH}")
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA busy_timeout=5000")
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for t in ["memories", "sessions", "metrics", "hook_state", "memory_history"]:
                if t in tables:
                    ok(f"Table '{t}' exists")
                else:
                    fail(f"Table '{t}' missing")
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            ok(f"{count} memories stored")
            missing_emb = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL").fetchone()[0]
            if missing_emb == 0:
                ok("All memories have embeddings")
            else:
                warn(f"{missing_emb} memories without embeddings (run --backfill)")
            wal = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if wal == "wal":
                ok("WAL mode enabled")
            else:
                warn(f"Journal mode is '{wal}', expected 'wal'")
            conn.close()
        except Exception as e:
            fail(f"Database error: {e}")
    else:
        fail(f"Not found at {DB_PATH}")

    # 2. Hooks
    print("\nHooks:")
    settings_path = os.path.join(claude_dir, "settings.json")
    if os.path.exists(settings_path):
        try:
            import json as _json
            with open(settings_path, encoding="utf-8") as f:
                settings = _json.load(f)
            hooks = settings.get("hooks", {})
            stop_hooks = hooks.get("Stop", [])
            prompt_hooks = hooks.get("UserPromptSubmit", [])
            stop_found = any("stop_hook.py" in str(h) for h in stop_hooks)
            prompt_found = any("prompt_hook.py" in str(h) for h in prompt_hooks)
            if stop_found:
                ok("Stop hook registered")
            else:
                fail("Stop hook not found in settings.json")
            if prompt_found:
                ok("UserPromptSubmit hook registered")
            else:
                fail("UserPromptSubmit hook not found in settings.json")
        except Exception as e:
            fail(f"Settings parse error: {e}")
    else:
        fail("~/.claude/settings.json not found")

    # Check hook files exist
    for hook_file in ["stop_hook.py", "prompt_hook.py"]:
        path = os.path.join(project_dir, "hooks", hook_file)
        if os.path.exists(path):
            ok(f"hooks/{hook_file} exists")
        else:
            fail(f"hooks/{hook_file} missing")

    # 3. Rules
    print("\nRules:")
    rules_path = os.path.join(claude_dir, "rules", "memory-system.md")
    if os.path.exists(rules_path):
        ok("Global rules file deployed")
    else:
        fail("~/.claude/rules/memory-system.md missing (run install.sh)")

    # 4. Daemon
    print("\nDaemon:")
    pid_path = os.path.join(cairn_dir, ".daemon.pid")
    sock_path = os.path.join(cairn_dir, ".daemon.sock")
    if os.path.exists(pid_path):
        try:
            with open(pid_path, encoding="utf-8") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            ok(f"Running (PID {pid})")
            # Test responsiveness
            try:
                from daemon import send_request
                resp = send_request({"action": "ping"})
                if resp and resp.get("status") == "ok":
                    ok("Responding to ping")
                else:
                    warn("PID alive but not responding to ping")
            except Exception:
                warn("Could not ping daemon")
        except ProcessLookupError:
            warn("Stale PID file (daemon not running)")
        except ValueError:
            warn("Corrupt PID file")
    else:
        warn("Not running (will auto-start on first embed)")

    # 5. Embeddings
    print("\nEmbeddings:")
    venv_python = os.path.join(project_dir, ".venv", "bin", "python3")
    if os.path.exists(venv_python):
        ok("Virtual environment found")
    else:
        fail("Virtual environment missing (run install.sh)")

    try:
        import embeddings as emb
        vec = emb.embed("health check test", allow_slow=False)
        if vec is not None:
            ok(f"Embedding generated ({len(vec)} dimensions)")
        else:
            warn("Daemon unavailable — embeddings will use slow path")
            try:
                vec = emb.embed("health check test", allow_slow=True)
                if vec is not None:
                    ok(f"Slow-path embedding works ({len(vec)} dimensions)")
                else:
                    fail("Embedding failed on both paths")
            except Exception as e:
                fail(f"Embedding error: {e}")
    except ImportError:
        fail("embeddings module not importable")

    # 6. Slash command
    print("\nSlash command:")
    cmd_path = os.path.join(claude_dir, "commands", "cairn.md")
    if os.path.exists(cmd_path):
        ok("/cairn command deployed")
    else:
        warn("/cairn command missing (run install.sh)")

    # Summary
    print(f"\n{'='*30}")
    total = passed + failed + warnings
    print(f"  {passed}/{total} passed, {failed} failed, {warnings} warnings")
    if failed == 0:
        print("  System healthy.")
    else:
        print("  Issues found — run ./install.sh to fix.")
    return failed


def audit(session_id=None):
    """Dump unaudited memories for LLM review.

    If session_id is provided, audits that session's chain.
    Otherwise, finds the most recent session and audits it.
    Watermark is always per-session to prevent cross-session interference."""
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")

    if not session_id:
        # Find the most recent session with memories
        row = conn.execute("""
            SELECT session_id FROM memories
            GROUP BY session_id
            ORDER BY MAX(id) DESC LIMIT 1
        """).fetchone()
        if not row:
            print("No sessions with memories found.")
            conn.close()
            return
        session_id = row[0]
        print(f"Auditing most recent session: {session_id[:12]}...\n")

    # Per-session watermark
    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'last_audit_id'",
        (session_id,)
    ).fetchone()
    last_audit_id = int(row[0]) if row and row[0] else 0

    # Audit this session + its chain
    memories = conn.execute("""
        SELECT m.id, m.type, m.topic, m.content, m.confidence, m.created_at, m.session_id
        FROM memories m
        JOIN sessions s ON m.session_id = s.session_id
        WHERE (s.session_id LIKE ? OR s.parent_session_id LIKE ?) AND m.id > ?
        ORDER BY m.id ASC
    """, (f"{session_id}%", f"{session_id}%", last_audit_id)).fetchall()

    if not memories:
        print("No unaudited memories.")
        conn.close()
        return

    print(f"=== Audit: {len(memories)} unaudited memories ===")
    print(f"(since memory ID {last_audit_id})\n")

    for mem in memories:
        mem_id, mem_type, topic, content, confidence, created_at, sess = mem
        print(f"[{mem_id}] {mem_type}/{topic} (conf={confidence:.2f}, {created_at})")
        print(f"    {content}")
        print()

    # Record the highest ID as the new audit watermark
    max_id = memories[-1][0]
    conn.execute(
        "INSERT OR REPLACE INTO hook_state (session_id, key, value) VALUES (?, 'last_audit_id', ?)",
        (audit_key, str(max_id))
    )
    conn.commit()
    conn.close()

    print(f"--- Audit watermark set to ID {max_id} ---")
    print("Review each memory above. For each one:")
    print("  - If accurate: confirm")
    print("  - If inaccurate but correctable: --update <id> <corrected content>")
    print("  - If unsalvageable: --delete <id>")
    print("  - If stale/superseded: --delete <id>")
    print("  - If vague: --update <id> <richer content>")
    print("\nSummary: total reviewed, confirmed, updated, deleted.")


def label_project(session_id, project_name):
    """Label all sessions in a chain with a project name."""
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    # Find root by walking up
    current = session_id
    while True:
        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id LIKE ?",
            (f"{current}%",)
        ).fetchone()
        if not row or not row[0]:
            break
        current = row[0]
    root = current

    # Collect all session IDs in chain
    chain_ids = conn.execute("""
        WITH RECURSIVE chain(sid) AS (
            SELECT session_id FROM sessions WHERE session_id LIKE ?
            UNION ALL
            SELECT s.session_id FROM sessions s JOIN chain c ON s.parent_session_id = c.sid
        )
        SELECT sid FROM chain
    """, (f"{root}%",)).fetchall()

    for (sid,) in chain_ids:
        conn.execute("UPDATE sessions SET project = ? WHERE session_id = ?", (project_name, sid))
        conn.execute("UPDATE memories SET project = ? WHERE session_id = ?", (project_name, sid))

    conn.commit()
    # Count affected memories
    mem_count = conn.execute("SELECT COUNT(*) FROM memories WHERE project = ?", (project_name,)).fetchone()[0]
    conn.close()
    print(f"Labelled {len(chain_ids)} sessions, {mem_count} memories as project '{project_name}'")


def list_projects():
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    projects = conn.execute("""
        SELECT project, COUNT(*) as memories,
               MIN(created_at) as first_seen,
               MAX(updated_at) as last_updated
        FROM memories
        WHERE project IS NOT NULL
        GROUP BY project
        ORDER BY last_updated DESC
    """).fetchall()
    if not projects:
        print("No labelled projects.")
        return
    for p in projects:
        print(f"  {p[0]}: {p[1]} memories (since {p[2]}, last updated {p[3]})")
    # Also show global (unassigned) count
    global_count = conn.execute("SELECT COUNT(*) FROM memories WHERE project IS NULL").fetchone()[0]
    if global_count:
        print(f"  (global): {global_count} memories with no project")
    conn.close()


def memories_for_project(project_name, limit=50):
    conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, type, topic, content, updated_at
        FROM memories
        WHERE project = ?
        ORDER BY updated_at DESC
        LIMIT ?
    """, (project_name, limit)).fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "--today":
        format_rows(list_by_date(since="today"))
    elif cmd == "--since" and len(sys.argv) > 2:
        until = None
        if "--until" in sys.argv:
            idx = sys.argv.index("--until")
            until = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        format_rows(list_by_date(since=sys.argv[2], until=until))
    elif cmd == "--until" and len(sys.argv) > 2:
        since = None
        if "--since" in sys.argv:
            idx = sys.argv.index("--since")
            since = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        format_rows(list_by_date(since=since, until=sys.argv[2]))
    elif cmd == "--recent":
        format_rows(list_recent())
    elif cmd == "--type" and len(sys.argv) > 2:
        format_rows(list_by_type(sys.argv[2]))
    elif cmd == "--semantic" and len(sys.argv) > 2:
        results = semantic_search(" ".join(sys.argv[2:]))
        if results is not None:
            format_rows(results)
    elif cmd == "--session" and len(sys.argv) > 2:
        format_rows(list_by_session(sys.argv[2]))
    elif cmd == "--chain" and len(sys.argv) > 2:
        show_session_chain(sys.argv[2])
    elif cmd == "--project" and len(sys.argv) > 2:
        format_rows(memories_for_project(" ".join(sys.argv[2:])))
    elif cmd == "--projects":
        list_projects()
    elif cmd == "--label" and len(sys.argv) > 3:
        label_project(sys.argv[2], " ".join(sys.argv[3:]))
    elif cmd == "--context" and len(sys.argv) > 2:
        show_context(int(sys.argv[2]))
    elif cmd == "--history" and len(sys.argv) > 2:
        show_history(int(sys.argv[2]))
    elif cmd == "--update" and len(sys.argv) > 3:
        update_memory(int(sys.argv[2]), " ".join(sys.argv[3:]))
    elif cmd == "--delete" and len(sys.argv) > 2:
        delete_memory(int(sys.argv[2]))
    elif cmd == "--compact":
        compact(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else None)
    elif cmd == "--review":
        review()
    elif cmd == "--verify-sources":
        verify_sources()
    elif cmd == "--backfill":
        backfill_embeddings()
    elif cmd == "--check":
        sys.exit(check())
    elif cmd == "--audit":
        if len(sys.argv) > 2:
            audit(sys.argv[2])
        else:
            # Default to most recent session
            conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute("SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
            conn.close()
            if row:
                audit(row[0])
            else:
                print("No sessions found.")
    elif cmd == "--stats":
        stats()
    else:
        format_rows(search(" ".join(sys.argv[1:])))
