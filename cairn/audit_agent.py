#!/usr/bin/env python3
"""Background audit agent — reviews unaudited memories against transcript context.

Usage:
  python3 audit_agent.py [session_id]

If no session_id, audits the most recent session with unaudited memories.

Extracts the relevant transcript segment, pairs it with unaudited memories,
and launches `claude -p` to review, enrich, archive, or fill gaps.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")
QUERY_PY = os.path.join(os.path.dirname(__file__), "query.py")


def get_unaudited(session_id: str) -> tuple[list[dict], int]:
    """Get unaudited memories for a session."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")

    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'last_audit_id'",
        (session_id,)
    ).fetchone()
    last_audit_id = int(row[0]) if row and row[0] else 0

    memories = conn.execute("""
        SELECT m.id, m.type, m.topic, m.content, m.confidence, m.created_at
        FROM memories m
        JOIN sessions s ON m.session_id = s.session_id
        WHERE (s.session_id LIKE ? OR s.parent_session_id LIKE ?) AND m.id > ?
        ORDER BY m.id ASC
    """, (f"{session_id}%", f"{session_id}%", last_audit_id)).fetchall()

    conn.close()
    return [
        {"id": m[0], "type": m[1], "topic": m[2], "content": m[3],
         "confidence": m[4], "created_at": m[5]}
        for m in memories
    ], last_audit_id


def get_transcript_segment(session_id: str, since_time: str) -> str:
    """Extract user/assistant text from transcript since a given time."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")

    row = conn.execute(
        "SELECT transcript_path FROM sessions WHERE session_id LIKE ?",
        (f"{session_id}%",)
    ).fetchone()
    conn.close()

    if not row or not row[0] or not os.path.exists(row[0]):
        return "(transcript not available)"

    lines = []
    try:
        with open(row[0], "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
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

                    if not content or not content.strip():
                        continue

                    # Strip memory blocks from assistant messages
                    if role == "assistant":
                        import re
                        content = re.sub(r"<memory>.*?</memory>", "", content, flags=re.DOTALL).strip()
                        if not content:
                            continue

                    # Truncate very long messages
                    if len(content) > 500:
                        content = content[:500] + "..."

                    lines.append(f"[{role}] {content}")
                except (json.JSONDecodeError, KeyError):
                    continue
    except (FileNotFoundError, PermissionError):
        return "(transcript not readable)"

    # Take the last portion — most relevant to recent memories
    if len(lines) > 100:
        lines = lines[-100:]

    return "\n".join(lines)


def find_session() -> str | None:
    """Find the most recent session with unaudited memories."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")

    sessions = conn.execute("""
        SELECT DISTINCT m.session_id, MAX(m.id) as max_id
        FROM memories m
        GROUP BY m.session_id
        ORDER BY max_id DESC
        LIMIT 10
    """).fetchall()

    for sess_id, _ in sessions:
        row = conn.execute(
            "SELECT value FROM hook_state WHERE session_id = ? AND key = 'last_audit_id'",
            (sess_id,)
        ).fetchone()
        last_id = int(row[0]) if row and row[0] else 0

        unaudited = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id = ? AND id > ?",
            (sess_id, last_id)
        ).fetchone()[0]

        if unaudited > 0:
            conn.close()
            return sess_id

    conn.close()
    return None


def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else find_session()

    if not session_id:
        print("No sessions with unaudited memories.")
        return

    memories, last_audit_id = get_unaudited(session_id)
    if not memories:
        print(f"No unaudited memories for session {session_id[:12]}.")
        return

    # Get the earliest memory's timestamp for transcript filtering
    earliest = memories[0]["created_at"]
    transcript = get_transcript_segment(session_id, earliest)

    # Build the prompt
    memory_text = "\n".join(
        f"[{m['id']}] {m['type']}/{m['topic']} (conf={m['confidence']:.2f})\n    {m['content']}"
        for m in memories
    )

    cairn_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    query_cmd = f"python3 {cairn_home}/cairn/query.py"

    # Look up project for session
    conn2 = sqlite3.connect(DB_PATH)
    proj_row = conn2.execute(
        "SELECT project FROM sessions WHERE session_id LIKE ?", (f"{session_id}%",)
    ).fetchone()
    project_name = proj_row[0] if proj_row and proj_row[0] else None
    conn2.close()
    project_flag = f" --project {project_name}" if project_name else ""

    prompt = f"""You are auditing Cairn memories for accuracy and completeness.

## Memories to review ({len(memories)} entries)

{memory_text}

## Conversation context (recent transcript)

{transcript}

## Instructions

### Step 1: Review existing memories
For each memory, cross-check against the transcript:
- **Accurate but thin**: Run `{query_cmd} --update <id> <richer content>` — add the why, alternatives considered, reasoning, or outcome
- **Accurate and complete**: Note as confirmed
- **Inaccurate**: Run `{query_cmd} --update <id> <corrected content>`
- **Superseded/wrong**: Run `{query_cmd} --archive <id> <reason>` — preserves the learning trail

### Step 2: Find gaps (CRITICAL)
Read through the ENTIRE transcript and identify:
- Decisions made that have no corresponding memory
- User corrections or redirections not captured
- Rejected approaches or failed attempts not recorded
- Facts discovered (schemas, paths, configs) not stored
- User preferences expressed but not captured

For each gap, add it:
`{query_cmd} --add <type> <topic> <content> --session {session_id}{project_flag}`

### Step 3: Advance watermark
Run: `{query_cmd} --audit {session_id}`

### Step 4: Summary
Report: reviewed, confirmed, enriched, archived, gaps filled (with details of what was added)."""

    print(f"Auditing {len(memories)} memories for session {session_id[:12]}...")
    print(f"Transcript: {len(transcript)} chars")
    print()

    # Launch claude -p
    result = subprocess.run(
        ["claude", "-p", prompt, "--allowedTools", "Bash"],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"Agent error: {result.stderr[:200]}", file=sys.stderr)


if __name__ == "__main__":
    main()
