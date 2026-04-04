#!/usr/bin/env python3
"""
Claude Code PreToolUse Hook for Cairn — Gotcha Injection.

Fires before Read/Edit/Write/MultiEdit tool uses. Queries the cairn DB for
correction-type memories associated with the target file and injects them
as warnings via additionalContext.

This creates a closed loop:
  LLM makes mistake → correction stored → file paths captured →
  next access to those files → warning injected → mistake prevented.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.dirname(__file__))

from hook_helpers import log, get_conn, record_metric

# Max corrections to inject per file access (avoid flooding context)
MAX_GOTCHA_INJECTIONS = 3


def find_corrections_for_file(file_path: str, session_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Find correction memories associated with a given file path.

    Matches by:
    1. Exact path match in associated_files JSON array
    2. Basename match (for when the same file is referenced by different absolute paths)
    """
    if not file_path:
        return []

    conn = get_conn()
    basename = os.path.basename(file_path)

    try:
        rows = conn.execute("""
            SELECT id, topic, content, associated_files, confidence, archived_reason
            FROM memories
            WHERE type = 'correction'
              AND associated_files IS NOT NULL
              AND archived_reason IS NULL
        """).fetchall()
    except sqlite3.Error as e:
        log(f"Gotcha query error: {e}")
        conn.close()
        return []

    conn.close()

    matches: list[dict[str, Any]] = []
    for row in rows:
        mid, topic, content, files_json, confidence, archived = row
        try:
            files = json.loads(files_json)
        except (json.JSONDecodeError, TypeError):
            continue

        # Check for exact path match or basename match
        matched = False
        for f in files:
            if f == file_path or os.path.basename(f) == basename:
                matched = True
                break

        if matched:
            matches.append({
                "id": mid,
                "topic": topic,
                "content": content,
                "confidence": confidence or 0.7,
            })

    return matches[:MAX_GOTCHA_INJECTIONS]


def main() -> None:
    raw = sys.stdin.read()
    hook_input = json.loads(raw)

    tool_name = hook_input.get("tool_name", "")
    session_id = hook_input.get("session_id", "")

    # Only fire for file-access tools
    if tool_name not in ("Read", "Edit", "Write", "MultiEdit"):
        sys.exit(0)

    # Extract file path from tool input
    tool_input = hook_input.get("tool_input") or hook_input.get("input") or {}
    file_path = tool_input.get("file_path") or tool_input.get("filePath") or ""

    if not file_path:
        sys.exit(0)

    # Query for corrections
    corrections = find_corrections_for_file(file_path, session_id)

    if not corrections:
        sys.exit(0)

    # Format warnings
    warnings: list[str] = []
    ids: list[str] = []
    for c in corrections:
        warnings.append(f"- [{c['topic']}] {c['content']}")
        ids.append(str(c["id"]))

    basename = os.path.basename(file_path)
    context_text = (
        f"CAIRN GOTCHA for {basename}:\n"
        + "\n".join(warnings)
        + f"\nSources: {', '.join(ids)}"
    )

    log(f"Gotcha injection: {len(corrections)} corrections for {basename}")
    record_metric(session_id, "gotcha_injected", basename, len(corrections))

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": context_text
        }
    }
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            log(f"PRETOOL HOOK CRASH: {e}")
        except Exception:
            pass
        sys.exit(0)
