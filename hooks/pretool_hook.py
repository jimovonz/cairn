#!/usr/bin/env python3
"""
Claude Code PreToolUse Hook for Cairn — File Context Injection.

Fires before Read/Edit/Write/MultiEdit tool uses. Queries the cairn DB for
memories associated with the target file and injects them as context.

Two injection paths:
  1. Corrections (gotcha) — warnings injected as CAIRN GOTCHA, highest priority
  2. All other types (decisions, facts, etc.) — injected as CAIRN CONTEXT FOR FILE

This creates a closed loop:
  LLM touches file → memories written → file paths captured →
  next access to that file → relevant context injected automatically.
"""

from __future__ import annotations

import json
import os
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import sys
from typing import Any, Optional

from hooks.hook_helpers import log, get_conn, record_metric, flush_metrics

# Max entries to inject per file access (avoid flooding context)
MAX_GOTCHA_INJECTIONS = 3
MAX_CONTEXT_INJECTIONS = 5


def find_memories_for_file(
    file_path: str,
    corrections_only: bool = False,
) -> list[dict[str, Any]]:
    """Find memories associated with a given file path.

    Matches by:
    1. Exact path match in associated_files JSON array
    2. Basename match (for when the same file is referenced by different absolute paths)

    If corrections_only=True, returns only correction-type memories (gotcha path).
    Otherwise returns all non-correction types (context path).
    """
    if not file_path:
        return []

    conn = get_conn()
    basename = os.path.basename(file_path)

    type_filter = "type = 'correction'" if corrections_only else "type != 'correction'"

    try:
        rows = conn.execute(f"""
            SELECT id, type, topic, content, associated_files, confidence
            FROM memories
            WHERE {type_filter}
              AND associated_files IS NOT NULL
              AND archived_reason IS NULL
        """).fetchall()
    except sqlite3.Error as e:
        log(f"File context query error: {e}")
        conn.close()
        return []

    conn.close()

    matches: list[dict[str, Any]] = []
    for row in rows:
        mid, mem_type, topic, content, files_json, confidence = row
        try:
            files = json.loads(files_json)
        except (json.JSONDecodeError, TypeError):
            continue

        for f in files:
            if f == file_path or os.path.basename(f) == basename:
                matches.append({
                    "id": mid,
                    "type": mem_type,
                    "topic": topic,
                    "content": content,
                    "confidence": confidence or 0.7,
                })
                break

    return matches


def main() -> None:
    raw = sys.stdin.read()
    hook_input = json.loads(raw)

    tool_name = hook_input.get("tool_name", "")
    session_id = hook_input.get("session_id", "") or hook_input.get("sessionId", "")

    # Only fire for file-access tools
    if tool_name not in ("Read", "Edit", "Write", "MultiEdit"):
        sys.exit(0)

    # Extract file path from tool input
    tool_input = hook_input.get("tool_input") or hook_input.get("input") or {}
    file_path = tool_input.get("file_path") or tool_input.get("filePath") or ""

    if not file_path:
        sys.exit(0)

    basename = os.path.basename(file_path)
    sections: list[str] = []

    # Path 1: corrections (gotcha warnings) — highest priority
    corrections = find_memories_for_file(file_path, corrections_only=True)
    if corrections:
        warnings = [f"- [{c['topic']}] {c['content']}" for c in corrections[:MAX_GOTCHA_INJECTIONS]]
        ids = [str(c["id"]) for c in corrections[:MAX_GOTCHA_INJECTIONS]]
        sections.append(
            f"CAIRN GOTCHA for {basename}:\n" + "\n".join(warnings) + f"\nSources: {', '.join(ids)}"
        )
        log(f"Gotcha injection: {len(corrections)} corrections for {basename}")
        record_metric(session_id, "gotcha_injected", basename, len(corrections))

    # Path 2: all other memory types (decisions, facts, skills, etc.)
    context_memories = find_memories_for_file(file_path, corrections_only=False)
    if context_memories:
        # Sort by confidence descending, cap at MAX_CONTEXT_INJECTIONS
        context_memories.sort(key=lambda m: m["confidence"], reverse=True)
        top = context_memories[:MAX_CONTEXT_INJECTIONS]
        lines = [f"- [{m['type']}/{m['topic']}] {m['content']}" for m in top]
        ids = [str(m["id"]) for m in top]
        sections.append(
            f"CAIRN CONTEXT for {basename}:\n" + "\n".join(lines) + f"\nSources: {', '.join(ids)}"
        )
        log(f"File context injection: {len(top)} memories for {basename}")
        record_metric(session_id, "file_context_injected", basename, len(top))

    if not sections:
        sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": "\n\n".join(sections)
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
