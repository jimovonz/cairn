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
import re
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import sys
from typing import Any, Optional

from hooks.hook_helpers import log, get_conn, record_metric, flush_metrics, load_hook_state, save_hook_state

# Max entries to inject per file access (avoid flooding context)
MAX_GOTCHA_INJECTIONS = 3
MAX_CONTEXT_INJECTIONS = 5


def _looks_like_code_search(command: str) -> Optional[str]:
    """Return a candidate symbol name if the Bash command looks like a code-symbol grep."""
    if "grep" not in command:
        return None
    # grep with a quoted identifier-like pattern
    m = re.search(r'\bgrep\b[^"\n]*["\']([A-Za-z_][A-Za-z0-9_]{2,})["\']', command)
    if m:
        return m.group(1)
    # grep with an unquoted identifier before a space/end (skip flag-only invocations)
    m = re.search(r'\bgrep\b(?:\s+-\S+)*\s+([A-Za-z_][A-Za-z0-9_]{2,})(?:\s|$)', command)
    if m:
        return m.group(1)
    return None


def find_memories_for_file(
    file_path: str,
    corrections_only: bool = False,
    current_session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Find memories associated with a given file path.

    Matches by:
    1. Exact path match in associated_files JSON array
    2. Basename match (for when the same file is referenced by different absolute paths)

    If corrections_only=True, returns only correction-type memories (gotcha path).
    Otherwise returns all non-correction types (context path).

    Memories written by current_session_id are excluded — they're already in
    the live conversation context, so re-injecting them is pure token noise.
    """
    if not file_path:
        return []

    conn = get_conn()
    basename = os.path.basename(file_path)

    type_filter = "type = 'correction'" if corrections_only else "type != 'correction'"

    try:
        rows = conn.execute(f"""
            SELECT id, type, topic, content, associated_files, confidence, session_id
            FROM memories
            WHERE {type_filter}
              AND associated_files IS NOT NULL
              AND archived_reason IS NULL
              AND deleted_at IS NULL
        """).fetchall()
    except sqlite3.Error as e:
        log(f"File context query error: {e}")
        conn.close()
        return []

    conn.close()

    matches: list[dict[str, Any]] = []
    for row in rows:
        mid, mem_type, topic, content, files_json, confidence, mem_session = row
        if current_session_id and mem_session == current_session_id:
            continue
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


# --- Bash file-access recovery -------------------------------------------------
# In environments where Read/Edit/Write are blocked and routed through Bash
# helpers (e.g. claude-context-hooks: cat/sed and cch-edit.py/cch-write.py), the
# native Read/Edit PreToolUse event never fires — so file-context injection would
# silently never trigger. Both hooks fire on the SAME PreToolUse:Bash event, so we
# recover the target file(s) from the command string and inject for them too.

# File-reading verbs whose operand is a path we should inject context for.
_BASH_FILE_VERBS = {"cat", "head", "tail", "sed", "less", "more", "bat", "nl", "view"}
# Helper editors are invoked as scripts; match by suffix to catch python3 .../cch-edit.py.
_BASH_EDITOR_SUFFIXES = ("cch-edit.py", "cch-write.py", "cch-edit", "cch-write")
# Only inject for source/text files — keeps DB/log/binary args from triggering queries.
_SOURCE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".java", ".rb", ".sh", ".sql", ".md", ".toml", ".yaml", ".yml",
}


def extract_bash_file_paths(command: str, max_files: int = 3) -> list[str]:
    """Recover source-file paths a Bash command is about to read or edit.

    Conservative by design: engages only when the command invokes a known
    file-reading/editing verb, and returns only existing files with a source
    extension. Returns realpaths, deduped, capped at max_files.
    """
    if not command:
        return []
    # cch-batch.py with heredoc: extract paths from each embedded command line
    if "cch-batch.py" in command and "\n" in command:
        batch_out: list[str] = []
        for line in command.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or "<<" in line or "cch-batch" in line or line == "EOF":
                continue
            for p in extract_bash_file_paths(line, max_files=max_files - len(batch_out)):
                if p not in batch_out:
                    batch_out.append(p)
                if len(batch_out) >= max_files:
                    return batch_out
        return batch_out
    import shlex
    try:
        toks = shlex.split(command, posix=True)
    except ValueError:
        toks = command.split()
    if not toks:
        return []
    verbs = {os.path.basename(t) for t in toks}
    is_editor = any(t.endswith(_BASH_EDITOR_SUFFIXES) for t in toks)
    if not (verbs & _BASH_FILE_VERBS or is_editor):
        return []
    out: list[str] = []
    for t in toks:
        if t.startswith("-"):
            continue
        if os.path.splitext(t)[1].lower() not in _SOURCE_EXTS:
            continue
        if not os.path.isfile(t):
            continue
        rp = os.path.realpath(t)
        if rp not in out:
            out.append(rp)
        if len(out) >= max_files:
            break
    return out


def sections_for_file(file_path: str, session_id: str, seen: set, graph_cfg) -> list[str]:
    """Build the gotcha / context / graph injection sections for one file.

    seen is the shared graph_files_seen set (mutated in place when a graph block
    is served). graph_cfg is (enabled, max_symbols, risk_threshold) or None.
    """
    basename = os.path.basename(file_path)
    sections: list[str] = []

    # Path 1: corrections (gotcha warnings) — highest priority
    corrections = find_memories_for_file(file_path, corrections_only=True, current_session_id=session_id)
    if corrections:
        warnings = [f"- [{c['topic']}] {c['content']}" for c in corrections[:MAX_GOTCHA_INJECTIONS]]
        ids = [str(c["id"]) for c in corrections[:MAX_GOTCHA_INJECTIONS]]
        sections.append(
            f"CAIRN GOTCHA for {basename}:\n" + "\n".join(warnings) + f"\nSources: {', '.join(ids)}"
        )
        log(f"Gotcha injection: {len(corrections)} corrections for {basename}")
        record_metric(session_id, "gotcha_injected", basename, len(corrections))

    # Path 2: all other memory types (decisions, facts, skills, etc.)
    context_memories = find_memories_for_file(file_path, corrections_only=False, current_session_id=session_id)
    if context_memories:
        context_memories.sort(key=lambda m: m["confidence"], reverse=True)
        top = context_memories[:MAX_CONTEXT_INJECTIONS]
        lines = [f"- [{m['type']}/{m['topic']}] {m['content']}" for m in top]
        ids = [str(m["id"]) for m in top]
        sections.append(
            f"CAIRN CONTEXT for {basename}:\n" + "\n".join(lines) + f"\nSources: {', '.join(ids)}"
        )
        log(f"File context injection: {len(top)} memories for {basename}")
        record_metric(session_id, "file_context_injected", basename, len(top))

    # Path 3: code-graph structural context — deterministic, no LLM. Once-per-file
    # via the shared seen cache. Fails open.
    if graph_cfg is not None:
        try:
            enabled, max_symbols, risk_threshold = graph_cfg
            key = os.path.realpath(file_path)
            if enabled and key not in seen:
                from cairn.graph import file_context_block
                block = file_context_block(file_path, max_symbols=max_symbols, risk_threshold=risk_threshold)
                if block:
                    sections.append(
                        f"CAIRN GRAPH for {basename} (code-review-graph — structure, no need to re-read):\n{block}"
                    )
                    seen.add(key)
                    record_metric(session_id, "graph_file_context_injected", basename)
        except Exception as _e:
            log(f"graph file-context failed open: {type(_e).__name__}: {_e}")

    return sections


def main() -> None:
    if os.environ.get("CAIRN_ENABLED", "1") == "0":
        sys.exit(0)
    raw = sys.stdin.read()
    hook_input = json.loads(raw)

    tool_name = hook_input.get("tool_name", "")
    session_id = hook_input.get("session_id", "") or hook_input.get("sessionId", "")
    cwd = hook_input.get("cwd", os.getcwd())

    tool_input = hook_input.get("tool_input") or hook_input.get("input") or {}

    # Determine the file(s) this tool call touches.
    file_paths: list[str] = []
    if tool_name in ("Read", "Edit", "Write", "MultiEdit"):
        fp = tool_input.get("file_path") or tool_input.get("filePath") or ""
        if fp:
            file_paths = [fp]
    elif tool_name == "Bash":
        # Read/Edit are routed through Bash helpers in some environments; recover paths.
        file_paths = extract_bash_file_paths(tool_input.get("command") or "")

    # Resolve graph config once (shared across files).
    graph_cfg = None
    try:
        from cairn.config import (
            GRAPH_FILE_CONTEXT_ENABLED,
            GRAPH_FILE_CONTEXT_MAX_SYMBOLS,
            GRAPH_RISK_TAIL_THRESHOLD,
        )
        graph_cfg = (GRAPH_FILE_CONTEXT_ENABLED, GRAPH_FILE_CONTEXT_MAX_SYMBOLS, GRAPH_RISK_TAIL_THRESHOLD)
    except Exception:
        graph_cfg = None

    # Shared once-per-file-per-session graph cache.
    seen_raw = load_hook_state(session_id, "graph_files_seen") or ""
    seen = set(seen_raw.split("\n")) if seen_raw else set()
    seen_before = len(seen)

    sections: list[str] = []
    for fp in file_paths:
        sections.extend(sections_for_file(fp, session_id, seen, graph_cfg))

    if len(seen) != seen_before:
        save_hook_state(session_id, "graph_files_seen", "\n".join(seen))

    # Grep hint: when the model searches for a code symbol, remind it about cairn-graph
    # (once per session, only when a graph is present for this repo).
    if tool_name == "Bash" and not load_hook_state(session_id, "graph_grep_hint_shown"):
        command = tool_input.get("command") or ""
        symbol = _looks_like_code_search(command)
        if symbol:
            gdb = os.path.join(cwd, ".code-review-graph", "graph.db")
            if os.path.exists(gdb):
                save_hook_state(session_id, "graph_grep_hint_shown", "1")
                sections.append(
                    f"CODE GRAPH AVAILABLE — prefer cairn-graph over grep for structural queries:\n"
                    f"  cairn-graph --location {symbol}      # where it is defined\n"
                    f"  cairn-graph --callers {symbol}       # who calls it\n"
                    f"  cairn-graph --callees {symbol}       # what it calls\n"
                    f"  cairn-graph --impact {symbol}        # one-line blast radius\n"
                    f"  cairn-graph --context-pack {symbol}  # body + callers + tests"
                )
                log(f"graph grep hint injected for symbol: {symbol}")
                record_metric(session_id, "graph_grep_hint_injected", symbol)

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
    finally:
        # main() exits via sys.exit() on every path; flush buffered metrics
        # (e.g. graph_file_context_injected) explicitly here rather than relying
        # solely on the atexit backstop. SystemExit propagates through finally;
        # a second flush is a no-op once the buffer is drained.
        try:
            flush_metrics()
        except Exception:
            pass
