#!/usr/bin/env python3
"""
Claude Code PostToolUse Hook for Cairn — Mid-Response Memory Checkpoints.

Fires after tool calls complete. Detects high-signal results (errors, large
output, edit decisions) and nudges the LLM to emit a <memory_note> tag
capturing the observation before continuing.

No extra LLM calls — the agent is already generating its next response.
The <memory_note> is a lightweight inline tag collected by the stop hook.

Gating:
  - Only fires for whitelisted tools (Bash, Edit, Write by default)
  - Cooldown: skips if a nudge fired within the last N tool calls
  - Subagents: disabled (they have opportunistic memory only)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from hooks.hook_helpers import log, record_metric, flush_metrics, load_hook_state, save_hook_state
from cairn.config import (
    CHECKPOINT_ENABLED,
    CHECKPOINT_COOLDOWN,
    CHECKPOINT_TOOLS,
    CHECKPOINT_ERROR_PATTERNS,
    CHECKPOINT_MIN_OUTPUT_LINES,
)

TOOL_WHITELIST: set[str] = {t.strip() for t in CHECKPOINT_TOOLS.split(",") if t.strip()}
ERROR_PATTERNS: list[str] = [p.strip().lower() for p in CHECKPOINT_ERROR_PATTERNS.split(",") if p.strip()]

NUDGE_TEXT = (
    "MEMORY CHECKPOINT: The previous tool call produced a notable result. "
    "Before continuing, emit a brief <memory_note> tag capturing what you just "
    "learned or observed. Format: <memory_note>type/topic: one-line observation"
    "</memory_note> — then continue your work. Types: fact, correction, decision, "
    "skill. Skip if trivial."
)


def _get_tool_count(session_id: str) -> int:
    """Get the running tool call count for this session."""
    raw = load_hook_state(session_id, "checkpoint_tool_count")
    return int(raw) if raw else 0


def _increment_tool_count(session_id: str) -> int:
    """Increment and return the tool call count."""
    count = _get_tool_count(session_id) + 1
    save_hook_state(session_id, "checkpoint_tool_count", str(count))
    return count


def _get_last_nudge_count(session_id: str) -> int:
    """Get the tool count at which the last nudge fired."""
    raw = load_hook_state(session_id, "checkpoint_last_nudge")
    return int(raw) if raw else 0


def _set_last_nudge_count(session_id: str, count: int) -> None:
    save_hook_state(session_id, "checkpoint_last_nudge", str(count))


def _is_high_signal_bash(tool_input: dict, tool_output: dict) -> tuple[bool, str]:
    """Check if a Bash tool result is high-signal (worth a checkpoint).

    Returns (is_high_signal, reason).
    """
    exit_code = tool_output.get("exitCode") or tool_output.get("exit_code")
    stdout = tool_output.get("stdout", "") or tool_output.get("output", "") or ""
    stderr = tool_output.get("stderr", "") or ""
    combined = f"{stdout}\n{stderr}".lower()

    # Non-zero exit code
    if exit_code and exit_code != 0:
        return True, f"non-zero exit ({exit_code})"

    # Error patterns in output
    for pattern in ERROR_PATTERNS:
        if pattern in combined:
            return True, f"error pattern: {pattern}"

    # Large output (complex results worth noting)
    line_count = stdout.count("\n") + 1
    if line_count >= CHECKPOINT_MIN_OUTPUT_LINES:
        return True, f"large output ({line_count} lines)"

    return False, ""


def _is_high_signal_edit(tool_input: dict, tool_output: dict) -> tuple[bool, str]:
    """Edit/Write calls are always high-signal — a decision was made."""
    file_path = tool_input.get("file_path") or tool_input.get("filePath") or "unknown"
    basename = os.path.basename(file_path)
    return True, f"file modified: {basename}"


def main() -> None:
    if not CHECKPOINT_ENABLED:
        sys.exit(0)

    raw = sys.stdin.read()
    hook_input = json.loads(raw)

    tool_name: str = hook_input.get("tool_name", "")
    session_id: str = hook_input.get("session_id", "") or hook_input.get("sessionId", "")
    is_subagent: bool = bool(hook_input.get("agent_id"))

    # Skip subagents — they have opportunistic memory only
    if is_subagent:
        sys.exit(0)

    # Only process whitelisted tools
    if tool_name not in TOOL_WHITELIST:
        sys.exit(0)

    tool_input: dict = hook_input.get("tool_input") or hook_input.get("input") or {}
    tool_output: dict = hook_input.get("tool_output") or hook_input.get("output") or {}

    # If tool_output is a string (some tools return plain text), wrap it
    if isinstance(tool_output, str):
        tool_output = {"output": tool_output}

    # Increment tool count
    current_count = _increment_tool_count(session_id)

    # Check cooldown — skip if a nudge has fired recently, but always allow the first
    last_nudge = _get_last_nudge_count(session_id)
    if last_nudge > 0 and current_count - last_nudge < CHECKPOINT_COOLDOWN:
        sys.exit(0)

    # Detect high-signal result
    high_signal = False
    reason = ""

    if tool_name == "Bash":
        high_signal, reason = _is_high_signal_bash(tool_input, tool_output)
    elif tool_name in ("Edit", "Write"):
        high_signal, reason = _is_high_signal_edit(tool_input, tool_output)

    if not high_signal:
        sys.exit(0)

    # Fire the nudge
    _set_last_nudge_count(session_id, current_count)
    log(f"Checkpoint nudge: {tool_name} — {reason} (tool #{current_count})")
    record_metric(session_id, "checkpoint_nudge", f"{tool_name}: {reason}", current_count)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": NUDGE_TEXT,
        }
    }
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            log(f"POSTTOOL HOOK CRASH: {e}")
        except Exception:
            pass
        sys.exit(0)
