"""Transcript format adapter.

Cairn was written against the Claude Code CLI JSONL transcript shape. The
GitHub Copilot VS Code extension (chatHooks@6, github.copilot-chat ≥0.37.9)
writes a structurally different JSONL — typed event records with a separate
session.start header and tool requests embedded inside assistant messages.

Both surfaces deliver a `transcript_path` field in the Stop hook payload, and
both files are valid JSONL, but the per-record shape is incompatible. This
module normalizes Copilot transcripts into the CLI shape that the rest of
cairn already understands, so the readers in hook_helpers.py, stop_hook.py,
enforcement.py, and storage.py do not each need a parallel implementation.

CLI shape (what the rest of cairn expects):
    {
      "type": "user" | "assistant" | ...,
      "message": {"role": ..., "content": str | [block, ...]},
      "uuid": ..., "parentUuid": ..., "sessionId": ...,
      "timestamp": ...,
    }
    where blocks are {"type":"text","text":...} or
    {"type":"tool_use","id":...,"name":...,"input":{...}}.

Copilot shape (what we adapt from):
    {"type": "session.start", "data": {"producer":"copilot-agent",
                                       "sessionId":...}, "id":..., "parentId":...}
    {"type": "user.message", "data":{"content":...,"attachments":[]}, ...}
    {"type": "assistant.message",
     "data": {"messageId":..., "content": str,
              "toolRequests":[{"toolCallId":...,"name":...,"arguments":str}],
              "reasoningText": str (optional)}, ...}

Schema detection happens once on the first meaningful line — Copilot writes a
`session.start` record with `data.producer` starting with "copilot". CLI
transcripts have no equivalent header.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional


# Copilot uses VS Code / Language Model tool names. Translate to the
# Claude-Code-equivalent names that enforcement.py and storage.py look for so
# their existing logic (file-path extraction, action-tool detection) keeps
# working without per-tool branching at every call site.
_COPILOT_TO_CLI_TOOL_NAMES = {
    "read_file": "Read",
    "create_file": "Write",
    "create_new_file": "Write",
    "replace_string_in_file": "Edit",
    "multi_replace_string_in_file": "MultiEdit",
    "insert_edit_into_file": "Edit",
    "apply_patch": "Edit",
    "run_in_terminal": "Bash",
    "get_terminal_output": "Bash",
    "grep_search": "Grep",
    "file_search": "Glob",
    "semantic_search": "Grep",
}


def _normalize_tool_input(arguments: Any) -> dict[str, Any]:
    """Copilot stores tool arguments as a JSON string; CLI as a dict.

    Also unifies the camelCase `filePath` Copilot uses with the snake_case
    `file_path` cairn's storage.extract_associated_files looks for first.
    """
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": arguments}
    elif isinstance(arguments, dict):
        parsed = dict(arguments)
    else:
        return {}

    if not isinstance(parsed, dict):
        return {"_value": parsed}

    if "filePath" in parsed and "file_path" not in parsed:
        parsed["file_path"] = parsed["filePath"]
    return parsed


def _detect_format(first_record: dict[str, Any]) -> str:
    """Return 'copilot' if the first parsed record is a Copilot session header.

    Anything else — including CLI transcripts that begin with permission-mode
    or file-history-snapshot records — is treated as CLI shape and passed
    through unchanged.
    """
    if first_record.get("type") == "session.start":
        producer = (first_record.get("data") or {}).get("producer", "")
        if isinstance(producer, str) and producer.startswith("copilot"):
            return "copilot"
    return "cli"


def _normalize_copilot_record(
    record: dict[str, Any],
    session_id: str,
) -> Optional[dict[str, Any]]:
    """Convert one Copilot record to the CLI shape, or return None to skip.

    session_id is threaded from the session.start header because individual
    Copilot records don't carry it the way CLI records do.
    """
    rec_type = record.get("type", "")
    data = record.get("data") or {}
    common = {
        "uuid": record.get("id", ""),
        "parentUuid": record.get("parentId"),
        "sessionId": session_id,
        "timestamp": record.get("timestamp", ""),
    }

    if rec_type == "user.message":
        content = data.get("content", "")
        return {
            **common,
            "type": "user",
            "message": {"role": "user", "content": content if isinstance(content, str) else ""},
        }

    if rec_type == "assistant.message":
        text = data.get("content", "") or ""
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        for tr in data.get("toolRequests", []) or []:
            if not isinstance(tr, dict):
                continue
            raw_name = tr.get("name", "")
            blocks.append({
                "type": "tool_use",
                "id": tr.get("toolCallId", ""),
                "name": _COPILOT_TO_CLI_TOOL_NAMES.get(raw_name, raw_name),
                "input": _normalize_tool_input(tr.get("arguments")),
            })
        return {
            **common,
            "type": "assistant",
            "message": {"role": "assistant", "content": blocks},
        }

    # session.start, assistant.turn_start/end, tool.execution_start/complete:
    # no CLI equivalent, no caller depends on them.
    return None


def iter_normalized_entries(transcript_path: str) -> Iterator[dict[str, Any]]:
    """Yield transcript entries in CLI shape regardless of source format.

    For CLI transcripts: yield each parsed line as-is (zero-cost passthrough).
    For Copilot transcripts: translate each record. Format detection happens
    on the first non-empty parseable line; subsequent lines stay on the same
    branch.

    Malformed lines are skipped silently — matching the behaviour of every
    existing reader in cairn that calls json.loads inside try/except.
    """
    if not transcript_path:
        return
    try:
        f = open(transcript_path, "r", encoding="utf-8")
    except (OSError, PermissionError):
        return

    try:
        fmt: Optional[str] = None
        copilot_session_id = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue

            if fmt is None:
                fmt = _detect_format(record)
                if fmt == "copilot":
                    copilot_session_id = (record.get("data") or {}).get("sessionId", "")

            if fmt == "cli":
                yield record
            else:
                normalized = _normalize_copilot_record(record, copilot_session_id)
                if normalized is not None:
                    yield normalized
    finally:
        f.close()


def detect_transcript_format(transcript_path: str) -> str:
    """Return 'cli', 'copilot', or 'unknown' for a transcript file.

    Used by register_session to record provenance in the sessions table and
    by tests. Reads at most one parseable record from the file.
    """
    if not transcript_path:
        return "unknown"
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(record, dict):
                    continue
                return _detect_format(record)
    except (OSError, PermissionError):
        return "unknown"
    return "unknown"
