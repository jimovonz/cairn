#!/usr/bin/env python3
"""Tests for hooks/transcript_adapter.py.

Verifies the adapter correctly normalizes both Claude Code CLI JSONL transcripts
and GitHub Copilot extension JSONL transcripts into the CLI-shape entries that
the rest of cairn (hook_helpers, stop_hook, enforcement, storage) expects.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hooks.transcript_adapter import (
    detect_transcript_format,
    iter_normalized_entries,
    _normalize_tool_input,
)


def _write_jsonl(records: list[dict]) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# --- Format detection ---

def test_detects_copilot_format_from_session_start():
    path = _write_jsonl([
        {"type": "session.start",
         "data": {"sessionId": "s1", "producer": "copilot-agent",
                  "version": 1, "copilotVersion": "0.37.9"},
         "id": "h1", "timestamp": "2026-04-15T00:00:00Z", "parentId": None},
    ])
    try:
        assert detect_transcript_format(path) == "copilot"
    finally:
        os.unlink(path)


def test_detects_cli_format_from_permission_mode_header():
    path = _write_jsonl([
        {"type": "permission-mode", "permissionMode": "default", "sessionId": "s1"},
        {"type": "user", "message": {"role": "user", "content": "hi"},
         "uuid": "u1", "parentUuid": None, "sessionId": "s1",
         "timestamp": "2026-04-15T00:00:00Z"},
    ])
    try:
        assert detect_transcript_format(path) == "cli"
    finally:
        os.unlink(path)


def test_unknown_for_missing_or_unreadable_file():
    assert detect_transcript_format("") == "unknown"
    assert detect_transcript_format("/nonexistent/path.jsonl") == "unknown"


# --- CLI passthrough ---

def test_cli_records_pass_through_unchanged():
    cli_records = [
        {"type": "user", "message": {"role": "user", "content": "hello"},
         "uuid": "u1", "parentUuid": None, "sessionId": "s1",
         "timestamp": "2026-04-15T00:00:00Z"},
        {"type": "assistant",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "hi"},
             {"type": "tool_use", "id": "t1", "name": "Bash",
              "input": {"command": "ls"}},
         ]},
         "uuid": "u2", "parentUuid": "u1", "sessionId": "s1",
         "timestamp": "2026-04-15T00:00:01Z"},
    ]
    path = _write_jsonl(cli_records)
    try:
        out = list(iter_normalized_entries(path))
        assert out == cli_records
    finally:
        os.unlink(path)


# --- Copilot normalization ---

def test_copilot_user_message_becomes_cli_user():
    path = _write_jsonl([
        {"type": "session.start",
         "data": {"sessionId": "sess-42", "producer": "copilot-agent"},
         "id": "h1", "timestamp": "2026-04-15T00:00:00Z", "parentId": None},
        {"type": "user.message",
         "data": {"content": "what does this code do?", "attachments": []},
         "id": "u1", "parentId": "h1", "timestamp": "2026-04-15T00:00:01Z"},
    ])
    try:
        out = list(iter_normalized_entries(path))
        # session.start produces no output; user.message produces one entry
        assert len(out) == 1
        e = out[0]
        assert e["type"] == "user"
        assert e["message"] == {"role": "user", "content": "what does this code do?"}
        assert e["uuid"] == "u1"
        assert e["parentUuid"] == "h1"
        assert e["sessionId"] == "sess-42"
        assert e["timestamp"] == "2026-04-15T00:00:01Z"
    finally:
        os.unlink(path)


def test_copilot_assistant_message_with_tool_requests():
    path = _write_jsonl([
        {"type": "session.start",
         "data": {"sessionId": "sess-1", "producer": "copilot-agent"},
         "id": "h1", "timestamp": "2026-04-15T00:00:00Z", "parentId": None},
        {"type": "assistant.message",
         "data": {
             "messageId": "m1",
             "content": "Let me read the file.",
             "toolRequests": [
                 {"toolCallId": "tc1", "name": "read_file",
                  "arguments": '{"filePath": "/tmp/foo.py", "startLine": 1, "endLine": 50}',
                  "type": "function"},
             ],
             "reasoningText": "I need to look at this",
         },
         "id": "a1", "parentId": "h1", "timestamp": "2026-04-15T00:00:02Z"},
    ])
    try:
        out = list(iter_normalized_entries(path))
        assert len(out) == 1
        e = out[0]
        assert e["type"] == "assistant"
        assert e["message"]["role"] == "assistant"
        blocks = e["message"]["content"]
        assert blocks[0] == {"type": "text", "text": "Let me read the file."}
        assert blocks[1]["type"] == "tool_use"
        # Copilot read_file → cairn-recognizable Read
        assert blocks[1]["name"] == "Read"
        assert blocks[1]["id"] == "tc1"
        # Arguments parsed from JSON string + filePath aliased to file_path
        assert blocks[1]["input"]["file_path"] == "/tmp/foo.py"
        assert blocks[1]["input"]["startLine"] == 1
        assert e["sessionId"] == "sess-1"
    finally:
        os.unlink(path)


def test_copilot_assistant_with_empty_text_omits_text_block():
    path = _write_jsonl([
        {"type": "session.start",
         "data": {"sessionId": "s1", "producer": "copilot-agent"},
         "id": "h1", "timestamp": "2026-04-15T00:00:00Z", "parentId": None},
        {"type": "assistant.message",
         "data": {"messageId": "m1", "content": "",
                  "toolRequests": [
                      {"toolCallId": "tc1", "name": "run_in_terminal",
                       "arguments": '{"command": "ls -la"}', "type": "function"},
                  ]},
         "id": "a1", "parentId": "h1", "timestamp": "2026-04-15T00:00:02Z"},
    ])
    try:
        out = list(iter_normalized_entries(path))
        blocks = out[0]["message"]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "Bash"
        assert blocks[0]["input"]["command"] == "ls -la"
    finally:
        os.unlink(path)


def test_copilot_skips_non_message_records():
    path = _write_jsonl([
        {"type": "session.start",
         "data": {"sessionId": "s1", "producer": "copilot-agent"},
         "id": "h1", "timestamp": "2026-04-15T00:00:00Z", "parentId": None},
        {"type": "assistant.turn_start", "data": {"turnId": "0.0"},
         "id": "ts1", "parentId": "h1", "timestamp": "2026-04-15T00:00:01Z"},
        {"type": "user.message", "data": {"content": "hi", "attachments": []},
         "id": "u1", "parentId": "ts1", "timestamp": "2026-04-15T00:00:02Z"},
        {"type": "assistant.turn_end", "data": {"turnId": "0.0"},
         "id": "te1", "parentId": "u1", "timestamp": "2026-04-15T00:00:03Z"},
        {"type": "tool.execution_start",
         "data": {"toolCallId": "tc1", "toolName": "read_file", "arguments": {}},
         "id": "ts2", "parentId": "te1", "timestamp": "2026-04-15T00:00:04Z"},
    ])
    try:
        out = list(iter_normalized_entries(path))
        # Only the user.message should have come through
        assert len(out) == 1
        assert out[0]["type"] == "user"
    finally:
        os.unlink(path)


# --- Robustness ---

def test_malformed_lines_are_skipped():
    path = _write_jsonl([
        {"type": "session.start",
         "data": {"sessionId": "s1", "producer": "copilot-agent"},
         "id": "h1", "timestamp": "2026-04-15T00:00:00Z", "parentId": None},
    ])
    with open(path, "a") as f:
        f.write("not json at all\n")
        f.write('{"truncated": \n')
        f.write(json.dumps({
            "type": "user.message",
            "data": {"content": "hi", "attachments": []},
            "id": "u1", "parentId": "h1",
            "timestamp": "2026-04-15T00:00:01Z",
        }) + "\n")
    try:
        out = list(iter_normalized_entries(path))
        assert len(out) == 1
        assert out[0]["message"]["content"] == "hi"
    finally:
        os.unlink(path)


def test_empty_path_returns_nothing():
    assert list(iter_normalized_entries("")) == []
    assert list(iter_normalized_entries("/nonexistent/file.jsonl")) == []


def test_normalize_tool_input_handles_string_dict_and_garbage():
    assert _normalize_tool_input('{"file_path": "/a"}') == {"file_path": "/a"}
    assert _normalize_tool_input({"file_path": "/b"}) == {"file_path": "/b"}
    # filePath gets aliased
    assert _normalize_tool_input({"filePath": "/c"})["file_path"] == "/c"
    # Existing file_path is not overwritten
    assert _normalize_tool_input({"filePath": "/c", "file_path": "/d"})["file_path"] == "/d"
    # Unparseable JSON string preserved
    assert _normalize_tool_input("not json")["_raw"] == "not json"
    # Non-string/dict argument
    assert _normalize_tool_input(None) == {}


# --- Real-data smoke test against a Copilot transcript on disk if present ---

def test_real_copilot_file_if_available():
    """Best-effort smoke test against any real Copilot transcript on this machine.

    Skips silently if no transcripts directory exists — this is a sanity check
    for development environments, not a hard test requirement.
    """
    base = os.path.expanduser("~/.config/Code/User/workspaceStorage")
    if not os.path.isdir(base):
        return
    found = []
    for ws in os.listdir(base):
        d = os.path.join(base, ws, "GitHub.copilot-chat", "transcripts")
        if os.path.isdir(d):
            for name in os.listdir(d):
                if name.endswith(".jsonl"):
                    found.append(os.path.join(d, name))
    if not found:
        return
    for path in found:
        fmt = detect_transcript_format(path)
        assert fmt in ("copilot", "cli", "unknown")
        # Should not crash, should yield at least zero entries
        entries = list(iter_normalized_entries(path))
        for e in entries:
            assert e.get("type") in ("user", "assistant")
            assert "message" in e
            assert "sessionId" in e


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
