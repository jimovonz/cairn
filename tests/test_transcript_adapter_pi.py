"""pi session files normalise to the CLI shape so transcript-reading checks
(declined-without-trying, asserted-absence) work on pi turns."""
import json

from hooks.transcript_adapter import detect_transcript_format, iter_normalized_entries
from hooks.hook_helpers import cairn_query_invoked_this_turn


def _write(tmp_path, records):
    p = tmp_path / "pi.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(p)


HEADER = {"type": "session", "version": 3, "id": "x", "cwd": "/tmp"}


def _msg(role, content, **extra):
    return {"type": "message", "message": dict(role=role, content=content, **extra)}


def test_pi_format_detected_and_normalised(tmp_path):
    path = _write(tmp_path, [
        HEADER,
        {"type": "custom_message", "customType": "cairn-context", "content": "ctx"},
        _msg("user", [{"type": "text", "text": "do it"}]),
        _msg("assistant", [{"type": "text", "text": "ok"},
                           {"type": "toolCall", "id": "c1", "name": "bash",
                            "arguments": {"command": "ls"}}]),
        _msg("toolResult", [{"type": "text", "text": "a b"}], toolCallId="c1", isError=False),
    ])
    assert detect_transcript_format(path) == "pi"
    entries = list(iter_normalized_entries(path))
    assert [e["type"] for e in entries] == ["user", "assistant", "user"]
    tool = entries[1]["message"]["content"][1]
    assert tool == {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}
    assert entries[2]["message"]["content"][0]["type"] == "tool_result"


def test_pi_cairn_query_tool_counts_as_cairn_search(tmp_path):
    path = _write(tmp_path, [
        HEADER,
        _msg("user", [{"type": "text", "text": "is there a key?"}]),
        _msg("assistant", [{"type": "toolCall", "id": "c1", "name": "cairn_query",
                            "arguments": {"query": "api key"}}]),
        _msg("toolResult", [{"type": "text", "text": "found"}], toolCallId="c1"),
        _msg("assistant", [{"type": "text", "text": "No key exists."}]),
    ])
    assert cairn_query_invoked_this_turn(path)
