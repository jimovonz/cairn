"""_response_has_tool_calls must see tool calls made earlier in the turn.
Tool results arrive as role=user entries; treating them as the turn boundary
hid every tool call before the last result, so check_declined_without_trying
blocked turns that had in fact tried."""
import json

from hooks.enforcement import _response_has_tool_calls


def _write(tmp_path, records):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(p)


def _user(content):
    return {"type": "user", "message": {"role": "user", "content": content}}


def _assistant(blocks):
    return {"type": "assistant", "message": {"role": "assistant", "content": blocks}}


def test_tool_call_before_last_tool_result_is_seen(tmp_path):
    path = _write(tmp_path, [
        _user("restart the service"),
        _assistant([{"type": "tool_use", "id": "t1", "name": "Bash",
                     "input": {"command": "systemctl --user restart x"}}]),
        _user([{"type": "tool_result", "tool_use_id": "t1", "content": "denied"}]),
        _assistant([{"type": "text", "text": "I can't restart it from here."}]),
    ])
    assert _response_has_tool_calls(path)


def test_tool_call_in_previous_turn_is_not_counted(tmp_path):
    path = _write(tmp_path, [
        _user("first"),
        _assistant([{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]),
        _user([{"type": "tool_result", "tool_use_id": "t1", "content": "a"}]),
        _assistant([{"type": "text", "text": "done"}]),
        _user("now restart it"),
        _assistant([{"type": "text", "text": "You'll need to restart it yourself."}]),
    ])
    assert not _response_has_tool_calls(path)
