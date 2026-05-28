"""Tests for cairn-session-extract — transcript cleaner for the calibration analyser.

Validates that the extractor:
- drops <cairn_context> and <system-reminder> blocks from user text
- drops thinking blocks and [cm] memory link-defs from assistant text
- drops tool_use / tool_result blocks in --signal-only mode
- preserves them as one-line summaries in --with-tools mode
- slices by --turn-range and --last-N-minutes
- isolates corrections via --corrections-only
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import session_extract as se


def _write_jsonl(path, entries):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _user(text, ts=None):
    return {"type": "user", "message": {"content": text}, "timestamp": ts or ""}


def _user_blocks(blocks, ts=None):
    return {"type": "user", "message": {"content": blocks}, "timestamp": ts or ""}


def _asst(text, ts=None):
    return {"type": "assistant", "message": {"content": text}, "timestamp": ts or ""}


def _asst_blocks(blocks, ts=None):
    return {"type": "assistant", "message": {"content": blocks}, "timestamp": ts or ""}


def test_strips_cairn_context_from_user_text():
    out = se._clean_user_text(
        "real question\n<cairn_context query=\"x\">\n<entry id=\"1\">stuff</entry>\n</cairn_context>"
    )
    assert "real question" in out
    assert "cairn_context" not in out
    assert "entry" not in out


def test_strips_system_reminder():
    out = se._clean_user_text(
        "hi\n<system-reminder>noise noise</system-reminder>\nbye"
    )
    assert "noise" not in out
    assert "hi" in out and "bye" in out


def test_strips_thinking_from_assistant():
    out = se._clean_assistant_text(
        "<thinking>internal monologue</thinking>real answer"
    )
    assert "internal monologue" not in out
    assert "real answer" in out


def test_strips_memory_link_def():
    out = se._clean_assistant_text(
        "the answer\n[cm]: # '{\"e\":[{\"t\":\"fact\",\"to\":\"x\",\"c\":\"y\"}],\"ok\":true,\"ctx\":\"s\",\"kw\":[\"x\"]}'"
    )
    assert "the answer" in out
    assert "[cm]" not in out


def test_signal_only_drops_tool_only_turns():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.jsonl")
        _write_jsonl(path, [
            _user("hello"),
            _asst_blocks([
                {"type": "thinking", "thinking": "secret"},
                {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}},
            ]),
            _user_blocks([{"type": "tool_result", "content": "file1\nfile2"}]),
            _asst("done"),
        ])
        turns = se.load_turns(path)
        filtered = se.filter_turns(turns, signal_only=True)
        texts = [t["text"] for t in filtered]
        assert "hello" in texts
        assert "done" in texts
        # tool-only turns have empty text — dropped in signal_only
        assert "" not in texts
        assert "secret" not in "".join(texts)


def test_with_tools_keeps_tool_summary():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.jsonl")
        _write_jsonl(path, [
            _user("hi"),
            _asst_blocks([
                {"type": "text", "text": "I'll run it."},
                {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}},
            ]),
        ])
        turns = se.load_turns(path)
        rendered = se.render(turns, with_tools=True)
        assert "tool_use Bash" in rendered
        assert "I'll run it." in rendered


def test_corrections_only_filters_user_redirects():
    turns = [
        {"idx": 0, "role": "user", "text": "tell me about X",
         "has_tools": False, "tool_summary": "", "timestamp": ""},
        {"idx": 1, "role": "assistant", "text": "X is...",
         "has_tools": False, "tool_summary": "", "timestamp": ""},
        {"idx": 2, "role": "user", "text": "No, that's wrong, I meant Y",
         "has_tools": False, "tool_summary": "", "timestamp": ""},
        {"idx": 3, "role": "user", "text": "actually focus on Z",
         "has_tools": False, "tool_summary": "", "timestamp": ""},
    ]
    out = se.filter_turns(turns, corrections_only=True)
    assert len(out) == 2
    assert all(t["idx"] in (2, 3) for t in out)


def test_turn_range_slices():
    turns = [
        {"idx": i, "role": "user", "text": f"t{i}",
         "has_tools": False, "tool_summary": "", "timestamp": ""}
        for i in range(10)
    ]
    out = se.filter_turns(turns, turn_range=(3, 5))
    assert [t["idx"] for t in out] == [3, 4, 5]


def test_last_n_minutes_slices():
    now = datetime.now(timezone.utc)
    turns = []
    for i in range(5):
        ts = (now - timedelta(minutes=(4 - i) * 20)).isoformat()
        turns.append({"idx": i, "role": "user", "text": f"t{i}",
                      "has_tools": False, "tool_summary": "", "timestamp": ts})
    # latest is turn 4. last 30 min should include only turn 4 (turn 3 is 20 min before turn 4 → 20<=30 → keep)
    out = se.filter_turns(turns, last_minutes=30)
    kept = [t["idx"] for t in out]
    assert 4 in kept
    assert 3 in kept
    assert 0 not in kept


def test_parse_range_ok_and_bad():
    assert se.parse_range("3-9") == (3, 9)
    try:
        se.parse_range("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_load_turns_skips_malformed_json():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.jsonl")
        with open(path, "w") as f:
            f.write('{"type":"user","message":{"content":"ok"}}\n')
            f.write("not json garbage\n")
            f.write('{"type":"assistant","message":{"content":"yes"}}\n')
        turns = se.load_turns(path)
        assert [t["text"] for t in turns] == ["ok", "yes"]


def test_main_entry_emits_text(capsys):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.jsonl")
        _write_jsonl(path, [_user("hello"), _asst("hi back")])
        rc = se.main([path])
        out = capsys.readouterr().out
        assert rc == 0
        assert "hello" in out
        assert "hi back" in out
        assert "## turn" in out


def test_main_entry_json_mode(capsys):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.jsonl")
        _write_jsonl(path, [_user("hello"), _asst("hi")])
        rc = se.main([path, "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        data = json.loads(out)
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[1]["role"] == "assistant"
