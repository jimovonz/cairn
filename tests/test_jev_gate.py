"""Jev quality gate: flag at capture, enforce once at the next turn, log every
outcome. External calls are stubbed; hook_state is an in-memory dict."""
import json

import pytest

from cairn import jev_gate as jg


@pytest.fixture
def gate(monkeypatch, tmp_path):
    monkeypatch.setenv("CAIRN_JEV_GATE", "1")
    import hooks.hook_helpers as hh
    state = {}
    monkeypatch.setattr(hh, "load_hook_state", lambda s, k: state.get((s, k)))
    monkeypatch.setattr(hh, "save_hook_state", lambda s, k, v: state.__setitem__((s, k), v))
    monkeypatch.setattr(hh, "delete_hook_state", lambda s, k: state.pop((s, k), None))
    monkeypatch.setattr(hh, "record_metric", lambda *a, **k: None)
    monkeypatch.setattr(jg, "JEV_GATE_LOG", str(tmp_path / "log.jsonl"))
    return state


def _flag(state, topics, shown=True):
    state[("s", jg.STATE_KEY)] = json.dumps({"flagged": [
        {"type": "fact", "topic": t, "content": f"{t} detail", "p": 0.9} for t in topics], "shown": shown})


def _log(gate_path):
    return [json.loads(l) for l in open(gate_path)]


def test_redact_strips_api_keys():
    s = jg.redact("Metlink key d52jOr9CHf9pWPyH047hY1rDZyu4geEt7FMONgb9 token=abc123 path /home/x/y.py")
    assert "d52jOr9" not in s and "abc123" not in s and "/home/x/y.py" in s


def test_assess_flags_only_above_threshold(gate, monkeypatch, tmp_path):
    monkeypatch.setattr(jg, "_key", lambda: "k")
    monkeypatch.setattr(jg, "turn_state", lambda p: "USER PROMPT: x")
    def fake_ask(key, state):
        bad = "wrong" in state
        return {"probabilities": {"supported": 0.05 if bad else 0.9, "inferred": 0.05 if bad else 0.05,
                                  "unsupported": 0.9 if bad else 0.05}}
    monkeypatch.setattr(jg, "_ask", fake_ask)
    flagged = jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "ok", "content": "fine"},
                                         {"type": "fact", "topic": "bad", "content": "wrong number"}])
    assert [f["topic"] for f in flagged] == ["bad"]
    assert json.loads(gate[("s", jg.STATE_KEY)])["shown"] is False


def test_assess_fails_open_without_key(gate, monkeypatch):
    monkeypatch.setattr(jg, "_key", lambda: None)
    assert jg.assess("s", "t", [{"type": "fact", "topic": "a", "content": "b"}]) == []
    assert ("s", jg.STATE_KEY) not in gate


def test_reminder_shown_once(gate):
    _flag(gate, ["a"], shown=False)
    first = jg.pending_reminder("s")
    assert "fact/a: a detail" in first and '"jv"' in first
    assert jg.pending_reminder("s") == ""


def test_obligation_not_checked_before_shown(gate):
    _flag(gate, ["a"], shown=False)
    assert jg.check_obligation("s", "no block", []) is None
    assert ("s", jg.STATE_KEY) in gate


def test_rebuttal_and_correction_clear_obligation(gate):
    _flag(gate, ["a", "b"])
    text = ("ok\n\n[cm]: # '{\"e\":[{\"t\":\"fact\",\"to\":\"b\",\"c\":\"fixed\"}],"
            "\"jv\":{\"a\":\"supported: tool result 2\"},\"ok\":true,\"ctx\":\"s\",\"kw\":[\"k\"]}'")
    assert jg.check_obligation("s", text, [{"type": "fact", "topic": "b", "content": "fixed"}]) is None
    assert ("s", jg.STATE_KEY) not in gate
    events = {r["flag"]["topic"]: r["event"] for r in _log(jg.JEV_GATE_LOG)}
    assert events == {"a": "rebutted", "b": "corrected"}


def test_unanswered_reprompts_once_then_clears(gate):
    _flag(gate, ["a"])
    reason = jg.check_obligation("s", "nothing", [])
    assert reason and "fact/a" in reason
    assert jg.check_obligation("s", "still nothing", []) is None
    assert ("s", jg.STATE_KEY) not in gate
    assert [r["event"] for r in _log(jg.JEV_GATE_LOG)] == ["unanswered_reprompted", "ignored"]


def test_disabled_is_inert(gate, monkeypatch):
    monkeypatch.setenv("CAIRN_JEV_GATE", "0")
    _flag(gate, ["a"], shown=False)
    assert jg.pending_reminder("s") == "" and jg.check_obligation("s", "x", []) is None


def test_turn_state_reads_pi_transcript_from_last_prompt(tmp_path):
    rec = lambda role, content, **kw: {"type": "message", "message": dict(role=role, content=content, **kw)}
    lines = [{"type": "session", "version": 3, "id": "x", "cwd": "/"},
             rec("user", [{"type": "text", "text": "old prompt"}]),
             rec("assistant", [{"type": "text", "text": "old answer"}]),
             rec("user", [{"type": "text", "text": "new prompt"}]),
             rec("assistant", [{"type": "toolCall", "id": "c", "name": "bash", "arguments": {"command": "ls"}}]),
             rec("toolResult", [{"type": "text", "text": "file.txt"}], toolCallId="c"),
             rec("assistant", [{"type": "text", "text": "found file.txt"}])]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    s = jg.turn_state(str(p))
    assert "new prompt" in s and "old prompt" not in s
    assert "[tool call] Bash" in s and "[tool result] file.txt" in s and "found file.txt" in s
