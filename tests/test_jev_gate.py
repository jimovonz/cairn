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
    monkeypatch.setattr(jg, "turn_number", lambda p: 4)
    monkeypatch.setattr(jg, "_ask", _fake_ask)
    flagged = jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "ok", "content": "fine"},
                                         {"type": "fact", "topic": "bad", "content": "wrong number"}])
    assert [f["topic"] for f in flagged] == ["bad"]
    assert json.loads(gate[("s", jg.STATE_KEY)])["shown"] is False
    rec = _log(jg.JEV_GATE_LOG)[0]
    # every score is logged, flagged or not, with the turn locator and cost
    assert {s["topic"]: s["flagged"] for s in rec["scores"]} == {"ok": False, "bad": True}
    assert rec["turn"] == 4 and rec["transcript"] == "t.jsonl" and rec["cost"] == 0.002


def _fake_ask(key, state):
    bad = "wrong" in state
    return ({"choice": "unsupported" if bad else "supported",
             "probabilities": {"supported": 0.05 if bad else 0.9, "inferred": 0.05,
                               "unsupported": 0.9 if bad else 0.05}}, 0.001, 0.4)


def test_new_flags_log_the_unanswered_obligation_they_replace(gate, monkeypatch):
    monkeypatch.setattr(jg, "_key", lambda: "k")
    monkeypatch.setattr(jg, "turn_state", lambda p: "USER PROMPT: x")
    monkeypatch.setattr(jg, "turn_number", lambda p: 1)
    monkeypatch.setattr(jg, "_ask", _fake_ask)
    _flag(gate, ["old"], shown=False)
    jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "new", "content": "wrong"}])
    events = [r["event"] for r in _log(jg.JEV_GATE_LOG)]
    assert "expired_unshown" in events


def test_report_separates_precision_and_coverage(gate, monkeypatch, tmp_path):
    monkeypatch.setattr(jg, "JEV_GATE_LABELS", str(tmp_path / "labels.jsonl"))
    monkeypatch.setattr(jg, "_deliveries", lambda s, t: (0, 0))
    scores = [{"type": "fact", "topic": f"t{i}", "content": "c", "p": p, "flagged": p >= 0.7}
              for i, p in enumerate([0.95, 0.8, 0.75, 0.2, 0.1])]
    with open(jg.JEV_GATE_LOG, "w") as f:
        f.write(json.dumps({"event": "assessed", "session": "s", "transcript": "t", "turn": 2,
                            "scores": scores, "cost": 0.005, "latency_max": 0.5}) + "\n")
        for ev in ("rebutted", "corrected", "rebutted"):
            f.write(json.dumps({"event": ev, "session": "s", "flag": {"topic": "t0"}}) + "\n")
    labels = [("t0", 0.95, True, "defect"), ("t1", 0.8, True, "fine"), ("t3", 0.2, False, "defect")]
    with open(jg.JEV_GATE_LABELS, "w") as f:
        for t, p, fl, lab in labels:
            f.write(json.dumps({"key": f"s|2|{t}", "p": p, "flagged": fl, "label": lab}) + "\n")
    r = jg.report()
    assert r["entries_scored"] == 5 and r["flag_rate"] == 0.6
    assert r["rebuttal_rate"] == round(2 / 3, 3)
    assert r["precision_labelled_flags"] == {"n": 2, "precision": 0.5}
    assert r["defects_among_labelled_unflagged"] == {"n": 1, "rate": 1.0}
    assert r["threshold_curve"]["0.9"]["labelled_precision"] == 1.0
    assert r["by_turn_position"]["1-3"]["n"] == 5


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
