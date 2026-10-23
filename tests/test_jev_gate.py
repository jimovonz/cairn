"""Jev quality gate: flag at capture, enforce once at the next turn, log every
score and outcome. External calls are stubbed; hook_state is an in-memory dict."""
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
    monkeypatch.setattr(jg, "JEV_GATE_LABELS", str(tmp_path / "labels.jsonl"))
    return state


def _log():
    return [json.loads(l) for l in open(jg.JEV_GATE_LOG)]


def _flag(state, topics, shown=True, kind="detail"):
    state[("s", jg.STATE_KEY)] = json.dumps({"flagged": [
        {"kind": kind, "type": "fact", "topic": t, "content": f"{t} detail", "p": 0.9} for t in topics], "shown": shown})


def _fake(block_quality="good", omission=0.1, user_correction=0.1, stale=None):
    """Answers keyed by which question set was asked."""
    def ask(key, state, questions):
        if "mismatch" in questions:
            bad = "wrong" in state.split("ENTRY UNDER REVIEW:")[-1]
            return ({"verdict": {"choice": "unsupported" if bad else "supported",
                                 "probabilities": {"supported": 0.05 if bad else 0.9, "inferred": 0.05}},
                     "mismatch": {"noul": 0.95 if bad else 0.1},
                     "durable": {"choice": "transient" if "PID" in state else "durable"},
                     "kind": {"choice": "fact"}, "standalone": {"noul": 0.9}}, 0.001, 0.4)
        if "quality" in questions:
            return ({"quality": {"choice": block_quality, "probabilities": {block_quality: 0.9}},
                     "omission": {"noul": omission}, "user_correction": {"noul": user_correction},
                     "task": {"choice": "debug"}}, 0.001, 0.4)
        return ({k: {"noul": (stale or {}).get(k, 0.1)} for k in questions}, 0.001, 0.4)
    return ask


@pytest.fixture
def wired(gate, monkeypatch):
    monkeypatch.setattr(jg, "_key", lambda: "k")
    monkeypatch.setattr(jg, "turn_state", lambda p: "USER PROMPT: x\n[assistant] did things")
    monkeypatch.setattr(jg, "turn_number", lambda p: 4)
    monkeypatch.setattr(jg, "delivered_memories", lambda s: [])
    monkeypatch.setattr(jg, "_ask", _fake())
    return gate


def test_redact_strips_api_keys():
    s = jg.redact("Metlink key d52jOr9CHf9pWPyH047hY1rDZyu4geEt7FMONgb9 token=abc123 path /home/x/y.py")
    assert "d52jOr9" not in s and "abc123" not in s and "/home/x/y.py" in s


def test_entry_detail_flag_uses_mismatch_threshold(wired):
    flags = jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "ok", "content": "fine"},
                                       {"type": "fact", "topic": "bad", "content": "wrong number"}])
    assert [(f["kind"], f["topic"]) for f in flags] == [("detail", "bad")]
    rec = _log()[0]
    assert {s["topic"]: s["flagged"] for s in rec["scores"]} == {"ok": False, "bad": True}
    assert rec["turn"] == 4 and rec["block"]["task"] == "debug" and rec["cost"] == 0.003


def test_log_only_entry_signals_recorded(wired):
    jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "state", "content": "running as PID 4411"}])
    s = _log()[0]["scores"][0]
    assert s["durable"] == "transient" and s["kind_pred"] == "fact" and s["standalone"] == 0.9


def test_empty_block_flagged_only_when_omission_high(wired, monkeypatch):
    monkeypatch.setattr(jg, "_ask", _fake(block_quality="generic_or_lazy", omission=0.2))
    assert jg.assess("s", "t.jsonl", []) == []          # empty but nothing durable missed
    monkeypatch.setattr(jg, "_ask", _fake(block_quality="generic_or_lazy", omission=0.7))
    flags = jg.assess("s", "t.jsonl", [])
    assert [f["kind"] for f in flags] == ["block"]


def test_nonempty_lazy_block_flagged(wired, monkeypatch):
    monkeypatch.setattr(jg, "_ask", _fake(block_quality="misses_finding"))
    flags = jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "ok", "content": "fine"}])
    assert [f["kind"] for f in flags] == ["block"] and "not captured" in flags[0]["content"]


def test_secret_flag_is_local_and_redacted(wired):
    flags = jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "keys",
                                        "content": "key d52jOr9CHf9pWPyH047hY1rDZyu4geEt7FMONgb9"}])
    assert flags[0]["kind"] == "secret" and "d52jOr9" not in flags[0]["content"]
    assert "d52jOr9" not in jg.pending_reminder("s")


def test_uncaptured_correction_logged_not_enforced(wired, monkeypatch):
    monkeypatch.setattr(jg, "_ask", _fake(user_correction=0.9))
    assert jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "ok", "content": "fine"}]) == []
    assert _log()[0]["block"]["uncaptured_correction"] is True


def test_stale_candidates_logged(wired, monkeypatch):
    monkeypatch.setattr(jg, "delivered_memories", lambda s: [
        {"id": 7, "type": "fact", "topic": "old", "content": "port is 8080"},
        {"id": 9, "type": "fact", "topic": "still", "content": "uses sqlite"}])
    monkeypatch.setattr(jg, "_ask", _fake(stale={"m7": 0.92}))
    jg.assess("s", "t.jsonl", [])
    st = next(r for r in _log() if r["event"] == "stale_candidates")
    assert st["memories"][0] == {"id": 7, "p": 0.92}
    assert jg.report()["stale_candidates_top"] == [{"id": 7, "max_p": 0.92, "times": 1}]


def test_new_flags_log_the_obligation_they_replace(wired):
    _flag(wired, ["old"], shown=False)
    jg.assess("s", "t.jsonl", [{"type": "fact", "topic": "new", "content": "wrong"}])
    assert "expired_unshown" in [r["event"] for r in _log()]


def test_assess_fails_open_without_key(gate, monkeypatch):
    monkeypatch.setattr(jg, "_key", lambda: None)
    monkeypatch.setattr(jg, "turn_state", lambda p: "x")
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
    assert {r["flag"]["topic"]: r["event"] for r in _log()} == {"a": "rebutted", "b": "corrected"}


def test_block_flag_answered_by_rebuttal_or_entries(gate):
    gate[("s", jg.STATE_KEY)] = json.dumps({"flagged": [{"kind": "block", "type": "block", "topic": "_block",
                                                         "content": "generic"}], "shown": True})
    text = "x\n\n[cm]: # '{\"jv\":{\"_block\":\"complete: chat turn\"},\"ok\":true,\"ctx\":\"s\",\"kw\":[\"k\"]}'"
    assert jg.check_obligation("s", text, []) is None
    assert _log()[-1]["event"] == "rebutted"


def test_unanswered_reprompts_once_then_clears(gate):
    _flag(gate, ["a"])
    reason = jg.check_obligation("s", "nothing", [])
    assert reason and "fact/a" in reason
    assert jg.check_obligation("s", "still nothing", []) is None
    assert ("s", jg.STATE_KEY) not in gate
    assert [r["event"] for r in _log()] == ["unanswered_reprompted", "ignored"]


def test_disabled_is_inert(gate, monkeypatch):
    monkeypatch.setenv("CAIRN_JEV_GATE", "0")
    _flag(gate, ["a"], shown=False)
    assert jg.pending_reminder("s") == "" and jg.check_obligation("s", "x", []) is None


def test_report_separates_precision_and_coverage(gate, monkeypatch):
    monkeypatch.setattr(jg, "_deliveries", lambda s, t: (0, 0))
    scores = [{"type": "fact", "topic": f"t{i}", "content": "c", "p": p, "flagged": p >= 0.8}
              for i, p in enumerate([0.95, 0.85, 0.82, 0.2, 0.1])]
    with open(jg.JEV_GATE_LOG, "w") as f:
        f.write(json.dumps({"event": "assessed", "session": "s", "transcript": "t", "turn": 2,
                            "scores": scores, "cost": 0.005, "latency_max": 0.5,
                            "flags": [{"kind": "detail", "topic": "t0"}]}) + "\n")
        for ev in ("rebutted", "corrected", "rebutted"):
            f.write(json.dumps({"event": ev, "session": "s", "flag": {"kind": "detail", "topic": "t0"}}) + "\n")
    with open(jg.JEV_GATE_LABELS, "w") as f:
        for t, p, fl, lab in [("t0", 0.95, True, "defect"), ("t1", 0.85, True, "fine"), ("t3", 0.2, False, "defect")]:
            f.write(json.dumps({"key": f"s|2|{t}", "p": p, "flagged": fl, "label": lab}) + "\n")
    r = jg.report()
    assert r["entries_scored"] == 5 and r["flag_rate"] == 0.6
    assert r["rebuttal_rate"] == round(2 / 3, 3)
    assert r["precision_labelled_flags"] == {"n": 2, "precision": 0.5}
    assert r["defects_among_labelled_unflagged"] == {"n": 1, "rate": 1.0}
    assert r["flags_by_kind"]["detail"] == {"raised": 1, "rebutted": 2, "corrected": 1}


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


def test_secret_detector_ignores_ids_and_hashes():
    assert not jg.has_secret("session f62c3c6d-fe31-4a1b-9c2d-1234567890ab")
    assert not jg.has_secret("commit 070cdb0d2355aa77b1c3e5f7a9b1c3d5e7f9a1b3")
    assert not jg.has_secret("branch l5-strengthen-the-retrieval-gate-for-now-x")
    assert jg.has_secret("key d52jOr9CHf9pWPyH047hY1rDZyu4geEt7FMONgb9")
    assert jg.has_secret("password: Alpine99!x")
