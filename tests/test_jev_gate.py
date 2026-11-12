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


@pytest.fixture
def test_redact_strips_api_keys():
    s = jg.redact("Metlink key d52jOr9CHf9pWPyH047hY1rDZyu4geEt7FMONgb9 token=abc123 path /home/x/y.py")
    assert "d52jOr9" not in s and "abc123" not in s and "/home/x/y.py" in s


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


def test_turn_state_can_target_an_earlier_turn(tmp_path):
    rec = lambda role, text: {"type": "message", "message": {"role": role, "content": [{"type": "text", "text": text}]}}
    lines = [{"type": "session", "version": 3, "id": "x", "cwd": "/"},
             rec("user", "first prompt"), rec("assistant", "first answer"),
             rec("user", "second prompt"), rec("assistant", "second answer")]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    s = jg.turn_state(str(p), turn_no=1)
    assert "first answer" in s and "second" not in s


@pytest.mark.parametrize("prompt,label", [
    ("WTF?!?! USE CAIRN!!!! you have gmail access", "missed_context"),
    ("If you do not know, do you not have explicit instructions to search?", "missed_context"),
    ("do it", "premature_stop"), ("Implement the rest", "premature_stop"), ("y", "premature_stop"),
    ("wtf!?!? Comitted and pushed what?!", "corrected"), ("No, IN YOUR MODEL", "corrected"),
    ("Is this proposal likely to improve Cairn?", "no_complaint"),
])
def test_classify_followup(prompt, label):
    assert jg.classify_followup(prompt) == label


def test_correction_label_arms_capture_requirement(gate):
    assert jg.label_previous_turn("s", "WTF?! that is wrong") == "corrected"
    reason = jg.check_correction_captured("s", [{"type": "fact", "topic": "x"}], "text")
    assert reason and "correction entry" in reason
    assert jg.check_correction_captured("s", [], "text") is None      # one-shot


def test_correction_capture_satisfied_or_declined(gate):
    jg.label_previous_turn("s", "No, the other machine")
    assert jg.check_correction_captured("s", [{"type": "correction", "topic": "x"}], "") is None
    jg.label_previous_turn("s", "No, the other machine")
    text = "ok\n\n[cm]: # '{\"jv\":{\"_correction\":\"none: user restated a preference\"},\"ok\":true}'"
    assert jg.check_correction_captured("s", [], text) is None
    assert _log()[-1]["event"] == "correction_declined"


def test_report_scores_checks_against_user_labels(gate, monkeypatch):
    monkeypatch.setattr(jg, "_deliveries", lambda s, t: (0, 0))
    with open(jg.JEV_GATE_LOG, "w") as f:
        for p, fired, legacy, follow in [(0.8, True, False, "do it"), (0.2, False, True, "thanks, next question?"),
                                         (0.7, True, False, "continue")]:
            f.write(json.dumps({"event": "stop_check", "session": "s", "p": p, "fired": fired, "legacy_fired": legacy}) + "\n")
            f.write(json.dumps({"event": "followup_label", "session": "s", "label": jg.classify_followup(follow)}) + "\n")
    v = jg.report()["vs_user_labels"]
    assert v["labels"] == {"premature_stop": 2, "no_complaint": 1}
    s = v["stop_check"]
    assert s["auc_vs_premature_stop"] == 1.0 and s["fired_then_premature_stop"] == 2
    assert s["legacy_fired_then_no_complaint"] == 1


# --- the single combined judgement at the stop ------------------------------

@pytest.fixture
def judge(gate, monkeypatch):
    import hooks.hook_helpers as hh
    monkeypatch.setattr(jg, "_key", lambda: "k")
    monkeypatch.setattr(jg, "turn_state", lambda *a, **k: "[USER PROMPT] fix it\n[assistant] fixed it")
    monkeypatch.setattr(jg, "turn_number", lambda p: 7)
    monkeypatch.setattr(jg, "regex_checks", lambda text, path: {"unkept_promise": False})
    s = {"calls": 0, "cairn_used": False, "states": [],
         "ans": {"stop_a": 0.2, "stop_b": 0.2, "ctx": 0.1, "omission": 0.1, "quality": "good"}}
    monkeypatch.setattr(hh, "cairn_query_invoked_this_turn", lambda p: s["cairn_used"])
    def ask(key, state, q):
        s["calls"] += 1; s["states"].append(state)
        a = {}
        for k, spec in q.items():
            if spec["type"] == "noul":
                v = s["ans"].get(k, 0.1)
                if k.startswith("e") and k[1:].isdigit():
                    v = 0.95 if "wrong" in state.split(f"ENTRY {int(k[1:]) + 1}:")[1].split("\n")[0] else 0.1
                a[k] = {"noul": v}
            else:
                c = s["ans"].get(k) or next(iter(spec["criteria"]))
                a[k] = {"choice": c, "probabilities": {c: 0.9}}
        return a, 0.0002, 0.5
    monkeypatch.setattr(jg, "_ask", ask)
    return s


def test_judge_turn_is_one_call_and_logs_all_events(judge):
    j = jg.judge_turn("s", "t.jsonl", "Done.", [{"type": "fact", "topic": "a", "content": "fine"},
                                                {"type": "fact", "topic": "b", "content": "wrong number"}])
    assert judge["calls"] == 1
    assert j.flags == [] and j.reprompt is None          # detail is logged, not enforced
    scored = next(r for r in _log() if r["event"] == "assessed")["scores"]
    assert {x["topic"]: x["flagged"] for x in scored} == {"a": False, "b": True}
    events = [r["event"] for r in _log()]
    assert {"stop_check", "context_check", "assessed"} <= set(events)
    assessed = next(r for r in _log() if r["event"] == "assessed")
    assert assessed["combined"] and assessed["turn"] == 7 and assessed["block"]["regex"] == {"unkept_promise": False}
    assert "ENTRY 2: fact/b: wrong number" in judge["states"][0] and "ASSISTANT'S FINAL REPLY" in judge["states"][0]


def test_judge_turn_stop_and_context_reprompts(judge):
    judge["ans"].update(stop_a=0.7, stop_b=0.6, ctx=0.9)
    j = jg.judge_turn("s", "t.jsonl", "Remaining: X. Want me to?", [])
    assert j.stop_reason and j.context_reason and "carry on" in j.reprompt and "search Cairn" in j.reprompt


def test_judge_turn_context_needs_cairn_unused_and_stop_respects_intent(judge):
    judge["ans"].update(stop_a=0.9, stop_b=0.9, ctx=0.95)
    judge["cairn_used"] = True
    j = jg.judge_turn("s", "t.jsonl", "reply", [], intent_resolved=True)
    assert j.reprompt is None


def test_judge_turn_block_and_secret_flags(judge):
    judge["ans"].update(omission=0.7)
    assert [f["kind"] for f in jg.judge_turn("s", "t.jsonl", "r", []).flags] == ["block"]
    judge["ans"].update(omission=0.1, quality="misses_finding")
    flags = jg.judge_turn("s", "t.jsonl", "r", [{"type": "fact", "topic": "k",
                                                 "content": "key d52jOr9CHf9pWPyH047hY1rDZyu4geEt7FMONgb9"}]).flags
    assert {f["kind"] for f in flags} == {"secret", "block"}
    assert all("d52jOr9" not in f["content"] for f in flags)


def test_judge_turn_fails_open(judge, monkeypatch):
    monkeypatch.setattr(jg, "_ask", lambda *a, **k: (None, 0.0, 4.0))
    j = jg.judge_turn("s", "t.jsonl", "r", [{"type": "fact", "topic": "a", "content": "b"}])
    assert j.reprompt is None and j.flags == [] and _log()[-1]["event"] == "judge_failed"
    monkeypatch.setattr(jg, "_key", lambda: None)
    assert jg.judge_turn("s", "t.jsonl", "r", []).reprompt is None


def test_save_flags_sets_obligation_and_logs_replaced(gate):
    _flag(gate, ["old"], shown=False)
    jg.save_flags("s", [{"kind": "detail", "type": "fact", "topic": "new", "content": "c", "p": 0.9}])
    assert json.loads(gate[("s", jg.STATE_KEY)])["flagged"][0]["topic"] == "new"
    assert _log()[-1]["event"] == "expired_unshown"


def test_assess_delivered_logs_memory_calibration_and_entry_signals(judge, monkeypatch):
    monkeypatch.setattr(jg, "delivered_memories", lambda s: [{"id": 7, "type": "fact", "topic": "t", "content": "c"}])
    monkeypatch.setattr(jg, "delivered_calibration", lambda s: [{"id": 3, "content": "be brief"}])
    judge["ans"].update({"u7": 0.8, "r7": 0.9, "m7": 0.75, "a3": 0.9, "c3": 0.7})
    jg.assess_delivered("s", "t.jsonl", [{"type": "fact", "topic": "x", "content": "y"}])
    d = next(r for r in _log() if r["event"] == "delivered_judged")
    assert d["memories"] == [{"id": 7, "stale": 0.75, "used": 0.8, "relevant": 0.9}]
    assert d["calibration"] == [{"row_id": 3, "applies": 0.9, "followed": 0.7}]
    assert d["entries"][0]["topic"] == "x" and "standalone" in d["entries"][0]
    assert jg.report()["stale_candidates_top"] == [{"id": 7, "max_p": 0.75, "times": 1}]


def test_detail_enforcement_can_be_restored(judge, monkeypatch):
    monkeypatch.setenv("CAIRN_JEV_DETAIL_ENFORCE", "1")
    j = jg.judge_turn("s", "t.jsonl", "Done.", [{"type": "fact", "topic": "b", "content": "wrong number"}])
    assert [f["kind"] for f in j.flags] == ["detail"]
