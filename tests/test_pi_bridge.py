"""The pi bridge must contribute the same write-path labels and provenance as
the Claude Code Stop hook, or every pi turn widens an unlabelled, unattributable
slice of the store (remediation programme, write-path gate)."""
import hooks.pi_bridge as pb
from cairn import config


def _freeze_provenance(monkeypatch):
    monkeypatch.setattr(config, "AB_TEST_ENABLED", False)
    monkeypatch.setattr(config, "GENERATION_PROMPT_VERSION", "genX")


def test_source_ref_carries_model_and_version(monkeypatch):
    _freeze_provenance(monkeypatch)
    assert pb._pi_source_ref("deepseek-flash", "s") == "pi:deepseek-flash:genX"
    assert pb._pi_source_ref("", "s") == "pi:unknown:genX"


def test_capture_writes_grades_engagement_and_provenance(monkeypatch, tmp_path):
    _freeze_provenance(monkeypatch)
    monkeypatch.setattr(pb, "_ensure_project", lambda *a, **k: None)

    import hooks.storage as storage
    import hooks.retrieval as retrieval
    import cairn.relevance as rel

    monkeypatch.setattr(retrieval, "layer2_cross_project_search", lambda *a, **k: None)
    seen = {}
    monkeypatch.setattr(storage, "insert_memories",
                        lambda entries, **k: seen.update(k) or len(entries))
    calls = {"grades": 0, "engagement": 0}
    monkeypatch.setattr(rel, "apply_relevance_grades",
                        lambda *a, **k: calls.__setitem__("grades", calls["grades"] + 1))
    monkeypatch.setattr(rel, "apply_engagement",
                        lambda *a, **k: calls.__setitem__("engagement", calls["engagement"] + 1))

    block = ('[cm]: # \'{"e":[{"t":"fact","to":"x","c":"y"}],"ok":true,'
             '"ctx":"s","kw":["a"],"rg":["5:3"]}\'')
    f = tmp_path / "r.txt"
    f.write_text("Answer\n\n" + block)

    class A:
        text_file = str(f)
        session = "s"
        transcript = ""
        cwd = ""
        model = "m"
        continuation = 0

    pb.cmd_capture(A())
    assert seen.get("source_ref") == "pi:m:genX"
    assert calls == {"grades": 1, "engagement": 1}


def test_cc_tool_name_maps_pi_spellings():
    assert pb._cc_tool_name("read") == "Read"
    assert pb._cc_tool_name("edit") == "Edit"
    assert pb._cc_tool_name("multi_edit") == "MultiEdit"
    assert pb._cc_tool_name("Bash") == "Bash"  # already Claude Code spelling


def test_extract_additional_context_unwraps_envelope():
    import json
    env = json.dumps({"hookSpecificOutput": {"additionalContext": "CAIRN GOTCHA: x"}})
    assert pb._extract_additional_context(env) == "CAIRN GOTCHA: x"
    assert pb._extract_additional_context("") == ""
    assert pb._extract_additional_context("plain text") == "plain text"


def test_cmd_pretool_prints_extracted_context(monkeypatch, tmp_path, capsys):
    import json
    import subprocess

    f = tmp_path / "p.json"
    f.write_text(json.dumps({"tool": "read", "input": {"file_path": "x.py"}}))

    class R:
        stdout = json.dumps({"hookSpecificOutput": {"additionalContext": "CAIRN GOTCHA: z"}})
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())

    class A:
        text_file = str(f)
        session = "s"
        transcript = ""
        cwd = ""
        model = ""

    pb.cmd_pretool(A())
    assert "CAIRN GOTCHA: z" in capsys.readouterr().out


def test_cc_tool_name_maps_bash():
    assert pb._cc_tool_name("bash") == "Bash"


class _Args:
    session = "s"
    transcript = ""
    cwd = ""
    model = "m"
    continuation = 0
    first = False
    query = "q"
    text_file = ""


def test_retrieve_appends_calibration_profile(monkeypatch, capsys):
    monkeypatch.setattr(pb, "_ensure_project", lambda *a, **k: None)
    import hooks.prompt_hook as ph
    import cairn.calibration_inject as ci
    monkeypatch.setattr(ph, "layer1_5_search", lambda *a, **k: "<cairn_context/>")
    monkeypatch.setattr(ci, "inject_for_prompt", lambda q, s: "<calibration_profile/>")
    pb.cmd_retrieve(_Args())
    out = capsys.readouterr().out
    assert "<cairn_context/>" in out and "<calibration_profile/>" in out


def test_capture_stores_memory_notes_deduped(monkeypatch, tmp_path):
    _freeze_provenance(monkeypatch)
    monkeypatch.setattr(pb, "_ensure_project", lambda *a, **k: None)
    import hooks.storage as storage
    import hooks.retrieval as retrieval
    import hooks.hook_helpers as hh
    monkeypatch.setattr(retrieval, "layer2_cross_project_search", lambda *a, **k: None)
    state = {}
    monkeypatch.setattr(hh, "load_hook_state", lambda s, k: state.get(k))
    monkeypatch.setattr(hh, "save_hook_state", lambda s, k, v: state.__setitem__(k, v))
    monkeypatch.setattr(hh, "record_metric", lambda *a, **k: None)
    batches = []
    monkeypatch.setattr(storage, "insert_memories",
                        lambda entries, **k: batches.append(entries) or len(entries))
    text = ("Found it. <memory_note>fact/new-thing: mid-turn finding</memory_note>\n"
            "Dup <memory_note>fact/x: same as final</memory_note>\n"
            "Template <memory_note>type/topic: observation</memory_note>\n\n"
            '[cm]: # \'{"e":[{"t":"fact","to":"x","c":"y"}],"ok":true,"ctx":"s","kw":["a"]}\'')
    f = tmp_path / "r.txt"
    f.write_text(text)
    a = _Args()
    a.text_file = str(f)
    pb.cmd_capture(a)
    notes = [e for batch in batches for e in batch if e.get("topic") == "new-thing"]
    assert len(notes) == 1
    assert not any(e.get("type") == "type" for batch in batches for e in batch)
    assert state["memory_notes_stored"] == "1"


def test_enforce_blocks_trailing_intent(monkeypatch, tmp_path, capsys):
    import hooks.hook_helpers as hh
    monkeypatch.setattr(hh, "record_metric", lambda *a, **k: None)
    f = tmp_path / "r.txt"
    f.write_text("I checked the config. Let me now run the migration.\n\n"
                 '[cm]: # \'{"ok":true,"ctx":"s","kw":["a"]}\'')
    a = _Args()
    a.text_file = str(f)
    pb.cmd_enforce(a)
    assert "stated intent to act" in capsys.readouterr().out


def test_enforce_intent_resolved_allows(monkeypatch, tmp_path, capsys):
    import hooks.hook_helpers as hh
    monkeypatch.setattr(hh, "record_metric", lambda *a, **k: None)
    f = tmp_path / "r.txt"
    f.write_text("I checked the config. Let me now run the migration.\n\n"
                 '[cm]: # \'{"ok":true,"ctx":"s","kw":["a"],"int":"resolved"}\'')
    a = _Args()
    a.text_file = str(f)
    pb.cmd_enforce(a)
    assert capsys.readouterr().out == ""
