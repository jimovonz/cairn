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
