"""Stale-assertion verification (spec 4.1).

Cairn has no passive decay by design. This closes the narrow gap decay was
proposed for — a memory made false because the world moved — but only for the
checkable subset, and only as a FLAG: measurement showed file-absence is weak
evidence, so archiving on it would destroy true knowledge.
"""
import cairn.verify_stale as vs


def _exists(present):
    return lambda p: p in present


def test_historical_types_are_never_verified():
    """A correction naming a file that has since been deleted is still true —
    archiving it would destroy the learning trail."""
    for t in ("correction", "decision", "preference", "person"):
        verdict, _ = vs.classify(t, {"/gone.py"}, "/repo", _exists(set()))
        assert verdict == "skip"


def test_present_tense_types_are_verified():
    verdict, _ = vs.classify("fact", {"/gone.py"}, "/repo", _exists({"/repo"}))
    assert verdict == "stale"


def test_present_file_is_ok():
    verdict, _ = vs.classify("fact", {"/there.py"}, "/repo",
                             _exists({"/repo", "/there.py"}))
    assert verdict == "ok"


def test_absent_repo_root_never_judges_relative_paths():
    """Not checked out here is indistinguishable from deleted, and must not
    archive a project's memories on a machine that lacks it."""
    verdict, _ = vs.classify("fact", {"src/x.py"}, "/repo", _exists(set()))
    assert verdict == "skip"


def test_absolute_paths_need_no_repo_root():
    verdict, _ = vs.classify("fact", {"/abs/gone.py"}, None, _exists(set()))
    assert verdict == "stale"


def test_partial_absence_is_ambiguous_not_stale():
    verdict, _ = vs.classify("fact", {"/a.py", "/b.py"}, None, _exists({"/a.py"}))
    assert verdict == "skip"


def test_ephemeral_paths_are_never_evidence():
    """The failure that motivated this: 52 true memories flagged because a /tmp
    harvest file had been cleaned up."""
    assert vs.is_ephemeral("/tmp/findings.txt")
    assert vs.is_ephemeral("/home/u/.cache/x.json")
    assert vs.is_ephemeral("/proj/node_modules/y.js")
    assert not vs.is_ephemeral("/proj/src/real.py")
    verdict, reason = vs.classify("fact", {"/tmp/gone.txt"}, None, _exists(set()))
    assert verdict == "skip"
    assert "ephemeral" in reason


def test_path_extraction_requires_separator_and_extension():
    """Bare words produce false staleness, and a false archive costs more than
    a missed one."""
    found = vs.extract_paths("see src/app.py and also the word config here")
    assert "src/app.py" in found
    assert not any(p == "config" for p in found)


def test_path_extraction_handles_absolute_and_dotted():
    found = vs.extract_paths("check /etc/hosts.conf and ./local/file.md")
    assert "/etc/hosts.conf" in found
    assert "./local/file.md" in found


def test_no_paths_means_no_verdict():
    verdict, _ = vs.classify("fact", set(), "/repo", _exists({"/repo"}))
    assert verdict == "skip"
