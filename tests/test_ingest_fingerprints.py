"""Ingest incremental-invalidation tests (spec 1.3, 1.4).

These functions had zero test coverage while a third-party review claimed
removals went undetected. They do not — section payload hashing catches
deletions. The real defect was that ARCHIVAL keyed on LLM-self-reported
source_sections, which these tests now pin.
"""
import pytest

import cairn.ingest as ingest


# --- fingerprints (1.4) ---

def test_identical_extractions_fingerprint_identically():
    a = ingest.compute_fingerprints({"docs": ["a", "b"]})
    b = ingest.compute_fingerprints({"docs": ["a", "b"]})
    assert a == b


def test_removing_a_document_changes_the_fingerprint():
    """The claim under review: removals are undetectable. They are not — the
    hash covers the whole section payload."""
    before = ingest.compute_fingerprints({"docs": ["a", "b", "c"]})
    after = ingest.compute_fingerprints({"docs": ["a", "b"]})
    assert before["docs"] != after["docs"]


def test_removing_the_last_document_changes_the_fingerprint():
    before = ingest.compute_fingerprints({"docs": ["a"]})
    after = ingest.compute_fingerprints({"docs": []})
    assert before["docs"] != after["docs"]


def test_reordering_is_not_a_change():
    """Payload is JSON-serialised with sort_keys, so dict ordering must not
    produce spurious re-distillation."""
    a = ingest.compute_fingerprints({"cfg": {"x": 1, "y": 2}})
    b = ingest.compute_fingerprints({"cfg": {"y": 2, "x": 1}})
    assert a == b


def test_extractor_version_bump_invalidates_without_content_change(monkeypatch):
    before = ingest._fingerprint_section("docs", ["a"])
    bumped = dict(ingest.EXTRACTOR_VERSIONS)
    bumped["docs"] = bumped["docs"] + 1
    monkeypatch.setattr(ingest, "EXTRACTOR_VERSIONS", bumped)
    assert ingest._fingerprint_section("docs", ["a"]) != before


def test_diff_reports_changed_and_new_sections():
    cached = {"docs": "x", "tree": "y"}
    current = {"docs": "CHANGED", "tree": "y", "todos": "NEW"}
    assert ingest.diff_sections(current, cached) == {"docs", "todos"}


def test_diff_reports_nothing_when_unchanged():
    fps = {"docs": "x", "tree": "y"}
    assert ingest.diff_sections(fps, fps) == set()


# --- attribution / invalidation key (1.3) ---

def test_declared_sections_accepted_when_subset_of_what_was_fed():
    assert ingest._attribute_sections(["docs"], {"docs", "tree"}) == ["docs"]


def test_undeclared_entry_falls_back_to_whole_input_set():
    """No declaration must not mean 'attributed to nothing' — that would leave
    the entry permanently un-archivable."""
    assert ingest._attribute_sections([], {"docs", "tree"}) == ["docs", "tree"]


def test_hallucinated_section_discards_the_whole_declaration():
    """A section that was never fed in is evidence the declaration is
    unreliable, so none of it is trusted."""
    assert ingest._attribute_sections(["docs", "ros2"], {"docs"}) == ["docs"]


def test_full_mode_keeps_the_declaration_verbatim():
    """distilled_sections=None is full ingestion, where everything prior is
    archived regardless and attribution gates nothing."""
    assert ingest._attribute_sections(["docs"], None) == ["docs"]


def test_attribution_is_deterministically_ordered():
    assert ingest._attribute_sections([], {"tree", "docs"}) == ["docs", "tree"]


@pytest.mark.parametrize("declared", [None, [], ["nope"]])
def test_unusable_declarations_all_fail_safe(declared):
    """Over-archiving costs a re-distillation; under-archiving leaves a false
    memory forever. Every unusable case must pick the aggressive side."""
    assert ingest._attribute_sections(declared, {"a", "b"}) == ["a", "b"]
