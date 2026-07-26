"""Per-document fingerprint granularity (spec 3.3).

Section-level fingerprints are correct but coarse: one changed docstring
invalidated the whole `docs` section and re-distilled every document in it.
These pin the narrowing without changing section-level change detection or
archival semantics.
"""
import cairn.ingest as ingest


DOCS = [{"file": "a.md", "content": "alpha"},
        {"file": "b.md", "content": "beta"},
        {"file": "c.md", "content": "gamma"}]


def test_list_of_dicts_splits_on_file_identity():
    units = ingest._section_units(DOCS)
    assert set(units) == {"a.md", "b.md", "c.md"}


def test_duplicate_identities_fall_back_to_one_opaque_unit():
    """Non-unique identity cannot address a unit, so pruning must not try."""
    dupes = [{"file": "a.md", "content": "1"}, {"file": "a.md", "content": "2"}]
    assert set(ingest._section_units(dupes)) == {"__whole__"}


def test_payload_without_identity_is_one_opaque_unit():
    assert set(ingest._section_units("just a string")) == {"__whole__"}
    assert set(ingest._section_units([1, 2, 3])) == {"__whole__"}


def test_dict_payload_splits_per_key():
    assert set(ingest._section_units({"x": 1, "y": 2})) == {"x", "y"}


def test_only_the_edited_document_is_reported_changed():
    cached = ingest._fingerprint_units("docs", DOCS)
    edited = [dict(d) for d in DOCS]
    edited[1]["content"] = "beta CHANGED"
    assert ingest.changed_units("docs", edited, cached) == {"b.md"}


def test_added_and_removed_documents():
    cached = ingest._fingerprint_units("docs", DOCS)
    added = DOCS + [{"file": "d.md", "content": "delta"}]
    assert ingest.changed_units("docs", added, cached) == {"d.md"}
    # A removal leaves no current unit to flag — section-level fingerprinting
    # still catches it, which is why unit pruning does not replace it.
    removed = DOCS[:2]
    assert ingest.changed_units("docs", removed, cached) == set()
    assert (ingest.compute_fingerprints({"docs": removed})["docs"]
            != ingest.compute_fingerprints({"docs": DOCS})["docs"])


def test_unknown_units_count_as_changed():
    """Empty cache must re-distil everything, never assume current."""
    assert ingest.changed_units("docs", DOCS, {}) == {"a.md", "b.md", "c.md"}


def test_pruning_keeps_only_requested_units_and_preserves_shape():
    out = ingest.prune_section_to_units(DOCS, {"b.md"})
    assert out == [{"file": "b.md", "content": "beta"}]


def test_pruning_an_opaque_section_returns_it_whole():
    """Emptying it would silently drop content the distiller needs."""
    assert ingest.prune_section_to_units("blob", set()) == "blob"


def test_prune_extractions_narrows_only_changed_sections():
    extractions = {"docs": DOCS, "tree": ["untouched"]}
    cached = {"docs": ingest._fingerprint_units("docs", DOCS)}
    edited = [dict(d) for d in DOCS]
    edited[2]["content"] = "gamma CHANGED"
    pruned, stats = ingest.prune_extractions(
        {"docs": edited, "tree": ["untouched"]}, {"docs"}, cached)
    assert pruned["docs"] == [{"file": "c.md", "content": "gamma CHANGED"}]
    assert pruned["tree"] == ["untouched"]      # unchanged section untouched
    assert stats["docs"] == (1, 3)


def test_no_cached_units_prunes_nothing():
    """Every unit reads as changed, so pruning would be a no-op with added risk."""
    pruned, stats = ingest.prune_extractions({"docs": DOCS}, {"docs"}, {})
    assert pruned["docs"] == DOCS
    assert stats == {}
