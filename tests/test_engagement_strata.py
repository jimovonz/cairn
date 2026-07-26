"""Engagement measurement bases are not comparable (spec 2S.1).

Untagged deliveries recorded only positives — non-engagement was never
distinguished from never-scored — so a rate computed across `engaged_method`
strata is invalid. A stratum is a usable basis only if it contains BOTH classes.
"""
import cairn.query as query


def test_stratum_with_both_classes_qualifies():
    rows = [("lexical", 1), ("lexical", 0), ("lexical", 0)]
    assert query._qualifying_strata(rows) == {"lexical"}


def test_positives_only_stratum_is_disqualified():
    """The untagged corpus: 1699 positives, zero negatives. A rate over it is
    100% by construction and means nothing."""
    rows = [(None, 1), (None, 1), (None, 1)]
    assert query._qualifying_strata(rows) == set()


def test_mixed_corpus_keeps_only_the_two_class_stratum():
    rows = [(None, 1), (None, 1), ("lexical", 1), ("lexical", 0),
            ("semantic-backfill", 1)]
    assert query._qualifying_strata(rows) == {"lexical"}


def test_nulls_are_ignored_when_qualifying():
    rows = [("lexical", None), ("lexical", 1), ("lexical", 0)]
    assert query._qualifying_strata(rows) == {"lexical"}


def test_stratum_of_only_nulls_does_not_qualify():
    assert query._qualifying_strata([("semantic", None)]) == set()


def test_unusable_stratum_is_neutralised_to_unlabelled():
    """A positives-only observation is not evidence of engagement rate — it is
    absence of measurement, so it must become None, not 1."""
    eng, score = query._neutralise_unusable_engagement(1, 0.9, None, {"lexical"})
    assert eng is None
    assert score is None


def test_qualifying_stratum_passes_through_untouched():
    eng, score = query._neutralise_unusable_engagement(0, 0.02, "lexical", {"lexical"})
    assert eng == 0
    assert score == 0.02


def test_neutralisation_makes_cross_stratum_rate_honest():
    """End-to-end: without neutralisation the blended rate is inflated by the
    positives-only stratum; with it, only the measurable stratum counts."""
    raw = [(None, 1), (None, 1), (None, 1), ("lexical", 1), ("lexical", 0)]
    qual = query._qualifying_strata(raw)
    recs = []
    for method, eng in raw:
        e, s = query._neutralise_unusable_engagement(eng, None, method, qual)
        recs.append({"key": "all", "engaged": e, "engaged_score": s, "grade": None})
    agg = query._aggregate_outcomes(recs)["all"]
    assert agg["n"] == 5
    assert agg["decided"] == 2          # only the lexical rows are measurements
    assert agg["engaged_pct"] == 50.0   # not 80.0
