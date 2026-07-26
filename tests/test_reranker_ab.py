"""Randomised reranker arm assignment (spec 2S.5).

Every reranker verdict to date is flag-day confounded: a model change is a
deployment, not an assignment, so models are compared across different periods
with different work in them. Assigning arms per request turns the already-
recorded `memory_deliveries.reranker_model` into a randomised treatment label.
"""
from collections import Counter

from cairn.config import pick_reranker_arm

ARMS = ("student", "ms-marco")


def test_no_arms_means_default_behaviour():
    assert pick_reranker_arm((), "anything") is None
    assert pick_reranker_arm(None, "anything") is None


def test_assignment_is_stable_for_the_same_request():
    """A retry must not switch treatment — that would contaminate the
    comparison it exists to make clean."""
    assert pick_reranker_arm(ARMS, "same query") == pick_reranker_arm(ARMS, "same query")


def test_assignment_is_balanced_across_requests():
    counts = Counter(pick_reranker_arm(ARMS, f"query-{i}") for i in range(2000))
    assert set(counts) == set(ARMS)
    for arm in ARMS:
        assert 0.45 < counts[arm] / 2000 < 0.55, counts


def test_three_arms_are_also_balanced():
    arms = ("a", "b", "c")
    counts = Counter(pick_reranker_arm(arms, f"q{i}") for i in range(3000))
    assert set(counts) == set(arms)
    for arm in arms:
        assert 0.28 < counts[arm] / 3000 < 0.39, counts


def test_single_arm_always_selected():
    assert pick_reranker_arm(("only",), "x") == "only"


def test_empty_key_still_assigns():
    assert pick_reranker_arm(ARMS, "") in ARMS
    assert pick_reranker_arm(ARMS, None) in ARMS
