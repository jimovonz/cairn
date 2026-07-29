"""Randomised reranker arm assignment (spec 2S.5).

Every reranker verdict to date is flag-day confounded: a model change is a
deployment, not an assignment, so models are compared across different periods
with different work in them. Assigning arms per request turns the already-
recorded `memory_deliveries.reranker_model` into a randomised treatment label.
"""
from collections import Counter

from cairn.config import pick_reranker_arm
from cairn.embeddings import _filter_by_ce_floor

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


def test_arm_floors_are_not_interchangeable(tmp_path):
    """Floors are per-model: the student is pairwise-trained with compressed
    logits (~-9.3) while ms-marco is pointwise (-3.0). Serving one arm the
    other's floor suppresses the wrong candidates and biases the comparison the
    experiment exists to make."""
    from cairn.config import (resolve_arm_floor, CROSS_ENCODER_SCORE_FLOOR,
                              CROSS_ENCODER_STUDENT_FLOOR)

    # A named pretrained model gets the static logit floor.
    assert resolve_arm_floor("cross-encoder/ms-marco-MiniLM-L-6-v2") == CROSS_ENCODER_SCORE_FLOOR

    # A local model dir ships its own floor.
    d = tmp_path / "student"
    d.mkdir()
    (d / "floor.txt").write_text("-9.3228\n")
    assert resolve_arm_floor(str(d)) == -9.3228

    # A local dir with no floor.txt falls back to suppression-off, never to the
    # ms-marco floor — guessing a floor for an uncalibrated model would silently
    # drop relevant memories.
    bare = tmp_path / "bare"
    bare.mkdir()
    assert resolve_arm_floor(str(bare)) == CROSS_ENCODER_STUDENT_FLOOR


def test_unreadable_floor_file_does_not_crash(tmp_path):
    from cairn.config import resolve_arm_floor, CROSS_ENCODER_STUDENT_FLOOR
    d = tmp_path / "m"
    d.mkdir()
    (d / "floor.txt").write_text("not-a-number")
    assert resolve_arm_floor(str(d)) == CROSS_ENCODER_STUDENT_FLOOR


def _rows(*scores):
    return [{"ce_score": s, "id": i} for i, s in enumerate(scores)]


def test_floor_applies_normally_when_no_arm_assigned():
    kept = _filter_by_ce_floor(_rows(2.0, -1.0, -8.0), floor=-3.0, arm_assigned=False)
    assert [r["ce_score"] for r in kept] == [2.0, -1.0]


def test_arm_assignment_keeps_every_candidate():
    """The defect this fixes: with each arm applying its own calibrated floor,
    the student delivered 4.7 memories per turn and ms-marco exactly 1.0, so
    delivery VOLUME was confounded with arm in an experiment about ORDERING."""
    rows = _rows(2.0, -1.0, -8.0, -20.0)
    kept = _filter_by_ce_floor(rows, floor=-3.0, arm_assigned=True)
    assert len(kept) == len(rows)


def test_both_arms_yield_equal_k_for_the_same_candidates():
    rows = _rows(6.8, 0.3, -4.0, -11.0)
    a = _filter_by_ce_floor(rows, floor=-3.0, arm_assigned=True)       # ms-marco floor
    b = _filter_by_ce_floor(rows, floor=-9.3228, arm_assigned=True)    # student floor
    assert len(a) == len(b) == len(rows)


def test_total_suppression_still_keeps_one_when_no_arm():
    """Existing safety net: never return nothing because the floor was harsh."""
    kept = _filter_by_ce_floor(_rows(-50.0, -60.0), floor=-3.0, arm_assigned=False)
    assert len(kept) == 1


def test_filter_does_not_mutate_its_input():
    rows = _rows(2.0, -8.0)
    _filter_by_ce_floor(rows, floor=-3.0, arm_assigned=True)
    assert len(rows) == 2
