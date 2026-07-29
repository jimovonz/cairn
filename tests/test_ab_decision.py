"""A/B promotion must not read 'unmeasurable' as '0% engaged' (spec 2S.1).

_arm_stats returns None for an arm with no two-class engagement stratum. The
previous `(pct or 0.0)` coercion turned that into the worst possible score, so
an arm could be promoted or rejected on a phantom gap.
"""
import cairn.ab_selfmod as ab
from cairn.ab_selfmod import AB_MIN_DELIVERIES_PER_ARM as MIN


def test_below_minimum_deliveries_keeps_running():
    status, _ = ab._ab_decision(MIN - 1, 50.0, MIN - 1, 90.0)
    assert status == "running"


def test_unmeasurable_arm_never_decides():
    """The bug this guards: pct_b=None previously became 0.0, producing a large
    negative gap and a spurious rejection."""
    status, reason = ab._ab_decision(MIN, 50.0, MIN, None)
    assert status == "running"
    assert "unmeasurable" in reason


def test_both_unmeasurable_never_decides():
    status, _ = ab._ab_decision(MIN, None, MIN, None)
    assert status == "running"


def test_clear_win_promotes():
    status, reason = ab._ab_decision(MIN, 40.0, MIN, 40.0 + ab.AB_PROMOTE_GAP_PCT)
    assert status == "promoted"
    assert "gap=" in reason


def test_clear_loss_rejects():
    status, _ = ab._ab_decision(MIN, 40.0, MIN, 40.0 + ab.AB_REJECT_GAP_PCT)
    assert status == "rejected"


def test_small_gap_is_inconclusive():
    status, _ = ab._ab_decision(MIN, 40.0, MIN, 40.0)
    assert status == "inconclusive"


def test_zero_percent_is_a_real_measurement_not_unknown():
    """A genuine 0.0 must still decide — only None means unmeasurable."""
    status, _ = ab._ab_decision(MIN, 40.0, MIN, 0.0)
    assert status == "rejected"
