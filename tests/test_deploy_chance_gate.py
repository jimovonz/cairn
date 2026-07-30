"""The deploy gate must not promote against a below-chance incumbent.

Pairwise agreement has a 50% null. The first live run scored the incumbent at
39.6% — below it. A ranker that loses to a coin is not merely weak, it is
anti-correlated with the labels, so "the student beat it by 26 points" reads as
decisive while proving nothing about the student. These tests pin the guard as
pure arithmetic so it holds without loading a model.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CHANCE = 0.5


def decide(ft_agr, inc_agr, deploy_margin=0.02):
    """Mirror of the gate in train_reranker.main (kept in sync by the tests
    below, which assert the live source still contains the guard)."""
    beats = ft_agr > (inc_agr or 0) + deploy_margin
    if (inc_agr is not None and inc_agr < CHANCE) or (ft_agr is not None and ft_agr < CHANCE):
        beats = False
    return beats


def test_below_chance_incumbent_blocks_a_large_apparent_win():
    """The exact shape of the first live run: 66.0% vs 39.6%."""
    assert decide(0.660, 0.396) is False


def test_below_chance_student_is_blocked_even_against_a_worse_incumbent():
    assert decide(0.45, 0.30) is False


def test_ordinary_win_above_chance_still_deploys():
    assert decide(0.70, 0.60) is True


def test_win_inside_the_margin_does_not_deploy():
    assert decide(0.605, 0.60) is False


def test_both_exactly_at_chance_does_not_deploy():
    """0.5 is the null itself, not a pass — and the margin excludes a tie anyway."""
    assert decide(0.50, 0.50) is False


def test_guard_is_present_in_the_shipped_gate():
    """Guards against the mirror above drifting from the real implementation."""
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "cairn", "train_reranker.py")).read()
    assert "CHANCE = 0.5" in src
    assert "inc_below_chance" in src and "student_below_chance" in src
    # The blocking assignment, not just the computation.
    assert "beats = False" in src
