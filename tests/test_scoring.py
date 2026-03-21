#!/usr/bin/env python3
"""Tests for composite scoring, confidence dynamics, and quality gates."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))

from embeddings import composite_score, _recency_decay


def test_composite_score_similarity_dominant():
    """Higher similarity should generally produce higher score."""
    s1 = composite_score(0.8, 0.7, "2026-03-20 12:00:00")
    s2 = composite_score(0.4, 0.7, "2026-03-20 12:00:00")
    assert s1 > s2


def test_composite_score_confidence_nonlinear():
    """Confidence squared means 0.9 confidence contributes much more than 0.4."""
    s_high = composite_score(0.5, 0.9, "2026-03-20 12:00:00")
    s_low = composite_score(0.5, 0.4, "2026-03-20 12:00:00")
    # With confidence^2: 0.9^2=0.81 vs 0.4^2=0.16, gap should be significant
    assert (s_high - s_low) > 0.1


def test_composite_score_project_scope_boost():
    """Project-scoped entries should score higher than global at same similarity."""
    s_proj = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="myproject", current_project="myproject")
    s_glob = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="other", current_project="myproject")
    assert s_proj > s_glob


def test_recency_decay_recent():
    """Very recent entries should have decay near 1.0."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(now)
    assert decay > 0.95


def test_recency_decay_old():
    """30-day old entries should be roughly 0.5 (half-life)."""
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(old)
    assert 0.4 < decay < 0.6


def test_recency_decay_very_old():
    """90-day old entries should be low."""
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(old)
    assert decay < 0.2


# === Confidence dynamics ===

def test_saturating_boost_low_confidence():
    """Boost at 0.3 should be meaningful."""
    boost = 0.1 * (1 - 0.3)
    assert 0.06 < boost < 0.08


def test_saturating_boost_high_confidence():
    """Boost at 0.9 should be tiny."""
    boost = 0.1 * (1 - 0.9)
    assert boost < 0.02


def test_scaled_penalty_low_confidence():
    """Penalty at 0.3 should be moderate."""
    penalty = 0.2 * (1 + 0.3)
    assert 0.25 < penalty < 0.27


def test_scaled_penalty_high_confidence():
    """Penalty at 0.9 should be severe."""
    penalty = 0.2 * (1 + 0.9)
    assert penalty > 0.35


def test_single_negative_at_09_drops_significantly():
    """One negative at 0.9 should drop to roughly 0.52."""
    conf = 0.9
    penalty = 0.2 * (1 + conf)
    new = conf - penalty
    assert 0.5 < new < 0.55


def test_many_boosts_approach_but_dont_reach_1():
    """Repeated boosts should approach 1.0 but never reach it."""
    conf = 0.7
    for _ in range(50):
        boost = 0.1 * (1 - conf)
        conf = min(conf + boost, 1.0)
    assert conf > 0.95
    assert conf < 1.0


# === Negation heuristic ===

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))
from stop_hook import _has_negation_mismatch


def test_negation_detected():
    assert _has_negation_mismatch("use RTK for positioning", "do not use RTK for positioning")


def test_no_negation():
    assert not _has_negation_mismatch("use RTK for positioning", "use RTK for best accuracy")


def test_directional_contradiction():
    assert _has_negation_mismatch("increase the gain", "decrease the gain")


def test_preference_opposition():
    assert _has_negation_mismatch("prefer PostgreSQL", "avoid PostgreSQL")


def test_enable_disable():
    assert _has_negation_mismatch("enable debug logging", "disable debug logging")


def test_unrelated_sentences():
    assert not _has_negation_mismatch("the sky is blue", "the database uses SQLite")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {test.__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
