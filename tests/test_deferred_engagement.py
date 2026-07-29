"""Deferred-value window and its self-validation gate (spec 2S.6).

The tool must not report layer rates until it can reproduce the LIVE engagement
verdict at window=1 — at that width it is the same scorer on the same turn, so
disagreement means the response reconstruction is wrong, and a deferred-value
curve built on mis-selected responses is indistinguishable from a real finding.
"""
from datetime import datetime

import cairn.deferred_engagement as de


def _t(minute):
    return datetime(2026, 7, 26, 12, minute, 0)


TURNS = [(_t(0), "first"), (_t(5), "second"), (_t(10), "third"), (_t(15), "fourth")]


def test_window_takes_responses_at_or_after_delivery():
    out = de._responses_within(TURNS, "2026-07-26 12:05:00", 2)
    assert out == ["second", "third"]


def test_window_is_capped_at_requested_size():
    assert len(de._responses_within(TURNS, "2026-07-26 12:00:00", 3)) == 3


def test_window_before_any_turn_takes_from_the_start():
    assert de._responses_within(TURNS, "2026-07-26 11:00:00", 1) == ["first"]


def test_unparseable_delivery_time_yields_no_window():
    assert de._responses_within(TURNS, "not-a-timestamp", 3) == []


def test_prompt_before_picks_the_most_recent_preceding_user_turn():
    users = [(_t(1), "early ask"), (_t(6), "later ask"), (_t(20), "future ask")]
    assert de._prompt_before(users, "2026-07-26 12:10:00") == "later ask"


def test_prompt_before_returns_empty_when_nothing_precedes():
    users = [(_t(20), "future ask")]
    assert de._prompt_before(users, "2026-07-26 12:00:00") == ""


def test_used_requires_the_live_detector_to_fire():
    """Any-token-overlap reported 76-90% where live reports ~15%; the shared
    scorer is what keeps windowed and live numbers comparable."""
    mem = "sqlite wal checkpoint corruption pysqlite3 guard"
    assert de._used(mem, "unrelated prompt", ["nothing relevant here at all"]) is False
    assert de._used(mem, "unrelated prompt",
                    ["the pysqlite3 guard prevents wal corruption"]) is True


def test_used_returns_none_when_memory_is_redundant_with_the_prompt():
    """No distinctive terms is undecidable, and must not score as a 0."""
    mem = "alpha beta"
    assert de._used(mem, "alpha beta", ["anything"]) is None


def test_used_fires_if_any_response_in_the_window_engages():
    mem = "sqlite wal checkpoint corruption pysqlite3 guard"
    assert de._used(mem, "prompt", ["irrelevant", "pysqlite3 guard fixed wal corruption"]) is True


def test_validation_gate_threshold_is_strict_enough_to_withhold_current_state():
    """Measured agreement was 39.5%; the gate must not be set below what the
    tool currently achieves, or it would report anyway."""
    assert de.VALIDATION_MIN_AGREEMENT > 0.5
