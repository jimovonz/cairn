"""Signal liveness (spec follow-up).

Four failures this session shared one shape: a signal whose failure was
indistinguishable from success. Delivery logging died for 20 hours behind a
fail-soft handler and was caught only because a human asked. This check exists to
make silence reportable — but only real silence, or it becomes noise and gets
ignored.
"""
from datetime import datetime, timedelta, timezone

import cairn.signal_liveness as sl


def test_fresh_signal_is_ok():
    assert sl.classify(ever_seen=True, age_hours=1.0, max_silence_h=24,
                       system_active=True) == "OK"


def test_silent_signal_while_system_active_is_a_fault():
    """The 20-hour outage: stops arriving while work is still happening."""
    assert sl.classify(ever_seen=True, age_hours=30.0, max_silence_h=24,
                       system_active=True) == "STALE"


def test_silent_signal_while_system_idle_is_expected():
    """Nobody used the system, so silence is not evidence of a fault. Flagging
    this would fire every weekend and train everyone to ignore the check."""
    assert sl.classify(ever_seen=True, age_hours=30.0, max_silence_h=24,
                       system_active=False) == "IDLE"


def test_never_seen_is_distinct_from_stopped():
    """A just-shipped instrument must not look broken, and a dead one must not
    look new."""
    assert sl.classify(ever_seen=False, age_hours=None, max_silence_h=24,
                       system_active=True) == "NEVER"


def test_rare_signal_is_never_stale():
    """enforcement_block is legitimately infrequent; a cadence expectation would
    make it permanently red."""
    assert sl.classify(ever_seen=True, age_hours=500.0, max_silence_h=None,
                       system_active=True) == "RARE"


def test_unreadable_timestamp_is_not_treated_as_healthy():
    """Optimistic-on-unknown is the exact instinct this module corrects."""
    assert sl.classify(ever_seen=True, age_hours=None, max_silence_h=24,
                       system_active=True) == "UNKNOWN"


def test_age_hours_parses_stored_utc_format():
    now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    assert abs(sl._age_hours("2026-07-29 09:00:00", now=now) - 3.0) < 0.01


def test_age_hours_handles_fractional_seconds_and_iso():
    now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    assert abs(sl._age_hours("2026-07-29T09:00:00", now=now) - 3.0) < 0.01
    assert abs(sl._age_hours("2026-07-29 09:00:00.123456", now=now) - 3.0) < 0.01


def test_age_hours_returns_none_for_garbage_not_zero():
    """Returning 0 would read as 'just arrived' — the healthiest possible value
    for the least trustworthy input."""
    assert sl._age_hours("not-a-timestamp") is None
    assert sl._age_hours(None) is None


def test_reference_signal_is_one_of_the_tracked_signals():
    """The idle/stale distinction is meaningless if the reference is not itself
    collected."""
    assert sl.REFERENCE_SIGNAL in [s["name"] for s in sl.SIGNALS]


def test_every_signal_declares_why_it_is_tracked():
    for s in sl.SIGNALS:
        assert s.get("why"), f"{s['name']} has no rationale"
        assert "max_silence_h" in s


def _seed(tmp_path, deliveries_age_h, hook_age_h):
    """Ephemeral + durable DBs with controlled signal ages."""
    import pysqlite3 as sqlite3
    import cairn.init_db as init_db

    eph = str(tmp_path / "eph.db")
    dur = str(tmp_path / "dur.db")
    init_db.init_ephemeral(eph)
    old = init_db.DB_PATH
    init_db.DB_PATH = dur
    try:
        init_db.init()
    finally:
        init_db.DB_PATH = old

    now = datetime.now(timezone.utc)
    stamp = lambda h: (now - timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
    c = sqlite3.connect(eph)
    c.execute("INSERT INTO metrics (event, created_at) VALUES ('hook_fired', ?)",
              (stamp(hook_age_h),))
    c.execute("INSERT INTO memory_deliveries (session_id, memory_id, delivered_at) "
              "VALUES ('s', 1, ?)", (stamp(deliveries_age_h),))
    c.commit()
    c.close()
    return eph, dur


def test_end_to_end_reports_stale_when_a_signal_stops(tmp_path):
    """The 20-hour outage, reproduced: work still happening, deliveries silent.
    This asserts the whole path — queries, age arithmetic, reference logic — not
    just the policy function."""
    eph, dur = _seed(tmp_path, deliveries_age_h=40, hook_age_h=0.5)
    rows, active = sl.collect(hours=24, eph_path=eph, durable_path=dur)
    by = {r["name"]: r for r in rows}
    assert active is True, "hook_fired within the window means the system was active"
    assert by["memory_deliveries"]["status"] == "STALE"


def test_end_to_end_reports_idle_when_nothing_was_running(tmp_path):
    """Same silence, but nobody was using the system — must not be a fault."""
    eph, dur = _seed(tmp_path, deliveries_age_h=40, hook_age_h=40)
    rows, active = sl.collect(hours=24, eph_path=eph, durable_path=dur)
    by = {r["name"]: r for r in rows}
    assert active is False
    assert by["memory_deliveries"]["status"] == "IDLE"


def test_render_exit_code_is_nonzero_only_when_action_is_needed(tmp_path, capsys):
    eph, dur = _seed(tmp_path, deliveries_age_h=40, hook_age_h=0.5)
    rows, active = sl.collect(hours=24, eph_path=eph, durable_path=dur)
    assert sl.render(rows, active, 24, (True, "responding")) == 1

    healthy_dir = tmp_path / "healthy"
    healthy_dir.mkdir()
    eph2, dur2 = _seed(healthy_dir, deliveries_age_h=0.2, hook_age_h=0.2)
    rows2, active2 = sl.collect(hours=24, eph_path=eph2, durable_path=dur2)
    assert sl.render(rows2, active2, 24, (True, "responding")) == 0
