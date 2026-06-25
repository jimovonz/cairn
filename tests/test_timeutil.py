"""cairn/timeutil.py — storage-UTC / display-local standardisation."""
from __future__ import annotations

import pytest

from cairn import timeutil


@pytest.fixture
def nzst(monkeypatch):
    # June = NZST (UTC+12, no DST) — deterministic for the fixed date used below.
    monkeypatch.setenv("CAIRN_TZ", "Pacific/Auckland")


def test_fmt_local_known_conversion(nzst):
    # The exact UTC->local conversion that caused the earlier misdiagnosis.
    assert timeutil.fmt_local("2026-06-25 21:04:34") == "2026-06-26 09:04:34 NZST"


def test_fmt_local_utc_identity(monkeypatch):
    monkeypatch.setenv("CAIRN_TZ", "UTC")
    assert timeutil.fmt_local("2026-06-25 21:04:34") == "2026-06-25 21:04:34 UTC"


def test_fmt_local_failsoft():
    assert timeutil.fmt_local("not a timestamp") == "not a timestamp"
    assert timeutil.fmt_local("") == ""
    assert timeutil.fmt_local(None) == ""


def test_parse_utc_variants(monkeypatch):
    monkeypatch.setenv("CAIRN_TZ", "UTC")
    # fractional seconds and ISO 'T' both parse; date-only does not (returns raw)
    assert timeutil.fmt_local("2026-06-25 21:04:34.123456") == "2026-06-25 21:04:34 UTC"
    assert timeutil.fmt_local("2026-06-25T21:04:34") == "2026-06-25 21:04:34 UTC"
    assert timeutil.fmt_local("2026-06-25") == "2026-06-25"  # date-only bucket: untouched


def test_day_bounds_utc_identity(monkeypatch):
    monkeypatch.setenv("CAIRN_TZ", "UTC")
    lo = timeutil.since_bound_utc("today")
    hi = timeutil.until_bound_utc("today")
    assert lo.endswith(" 00:00:00")
    assert hi.endswith(" 23:59:59")
    assert lo[:10] == hi[:10]  # same UTC day when TZ is UTC


def test_day_bounds_offset_zone(monkeypatch):
    # Etc/GMT-12 == UTC+12 (POSIX sign inversion). Local midnight -> previous UTC
    # day 12:00:00; local end-of-day -> 11:59:59 UTC. Proves the offset math.
    monkeypatch.setenv("CAIRN_TZ", "Etc/GMT-12")
    assert timeutil.since_bound_utc("today").endswith(" 12:00:00")
    assert timeutil.until_bound_utc("today").endswith(" 11:59:59")
    # since (local 00:00) is the UTC day BEFORE until (local 23:59) here
    assert timeutil.since_bound_utc("today") < timeutil.until_bound_utc("today")


def test_resolve_relative_local(monkeypatch):
    monkeypatch.setenv("CAIRN_TZ", "UTC")
    today = timeutil.resolve_relative_local("today")
    assert today.hour == 0 and today.minute == 0 and today.second == 0
    y = timeutil.resolve_relative_local("yesterday")
    assert (today - y).days == 1
    # Nh is an exact instant ~2h before now (not midnight)
    h = timeutil.resolve_relative_local("2h")
    delta = timeutil.now_local() - h
    assert 7000 < delta.total_seconds() < 7400
    assert timeutil.resolve_relative_local("garbage") is None


def test_iso_string_interpreted_local(monkeypatch):
    monkeypatch.setenv("CAIRN_TZ", "Etc/GMT-12")  # UTC+12
    # an ISO local date -> its UTC since-bound is the prior UTC day at 12:00
    assert timeutil.since_bound_utc("2026-03-10") == "2026-03-09 12:00:00"
