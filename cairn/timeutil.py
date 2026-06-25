"""Time standardisation — storage is UTC (canonical), presentation is LOCAL.

cairn stores every timestamp as naive UTC (SQLite `CURRENT_TIMESTAMP` /
`datetime('now')`). That is correct and unambiguous for storage, sync ordering,
and comparison — but humans (and the agent) think in one local timezone, so
showing raw UTC and bucketing "today" in UTC causes systematic confusion.

This module is the SINGLE source of truth for the local timezone and for every
UTC<->local conversion. Rules:
  * NEVER display a stored timestamp raw — render it with `fmt_local`.
  * NEVER compare a stored (UTC) value against a local wall-clock — convert the
    local boundary to UTC with `since_bound_utc` / `until_bound_utc` first.

Local timezone resolution order: env `CAIRN_TZ` (IANA name, e.g.
`Pacific/Auckland`) -> `config.CAIRN_TZ` -> the system zone (`/etc/localtime`) ->
a fixed offset from the current clock (last-resort, no historical DST).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

_SQLITE_FMT = "%Y-%m-%d %H:%M:%S"


def _tz_name() -> Optional[str]:
    name = os.environ.get("CAIRN_TZ")
    if name:
        return name
    try:
        from cairn import config
        return getattr(config, "CAIRN_TZ", None)
    except Exception:
        return None


def local_tz():
    """Resolve the configured local timezone (see module docstring for order)."""
    name = _tz_name()
    if name and ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    if ZoneInfo is not None:
        try:
            link = os.readlink("/etc/localtime")
            if "zoneinfo/" in link:
                return ZoneInfo(link.split("zoneinfo/")[-1])
        except Exception:
            pass
    # Last resort: the system's current fixed offset (no historical DST accuracy).
    return datetime.now().astimezone().tzinfo or timezone.utc


def _parse_utc(ts) -> Optional[datetime]:
    """Parse a stored UTC timestamp (naive 'YYYY-MM-DD HH:MM:SS[.ffffff]' or ISO,
    or a datetime) into a UTC-aware datetime. None if unparseable."""
    if ts is None or ts == "":
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        s = str(ts).strip().replace("T", " ")
        # drop fractional seconds and any explicit offset suffix
        s = re.split(r"[.+]", s)[0].strip()
        try:
            dt = datetime.strptime(s[:19], _SQLITE_FMT)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_local(ts) -> Optional[datetime]:
    dt = _parse_utc(ts)
    return dt.astimezone(local_tz()) if dt else None


def fmt_local(ts, with_tz: bool = True) -> str:
    """Render a stored UTC timestamp in local time, e.g. '2026-06-26 09:04:34 NZST'.
    Returns the raw value unchanged if it can't be parsed (never raises)."""
    dt = to_local(ts)
    if dt is None:
        return str(ts) if ts else ""
    return dt.strftime(_SQLITE_FMT + (" %Z" if with_tz else ""))


def now_local() -> datetime:
    return datetime.now(local_tz())


def _to_utc_str(dt_local: datetime) -> str:
    return dt_local.astimezone(timezone.utc).strftime(_SQLITE_FMT)


def resolve_relative_local(s: str) -> Optional[datetime]:
    """Map today | yesterday | Nh | Nd | Nw | Nm | ISO(date|datetime) to a
    LOCAL-aware datetime (the lower instant). Day-grained values resolve to local
    midnight; Nh is an exact instant; ISO strings are interpreted as LOCAL."""
    now = now_local()
    lt = local_tz()
    s = (s or "").strip()
    midnight = lambda d: d.replace(hour=0, minute=0, second=0, microsecond=0)
    if s == "today":
        return midnight(now)
    if s == "yesterday":
        return midnight(now - timedelta(days=1))
    m = re.fullmatch(r"(\d+)([hdwm])", s)
    if m:
        n, u = int(m.group(1)), m.group(2)
        if u == "h":
            return now - timedelta(hours=n)
        if u == "d":
            return midnight(now - timedelta(days=n))
        if u == "w":
            return midnight(now - timedelta(weeks=n))
        if u == "m":
            return midnight(now - timedelta(days=30 * n))
    try:
        if len(s) <= 10:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=lt)
        return datetime.strptime(s[:19], _SQLITE_FMT).replace(tzinfo=lt)
    except ValueError:
        return None


def since_bound_utc(s: str) -> Optional[str]:
    """Lower UTC bound (stored-string form) for `--since <s>` interpreted in local."""
    dt = resolve_relative_local(s)
    return _to_utc_str(dt) if dt else None


def until_bound_utc(s: str) -> Optional[str]:
    """Upper UTC bound for `--until <s>` (local) — day-grained values extend to the
    end of that local day so `--until today` includes all of today."""
    dt = resolve_relative_local(s)
    if dt is None:
        return None
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        dt = dt.replace(hour=23, minute=59, second=59)
    return _to_utc_str(dt)
