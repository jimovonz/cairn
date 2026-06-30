#!/usr/bin/env python3
"""Capture watchdog — detect the silent memory-capture stall before it hides for days.

Background: a writer (historically the sync service holding an idle write
transaction) can hold SQLite's single-writer lock on cairn.db, so every Stop-hook
memory insert times out and gets queued to the ephemeral ``pending_writes`` table.
Nothing is lost, but capture silently stops until the lock clears — once for ~2
days before this watchdog existed. This checks the two leading indicators:

  1. cairn.db write-lock availability (a non-mutating ``BEGIN IMMEDIATE`` probe —
     acquires the write lock then rolls back, so it neither writes nor contends
     beyond a short busy_timeout).
  2. ``pending_writes`` queue depth — memories waiting to drain.

Plus an informational capture-staleness age (MAX(created_at)); staleness alone is
not failed on (an idle session is normal), only flagged alongside a real signal.

Exit code: 0 healthy, 1 unhealthy (held lock, or queue over threshold). Wire into
the hourly cron or the daemon healthcheck so a stalled capture alerts in minutes.

Thresholds (env overrides):
  CAIRN_CAPTURE_PENDING_WARN   queue depth that fails the check (default 25)
  CAIRN_CAPTURE_STALE_HOURS    age that flags staleness as notable (default 12)
"""
from __future__ import annotations

import os
import sys

# Single-SQLite-library guard (see CLAUDE.md / tests/test_sqlite_guard.py): a
# stdlib-vs-pysqlite3 mix on a WAL cairn DB risks corruption.
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err

# Re-exec under the cairn venv before the guard above can trip under a bare
# python3. No-op inside a venv.
if __name__ == "__main__":
    _venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "python3")
    if os.path.exists(_venv) and sys.prefix == sys.base_prefix:
        os.execv(_venv, [_venv] + sys.argv)

PENDING_WARN = int(os.environ.get("CAIRN_CAPTURE_PENDING_WARN", "25"))
STALE_HOURS = float(os.environ.get("CAIRN_CAPTURE_STALE_HOURS", "12"))


def _durable_db():
    from cairn import init_db
    return init_db.DB_PATH


def _ephemeral_db():
    from cairn import config
    return config.EPHEMERAL_DB_PATH


def _lock_held(db_path):
    """True if cairn.db's write lock is held by another connection. Non-mutating:
    BEGIN IMMEDIATE acquires the reserved/write lock, then we roll back."""
    try:
        conn = sqlite3.connect(db_path, timeout=0)
        try:
            conn.execute("PRAGMA busy_timeout=2000")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
            return False, "writable"
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            return True, str(e)
        return True, f"unexpected: {e}"
    except Exception as e:  # noqa: BLE001
        return True, f"open failed: {e}"


def _pending_depth(eph_path):
    try:
        conn = sqlite3.connect(f"file:{eph_path}?mode=ro", uri=True)
        try:
            return conn.execute("SELECT COUNT(*) FROM pending_writes").fetchone()[0]
        finally:
            conn.close()
    except Exception:
        return None  # table/db absent -> no queue


def _capture_age_hours(db_path):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT (julianday('now') - julianday(MAX(created_at))) * 24 FROM memories"
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception:
        return None


def check():
    db, eph = _durable_db(), _ephemeral_db()
    locked, detail = _lock_held(db)
    pending = _pending_depth(eph)
    age = _capture_age_hours(db)

    unhealthy = False
    print("cairn capture watchdog:")
    if locked:
        print(f"  [CRIT] cairn.db write lock HELD - capture is stalling ({detail})")
        unhealthy = True
    else:
        print("  [ok]   cairn.db write-lock available")

    if pending is None:
        print("  [ok]   pending_writes: none queued")
    elif pending > PENDING_WARN:
        print(f"  [CRIT] pending_writes={pending} (> {PENDING_WARN}) - memories not draining")
        unhealthy = True
    elif pending > 0:
        print(f"  [warn] pending_writes={pending} (draining)")
    else:
        print("  [ok]   pending_writes: 0")

    if age is None:
        print("  [ok]   capture age: no memories yet / unknown")
    elif age > STALE_HOURS:
        flag = "CRIT" if (locked or (pending or 0) > PENDING_WARN) else "warn"
        suffix = "" if flag == "CRIT" else " - may just be idle"
        print(f"  [{flag}] last capture {age:.1f}h ago (> {STALE_HOURS}h){suffix}")
    else:
        print(f"  [ok]   last capture {age:.1f}h ago")

    print("STATUS:", "UNHEALTHY" if unhealthy else "healthy")
    return 1 if unhealthy else 0


if __name__ == "__main__":
    sys.exit(check())
