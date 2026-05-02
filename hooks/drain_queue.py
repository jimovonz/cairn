#!/usr/bin/env python3
"""Drain worker for the Stop hook pending_writes queue.

Acquires an exclusive flock on a sentinel file. If another drain holds it,
exits immediately — the incumbent will see our newly-enqueued rows on its
next loop iteration. The holder loops SELECT ... ORDER BY seq until empty,
processing each row by calling the original synchronous write functions
(storage.insert_memories, storage.apply_confidence_updates) and DELETEing
on success.

Per-row failures are logged and the row is moved to a DLQ-style state by
incrementing a retry counter; rows exceeding MAX_RETRIES are deleted to
prevent indefinite blocking of subsequent jobs.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import sys
import time
from typing import Optional

# Ensure repo importable when invoked as a detached subprocess
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cairn.config import EPHEMERAL_DB_PATH  # noqa: E402
from hooks.hook_helpers import get_ephemeral_conn, log, log_error  # noqa: E402


LOCK_PATH = os.path.join(os.path.dirname(EPHEMERAL_DB_PATH), ".drain.lock")
MAX_RETRIES = 3
IDLE_EXIT_AFTER_S = 2.0  # if no rows for this long, exit


def _acquire_lock():
    """Try to acquire the exclusive drain lock. Returns the fd on success,
    None if another drainer is already running."""
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return None
        raise
    return fd


def _release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _process_one(row, conn) -> bool:
    """Process a single pending_writes row. Returns True on success."""
    seq, kind, payload_json, session_id, transcript_path = row
    payload = json.loads(payload_json)

    if kind == "insert_memories":
        from hooks.storage import insert_memories
        insert_memories(payload["entries"], session_id=session_id,
                        transcript_path=transcript_path)
    elif kind == "confidence_updates":
        from hooks.storage import apply_confidence_updates
        apply_confidence_updates(payload["updates"], session_id=session_id)
    else:
        log_error(f"drain: unknown kind {kind!r} for seq={seq}, dropping")
        return True  # treat unknown kind as processed (delete)
    return True


def _drain_loop(fd: int) -> None:
    """Drain rows in seq order until idle for IDLE_EXIT_AFTER_S."""
    last_activity = time.monotonic()
    processed = 0
    while True:
        conn = get_ephemeral_conn()
        try:
            row = conn.execute(
                "SELECT seq, kind, payload, session_id, transcript_path "
                "FROM pending_writes ORDER BY seq LIMIT 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            if time.monotonic() - last_activity > IDLE_EXIT_AFTER_S:
                if processed:
                    log(f"drain: processed {processed} rows, exiting (idle)")
                return
            time.sleep(0.05)
            continue

        seq = row[0]
        _last_error: Optional[str] = None
        try:
            ok = _process_one(row, None)
        except Exception as e:
            _last_error = f"{type(e).__name__}: {e}"
            log_error(f"drain: seq={seq} failed: {_last_error}")
            ok = False

        del_conn = get_ephemeral_conn()
        try:
            if ok:
                del_conn.execute("DELETE FROM pending_writes WHERE seq = ?", (seq,))
                del_conn.commit()
            else:
                # Increment attempts; only drop after MAX_RETRIES so transient
                # errors (DB locked, corruption mid-recovery) do not lose entries.
                attempts_row = del_conn.execute(
                    "SELECT attempts FROM pending_writes WHERE seq = ?", (seq,)
                ).fetchone()
                attempts = (attempts_row[0] if attempts_row else 0) + 1
                if attempts >= MAX_RETRIES:
                    del_conn.execute("DELETE FROM pending_writes WHERE seq = ?", (seq,))
                    log_error(f"drain: seq={seq} dropped after {attempts} attempts (poison row)")
                    del_conn.commit()
                else:
                    del_conn.execute(
                        "UPDATE pending_writes SET attempts = ?, last_error = ? WHERE seq = ?",
                        (attempts, str(_last_error)[:500] if _last_error else None, seq),
                    )
                    log(f"drain: seq={seq} attempt {attempts}/{MAX_RETRIES} failed, leaving in queue")
                    del_conn.commit()
                    # Stop on first failure — do not hammer a broken DB. Next drain
                    # spawn (after next user turn) retries.
                    return
        finally:
            del_conn.close()

        processed += 1
        last_activity = time.monotonic()


def main() -> int:
    fd = _acquire_lock()
    if fd is None:
        # Another drainer is running; it will pick up our rows.
        return 0
    try:
        _drain_loop(fd)
    finally:
        _release_lock(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
