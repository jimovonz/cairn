"""Pending-writes queue for the Stop hook.

Stop hook critical path:
  parse → validate → enqueue (this module) → spawn detached drain → exit

Drain worker (hooks/drain_queue.py):
  flock(LOCK_FILE, LOCK_EX | LOCK_NB) → if held, exit (incumbent will see new
  rows on next iteration) → loop SELECT * FROM pending_writes ORDER BY seq →
  call storage.insert_memories / apply_confidence_updates → DELETE row → release.

Trade-off: turn N+1's UserPromptSubmit hook may fire before the drain has
flushed turn N's entries. Acceptable because:
  - drain runs concurrently with prompt-hook retrieval (no serialization needed)
  - human typing latency is multi-second; drain at <100ms/entry catches up easily
  - if strict ordering ever matters, the prompt hook can flock-wait briefly
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

from hooks.hook_helpers import get_ephemeral_conn, log, log_error


def enqueue_memories(entries: list[dict], session_id: Optional[str],
                     transcript_path: Optional[str]) -> int:
    """Append a memory-insert job to the queue. Returns the seq assigned."""
    if not entries:
        return 0
    payload = json.dumps({"entries": entries})
    conn = get_ephemeral_conn()
    try:
        cur = conn.execute(
            "INSERT INTO pending_writes (kind, payload, session_id, transcript_path) "
            "VALUES (?, ?, ?, ?)",
            ("insert_memories", payload, session_id, transcript_path),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def enqueue_confidence_updates(updates: list[tuple], session_id: Optional[str]) -> int:
    """Append a confidence-update job to the queue."""
    if not updates:
        return 0
    payload = json.dumps({"updates": updates})
    conn = get_ephemeral_conn()
    try:
        cur = conn.execute(
            "INSERT INTO pending_writes (kind, payload, session_id) VALUES (?, ?, ?)",
            ("confidence_updates", payload, session_id),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def spawn_drain() -> None:
    """Fork a detached drain worker. If one already holds the flock, the new
    spawn exits cheaply; the incumbent will see our rows on its next iteration."""
    drain_path = os.path.join(os.path.dirname(__file__), "drain_queue.py")
    try:
        subprocess.Popen(
            [sys.executable, drain_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as e:
        log_error(f"failed to spawn drain worker: {type(e).__name__}: {e}")
