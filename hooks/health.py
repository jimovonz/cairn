"""Detect systemic Cairn failures, manage sentinel file, emit alerts."""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Optional

from cairn.config import (
    SENTINEL_PATH, DAEMON_FAIL_THRESHOLD, EMBEDDING_FAIL_WINDOW,
    EMBEDDING_FAIL_RATE_THRESHOLD, HOOK_CRASH_WINDOW_MINUTES, HOOK_CRASH_THRESHOLD,
)


def record_failure(kind: str, detail: str = "") -> None:
    """Record a failure event. If threshold crossed, write sentinel."""
    try:
        from hooks.hook_helpers import get_ephemeral_conn, record_metric
        record_metric("", f"health_failure_{kind}", detail)
        conn = get_ephemeral_conn()
        conn.execute(
            "INSERT INTO metrics (event, detail, value) VALUES (?, ?, 1)",
            (f"health_failure_{kind}", detail)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
    info = check_systemic(kind)
    if info:
        write_sentinel(info)


def record_success(kind: str) -> None:
    """Record a success event. If sentinel matches kind, clear it."""
    try:
        from hooks.hook_helpers import get_ephemeral_conn
        conn = get_ephemeral_conn()
        conn.execute(
            "INSERT INTO metrics (event, value) VALUES (?, 1)",
            (f"health_success_{kind}",)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
    existing = sentinel_info()
    if existing and existing.get("reason", "").startswith(kind):
        clear_sentinel()


def check_systemic(kind: str = "") -> Optional[dict]:
    """Evaluate failure thresholds. Return dict if systemic, None otherwise."""
    try:
        from hooks.hook_helpers import get_ephemeral_conn
        conn = get_ephemeral_conn()

        if kind in ("daemon", ""):
            rows = conn.execute(
                "SELECT COUNT(*) FROM metrics WHERE event = 'health_failure_daemon' "
                "AND created_at > datetime('now', '-5 minutes')"
            ).fetchone()
            if rows and rows[0] >= DAEMON_FAIL_THRESHOLD:
                conn.close()
                return {
                    "reason": "daemon_unreachable",
                    "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "count": rows[0],
                }

        if kind in ("embedding", ""):
            total = conn.execute(
                f"SELECT COUNT(*) FROM metrics WHERE event LIKE 'health_%_embedding' "
                f"ORDER BY created_at DESC LIMIT {EMBEDDING_FAIL_WINDOW}"
            ).fetchone()[0]
            if total >= EMBEDDING_FAIL_WINDOW:
                fails = conn.execute(
                    f"SELECT COUNT(*) FROM ("
                    f"  SELECT event FROM metrics WHERE event LIKE 'health_%_embedding' "
                    f"  ORDER BY created_at DESC LIMIT {EMBEDDING_FAIL_WINDOW}"
                    f") WHERE event = 'health_failure_embedding'"
                ).fetchone()[0]
                if fails / max(total, 1) > EMBEDDING_FAIL_RATE_THRESHOLD:
                    conn.close()
                    return {
                        "reason": "embedding_failure_rate",
                        "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "count": fails,
                    }

        if kind in ("hook_crash", ""):
            crashes = conn.execute(
                "SELECT COUNT(*) FROM metrics WHERE event = 'health_failure_hook_crash' "
                f"AND created_at > datetime('now', '-{HOOK_CRASH_WINDOW_MINUTES} minutes')"
            ).fetchone()[0]
            if crashes >= HOOK_CRASH_THRESHOLD:
                conn.close()
                return {
                    "reason": "hook_crash_rate",
                    "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "count": crashes,
                }

        conn.close()
    except Exception:
        pass
    return None


def write_sentinel(info: dict) -> None:
    """Write .impaired atomically. Call notify-send (best-effort)."""
    try:
        tmp = SENTINEL_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(info, f)
        os.replace(tmp, SENTINEL_PATH)
    except Exception:
        return
    try:
        subprocess.Popen(
            ["notify-send", "-u", "critical", "Cairn impaired", info.get("reason", "unknown")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def clear_sentinel() -> None:
    """Remove .impaired if present."""
    try:
        os.remove(SENTINEL_PATH)
    except FileNotFoundError:
        pass


def sentinel_info() -> Optional[dict]:
    """Read .impaired, return parsed dict or None."""
    try:
        with open(SENTINEL_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
