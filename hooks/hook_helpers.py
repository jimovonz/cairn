"""Shared helpers for Cairn hooks — DB access, logging, metrics, embedder."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from types import ModuleType
from typing import Optional

CAIRN_DIR = os.path.join(os.path.dirname(__file__), "..", "cairn")
DB_PATH = os.environ.get("CAIRN_DB_PATH", os.path.join(CAIRN_DIR, "cairn.db"))
LOG_PATH = os.path.join(CAIRN_DIR, "hook.log")

sys.path.insert(0, CAIRN_DIR)


def log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def get_conn() -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def record_metric(session_id: str, event: str, detail: Optional[str] = None,
                  value: Optional[float] = None) -> None:
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO metrics (event, session_id, detail, value) VALUES (?, ?, ?, ?)",
            (event, session_id, detail, value)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_embedder() -> Optional[ModuleType]:
    """Lazy-load the embeddings module."""
    if os.environ.get("CAIRN_SKIP_EMBEDDER"):
        return None
    try:
        import embeddings
        return embeddings
    except ImportError:
        return None


def get_session_project(conn: sqlite3.Connection, session_id: str) -> Optional[str]:
    """Look up the project label for a session."""
    if not session_id:
        return None
    row = conn.execute("SELECT project FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return row[0] if row else None
