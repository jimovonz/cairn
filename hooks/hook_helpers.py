"""Shared helpers for Cairn hooks — DB access, logging, metrics, embedder."""

import json
import os
import sqlite3
import sys

CAIRN_DIR = os.path.join(os.path.dirname(__file__), "..", "cairn")
DB_PATH = os.path.join(CAIRN_DIR, "cairn.db")
LOG_PATH = os.path.join(CAIRN_DIR, "hook.log")

sys.path.insert(0, CAIRN_DIR)


def log(msg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def get_conn():
    """Create a SQLite connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def record_metric(session_id, event, detail=None, value=None):
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


def get_embedder():
    """Lazy-load the embeddings module."""
    try:
        import embeddings
        return embeddings
    except ImportError:
        return None


def get_session_project(conn, session_id):
    """Look up the project label for a session."""
    if not session_id:
        return None
    row = conn.execute("SELECT project FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return row[0] if row else None
