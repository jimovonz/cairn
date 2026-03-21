"""Shared database connection helper. All Cairn code should use connect() from here
to ensure consistent WAL mode and busy timeout across concurrent sessions."""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")


def connect(db_path=None):
    """Open a SQLite connection with WAL mode and busy timeout for concurrent access."""
    from config import DB_BUSY_TIMEOUT_MS
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
