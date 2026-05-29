"""get_ephemeral_conn self-heals the ephemeral DB.

Regression coverage for the recurring "no such table: hook_state" loss: the old
probe checked only `metrics`, so a DB missing `hook_state` (schema drift / partial
rebuild after a corruption reset) was never repaired. Also covers rebuild of a
genuinely corrupt image.
"""

import os
import sqlite3
import tempfile
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import config, init_db
import hooks.hook_helpers as hh


def _fresh(tmp):
    return os.path.join(tmp, "cairn-ephemeral.db")


def test_recreates_missing_hook_state_table():
    tmp = tempfile.mkdtemp()
    eph = _fresh(tmp)
    init_db.init_ephemeral(eph)
    # Simulate drift: metrics present, hook_state dropped — the exact state that
    # the metrics-only probe failed to repair.
    c = sqlite3.connect(eph)
    c.execute("DROP TABLE hook_state")
    c.commit()
    c.close()
    with patch.object(config, "EPHEMERAL_DB_PATH", eph):
        conn = hh.get_ephemeral_conn()
        conn.execute("SELECT 1 FROM hook_state LIMIT 0")  # must not raise
        conn.close()
        # round-trip through the public helpers
        hh.save_hook_state("sess", "graph_files_seen", "/a\n/b")
        assert hh.load_hook_state("sess", "graph_files_seen") == "/a\n/b"


def test_inits_when_db_file_absent():
    tmp = tempfile.mkdtemp()
    eph = _fresh(tmp)  # never created
    with patch.object(config, "EPHEMERAL_DB_PATH", eph):
        conn = hh.get_ephemeral_conn()
        for t in ("metrics", "hook_state", "pending_writes", "pair_assessments"):
            conn.execute(f"SELECT 1 FROM {t} LIMIT 0")
        conn.close()


def test_rebuilds_corrupt_image():
    tmp = tempfile.mkdtemp()
    eph = _fresh(tmp)
    # A non-SQLite file → "file is not a database" (a corruption-class error).
    with open(eph, "wb") as f:
        f.write(b"NOT-A-SQLITE-DB" * 200)
    with patch.object(config, "EPHEMERAL_DB_PATH", eph):
        conn = hh.get_ephemeral_conn()
        for t in ("metrics", "hook_state", "pending_writes", "pair_assessments"):
            conn.execute(f"SELECT 1 FROM {t} LIMIT 0")  # rebuilt, usable
        conn.close()


def test_healthy_db_untouched():
    """A complete DB with data is returned as-is (no needless rebuild)."""
    tmp = tempfile.mkdtemp()
    eph = _fresh(tmp)
    init_db.init_ephemeral(eph)
    c = sqlite3.connect(eph)
    c.execute("INSERT INTO hook_state (session_id, key, value) VALUES ('s', 'k', 'keep')")
    c.commit()
    c.close()
    with patch.object(config, "EPHEMERAL_DB_PATH", eph):
        conn = hh.get_ephemeral_conn()
        row = conn.execute("SELECT value FROM hook_state WHERE session_id='s' AND key='k'").fetchone()
        conn.close()
        assert row and row[0] == "keep"  # data survived (DB not rebuilt)
