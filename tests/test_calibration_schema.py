"""Phase 1 calibration schema migration tests.

Verifies init_db.init() creates calibration_rows in the durable DB and
init_ephemeral() creates calibration_deliveries. Both should be idempotent
and survive being re-run against an existing DB.
"""

import os
import sys
import tempfile
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import init_db


def _table_columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _table_exists(conn, table):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def test_init_creates_calibration_rows():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "test.db")
        with patch.object(init_db, "DB_PATH", db):
            init_db.init()
        conn = sqlite3.connect(db)
        try:
            assert _table_exists(conn, "calibration_rows")
            cols = set(_table_columns(conn, "calibration_rows"))
            expected = {
                "id", "content", "kw", "qf", "source", "confidence",
                "pinned", "layer", "session_scope", "superseded_by",
                "archived_at", "archive_reason", "delivered_count",
                "followed_count", "ignored_count", "corrected_count",
                "created_at", "updated_at", "embedding",
            }
            missing = expected - cols
            assert not missing, f"missing columns: {missing}"
        finally:
            conn.close()


def test_init_indexes_present():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "test.db")
        with patch.object(init_db, "DB_PATH", db):
            init_db.init()
        conn = sqlite3.connect(db)
        try:
            names = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            for idx in ("idx_cal_rows_source", "idx_cal_rows_layer",
                        "idx_cal_rows_archived"):
                assert idx in names, f"missing index {idx}"
        finally:
            conn.close()


def test_init_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "test.db")
        with patch.object(init_db, "DB_PATH", db):
            init_db.init()
            init_db.init()  # second run must not raise
        conn = sqlite3.connect(db)
        try:
            assert _table_exists(conn, "calibration_rows")
        finally:
            conn.close()


def test_schema_version_5_recorded():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "test.db")
        with patch.object(init_db, "DB_PATH", db):
            init_db.init()
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT description FROM schema_version WHERE version = 5"
            ).fetchone()
            assert row is not None
            assert "calibration" in row[0].lower()
        finally:
            conn.close()


def test_init_ephemeral_creates_calibration_deliveries():
    with tempfile.TemporaryDirectory() as td:
        eph = os.path.join(td, "eph.db")
        init_db.init_ephemeral(eph)
        conn = sqlite3.connect(eph)
        try:
            assert _table_exists(conn, "calibration_deliveries")
            cols = set(_table_columns(conn, "calibration_deliveries"))
            expected = {
                "id", "session_id", "turn_index", "row_id", "delivered_at",
                "similarity", "outcome", "outcome_evidence",
            }
            missing = expected - cols
            assert not missing, f"missing columns: {missing}"
        finally:
            conn.close()


def test_init_ephemeral_indexes_present():
    with tempfile.TemporaryDirectory() as td:
        eph = os.path.join(td, "eph.db")
        init_db.init_ephemeral(eph)
        conn = sqlite3.connect(eph)
        try:
            names = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            for idx in ("idx_cal_deliv_session", "idx_cal_deliv_row",
                        "idx_cal_deliv_outcome"):
                assert idx in names, f"missing index {idx}"
        finally:
            conn.close()


def test_init_ephemeral_idempotent():
    with tempfile.TemporaryDirectory() as td:
        eph = os.path.join(td, "eph.db")
        init_db.init_ephemeral(eph)
        init_db.init_ephemeral(eph)
        conn = sqlite3.connect(eph)
        try:
            assert _table_exists(conn, "calibration_deliveries")
        finally:
            conn.close()


def test_calibration_row_insert_round_trip():
    """A row written with the documented schema must survive a round-trip."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "test.db")
        with patch.object(init_db, "DB_PATH", db):
            init_db.init()
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO calibration_rows (content, source, confidence, "
                "kw, qf, layer) VALUES (?, ?, ?, ?, ?, ?)",
                ("prefers terse answers", "explicit", 0.9,
                 "terse,brevity", '["how should I respond","keep it short"]',
                 "general"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT content, source, confidence, layer, pinned, "
                "delivered_count, followed_count "
                "FROM calibration_rows WHERE id = 1"
            ).fetchone()
            assert row[0] == "prefers terse answers"
            assert row[1] == "explicit"
            assert row[2] == 0.9
            assert row[3] == "general"
            assert row[4] == 0  # pinned default
            assert row[5] == 0  # delivered_count default
            assert row[6] == 0
        finally:
            conn.close()
