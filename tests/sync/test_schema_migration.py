"""Schema migration v3 → v4: idempotency + backfill correctness."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


def _v3_database(tmp_path: Path) -> Path:
    """Create a database at schema_version=3 with one legacy memory row."""
    db = tmp_path / "v3.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT, topic TEXT, content TEXT,
            embedding BLOB, session_id TEXT, project TEXT,
            confidence REAL DEFAULT 0.7,
            origin_id TEXT, user_id TEXT, updated_by TEXT,
            keywords TEXT, depth INTEGER, archived_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE memory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER, content TEXT, session_id TEXT,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE memory_annotation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER, direction TEXT, reason TEXT, session_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TIMESTAMP, description TEXT);
        INSERT INTO schema_version (version, description) VALUES (3, 'pre-migration');
        INSERT INTO memories (type, topic, content, origin_id) VALUES ('fact', 't', 'c', 'legacy-uuid');
        INSERT INTO memory_history (memory_id, content) VALUES (1, 'old content');
    """)
    conn.commit()
    conn.close()
    return db


def test_v4_migration_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_STATE_DIR", str(tmp_path / "state"))
    db = _v3_database(tmp_path)

    from cairn import init_db
    init_db.DB_PATH = str(db)

    init_db.init()  # first apply
    init_db.init()  # second apply — must be no-op

    conn = sqlite3.connect(str(db))
    versions = [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()]
    assert 4 in versions, "v4 not recorded"
    assert versions == sorted(set(versions)), "duplicate version inserts on re-apply"


def test_backfill_preserves_existing_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_STATE_DIR", str(tmp_path / "state"))
    db = _v3_database(tmp_path)

    from cairn import init_db
    init_db.DB_PATH = str(db)
    init_db.init()

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT origin_id, created_by_node, updated_by_node, lamport, visibility, embedding_model_version "
        "FROM memories WHERE origin_id = 'legacy-uuid'"
    ).fetchone()
    assert row is not None, "legacy row lost"
    origin, cb_node, ub_node, lam, vis, model = row
    assert origin == "legacy-uuid"
    assert cb_node and len(cb_node) > 8, "created_by_node not backfilled"
    assert ub_node == cb_node, "updated_by_node should match created on backfill"
    assert lam and lam > 0, "lamport must be backfilled to a positive value"
    assert vis == "team"
    assert model and "@" in model, f"model_version format unexpected: {model!r}"

    hist = conn.execute(
        "SELECT history_uuid, memory_origin, changed_by_node, lamport FROM memory_history WHERE memory_id = 1"
    ).fetchone()
    assert hist[0] is not None, "history_uuid not backfilled"
    assert hist[1] == "legacy-uuid", "memory_origin not anchored to memories.origin_id"


def test_v4_creates_required_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_STATE_DIR", str(tmp_path / "state"))
    db = _v3_database(tmp_path)
    from cairn import init_db
    init_db.DB_PATH = str(db)
    init_db.init()
    conn = sqlite3.connect(str(db))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    # sync_state + discovered_peers relocated to the ephemeral DB (schema v12);
    # they are no longer durable tables.
    assert {"node_state", "confidence_log", "sync_peers"}.issubset(tables)
    assert "sync_state" not in tables and "discovered_peers" not in tables


def test_node_id_persists_across_init(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_STATE_DIR", str(tmp_path / "state"))
    db = _v3_database(tmp_path)
    from cairn import init_db
    init_db.DB_PATH = str(db)
    init_db.init()
    nid_path = tmp_path / "state" / "node_id"
    assert nid_path.exists()
    nid_first = nid_path.read_text().strip()

    init_db.init()
    nid_second = nid_path.read_text().strip()
    assert nid_first == nid_second, "node_id must be stable across init() calls"
