#!/usr/bin/env python3
"""Regression tests for the silent vec-index write failure (no such module: vec0).

Rows were written with embeddings but never inserted into the `memories_vec`
ANN index, leaving them keyword-findable but invisible to `--semantic`. The fix
makes `upsert_vec_index` self-load the sqlite-vec extension, adds a `--heal-vec`
backfill for the embedded-but-unindexed gap, and surfaces the gap in `--stats`.
See cairn/SPEC-vec-index-write-fix.md.
"""
import os
import sys
import tempfile
from io import StringIO

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3

import numpy as np
import pytest

import cairn.embeddings as emb
import cairn.query as query

DIM = 384
TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def make_vec(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def fresh_db():
    """Create a fresh cairn DB with the tables the vec-write paths touch,
    including the vec0 virtual table (requires the extension loaded)."""
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"test_vec_{_counter[0]}.db")
    conn = emb.connect_db(db_path)  # loads sqlite_vec
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB,
        session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        archived_reason TEXT, keywords TEXT, depth INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        origin_id TEXT, deleted_at TIMESTAMP, topic_embedding BLOB)""")
    conn.execute("""CREATE TABLE memory_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id INTEGER, content TEXT)""")
    conn.execute("""CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY, project TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE VIRTUAL TABLE memories_vec USING vec0(
        memory_id INTEGER PRIMARY KEY, embedding float[384])""")
    conn.commit()
    return db_path, conn


def vec_count(conn, mem_id):
    return conn.execute(
        "SELECT COUNT(*) FROM memories_vec WHERE memory_id = ?", (mem_id,)
    ).fetchone()[0]


def insert_embedded_row(conn, vec, content="hello world"):
    """Insert a memory WITH an embedding but WITHOUT a vec-index row (simulates
    the bug)."""
    blob = emb.to_blob(vec)
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
        ("fact", "topic", content, blob))
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def capture_stdout(func, *args, **kwargs):
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_upsert_loads_vec():
    """upsert_vec_index self-loads the extension: succeeds even on a bare
    connection that never called _load_vec."""
    db_path, setup_conn = fresh_db()
    setup_conn.close()

    bare = sqlite3.connect(db_path)  # deliberately NOT vec-loaded
    bare.execute("INSERT INTO memories (type, topic, content) VALUES ('fact','t','c')")
    mem_id = bare.execute("SELECT last_insert_rowid()").fetchone()[0]

    ok = emb.upsert_vec_index(bare, mem_id, emb.to_blob(make_vec(1)))
    assert ok is True
    assert vec_count(bare, mem_id) == 1
    bare.close()


def test_upsert_idempotent():
    """Upserting twice leaves exactly one vec row."""
    db_path, conn = fresh_db()
    mem_id = insert_embedded_row(conn, make_vec(2))
    blob = emb.to_blob(make_vec(2))
    assert emb.upsert_vec_index(conn, mem_id, blob) is True
    assert emb.upsert_vec_index(conn, mem_id, blob) is True
    assert vec_count(conn, mem_id) == 1
    conn.close()


def test_add_memory_is_semantically_retrievable(monkeypatch):
    """add_memory indexes the row: it lands in memories_vec AND find_similar
    returns it."""
    db_path, setup_conn = fresh_db()
    setup_conn.close()

    vec = make_vec(3)
    monkeypatch.setattr(query, "DB_PATH", db_path)
    monkeypatch.setattr(emb, "embed", lambda text, **kw: vec)

    capture_stdout(query.add_memory, "fact", "mytopic", "distinctive content")

    conn = emb.connect_db(db_path)
    new_id = conn.execute(
        "SELECT id FROM memories WHERE topic='mytopic'").fetchone()[0]
    assert vec_count(conn, new_id) == 1

    results = emb.find_similar(conn, "distinctive content", rerank=False)
    assert any(r["id"] == new_id for r in results)
    conn.close()


def test_add_memory_reports_index_state(monkeypatch):
    """The add_memory print distinguishes a successful index from a failure."""
    db_path, setup_conn = fresh_db()
    setup_conn.close()
    monkeypatch.setattr(query, "DB_PATH", db_path)
    monkeypatch.setattr(emb, "embed", lambda text, **kw: make_vec(4))
    out = capture_stdout(query.add_memory, "fact", "t", "c")
    assert "with embedding+index" in out


def test_heal_vec(monkeypatch):
    """heal_vec drives the embedded-but-unindexed gap to 0 and is idempotent."""
    db_path, conn = fresh_db()
    mem_id = insert_embedded_row(conn, make_vec(5))
    conn.close()
    monkeypatch.setattr(query, "DB_PATH", db_path)

    out = capture_stdout(query.heal_vec)
    assert "Healed 1" in out and "remaining gap 0" in out

    check = emb.connect_db(db_path)
    assert vec_count(check, mem_id) == 1
    check.close()

    # idempotent: second run finds nothing
    out2 = capture_stdout(query.heal_vec)
    assert "gap is 0" in out2


def test_stats_reports_gap(monkeypatch):
    """--stats surfaces a non-zero vec-index gap with the heal hint."""
    db_path, conn = fresh_db()
    insert_embedded_row(conn, make_vec(6))  # embedded, not indexed -> gap 1
    conn.close()
    monkeypatch.setattr(query, "DB_PATH", db_path)

    out = capture_stdout(query.stats)
    assert "gap 1" in out
    assert "--heal-vec" in out
