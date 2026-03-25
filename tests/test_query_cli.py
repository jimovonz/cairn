#!/usr/bin/env python3
"""Tests for query.py CLI commands against a test database."""

import sys
import os
import sqlite3
import tempfile
import shutil
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"cli_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE memory_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER, content TEXT, session_id TEXT,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content); END""")
    conn.execute("""CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content) VALUES ('delete', old.id, old.topic, old.content); END""")
    conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content, session_id, changed_at)
        VALUES (old.id, old.content, old.session_id, old.updated_at); END""")
    conn.commit()
    return db_path, conn


def seed_db(conn, count=5):
    """Insert test memories."""
    types = ["fact", "decision", "preference", "correction", "project"]
    for i in range(count):
        conn.execute(
            "INSERT INTO memories (type, topic, content, project, confidence, session_id) VALUES (?, ?, ?, ?, ?, ?)",
            (types[i % len(types)], f"topic-{i}", f"content for item {i}", "TestProject", 0.5 + i * 0.1, "sess-test")
        )
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('sess-test', 'TestProject')")
    conn.execute("INSERT INTO metrics (event, session_id, value) VALUES ('hook_fired', 'sess-test', 1)")
    conn.commit()


def capture_output(func, *args, **kwargs):
    """Capture stdout from a function."""
    captured = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return captured.getvalue()


import query


# ============================================================
# --recent
# ============================================================

def test_list_recent():
    db_path, conn = fresh_db()
    seed_db(conn)
    with patch.object(query, 'DB_PATH', db_path):
        rows = query.list_recent(limit=3)
    assert len(rows) >= 1
    assert len(rows) <= 3
    # Verify rows have expected fields
    output = capture_output(query.format_rows, rows)
    assert "topic-" in output
    conn.close()


def test_list_recent_empty_db():
    db_path, conn = fresh_db()
    with patch.object(query, 'DB_PATH', db_path):
        output = capture_output(query.format_rows, query.list_recent())
    assert "No results" in output
    conn.close()


# ============================================================
# --type
# ============================================================

def test_list_by_type():
    db_path, conn = fresh_db()
    seed_db(conn, 10)
    with patch.object(query, 'DB_PATH', db_path):
        rows = query.list_by_type("fact")
    assert len(rows) >= 1
    assert all(r["type"] == "fact" for r in rows)
    conn.close()


# ============================================================
# Full-text search
# ============================================================

def test_fts_search():
    db_path, conn = fresh_db()
    seed_db(conn)
    with patch.object(query, 'DB_PATH', db_path):
        rows = query.search("item")
    assert len(rows) >= 1
    conn.close()


def test_fts_search_no_results():
    db_path, conn = fresh_db()
    seed_db(conn)
    with patch.object(query, 'DB_PATH', db_path):
        rows = query.search("zzzznonexistent")
    assert len(rows) == 0
    conn.close()


# ============================================================
# --stats
# ============================================================

def test_stats_output():
    db_path, conn = fresh_db()
    seed_db(conn)
    with patch.object(query, 'DB_PATH', db_path):
        output = capture_output(query.stats)
    assert "Total memories:" in output
    assert "5" in output  # We seeded 5
    conn.close()


# ============================================================
# --review
# ============================================================

def test_review_shows_low_confidence():
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'low', 'low conf', 0.2)")
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'high', 'high conf', 0.8)")
    conn.commit()
    with patch.object(query, 'DB_PATH', db_path):
        output = capture_output(query.review)
    assert "low" in output.lower() or "Suppressed" in output
    conn.close()


# ============================================================
# --delete
# ============================================================

def test_delete_memory():
    db_path, conn = fresh_db()
    seed_db(conn)
    count_before = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    with patch.object(query, 'DB_PATH', db_path):
        query.delete_memory(1)
    count_after = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count_after == count_before - 1
    conn.close()


def test_delete_nonexistent():
    db_path, conn = fresh_db()
    with patch.object(query, 'DB_PATH', db_path):
        output = capture_output(query.delete_memory, 999)
    assert "No memory" in output
    conn.close()


# ============================================================
# --history
# ============================================================

def test_show_history_with_versions():
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO memories (type, topic, content) VALUES ('decision', 'db', 'PostgreSQL')")
    conn.commit()
    conn.execute("UPDATE memories SET content = 'SQLite' WHERE id = 1")
    conn.commit()
    with patch.object(query, 'DB_PATH', db_path):
        output = capture_output(query.show_history, 1)
    assert "PostgreSQL" in output  # Old version in history
    assert "SQLite" in output  # Current
    conn.close()


# ============================================================
# --compact
# ============================================================

def test_compact_output():
    db_path, conn = fresh_db()
    seed_db(conn)
    with patch.object(query, 'DB_PATH', db_path):
        output = capture_output(query.compact, "TestProject")
    assert "TestProject" in output
    assert "memories" in output.lower()
    conn.close()


# ============================================================
# --projects
# ============================================================

def test_list_projects():
    db_path, conn = fresh_db()
    seed_db(conn)
    with patch.object(query, 'DB_PATH', db_path):
        output = capture_output(query.list_projects)
    assert "TestProject" in output
    conn.close()


def cleanup():
    shutil.rmtree(TEST_DIR, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
    cleanup()
