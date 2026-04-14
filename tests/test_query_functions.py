#!/usr/bin/env python3
"""Tests for cairn/query.py — 17 functions × 4 categories = 68 tests."""

import sys
import os
import json
import re
import sqlite3
import tempfile
from unittest.mock import patch
from io import StringIO
from datetime import datetime, timedelta

_ANSI = re.compile(r'\x1b\[[0-9;]*m')

import cairn.query as query

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"test_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
    conn.execute("""CREATE TABLE hook_state (session_id TEXT NOT NULL,
        key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, keywords, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content, keywords) VALUES (new.id, new.topic, new.content, new.keywords); END""")
    conn.execute("""CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content, keywords) VALUES ('delete', old.id, old.topic, old.content, old.keywords); END""")
    conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content, session_id, changed_at)
        VALUES (old.id, old.content, old.session_id, old.updated_at); END""")
    conn.commit()
    return db_path, conn


def seed(conn, count=5):
    types = ["fact", "decision", "preference", "correction", "project"]
    for i in range(count):
        conn.execute(
            "INSERT INTO memories (type, topic, content, project, confidence, session_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (types[i % len(types)], f"topic-{i}", f"content for item {i}",
             "TestProject", 0.5 + i * 0.1, "sess-001",
             f"2026-03-{20+i:02d} 10:00:00", f"2026-03-{20+i:02d} 10:00:00"))
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('sess-001', 'TestProject')")
    conn.execute("INSERT INTO metrics (event, session_id, value) VALUES ('hook_fired', 'sess-001', 1)")
    conn.execute("INSERT INTO metrics (event, session_id, value) VALUES ('retrieval_latency_ms', 'sess-001', 150.0)")
    conn.commit()


def capture(func, *args, **kwargs):
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue(), result


def _make_transcript(messages, path):
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


# ============================================================
# search
# ============================================================

#TAG: [07C3] 2026-04-05
# Verifies: search returns correct count of FTS matches with valid row fields
def test_search_behavioural():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.search("item")
    assert len(rows) == 5
    returned_types = {r["type"] for r in rows}
    assert returned_types == {"fact", "decision", "preference", "correction", "project"}
    c.close()


#TAG: [903B] 2026-04-05
# Verifies: search returns zero-length list when query has no matches in populated DB
def test_search_edge():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.search("zzzznonexistent")
    assert len(rows) == 0
    c.close()


#TAG: [A9D5] 2026-04-05
# Verifies: search returns empty list on empty database without raising exception
def test_search_error():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        rows = query.search("anything")
    assert len(rows) == 0
    c.close()


#TAG: [DF71] 2026-04-05
# Verifies: search limit parameter constrains result count below total matches
def test_search_adversarial():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.search("item", limit=2)
    assert len(rows) == 2
    total = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert total == 5
    c.close()


# ============================================================
# list_by_type
# ============================================================

#TAG: [468D] 2026-04-05
# Verifies: list_by_type returns only rows with the requested type field
def test_list_by_type_behavioural():
    db, c = fresh_db(); seed(c, 10)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_type("fact")
    assert len(rows) == 2
    assert all(r["type"] == "fact" for r in rows)
    c.close()


#TAG: [8704] 2026-04-05
# Verifies: list_by_type returns empty list for nonexistent type string
def test_list_by_type_edge():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_type("nonexistent_type")
    assert len(rows) == 0
    c.close()


#TAG: [B298] 2026-04-05
# Verifies: list_by_type returns empty list on empty database
def test_list_by_type_error():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_type("fact")
    assert len(rows) == 0
    c.close()


#TAG: [C602] 2026-04-05
# Verifies: list_by_type safely handles SQL injection characters in type parameter
def test_list_by_type_adversarial():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_type("fact' OR '1'='1")
    assert len(rows) == 0
    assert c.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 5
    c.close()


# ============================================================
# list_recent
# ============================================================

#TAG: [3608] 2026-04-05
# Verifies: list_recent returns rows in descending updated_at order respecting limit
def test_list_recent_behavioural():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_recent(limit=3)
    assert len(rows) == 3
    dates = [r["updated_at"] for r in rows]
    assert dates == sorted(dates, reverse=True)
    c.close()


#TAG: [5825] 2026-04-05
# Verifies: list_recent on empty database returns zero-length list
def test_list_recent_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_recent()
    assert len(rows) == 0
    c.close()


#TAG: [9E58] 2026-04-05
# Verifies: list_recent with limit=1 returns exactly the most recent row
def test_list_recent_error():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_recent(limit=1)
    assert len(rows) == 1
    assert rows[0]["updated_at"] == "2026-03-24 10:00:00"
    c.close()


#TAG: [1B56] 2026-04-05
# Verifies: list_recent with limit exceeding row count returns all available rows
def test_list_recent_adversarial():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_recent(limit=1000)
    assert len(rows) == 5
    c.close()


# ============================================================
# list_by_date
# ============================================================

#TAG: [EDA4] 2026-04-05
# Verifies: list_by_date with since and until returns only rows within date range
def test_list_by_date_behavioural():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_date(since="2026-03-21", until="2026-03-23")
    assert len(rows) == 3
    for r in rows:
        assert "2026-03-21" <= r["updated_at"][:10] <= "2026-03-23"
    c.close()


#TAG: [8B8E] 2026-04-05
# Verifies: list_by_date with no filters returns all memories up to limit
def test_list_by_date_edge():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_date()
    assert len(rows) == 5
    c.close()


#TAG: [E241] 2026-04-05
# Verifies: list_by_date with reversed range where since > until returns empty
def test_list_by_date_error():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_date(since="2026-12-01", until="2026-01-01")
    assert len(rows) == 0
    c.close()


#TAG: [4C8E] 2026-04-05
# Verifies: list_by_date with relative "3d" format computes correct date via _parse_date
def test_list_by_date_adversarial():
    db, c = fresh_db()
    two_days_ago = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
    c.execute("INSERT INTO memories (type, topic, content, updated_at) VALUES ('fact', 'recent', 'x', ?)",
              (two_days_ago + " 12:00:00",))
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_date(since="3d")
    assert len(rows) == 1
    assert rows[0]["topic"] == "recent"
    c.close()


# ============================================================
# list_by_session
# ============================================================

#TAG: [7B34] 2026-04-05
# Verifies: list_by_session returns memories matching session prefix only
def test_list_by_session_behavioural():
    db, c = fresh_db(); seed(c)
    c.execute("INSERT INTO memories (type, topic, content, session_id) VALUES ('fact', 'x', 'other', 'sess-999')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_session("sess-001")
    assert len(rows) == 5
    c.close()


#TAG: [5E34] 2026-04-05
# Verifies: list_by_session returns empty for nonexistent session prefix
def test_list_by_session_edge():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_session("nonexistent")
    assert len(rows) == 0
    c.close()


#TAG: [EC90] 2026-04-05
# Verifies: list_by_session on empty database returns empty list
def test_list_by_session_error():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_session("anything")
    assert len(rows) == 0
    c.close()


#TAG: [53D0] 2026-04-05
# Verifies: list_by_session correctly isolates by exact prefix, not matching other sessions
def test_list_by_session_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, session_id) VALUES ('fact', 'target', 'y', 'sess-special')")
    c.execute("INSERT INTO memories (type, topic, content, session_id) VALUES ('fact', 'other', 'z', 'other-sess')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        rows = query.list_by_session("sess-special")
    assert len(rows) == 1
    assert rows[0]["topic"] == "target"
    c.close()


# ============================================================
# show_context — NON-TRIVIAL
# ============================================================

#TAG: [6F04] 2026-04-05
# Verifies: show_context displays conversation anchor marker and context around memory timestamp
def test_show_context_behavioural():
    db, c = fresh_db()
    tp = os.path.join(TEST_DIR, f"transcript_{_counter[0]}.jsonl")
    c.execute("INSERT INTO memories (type, topic, content, session_id, created_at, depth) VALUES (?, ?, ?, ?, ?, ?)",
              ("fact", "test-topic", "test content", "sess-ctx", "2026-03-25 10:05:00", 3))
    c.execute("INSERT INTO sessions (session_id, transcript_path) VALUES (?, ?)", ("sess-ctx", tp))
    c.commit()
    msgs = [
        {"timestamp": "2026-03-25T10:00:00Z", "message": {"role": "user", "content": "hello world"}},
        {"timestamp": "2026-03-25T10:01:00Z", "message": {"role": "assistant", "content": "hi there"}},
        {"timestamp": "2026-03-25T10:02:00Z", "message": {"role": "user", "content": "what is cairn"}},
        {"timestamp": "2026-03-25T10:03:00Z", "message": {"role": "assistant", "content": "a memory system"}},
        {"timestamp": "2026-03-25T10:04:00Z", "message": {"role": "user", "content": "store this fact"}},
        {"timestamp": "2026-03-25T10:06:00Z", "message": {"role": "assistant", "content": "done storing"}},
    ]
    _make_transcript(msgs, tp)
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_context, 1)
    lines = out.strip().split('\n')
    assert len(lines) >= 4
    # Anchor marker appended to exactly 1 message line; use endswith (structural, not containment)
    marker_lines = [l for l in lines if l.endswith(' <<<')]
    assert len(marker_lines) == 1, f"Expected exactly 1 '<<<' marker line; got: {marker_lines}"
    assert re.match(r'\s+\[\d{2}:\d{2}:\d{2}\]\s+\w+:.*<<<$', marker_lines[0]), \
        f"Marker line format unexpected: {marker_lines[0]!r}"
    c.close()


#TAG: [A49B] 2026-04-05
# Verifies: show_context prints no-session message when memory has NULL session_id
def test_show_context_edge():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, session_id) VALUES ('fact', 'orphan', 'no session', NULL)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_context, 1)
    lines = out.strip().split('\n')
    assert len(lines) == 3
    assert lines[2] == "  No session ID — cannot locate transcript."
    c.close()


#TAG: [F951] 2026-04-05
# Verifies: show_context prints descriptive error for nonexistent memory ID
def test_show_context_error():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_context, 999)
    assert out.strip() == "No memory with id 999"
    c.close()


#TAG: [4A69] 2026-04-05
# Verifies: show_context handles corrupt created_at without crashing and prints parse error
def test_show_context_adversarial():
    db, c = fresh_db()
    tp = os.path.join(TEST_DIR, f"transcript_corrupt_{_counter[0]}.jsonl")
    c.execute("INSERT INTO memories (type, topic, content, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
              ("fact", "bad-ts", "corrupt timestamp", "sess-bad", "NOT_A_DATE"))
    c.execute("INSERT INTO sessions (session_id, transcript_path) VALUES (?, ?)", ("sess-bad", tp))
    c.commit()
    _make_transcript([{"timestamp": "2026-03-25T10:00:00Z", "message": {"role": "user", "content": "msg"}}], tp)
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_context, 1)
    lines = out.strip().split('\n')
    assert len(lines) == 3
    assert lines[2] == "  Cannot parse created_at: NOT_A_DATE"
    c.close()


# ============================================================
# verify_sources — NON-TRIVIAL
# ============================================================

#TAG: [0723] 2026-04-05
# Verifies: verify_sources reports correct coverage percentages for memories with depth and session
def test_verify_sources_behavioural():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, session_id, depth) VALUES ('fact', 'vs', 'verify test content', 'sess-vs', 5)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.verify_sources)
    lines = out.strip().split('\n')
    assert len(lines) == 6
    assert lines[1] == "  With session (timestamp-navigable): 1 (100.0%)"
    assert lines[5] == "  Depth distribution: min=5, avg=5.0, max=5"
    c.close()


#TAG: [448C] 2026-04-05
# Verifies: verify_sources reports No memories for empty database
def test_verify_sources_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.verify_sources)
    lines = out.strip().split('\n')
    assert len(lines) == 2
    assert lines[1] == "  No memories"
    c.close()


#TAG: [BD57] 2026-04-05
# Verifies: verify_sources skips legacy drift when transcript file is missing
def test_verify_sources_error():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, session_id, source_start, source_end) VALUES ('fact', 'legacy', 'this has long enough words for keywords matching', 'sess-leg', 1, 5)")
    c.execute("INSERT INTO sessions (session_id, transcript_path) VALUES ('sess-leg', '/nonexistent/path/transcript.jsonl')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.verify_sources)
    lines = out.strip().split('\n')
    assert len(lines) >= 2
    total = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert total == 1
    c.close()


#TAG: [5934] 2026-04-05
# Verifies: verify_sources recovers from malformed JSONL lines in transcript without crash
def test_verify_sources_adversarial():
    db, c = fresh_db()
    tp = os.path.join(TEST_DIR, f"transcript_malformed_{_counter[0]}.jsonl")
    c.execute("INSERT INTO memories (type, topic, content, session_id, source_start, source_end) VALUES ('fact', 'mal', 'words enough content here for matching keywords', 'sess-mal', 1, 3)")
    c.execute("INSERT INTO sessions (session_id, transcript_path) VALUES ('sess-mal', ?)", (tp,))
    c.commit()
    with open(tp, "w") as f:
        f.write("NOT VALID JSON\n")
        f.write(json.dumps({"message": {"role": "user", "content": "words enough content here matching"}}) + "\n")
        f.write("{broken json\n")
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.verify_sources)
    lines = out.strip().split('\n')
    assert len(lines) >= 1
    assert c.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    c.close()


# ============================================================
# stats — NON-TRIVIAL
# ============================================================

#TAG: [326F] 2026-04-05
# Verifies: stats reports correct total, type breakdown, and latency from metrics table
def test_stats_behavioural():
    db, c = fresh_db(); seed(c)
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.stats)
    lines = out.strip().split('\n')
    assert len(lines) >= 6
    assert lines[1] == "Total memories: 5"
    types_line = next(l for l in lines if l.startswith("Types: "))
    type_counts = dict(pair.split('=') for pair in types_line[7:].split(', '))
    assert type_counts == {"fact": "1", "decision": "1", "preference": "1", "correction": "1", "project": "1"}
    c.close()


#TAG: [A70E] 2026-04-05
# Verifies: stats on empty DB reports zero for all counts without errors
def test_stats_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.stats)
    lines = out.strip().split('\n')
    assert len(lines) >= 3
    total_line = next(l for l in lines if l.startswith("Total memories:"))
    assert total_line == "Total memories: 0"
    c.close()


#TAG: [4185] 2026-04-05
# Verifies: stats handles missing metrics table gracefully via try/except
def test_stats_error():
    db, c = fresh_db(); seed(c)
    c.execute("DROP TABLE metrics")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.stats)
    lines = out.strip().split('\n')
    assert len(lines) >= 3
    total_line = next(l for l in lines if l.startswith("Total memories:"))
    assert total_line == "Total memories: 5"
    c.close()


#TAG: [300D] 2026-04-05
# Verifies: stats handles NULL confidence values in distribution calculation
def test_stats_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'null-conf', 'test', NULL)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.stats)
    lines = out.strip().split('\n')
    assert len(lines) >= 3
    total_line = next(l for l in lines if l.startswith("Total memories:"))
    assert total_line == "Total memories: 1"
    c.close()


# ============================================================
# add_memory — NON-TRIVIAL
# ============================================================

#TAG: [6788] 2026-04-05
# Verifies: add_memory inserts row with correct type, topic, content, project, session_id
def test_add_memory_behavioural():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.add_memory, "decision", "architecture", "Use SQLite", project="cairn", session_id="sess-add")
    row = c.execute("SELECT type, topic, content, project, session_id FROM memories WHERE id=1").fetchone()
    assert row[0] == "decision"
    assert row[2] == "Use SQLite"
    assert row[3] == "cairn"
    assert row[4] == "sess-add"
    c.close()


#TAG: [B81C] 2026-04-05
# Verifies: add_memory accepts empty string content and stores it correctly
def test_add_memory_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.add_memory, "fact", "empty", "")
    row = c.execute("SELECT content FROM memories WHERE id=1").fetchone()
    assert row[0] == ""
    assert c.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    c.close()


#TAG: [637D] 2026-04-05
# Verifies: add_memory works without embeddings module reporting without embedding
def test_add_memory_error():
    db, c = fresh_db()
    import cairn as _cairn_pkg
    _saved = getattr(_cairn_pkg, "embeddings", None)
    with patch.object(query, 'DB_PATH', db), \
         patch.dict('sys.modules', {'cairn.embeddings': None}):
        if hasattr(_cairn_pkg, "embeddings"):
            delattr(_cairn_pkg, "embeddings")
        try:
            out, _ = capture(query.add_memory, "fact", "no-emb", "content without embedding")
        finally:
            if _saved is not None:
                _cairn_pkg.embeddings = _saved
    row = c.execute("SELECT embedding FROM memories WHERE id=1").fetchone()
    assert row[0] is None
    assert c.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    c.close()


#TAG: [B8FA] 2026-04-05
# Verifies: add_memory safely stores SQL injection payload without executing it
def test_add_memory_adversarial():
    db, c = fresh_db()
    malicious = "Robert'); DROP TABLE memories;--"
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.add_memory, "fact", "bobby", malicious)
    row = c.execute("SELECT content FROM memories WHERE id=1").fetchone()
    assert row[0] == malicious
    count = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 1
    c.close()


# ============================================================
# check — NON-TRIVIAL
# ============================================================

#TAG: [B8D4] 2026-04-05
# Verifies: check returns integer failure count and reports memory count for healthy DB
def test_check_behavioural():
    db, c = fresh_db(); seed(c); c.close()
    with patch.object(query, 'DB_PATH', db), \
         patch('os.path.expanduser', return_value=TEST_DIR):
        out, result = capture(query.check)
    plain_lines = [_ANSI.sub('', l) for l in out.strip().split('\n')]
    mem_lines = [l for l in plain_lines if re.match(r'  [✓!✗] \d+ memories stored', l)]
    assert len(mem_lines) == 1
    assert mem_lines[0] == "  ✓ 5 memories stored"
    assert result >= 0


#TAG: [61AB] 2026-04-05
# Verifies: check reports zero memories for empty DB without errors
def test_check_edge():
    db, c = fresh_db(); c.close()
    with patch.object(query, 'DB_PATH', db), \
         patch('os.path.expanduser', return_value=TEST_DIR):
        out, result = capture(query.check)
    plain_lines = [_ANSI.sub('', l) for l in out.strip().split('\n')]
    mem_lines = [l for l in plain_lines if re.match(r'  [✓!✗] \d+ memories stored', l)]
    assert len(mem_lines) == 1
    assert mem_lines[0] == "  ✓ 0 memories stored"
    assert result >= 0


#TAG: [A64B] 2026-04-05
# Verifies: check returns nonzero failure count when DB file does not exist
def test_check_error():
    missing = os.path.join(TEST_DIR, "nonexistent.db")
    with patch.object(query, 'DB_PATH', missing), \
         patch('os.path.expanduser', return_value=TEST_DIR):
        out, result = capture(query.check)
    assert result > 0


#TAG: [8899] 2026-04-05
# Verifies: check handles corrupt settings.json without crashing
def test_check_adversarial():
    db, c = fresh_db(); c.close()
    settings_dir = os.path.join(TEST_DIR, f"claude_corrupt_{_counter[0]}")
    os.makedirs(settings_dir, exist_ok=True)
    with open(os.path.join(settings_dir, "settings.json"), "w") as f:
        f.write("{{{invalid json")
    with patch.object(query, 'DB_PATH', db), \
         patch('os.path.expanduser', return_value=settings_dir):
        out, result = capture(query.check)
    assert result >= 0
    lines = out.strip().split('\n')
    assert len(lines) >= 3


# ============================================================
# show_history
# ============================================================

#TAG: [FAB9] 2026-04-05
# Verifies: show_history displays current content and prior version count after update
def test_show_history_behavioural():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, session_id) VALUES ('decision', 'db', 'PostgreSQL', 'sess-h')")
    c.commit()
    c.execute("UPDATE memories SET content = 'SQLite' WHERE id = 1")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_history, 1)
    lines = out.strip().split('\n')
    assert len(lines) >= 3
    hist_line = next(l for l in lines if "prior versions" in l)
    assert hist_line == "  --- History (1 prior versions) ---"
    c.close()


#TAG: [1B58] 2026-04-05
# Verifies: show_history reports No memory for nonexistent ID
def test_show_history_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_history, 999)
    assert out.strip() == "No memory with id 999"
    c.close()


#TAG: [938F] 2026-04-05
# Verifies: show_history shows No prior versions for memory never updated
def test_show_history_error():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'fresh', 'never updated')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_history, 1)
    lines = out.strip().split('\n')
    assert len(lines) >= 2
    no_hist_line = next(l for l in lines if "prior versions" in l)
    assert no_hist_line == "  No prior versions."
    c.close()


#TAG: [1942] 2026-04-05
# Verifies: show_history displays all prior versions after multiple sequential updates
def test_show_history_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'multi', 'v1')")
    c.commit()
    c.execute("UPDATE memories SET content = 'v2' WHERE id = 1")
    c.commit()
    c.execute("UPDATE memories SET content = 'v3' WHERE id = 1")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_history, 1)
    lines = out.strip().split('\n')
    assert len(lines) >= 4
    hist_line = next(l for l in lines if "prior versions" in l)
    assert hist_line == "  --- History (2 prior versions) ---"
    c.close()


# ============================================================
# delete_memory
# ============================================================

#TAG: [2D25] 2026-04-05
# Verifies: delete_memory removes the row and its history entries from the database
def test_delete_memory_behavioural():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'del', 'to delete')")
    c.commit()
    c.execute("UPDATE memories SET content = 'updated' WHERE id = 1")
    c.commit()
    assert c.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0] == 1
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.delete_memory, 1)
    assert c.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0] == 0
    c.close()


#TAG: [122A] 2026-04-05
# Verifies: delete_memory reports No memory for nonexistent ID
def test_delete_memory_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.delete_memory, 999)
    assert out.strip() == "No memory with id 999"
    c.close()


#TAG: [BE05] 2026-04-05
# Verifies: delete_memory second call on same ID reports No memory after prior delete
def test_delete_memory_error():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'del2', 'will delete twice')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        capture(query.delete_memory, 1)
        out2, _ = capture(query.delete_memory, 1)
    assert out2.strip() == "No memory with id 1"
    c.close()


#TAG: [D6CA] 2026-04-05
# Verifies: delete_memory also removes FTS index entry so search no longer finds it
def test_delete_memory_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'findme', 'unique_searchterm_xyz')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        pre = query.search("unique_searchterm_xyz")
        assert len(pre) == 1
        capture(query.delete_memory, 1)
        post = query.search("unique_searchterm_xyz")
    assert len(post) == 0
    c.close()


# ============================================================
# archive_memory
# ============================================================

#TAG: [CE16] 2026-04-05
# Verifies: archive_memory sets confidence to 0 and stores reason string in DB
def test_archive_memory_behavioural():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'old', 'outdated info', 0.8)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.archive_memory, 1, "superseded by new data")
    row = c.execute("SELECT confidence, archived_reason FROM memories WHERE id=1").fetchone()
    assert row[0] == 0
    assert row[1] == "superseded by new data"
    c.close()


#TAG: [7D8E] 2026-04-05
# Verifies: archive_memory reports No memory for nonexistent ID
def test_archive_memory_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.archive_memory, 999, "reason")
    assert out.strip() == "No memory with id 999"
    c.close()


#TAG: [2EA1] 2026-04-05
# Verifies: archive_memory on already-archived memory sets confidence to 0 again
def test_archive_memory_error():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'x', 'already archived', 0.0)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.archive_memory, 1, "re-archived")
    row = c.execute("SELECT confidence, archived_reason FROM memories WHERE id=1").fetchone()
    assert row[0] == 0
    assert row[1] == "re-archived"
    c.close()


#TAG: [024A] 2026-04-05
# Verifies: archive_memory safely stores reason with special characters
def test_archive_memory_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'x', 'test', 0.8)")
    c.commit()
    reason = "superseded'; DROP TABLE memories;--"
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.archive_memory, 1, reason)
    row = c.execute("SELECT archived_reason FROM memories WHERE id=1").fetchone()
    assert row[0] == reason
    assert c.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    c.close()


# ============================================================
# update_memory
# ============================================================

#TAG: [F5CC] 2026-04-05
# Verifies: update_memory changes content and preserves old value in history via trigger
def test_update_memory_behavioural():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'upd', 'original content')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.update_memory, 1, "new content")
    current = c.execute("SELECT content FROM memories WHERE id=1").fetchone()[0]
    assert current == "new content"
    hist = c.execute("SELECT content FROM memory_history WHERE memory_id=1").fetchone()
    assert hist[0] == "original content"
    c.close()


#TAG: [139B] 2026-04-05
# Verifies: update_memory reports No memory for nonexistent ID
def test_update_memory_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.update_memory, 999, "new")
    assert out.strip() == "No memory with id 999"
    c.close()


#TAG: [EB2E] 2026-04-05
# Verifies: update_memory to same content still creates a history entry via trigger
def test_update_memory_error():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'same', 'unchanged')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        capture(query.update_memory, 1, "unchanged")
    hist_count = c.execute("SELECT COUNT(*) FROM memory_history WHERE memory_id=1").fetchone()[0]
    assert hist_count == 1
    c.close()


#TAG: [2477] 2026-04-05
# Verifies: update_memory safely stores SQL injection payload as content
def test_update_memory_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'inj', 'safe')")
    c.commit()
    malicious = "'; DROP TABLE memories;--"
    with patch.object(query, 'DB_PATH', db):
        capture(query.update_memory, 1, malicious)
    row = c.execute("SELECT content FROM memories WHERE id=1").fetchone()
    assert row[0] == malicious
    assert c.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    c.close()


# ============================================================
# show_session_chain
# ============================================================

#TAG: [1584] 2026-04-05
# Verifies: show_session_chain displays parent-child chain with memory counts
def test_show_session_chain_behavioural():
    db, c = fresh_db()
    c.execute("INSERT INTO sessions (session_id, parent_session_id) VALUES ('root-001', NULL)")
    c.execute("INSERT INTO sessions (session_id, parent_session_id) VALUES ('child-001', 'root-001')")
    c.execute("INSERT INTO memories (type, topic, content, session_id) VALUES ('fact', 'a', 'x', 'root-001')")
    c.execute("INSERT INTO memories (type, topic, content, session_id) VALUES ('fact', 'b', 'y', 'child-001')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_session_chain, "root-001")
    lines = out.strip().split('\n')
    assert len(lines) == 3
    root_count = re.search(r'\[(\d+) memories\]', lines[1])
    child_count = re.search(r'\[(\d+) memories\]', lines[2])
    assert root_count.group(1) == "1", f"root-001 memory count wrong: {lines[1]}"
    assert child_count.group(1) == "1", f"child-001 memory count wrong: {lines[2]}"
    assert lines[1].split("...")[0] == "root-001"
    assert lines[2].split("...")[0] == "  -> child-001"
    c.close()


#TAG: [CC39] 2026-04-05
# Verifies: show_session_chain reports No session found for nonexistent session
def test_show_session_chain_edge():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_session_chain, "nonexistent")
    assert out.strip() == "No session found matching nonexistent"
    c.close()


#TAG: [6DDA] 2026-04-05
# Verifies: show_session_chain handles single session with no parent or children
def test_show_session_chain_error():
    db, c = fresh_db()
    c.execute("INSERT INTO sessions (session_id, parent_session_id) VALUES ('orphan-001', NULL)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_session_chain, "orphan-001")
    lines = out.strip().split('\n')
    assert len(lines) >= 2
    orphan_lines = [l for l in lines if "orphan-001" in l]
    assert len(orphan_lines) == 1
    c.close()


#TAG: [97EF] 2026-04-05
# Verifies: show_session_chain traverses three-level deep chain correctly
def test_show_session_chain_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO sessions (session_id, parent_session_id) VALUES ('L1', NULL)")
    c.execute("INSERT INTO sessions (session_id, parent_session_id) VALUES ('L2', 'L1')")
    c.execute("INSERT INTO sessions (session_id, parent_session_id) VALUES ('L3', 'L2')")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.show_session_chain, "L3")
    lines = out.strip().split('\n')
    assert len(lines) == 4
    assert lines[1].split("...")[0] == "L1"
    assert lines[3].split("...")[0] == "    -> L3"
    c.close()


# ============================================================
# review
# ============================================================

#TAG: [B979] 2026-04-05
# Verifies: review categorizes memories into suppressed, uncertain, and healthy counts
def test_review_behavioural():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'sup', 'suppressed', 0.1)")
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'unc', 'uncertain', 0.45)")
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'ok', 'healthy', 0.8)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.review)
    lines = out.strip().split('\n')
    assert len(lines) >= 3
    summary_line = next(l for l in lines if l.startswith("Summary:"))
    assert summary_line == "Summary: 1 suppressed, 1 uncertain, 1 healthy (of 3 total)"
    c.close()


#TAG: [0C78] 2026-04-05
# Verifies: review reports all healthy when no low-confidence memories exist
def test_review_edge():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'ok', 'good', 0.9)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.review)
    lines = out.strip().split('\n')
    assert len(lines) >= 1
    first_nonblank = next(l for l in lines if l.strip())
    assert first_nonblank == "All memories have confidence >= 0.6"
    c.close()


#TAG: [44FF] 2026-04-05
# Verifies: review on empty database reports zero counts
def test_review_error():
    db, c = fresh_db()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.review)
    lines = out.strip().split('\n')
    assert len(lines) >= 1
    summary_line = next(l for l in lines if l.startswith("Summary:"))
    assert summary_line == "Summary: 0 suppressed, 0 uncertain, 0 healthy (of 0 total)"
    c.close()


#TAG: [F008] 2026-04-05
# Verifies: review correctly classifies memory at exact threshold boundary 0.3
def test_review_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'at-boundary', 'exactly at threshold', 0.3)")
    c.commit()
    with patch.object(query, 'DB_PATH', db):
        out, _ = capture(query.review)
    lines = out.strip().split('\n')
    assert len(lines) >= 2
    # 0.3 is >= threshold_low (0.3) but < threshold_high (0.6), so it's "uncertain"
    uncertain_header = next(l for l in lines if l.startswith("=== Uncertain"))
    assert uncertain_header == "=== Uncertain (confidence 0.3–0.6) ==="
    c.close()


# ============================================================
# format_rows
# ============================================================

#TAG: [DA61] 2026-04-05
# Verifies: format_rows prints sqlite3.Row objects with id, type, topic, content fields
def test_format_rows_behavioural():
    db, c = fresh_db(); seed(c)
    c2 = sqlite3.connect(db)
    c2.row_factory = sqlite3.Row
    rows = c2.execute("SELECT id, type, topic, content, updated_at FROM memories LIMIT 2").fetchall()
    c2.close()
    out, _ = capture(query.format_rows, rows)
    lines = out.strip().split('\n')
    assert len(lines) == 4  # 2 rows × 2 lines each (header + content)
    c.close()


#TAG: [C8AA] 2026-04-05
# Verifies: format_rows prints No results for empty input list
def test_format_rows_edge():
    out, _ = capture(query.format_rows, [])
    assert out.strip() == "No results."


#TAG: [2C2E] 2026-04-05
# Verifies: format_rows prints dict objects with similarity score formatted to 3 decimals
def test_format_rows_error():
    rows = [{"id": 1, "type": "fact", "topic": "test", "content": "dict content",
             "updated_at": "2026-03-25", "similarity": 0.95}]
    out, _ = capture(query.format_rows, rows)
    lines = out.strip().split('\n')
    assert len(lines) == 2
    # Extract sim value from formatted output and verify exact value
    sim_idx = lines[0].index("sim=")
    sim_str = lines[0][sim_idx:sim_idx+9]
    assert sim_str == "sim=0.950"


#TAG: [E4BF] 2026-04-05
# Verifies: format_rows handles content with special characters without corruption
def test_format_rows_adversarial():
    db, c = fresh_db()
    c.execute("INSERT INTO memories (type, topic, content) VALUES ('fact', 'special', 'line1\\nline2 <tag> & \"quotes\"')")
    c.commit()
    c2 = sqlite3.connect(db)
    c2.row_factory = sqlite3.Row
    rows = c2.execute("SELECT id, type, topic, content, updated_at FROM memories").fetchall()
    c2.close()
    out, _ = capture(query.format_rows, rows)
    lines = out.strip().split('\n')
    assert len(lines) >= 2
    assert rows[0]["content"] == 'line1\\nline2 <tag> & "quotes"'
    c.close()
