#!/usr/bin/env python3
"""Tests for project_bootstrap function in prompt_hook.py."""

import re
import sys
import os
import json
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import tempfile
from unittest.mock import patch

import pytest


import hooks.hook_helpers as hook_helpers

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    """Create a fresh test database with required schema."""
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"pb_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    return db_path, conn


def insert_memory(conn, project, mem_type="project", topic="test-topic",
                  content="test content", confidence=0.7, archived_reason=None,
                  updated_at="2026-04-01 10:00:00"):
    """Insert a memory row and return its ID."""
    conn.execute(
        """INSERT INTO memories (type, topic, content, project, confidence,
           archived_reason, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (mem_type, topic, content, project, confidence, archived_reason, updated_at)
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def parse_entries(xml_str):
    """Extract entry dicts from cairn_context XML output."""
    entries = []
    for m in re.finditer(
        r'<entry\s+([^>]+)>([^<]+)</entry>', xml_str
    ):
        attrs_str, content = m.group(1), m.group(2)
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', attrs_str))
        attrs["content"] = content
        entries.append(attrs)
    return entries


#TAG: [8933] 2026-04-05 src:A6CB
# Verifies: returns structured cairn_context XML with correct entries, attributes, ordering, type filtering, archived exclusion, and metric recording
@pytest.mark.behavioural
def test_project_bootstrap_behavioural():
    from hooks.prompt_hook import project_bootstrap
    db_path, conn = fresh_db()
    id1 = insert_memory(conn, "myproject", mem_type="project", topic="arch",
                        content="Uses SQLite for storage", confidence=0.8,
                        updated_at="2026-04-01 12:00:00")
    id2 = insert_memory(conn, "myproject", mem_type="fact", topic="env",
                        content="Python 3.11 required", confidence=0.6,
                        updated_at="2026-04-01 11:00:00")
    insert_memory(conn, "myproject", mem_type="preference", topic="style",
                  content="Prefers short functions", confidence=0.9,
                  updated_at="2026-04-01 10:00:00")
    # Archived memory — should be excluded
    insert_memory(conn, "myproject", mem_type="project", topic="old",
                  content="archived decision", archived_reason="superseded")
    # Wrong type — correction not in default PROJECT_BOOTSTRAP_TYPES
    insert_memory(conn, "myproject", mem_type="correction", topic="fix",
                  content="correction mem")
    # Wrong type — skill not in default PROJECT_BOOTSTRAP_TYPES
    insert_memory(conn, "myproject", mem_type="skill", topic="cmd",
                  content="skill mem")
    conn.close()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = project_bootstrap("sess-beh", "/home/user/Projects/myproject")
        hook_helpers.flush_metrics()

    # Structural: starts with XML tag, contains required layers
    assert result.startswith('<cairn_context')
    assert result.endswith('</cairn_context>')

    # Parse entries and verify count — 3 valid (project, fact, preference), not 6
    entries = parse_entries(result)
    assert len(entries) == 3

    # Verify ordering: most recent first (project@12:00 > fact@11:00 > preference@10:00)
    assert entries[0]["content"] == "Uses SQLite for storage"
    assert entries[1]["content"] == "Python 3.11 required"
    assert entries[2]["content"] == "Prefers short functions"

    # Verify entry attributes match inserted data
    assert entries[0]["type"] == "project"
    assert entries[0]["topic"] == "arch"
    assert entries[0]["confidence"] == "0.80"
    assert entries[1]["type"] == "fact"
    assert entries[1]["confidence"] == "0.60"
    assert entries[2]["type"] == "preference"
    assert entries[2]["confidence"] == "0.90"

    # Verify reliability mapping: 0.8 >= 0.6 → strong, 0.6 >= 0.6 → strong, 0.9 → strong
    assert entries[0]["reliability"] == "strong"
    assert entries[1]["reliability"] == "strong"

    # Verify excluded types are not present
    contents = [e["content"] for e in entries]
    assert "archived decision" not in contents
    assert "correction mem" not in contents
    assert "skill mem" not in contents

    # Verify project attribute in XML header
    assert 'current_project="myproject"' in result.split('\n')[0]
    assert 'layer="project-bootstrap"' in result.split('\n')[0]

    # Verify metric was recorded
    verify_conn = sqlite3.connect(db_path)
    row = verify_conn.execute(
        "SELECT event, session_id, detail, value FROM metrics WHERE event = 'project_bootstrap_injected'"
    ).fetchone()
    verify_conn.close()
    assert row[0] == "project_bootstrap_injected"
    assert row[1] == "sess-beh"
    assert row[2] == "myproject"
    assert row[3] == 3  # 3 valid memories injected


#TAG: [12BE] 2026-04-05 src:A6CB
# Verifies: returns None for empty cwd, blocked names, root path, dot path, disabled config, no-match project, and respects PROJECT_BOOTSTRAP_MAX cap
@pytest.mark.edge
def test_project_bootstrap_edge():
    from hooks.prompt_hook import project_bootstrap
    import cairn.config as config
    db_path, conn = fresh_db()
    # Insert 10 memories to test max cap
    for i in range(10):
        insert_memory(conn, "captest", mem_type="project",
                      topic=f"topic-{i}", content=f"memory-{i:03d}",
                      updated_at=f"2026-04-01 {10+i:02d}:00:00")
    conn.close()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        # Empty / invalid CWD returns None
        assert project_bootstrap("s1", "") is None
        assert project_bootstrap("s1", "/") is None
        assert project_bootstrap("s1", "/home") is None
        assert project_bootstrap("s1", "/tmp") is None
        assert project_bootstrap("s1", "/temp") is None
        assert project_bootstrap("s1", ".") is None

        # Non-matching project returns None (no memories for "nomatch")
        assert project_bootstrap("s1", "/home/user/nomatch") is None

    # Disabled via config returns None
    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(config, 'PROJECT_BOOTSTRAP_ENABLED', False):
        assert project_bootstrap("s1", "/home/user/captest") is None

    # Max cap: 10 inserted but only PROJECT_BOOTSTRAP_MAX (5) returned
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = project_bootstrap("s-cap", "/home/user/captest")

    entries = parse_entries(result)
    assert len(entries) == 5
    # Most recent 5: memories 9,8,7,6,5 (updated_at 19:00 down to 15:00)
    assert entries[0]["content"] == "memory-009"
    assert entries[4]["content"] == "memory-005"
    # Memory 4 (14:00) should NOT be present — beyond the cap
    capped_contents = [e["content"] for e in entries]
    assert "memory-004" not in capped_contents


#TAG: [4FDC] 2026-04-05 src:A6CB
# Verifies: returns None without crashing when DB is broken, table is missing, or query fails
@pytest.mark.error
def test_project_bootstrap_error():
    from hooks.prompt_hook import project_bootstrap
    db_path, conn = fresh_db()
    conn.close()

    # Drop the memories table so the SQL query will fail
    broken_conn = sqlite3.connect(db_path)
    broken_conn.execute("DROP TABLE memories")
    broken_conn.commit()
    broken_conn.close()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = project_bootstrap("sess-err", "/home/user/someproject")

    # Must return None, not raise
    assert result is None

    # Also test with a completely non-existent DB path
    with patch.object(hook_helpers, 'DB_PATH', "/nonexistent/path/db.sqlite"):
        result2 = project_bootstrap("sess-err2", "/home/user/someproject")
    assert result2 is None


#TAG: [8C63] 2026-04-05 src:A6CB
# Verifies: handles NULL confidence (defaults to 0.7), malformed updated_at (days=0), and produces valid XML structure despite bad data
@pytest.mark.adversarial
def test_project_bootstrap_adversarial():
    from hooks.prompt_hook import project_bootstrap
    db_path, conn = fresh_db()

    # Memory with NULL confidence
    conn.execute(
        """INSERT INTO memories (type, topic, content, project, confidence, updated_at)
           VALUES ('project', 'null-conf', 'no confidence set', 'weirdproject', NULL,
                   '2026-04-01 10:00:00')"""
    )
    # Memory with malformed updated_at
    conn.execute(
        """INSERT INTO memories (type, topic, content, project, confidence, updated_at)
           VALUES ('fact', 'bad-date', 'bad timestamp value', 'weirdproject', 0.5, 'not-a-date')"""
    )
    # Memory with empty string archived_reason (should still be included — only non-empty excludes)
    conn.execute(
        """INSERT INTO memories (type, topic, content, project, confidence, archived_reason, updated_at)
           VALUES ('project', 'empty-arch', 'empty archived reason', 'weirdproject', 0.3, '',
                   '2026-04-01 09:00:00')"""
    )
    conn.commit()
    conn.close()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = project_bootstrap("sess-adv", "/home/user/weirdproject")

    entries = parse_entries(result)
    assert len(entries) == 3

    # Find entries by content
    by_content = {e["content"]: e for e in entries}

    # NULL confidence → defaults to 0.7 in code
    null_conf_entry = by_content["no confidence set"]
    assert null_conf_entry["confidence"] == "0.70"

    # Malformed date → recency_days should be 0 (fallback), confidence correct
    bad_date_entry = by_content["bad timestamp value"]
    assert bad_date_entry["recency_days"] == "0"
    assert bad_date_entry["confidence"] == "0.50"

    # Low confidence (0.3) → reliability "weak" (score < 0.4)
    empty_arch_entry = by_content["empty archived reason"]
    assert empty_arch_entry["reliability"] == "weak"
    assert empty_arch_entry["confidence"] == "0.30"

    # Verify overall XML structure is well-formed despite bad data
    assert result.startswith('<cairn_context')
    assert result.endswith('</cairn_context>')
    assert result.count('<scope ') == 1
    assert result.count('</scope>') == 1
