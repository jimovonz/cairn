#!/usr/bin/env python3
"""Tests for delivery-count dampening and per-file S/N scoping.

Covers the 2026-06-12 retrieval-quality changes:
- delivery_counts lifetime counters (record_layer_delivery / overdelivered_ids)
- dampened entries retired BEFORE the top-N cut (slots free up, unlike the
  session served-ledger which never backfills)
- generic basenames never basename-match (weak cross-project keys)
- associated-files write hygiene (junk filter, edited-preference, cap)
"""

import json
import os
import sys
import tempfile
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import pytest
from unittest.mock import patch

import hooks.hook_helpers as hook_helpers
import hooks.pretool_hook as pretool_hook
import hooks.storage as storage
from hooks.pretool_hook import find_memories_for_file, sections_for_file

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"dampen_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        associated_files TEXT, keywords TEXT, confidence REAL DEFAULT 0.7,
        archived_reason TEXT, session_id TEXT, project TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP, topic_embedding BLOB)""")
    conn.execute("""CREATE TABLE metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT, session_id TEXT,
        detail TEXT, value REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE hook_state (
        session_id TEXT, key TEXT, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


#TAG: [DC01] 2026-06-12
# Verifies: record_layer_delivery creates and increments per-memory lifetime
# counters; overdelivered_ids returns exactly the ids at/above threshold
@pytest.mark.behavioural
def test_record_layer_delivery_behavioural():
    db_path, conn = fresh_db()
    conn.close()

    with patch.object(hook_helpers, "DB_PATH", db_path), \
         patch("cairn.config.EPHEMERAL_DB_PATH", db_path):
        hook_helpers.record_layer_delivery("s1", "per-file", [7, 8])
        hook_helpers.record_layer_delivery("s2", "L1.5", [7])
        hook_helpers.record_layer_delivery("s3", "per-file", [7])
        over = hook_helpers.overdelivered_ids(3)
        under = hook_helpers.overdelivered_ids(4)

    check = sqlite3.connect(db_path)
    rows = dict(check.execute("SELECT memory_id, count FROM delivery_counts").fetchall())
    check.close()
    assert rows == {7: 3, 8: 1}
    assert over == {7}
    assert under == set()


#TAG: [DC02] 2026-06-12
# Verifies: dampened ids are retired BEFORE the top-N cut, so the next-ranked
# memory takes the freed slot (unlike the served ledger, which never backfills)
@pytest.mark.behavioural
def test_sections_for_file_behavioural():
    db_path, conn = fresh_db()
    # Three corrections on the same file; MAX_GOTCHA_INJECTIONS is 3 but we
    # dampen the first — remaining two must serve, slot freed.
    for i in range(1, 4):
        conn.execute(
            "INSERT INTO memories (type, topic, content, associated_files, confidence)"
            " VALUES ('correction', 'gotcha-' || ?, 'Warning number ' || ?, ?, 0.9)",
            (i, i, json.dumps(["/proj/app.py"]))
        )
    conn.commit()
    conn.close()

    with patch.object(hook_helpers, "DB_PATH", db_path), \
         patch("cairn.config.EPHEMERAL_DB_PATH", db_path), \
         patch("hooks.pretool_hook.log"), patch("hooks.pretool_hook.record_metric"):
        sections, new_ids = sections_for_file(
            "/proj/app.py", "sess-d", set(), None, set(), dampened={1})

    assert new_ids == [2, 3]
    lines = sections[0].split("\n")
    assert lines[0] == "CAIRN GOTCHA for app.py:"
    assert lines[1] == "- [gotcha-2] Warning number 2"
    assert lines[2] == "- [gotcha-3] Warning number 3"
    assert lines[3] == "Sources: 2, 3"


#TAG: [DC03] 2026-06-12
# Verifies: generic basenames (README.md) never basename-match even for the
# same project; exact path matches still serve
@pytest.mark.adversarial
def test_find_memories_for_file_adversarial_generic():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, project)"
        " VALUES ('fact', 'readme-evergreen', 'Cross-project README noise', ?, 0.95, 'proj')",
        (json.dumps(["/other/repo/README.md"]),)
    )
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, project)"
        " VALUES ('fact', 'readme-exact', 'This exact README', ?, 0.6, 'proj')",
        (json.dumps(["/proj/README.md"]),)
    )
    conn.commit()
    conn.close()

    with patch.object(hook_helpers, "DB_PATH", db_path), \
         patch("cairn.config.EPHEMERAL_DB_PATH", db_path), \
         patch("hooks.pretool_hook.log"):
        results = find_memories_for_file("/proj/README.md", corrections_only=False,
                                         project="proj")

    topics = [r["topic"] for r in results]
    assert topics == ["readme-exact"]


#TAG: [DC04] 2026-06-12
# Verifies: junk paths (venv, /tmp, logs, dbs) are excluded at write time and
# the association list is capped at ASSOC_FILES_MAX
@pytest.mark.edge
def test_extract_associated_files_edge_hygiene(tmp_path):
    from cairn.config import ASSOC_FILES_MAX
    lines = [
        json.dumps({"tool_name": "Edit", "parameters": {"file_path": "/repo/.venv/lib/pkg.py"}}),
        json.dumps({"tool_name": "Edit", "parameters": {"file_path": "/tmp/scratch.py"}}),
        json.dumps({"tool_name": "Edit", "parameters": {"file_path": "/repo/hook.log"}}),
        json.dumps({"tool_name": "Edit", "parameters": {"file_path": "/repo/cairn.db"}}),
    ]
    # 10 legitimate edited files — must be capped to ASSOC_FILES_MAX
    for i in range(10):
        lines.append(json.dumps({"tool_name": "Edit",
                                 "parameters": {"file_path": f"/repo/src/mod_{i}.py"}}))
    (tmp_path / "t.jsonl").write_text("\n".join(lines))

    result = storage.extract_associated_files(str(tmp_path / "t.jsonl"), lookback=0)

    assert len(result) == ASSOC_FILES_MAX
    assert result == [f"/repo/src/mod_{i}.py" for i in range(ASSOC_FILES_MAX)]


#TAG: [DC05] 2026-06-12
# Verifies: cch-edit/cch-write Bash helper targets count as edited files and
# win the edited-preference over plain read paths
@pytest.mark.behavioural
def test_extract_associated_files_behavioural_bash_helpers(tmp_path):
    lines = [
        json.dumps({"tool_name": "Bash", "parameters": {"command": "cat /repo/src/reader.py"}}),
        json.dumps({"tool_name": "Bash", "parameters": {"command": "cch-edit.py /repo/src/target.py 'old' 'new'"}}),
    ]
    (tmp_path / "t.jsonl").write_text("\n".join(lines))

    result = storage.extract_associated_files(str(tmp_path / "t.jsonl"), lookback=0)

    assert result == ["/repo/src/target.py"]


#TAG: [DC06] 2026-06-12
# Verifies: _maybe_optimize_fts merges FTS segments above the threshold and
# no-ops below it (write-path self-maintenance for MATCH query latency)
@pytest.mark.behavioural
def test_maybe_optimize_fts_behavioural(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "fts.db"))
    conn.execute("CREATE VIRTUAL TABLE memories_fts USING fts5(topic, content)")
    # One commit per insert fragments the index into many segments
    for i in range(40):
        conn.execute("INSERT INTO memories_fts (topic, content) VALUES (?, ?)",
                     (f"topic-{i}", f"content body number {i}"))
        conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM memories_fts_data").fetchone()[0]
    assert before > 10

    with patch("hooks.storage.log"), \
         patch.object(hook_helpers, "record_metric"):
        # Below threshold: no-op
        with patch("cairn.config.FTS_OPTIMIZE_SEGMENT_THRESHOLD", before + 1):
            storage._maybe_optimize_fts(conn)
        unchanged = conn.execute("SELECT COUNT(*) FROM memories_fts_data").fetchone()[0]
        assert unchanged == before

        # At/above threshold: segments merge
        with patch("cairn.config.FTS_OPTIMIZE_SEGMENT_THRESHOLD", 10):
            storage._maybe_optimize_fts(conn)
        merged = conn.execute("SELECT COUNT(*) FROM memories_fts_data").fetchone()[0]
        assert merged < before
    conn.close()
