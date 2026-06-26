#!/usr/bin/env python3
"""Tests for hooks/storage.py — memory storage, deduplication, confidence updates, quality gates."""

import json
import os
import sys
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import pytest
from unittest.mock import patch, MagicMock

# Add hooks and cairn directories to path

import hooks.storage as storage


def _init_db(path):
    """Create a file-based SQLite DB with the memories schema."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            session_id TEXT,
            project TEXT,
            confidence REAL DEFAULT 0.7,
            source_start INTEGER,
            source_end INTEGER,
            anchor_line INTEGER,
            depth INTEGER,
            archived_reason TEXT,
            associated_files TEXT,
            keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            origin_id TEXT,
            user_id TEXT,
            updated_by TEXT,
            team_id TEXT,
            source_ref TEXT,
            deleted_at TIMESTAMP,
            synced_at TIMESTAMP,
            facts TEXT,
        topic_embedding BLOB)
    """)
    conn.execute("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            project TEXT,
            transcript_path TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            session_id TEXT,
            detail TEXT,
            value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    """Provide a temp DB path, initialized with schema."""
    path = str(tmp_path / "test.db")
    _init_db(path)
    return path


@pytest.fixture(autouse=True)
def patch_storage(db_path):
    """Patch storage module's imported references to use our temp DB."""
    def _get_conn():
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    with patch.object(storage, "get_conn", side_effect=_get_conn), \
         patch.object(storage, "log"), \
         patch.object(storage, "record_metric"), \
         patch.object(storage, "get_session_project", return_value=None), \
         patch.object(storage.hook_helpers, "get_embedder", return_value=None):
        yield


def _query(db_path, sql, params=()):
    """Run a query against the test DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def _execute(db_path, sql, params=()):
    """Execute a write against the test DB."""
    conn = sqlite3.connect(db_path)
    conn.execute(sql, params)
    conn.commit()
    conn.close()


# ============================================================
# apply_confidence_updates
# ============================================================

#TAG: [9FE4] 2026-04-05
# Verifies: corroboration (+) boosts confidence via saturating formula; contradiction (-!) sets archived_reason
@pytest.mark.behavioural
def test_apply_confidence_updates_behavioural(db_path):
    from cairn.config import CONFIDENCE_BOOST, CONFIDENCE_DEFAULT, CONFIDENCE_MAX
    _execute(db_path, "INSERT INTO memories (id, type, topic, content, confidence) VALUES (1, 'fact', 'a', 'content here for testing', ?)", (CONFIDENCE_DEFAULT,))
    _execute(db_path, "INSERT INTO memories (id, type, topic, content, confidence) VALUES (2, 'fact', 'b', 'other content for test', 0.7)")

    applied = storage.apply_confidence_updates([(1, "+", None), (2, "-!", "superseded by new info")])

    assert applied == 2
    conf_rows = _query(db_path, "SELECT confidence FROM memories WHERE id = 1")
    expected = min(CONFIDENCE_DEFAULT + CONFIDENCE_BOOST * (1 - CONFIDENCE_DEFAULT), CONFIDENCE_MAX)
    assert abs(conf_rows[0][0] - expected) < 1e-9
    reason_rows = _query(db_path, "SELECT archived_reason FROM memories WHERE id = 2")
    assert reason_rows[0][0] == "superseded by new info"


#TAG: [4FF2] 2026-04-05
# Verifies: empty updates list returns 0 immediately without any DB interaction
@pytest.mark.edge
def test_apply_confidence_updates_edge():
    assert storage.apply_confidence_updates([]) == 0


#TAG: [7FAA] 2026-04-05
# Verifies: update referencing non-existent memory ID is skipped and returns 0 applied count
@pytest.mark.error
def test_apply_confidence_updates_error(db_path):
    applied = storage.apply_confidence_updates([(9999, "+", None)])
    assert applied == 0


#TAG: [AAF6] 2026-04-05
# Verifies: irrelevant (-) increments applied count but leaves confidence unchanged at original value
@pytest.mark.adversarial
def test_apply_confidence_updates_adversarial(db_path):
    _execute(db_path, "INSERT INTO memories (id, type, topic, content, confidence) VALUES (1, 'fact', 't', 'some long content value here', 0.7)")
    applied = storage.apply_confidence_updates([(1, "-", None)])
    assert applied == 1
    rows = _query(db_path, "SELECT confidence FROM memories WHERE id = 1")
    assert rows[0][0] == 0.7


# ============================================================
# extract_associated_files
# ============================================================

#TAG: [7A3E] 2026-04-05
# Verifies: JSONL with Read, Edit, Write, MultiEdit tool calls returns unique file paths in insertion order
@pytest.mark.behavioural
def test_extract_associated_files_behavioural(tmp_path):
    lines = [
        json.dumps({"tool_name": "Read", "parameters": {"file_path": "/hooks/storage.py"}}),
        json.dumps({"tool_name": "Edit", "parameters": {"file_path": "/hooks/pretool_hook.py"}}),
        json.dumps({"tool_name": "Write", "parameters": {"file_path": "/cairn/query.py"}}),
        json.dumps({"tool_name": "Read", "parameters": {"file_path": "/hooks/storage.py"}}),  # duplicate
        json.dumps({"tool_name": "MultiEdit", "input": {"edits": [
            {"file_path": "/tests/test_storage.py"},
            {"filePath": "/cairn/config.py"},
        ]}}),
    ]
    (tmp_path / "t.jsonl").write_text("\n".join(lines))

    result = storage.extract_associated_files(str(tmp_path / "t.jsonl"))

    # Edited-files preference: the Read of /hooks/storage.py is dropped because
    # Edit/Write/MultiEdit targets exist in the window — edits mark what the
    # memory is actually about.
    assert result == [
        "/hooks/pretool_hook.py",
        "/cairn/query.py",
        "/tests/test_storage.py",
        "/cairn/config.py",
    ]


#TAG: [5142] 2026-04-05
# Verifies: empty transcript file returns [] and empty string argument returns []
@pytest.mark.edge
def test_extract_associated_files_edge(tmp_path):
    (tmp_path / "empty.jsonl").write_text("")
    assert storage.extract_associated_files(str(tmp_path / "empty.jsonl")) == []
    assert storage.extract_associated_files("") == []


#TAG: [663A] 2026-04-05
# Verifies: non-existent transcript path returns [] with FileNotFoundError silently caught
@pytest.mark.error
def test_extract_associated_files_error():
    result = storage.extract_associated_files("/nonexistent/path/no_such_file.jsonl")
    assert result == []


#TAG: [7242] 2026-04-05
# Verifies: malformed JSON lines are skipped; Bash tool file paths extracted via regex; valid entries processed
@pytest.mark.adversarial
def test_extract_associated_files_adversarial(tmp_path):
    lines = [
        "NOT VALID JSON{{{",
        json.dumps({"tool_name": "Bash", "parameters": {"command": "python3 /hooks/storage.py /cairn/daemon.py"}}),
        json.dumps({"tool_name": "Read", "parameters": {"file_path": "/cairn/config.py"}}),
        "",
        "{incomplete json",
    ]
    (tmp_path / "t.jsonl").write_text("\n".join(lines))

    result = storage.extract_associated_files(str(tmp_path / "t.jsonl"))

    assert result == ["/hooks/storage.py", "/cairn/daemon.py", "/cairn/config.py"]
    assert len(result) == len(set(result))


# ============================================================
# insert_memories
# ============================================================

#TAG: [D8E8] 2026-04-05
# Verifies: single valid entry stored correctly; associated_files JSON column set from transcript_path
@pytest.mark.behavioural
def test_insert_memories_behavioural(db_path, tmp_path):
    lines = [json.dumps({"tool_name": "Read", "parameters": {"file_path": "/hooks/storage.py"}})]
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(lines))

    entries = [{"type": "correction", "topic": "boundary", "content": "Fixed off-by-one in loop boundary — exclusive vs inclusive upper bound"}]
    with patch.object(storage, "inline_backfill"):
        count = storage.insert_memories(entries, session_id="s1", transcript_path=str(transcript))

    assert count == 1
    rows = _query(db_path, "SELECT content, associated_files FROM memories WHERE topic = 'boundary'")
    assert rows[0][0] == "Fixed off-by-one in loop boundary — exclusive vs inclusive upper bound"
    assert json.loads(rows[0][1]) == ["/hooks/storage.py"]


# Step 3a: write provenance (memories.source_ref) is stamped at the CALL SITE that
# knows the writer, never blanket-defaulted in this shared sink. Precedence per
# entry: entry["source_ref"] > the source_ref param > NULL ("unknown provenance" —
# honest and excludable from the A/B cohort, vs a false version that poisons it).
@pytest.mark.behavioural
def test_union_keywords():
    # Keyword union on dedup: re-encounters ENRICH a memory's keywords, not clobber.
    assert storage._union_keywords("a,b", "b,c") == "a,b,c"
    assert storage._union_keywords("Foo,bar", "foo,BAZ") == "Foo,bar,BAZ"  # case-insensitive dedup, first casing kept
    assert storage._union_keywords("a, b ", " c ,a") == "a,b,c"            # whitespace stripped
    assert storage._union_keywords(None, "x") == "x"
    assert storage._union_keywords("x", None) == "x"
    assert storage._union_keywords(None, None) is None
    assert storage._union_keywords("", "") is None


def test_insert_memories_source_ref_provenance(db_path):
    LONG = "A sufficiently long memory content to clear the quality gate"
    with patch.object(storage, "inline_backfill"):
        # 1. No source_ref anywhere -> NULL (the sink imposes no default).
        storage.insert_memories([{"type": "fact", "topic": "prov_null", "content": LONG}])
        # 2. Call-level param is the fallback.
        storage.insert_memories([{"type": "fact", "topic": "prov_param", "content": LONG}],
                                source_ref="genTEST-v9")
        # 3. Per-entry source_ref overrides the call-level param.
        storage.insert_memories([{"type": "fact", "topic": "prov_entry", "content": LONG,
                                  "source_ref": "review-writeback"}],
                                source_ref="genTEST-v9")
    def _sr(topic):
        return _query(db_path, "SELECT source_ref FROM memories WHERE topic = ?", (topic,))[0][0]
    assert _sr("prov_null") is None
    assert _sr("prov_param") == "genTEST-v9"
    assert _sr("prov_entry") == "review-writeback"


#TAG: [24F1] 2026-04-05
# Verifies: empty list returns 0; short content rejected by quality gate; no transcript_path → associated_files NULL
@pytest.mark.edge
def test_insert_memories_edge(db_path):
    assert storage.insert_memories([]) == 0

    with patch.object(storage, "inline_backfill"):
        count_short = storage.insert_memories([{"type": "fact", "topic": "t", "content": "short"}])
    assert count_short == 0

    with patch.object(storage, "inline_backfill"):
        storage.insert_memories([{"type": "fact", "topic": "no-files", "content": "Memory without any transcript file association context provided"}])
    rows = _query(db_path, "SELECT associated_files FROM memories WHERE topic = 'no-files'")
    assert rows[0][0] is None


#TAG: [8A94] 2026-04-05
# Verifies: ConnectionError from embedder is caught and memory is inserted without embedding blob
@pytest.mark.error
def test_insert_memories_error(db_path):
    mock_emb = MagicMock()
    mock_emb.embed.side_effect = ConnectionError("daemon down")

    with patch.object(storage.hook_helpers, "get_embedder", return_value=mock_emb), \
         patch.object(storage, "inline_backfill"):
        count = storage.insert_memories(
            [{"type": "fact", "topic": "resilient", "content": "Memory stored despite ConnectionError from embedding daemon"}],
            session_id="s1"
        )

    assert count == 1
    rows = _query(db_path, "SELECT embedding FROM memories WHERE topic = 'resilient'")
    assert rows[0][0] is None


#TAG: [28B4] 2026-04-05
# Verifies: same type+topic with negation mismatch annotates old row archived_reason as superseded
@pytest.mark.adversarial
def test_insert_memories_adversarial(db_path):
    _execute(db_path, "INSERT INTO memories (id, type, topic, content) VALUES (1, 'decision', 'feat', 'Enable feature X for all users')")

    with patch.object(storage, "inline_backfill"):
        count = storage.insert_memories(
            [{"type": "decision", "topic": "feat", "content": "Disable feature X due to performance regression"}],
            session_id="s1"
        )

    assert count == 1
    rows = _query(db_path, "SELECT archived_reason FROM memories WHERE id = 1")
    assert isinstance(rows[0][0], str) and rows[0][0].startswith("superseded")


def test_null_embedding_trigger_on_content_edit(tmp_path):
    """Trigger nulls embedding when content changes without setting a new embedding."""
    import cairn.init_db as init_db
    db_path = str(tmp_path / "trigger_test.db")
    old_db_path = init_db.DB_PATH
    init_db.DB_PATH = db_path
    try:
        init_db.init()
    finally:
        init_db.DB_PATH = old_db_path

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    fake_embedding = b'\x00' * 16
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
        ("fact", "test-topic", "original content", fake_embedding)
    )
    conn.commit()

    row = conn.execute("SELECT embedding FROM memories WHERE id = 1").fetchone()
    assert row[0] == fake_embedding

    conn.execute("UPDATE memories SET content = 'updated content' WHERE id = 1")
    conn.commit()

    row = conn.execute("SELECT embedding FROM memories WHERE id = 1").fetchone()
    assert row[0] is None

    conn.close()


def test_null_embedding_trigger_skips_when_embedding_also_set(tmp_path):
    """Trigger does NOT null embedding when content and embedding change together."""
    import cairn.init_db as init_db
    db_path = str(tmp_path / "trigger_test2.db")
    old_db_path = init_db.DB_PATH
    init_db.DB_PATH = db_path
    try:
        init_db.init()
    finally:
        init_db.DB_PATH = old_db_path

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    fake_embedding = b'\x00' * 16
    new_embedding = b'\x01' * 16
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
        ("fact", "test-topic", "original content", fake_embedding)
    )
    conn.commit()

    conn.execute(
        "UPDATE memories SET content = 'updated content', embedding = ? WHERE id = 1",
        (new_embedding,)
    )
    conn.commit()

    row = conn.execute("SELECT embedding FROM memories WHERE id = 1").fetchone()
    assert row[0] == new_embedding

    conn.close()


# ============================================================
# explicit per-entry associated_files (review write-back keying)
# ============================================================

# Verifies: an entry-level associated_files list is stored verbatim and takes
# precedence over transcript-derived files — the keying path the review
# write-back relies on so cairn-graph --knowledge surfaces findings by file.
@pytest.mark.behavioural
def test_insert_memories_explicit_associated_files(db_path, tmp_path):
    # A transcript that would otherwise yield a *different* edited file.
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({
        "type": "assistant",
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "/some/other/edited.py"}}
        ]}
    }) + "\n")
    entries = [{
        "type": "correction", "topic": "explicit-files",
        "content": "A finding keyed explicitly to a changed file via the associated_files override.",
        "associated_files": ["/repo/hooks/storage.py", "hooks/storage.py"],
    }]
    count = storage.insert_memories(entries, session_id="s1", transcript_path=str(transcript))
    assert count == 1
    rows = _query(db_path, "SELECT associated_files FROM memories WHERE topic = 'explicit-files'")
    assert len(rows) == 1
    stored = json.loads(rows[0][0])
    assert stored == ["/repo/hooks/storage.py", "hooks/storage.py"]
    # The transcript-derived path must NOT override the explicit list.
    assert "/some/other/edited.py" not in stored


# Verifies: non-string / empty members in an explicit associated_files list are
# filtered out (defensive against malformed write-back input).
@pytest.mark.edge
def test_insert_memories_explicit_associated_files_filtered(db_path):
    entries = [{
        "type": "fact", "topic": "filtered-files",
        "content": "A fact whose explicit file list contains junk that must be dropped before storage.",
        "associated_files": ["good/path.py", "", None, 123, "  "],
    }]
    storage.insert_memories(entries, session_id="s1")
    rows = _query(db_path, "SELECT associated_files FROM memories WHERE topic = 'filtered-files'")
    stored = json.loads(rows[0][0])
    assert stored == ["good/path.py"]
