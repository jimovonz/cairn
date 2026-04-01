#!/usr/bin/env python3
"""Tests for hooks/storage.py — memory storage, deduplication, confidence updates, quality gates."""

import os
import sys
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from types import ModuleType

# Add hooks and cairn directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))

import storage


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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
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

#TAG: [D377] 2026-04-02
# Verifies: corroboration (+) boosts confidence using saturating formula BOOST * (1 - current)
@pytest.mark.behavioural
def test_apply_confidence_corroboration_boosts(db_path):
    """Corroboration (+) should increase confidence toward max using BOOST * (1 - current)."""
    from config import CONFIDENCE_BOOST, CONFIDENCE_DEFAULT, CONFIDENCE_MAX

    _execute(db_path,
        "INSERT INTO memories (id, type, topic, content, confidence) VALUES (1, 'fact', 'test', 'some content', ?)",
        (CONFIDENCE_DEFAULT,))

    applied = storage.apply_confidence_updates([(1, "+", None)])
    assert applied == 1

    rows = _query(db_path, "SELECT confidence FROM memories WHERE id = 1")
    expected = min(CONFIDENCE_DEFAULT + CONFIDENCE_BOOST * (1 - CONFIDENCE_DEFAULT), CONFIDENCE_MAX)
    assert abs(rows[0][0] - expected) < 1e-9, f"Expected {expected}, got {rows[0][0]}"


#TAG: [8877] 2026-04-02
# Verifies: contradiction (-!) sets archived_reason to the provided reason string
@pytest.mark.behavioural
def test_apply_confidence_contradiction_annotates(db_path):
    """Contradiction (-!) should set archived_reason on the memory."""
    _execute(db_path,
        "INSERT INTO memories (id, type, topic, content, confidence) VALUES (1, 'fact', 'test', 'old info', 0.7)")

    reason = "superseded by new approach"
    applied = storage.apply_confidence_updates([(1, "-!", reason)])
    assert applied == 1

    rows = _query(db_path, "SELECT archived_reason FROM memories WHERE id = 1")
    assert rows[0][0] == reason


#TAG: [42E3] 2026-04-02
# Verifies: empty updates list returns 0 without any DB interaction
@pytest.mark.edge
def test_apply_confidence_empty_list():
    """Empty updates list should return 0 immediately."""
    assert storage.apply_confidence_updates([]) == 0


#TAG: [039F] 2026-04-02
# Verifies: update referencing non-existent memory ID returns 0 applied count
@pytest.mark.error
def test_apply_confidence_missing_memory(db_path):
    """Update for a non-existent memory ID should be skipped, returning 0."""
    applied = storage.apply_confidence_updates([(9999, "+", None)])
    assert applied == 0


#TAG: [EE63] 2026-04-02
# Verifies: irrelevant (-) direction increments applied count but leaves confidence at 0.7
@pytest.mark.adversarial
def test_apply_confidence_irrelevant_no_change(db_path):
    """Irrelevant (-) should not change the confidence value."""
    _execute(db_path,
        "INSERT INTO memories (id, type, topic, content, confidence) VALUES (1, 'fact', 'test', 'content', 0.7)")

    applied = storage.apply_confidence_updates([(1, "-", None)])
    assert applied == 1

    rows = _query(db_path, "SELECT confidence FROM memories WHERE id = 1")
    assert rows[0][0] == 0.7


# ============================================================
# insert_memories — behavioural
# ============================================================

#TAG: [7EDD] 2026-04-02
# Verifies: single valid entry is stored with correct type, topic, content in DB
@pytest.mark.behavioural
def test_insert_memories_basic(db_path):
    """Basic insert of a single valid memory should store it and return 1."""
    entries = [{"type": "fact", "topic": "db-choice", "content": "Use SQLite for persistence — zero-config, WAL mode"}]
    with patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT type, topic, content FROM memories WHERE topic = 'db-choice'")
    assert len(rows) == 1
    assert rows[0][0] == "fact"
    assert rows[0][2] == "Use SQLite for persistence — zero-config, WAL mode"


#TAG: [D81E] 2026-04-02
# Verifies: same type+topic without embedder inserts as distinct variant (sim=0.0 < threshold)
@pytest.mark.behavioural
def test_insert_memories_same_topic_distinct_variant(db_path):
    """Same type+topic with no embedder → sim=0.0 < DISTINCT_VARIANT threshold → inserts new row."""
    entries1 = [{"type": "fact", "topic": "version", "content": "Using Python 3.10 for the project runtime"}]
    entries2 = [{"type": "fact", "topic": "version", "content": "Upgraded to Python 3.11 for performance gains"}]
    with patch.object(storage, "_trigger_background_backfill"):
        storage.insert_memories(entries1, session_id="s1")
        storage.insert_memories(entries2, session_id="s2")

    rows = _query(db_path, "SELECT content FROM memories WHERE type = 'fact' AND topic = 'version' ORDER BY id")
    assert len(rows) == 2
    assert rows[0][0] == "Using Python 3.10 for the project runtime"
    assert rows[1][0] == "Upgraded to Python 3.11 for performance gains"


#TAG: [841D] 2026-04-02
# Verifies: write throttle keeps correction (priority 0) over project entries (priority 7)
@pytest.mark.behavioural
def test_insert_memories_write_throttle(db_path):
    """When entries exceed MAX_MEMORIES_PER_RESPONSE, corrections survive over projects."""
    from config import MAX_MEMORIES_PER_RESPONSE

    entries = [
        {"type": "project", "topic": f"proj{i}", "content": f"Project metadata entry number {i} for testing"} for i in range(MAX_MEMORIES_PER_RESPONSE + 3)
    ] + [
        {"type": "correction", "topic": "bugfix", "content": "Fixed off-by-one in loop counter — boundary check needed"},
    ]
    with patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count <= MAX_MEMORIES_PER_RESPONSE
    rows = _query(db_path, "SELECT content FROM memories WHERE topic = 'bugfix'")
    assert len(rows) == 1, "Correction should survive write throttle"


#TAG: [7BA2] 2026-04-02
# Verifies: semantic dedup updates existing row when find_nearest similarity >= DEDUP_THRESHOLD
@pytest.mark.behavioural
def test_insert_memories_semantic_dedup(db_path):
    """When find_nearest match >= DEDUP_THRESHOLD, existing row is updated."""
    from config import DEDUP_THRESHOLD

    _execute(db_path,
        "INSERT INTO memories (id, type, topic, content, embedding) VALUES (1, 'fact', 'db', 'Use SQLite for storage', ?)",
        (b"\x00" * (384 * 4),))

    mock_emb = MagicMock()
    mock_emb.embed.return_value = [0.1] * 384
    mock_emb.to_blob.return_value = b"\x01" * (384 * 4)
    mock_emb.find_nearest.return_value = [{
        "id": 1,
        "content": "Use SQLite for storage",
        "similarity": DEDUP_THRESHOLD + 0.01,
    }]
    mock_emb.upsert_vec_index = MagicMock()

    entries = [{"type": "fact", "topic": "database", "content": "SQLite selected for persistent storage layer"}]
    with patch.object(storage.hook_helpers, "get_embedder", return_value=mock_emb), \
         patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT COUNT(*) FROM memories")
    assert rows[0][0] == 1
    updated = _query(db_path, "SELECT content FROM memories WHERE id = 1")
    assert updated[0][0] == "SQLite selected for persistent storage layer"


# ============================================================
# insert_memories — edge
# ============================================================

#TAG: [58F4] 2026-04-02
# Verifies: empty entries list returns 0 without any DB or embedder interaction
@pytest.mark.edge
def test_insert_memories_empty_list():
    """Empty entries list should return 0 immediately."""
    assert storage.insert_memories([]) == 0


#TAG: [3356] 2026-04-02
# Verifies: quality gate rejects content matching "no context available" pattern
@pytest.mark.edge
def test_insert_memories_quality_gate_rejects_pattern(db_path):
    """Memory with empty-pattern content should be rejected."""
    entries = [{"type": "fact", "topic": "test", "content": "No context available for this topic"}]
    with patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 0
    rows = _query(db_path, "SELECT COUNT(*) FROM memories")
    assert rows[0][0] == 0


#TAG: [1DCD] 2026-04-02
# Verifies: content under 10 characters (after strip) is rejected by quality gate
@pytest.mark.edge
def test_insert_memories_quality_gate_short_content(db_path):
    """Content shorter than 10 characters should be rejected."""
    entries = [{"type": "fact", "topic": "tiny", "content": "short"}]
    with patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 0
    rows = _query(db_path, "SELECT COUNT(*) FROM memories")
    assert rows[0][0] == 0


#TAG: [3BC1] 2026-04-02
# Verifies: entry with no content key defaults to "" which is rejected by quality gate
@pytest.mark.edge
def test_insert_memories_no_content_key(db_path):
    """Entry with no content key defaults to '' and is rejected."""
    entries = [{"type": "fact", "topic": "empty"}]
    with patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 0
    rows = _query(db_path, "SELECT COUNT(*) FROM memories")
    assert rows[0][0] == 0


# ============================================================
# insert_memories — error
# ============================================================

#TAG: [25EE] 2026-04-02
# Verifies: ConnectionError from embedder is caught, memory stored without embedding
@pytest.mark.error
def test_insert_memories_embedding_connection_error(db_path):
    """When embedder raises ConnectionError, memory is still inserted without embedding."""
    mock_emb = MagicMock()
    mock_emb.embed.side_effect = ConnectionError("daemon down")

    entries = [{"type": "fact", "topic": "resilient", "content": "This memory should be stored despite embedding failure"}]
    with patch.object(storage.hook_helpers, "get_embedder", return_value=mock_emb), \
         patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT content, embedding FROM memories WHERE topic = 'resilient'")
    assert rows[0][0] == "This memory should be stored despite embedding failure"
    assert rows[0][1] is None


#TAG: [F30B] 2026-04-02
# Verifies: embed() returning None triggers daemon_unavailable path, memory stored without embedding
@pytest.mark.error
def test_insert_memories_embed_returns_none(db_path):
    """When embed() returns None (daemon unavailable), memory stored without embedding."""
    mock_emb = MagicMock()
    mock_emb.embed.return_value = None

    entries = [{"type": "decision", "topic": "approach", "content": "Chose event-driven architecture over polling for real-time updates"}]
    with patch.object(storage.hook_helpers, "get_embedder", return_value=mock_emb), \
         patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT embedding FROM memories WHERE topic = 'approach'")
    assert rows[0][0] is None


# ============================================================
# insert_memories — adversarial
# ============================================================

#TAG: [161B] 2026-04-02
# Verifies: entry missing type/topic keys uses defaults "fact"/"unknown"
@pytest.mark.adversarial
def test_insert_memories_missing_keys_defaults(db_path):
    """Entry with no type or topic should use defaults and still insert."""
    entries = [{"content": "A valid memory content that is long enough to pass quality gate"}]
    with patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT type, topic FROM memories")
    assert rows[0][0] == "fact"
    assert rows[0][1] == "unknown"


#TAG: [1856] 2026-04-02
# Verifies: negation mismatch on same type+topic sets archived_reason with "superseded" on old row
@pytest.mark.adversarial
def test_insert_memories_negation_supersedes(db_path):
    """Same type+topic with negation mismatch should annotate old as superseded."""
    _execute(db_path,
        "INSERT INTO memories (id, type, topic, content) VALUES (1, 'decision', 'feature-x', 'Enable feature X for all users')")

    entries = [{"type": "decision", "topic": "feature-x", "content": "Disable feature X due to performance regression"}]
    with patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT archived_reason FROM memories WHERE id = 1")
    assert isinstance(rows[0][0], str) and len(rows[0][0]) > 0, "archived_reason should be a non-empty string"
    assert rows[0][0].split(":")[0] == "superseded", f"Expected 'superseded' prefix, got: {rows[0][0]}"


# ============================================================
# Additional insert_memories paths
# ============================================================

#TAG: [47CF] 2026-04-02
# Verifies: project label from get_session_project is stored on inserted memories
@pytest.mark.behavioural
def test_insert_memories_project_label(db_path):
    """Memory should inherit project label from session."""
    entries = [{"type": "fact", "topic": "config", "content": "Configuration loaded from cairn/config.py module"}]
    with patch.object(storage, "get_session_project", return_value="cairn"), \
         patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT project FROM memories WHERE topic = 'config'")
    assert rows[0][0] == "cairn"


#TAG: [4C3E] 2026-04-02
# Verifies: backfill is triggered with count >= 1 when memories lack embeddings
@pytest.mark.behavioural
def test_insert_memories_triggers_backfill(db_path):
    """After inserting without embeddings, _trigger_background_backfill should be called."""
    entries = [{"type": "fact", "topic": "trigger", "content": "Memory that triggers background backfill process"}]
    with patch.object(storage, "_trigger_background_backfill") as mock_backfill:
        storage.insert_memories(entries, session_id="s1")

    mock_backfill.assert_called_once()
    assert mock_backfill.call_args[0][0] >= 1


#TAG: [F991] 2026-04-02
# Verifies: cross-topic negation via embedding similarity annotates old memory as superseded
@pytest.mark.behavioural
def test_insert_memories_negation_cross_topic(db_path):
    """Cross-topic negation via embedding similarity should annotate old as superseded."""
    from config import NEGATION_SIM_FLOOR, DEDUP_THRESHOLD

    _execute(db_path,
        "INSERT INTO memories (id, type, topic, content, embedding) VALUES (1, 'decision', 'auth', 'Use JWT tokens for authentication', ?)",
        (b"\x00" * (384 * 4),))

    sim = (NEGATION_SIM_FLOOR + DEDUP_THRESHOLD) / 2
    mock_emb = MagicMock()
    mock_emb.embed.return_value = [0.1] * 384
    mock_emb.to_blob.return_value = b"\x01" * (384 * 4)
    mock_emb.find_nearest.return_value = [{
        "id": 1,
        "content": "Use JWT tokens for authentication",
        "similarity": sim,
    }]
    mock_emb.upsert_vec_index = MagicMock()

    entries = [{"type": "decision", "topic": "auth-change", "content": "Avoid JWT tokens — replaced with session cookies"}]
    with patch.object(storage.hook_helpers, "get_embedder", return_value=mock_emb), \
         patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT archived_reason FROM memories WHERE id = 1")
    assert isinstance(rows[0][0], str) and len(rows[0][0]) > 0, "archived_reason should be a non-empty string"
    assert rows[0][0].split(":")[0] == "superseded", f"Expected 'superseded' prefix, got: {rows[0][0]}"


#TAG: [D883] 2026-04-02
# Verifies: depth field from entry dict is stored in the database
@pytest.mark.behavioural
def test_insert_memories_depth_stored(db_path):
    """The depth field should be stored when provided in entry."""
    entries = [{"type": "fact", "topic": "depth-test", "content": "Memory with depth value set to five turns back", "depth": 5}]
    with patch.object(storage, "_trigger_background_backfill"):
        count = storage.insert_memories(entries, session_id="s1")

    assert count == 1
    rows = _query(db_path, "SELECT depth FROM memories WHERE topic = 'depth-test'")
    assert rows[0][0] == 5
