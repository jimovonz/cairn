#!/usr/bin/env python3
"""Tests for hooks/retrieval.py — retrieve_context, layer2_cross_project_search,
get_adaptive_threshold_boost, is_context_cached, and cache helpers.

Uses in-memory SQLite with mock embeddings to exercise real code paths."""

import sys
import os
import json
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import tempfile
import re
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


import hooks.hook_helpers as hook_helpers
import hooks.retrieval as retrieval

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def make_vector(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


def make_blob(seed):
    return make_vector(seed).tobytes()


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"rh_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, keywords, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content, keywords) VALUES (new.id, new.topic, new.content, new.keywords); END""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


def seed_project_session(conn, session_id="test-session", project="TestProject"):
    conn.execute("INSERT INTO sessions (session_id, project) VALUES (?, ?)", (session_id, project))
    conn.commit()


def parse_xml_attr(xml, attr_name):
    """Extract attribute value from XML string."""
    m = re.search(rf'{attr_name}="([^"]*)"', xml)
    return m.group(1) if m else None


# ============================================================
# get_adaptive_threshold_boost — 4 tests
# ============================================================

#TAG: [5396] 2026-04-05
# Verifies: returns 0.10 boost when harmful+neutral rate exceeds 50%
@pytest.mark.behavioural
def test_get_adaptive_threshold_boost_high_harmful_rate():
    db_path, conn = fresh_db()
    for _ in range(6):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_harmful', 's1')")
    for _ in range(4):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_useful', 's1')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        boost = retrieval.get_adaptive_threshold_boost()

    assert boost == 0.10, f"Expected 0.10 for 60% harmful rate, got {boost}"
    conn.close()


#TAG: [C8A0] 2026-04-05
# Verifies: returns 0.0 when fewer than 5 total metrics exist
@pytest.mark.edge
def test_get_adaptive_threshold_boost_insufficient_data():
    db_path, conn = fresh_db()
    for _ in range(3):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_harmful', 's1')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        boost = retrieval.get_adaptive_threshold_boost()

    assert boost == 0.0, f"Expected 0.0 for < 5 total metrics, got {boost}"
    conn.close()


#TAG: [2488] 2026-04-05
# Verifies: returns 0.0 when database access raises sqlite3.Error
@pytest.mark.error
def test_get_adaptive_threshold_boost_db_error():
    # Use the same sqlite3 module as retrieval.py (may be pysqlite3)
    from hooks import retrieval as _ret_mod
    with patch('hooks.retrieval.get_conn', side_effect=_ret_mod.sqlite3.OperationalError("locked")):
        boost = retrieval.get_adaptive_threshold_boost()

    assert boost == 0.0, f"Expected 0.0 on DB error, got {boost}"


#TAG: [1685] 2026-04-05
# Verifies: returns 0.05 (not 0.10) when harmful rate is exactly 50% boundary
@pytest.mark.adversarial
def test_get_adaptive_threshold_boost_boundary_fifty_percent():
    db_path, conn = fresh_db()
    # 5 harmful + 5 useful = exactly 50% harmful rate
    for _ in range(5):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_harmful', 's1')")
    for _ in range(5):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_useful', 's1')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        boost = retrieval.get_adaptive_threshold_boost()

    # 50% is NOT > 0.5, so falls to the > 0.3 branch => 0.05
    assert boost == 0.05, f"Expected 0.05 at exactly 50% harmful rate, got {boost}"
    conn.close()


# ============================================================
# retrieve_context — 4 tests
# ============================================================

#TAG: [D8AC] 2026-04-05
# Verifies: returns XML with both project and global scopes containing correct entries
@pytest.mark.behavioural
def test_retrieve_context_project_and_global():
    db_path, conn = fresh_db()
    seed_project_session(conn, "s1", "ProjA")
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("fact", "auth", "JWT authentication tokens", make_blob(100), "ProjA", 0.8, "other-session"))
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("fact", "deploy", "Deploy to production server", make_blob(200), "ProjB", 0.7, "other-session"))
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [
        {"id": 1, "type": "fact", "topic": "auth", "content": "JWT authentication tokens",
         "similarity": 0.60, "confidence": 0.8, "score": 0.7,
         "updated_at": "2026-03-20 12:00:00", "project": "ProjA", "session_id": "other-session"},
        {"id": 2, "type": "fact", "topic": "deploy", "content": "Deploy to production server",
         "similarity": 0.55, "confidence": 0.7, "score": 0.5,
         "updated_at": "2026-03-20 12:00:00", "project": "ProjB", "session_id": "other-session"},
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context("authentication and deployment", session_id="s1")

    # Verify overall XML structure
    assert result[:len('<cairn_context')] == '<cairn_context'
    assert result[-len('</cairn_context>'):] == '</cairn_context>'
    # Verify both scopes present with correct structure
    project_scope = re.search(r'<scope level="project" name="ProjA" weight="high">(.*?)</scope>', result, re.DOTALL)
    global_scope = re.search(r'<scope level="global" weight="low">(.*?)</scope>', result, re.DOTALL)
    # Verify entries are in correct scopes — extract id attribute and compare exactly
    proj_entry = re.search(r'<entry id="(\d+)"', project_scope.group(1))
    assert proj_entry is not None and proj_entry.group(1) == "1", \
        f"Expected project scope to contain entry id=\"1\"; got: {project_scope.group(1)!r}"
    global_entry = re.search(r'<entry id="(\d+)"', global_scope.group(1))
    assert global_entry is not None and global_entry.group(1) == "2", \
        f"Expected global scope to contain entry id=\"2\"; got: {global_scope.group(1)!r}"
    conn.close()


#TAG: [4CBE] 2026-04-05
# Verifies: returns None and records context_empty metric when no results match
@pytest.mark.edge
def test_retrieve_context_no_results_returns_none():
    db_path, conn = fresh_db()
    seed_project_session(conn, "s1", "ProjA")

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = []

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch('hooks.retrieval.record_metric') as mock_metric, \
         patch('hooks.retrieval.log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context("zzz nonexistent gibberish", session_id="s1")

    assert result is None
    # Verify context_empty metric recorded with the correct detail
    metric_calls = [c for c in mock_metric.call_args_list if c[0][1] == "context_empty"]
    assert len(metric_calls) == 1
    assert metric_calls[0][0][2] == "zzz nonexistent gibberish"
    conn.close()


#TAG: [7C75] 2026-04-05
# Verifies: embedding ConnectionError is caught and FTS fallback returns correct entry
@pytest.mark.error
def test_retrieve_context_embedding_error_fts_fallback():
    db_path, conn = fresh_db()
    seed_project_session(conn, "s1", "ProjA")
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("fact", "database", "SQLite database configuration", make_blob(100), "ProjA", 0.8, "other-session"))
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.find_similar.side_effect = ConnectionError("daemon down")

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context("SQLite database", session_id="s1")

    # FTS should pick up the "SQLite" and "database" terms and return valid XML
    assert result[:len('<cairn_context')] == '<cairn_context'
    assert result[-len('</cairn_context>'):] == '</cairn_context>'
    # Verify the FTS-found entry appears in a scope block with correct id and content
    entry_match = re.search(r'<entry[^>]*id="1"[^>]*>([^<]+)</entry>', result)
    assert entry_match.group(1) == "SQLite database configuration"
    conn.close()


#TAG: [29DD] 2026-04-05
# Verifies: context_need of only stopwords does not crash and produces valid output or None
@pytest.mark.adversarial
def test_retrieve_context_only_stopwords():
    db_path, conn = fresh_db()
    seed_project_session(conn, "s1", "ProjA")
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("fact", "misc", "the and or but", make_blob(100), "ProjA", 0.7, "other-session"))
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = []

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch('hooks.retrieval.record_metric') as mock_metric, \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context("the and or but", session_id="s1")

    # All words are stopwords; the code falls through to len>2 filter then raw words.
    # Either we get None (FTS didn't match) or valid XML. No crash.
    if result is None:
        # Verify empty metric was recorded
        empty_calls = [c for c in mock_metric.call_args_list if c[0][1] == "context_empty"]
        assert len(empty_calls) == 1
    else:
        assert result[:len('<cairn_context')] == '<cairn_context'
        assert result[-len('</cairn_context>'):] == '</cairn_context>'
    conn.close()


# ============================================================
# layer2_cross_project_search — 4 tests
# ============================================================

#TAG: [3E69] 2026-04-05
# Verifies: stages cross-project results into hook_state with correct XML structure and entry data
@pytest.mark.behavioural
def test_layer2_cross_project_search_stages_results():
    db_path, conn = fresh_db()
    seed_project_session(conn, "s1", "ProjA")

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [{
        "id": 10, "type": "fact", "topic": "cross-info",
        "content": "Cross-project useful info",
        "similarity": 0.75, "confidence": 0.8, "score": 0.7,
        "updated_at": "2026-03-20 12:00:00", "project": "ProjB",
    }]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'):
        retrieval.layer2_cross_project_search(["authentication", "JWT"], session_id="s1")

    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 's1' AND key = 'staged_context'"
    ).fetchone()
    xml = row[0]
    # Verify XML structure
    assert xml[:len('<cairn_context')] == '<cairn_context'
    assert parse_xml_attr(xml, 'layer') == 'cross-project'
    # Verify entry attributes match input data
    entry_match = re.search(r'<entry ([^>]*)>([^<]+)</entry>', xml)
    entry_attrs = entry_match.group(1)
    assert parse_xml_attr(entry_attrs, 'id') == '10'
    assert parse_xml_attr(entry_attrs, 'confidence') == '0.80'
    assert parse_xml_attr(entry_attrs, 'project') == 'ProjB'
    assert entry_match.group(2) == "Cross-project useful info"
    conn.close()


#TAG: [FE9B] 2026-04-05
# Verifies: empty keywords list returns immediately without calling get_embedder
@pytest.mark.edge
def test_layer2_cross_project_search_empty_keywords():
    mock_emb = MagicMock()
    with patch.object(hook_helpers, 'get_embedder', return_value=mock_emb) as mock_get:
        retrieval.layer2_cross_project_search([], session_id="s1")

    mock_get.assert_not_called()


#TAG: [C1B3] 2026-04-05
# Verifies: find_similar exception is caught, no context staged, no crash
@pytest.mark.error
def test_layer2_cross_project_search_search_error():
    db_path, conn = fresh_db()
    seed_project_session(conn, "s1", "ProjA")

    mock_emb = MagicMock()
    mock_emb.find_similar.side_effect = RuntimeError("embedding service down")

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'log'):
        retrieval.layer2_cross_project_search(["test", "keywords"], session_id="s1")

    # Verify no staged context written AND no other hook_state keys leaked
    all_rows = conn.execute("SELECT * FROM hook_state").fetchall()
    assert len(all_rows) == 0, f"Expected empty hook_state after error, got {len(all_rows)} rows"
    conn.close()


#TAG: [3684] 2026-04-05
# Verifies: all same-project results filtered out, nothing staged
@pytest.mark.adversarial
def test_layer2_cross_project_search_all_same_project():
    db_path, conn = fresh_db()
    seed_project_session(conn, "s1", "ProjA")

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [
        {"id": 1, "type": "fact", "topic": "same", "content": "Same project",
         "similarity": 0.8, "confidence": 0.9, "score": 0.8,
         "updated_at": "2026-03-20 12:00:00", "project": "ProjA"},
        {"id": 2, "type": "fact", "topic": "also-same", "content": "Also same",
         "similarity": 0.7, "confidence": 0.8, "score": 0.7,
         "updated_at": "2026-03-20 12:00:00", "project": "ProjA"},
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'):
        retrieval.layer2_cross_project_search(["test"], session_id="s1")

    all_rows = conn.execute("SELECT * FROM hook_state").fetchall()
    assert len(all_rows) == 0, f"Expected no staged context for same-project results, got {len(all_rows)} rows"
    conn.close()


# ============================================================
# is_context_cached — 4 tests
# ============================================================

#TAG: [3C92] 2026-04-05
# Verifies: returns True when embedding similarity meets CONTEXT_CACHE_SIM_THRESHOLD
@pytest.mark.behavioural
def test_is_context_cached_semantic_match():
    mock_emb = MagicMock()
    query_vec = make_vector(100)
    cached_vec = make_vector(100)
    mock_emb.embed.return_value = query_vec
    mock_emb.cosine_similarity.return_value = 0.95
    mock_emb.to_blob = lambda v: v.tobytes()

    served = [{"text": "old query", "embedding_hex": cached_vec.tobytes().hex()}]
    result = retrieval.is_context_cached("new query", served, mock_emb)

    assert result is True


#TAG: [7DA2] 2026-04-05
# Verifies: returns False when served_needs is empty and no embedder
@pytest.mark.edge
def test_is_context_cached_empty_no_embedder():
    result = retrieval.is_context_cached("anything", [], None)
    assert result is False


#TAG: [2C72] 2026-04-05
# Verifies: embedder exception falls back to exact text match returning True for match and False for mismatch
@pytest.mark.error
def test_is_context_cached_embed_error_falls_back():
    mock_emb = MagicMock()
    mock_emb.embed.side_effect = RuntimeError("model unavailable")

    served = [{"text": "exact match query"}]
    result_match = retrieval.is_context_cached("exact match query", served, mock_emb)
    result_nomatch = retrieval.is_context_cached("different query", served, mock_emb)

    assert result_match is True, "Should fall back to exact text match"
    assert result_nomatch is False, "Should not match different text"


#TAG: [5A04] 2026-04-05
# Verifies: missing embedding_hex key triggers fallback to text match
@pytest.mark.adversarial
def test_is_context_cached_missing_embedding_hex():
    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(100)

    served = [{"text": "some query"}]  # No embedding_hex key
    result_match = retrieval.is_context_cached("some query", served, mock_emb)
    result_nomatch = retrieval.is_context_cached("other query", served, mock_emb)

    # Falls back to text match due to KeyError on cached["embedding_hex"]
    assert result_match is True
    assert result_nomatch is False


# ============================================================
# load_context_cache — 1 test
# ============================================================

#TAG: [B573] 2026-04-05
# Verifies: returns parsed JSON list with correct structure from hook_state
@pytest.mark.behavioural
def test_load_context_cache_returns_parsed():
    db_path, conn = fresh_db()
    data = [{"text": "cached query", "embedding_hex": "aabb"}]
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'context_cache', ?)",
        ("s1", json.dumps(data)))
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = retrieval.load_context_cache("s1")

    assert len(result) == 1
    assert result[0]["text"] == "cached query"
    assert result[0]["embedding_hex"] == "aabb"
    conn.close()


# ============================================================
# save_context_cache — 1 test
# ============================================================

#TAG: [436C] 2026-04-05
# Verifies: persists served_needs to hook_state and round-trips correctly
@pytest.mark.behavioural
def test_save_context_cache_persists():
    db_path, conn = fresh_db()

    data = [{"text": "saved query"}]
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        retrieval.save_context_cache("s1", data)

    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 's1' AND key = 'context_cache'"
    ).fetchone()
    parsed = json.loads(row[0])
    assert len(parsed) == 1
    assert parsed[0]["text"] == "saved query"
    conn.close()


# ============================================================
# add_to_context_cache — 2 tests
# ============================================================

#TAG: [0D08] 2026-04-05
# Verifies: appends entry with embedding_hex whose length matches 384-dim float32 vector
@pytest.mark.behavioural
def test_add_to_context_cache_with_embedder():
    mock_emb = MagicMock()
    vec = make_vector(42)
    mock_emb.embed.return_value = vec
    mock_emb.to_blob.return_value = vec.tobytes()

    served = []
    result = retrieval.add_to_context_cache("new query", served, mock_emb)

    assert len(result) == 1
    assert result[0]["text"] == "new query"
    # 384 floats * 4 bytes = 1536 bytes = 3072 hex chars
    assert len(result[0]["embedding_hex"]) == 3072


#TAG: [1B62] 2026-04-05
# Verifies: appends text-only entry without embedding_hex when embedder is None
@pytest.mark.edge
def test_add_to_context_cache_without_embedder():
    served = [{"text": "existing"}]
    result = retrieval.add_to_context_cache("another query", served, None)

    assert len(result) == 2
    assert result[1]["text"] == "another query"
    assert set(result[1].keys()) == {"text"}


# ============================================================
# add_to_context_cache — error + adversarial
# ============================================================

#TAG: [701D] 2026-04-05
# Verifies: embed() exception is caught and entry is stored with only text key, no embedding_hex
@pytest.mark.error
def test_add_to_context_cache_embed_error():
    mock_emb = MagicMock()
    mock_emb.embed.side_effect = RuntimeError("model load failed")

    served = []
    result = retrieval.add_to_context_cache("crash query", served, mock_emb)

    assert len(result) == 1
    assert result[0]["text"] == "crash query"
    assert set(result[0].keys()) == {"text"}


#TAG: [D848] 2026-04-05
# Verifies: to_blob() exception is caught and entry is stored with only text key, no embedding_hex
@pytest.mark.adversarial
def test_add_to_context_cache_to_blob_error():
    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(99)
    mock_emb.to_blob.side_effect = ValueError("cannot serialize")

    served = []
    result = retrieval.add_to_context_cache("blob fail query", served, mock_emb)

    assert len(result) == 1
    assert result[0]["text"] == "blob fail query"
    assert set(result[0].keys()) == {"text"}


# ============================================================
# load_context_cache — edge, error, adversarial
# ============================================================

#TAG: [94F7] 2026-04-05
# Verifies: returns empty list when no cache entry exists for the session
@pytest.mark.edge
def test_load_context_cache_missing_session():
    db_path, conn = fresh_db()
    # No hook_state rows inserted

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = retrieval.load_context_cache("nonexistent-session")

    assert result == []
    conn.close()


#TAG: [5CFC] 2026-04-05
# Verifies: returns empty list when stored value is corrupt JSON (JSONDecodeError path)
@pytest.mark.error
def test_load_context_cache_corrupt_json():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'context_cache', ?)",
        ("s1", "THIS IS NOT JSON {{{"))
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = retrieval.load_context_cache("s1")

    assert result == []
    conn.close()


#TAG: [D398] 2026-04-05
# Verifies: returns empty list when hook_state row has NULL value (falsy guard path)
@pytest.mark.adversarial
def test_load_context_cache_null_value():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'context_cache', NULL)",
        ("s1",))
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = retrieval.load_context_cache("s1")

    assert result == []
    conn.close()


# ============================================================
# save_context_cache — edge, error, adversarial
# ============================================================

#TAG: [AC01] 2026-04-05
# Verifies: empty list is serialised as JSON array and round-trips back to empty list
@pytest.mark.edge
def test_save_context_cache_empty_list():
    db_path, conn = fresh_db()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        retrieval.save_context_cache("s1", [])

    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 's1' AND key = 'context_cache'"
    ).fetchone()
    assert row[0] == "[]"
    conn.close()


#TAG: [B450] 2026-04-05
# Verifies: INSERT OR REPLACE overwrites an existing cache entry with the new value
@pytest.mark.error
def test_save_context_cache_overwrites_existing():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES ('s1', 'context_cache', '[{\"text\":\"old\"}]')")
    conn.commit()

    new_data = [{"text": "new-entry"}]
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        retrieval.save_context_cache("s1", new_data)

    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 's1' AND key = 'context_cache'"
    ).fetchone()
    import json as _json
    parsed = _json.loads(row[0])
    assert len(parsed) == 1
    assert parsed[0]["text"] == "new-entry"
    conn.close()


#TAG: [FD27] 2026-04-05
# Verifies: data containing quotes and unicode round-trips correctly through JSON serialisation
@pytest.mark.adversarial
def test_save_context_cache_special_chars_roundtrip():
    db_path, conn = fresh_db()
    data = [{"text": 'query with "quotes" and \u00e9 unicode'}]

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        retrieval.save_context_cache("s1", data)

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = retrieval.load_context_cache("s1")

    assert len(result) == 1
    assert result[0]["text"] == 'query with "quotes" and \u00e9 unicode'
    conn.close()
