#!/usr/bin/env python3
"""Tests for retrieve_context in hooks/retrieval.py — RRF fusion focus.

Covers: dual-match RRF boost, FTS-only composite_score path, same-session
exclusion across both ranking methods, and max_per_scope cap.

Uses real FTS5 via in-memory SQLite; only the embedder is mocked.
"""

import sys
import os
import re
import sqlite3
import tempfile

import pytest
from unittest.mock import patch, MagicMock


import hooks.hook_helpers as hook_helpers
import hooks.retrieval as retrieval
import cairn.embeddings as embeddings_mod

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_db():
    """Create a fresh file-based SQLite DB with memories, FTS5, and all required tables."""
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"rrf2_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        embedding BLOB, session_id TEXT, project TEXT,
        confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER,
        depth INTEGER, archived_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content)
        VALUES (new.id, new.topic, new.content);
    END""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


def insert_memory(conn, content="test content", project=None,
                  session_id="other-session", confidence=0.7, topic="test"):
    """Insert a memory and return its auto-assigned id."""
    conn.execute(
        "INSERT INTO memories "
        "(type, topic, content, project, session_id, confidence, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
        ("fact", topic, content, project, session_id, confidence)
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def entry_ids_in_order(xml: str) -> list:
    """Return all <entry id="N"> IDs in document order as integers."""
    return [int(m) for m in re.findall(r'<entry[^>]*\bid="(\d+)"', xml)]


# ---------------------------------------------------------------------------
# retrieve_context — 4 tests (behavioural, edge, error, adversarial)
# ---------------------------------------------------------------------------

#TAG: [4C45] 2026-04-05
# Verifies: dual-match memory found by both semantic and FTS has a higher RRF fused score and appears before a semantic-only memory with the same baseline score
@pytest.mark.behavioural
def test_retrieve_context_rrf_dual_match_outranks_single_method():
    """Memory appearing in both semantic and FTS ranked lists must outrank semantic-only
    when both start with the same baseline composite score."""
    db_path, conn = fresh_db()
    # id=1: content contains "deployment pipeline" — will match FTS AND be in semantic results
    id_dual = insert_memory(conn,
                            content="deployment pipeline configuration guide",
                            session_id="other")
    # id=2: content is noise — will NOT match FTS, only in semantic results
    id_single = insert_memory(conn,
                              content="xyzzy noop unrelated placeholder entry",
                              session_id="other")
    assert id_dual == 1 and id_single == 2

    mock_emb = MagicMock()
    # Both returned by semantic with identical baseline scores (so RRF is the only differentiator)
    mock_emb.find_similar.return_value = [
        {"id": 1, "type": "fact", "topic": "test",
         "content": "deployment pipeline configuration guide",
         "similarity": 0.30, "confidence": 0.7, "score": 0.40,
         "updated_at": "2026-01-01 12:00:00", "project": None, "session_id": "other",
         "depth": None, "archived_reason": None},
        {"id": 2, "type": "fact", "topic": "test",
         "content": "xyzzy noop unrelated placeholder entry",
         "similarity": 0.30, "confidence": 0.7, "score": 0.40,
         "updated_at": "2026-01-01 12:00:00", "project": None, "session_id": "other",
         "depth": None, "archived_reason": None},
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context("deployment pipeline", session_id=None)

    assert result is not None, "Expected XML output; got None"
    ids = entry_ids_in_order(result)
    assert len(ids) == 2, f"Expected 2 entries, got {len(ids)}: {ids}"
    # Dual-match (id=1) must be first — it received RRF contribution from both methods
    assert ids[0] == 1, (
        f"Dual-match id=1 must outrank semantic-only id=2; got order {ids}"
    )
    assert ids[1] == 2, (
        f"Semantic-only id=2 must be second; got order {ids}"
    )


#TAG: [2253] 2026-04-05
# Verifies: when session_id is provided, memories with matching session_id are excluded from BOTH semantic_ranked and fts_ranked loops so they never reach the fused result set
@pytest.mark.edge
def test_retrieve_context_same_session_excluded_from_both_paths():
    """A memory whose session_id matches the caller must be suppressed in both the
    semantic loop and the FTS loop — never appearing in fused output."""
    db_path, conn = fresh_db()
    # id=1 belongs to the calling session — both loops must filter it
    id_same = insert_memory(conn,
                            content="keyword phrase retrieval exclusion test",
                            session_id="active-session")
    # id=2 belongs to a different session — must appear in output
    id_other = insert_memory(conn,
                             content="keyword phrase retrieval exclusion test",
                             session_id="other-session")
    assert id_same == 1 and id_other == 2

    mock_emb = MagicMock()
    # Semantic returns both; the session filter inside retrieve_context must exclude id=1
    mock_emb.find_similar.return_value = [
        {"id": 1, "type": "fact", "topic": "test",
         "content": "keyword phrase retrieval exclusion test",
         "similarity": 0.40, "confidence": 0.7, "score": 0.50,
         "updated_at": "2026-01-01 12:00:00", "project": None,
         "session_id": "active-session", "depth": None, "archived_reason": None},
        {"id": 2, "type": "fact", "topic": "test",
         "content": "keyword phrase retrieval exclusion test",
         "similarity": 0.40, "confidence": 0.7, "score": 0.50,
         "updated_at": "2026-01-01 12:00:00", "project": None,
         "session_id": "other-session", "depth": None, "archived_reason": None},
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context(
            "keyword phrase retrieval", session_id="active-session")

    # FTS also matches both — id=1 must be absent from fused output
    ids = entry_ids_in_order(result)
    assert ids == [2], (
        f"Expected exactly [2] (id=1 same-session excluded from both loops); got {ids}"
    )


#TAG: [56AE] 2026-04-05
# Verifies: FTS-only code path (semantic raises ConnectionError) calls embeddings.composite_score with fts_sim=0.35 — not the hardcoded 0.30 fallback — verified by call count and first positional argument
@pytest.mark.error
def test_retrieve_context_fts_only_uses_composite_score_with_correct_sim():
    """When semantic search raises ConnectionError, the FTS-only branch must call
    composite_score(fts_sim=0.35, ...) rather than using a hardcoded 0.30 score."""
    db_path, conn = fresh_db()
    insert_memory(conn, content="sqlite configuration storage backend",
                  session_id="other")
    conn.close()

    mock_emb = MagicMock()
    mock_emb.find_similar.side_effect = ConnectionError("embedding unavailable")

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0), \
         patch.object(embeddings_mod, 'composite_score', return_value=0.77) as mock_cs:
        result = retrieval.retrieve_context("sqlite storage", session_id=None)

    # composite_score must be called exactly once for the FTS-only entry
    assert mock_cs.call_count == 1, (
        f"Expected composite_score called once; got {mock_cs.call_count}"
    )
    # First positional arg must be 0.35 (the FTS baseline similarity constant)
    first_arg = mock_cs.call_args[0][0]
    assert first_arg == pytest.approx(0.35), (
        f"Expected composite_score(0.35, ...) but first arg was {first_arg!r}"
    )
    # With composite_score=0.77 the fused score ≈ 0.87 → reliability="strong".
    # If the 0.30 hardcode were used the score ≈ 0.40 → "moderate".
    reliability_values = re.findall(r'reliability="([^"]+)"', result)
    assert reliability_values == ["strong"], (
        f"Expected ['strong'] reliability (score≈0.87 from composite_score=0.77); "
        f"got {reliability_values!r} — suggests 0.30 fallback was used"
    )


#TAG: [33FE] 2026-04-05
# Verifies: max_per_scope=N hard-caps XML output to exactly N entries per scope even when many FTS results pass the global similarity threshold
@pytest.mark.adversarial
def test_retrieve_context_max_per_scope_caps_fts_output_exactly():
    """max_per_scope=2 must limit the XML output to exactly 2 entries even when
    5 FTS results all pass the global similarity threshold."""
    db_path, conn = fresh_db()
    # Insert 5 memories matching FTS for "sqlite database"
    for i in range(5):
        insert_memory(conn,
                      content=f"sqlite database storage configuration item {i + 1}",
                      session_id="other")
    conn.close()

    # No semantic results — all 5 go through FTS-only path
    mock_emb = MagicMock()
    mock_emb.find_similar.side_effect = ConnectionError("no semantic")

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context(
            "sqlite database", session_id=None, max_per_scope=2)

    assert result is not None, "Expected non-None XML output"
    ids = entry_ids_in_order(result)
    # Hard cap: exactly 2 entries regardless of how many pass threshold
    assert len(ids) == 2, (
        f"max_per_scope=2 must yield exactly 2 entries; got {len(ids)}: {ids}"
    )
    # Top 2 by FTS rank — verify they are from the first 5 inserted (ids 1-5)
    assert all(1 <= i <= 5 for i in ids), (
        f"Expected ids in range [1-5]; got {ids}"
    )
