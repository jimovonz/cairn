#!/usr/bin/env python3
"""Tests for retrieve_context RRF fusion logic in hooks/retrieval.py.

Focused on: dual-match RRF boost, same-session exclusion across both ranking
methods, FTS-only composite_score usage, and max_per_scope cap.

Uses real FTS5 SQLite (file-based), mocked embedder only — per reviewer hint.
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
import cairn.embeddings as embeddings_mod  # imported here so patch.object can target it

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_db():
    """Create a fresh file-based SQLite DB with all tables and FTS5 trigger."""
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"rrf_{_counter[0]}.db")
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
                  session_id="other-session", confidence=0.7):
    """Insert a memory and return its auto-assigned id."""
    conn.execute(
        "INSERT INTO memories "
        "(type, topic, content, project, session_id, confidence, updated_at) "
        "VALUES (?, 'test', ?, ?, ?, ?, datetime('now'))",
        ("fact", content, project, session_id, confidence)
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def extract_entry_ids(xml: str) -> list:
    """Return all entry IDs in document order as integers."""
    return [int(m) for m in re.findall(r'<entry[^>]*\bid="(\d+)"', xml)]


# ---------------------------------------------------------------------------
# retrieve_context — 4 RRF-focused tests
# ---------------------------------------------------------------------------

#TAG: [8615] 2026-04-05
# Verifies: dual-match memory found by both semantic and FTS scores higher via RRF than semantic-only match with same baseline score appearing first in output
@pytest.mark.behavioural
def test_rrf_dual_match_scores_higher_than_single():
    db_path, conn = fresh_db()
    # id=1: content matches FTS query "deployment pipeline" → will be in both ranked lists
    id_a = insert_memory(conn, content="deployment pipeline configuration", session_id="other")
    # id=2: content does NOT match FTS → semantic-only
    id_b = insert_memory(conn, content="xyzzy noop placeholder abcdef", session_id="other")
    assert id_a == 1 and id_b == 2

    mock_emb = MagicMock()
    # Both memories returned by semantic with identical baseline scores
    mock_emb.find_similar.return_value = [
        {"id": 1, "type": "fact", "topic": "test",
         "content": "deployment pipeline configuration",
         "similarity": 0.30, "confidence": 0.7, "score": 0.40,
         "updated_at": "2026-04-01 12:00:00", "project": None, "session_id": "other"},
        {"id": 2, "type": "fact", "topic": "test",
         "content": "xyzzy noop placeholder abcdef",
         "similarity": 0.30, "confidence": 0.7, "score": 0.40,
         "updated_at": "2026-04-01 12:00:00", "project": None, "session_id": "other"},
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context("deployment pipeline", session_id=None)

    # Both entries must appear — verify count and structure
    ids_in_order = extract_entry_ids(result)
    assert len(ids_in_order) == 2, (
        f"Expected 2 entries in output, got {len(ids_in_order)}. XML:\n{result}"
    )
    # Dual-match (id=1) must rank above semantic-only (id=2) due to higher RRF contribution
    assert ids_in_order[0] == 1, (
        f"Dual-match id=1 should be first (higher RRF score); got order {ids_in_order}"
    )
    assert ids_in_order[1] == 2, (
        f"Semantic-only id=2 should be second; got order {ids_in_order}"
    )
    conn.close()


#TAG: [C6E0] 2026-04-05
# Verifies: same-session memories are excluded independently by both semantic and FTS ranking loops — neither leaks same-session entries into fused result set
@pytest.mark.edge
def test_same_session_excluded_from_both_ranking_methods():
    db_path, conn = fresh_db()
    # id=1: SAME session as caller — must be excluded from both semantic and FTS ranked lists
    id_same = insert_memory(
        conn, content="keyword phrase retrieval test", session_id="caller-session")
    # id=2: different session — must appear in final output
    id_other = insert_memory(
        conn, content="keyword phrase retrieval test", session_id="other-session")
    assert id_same == 1 and id_other == 2

    mock_emb = MagicMock()
    # Semantic mock returns BOTH; the session filter inside retrieve_context excludes id=1
    mock_emb.find_similar.return_value = [
        {"id": 1, "type": "fact", "topic": "test",
         "content": "keyword phrase retrieval test",
         "similarity": 0.40, "confidence": 0.7, "score": 0.50,
         "updated_at": "2026-04-01 12:00:00", "project": None,
         "session_id": "caller-session"},
        {"id": 2, "type": "fact", "topic": "test",
         "content": "keyword phrase retrieval test",
         "similarity": 0.40, "confidence": 0.7, "score": 0.50,
         "updated_at": "2026-04-01 12:00:00", "project": None,
         "session_id": "other-session"},
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context(
            "keyword phrase retrieval", session_id="caller-session")

    # FTS also finds both memories; id=1 (session_id matches caller) must be excluded
    ids = extract_entry_ids(result)
    assert ids == [2], (
        f"Expected exactly [2] in output (id=1 same-session excluded); got {ids}"
    )


#TAG: [08FE] 2026-04-05
# Verifies: FTS-only code path calls composite_score(fts_sim=0.35) not the 0.30 default — distinguished by reliability=strong vs moderate in output XML
@pytest.mark.error
def test_fts_only_calls_composite_score_not_hardcoded_baseline():
    db_path, conn = fresh_db()
    insert_memory(conn, content="sqlite configuration storage", session_id="other")
    conn.close()

    # Semantic unavailable → semantic_ranked empty → FTS-only code path runs
    mock_emb = MagicMock()
    mock_emb.find_similar.side_effect = ConnectionError("embedding service down")

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0), \
         patch.object(embeddings_mod, 'composite_score', return_value=0.77) as mock_cscore:
        result = retrieval.retrieve_context("sqlite storage", session_id=None)

    # composite_score must have been called exactly once (not bypassed with 0.30 fallback)
    assert mock_cscore.call_count == 1, (
        f"composite_score call_count={mock_cscore.call_count}; expected 1 — FTS-only path may have used 0.30 fallback"
    )
    # First positional arg must be fts_sim = 0.35 (the FTS baseline similarity)
    first_arg = mock_cscore.call_args[0][0]
    assert first_arg == pytest.approx(0.35), (
        f"composite_score called with sim={first_arg!r}, expected 0.35"
    )
    # With composite_score=0.77: final score ≈ 0.77 + rrf_boost(≈0.10) = 0.87 → "strong"
    # With 0.30 fallback:        final score ≈ 0.30 + rrf_boost(≈0.10) = 0.40 → "moderate"
    # Verify reliability attribute is exactly "strong" (not moderate/weak)
    reliability_values = re.findall(r'reliability="([^"]+)"', result)
    assert len(reliability_values) == 1, (
        f"Expected exactly 1 entry with reliability attribute, got {len(reliability_values)}"
    )
    assert reliability_values[0] == "strong", (
        f"Expected reliability=strong (score≈0.87 from composite_score=0.77); "
        f"got {reliability_values[0]!r}, suggesting 0.30 fallback was used instead."
    )


#TAG: [EC13] 2026-04-05
# Verifies: max_per_scope=N hard-caps output entries per scope to exactly N even when more results pass similarity thresholds
@pytest.mark.adversarial
def test_max_per_scope_caps_output_entries():
    db_path, conn = fresh_db()
    # Insert 5 global memories all matching FTS for "sqlite database"
    for i in range(5):
        insert_memory(
            conn,
            content=f"sqlite database storage configuration item {i + 1}",
            session_id="other",
        )
    conn.close()

    # No semantic (ConnectionError) → all 5 are FTS-only results
    mock_emb = MagicMock()
    mock_emb.find_similar.side_effect = ConnectionError("no embedder")

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'), \
         patch.object(retrieval, 'get_adaptive_threshold_boost', return_value=0.0):
        result = retrieval.retrieve_context(
            "sqlite database", session_id=None, max_per_scope=2)

    # All 5 entries pass the global similarity threshold (sim=0.35 >= 0.25),
    # but max_per_scope=2 must cap the XML output to exactly 2 entries
    ids = extract_entry_ids(result)
    assert len(ids) == 2, (
        f"max_per_scope=2 should yield exactly 2 entries; got {len(ids)}: {ids}"
    )
    # Verify these are the top 2 by score (ids 1 and 2, FTS rank order)
    assert set(ids) == {1, 2}, (
        f"Expected ids {{1, 2}} (top 2 by FTS rank); got {ids}"
    )
