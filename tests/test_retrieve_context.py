#!/usr/bin/env python3
"""Tests for retrieve_context in hooks/retrieval.py — RRF fusion, thresholds, XML output."""

import sys
import os
import sqlite3
import tempfile
import re

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

import hook_helpers
import retrieval
from config import RRF_K

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def make_vector(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


def make_blob(seed):
    return make_vector(seed).tobytes()


def fresh_db():
    """Create a fresh SQLite DB with all required tables and FTS5."""
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"ret_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER,
        archived_reason TEXT, associated_files TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_event ON metrics(event)")
    conn.execute("""CREATE TABLE IF NOT EXISTS hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content)
        VALUES (new.id, new.topic, new.content); END""")
    conn.commit()
    return db_path, conn


def insert_memory(conn, type_="fact", topic="test", content="test content",
                  project=None, confidence=0.7, session_id="other", depth=None,
                  archived_reason=None, updated_at=None):
    """Insert a memory into the test DB and return its rowid."""
    if updated_at:
        conn.execute(
            "INSERT INTO memories (type, topic, content, embedding, project, confidence, "
            "session_id, depth, archived_reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (type_, topic, content, make_blob(abs(hash(content)) % 10000),
             project, confidence, session_id, depth, archived_reason, updated_at)
        )
    else:
        conn.execute(
            "INSERT INTO memories (type, topic, content, embedding, project, confidence, "
            "session_id, depth, archived_reason) VALUES (?,?,?,?,?,?,?,?,?)",
            (type_, topic, content, make_blob(abs(hash(content)) % 10000),
             project, confidence, session_id, depth, archived_reason)
        )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def make_semantic_result(mid, type_="fact", topic="test", content="test",
                         project=None, confidence=0.7, session_id="other",
                         similarity=0.65, score=0.55, depth=None,
                         archived_reason=None, updated_at="2026-04-01 10:00:00"):
    """Build a result dict matching the format returned by emb.find_similar."""
    return {
        "id": mid, "type": type_, "topic": topic, "content": content,
        "updated_at": updated_at, "project": project, "confidence": confidence,
        "session_id": session_id, "depth": depth, "archived_reason": archived_reason,
        "similarity": similarity, "score": score,
    }


# ============================================================
# Behavioural
# ============================================================

#TAG: [R1A0]
# Verifies: RRF dual-match (semantic+FTS) produces higher fused score than single-match (FTS-only)
@pytest.mark.behavioural
def test_rrf_dual_match_ranks_higher_than_single():
    """Memory found by both semantic and FTS should appear before FTS-only match."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    # Memory 1: will be returned by both semantic (mocked) and FTS (keyword "JWT" + "authentication")
    m1 = insert_memory(conn, type_="decision", topic="auth-jwt",
                       content="Use JWT tokens for stateless authentication",
                       project="proj", confidence=0.8, session_id="other")
    # Memory 2: will only match FTS (keyword "authentication") — not returned by semantic
    m2 = insert_memory(conn, type_="fact", topic="auth-sessions",
                       content="Session authentication was deprecated last quarter",
                       project="proj", confidence=0.7, session_id="other")

    mock_emb = MagicMock()
    # Semantic returns only memory 1 (both calls return the same for simplicity)
    mock_emb.find_similar.return_value = [
        make_semantic_result(m1, type_="decision", topic="auth-jwt",
                             content="Use JWT tokens for stateless authentication",
                             project="proj", confidence=0.8, similarity=0.65, score=0.55),
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("JWT authentication tokens", session_id="s1")

    # Both memories should appear. Dual-match (memory 1) should come first in output.
    # Extract all entry ids in document order to verify ranking
    entry_ids = [int(m.group(1)) for m in re.finditer(r'<entry id="(\d+)"', result)]
    assert m1 in entry_ids, f"Dual-match memory id={m1} must appear in output"
    assert m2 in entry_ids, f"Single-match memory id={m2} must appear in output"
    assert entry_ids.index(m1) < entry_ids.index(m2), (
        f"Dual-match id={m1} should rank before single-match id={m2}, got order: {entry_ids}"
    )


#TAG: [R1B0]
# Verifies: XML output contains correct structure — cairn_context root, instruction, project and global scopes
@pytest.mark.behavioural
def test_xml_output_structure_with_project_and_global():
    """retrieve_context should produce well-formed XML with both scopes when applicable."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'myproj')")
    # Project-scoped memory
    m1 = insert_memory(conn, type_="fact", topic="proj-fact",
                       content="Project-local fact about database schema",
                       project="myproj", confidence=0.8, session_id="other", depth=3)
    # Global memory (different project)
    m2 = insert_memory(conn, type_="skill", topic="global-skill",
                       content="Global skill about database migration patterns",
                       project="otherproj", confidence=0.6, session_id="other")

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [
        make_semantic_result(m1, type_="fact", topic="proj-fact",
                             content="Project-local fact about database schema",
                             project="myproj", confidence=0.8, similarity=0.70, score=0.60,
                             depth=3),
        make_semantic_result(m2, type_="skill", topic="global-skill",
                             content="Global skill about database migration patterns",
                             project="otherproj", confidence=0.6, similarity=0.55, score=0.45),
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("database schema migration", session_id="s1")

    # Verify complete XML structure via line-by-line parse
    lines = result.strip().split("\n")
    assert lines[0] == '<cairn_context query="database schema migration" current_project="myproj">'
    assert lines[-1] == "</cairn_context>"
    # Exactly 1 instruction element
    assert sum(1 for l in lines if "<instruction>" in l) == 1
    # Exactly 1 project scope and 1 global scope
    assert sum(1 for l in lines if 'level="project"' in l) == 1
    assert sum(1 for l in lines if 'level="global"' in l) == 1
    # Project scope references the correct project name and weight
    proj_scope_line = [l for l in lines if 'level="project"' in l][0]
    scope_attrs = re.search(r'<scope level="project" name="(\w+)" weight="(\w+)">', proj_scope_line)
    assert scope_attrs.group(1) == "myproj"
    assert scope_attrs.group(2) == "high"
    # Entry with depth=3 should have ctx="y" attribute — verify exactly once
    entry_with_ctx = [l for l in lines if f'id="{m1}"' in l][0]
    assert entry_with_ctx.count('ctx="y"') == 1
    # Reliability label must be "strong" for score 0.60 — verify exact value
    assert entry_with_ctx.count('reliability="strong"') == 1


# ============================================================
# Edge
# ============================================================

#TAG: [R2A0]
# Verifies: with no embedder (None), FTS-only path still produces valid output via composite_score
@pytest.mark.edge
def test_no_embedder_fts_only_produces_output():
    """When get_embedder returns None, FTS results still go through RRF and produce XML."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    m1 = insert_memory(conn, type_="correction", topic="sqlite-wal",
                       content="SQLite WAL mode fixed concurrent access deadlock",
                       project="proj", confidence=0.8, session_id="other")

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=None):
        result = retrieval.retrieve_context("SQLite WAL deadlock", session_id="s1")

    # FTS should find the memory. Verify it's a complete XML document with the expected entry.
    lines = result.strip().split("\n")
    assert lines[0].startswith("<cairn_context")
    assert lines[-1] == "</cairn_context>"
    # The FTS-found entry should have a computed score (not hardcoded 0.30)
    # and should reference our memory's content
    entry_line = [l for l in lines if f'id="{m1}"' in l][0]
    # Extract and verify the entry content matches exactly what was inserted
    m_content = re.search(r'>(.+)</entry>', entry_line)
    extracted_content = m_content.group(1) if m_content else ""
    assert extracted_content == "SQLite WAL mode fixed concurrent access deadlock"
    # FTS-only results get a computed composite score — verify reliability label is a valid enum value
    m_rel = re.search(r'reliability="(\w+)"', entry_line)
    extracted_rel = m_rel.group(1) if m_rel else ""
    assert extracted_rel in {"weak", "moderate", "strong"}


#TAG: [R2B0]
# Verifies: max_per_scope parameter caps project and global results independently
@pytest.mark.edge
def test_max_per_scope_caps_results():
    """max_per_scope=1 should limit to at most 1 project and 1 global result."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    # 3 project memories
    for i in range(3):
        insert_memory(conn, type_="fact", topic=f"proj-fact-{i}",
                      content=f"Project memory number {i} about caching strategy",
                      project="proj", confidence=0.8, session_id="other")
    # 3 global memories
    for i in range(3):
        insert_memory(conn, type_="fact", topic=f"global-fact-{i}",
                      content=f"Global memory number {i} about caching strategy",
                      project="otherproj", confidence=0.6, session_id="other")

    mock_emb = MagicMock()
    sem_results = []
    for i in range(3):
        sem_results.append(make_semantic_result(
            i + 1, topic=f"proj-fact-{i}",
            content=f"Project memory number {i} about caching strategy",
            project="proj", confidence=0.8, similarity=0.60 - i * 0.05,
            score=0.55 - i * 0.05))
    for i in range(3):
        sem_results.append(make_semantic_result(
            i + 4, topic=f"global-fact-{i}",
            content=f"Global memory number {i} about caching strategy",
            project="otherproj", confidence=0.6, similarity=0.55 - i * 0.05,
            score=0.45 - i * 0.05))
    mock_emb.find_similar.return_value = sem_results

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("caching strategy", session_id="s1",
                                            max_per_scope=1)

    # Count entry tags in each scope — must be exactly 1 per scope
    project_scope_match = re.search(
        r'<scope level="project".*?>(.*?)</scope>', result, re.DOTALL)
    global_scope_match = re.search(
        r'<scope level="global".*?>(.*?)</scope>', result, re.DOTALL)
    project_scope_count = result.count('<scope level="project"')
    assert project_scope_count == 1, f"Expected exactly 1 project scope, got {project_scope_count}"
    project_entry_count = project_scope_match.group(1).count("<entry ") if project_scope_match else 0
    assert project_entry_count == 1, f"Expected 1 project entry, got {project_entry_count}"
    if global_scope_match:
        global_entries = global_scope_match.group(1).count("<entry ")
        assert global_entries == 1, f"Expected 1 global entry, got {global_entries}"


# ============================================================
# Error
# ============================================================

#TAG: [R3A0]
# Verifies: ConnectionError from embedder.find_similar is caught and FTS results still returned
@pytest.mark.error
def test_embedder_connection_error_falls_back_to_fts():
    """When semantic search raises ConnectionError, FTS results should still be returned."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    m1 = insert_memory(conn, type_="fact", topic="redis-cache",
                       content="Redis caching layer reduces database query latency",
                       project="proj", confidence=0.7, session_id="other")

    mock_emb = MagicMock()
    mock_emb.find_similar.side_effect = ConnectionError("Embedding service unreachable")

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("Redis caching latency", session_id="s1")

    # FTS should still return results despite semantic failure
    # Verify the entry exists in output and has the correct id
    entry_ids = [int(m.group(1)) for m in re.finditer(r'<entry id="(\d+)"', result)]
    assert m1 in entry_ids, f"FTS-found memory id={m1} must appear despite semantic failure"
    # Verify the result is a well-formed cairn_context document
    assert result.strip().split("\n")[-1] == "</cairn_context>"


#TAG: [R3B0]
# Verifies: memory with NULL confidence in DB gets default 0.7 in FTS path instead of crashing
@pytest.mark.error
def test_null_confidence_defaults_to_0_7():
    """FTS path should handle NULL confidence by defaulting to 0.7."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    # Insert with explicit NULL confidence
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) "
        "VALUES (?,?,?,?,?,?,?)",
        ("fact", "null-conf", "Memory with null confidence about deployment pipeline",
         make_blob(42), "proj", None, "other")
    )
    conn.commit()
    mem_id = conn.execute("SELECT id FROM memories WHERE topic='null-conf'").fetchone()[0]

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = []  # No semantic results

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("deployment pipeline confidence", session_id="s1")

    # FTS should find the memory via keywords as the sole result
    assert result is not None, "NULL-confidence memory must still be retrievable via FTS"
    entry_ids = [int(m.group(1)) for m in re.finditer(r'<entry id="(\d+)"', result)]
    assert len(entry_ids) == 1, f"Expected exactly 1 FTS result, got {len(entry_ids)}"
    assert entry_ids[0] == mem_id, f"Expected entry id={mem_id}, got {entry_ids[0]}"


# ============================================================
# Adversarial
# ============================================================

#TAG: [R4A0]
# Verifies: same-session memories are excluded from BOTH FTS and semantic results simultaneously
@pytest.mark.adversarial
def test_same_session_excluded_from_both_methods():
    """Memories from the querying session must be excluded from both search methods."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    # Only memory in DB belongs to the same session
    m1 = insert_memory(conn, type_="fact", topic="self-ref",
                       content="Important fact about recursive self reference patterns",
                       project="proj", confidence=0.9, session_id="s1")

    mock_emb = MagicMock()
    # Semantic also returns this same-session memory with high similarity
    mock_emb.find_similar.return_value = [
        make_semantic_result(m1, topic="self-ref",
                             content="Important fact about recursive self reference patterns",
                             project="proj", confidence=0.9, session_id="s1",
                             similarity=0.95, score=0.80),
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("recursive self reference", session_id="s1")

    # Must be None — the only candidate is same-session and should be filtered by both methods
    assert result is None


#TAG: [R4B0]
# Verifies: context_need with SQL metacharacters and embedded quotes doesn't crash or inject
@pytest.mark.adversarial
def test_sql_metacharacters_in_context_need():
    """SQL metacharacters in query should be handled safely without crashes."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    insert_memory(conn, type_="fact", topic="safe-topic",
                  content="Normal memory about application safety measures",
                  project="proj", confidence=0.7, session_id="other")

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = []

    malicious_query = '''Robert"); DROP TABLE memories; --'''

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        # Should not raise — parameterised queries prevent injection
        result = retrieval.retrieve_context(malicious_query, session_id="s1")

    # Table should still exist with its row intact
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 1, "memories table must survive SQL injection attempt"
    # Verify the memory content is unchanged (injection didn't corrupt data)
    content = conn.execute("SELECT content FROM memories WHERE id=1").fetchone()[0]
    assert content == "Normal memory about application safety measures"
