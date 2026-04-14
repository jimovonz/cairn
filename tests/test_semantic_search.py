#!/usr/bin/env python3
"""Tests for semantic_search in query.py — 4 categories × 1 target + simple function tests."""

import sys
import os
import sqlite3
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO

import pytest


import cairn.query as query
import cairn.embeddings as emb

TEST_DIR = tempfile.mkdtemp()
_counter = [0]

# Deterministic 384-dim vectors (all-MiniLM-L6-v2 output size)
DIM = 384


def make_vec(seed_value):
    """Create a deterministic normalized 384-dim vector from a seed."""
    rng = np.random.RandomState(seed_value)
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


# Pre-compute some vectors with known relationships
VEC_QUERY = make_vec(42)
VEC_SIMILAR = VEC_QUERY + make_vec(99) * 0.05  # Very similar to query
VEC_SIMILAR /= np.linalg.norm(VEC_SIMILAR)
VEC_SIMILAR = VEC_SIMILAR.astype(np.float32)

VEC_DIFFERENT = make_vec(7)  # Unrelated
VEC_ORTHOGONAL = np.zeros(DIM, dtype=np.float32)
VEC_ORTHOGONAL[0] = 1.0  # Unit vector along axis 0


def fresh_db():
    """Create a fresh SQLite DB with cairn schema."""
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"test_sem_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB,
        session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER,
        depth INTEGER, archived_reason TEXT, keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE memory_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER, content TEXT, session_id TEXT,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY, parent_session_id TEXT,
        project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT,
        session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, keywords, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content, keywords) VALUES (new.id, new.topic, new.content, new.keywords);
    END""")
    conn.commit()
    return db_path, conn


def insert_memory(conn, topic, content, vec, mem_type="fact", project=None,
                  confidence=0.7, updated_at=None, archived_reason=None):
    """Insert a memory with a pre-computed embedding vector."""
    blob = emb.to_blob(vec)
    ua = updated_at or "2026-03-28 12:00:00"
    conn.execute(
        """INSERT INTO memories (type, topic, content, embedding, project, confidence,
           updated_at, created_at, archived_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (mem_type, topic, content, blob, project, confidence, ua, ua, archived_reason))
    conn.commit()


def capture_stdout(func, *args, **kwargs):
    """Capture stdout from a function call."""
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue(), result


# ============================================================
# semantic_search — 4 categories (behavioural, edge, error, adversarial)
# ============================================================

#TAG: [63F3] 2026-04-05
# Verifies: semantic_search returns similar memories ranked by composite score when embeddings module is available
def test_semantic_search_behavioural():
    db_path, conn = fresh_db()
    # Insert two memories: one similar to query, one dissimilar
    insert_memory(conn, "python-setup", "how to set up Python virtualenv", VEC_SIMILAR,
                  project="TestProj", confidence=0.8)
    insert_memory(conn, "cooking-recipe", "best pasta recipe for dinner", VEC_DIFFERENT,
                  project="TestProj", confidence=0.8)
    conn.close()

    with patch.object(query, "DB_PATH", db_path), \
         patch.object(emb, "embed", return_value=VEC_QUERY), \
         patch.object(emb, "_load_vec", return_value=False):
        results = query.semantic_search("python setup guide", limit=10, threshold=0.3)

    assert isinstance(results, list), "Should return a list"
    assert len(results) >= 1, "Should find at least the similar memory"
    # The similar vector should be the top result
    top = results[0]
    assert top["topic"] == "python-setup", f"Top result should be python-setup, got {top['topic']}"
    assert top["similarity"] > 0.9, f"Similar vec should have high similarity, got {top['similarity']}"
    # Verify composite score is computed (not zero)
    assert top["score"] > 0, "Composite score should be positive"


#TAG: [699D] 2026-04-05
# Verifies: semantic_search returns empty list when no memories exceed the similarity threshold
def test_semantic_search_edge_empty_db():
    db_path, conn = fresh_db()
    # Insert only a very dissimilar memory
    insert_memory(conn, "unrelated", "completely unrelated content", VEC_ORTHOGONAL)
    conn.close()

    with patch.object(query, "DB_PATH", db_path), \
         patch.object(emb, "embed", return_value=VEC_QUERY), \
         patch.object(emb, "_load_vec", return_value=False):
        results = query.semantic_search("python setup guide", limit=10, threshold=0.5)

    assert isinstance(results, list), "Should return a list"
    assert len(results) == 0, f"Should return empty list for dissimilar content, got {len(results)} results"


#TAG: [5CB4] 2026-04-05
# Verifies: semantic_search returns None and prints fallback message when embeddings module is unavailable
def test_semantic_search_error_import():
    # Simulate embeddings module not being importable
    # Better approach: patch the import inside semantic_search
    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def mock_import(name, *args, **kwargs):
        if name == "cairn.embeddings" or (name == "cairn" and "embeddings" in (args[2] if len(args) > 2 and args[2] else [])):
            raise ImportError("no module")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        output, result = capture_stdout(query.semantic_search, "test query")

    assert result is None, "Should return None when embeddings unavailable"
    # Verify fallback message is printed to stdout
    fallback_idx = output.find("falling back to FTS")
    assert fallback_idx >= 0, f"Should print fallback message, got: {output!r}"


#TAG: [0E2E] 2026-04-05
# Verifies: semantic_search handles non-string query input without crashing (adversarial type violation)
def test_semantic_search_adversarial_bad_query_type():
    db_path, conn = fresh_db()
    insert_memory(conn, "test", "test content", VEC_SIMILAR)
    conn.close()

    # Pass a numeric query — embed() receives it, may produce garbage or error
    # The function should not raise an unhandled exception
    def embed_passthrough(text, **kw):
        # Simulate embed returning a valid vector regardless of input type
        return VEC_QUERY

    with patch.object(query, "DB_PATH", db_path), \
         patch.object(emb, "embed", side_effect=embed_passthrough), \
         patch.object(emb, "_load_vec", return_value=False):
        # Pass integer — tests contract violation resilience
        try:
            results = query.semantic_search(12345, limit=5, threshold=0.3)
            # If it returns, it should be a list (possibly empty)
            assert isinstance(results, (list, type(None))), \
                f"Should return list or None, got {type(results)}"
        except (TypeError, AttributeError):
            pass  # Acceptable — propagating type error from embed is fine


#TAG: [F8E9] 2026-04-05
# Verifies: garbage gate rejects results when best similarity is below MIN_INJECTION_SIMILARITY (0.35)
@pytest.mark.behavioural
def test_semantic_search_garbage_gate():
    db_path, conn = fresh_db()
    # Insert memory with moderate but sub-threshold similarity
    # Create a vector that will have cosine sim below 0.35 with VEC_QUERY
    mixed = VEC_QUERY * 0.15 + VEC_DIFFERENT * 0.85
    mixed /= np.linalg.norm(mixed)
    mixed = mixed.astype(np.float32)
    insert_memory(conn, "weak-match", "weakly related content", mixed)
    conn.close()

    with patch.object(query, "DB_PATH", db_path), \
         patch.object(emb, "embed", return_value=VEC_QUERY), \
         patch.object(emb, "_load_vec", return_value=False):
        results = query.semantic_search("unrelated query", limit=10, threshold=0.1)

    # The garbage gate (MIN_INJECTION_SIMILARITY=0.35) should filter this out
    # The mixed vector has cosine_sim ~0.3 with VEC_QUERY
    sim = float(np.dot(VEC_QUERY, mixed))
    assert sim < 0.35, f"Test setup: mixed vector similarity should be below 0.35, got {sim:.3f}"
    assert isinstance(results, list), f"Should return list, got {type(results)}"
    assert len(results) == 0, f"Garbage gate should reject results with sim={sim:.3f} < 0.35"


#TAG: [E555] 2026-04-05
# Verifies: semantic_search respects limit parameter and returns at most limit results
def test_semantic_search_edge_limit():
    db_path, conn = fresh_db()
    # Insert 10 memories all similar to query
    for i in range(10):
        # Create slightly varied vectors, all similar to query
        noise = make_vec(100 + i) * 0.02
        v = VEC_QUERY + noise
        v /= np.linalg.norm(v)
        v = v.astype(np.float32)
        insert_memory(conn, f"topic-{i}", f"content about topic number {i} unique-{i}",
                      v, mem_type=["fact", "decision", "preference", "correction", "project"][i % 5])
    conn.close()

    with patch.object(query, "DB_PATH", db_path), \
         patch.object(emb, "embed", return_value=VEC_QUERY), \
         patch.object(emb, "_load_vec", return_value=False):
        results = query.semantic_search("test query", limit=3, threshold=0.1)

    assert isinstance(results, list), "Should return a list"
    # After diversity filter, could be fewer, but should not exceed limit
    assert len(results) <= 3, f"Should respect limit=3, got {len(results)} results"


#TAG: [3A61] 2026-04-05
# Verifies: semantic_search handles negative threshold and zero limit without crashing
def test_semantic_search_adversarial_bad_params():
    db_path, conn = fresh_db()
    insert_memory(conn, "test", "test content", VEC_SIMILAR)
    conn.close()

    with patch.object(query, "DB_PATH", db_path), \
         patch.object(emb, "embed", return_value=VEC_QUERY), \
         patch.object(emb, "_load_vec", return_value=False):
        # Negative threshold — should not crash, may return all or none
        results = query.semantic_search("test", limit=10, threshold=-1.0)
        assert isinstance(results, (list, type(None)))

        # Zero limit — find_similar interprets this; should not crash
        results2 = query.semantic_search("test", limit=0, threshold=0.5)
        assert isinstance(results2, (list, type(None)))


def run_tests():
    """Simple test runner — avoids pytest import-time issues with query.py's os.execv."""
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    errors = []
    for t in tests:
        name = t.__name__
        try:
            t()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            errors.append((name, traceback.format_exc()))
            print(f"  FAIL: {name}: {e}")

    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print("\n--- Failures ---")
        for name, tb in errors:
            print(f"\n{name}:\n{tb}")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
