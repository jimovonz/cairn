#!/usr/bin/env python3
"""Tests for the retrieval pipeline — retrieve_context, layer1_search, layer2_cross_project_search,
find_nearest, adaptive thresholds, and insert_memories dedup/contradiction paths.

These are the high-risk functions that determine what gets stored, what gets retrieved,
and what gets injected. Uses in-memory DB with mock embeddings to exercise real code paths."""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

TEST_DIR = tempfile.mkdtemp()
_counter = [0]

# Deterministic vectors: seed N always produces the same 384-dim unit vector
def make_vector(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)

def make_blob(seed):
    return make_vector(seed).tobytes()

def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"retr_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER,
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
    conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content, session_id, changed_at)
        VALUES (old.id, old.content, old.session_id, old.updated_at); END""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content); END""")
    conn.execute("""CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content) VALUES ('delete', old.id, old.topic, old.content); END""")
    conn.commit()
    return db_path, conn


def seed_memories(conn, project="TestProject"):
    """Insert a diverse set of memories with deterministic embeddings."""
    memories = [
        ("decision", "auth-method", "Use JWT for stateless authentication", 100, 0.8),
        ("decision", "db-choice", "Use SQLite for local storage", 200, 0.9),
        ("fact", "user-location", "User is in New Zealand", 300, 0.7),
        ("preference", "code-style", "User prefers snake_case in Python", 400, 0.6),
        ("correction", "hook-format", "Hook settings need nested structure", 500, 0.7),
        ("fact", "bird-sighting", "Pukeko observed on lawn — blue bird red beak", 600, 0.8),
    ]
    ids = []
    for mem_type, topic, content, seed, conf in memories:
        conn.execute(
            "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
            (mem_type, topic, content, make_blob(seed), project, conf, "seed-session"))
        ids.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('seed-session', ?)", (project,))
    conn.commit()
    return ids


# ============================================================
# find_nearest — used for dedup
# ============================================================

def test_find_nearest_exact_match():
    """find_nearest with same vector should return similarity ~1.0."""
    import embeddings as emb
    db_path, conn = fresh_db()
    seed_memories(conn)
    query_vec = make_vector(100)  # Same as auth-method

    with patch.object(emb, 'embed', return_value=query_vec), \
         patch.object(emb, '_load_vec', return_value=False):
        results = emb.find_nearest(conn, "decision auth-method Use JWT", limit=1)

    assert len(results) == 1
    assert results[0]["similarity"] > 0.99
    assert results[0]["topic"] == "auth-method"
    conn.close()


def test_find_nearest_no_match():
    """find_nearest with unrelated vector should still return something (no filtering)."""
    import embeddings as emb
    db_path, conn = fresh_db()
    seed_memories(conn)
    query_vec = make_vector(999)

    with patch.object(emb, 'embed', return_value=query_vec), \
         patch.object(emb, '_load_vec', return_value=False):
        results = emb.find_nearest(conn, "completely unrelated text", limit=1)

    assert len(results) == 1  # Always returns top-1 — no threshold
    assert results[0]["similarity"] < 0.3  # But similarity should be low
    conn.close()


# ============================================================
# insert_memories — dedup path
# ============================================================

def test_insert_dedup_near_identical():
    """Inserting near-identical content should update, not create new entry."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.commit()

    # First insert
    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(100)
    mock_emb.to_blob.return_value = make_blob(100)
    mock_emb.find_nearest.return_value = []  # No existing match
    mock_emb.upsert_vec_index = MagicMock()
    mock_emb.cosine_similarity = lambda a, b: float(np.dot(a, b))

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb):
        stop_hook.insert_memories([{"type": "fact", "topic": "test", "content": "first version"}], session_id="s1")

    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1

    # Second insert — near identical (find_nearest returns high similarity)
    existing = conn.execute("SELECT id, type, topic, content, confidence FROM memories").fetchone()
    mock_emb.find_nearest.return_value = [{
        "id": existing[0], "type": existing[1], "topic": existing[2],
        "content": existing[3], "similarity": 0.98, "confidence": existing[4]
    }]

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb):
        stop_hook.insert_memories([{"type": "fact", "topic": "test", "content": "first version slightly rephrased"}], session_id="s1")

    # Should still be 1 memory (updated, not duplicated)
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    conn.close()


def test_insert_distinct_variant_preserved():
    """Same type+topic but low similarity should create a new entry (distinct variant)."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.commit()

    # Insert first
    conn.execute("INSERT INTO memories (type, topic, content, embedding, session_id, project) VALUES (?,?,?,?,?,?)",
                 ("decision", "positioning", "Use RTK for accuracy", make_blob(100), "s1", "P"))
    conn.commit()

    # Second insert — same type+topic but different content, low embedding similarity
    mock_emb = MagicMock()
    mock_emb.embed.side_effect = [make_vector(200), make_vector(200), make_vector(100)]
    mock_emb.to_blob.return_value = make_blob(200)
    mock_emb.find_nearest.return_value = []  # Below dedup threshold
    mock_emb.cosine_similarity.return_value = 0.3  # Low similarity — distinct variant
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb):
        stop_hook.insert_memories([{
            "type": "decision", "topic": "positioning",
            "content": "Use GNSS fallback under canopy"
        }], session_id="s1")

    # Should be 2 memories — both preserved
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 2, f"Expected 2 distinct variants, got {count}"
    conn.close()


def test_insert_contradiction_overwrites_with_confidence_drop():
    """Same type+topic, different content, high similarity → contradiction overwrite."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.execute("INSERT INTO memories (type, topic, content, embedding, session_id, project, confidence) VALUES (?,?,?,?,?,?,?)",
                 ("decision", "db", "Use PostgreSQL", make_blob(100), "s1", "P", 0.9))
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.side_effect = [make_vector(101), make_vector(101), make_vector(100)]  # Similar but not identical
    mock_emb.to_blob.return_value = make_blob(101)
    mock_emb.find_nearest.return_value = []
    mock_emb.cosine_similarity.return_value = 0.95  # High similarity — true update, not variant
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb), \
         patch.object(stop_hook, 'record_metric'):
        stop_hook.insert_memories([{
            "type": "decision", "topic": "db",
            "content": "Use SQLite instead"
        }], session_id="s1")

    # Should still be 1 memory — overwritten
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    mem = conn.execute("SELECT content, confidence FROM memories").fetchone()
    assert mem[0] == "Use SQLite instead"
    assert mem[1] == 0.7  # Fresh default confidence
    # Old version in history
    assert conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0] >= 1
    old = conn.execute("SELECT content FROM memory_history").fetchone()
    assert old[0] == "Use PostgreSQL"
    conn.close()


# ============================================================
# insert_memories — no embedding (daemon down)
# ============================================================

def test_insert_without_embedding():
    """When embed returns None (daemon down), memory stored without embedding."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = None  # Daemon not available

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb), \
         patch.object(stop_hook, 'log'):
        stop_hook.insert_memories([{
            "type": "fact", "topic": "no-embed",
            "content": "stored without embedding"
        }], session_id="s1")

    mem = conn.execute("SELECT content, embedding FROM memories").fetchone()
    assert mem[0] == "stored without embedding"
    assert mem[1] is None  # No embedding
    conn.close()


# ============================================================
# retrieve_context — full pipeline with quality gates
# ============================================================

def test_retrieve_returns_structured_xml():
    """retrieve_context should return well-formed brain_context XML."""
    import stop_hook
    db_path, conn = fresh_db()
    seed_memories(conn, project="TestProject")

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [{
        "id": 1, "type": "decision", "topic": "auth-method",
        "content": "Use JWT for stateless authentication",
        "similarity": 0.65, "confidence": 0.8, "score": 0.7,
        "updated_at": "2026-03-20 12:00:00", "project": "TestProject",
        "source_start": 5, "source_end": 10
    }]

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb), \
         patch.object(stop_hook, 'record_metric'), \
         patch.object(stop_hook, 'log'):
        result = stop_hook.retrieve_context("authentication approach", session_id="seed-session")

    assert result is not None
    assert "<brain_context" in result
    assert "JWT" in result
    assert 'reliability=' in result
    assert 'id="1"' in result
    conn.close()


def test_retrieve_returns_none_when_no_match():
    """retrieve_context should return None when nothing matches."""
    import stop_hook
    db_path, conn = fresh_db()
    seed_memories(conn)

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = []

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb), \
         patch.object(stop_hook, 'record_metric'), \
         patch.object(stop_hook, 'log'):
        result = stop_hook.retrieve_context("zzz nonexistent topic", session_id="seed-session")

    assert result is None
    conn.close()


def test_retrieve_separates_project_and_global():
    """Project memories should be in project scope, others in global."""
    import stop_hook
    db_path, conn = fresh_db()
    seed_memories(conn, project="ProjectA")
    # Add a global memory from a different project
    conn.execute("INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
                 ("preference", "global-pref", "User likes dark mode", make_blob(700), "ProjectB", 0.7, "other-session"))
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [
        {"id": 1, "type": "decision", "topic": "auth", "content": "JWT auth",
         "similarity": 0.7, "confidence": 0.8, "score": 0.7,
         "updated_at": "2026-03-20 12:00:00", "project": "ProjectA",
         "source_start": None, "source_end": None},
        {"id": 7, "type": "preference", "topic": "global-pref", "content": "dark mode",
         "similarity": 0.55, "confidence": 0.7, "score": 0.5,
         "updated_at": "2026-03-20 12:00:00", "project": "ProjectB",
         "source_start": None, "source_end": None},
    ]

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb), \
         patch.object(stop_hook, 'record_metric'), \
         patch.object(stop_hook, 'log'):
        result = stop_hook.retrieve_context("preferences and auth", session_id="seed-session")

    assert result is not None
    assert 'level="project"' in result
    assert 'level="global"' in result
    conn.close()


# ============================================================
# get_adaptive_threshold_boost
# ============================================================

def test_adaptive_threshold_no_data():
    """No recent metrics → no boost."""
    import stop_hook
    db_path, conn = fresh_db()

    with patch.object(stop_hook, 'DB_PATH', db_path):
        boost = stop_hook.get_adaptive_threshold_boost()

    assert boost == 0.0
    conn.close()


def test_adaptive_threshold_with_harmful_outcomes():
    """High rate of harmful outcomes → positive boost."""
    import stop_hook
    db_path, conn = fresh_db()

    # Insert mostly harmful outcomes
    for _ in range(8):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_harmful', 's1')")
    for _ in range(2):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_useful', 's1')")
    conn.commit()

    with patch.object(stop_hook, 'DB_PATH', db_path):
        boost = stop_hook.get_adaptive_threshold_boost()

    assert boost > 0, f"Expected positive boost for high harmful rate, got {boost}"
    conn.close()


def test_adaptive_threshold_with_good_outcomes():
    """Mostly useful outcomes → no boost."""
    import stop_hook
    db_path, conn = fresh_db()

    for _ in range(8):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_useful', 's1')")
    for _ in range(1):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_neutral', 's1')")
    conn.commit()

    with patch.object(stop_hook, 'DB_PATH', db_path):
        boost = stop_hook.get_adaptive_threshold_boost()

    assert boost == 0.0
    conn.close()


# ============================================================
# layer2_cross_project_search
# ============================================================

def test_layer2_stages_cross_project_results():
    """Layer 2 should find and stage results from OTHER projects only."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'ProjectA')")
    conn.execute("INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
                 ("fact", "cross-topic", "Useful cross-project fact", make_blob(100), "ProjectB", 0.8, "s2"))
    conn.commit()

    staged_path = os.path.join(TEST_DIR, f".staged_l2_{_counter[0]}")
    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [{
        "id": 1, "type": "fact", "topic": "cross-topic",
        "content": "Useful cross-project fact",
        "similarity": 0.7, "confidence": 0.8, "score": 0.7,
        "updated_at": "2026-03-20 12:00:00", "project": "ProjectB",
        "source_start": None, "source_end": None
    }]

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'STAGED_PATH', staged_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb), \
         patch.object(stop_hook, 'record_metric'), \
         patch.object(stop_hook, 'log'):
        stop_hook.layer2_cross_project_search(["authentication", "JWT"], session_id="s1")

    # Check staged file exists and has content
    assert os.path.exists(staged_path)
    with open(staged_path) as f:
        staged = json.load(f)
    assert "s1" in staged
    assert "cross-project" in staged["s1"].lower() or "brain_context" in staged["s1"]
    conn.close()


def test_layer2_excludes_current_project():
    """Layer 2 should NOT stage results from the CURRENT project."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'ProjectA')")
    conn.execute("INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
                 ("fact", "same-proj", "Same project fact", make_blob(100), "ProjectA", 0.8, "s1"))
    conn.commit()

    staged_path = os.path.join(TEST_DIR, f".staged_exclude_{_counter[0]}")
    mock_emb = MagicMock()
    # Return result from same project — should be filtered out
    mock_emb.find_similar.return_value = [{
        "id": 1, "type": "fact", "topic": "same-proj",
        "content": "Same project fact",
        "similarity": 0.7, "confidence": 0.8, "score": 0.7,
        "updated_at": "2026-03-20 12:00:00", "project": "ProjectA",
        "source_start": None, "source_end": None
    }]

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'STAGED_PATH', staged_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb), \
         patch.object(stop_hook, 'record_metric'), \
         patch.object(stop_hook, 'log'):
        stop_hook.layer2_cross_project_search(["some", "keywords"], session_id="s1")

    # Should NOT stage anything — all results were same project
    if os.path.exists(staged_path):
        with open(staged_path) as f:
            staged = json.load(f)
        assert "s1" not in staged, "Same-project results should not be staged"
    conn.close()


# ============================================================
# register_session — parent chain
# ============================================================

def test_register_session_new():
    """New session should be registered with project from cwd."""
    import stop_hook
    db_path, conn = fresh_db()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'log'):
        stop_hook.register_session("sess-new", "")

    row = conn.execute("SELECT session_id FROM sessions WHERE session_id = 'sess-new'").fetchone()
    assert row is not None
    conn.close()


def test_register_session_idempotent():
    """Registering same session twice should not create duplicates."""
    import stop_hook
    db_path, conn = fresh_db()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'log'):
        stop_hook.register_session("sess-idem", "")
        stop_hook.register_session("sess-idem", "")

    count = conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = 'sess-idem'").fetchone()[0]
    assert count == 1
    conn.close()


# ============================================================
# auto_label_project — edge cases
# ============================================================

def test_auto_label_from_deep_path():
    """Should extract project name from deeply nested path."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES ('s-deep')")
    conn.commit()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'log'):
        stop_hook.auto_label_project("s-deep", "/home/user/Projects/robotics/nav-system")

    proj = conn.execute("SELECT project FROM sessions WHERE session_id = 's-deep'").fetchone()[0]
    assert proj == "nav-system"
    conn.close()


def test_auto_label_skips_root():
    """Should not label with '/' or empty path."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES ('s-root')")
    conn.commit()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'log'):
        stop_hook.auto_label_project("s-root", "/")

    proj = conn.execute("SELECT project FROM sessions WHERE session_id = 's-root'").fetchone()[0]
    assert proj is None
    conn.close()


def test_auto_label_skips_home():
    """Should not label with 'home'."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES ('s-home')")
    conn.commit()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'log'):
        stop_hook.auto_label_project("s-home", "/home")

    proj = conn.execute("SELECT project FROM sessions WHERE session_id = 's-home'").fetchone()[0]
    assert proj is None
    conn.close()


def test_auto_label_does_not_overwrite():
    """Should not overwrite an existing project label."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s-existing', 'AlreadyLabelled')")
    conn.commit()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'log'):
        stop_hook.auto_label_project("s-existing", "/home/user/different-project")

    proj = conn.execute("SELECT project FROM sessions WHERE session_id = 's-existing'").fetchone()[0]
    assert proj == "AlreadyLabelled"
    conn.close()


# ============================================================
# upsert_vec_index
# ============================================================

def test_upsert_vec_index_no_crash_without_table():
    """upsert_vec_index should silently fail if vec table doesn't exist."""
    import embeddings as emb
    db_path, conn = fresh_db()
    # No vec table created — should not crash
    emb.upsert_vec_index(conn, 1, make_blob(100))
    conn.close()


# ============================================================
# Negation heuristic in insert pipeline
# ============================================================

def test_negation_dampening_in_insert():
    """Similar memories with negation mismatch should have confidence reduced."""
    import stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.execute("INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
                 ("fact", "gnss-accuracy", "GNSS is reliable in open sky", make_blob(100), "P", 0.8, "s1"))
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(101)  # Similar
    mock_emb.to_blob.return_value = make_blob(101)
    # find_nearest returns something similar (0.7) but below dedup threshold (0.95)
    mock_emb.find_nearest.return_value = [{
        "id": 1, "type": "fact", "topic": "gnss-accuracy",
        "content": "GNSS is reliable in open sky",
        "similarity": 0.75, "confidence": 0.8
    }]
    mock_emb.cosine_similarity = lambda a, b: 0.75
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(stop_hook, 'DB_PATH', db_path), \
         patch.object(stop_hook, 'get_embedder', return_value=mock_emb), \
         patch.object(stop_hook, 'record_metric'), \
         patch.object(stop_hook, 'log'):
        stop_hook.insert_memories([{
            "type": "fact", "topic": "gnss-canopy",
            "content": "GNSS is not reliable under canopy"
        }], session_id="s1")

    # Original memory should have reduced confidence (negation detected)
    old_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    assert old_conf < 0.8, f"Expected confidence drop from negation, got {old_conf}"
    # New memory should also be inserted
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 2
    conn.close()


def cleanup():
    shutil.rmtree(TEST_DIR, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
    cleanup()
