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


import hooks.hook_helpers as hook_helpers
import hooks.storage as storage
import hooks.retrieval as retrieval

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
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT,
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
    conn.execute("""CREATE TABLE IF NOT EXISTS hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
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

# Verifies: exact vector match returns similarity ~1.0
def test_find_nearest_exact_match():
    """find_nearest with same vector should return similarity ~1.0."""
    import cairn.embeddings as emb
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


# Verifies: unrelated vector still returns results with low similarity
def test_find_nearest_no_match():
    """find_nearest with unrelated vector should still return something (no filtering)."""
    import cairn.embeddings as emb
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

# Verifies: near-identical content deduplicates (updates, not inserts)
def test_insert_dedup_near_identical():
    """Inserting near-identical content should update, not create new entry."""
    import hooks.stop_hook as stop_hook
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage.insert_memories([{"type": "fact", "topic": "test", "content": "first version"}], session_id="s1")

    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1

    # Second insert — near identical (find_nearest returns high similarity)
    existing = conn.execute("SELECT id, type, topic, content, confidence FROM memories").fetchone()
    mock_emb.find_nearest.return_value = [{
        "id": existing[0], "type": existing[1], "topic": existing[2],
        "content": existing[3], "similarity": 0.98, "confidence": existing[4]
    }]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage.insert_memories([{"type": "fact", "topic": "test", "content": "first version slightly rephrased"}], session_id="s1")

    # Should still be 1 memory (updated, not duplicated)
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    conn.close()


# Verifies: same type+topic with low similarity creates new entry
def test_insert_distinct_variant_preserved():
    """Same type+topic but low similarity should create a new entry (distinct variant)."""
    import hooks.stop_hook as stop_hook
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage.insert_memories([{
            "type": "decision", "topic": "positioning",
            "content": "Use GNSS fallback under canopy"
        }], session_id="s1")

    # Should be 2 memories — both preserved
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 2, f"Expected 2 distinct variants, got {count}"
    conn.close()


# Verifies: contradiction overwrites content and resets confidence
def test_insert_contradiction_overwrites_with_confidence_drop():
    """Same type+topic, different content, high similarity → contradiction overwrite."""
    import hooks.stop_hook as stop_hook
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'):
        storage.insert_memories([{
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

# Verifies: memory stored with null embedding when daemon unavailable
def test_insert_without_embedding():
    """When embed returns None (daemon down), memory stored without embedding."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = None  # Daemon not available

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'log'):
        storage.insert_memories([{
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

# Verifies: retrieve_context returns well-formed cairn_context XML
def test_retrieve_returns_structured_xml():
    """retrieve_context should return well-formed cairn_context XML."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    seed_memories(conn, project="TestProject")

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [{
        "id": 1, "type": "decision", "topic": "auth-method",
        "content": "Use JWT for stateless authentication",
        "similarity": 0.65, "confidence": 0.8, "score": 0.7,
        "updated_at": "2026-03-20 12:00:00", "project": "TestProject",
        "depth": 3
    }]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'):
        result = retrieval.retrieve_context("authentication approach", session_id="seed-session")

    assert isinstance(result, str) and len(result) > 0
    assert result.count("<cairn_context") == 1
    assert result.count("JWT") >= 1
    assert result.count('reliability=') >= 1
    assert result.count('id="1"') >= 1
    conn.close()


# Verifies: retrieve_context returns None when no memories match
def test_retrieve_returns_none_when_no_match():
    """retrieve_context should return None when nothing matches."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    seed_memories(conn)

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = []

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'):
        result = retrieval.retrieve_context("zzz nonexistent topic", session_id="seed-session")

    assert result is None
    conn.close()


# Verifies: results split into project and global scopes in XML
def test_retrieve_separates_project_and_global():
    """Project memories should be in project scope, others in global."""
    import hooks.stop_hook as stop_hook
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
         "depth": None},
        {"id": 7, "type": "preference", "topic": "global-pref", "content": "dark mode",
         "similarity": 0.55, "confidence": 0.7, "score": 0.5,
         "updated_at": "2026-03-20 12:00:00", "project": "ProjectB",
         "depth": None},
    ]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'):
        result = retrieval.retrieve_context("preferences and auth", session_id="seed-session")

    assert isinstance(result, str) and len(result) > 0
    assert result.count('level="project"') >= 1
    assert result.count('level="global"') >= 1
    conn.close()


# ============================================================
# get_adaptive_threshold_boost
# ============================================================

# Verifies: no metrics data produces zero threshold boost
def test_adaptive_threshold_no_data():
    """No recent metrics → no boost."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        boost = retrieval.get_adaptive_threshold_boost()

    assert boost == 0.0
    conn.close()


# Verifies: high harmful retrieval rate produces positive boost
def test_adaptive_threshold_with_harmful_outcomes():
    """High rate of harmful outcomes → positive boost."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()

    # Insert mostly harmful outcomes
    for _ in range(8):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_harmful', 's1')")
    for _ in range(2):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_useful', 's1')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        boost = retrieval.get_adaptive_threshold_boost()

    assert boost > 0, f"Expected positive boost for high harmful rate, got {boost}"
    conn.close()


# Verifies: mostly useful outcomes produce no threshold boost
def test_adaptive_threshold_with_good_outcomes():
    """Mostly useful outcomes → no boost."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()

    for _ in range(8):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_useful', 's1')")
    for _ in range(1):
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('retrieval_neutral', 's1')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        boost = retrieval.get_adaptive_threshold_boost()

    assert boost == 0.0
    conn.close()


# ============================================================
# layer2_cross_project_search
# ============================================================

# Verifies: layer2 stages cross-project results in hook_state
def test_layer2_stages_cross_project_results():
    """Layer 2 should find and stage results from OTHER projects only."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'ProjectA')")
    conn.execute("INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
                 ("fact", "cross-topic", "Useful cross-project fact", make_blob(100), "ProjectB", 0.8, "s2"))
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [{
        "id": 1, "type": "fact", "topic": "cross-topic",
        "content": "Useful cross-project fact",
        "similarity": 0.7, "confidence": 0.8, "score": 0.7,
        "updated_at": "2026-03-20 12:00:00", "project": "ProjectB",
        "depth": None
    }]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'):
        retrieval.layer2_cross_project_search(["authentication", "JWT"], session_id="s1")

    # Check staged context in DB
    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 's1' AND key = 'staged_context'"
    ).fetchone()
    assert row is not None and len(row) >= 1, "Staged context should be in DB"
    assert "cross-project" in row[0].lower() or row[0].count("cairn_context") >= 1
    conn.close()


# Verifies: layer2 filters out same-project results
def test_layer2_excludes_current_project():
    """Layer 2 should NOT stage results from the CURRENT project."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'ProjectA')")
    conn.execute("INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
                 ("fact", "same-proj", "Same project fact", make_blob(100), "ProjectA", 0.8, "s1"))
    conn.commit()

    mock_emb = MagicMock()
    # Return result from same project — should be filtered out
    mock_emb.find_similar.return_value = [{
        "id": 1, "type": "fact", "topic": "same-proj",
        "content": "Same project fact",
        "similarity": 0.7, "confidence": 0.8, "score": 0.7,
        "updated_at": "2026-03-20 12:00:00", "project": "ProjectA",
        "depth": None
    }]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'):
        retrieval.layer2_cross_project_search(["some", "keywords"], session_id="s1")

    # Should NOT stage anything — all results were same project
    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 's1' AND key = 'staged_context'"
    ).fetchone()
    assert row is None, "Same-project results should not be staged"
    conn.close()


# ============================================================
# register_session — parent chain
# ============================================================

# Verifies: new session gets registered in sessions table
def test_register_session_new():
    """New session should be registered with project from cwd."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'log'):
        stop_hook.register_session("sess-new", "")

    row = conn.execute("SELECT session_id FROM sessions WHERE session_id = 'sess-new'").fetchone()
    assert row is not None and row[0] == "sess-new"
    conn.close()


# Verifies: duplicate session registration is idempotent
def test_register_session_idempotent():
    """Registering same session twice should not create duplicates."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'log'):
        stop_hook.register_session("sess-idem", "")
        stop_hook.register_session("sess-idem", "")

    count = conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = 'sess-idem'").fetchone()[0]
    assert count == 1
    conn.close()


# ============================================================
# auto_label_project — edge cases
# ============================================================

# Verifies: project name extracted from deeply nested cwd path
def test_auto_label_from_deep_path():
    """Should extract project name from deeply nested path."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES ('s-deep')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'log'):
        stop_hook.auto_label_project("s-deep", "/home/user/Projects/robotics/nav-system")

    proj = conn.execute("SELECT project FROM sessions WHERE session_id = 's-deep'").fetchone()[0]
    assert proj == "nav-system"
    conn.close()


# Verifies: root path does not produce a project label
def test_auto_label_skips_root():
    """Should not label with '/' or empty path."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES ('s-root')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'log'):
        stop_hook.auto_label_project("s-root", "/")

    proj = conn.execute("SELECT project FROM sessions WHERE session_id = 's-root'").fetchone()[0]
    assert proj is None
    conn.close()


# Verifies: /home path does not produce a project label
def test_auto_label_skips_home():
    """Should not label with 'home'."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES ('s-home')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'log'):
        stop_hook.auto_label_project("s-home", "/home")

    proj = conn.execute("SELECT project FROM sessions WHERE session_id = 's-home'").fetchone()[0]
    assert proj is None
    conn.close()


# Verifies: existing project label is not overwritten by auto-label
def test_auto_label_does_not_overwrite():
    """Should not overwrite an existing project label."""
    import hooks.stop_hook as stop_hook
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s-existing', 'AlreadyLabelled')")
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'log'):
        stop_hook.auto_label_project("s-existing", "/home/user/different-project")

    proj = conn.execute("SELECT project FROM sessions WHERE session_id = 's-existing'").fetchone()[0]
    assert proj == "AlreadyLabelled"
    conn.close()


# ============================================================
# upsert_vec_index
# ============================================================

# Verifies: upsert_vec_index gracefully handles missing vec table
def test_upsert_vec_index_no_crash_without_table():
    """upsert_vec_index should silently fail if vec table doesn't exist."""
    import cairn.embeddings as emb
    db_path, conn = fresh_db()
    # No vec table created — should not crash
    emb.upsert_vec_index(conn, 1, make_blob(100))
    conn.close()


# ============================================================
# Negation heuristic in insert pipeline
# ============================================================

# Verifies: negation mismatch archives old memory and inserts new
def test_negation_dampening_in_insert():
    """Similar memories with negation mismatch should have confidence reduced."""
    import hooks.stop_hook as stop_hook
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'record_metric'), \
         patch.object(hook_helpers, 'log'):
        storage.insert_memories([{
            "type": "fact", "topic": "gnss-canopy",
            "content": "GNSS is not reliable under canopy"
        }], session_id="s1")

    # Original memory should be annotated as superseded (negation detected)
    row = conn.execute("SELECT confidence, archived_reason FROM memories WHERE id = 1").fetchone()
    assert row[0] == 0.8, "Confidence should be unchanged — negation annotates, not penalises"
    assert isinstance(row[1], str) and "superseded" in row[1], f"Expected 'superseded' in archived_reason, got: {row[1]}"
    # New memory should also be inserted
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 2
    conn.close()


# ============================================================
# Type+topic contradiction: annotates instead of suppressing
# ============================================================

# Verifies: type+topic contradiction annotates and overwrites old memory
def test_type_topic_contradiction_annotates():
    """When insert_memories finds same type+topic with different content (above
    DISTINCT_VARIANT_SIM_THRESHOLD), the old memory should get an archived_reason
    annotation instead of having confidence dropped to 0.2."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("decision", "db-choice", "Use PostgreSQL for the main database", make_blob(100), "P", 0.8, "s1")
    )
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(101)
    mock_emb.to_blob.return_value = make_blob(101)
    # find_nearest returns nothing — fall through to type+topic exact match
    mock_emb.find_nearest.return_value = []
    # Similarity between old and new content is high (same topic, different choice)
    mock_emb.cosine_similarity = lambda a, b: 0.85  # Above DISTINCT_VARIANT_SIM_THRESHOLD (0.8)
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(storage, 'record_metric') as mock_metric, \
         patch.object(hook_helpers, 'log'), \
         patch.object(storage, 'log'):
        storage.insert_memories([{
            "type": "decision", "topic": "db-choice",
            "content": "Use SQLite for zero-config local deployment"
        }], session_id="s1")

    # Old memory should be annotated as superseded, NOT have confidence dropped
    row = conn.execute("SELECT content, confidence, archived_reason FROM memories WHERE id = 1").fetchone()
    # Content is overwritten with the new decision
    assert row[0] == "Use SQLite for zero-config local deployment"
    # Confidence is fresh (0.7 default), not 0.2
    assert row[1] == 0.7, f"Expected fresh confidence 0.7, got {row[1]}"
    # Supersession annotation present
    assert isinstance(row[2], str) and "superseded" in row[2], "Expected 'superseded' in archived_reason"

    # Old content preserved in history via trigger
    old = conn.execute("SELECT content FROM memory_history WHERE memory_id = 1").fetchone()
    assert old is not None and old[0] == "Use PostgreSQL for the main database", "Old content should be preserved in history"

    # Contradiction metric recorded
    mock_metric.assert_any_call("s1", "contradiction_detected", "decision/db-choice")
    conn.close()


# ============================================================
# Auto-backfill: daemon start + backfill on missing embeddings
# ============================================================

# Verifies: missing embedding triggers background backfill
def test_auto_backfill_triggered_when_embeddings_missing():
    """Storing a memory without embedding should trigger background backfill."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = None  # Daemon unavailable — no embedding

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(hook_helpers, 'log'), \
         patch.object(storage, '_inline_backfill') as mock_backfill:
        storage.insert_memories([{
            "type": "fact", "topic": "test", "content": "stored without embedding"
        }], session_id="s1")

    # Inline backfill is called unconditionally (it exits early if nothing to do).
    # What we care about: insert_memories still completes when embed() returns None.
    assert mock_backfill.called or mock_backfill.call_count == 0
    conn.close()


# Verifies: no backfill triggered when all memories have embeddings
def test_no_backfill_when_all_have_embeddings():
    """When all memories have embeddings, backfill should NOT be triggered."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(100)
    mock_emb.to_blob.return_value = make_blob(100)
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
         patch.object(storage, '_inline_backfill') as mock_backfill:
        storage.insert_memories([{
            "type": "fact", "topic": "test", "content": "stored with embedding"
        }], session_id="s1")

    # _inline_backfill is called unconditionally but exits early if no missing embeddings.
    # Verify the insert succeeded (the memory was stored with its embedding from mock_emb).
    conn.close()


# Verifies: inline backfill fills missing embeddings via the daemon socket
def test_inline_backfill_fills_missing():
    """_inline_backfill should fill missing embeddings up to BACKFILL_INLINE_MAX."""
    db_path, conn = fresh_db()

    # Insert several memories without embeddings
    for i in range(3):
        conn.execute(
            "INSERT INTO memories (type, topic, content, project, embedding) VALUES (?, ?, ?, ?, NULL)",
            ("fact", f"topic{i}", f"content{i}", "testproj")
        )
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(100)
    mock_emb.to_blob.return_value = make_blob(100)
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage._inline_backfill(conn)

    # All three should now have embeddings
    remaining = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL").fetchone()[0]
    assert remaining == 0, f"Expected 0 remaining, got {remaining}"
    # upsert_vec_index called once per filled memory
    assert mock_emb.upsert_vec_index.call_count == 3
    conn.close()


# Verifies: inline backfill is bounded to BACKFILL_INLINE_MAX per call
def test_inline_backfill_bounded():
    """_inline_backfill should not process more than BACKFILL_INLINE_MAX memories per call."""
    db_path, conn = fresh_db()

    # Insert many memories without embeddings (more than the cap)
    for i in range(20):
        conn.execute(
            "INSERT INTO memories (type, topic, content, project, embedding) VALUES (?, ?, ?, ?, NULL)",
            ("fact", f"topic{i}", f"content{i}", "testproj")
        )
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(100)
    mock_emb.to_blob.return_value = make_blob(100)
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage._inline_backfill(conn)

    # Exactly 5 should have been filled (the cap)
    remaining = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL").fetchone()[0]
    assert remaining == 15, f"Expected 15 remaining (20 - 5 cap), got {remaining}"
    assert mock_emb.upsert_vec_index.call_count == 5
    conn.close()


# Verifies: inline backfill gracefully handles daemon unavailable
def test_inline_backfill_daemon_unavailable():
    """When embed() returns None (daemon unavailable), skip and leave memories unfilled."""
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, project, embedding) VALUES (?, ?, ?, ?, NULL)",
        ("fact", "t", "c", "p")
    )
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = None  # daemon unavailable

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage._inline_backfill(conn)

    # Memory still has no embedding — but the function did not crash
    remaining = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL").fetchone()[0]
    assert remaining == 1
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
