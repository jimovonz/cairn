#!/usr/bin/env python3
"""Tests for the memory consolidation pipeline."""

import os
import sys
import tempfile

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Ensure cairn package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"consolidation_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT,
        keywords TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        origin_id TEXT,
        user_id TEXT,
        updated_by TEXT,
        team_id TEXT,
        source_ref TEXT,
        deleted_at TIMESTAMP,
        synced_at TIMESTAMP)""")
    conn.execute("""CREATE TABLE memory_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER, content TEXT, session_id TEXT,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS pair_assessments (
        memory_id_a INTEGER NOT NULL, memory_id_b INTEGER NOT NULL,
        mode TEXT NOT NULL, verdict TEXT NOT NULL, reason TEXT,
        assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (memory_id_a, memory_id_b, mode))""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, keywords, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content, keywords) VALUES (new.id, new.topic, new.content, new.keywords); END""")
    conn.execute("""CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content, keywords) VALUES ('delete', old.id, old.topic, old.content, old.keywords); END""")
    conn.commit()
    return db_path, conn


def make_embedding(seed: int = 0) -> bytes:
    """Create a deterministic normalized 384-dim embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


def make_similar_embedding(base_seed: int, noise: float = 0.05) -> bytes:
    """Create an embedding similar to make_embedding(base_seed) with small noise."""
    rng_base = np.random.RandomState(base_seed)
    vec = rng_base.randn(384).astype(np.float32)
    rng_noise = np.random.RandomState(base_seed + 1000)
    vec = vec + rng_noise.randn(384).astype(np.float32) * noise
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


def make_different_embedding(seed: int = 999) -> bytes:
    """Create an embedding distinctly different from seed=0."""
    return make_embedding(seed)


# === Phase 1: Clustering tests ===

class TestFindClusters:
    def test_finds_similar_memories(self):
        """Two memories with near-identical embeddings should cluster."""
        db_path, conn = fresh_db()
        emb1 = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.02)
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "topic-a", "content A", emb1))
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "topic-b", "content B similar to A", emb2))
        conn.commit()

        from cairn.embeddings import find_clusters
        clusters = find_clusters(conn, similarity_threshold=0.85)
        conn.close()
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_does_not_cluster_different_memories(self):
        """Two memories with different embeddings should not cluster."""
        db_path, conn = fresh_db()
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "topic-a", "content A", make_embedding(1)))
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "topic-b", "content B", make_embedding(999)))
        conn.commit()

        from cairn.embeddings import find_clusters
        clusters = find_clusters(conn, similarity_threshold=0.85)
        conn.close()
        assert len(clusters) == 0

    def test_excludes_archived_memories(self):
        """Archived memories should not appear in clusters."""
        db_path, conn = fresh_db()
        emb = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.02)
        conn.execute("INSERT INTO memories (type, topic, content, embedding, archived_reason) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "topic-a", "content A", emb, "old reason"))
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "topic-b", "content B", emb2))
        conn.commit()

        from cairn.embeddings import find_clusters
        clusters = find_clusters(conn, similarity_threshold=0.85)
        conn.close()
        assert len(clusters) == 0

    def test_respects_max_cluster_size(self):
        """Clusters should not exceed max_cluster_size."""
        db_path, conn = fresh_db()
        base = make_embedding(42)
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "t0", "base content", base))
        for i in range(5):
            conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                          ("fact", f"t{i+1}", f"similar content {i}", make_similar_embedding(42, noise=0.01 + i * 0.001)))
        conn.commit()

        from cairn.embeddings import find_clusters
        clusters = find_clusters(conn, similarity_threshold=0.80, max_cluster_size=3)
        conn.close()
        for cluster in clusters:
            assert len(cluster) <= 3

    def test_clusters_sorted_by_recency(self):
        """Within a cluster, entries should be sorted newest-first."""
        db_path, conn = fresh_db()
        emb1 = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.02)
        conn.execute("INSERT INTO memories (type, topic, content, embedding, updated_at) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "old", "old content", emb1, "2026-01-01"))
        conn.execute("INSERT INTO memories (type, topic, content, embedding, updated_at) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "new", "new content", emb2, "2026-04-17"))
        conn.commit()

        from cairn.embeddings import find_clusters
        clusters = find_clusters(conn, similarity_threshold=0.85)
        conn.close()
        assert clusters[0][0]["topic"] == "new"
        assert clusters[0][1]["topic"] == "old"


# === Phase 2: NLI scoring tests ===

class TestNLIScoring:
    def test_filters_non_entailing_pairs(self):
        """Entries that don't entail each other should be filtered out."""
        from cairn.consolidate import score_cluster_nli

        cluster = [
            {"id": 1, "content": "The sky is blue"},
            {"id": 2, "content": "Water is wet"},
        ]

        with patch("cairn.embeddings._daemon_nli") as mock_nli:
            mock_nli.return_value = [
                [-2.0, -1.0, 3.0],  # 1->2: neutral
                [-2.0, -1.0, 3.0],  # 2->1: neutral
            ]
            with patch("cairn.config.NLI_ENABLED", True):
                result = score_cluster_nli(cluster)
        assert len(result) == 0

    def test_keeps_entailing_pairs(self):
        """Entries with bidirectional entailment should be kept."""
        from cairn.consolidate import score_cluster_nli

        cluster = [
            {"id": 1, "content": "SQLite uses WAL mode"},
            {"id": 2, "content": "The database uses WAL journaling"},
        ]

        with patch("cairn.embeddings._daemon_nli") as mock_nli:
            mock_nli.return_value = [
                [-3.0, 4.0, -1.0],  # 1->2: strong entailment
                [-3.0, 3.5, -1.0],  # 2->1: strong entailment
            ]
            with patch("cairn.config.NLI_ENABLED", True):
                result = score_cluster_nli(cluster)
        assert len(result) == 2

    def test_nli_disabled_returns_full_cluster(self):
        """When NLI is disabled, return the full cluster unchanged."""
        from cairn.consolidate import score_cluster_nli

        cluster = [{"id": 1, "content": "a"}, {"id": 2, "content": "b"}]
        with patch("cairn.config.NLI_ENABLED", False):
            result = score_cluster_nli(cluster)
        assert len(result) == 2

    def test_nli_daemon_failure_returns_full_cluster(self):
        """If daemon NLI fails, return the full cluster (graceful degradation)."""
        from cairn.consolidate import score_cluster_nli

        cluster = [{"id": 1, "content": "a"}, {"id": 2, "content": "b"}]
        with patch("cairn.embeddings._daemon_nli", return_value=None):
            with patch("cairn.config.NLI_ENABLED", True):
                result = score_cluster_nli(cluster)
        assert len(result) == 2


# === Phase 4: Execution tests ===

class TestExecuteConsolidation:
    def test_creates_meta_memory_and_archives_sources(self):
        """Consolidation should create a new entry and archive sources."""
        db_path, conn = fresh_db()
        emb1 = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.02)
        conn.execute("INSERT INTO memories (type, topic, content, embedding, project, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                      ("fact", "db-corruption", "DB corrupted again", emb1, "cairn", 0.7))
        conn.execute("INSERT INTO memories (type, topic, content, embedding, project, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                      ("fact", "db-corruption-fix", "DB corruption fixed", emb2, "cairn", 0.7))
        conn.commit()

        cluster = [
            {"id": 1, "type": "fact", "topic": "db-corruption", "content": "DB corrupted again",
             "project": "cairn", "confidence": 0.7, "updated_at": "2026-04-17"},
            {"id": 2, "type": "fact", "topic": "db-corruption-fix", "content": "DB corruption fixed",
             "project": "cairn", "confidence": 0.7, "updated_at": "2026-04-16"},
        ]

        with patch("cairn.embeddings.embed", return_value=np.random.randn(384).astype(np.float32)):
            with patch("cairn.embeddings.to_blob", return_value=make_embedding(99)):
                with patch("cairn.embeddings.upsert_vec_index"):
                    from cairn.consolidate import execute_consolidation
                    new_id = execute_consolidation(conn, cluster, "DB corruption occurred and was fixed")

        assert new_id is not None

        # Check meta-memory was created
        meta = conn.execute("SELECT type, topic, content, project FROM memories WHERE id = ?", (new_id,)).fetchone()
        assert meta[0] == "fact"
        assert meta[1] == "db-corruption"
        assert "DB corruption occurred and was fixed" in meta[2]
        assert meta[3] == "cairn"

        # Check sources were archived
        for source_id in [1, 2]:
            row = conn.execute("SELECT confidence, archived_reason FROM memories WHERE id = ?", (source_id,)).fetchone()
            assert row[0] == 0  # confidence zeroed
            assert row[1] == f"consolidated:{new_id}"

        conn.close()

    def test_consolidation_is_reversible(self):
        """Archived sources can be recovered by clearing archived_reason."""
        db_path, conn = fresh_db()
        conn.execute("INSERT INTO memories (type, topic, content, embedding, confidence) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "t", "original content", make_embedding(42), 0.8))
        conn.commit()

        # Simulate archival
        conn.execute("UPDATE memories SET confidence = 0, archived_reason = 'consolidated:999' WHERE id = 1")
        conn.commit()

        # Reverse it
        conn.execute("UPDATE memories SET confidence = 0.8, archived_reason = NULL WHERE id = 1")
        conn.commit()

        row = conn.execute("SELECT confidence, archived_reason, content FROM memories WHERE id = 1").fetchone()
        assert row[0] == 0.8
        assert row[1] is None
        assert row[2] == "original content"
        conn.close()


# === Integration: dry-run test ===

class TestDryRun:
    def test_dry_run_makes_no_changes(self):
        """Dry run should report candidates but not modify the database."""
        db_path, conn = fresh_db()
        emb1 = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.02)
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "topic-a", "content A about thing", emb1))
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "topic-a2", "content A about same thing", emb2))
        conn.commit()

        initial_count = conn.execute("SELECT count(*) FROM memories WHERE archived_reason IS NULL").fetchone()[0]

        from cairn.consolidate import run_consolidation
        with patch("cairn.consolidate.DB_PATH", db_path):
            with patch("cairn.consolidate.score_cluster_nli", side_effect=lambda c: c):
                summary = run_consolidation(execute=False)

        final_count = conn.execute("SELECT count(*) FROM memories WHERE archived_reason IS NULL").fetchone()[0]
        assert final_count == initial_count
        assert summary["consolidated"] == 0
        conn.close()


# === NLI model integration test (requires model) ===

_has_nli_model = False
try:
    from sentence_transformers import CrossEncoder
    _model_path = os.path.expanduser("~/.cache/huggingface/hub")
    _has_nli_model = any("nli" in d.lower() for d in os.listdir(_model_path)) if os.path.exists(_model_path) else False
except Exception:
    pass


@pytest.mark.skipif(not _has_nli_model, reason="NLI model not cached locally")
class TestNLIModelIntegration:
    def test_entailment_detected_for_paraphrases(self):
        """Real NLI model should detect entailment between paraphrases."""
        model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
        scores = model.predict([
            ["The database was corrupted", "DB corruption occurred"],
            ["DB corruption occurred", "The database was corrupted"],
        ])
        # Entailment is index 1
        assert scores[0][1] > 0  # forward entailment
        assert scores[1][1] > 0  # reverse entailment

    def test_no_entailment_for_unrelated(self):
        """Real NLI model should not detect entailment between unrelated content."""
        model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
        scores = model.predict([
            ["The database was corrupted", "The user prefers dark mode"],
        ])
        assert scores[0][1] < scores[0][2]  # entailment < neutral

    def test_contradiction_detected_for_negations(self):
        """Real NLI model should detect contradiction between opposing statements."""
        model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
        scores = model.predict([
            ["The database is corrupted", "The database is not corrupted"],
            ["Recency weighting is wrong for memory systems", "Recency weighting is correct for memory systems"],
        ])
        # Contradiction is index 0
        assert scores[0][0] > scores[0][1]  # contradiction > entailment
        assert scores[1][0] > scores[1][1]  # contradiction > entailment

    def test_no_contradiction_for_compatible_statements(self):
        """Real NLI model should not detect contradiction between compatible facts."""
        model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
        scores = model.predict([
            ["The database uses WAL mode", "WAL mode provides better concurrency"],
        ])
        assert scores[0][0] < scores[0][2]  # contradiction < neutral


# === Contradiction detection unit tests ===

class TestContradictionDetection:
    def test_find_contradiction_pairs_returns_similar_pairs(self):
        """Pairs above similarity threshold should be returned."""
        db_path, conn = fresh_db()
        emb1 = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.05)
        conn.execute("INSERT INTO memories (type, topic, content, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "db", "DB is corrupted", emb1, "2026-04-01"))
        conn.execute("INSERT INTO memories (type, topic, content, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "db", "DB is not corrupted", emb2, "2026-04-17"))
        conn.commit()

        from cairn.consolidate import find_contradiction_pairs
        with patch("cairn.consolidate.DB_PATH", db_path):
            pairs = find_contradiction_pairs(conn)
        conn.close()
        assert len(pairs) == 1
        assert pairs[0]["older"]["id"] == 1
        assert pairs[0]["newer"]["id"] == 2
        assert pairs[0]["same_topic"] is True

    def test_excludes_archived_from_contradiction_pairs(self):
        """Archived memories should not appear in contradiction pairs."""
        db_path, conn = fresh_db()
        emb1 = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.02)
        conn.execute("INSERT INTO memories (type, topic, content, embedding, archived_reason) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "db", "DB is broken", emb1, "already archived"))
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "db", "DB is fixed", emb2))
        conn.commit()

        from cairn.consolidate import find_contradiction_pairs
        with patch("cairn.consolidate.DB_PATH", db_path):
            pairs = find_contradiction_pairs(conn)
        conn.close()
        assert len(pairs) == 0

    def test_nli_filters_non_contradictions(self):
        """Pairs without NLI contradiction should be filtered."""
        from cairn.consolidate import score_contradictions_nli

        pairs = [{
            "older": {"id": 1, "content": "WAL mode is enabled"},
            "newer": {"id": 2, "content": "WAL provides concurrency"},
            "similarity": 0.8,
        }]

        with patch("cairn.embeddings._daemon_nli") as mock_nli:
            # Low contradiction, high neutral
            mock_nli.return_value = [
                [-3.0, -1.0, 4.0],  # fwd: neutral
                [-3.0, -1.0, 4.0],  # rev: neutral
            ]
            with patch("cairn.config.NLI_ENABLED", True):
                result = score_contradictions_nli(pairs)
        assert len(result) == 0

    def test_nli_keeps_genuine_contradictions(self):
        """Pairs with high NLI contradiction score should be kept."""
        from cairn.consolidate import score_contradictions_nli

        pairs = [{
            "older": {"id": 1, "content": "DB is corrupted"},
            "newer": {"id": 2, "content": "DB is not corrupted"},
            "similarity": 0.9,
        }]

        with patch("cairn.embeddings._daemon_nli") as mock_nli:
            mock_nli.return_value = [
                [4.0, -2.0, -1.0],  # fwd: strong contradiction
                [3.5, -2.0, -1.0],  # rev: strong contradiction
            ]
            with patch("cairn.config.NLI_ENABLED", True):
                result = score_contradictions_nli(pairs)
        assert len(result) == 1
        assert result[0]["nli_contradiction_score"] == 4.0

    def test_execute_supersession_archives_older(self):
        """execute_supersession should archive the older memory."""
        db_path, conn = fresh_db()
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "db", "DB is broken", make_embedding(42)))
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "db", "DB is fixed", make_similar_embedding(42, noise=0.02)))
        conn.commit()

        superseded = [{
            "older": {"id": 1, "content": "DB is broken"},
            "newer": {"id": 2, "content": "DB is fixed"},
            "reason": "DB was repaired",
        }]

        from cairn.consolidate import execute_supersession
        count = execute_supersession(conn, superseded)
        assert count == 1

        row = conn.execute("SELECT archived_reason FROM memories WHERE id = 1").fetchone()
        assert "superseded" in row[0]
        assert "#2" in row[0]

        # Newer should be untouched
        row2 = conn.execute("SELECT archived_reason FROM memories WHERE id = 2").fetchone()
        assert row2[0] is None
        conn.close()

    def test_assessed_pairs_skipped_on_rerun(self):
        """Previously assessed pairs should not appear in candidates."""
        db_path, conn = fresh_db()
        emb1 = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.02)
        conn.execute("INSERT INTO memories (type, topic, content, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "db", "DB is broken", emb1, "2026-04-01"))
        conn.execute("INSERT INTO memories (type, topic, content, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                      ("fact", "db", "DB is fixed", emb2, "2026-04-17"))
        conn.commit()

        from cairn.consolidate import find_contradiction_pairs
        with patch("cairn.consolidate.DB_PATH", db_path):
            pairs_first = find_contradiction_pairs(conn)
        assert len(pairs_first) == 1

        # Record assessment
        conn.execute("INSERT INTO pair_assessments (memory_id_a, memory_id_b, mode, verdict) VALUES (1, 2, 'contradiction', 'complementary')")
        conn.commit()

        with patch("cairn.consolidate.DB_PATH", db_path):
            pairs_second = find_contradiction_pairs(conn)
        assert len(pairs_second) == 0
        conn.close()

    def test_contradiction_dry_run_makes_no_changes(self):
        """Dry run should not modify the database."""
        db_path, conn = fresh_db()
        emb1 = make_embedding(42)
        emb2 = make_similar_embedding(42, noise=0.02)
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "db", "DB is broken", emb1))
        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES (?, ?, ?, ?)",
                      ("fact", "db", "DB is fixed", emb2))
        conn.commit()

        initial_count = conn.execute("SELECT count(*) FROM memories WHERE archived_reason IS NULL").fetchone()[0]

        from cairn.consolidate import run_contradiction_detection
        with patch("cairn.consolidate.DB_PATH", db_path):
            with patch("cairn.consolidate.score_contradictions_nli", return_value=[]):
                summary = run_contradiction_detection(execute=False)

        final_count = conn.execute("SELECT count(*) FROM memories WHERE archived_reason IS NULL").fetchone()[0]
        assert final_count == initial_count
        conn.close()
