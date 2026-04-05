#!/usr/bin/env python3
"""Retrieval quality benchmarks — measures precision, recall, and MRR with ground truth.

Uses REAL embeddings (all-MiniLM-L6-v2) to test whether the retrieval pipeline
returns the right memories for specific queries. This is the quality counterpart
to test_retrieval_benchmark.py which tests latency.

Ground truth: A synthetic DB with semantically distinct memory clusters, each with
defined queries that should retrieve specific memories. Measures:
- Precision@K: fraction of top-K results that are relevant
- Recall@K: fraction of relevant memories found in top-K
- MRR: reciprocal rank of first relevant result
- Method comparison: FTS-only vs semantic-only vs hybrid RRF

Run with: pytest tests/test_retrieval_quality.py -v
(Takes ~5-10s for model load on first run, fast thereafter)
"""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
import time
from unittest.mock import patch
from typing import Any

import numpy as np
import pytest

try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2")
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False

pytestmark = pytest.mark.skipif(not HAS_MODEL, reason="Embedding model not available")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

TEST_DIR = tempfile.mkdtemp()

# --- Metrics ---

def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of relevant memories found in top-K."""
    if not relevant_ids:
        return 1.0  # vacuously true
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean reciprocal rank — reciprocal of the rank of the first relevant result."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# --- Ground truth DB ---

# Each cluster is a thematic group of memories with queries that should find them.
# Memories are semantically distinct across clusters to test discrimination.
CLUSTERS = [
    {
        "name": "authentication",
        "memories": [
            ("decision", "auth-strategy", "JWT with RS256 refresh tokens chosen over session cookies for stateless API — cookies rejected due to CORS complexity with mobile clients"),
            ("correction", "auth-expiry-bug", "Token expiry validation was checking issued_at instead of expires_at in auth middleware — caused premature token rejection after clock skew"),
            ("skill", "auth-testing", "Use PyJWT decode with verify=False to inspect token claims in tests without needing the signing key"),
            ("fact", "auth-rate-limit", "Authentication endpoints rate-limited to 5 attempts per minute per IP using Redis sliding window"),
        ],
        "queries": [
            ("How does authentication work in the API?", {0, 1, 2, 3}),  # all relevant
            ("What token strategy did we choose?", {0}),  # decision specifically
            ("Was there a bug with token expiry?", {1}),  # correction specifically
            ("How to test JWT tokens?", {2}),  # skill specifically
        ],
    },
    {
        "name": "database",
        "memories": [
            ("decision", "db-choice", "PostgreSQL 15 with pgvector extension chosen over MongoDB — needed ACID transactions and vector similarity for embeddings"),
            ("skill", "db-migration", "Use alembic upgrade head --sql to preview migration SQL before applying — catches destructive changes before they hit production"),
            ("correction", "db-connection-leak", "Connection pool exhaustion caused by unclosed connections in error paths — fixed by adding context manager to all DB access functions"),
            ("fact", "db-wal-mode", "SQLite WAL mode enabled for concurrent read access — busy timeout set to 5 seconds to prevent lock contention"),
        ],
        "queries": [
            ("What database are we using and why?", {0}),
            ("How to run database migrations safely?", {1}),
            ("Was there a connection pool issue?", {2}),
            ("How is SQLite configured for concurrency?", {3}),
        ],
    },
    {
        "name": "deployment",
        "memories": [
            ("workflow", "deploy-process", "Deploy via GitHub Actions: PR merge to main triggers build, test, docker push, then rolling update on GCE instance group"),
            ("decision", "deploy-docker-pins", "Always pin Docker image versions in compose files — rejected latest tag due to reproducibility failures in staging"),
            ("correction", "deploy-env-leak", "Production .env file was accidentally committed — rotated all secrets, added .env to .gitignore, and enabled pre-commit hook to catch secrets"),
            ("skill", "deploy-rollback", "Quick rollback: gcloud compute instance-groups managed rolling-action replace with --max-unavailable=0 for zero-downtime"),
        ],
        "queries": [
            ("How do we deploy to production?", {0}),
            ("Should we use latest tag for Docker images?", {1}),
            ("Was there a security incident with env files?", {2}),
            ("How to rollback a bad deployment?", {3}),
        ],
    },
    {
        "name": "testing",
        "memories": [
            ("preference", "test-style", "Prefer pytest fixtures over setUp/tearDown — cleaner dependency injection and better isolation between tests"),
            ("skill", "test-mocking", "Use unittest.mock.patch as context manager not decorator when you need the mock object reference for assertions"),
            ("fact", "test-coverage", "Test suite has 486 tests across 23 files — coverage at 87% with hooks/ at 92% and cairn/ at 81%"),
            ("correction", "test-flaky", "Flaky test in test_daemon was due to socket cleanup race — fixed by adding retry with 100ms backoff on ConnectionRefused"),
        ],
        "queries": [
            ("What testing framework and style do we use?", {0}),
            ("How should I mock dependencies in tests?", {1}),
            ("How many tests do we have?", {2}),
            ("Were there any flaky tests?", {3}),
        ],
    },
    {
        "name": "noise",  # Unrelated memories to test discrimination
        "memories": [
            ("fact", "weather-nz", "Tauranga weather averages 22C in summer with occasional subtropical storms from the Coral Sea"),
            ("person", "vet-contact", "Dr Sarah Chen at Bay Vet Clinic — handles Milo annual checkup, phone 07-555-0123"),
            ("fact", "recipe-sourdough", "Sourdough starter needs feeding every 12 hours at room temperature — 1:1:1 ratio starter:flour:water by weight"),
            ("preference", "music-focus", "Lo-fi hip hop or ambient electronic for coding sessions — no lyrics, steady tempo around 80-90 BPM"),
        ],
        "queries": [],  # No queries — these are distractors only
    },
]


def build_quality_db():
    """Build the ground-truth test database with real embeddings."""
    db_path = os.path.join(TEST_DIR, "quality.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        embedding BLOB, session_id TEXT, project TEXT,
        confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER,
        depth INTEGER, archived_reason TEXT, associated_files TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE memory_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER NOT NULL, content TEXT NOT NULL,
        session_id TEXT, changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY, parent_session_id TEXT, project TEXT,
        transcript_path TEXT, started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT NOT NULL, session_id TEXT, detail TEXT,
        value REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key)
    )""")
    conn.execute("CREATE INDEX idx_memories_type ON memories(type)")
    conn.execute("CREATE INDEX idx_memories_topic ON memories(topic)")
    conn.execute("CREATE INDEX idx_memories_project ON memories(project)")
    conn.execute("CREATE INDEX idx_metrics_event ON metrics(event)")
    conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content, session_id, changed_at)
        VALUES (old.id, old.content, old.session_id, old.updated_at);
    END""")
    conn.execute("""CREATE VIRTUAL TABLE memories_fts USING fts5(
        topic, content, content=memories, content_rowid=id
    )""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content)
        VALUES (new.id, new.topic, new.content);
    END""")
    conn.execute("""CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content)
        VALUES ('delete', old.id, old.topic, old.content);
    END""")
    conn.execute("""CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content)
        VALUES ('delete', old.id, old.topic, old.content);
        INSERT INTO memories_fts(rowid, topic, content)
        VALUES (new.id, new.topic, new.content);
    END""")

    # Register a session for project scoping
    conn.execute(
        "INSERT INTO sessions (session_id, project) VALUES ('quality-sid', 'qualitytest')"
    )

    # Embed and insert all memories
    import embeddings
    id_offset = 0
    cluster_id_map: dict[str, list[int]] = {}

    for cluster in CLUSTERS:
        ids = []
        for mtype, topic, content in cluster["memories"]:
            embed_text = f"qualitytest {mtype} {topic} {content}"
            vec = embeddings.embed(embed_text)
            conn.execute(
                "INSERT INTO memories (type, topic, content, embedding, project, session_id, confidence) "
                "VALUES (?, ?, ?, ?, 'qualitytest', 'seed-session', 0.7)",
                (mtype, topic, content, embeddings.to_blob(vec))
            )
            ids.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        cluster_id_map[cluster["name"]] = ids

    conn.commit()
    return db_path, conn, cluster_id_map


# --- Query runners ---

def run_semantic_search(db_path: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Run semantic-only search (no FTS)."""
    import embeddings
    import hook_helpers

    original_db = hook_helpers.DB_PATH
    try:
        hook_helpers.DB_PATH = db_path
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        results = embeddings.find_similar(conn, query, threshold=0.0, limit=limit,
                                          current_project="qualitytest")
        conn.close()
        return results
    finally:
        hook_helpers.DB_PATH = original_db


def run_fts_search(db_path: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Run FTS5-only search."""
    import re
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")

    _STOPWORDS = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "its", "was", "are", "be",
        "has", "had", "have", "do", "did", "does", "will", "can", "could",
        "would", "should", "may", "might", "not", "no", "what", "when",
        "where", "who", "how", "why", "that", "this", "these", "those",
    })

    words = re.findall(r'\w+', query.lower())
    meaningful = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
    if not meaningful:
        meaningful = words
    fts_query = " OR ".join(f'"{w}"' for w in meaningful)

    rows = conn.execute("""
        SELECT m.id, m.type, m.topic, m.content, m.updated_at, m.project,
               m.session_id, m.confidence, rank
        FROM memories_fts f JOIN memories m ON f.rowid = m.id
        WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?
    """, (fts_query, limit)).fetchall()
    conn.close()

    return [{"id": r[0], "type": r[1], "topic": r[2], "content": r[3],
             "updated_at": r[4], "project": r[5], "session_id": r[6],
             "confidence": r[7] or 0.7, "bm25_rank": -r[8]}
            for r in rows]


def run_rrf_search(db_path: str, query: str) -> list[dict[str, Any]]:
    """Run full hybrid RRF pipeline via retrieve_context."""
    import hook_helpers
    from retrieval import retrieve_context

    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH
    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, "quality.log")
        result_xml = retrieve_context(query, session_id="quality-sid")
        if not result_xml:
            return []
        # Parse IDs from XML
        import re
        ids = [int(m) for m in re.findall(r'id="(\d+)"', result_xml)]
        return [{"id": i} for i in ids]
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log


# --- Fixtures ---

@pytest.fixture(scope="module")
def quality_db():
    """Build the quality test DB once per module (real embeddings, ~3s)."""
    db_path, conn, cluster_id_map = build_quality_db()
    conn.close()
    yield db_path, cluster_id_map


def _build_ground_truth(cluster_id_map: dict[str, list[int]]):
    """Build list of (query_text, relevant_memory_ids) from CLUSTERS."""
    queries = []
    for cluster in CLUSTERS:
        if not cluster["queries"]:
            continue
        ids = cluster_id_map[cluster["name"]]
        for query_text, relevant_indices in cluster["queries"]:
            relevant_ids = {ids[i] for i in relevant_indices}
            queries.append((query_text, relevant_ids))
    return queries


# --- Quality tests ---

class TestSemanticSearchQuality:
    """Tests semantic search returns the right memories."""

    @pytest.mark.behavioural
    def test_semantic_mrr_above_threshold(self, quality_db):
        """Average MRR across all queries should be above 0.5 (first relevant in top 2)."""
        db_path, cluster_id_map = quality_db
        ground_truth = _build_ground_truth(cluster_id_map)

        mrr_scores = []
        for query_text, relevant_ids in ground_truth:
            results = run_semantic_search(db_path, query_text, limit=10)
            retrieved_ids = [r["id"] for r in results]
            mrr_scores.append(mrr(retrieved_ids, relevant_ids))

        avg_mrr = sum(mrr_scores) / len(mrr_scores)
        assert avg_mrr >= 0.5, \
            f"Average MRR {avg_mrr:.3f} below threshold 0.5. Per-query: {list(zip([q[0][:40] for q in ground_truth], [f'{s:.2f}' for s in mrr_scores]))}"

    @pytest.mark.behavioural
    def test_semantic_precision_at_3(self, quality_db):
        """Average P@3 should be above 0.3 (at least 1 relevant in top 3)."""
        db_path, cluster_id_map = quality_db
        ground_truth = _build_ground_truth(cluster_id_map)

        p3_scores = []
        for query_text, relevant_ids in ground_truth:
            results = run_semantic_search(db_path, query_text, limit=10)
            retrieved_ids = [r["id"] for r in results]
            p3_scores.append(precision_at_k(retrieved_ids, relevant_ids, 3))

        avg_p3 = sum(p3_scores) / len(p3_scores)
        assert avg_p3 >= 0.3, \
            f"Average P@3 {avg_p3:.3f} below threshold 0.3"

    @pytest.mark.behavioural
    def test_semantic_recall_at_5(self, quality_db):
        """Average R@5 should be above 0.5 (find at least half the relevant memories)."""
        db_path, cluster_id_map = quality_db
        ground_truth = _build_ground_truth(cluster_id_map)

        r5_scores = []
        for query_text, relevant_ids in ground_truth:
            results = run_semantic_search(db_path, query_text, limit=10)
            retrieved_ids = [r["id"] for r in results]
            r5_scores.append(recall_at_k(retrieved_ids, relevant_ids, 5))

        avg_r5 = sum(r5_scores) / len(r5_scores)
        assert avg_r5 >= 0.5, \
            f"Average R@5 {avg_r5:.3f} below threshold 0.5"

    @pytest.mark.behavioural
    def test_semantic_discriminates_clusters(self, quality_db):
        """Auth queries should not return deployment/testing/noise memories in top 3."""
        db_path, cluster_id_map = quality_db
        auth_ids = set(cluster_id_map["authentication"])
        noise_ids = set(cluster_id_map["noise"])

        results = run_semantic_search(db_path, "JWT authentication token strategy", limit=5)
        top_3_ids = {r["id"] for r in results[:3]}

        # Should have at least one auth memory in top 3
        assert top_3_ids & auth_ids, \
            f"Auth query should find auth memories in top 3. Got IDs: {top_3_ids}"

        # Should not have noise in top 3
        assert not (top_3_ids & noise_ids), \
            f"Auth query should not return noise memories in top 3. Got IDs: {top_3_ids}"


class TestFTSSearchQuality:
    """Tests FTS5 keyword search returns the right memories."""

    @pytest.mark.behavioural
    def test_fts_finds_exact_terms(self, quality_db):
        """FTS should find memories containing exact query terms."""
        db_path, _ = quality_db

        results = run_fts_search(db_path, "JWT refresh tokens")
        retrieved_content = " ".join(r["content"] for r in results)
        assert "JWT" in retrieved_content or "refresh" in retrieved_content, \
            "FTS should find memories with exact keyword matches"

    @pytest.mark.behavioural
    def test_fts_finds_technical_terms(self, quality_db):
        """FTS should find memories with technical terms like 'alembic', 'pgvector'."""
        db_path, cluster_id_map = quality_db
        db_ids = set(cluster_id_map["database"])

        results = run_fts_search(db_path, "alembic migration")
        retrieved_ids = {r["id"] for r in results}
        assert retrieved_ids & db_ids, \
            "FTS should find database memories when querying 'alembic migration'"

    @pytest.mark.behavioural
    def test_fts_mrr_above_threshold(self, quality_db):
        """FTS average MRR should be reasonable (above 0.3)."""
        db_path, cluster_id_map = quality_db
        ground_truth = _build_ground_truth(cluster_id_map)

        mrr_scores = []
        for query_text, relevant_ids in ground_truth:
            results = run_fts_search(db_path, query_text, limit=10)
            retrieved_ids = [r["id"] for r in results]
            mrr_scores.append(mrr(retrieved_ids, relevant_ids))

        avg_mrr = sum(mrr_scores) / len(mrr_scores)
        # FTS is weaker than semantic for natural language queries — lower bar
        assert avg_mrr >= 0.3, \
            f"FTS average MRR {avg_mrr:.3f} below threshold 0.3"


class TestHybridRRFQuality:
    """Tests the full hybrid RRF pipeline quality."""

    @pytest.mark.behavioural
    def test_rrf_finds_relevant_memories(self, quality_db):
        """RRF pipeline should find relevant memories for clear queries."""
        db_path, cluster_id_map = quality_db
        auth_ids = set(cluster_id_map["authentication"])

        results = run_rrf_search(db_path, "authentication token strategy and JWT decisions")
        retrieved_ids = {r["id"] for r in results}

        assert retrieved_ids & auth_ids, \
            f"RRF should find auth memories. Got IDs: {retrieved_ids}, expected some of: {auth_ids}"

    @pytest.mark.behavioural
    def test_rrf_discriminates_noise(self, quality_db):
        """RRF should not return noise memories for targeted queries."""
        db_path, cluster_id_map = quality_db
        noise_ids = set(cluster_id_map["noise"])

        results = run_rrf_search(db_path, "database migration with alembic PostgreSQL")
        retrieved_ids = {r["id"] for r in results}

        assert not (retrieved_ids & noise_ids), \
            f"RRF should not return noise memories. Got noise IDs: {retrieved_ids & noise_ids}"

    @pytest.mark.behavioural
    def test_rrf_beats_fts_on_semantic_queries(self, quality_db):
        """RRF should match or beat FTS-only on natural language queries."""
        db_path, cluster_id_map = quality_db
        ground_truth = _build_ground_truth(cluster_id_map)

        rrf_mrr_scores = []
        fts_mrr_scores = []

        for query_text, relevant_ids in ground_truth:
            rrf_results = run_rrf_search(db_path, query_text)
            rrf_ids = [r["id"] for r in rrf_results]
            rrf_mrr_scores.append(mrr(rrf_ids, relevant_ids))

            fts_results = run_fts_search(db_path, query_text, limit=10)
            fts_ids = [r["id"] for r in fts_results]
            fts_mrr_scores.append(mrr(fts_ids, relevant_ids))

        avg_rrf = sum(rrf_mrr_scores) / len(rrf_mrr_scores)
        avg_fts = sum(fts_mrr_scores) / len(fts_mrr_scores)

        # RRF should be at least as good as FTS alone (it adds semantic)
        assert avg_rrf >= avg_fts * 0.9, \
            f"RRF MRR ({avg_rrf:.3f}) should be competitive with FTS ({avg_fts:.3f})"


class TestCrossClusterDiscrimination:
    """Tests that retrieval correctly discriminates between topic clusters."""

    @pytest.mark.behavioural
    def test_cluster_isolation(self, quality_db):
        """Each cluster's queries should primarily return that cluster's memories."""
        db_path, cluster_id_map = quality_db

        correct = 0
        total = 0

        for cluster in CLUSTERS:
            if not cluster["queries"]:
                continue
            cluster_ids = set(cluster_id_map[cluster["name"]])

            for query_text, _ in cluster["queries"]:
                results = run_semantic_search(db_path, query_text, limit=3)
                if results:
                    top_id = results[0]["id"]
                    if top_id in cluster_ids:
                        correct += 1
                    total += 1

        accuracy = correct / total if total else 0
        assert accuracy >= 0.6, \
            f"Cluster isolation accuracy {accuracy:.1%} ({correct}/{total}) below 60% threshold"

    @pytest.mark.behavioural
    def test_noise_never_dominates(self, quality_db):
        """Noise memories should never be the top result for any topical query."""
        db_path, cluster_id_map = quality_db
        noise_ids = set(cluster_id_map["noise"])

        for cluster in CLUSTERS:
            if not cluster["queries"]:
                continue
            for query_text, _ in cluster["queries"]:
                results = run_semantic_search(db_path, query_text, limit=3)
                if results:
                    top_id = results[0]["id"]
                    assert top_id not in noise_ids, \
                        f"Noise memory {top_id} was top result for: '{query_text}'"


class TestQualityMetricsSummary:
    """Aggregate quality metrics across all methods — outputs a summary table."""

    @pytest.mark.behavioural
    def test_print_quality_summary(self, quality_db):
        """Print a summary of quality metrics for all methods (always passes, diagnostic)."""
        db_path, cluster_id_map = quality_db
        ground_truth = _build_ground_truth(cluster_id_map)

        methods = {
            "Semantic": lambda q: [r["id"] for r in run_semantic_search(db_path, q, limit=10)],
            "FTS5": lambda q: [r["id"] for r in run_fts_search(db_path, q, limit=10)],
            "Hybrid RRF": lambda q: [r["id"] for r in run_rrf_search(db_path, q)],
        }

        print("\n\n=== Retrieval Quality Summary ===")
        print(f"{'Method':<15} {'Avg MRR':>8} {'Avg P@3':>8} {'Avg R@5':>8} {'Queries':>8}")
        print("-" * 55)

        for method_name, search_fn in methods.items():
            mrr_scores = []
            p3_scores = []
            r5_scores = []

            for query_text, relevant_ids in ground_truth:
                retrieved_ids = search_fn(query_text)
                mrr_scores.append(mrr(retrieved_ids, relevant_ids))
                p3_scores.append(precision_at_k(retrieved_ids, relevant_ids, 3))
                r5_scores.append(recall_at_k(retrieved_ids, relevant_ids, 5))

            avg_mrr = sum(mrr_scores) / len(mrr_scores)
            avg_p3 = sum(p3_scores) / len(p3_scores)
            avg_r5 = sum(r5_scores) / len(r5_scores)

            print(f"{method_name:<15} {avg_mrr:>8.3f} {avg_p3:>8.3f} {avg_r5:>8.3f} {len(ground_truth):>8}")

        print()

        # Per-query breakdown for semantic (most important method)
        print(f"{'Query':<50} {'MRR':>6} {'P@3':>6} {'R@5':>6}")
        print("-" * 70)
        for query_text, relevant_ids in ground_truth:
            results = run_semantic_search(db_path, query_text, limit=10)
            retrieved_ids = [r["id"] for r in results]
            m = mrr(retrieved_ids, relevant_ids)
            p = precision_at_k(retrieved_ids, relevant_ids, 3)
            r = recall_at_k(retrieved_ids, relevant_ids, 5)
            q_short = query_text[:48]
            print(f"{q_short:<50} {m:>6.2f} {p:>6.2f} {r:>6.2f}")

        print()
        # This test always passes — it's diagnostic output
        assert True


# --- Cleanup ---

def teardown_module():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
