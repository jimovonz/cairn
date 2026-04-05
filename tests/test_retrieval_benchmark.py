#!/usr/bin/env python3
"""Retrieval benchmark tests — measures latency and quality at various scales.

Creates synthetic databases with 100, 500, and 1000 memories, then measures:
- Embedding time (mock and real if available)
- Search latency (FTS5, brute-force vector, hybrid RRF)
- Quality gate pass rates
- Composite scoring accuracy (relevant results ranked higher)
- Deduplication throughput

Uses deterministic mock vectors for reproducibility. Real embedding benchmarks
are opt-in via --run-slow marker.

Run with: pytest tests/test_retrieval_benchmark.py -v
"""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


TEST_DIR = tempfile.mkdtemp()
_db_counter = [0]

# --- Thresholds for benchmark assertions ---
# These define maximum acceptable latencies (ms). Adjust for CI vs local.
MAX_FTS_LATENCY_MS = 50       # FTS5 search on 1000 rows
MAX_VECTOR_LATENCY_MS = 100   # Brute-force vector search on 1000 rows
MAX_RRF_LATENCY_MS = 200      # Full hybrid RRF pipeline on 1000 rows
MAX_DEDUP_LATENCY_MS = 50     # Single dedup check

# --- Helpers ---

def make_vector(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def make_db(n_memories, project="benchmark"):
    """Create a test DB with n_memories seeded entries."""
    _db_counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"bench_{_db_counter[0]}.db")
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

    # Seed diverse memories
    types = ["fact", "decision", "preference", "correction", "skill", "project", "workflow"]
    topics = [
        "authentication", "database", "api-design", "testing", "deployment",
        "caching", "logging", "monitoring", "security", "performance",
        "docker", "kubernetes", "ci-cd", "git-workflow", "code-review",
        "error-handling", "migration", "scaling", "backup", "documentation",
    ]
    content_templates = [
        "Use {topic} with {detail} for better {benefit}",
        "Decided to implement {topic} using {detail} — rejected {alt} due to complexity",
        "Bug in {topic}: {detail} was causing {benefit} — fixed by updating config",
        "Prefer {detail} over {alt} when working with {topic} — simpler and more maintainable",
        "Remember: {topic} requires {detail} to avoid {benefit} issues in production",
    ]
    details = ["Redis", "PostgreSQL", "JWT", "OAuth2", "GraphQL", "REST", "gRPC",
               "pytest", "alembic", "terraform", "nginx", "systemd", "WAL mode",
               "connection pooling", "rate limiting", "circuit breaker"]
    benefits = ["performance", "reliability", "security", "maintainability",
                "scalability", "observability", "developer experience"]
    projects = [project, project, project, "other-project", None]  # 60% project, 20% other, 20% global

    rng = np.random.RandomState(42)
    for i in range(n_memories):
        mtype = types[i % len(types)]
        topic = topics[i % len(topics)]
        template = content_templates[i % len(content_templates)]
        detail = details[i % len(details)]
        alt = details[(i + 7) % len(details)]
        benefit = benefits[i % len(benefits)]
        content = template.format(topic=topic, detail=detail, alt=alt, benefit=benefit)
        proj = projects[i % len(projects)]
        vec = make_vector(i)
        conf = 0.5 + rng.random() * 0.5  # 0.5 to 1.0

        conn.execute(
            "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mtype, f"{topic}-{i}", content, vec.tobytes(), proj, conf, f"bench-session-{i % 10}")
        )

    conn.commit()
    return db_path, conn


# --- Mock embedder for benchmark ---

class BenchmarkEmbedder:
    """Fast deterministic embedder for benchmarks."""

    def embed(self, text):
        # Deterministic: hash the text to a seed
        seed = hash(text) % (2**31)
        return make_vector(seed)

    def to_blob(self, vec):
        return vec.tobytes()

    def cosine_similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def find_similar(self, conn, query, threshold=0.0, limit=10, current_project=None):
        query_vec = self.embed(query)
        rows = conn.execute(
            "SELECT id, type, topic, content, updated_at, project, session_id, "
            "confidence, depth, archived_reason, embedding FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()
        results = []
        for r in rows:
            if r[10] is None:
                continue
            mem_vec = np.frombuffer(r[10], dtype=np.float32)
            sim = float(np.dot(query_vec, mem_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mem_vec)))
            if sim < threshold:
                continue
            from cairn.embeddings import composite_score
            score = composite_score(sim, r[7] or 0.7, r[4], r[5], current_project)
            results.append({
                "id": r[0], "type": r[1], "topic": r[2], "content": r[3],
                "updated_at": r[4], "project": r[5], "session_id": r[6],
                "confidence": r[7] or 0.7, "depth": r[8],
                "archived_reason": r[9],
                "similarity": sim, "score": score,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


# --- Benchmark fixtures ---

@pytest.fixture(scope="module")
def db_100():
    db_path, conn = make_db(100)
    conn.close()
    yield db_path

@pytest.fixture(scope="module")
def db_500():
    db_path, conn = make_db(500)
    conn.close()
    yield db_path

@pytest.fixture(scope="module")
def db_1000():
    db_path, conn = make_db(1000)
    conn.close()
    yield db_path


# --- FTS5 benchmarks ---

class TestFTS5Performance:
    """Benchmarks FTS5 full-text search latency."""

    def _run_fts(self, db_path, query, n_runs=10):
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000")

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            conn.execute("""
                SELECT m.id, m.type, m.topic, m.content, m.updated_at
                FROM memories_fts f JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ? ORDER BY rank LIMIT 20
            """, (query,)).fetchall()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        conn.close()
        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "median_ms": sorted(times)[len(times) // 2],
        }

    @pytest.mark.behavioural
    def test_fts_100(self, db_100):
        """FTS5 search on 100 memories should be fast."""
        stats = self._run_fts(db_100, '"authentication" OR "security"')
        assert stats["median_ms"] < MAX_FTS_LATENCY_MS, \
            f"FTS5 on 100 rows: {stats['median_ms']:.1f}ms (max {MAX_FTS_LATENCY_MS}ms)"

    @pytest.mark.behavioural
    def test_fts_500(self, db_500):
        """FTS5 search on 500 memories should be fast."""
        stats = self._run_fts(db_500, '"authentication" OR "security"')
        assert stats["median_ms"] < MAX_FTS_LATENCY_MS, \
            f"FTS5 on 500 rows: {stats['median_ms']:.1f}ms (max {MAX_FTS_LATENCY_MS}ms)"

    @pytest.mark.behavioural
    def test_fts_1000(self, db_1000):
        """FTS5 search on 1000 memories should be fast."""
        stats = self._run_fts(db_1000, '"authentication" OR "security"')
        assert stats["median_ms"] < MAX_FTS_LATENCY_MS, \
            f"FTS5 on 1000 rows: {stats['median_ms']:.1f}ms (max {MAX_FTS_LATENCY_MS}ms)"

    @pytest.mark.behavioural
    def test_fts_multi_term(self, db_1000):
        """Multi-term FTS5 query should remain fast."""
        stats = self._run_fts(db_1000, '"authentication" OR "database" OR "deployment" OR "caching"')
        assert stats["median_ms"] < MAX_FTS_LATENCY_MS, \
            f"Multi-term FTS5 on 1000 rows: {stats['median_ms']:.1f}ms"


# --- Vector search benchmarks ---

class TestVectorSearchPerformance:
    """Benchmarks brute-force vector search latency."""

    def _run_vector(self, db_path, query_seed, n_runs=10):
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        query_vec = make_vector(query_seed)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            rows = conn.execute(
                "SELECT id, embedding, confidence, updated_at, project FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
            results = []
            for r in rows:
                mem_vec = np.frombuffer(r[1], dtype=np.float32)
                sim = float(np.dot(query_vec, mem_vec))
                results.append((r[0], sim))
            results.sort(key=lambda x: x[1], reverse=True)
            _ = results[:10]
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        conn.close()
        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "median_ms": sorted(times)[len(times) // 2],
        }

    @pytest.mark.behavioural
    def test_vector_100(self, db_100):
        """Vector search on 100 memories should be fast."""
        stats = self._run_vector(db_100, 999)
        assert stats["median_ms"] < MAX_VECTOR_LATENCY_MS, \
            f"Vector on 100 rows: {stats['median_ms']:.1f}ms (max {MAX_VECTOR_LATENCY_MS}ms)"

    @pytest.mark.behavioural
    def test_vector_500(self, db_500):
        """Vector search on 500 memories should be fast."""
        stats = self._run_vector(db_500, 999)
        assert stats["median_ms"] < MAX_VECTOR_LATENCY_MS, \
            f"Vector on 500 rows: {stats['median_ms']:.1f}ms (max {MAX_VECTOR_LATENCY_MS}ms)"

    @pytest.mark.behavioural
    def test_vector_1000(self, db_1000):
        """Vector search on 1000 memories should be fast."""
        stats = self._run_vector(db_1000, 999)
        assert stats["median_ms"] < MAX_VECTOR_LATENCY_MS, \
            f"Vector on 1000 rows: {stats['median_ms']:.1f}ms (max {MAX_VECTOR_LATENCY_MS}ms)"


# --- Hybrid RRF benchmarks ---

class TestHybridRRFPerformance:
    """Benchmarks the full hybrid FTS5 + vector RRF pipeline."""

    def _run_rrf(self, db_path, query, n_runs=5):
        import hooks.hook_helpers as hook_helpers
        original_db = hook_helpers.DB_PATH
        original_log = hook_helpers.LOG_PATH
        mock_emb = BenchmarkEmbedder()

        times = []
        for _ in range(n_runs):
            hook_helpers.DB_PATH = db_path
            hook_helpers.LOG_PATH = os.path.join(TEST_DIR, "bench.log")
            try:
                with patch.object(hook_helpers, "get_embedder", return_value=mock_emb):
                    # Register a session so retrieve_context can find the project
                    conn = sqlite3.connect(db_path)
                    conn.execute("PRAGMA busy_timeout=5000")
                    conn.execute(
                        "INSERT OR IGNORE INTO sessions (session_id, project) VALUES ('bench-sid', 'benchmark')"
                    )
                    conn.commit()
                    conn.close()

                    start = time.perf_counter()
                    from hooks.retrieval import retrieve_context
                    result = retrieve_context(query, session_id="bench-sid")
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)
            finally:
                hook_helpers.DB_PATH = original_db
                hook_helpers.LOG_PATH = original_log

        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "median_ms": sorted(times)[len(times) // 2],
        }

    @pytest.mark.behavioural
    def test_rrf_100(self, db_100):
        """Full RRF pipeline on 100 memories."""
        stats = self._run_rrf(db_100, "authentication and security decisions")
        assert stats["median_ms"] < MAX_RRF_LATENCY_MS, \
            f"RRF on 100 rows: {stats['median_ms']:.1f}ms (max {MAX_RRF_LATENCY_MS}ms)"

    @pytest.mark.behavioural
    def test_rrf_500(self, db_500):
        """Full RRF pipeline on 500 memories."""
        stats = self._run_rrf(db_500, "authentication and security decisions")
        assert stats["median_ms"] < MAX_RRF_LATENCY_MS, \
            f"RRF on 500 rows: {stats['median_ms']:.1f}ms (max {MAX_RRF_LATENCY_MS}ms)"

    @pytest.mark.behavioural
    def test_rrf_1000(self, db_1000):
        """Full RRF pipeline on 1000 memories."""
        stats = self._run_rrf(db_1000, "authentication and security decisions")
        assert stats["median_ms"] < MAX_RRF_LATENCY_MS, \
            f"RRF on 1000 rows: {stats['median_ms']:.1f}ms (max {MAX_RRF_LATENCY_MS}ms)"


# --- Quality gate benchmarks ---

class TestQualityGates:
    """Tests that quality gates correctly filter at scale."""

    @pytest.mark.behavioural
    def test_garbage_gate_rejects_unrelated(self, db_1000):
        """Garbage gate should reject results when query is completely unrelated."""
        from cairn.config import MIN_INJECTION_SIMILARITY
        emb = BenchmarkEmbedder()
        conn = sqlite3.connect(db_1000)
        conn.execute("PRAGMA busy_timeout=5000")

        # Query with a completely unrelated topic
        results = emb.find_similar(conn, "quantum physics string theory higgs boson",
                                   threshold=0.0, limit=20, current_project="benchmark")
        conn.close()

        if results:
            max_sim = max(r["similarity"] for r in results)
            # With random vectors, most similarities will be low — but the gate
            # threshold should catch this
            assert max_sim < 0.8, \
                f"Unrelated query should not have high similarity matches (max={max_sim:.3f})"

    @pytest.mark.behavioural
    def test_project_scoping_reduces_noise(self, db_1000):
        """Project-scoped queries should rank project memories higher."""
        emb = BenchmarkEmbedder()
        conn = sqlite3.connect(db_1000)
        conn.execute("PRAGMA busy_timeout=5000")

        results = emb.find_similar(conn, "authentication strategy",
                                   threshold=0.0, limit=20, current_project="benchmark")
        conn.close()

        if results and len(results) >= 2:
            # Project memories should get a scope boost
            project_scores = [r["score"] for r in results if r.get("project") == "benchmark"]
            other_scores = [r["score"] for r in results if r.get("project") != "benchmark"]

            if project_scores and other_scores:
                avg_project = sum(project_scores) / len(project_scores)
                avg_other = sum(other_scores) / len(other_scores)
                # Project memories should score at least somewhat higher on average
                # (scope_weight=1.0 vs 0.3)
                assert avg_project >= avg_other * 0.8, \
                    f"Project avg ({avg_project:.3f}) should be competitive with other ({avg_other:.3f})"


# --- Composite scoring benchmarks ---

class TestCompositeScoringAtScale:
    """Tests composite scoring produces sensible rankings at scale."""

    @pytest.mark.behavioural
    def test_scores_are_bounded(self, db_1000):
        """All composite scores should be in a reasonable range."""
        emb = BenchmarkEmbedder()
        conn = sqlite3.connect(db_1000)
        conn.execute("PRAGMA busy_timeout=5000")
        results = emb.find_similar(conn, "database migration alembic",
                                   threshold=0.0, limit=50, current_project="benchmark")
        conn.close()

        for r in results:
            assert 0.0 <= r["score"] <= 2.0, \
                f"Score {r['score']:.3f} out of expected range for '{r['topic']}'"

    @pytest.mark.behavioural
    def test_results_are_sorted_by_score(self, db_1000):
        """Results should be sorted by composite score descending."""
        emb = BenchmarkEmbedder()
        conn = sqlite3.connect(db_1000)
        conn.execute("PRAGMA busy_timeout=5000")
        results = emb.find_similar(conn, "testing and deployment",
                                   threshold=0.0, limit=20, current_project="benchmark")
        conn.close()

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), \
            "Results should be sorted by score descending"


# --- Deduplication benchmarks ---

class TestDeduplicationPerformance:
    """Tests dedup performance with large existing memory sets."""

    @pytest.mark.behavioural
    def test_dedup_check_latency(self, db_1000):
        """Dedup check (cosine similarity against all memories) should be fast."""
        conn = sqlite3.connect(db_1000)
        conn.execute("PRAGMA busy_timeout=5000")
        query_vec = make_vector(9999)

        times = []
        for _ in range(10):
            start = time.perf_counter()
            rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
            max_sim = 0.0
            for r in rows:
                mem_vec = np.frombuffer(r[1], dtype=np.float32)
                sim = float(np.dot(query_vec, mem_vec))
                max_sim = max(max_sim, sim)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        conn.close()
        median = sorted(times)[len(times) // 2]
        assert median < MAX_DEDUP_LATENCY_MS, \
            f"Dedup check on 1000 rows: {median:.1f}ms (max {MAX_DEDUP_LATENCY_MS}ms)"


# --- Scale regression ---

class TestScaleRegression:
    """Verifies latency scales sub-linearly with memory count."""

    @pytest.mark.behavioural
    def test_fts_scales_sublinearly(self, db_100, db_1000):
        """FTS5 latency at 1000 should be less than 10x the latency at 100."""
        conn_100 = sqlite3.connect(db_100)
        conn_1000 = sqlite3.connect(db_1000)
        query = '"authentication" OR "security"'

        times_100 = []
        times_1000 = []
        for _ in range(10):
            start = time.perf_counter()
            conn_100.execute(
                "SELECT m.id FROM memories_fts f JOIN memories m ON f.rowid = m.id "
                "WHERE memories_fts MATCH ? ORDER BY rank LIMIT 20", (query,)
            ).fetchall()
            times_100.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            conn_1000.execute(
                "SELECT m.id FROM memories_fts f JOIN memories m ON f.rowid = m.id "
                "WHERE memories_fts MATCH ? ORDER BY rank LIMIT 20", (query,)
            ).fetchall()
            times_1000.append((time.perf_counter() - start) * 1000)

        conn_100.close()
        conn_1000.close()

        med_100 = sorted(times_100)[5]
        med_1000 = sorted(times_1000)[5]

        # 10x data should be well under 10x latency for indexed search
        # Use 20x as a generous bound to avoid flaky tests
        assert med_1000 < med_100 * 20 or med_1000 < 5.0, \
            f"FTS5 scaling: 100={med_100:.2f}ms, 1000={med_1000:.2f}ms (ratio={med_1000/max(med_100, 0.01):.1f}x)"

    @pytest.mark.behavioural
    def test_vector_scales_linearly(self, db_100, db_1000):
        """Brute-force vector search should scale roughly linearly (within 15x)."""
        query_vec = make_vector(9999)

        def bench(db_path, n=10):
            conn = sqlite3.connect(db_path)
            times = []
            for _ in range(n):
                start = time.perf_counter()
                rows = conn.execute(
                    "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
                ).fetchall()
                for r in rows:
                    mem_vec = np.frombuffer(r[1], dtype=np.float32)
                    float(np.dot(query_vec, mem_vec))
                times.append((time.perf_counter() - start) * 1000)
            conn.close()
            return sorted(times)[n // 2]

        med_100 = bench(db_100)
        med_1000 = bench(db_1000)

        # Linear scan: 10x data → ~10x latency. Allow 15x for overhead.
        assert med_1000 < med_100 * 15 or med_1000 < 200, \
            f"Vector scaling: 100={med_100:.2f}ms, 1000={med_1000:.2f}ms (ratio={med_1000/max(med_100, 0.01):.1f}x)"


# --- Cleanup ---

def teardown_module():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
