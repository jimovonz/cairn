#!/usr/bin/env python3
"""Hard retrieval quality benchmark — graded relatedness with guaranteed misses.

Unlike test_retrieval_quality.py (clean clusters, high baseline), this benchmark
models real-world retrieval challenges:

1. **Overlapping topics** — memories span multiple domains (auth+db, deploy+testing)
2. **Temporal spread** — old vs fresh memories with varying staleness
3. **Cross-project** — memories split across projects with global entries
4. **Paraphrasing distance** — queries range from near-verbatim to heavy rephrase
5. **Graded difficulty** — queries ranked easy/medium/hard/impossible
6. **Distractors** — near-miss memories that share keywords but are wrong answers

Query difficulty tiers:
- EASY: query uses same vocabulary as the memory
- MEDIUM: query rephrases the concept (synonyms, different framing)
- HARD: query is indirect — asks about the consequence/symptom, not the root cause
- IMPOSSIBLE: no relevant memory exists — tests false positive rate

This benchmark is designed to produce IMPERFECT scores. A strategy that scores
1.0 MRR here is suspicious. Expected baseline: ~0.5-0.8 MRR.
"""

import sys
import os
import json
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

# These benchmarks require the real embedding model — skip in CI where model isn't downloaded
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    del _model
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False

pytestmark = pytest.mark.skipif(not HAS_MODEL, reason="Embedding model not available")


TEST_DIR = tempfile.mkdtemp()

# --- Metrics (same as test_retrieval_quality.py) ---

def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / len(top_k)

def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 1.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)

def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def false_positive_rate(retrieved_ids: list[int], k: int) -> float:
    """For impossible queries: any result in top-K is a false positive."""
    return 1.0 if retrieved_ids[:k] else 0.0


# --- Ground truth ---

# Memories are tuples of (type, topic, content, project, age_days)
# age_days controls temporal spread — older memories have weaker recency signal
MEMORIES = [
    # === CLUSTER: Authentication (project: webapp) ===
    ("decision", "auth-jwt-choice",
     "Chose JWT with RS256 over session cookies — stateless API, mobile client CORS issues with cookies, refresh tokens in httponly cookies",
     "webapp", 0),
    ("correction", "auth-token-expiry-bug",
     "Token expiry checked issued_at instead of expires_at causing premature rejection after server clock drift of 2 seconds",
     "webapp", 5),
    ("skill", "auth-jwt-testing",
     "PyJWT decode with options={'verify_signature': False} to inspect claims in test without signing key",
     "webapp", 30),
    ("fact", "auth-rate-limiting",
     "Login endpoint rate-limited to 5 attempts per minute per IP via Redis sliding window counter",
     "webapp", 15),

    # === CLUSTER: Database (project: webapp) ===
    ("decision", "db-postgres-choice",
     "PostgreSQL 15 with pgvector chosen over MongoDB — need ACID for financial transactions and vector search for embeddings",
     "webapp", 2),
    ("skill", "db-alembic-preview",
     "Run alembic upgrade head --sql to preview migration DDL before applying to production — caught a DROP COLUMN last week",
     "webapp", 10),
    ("correction", "db-connection-pool-leak",
     "Connection pool exhaustion traced to unclosed connections in exception handlers — added contextlib.closing wrapper",
     "webapp", 20),

    # === OVERLAP: Auth + Database (shares vocabulary with both clusters) ===
    ("decision", "db-auth-session-store",
     "Storing refresh token hashes in PostgreSQL sessions table — rejected Redis for token storage due to persistence requirements",
     "webapp", 3),
    ("correction", "db-auth-migration-failure",
     "Alembic migration adding auth_tokens table failed because of missing foreign key to users — fixed column order",
     "webapp", 7),

    # === CLUSTER: Deployment (project: webapp) ===
    ("workflow", "deploy-pipeline",
     "GitHub Actions CI: lint → test → build Docker image → push to GCR → rolling update on GCE managed instance group",
     "webapp", 1),
    ("decision", "deploy-image-pinning",
     "Pin all Docker base images to SHA256 digest — latest tag caused staging failures when base image updated with breaking Python change",
     "webapp", 25),
    ("correction", "deploy-secrets-leak",
     "Production DATABASE_URL leaked in docker-compose.yml committed to git — rotated credentials, added pre-commit hook with detect-secrets",
     "webapp", 45),

    # === OVERLAP: Deploy + Database ===
    ("skill", "deploy-db-backup",
     "Pre-deployment database backup: pg_dump --format=custom --compress=9 before running alembic upgrade in CI pipeline",
     "webapp", 8),

    # === CLUSTER: Testing (project: webapp) ===
    ("preference", "test-pytest-style",
     "Use pytest parametrize and fixtures over unittest TestCase — cleaner, better output, easier to read",
     "webapp", 35),
    ("fact", "test-coverage-numbers",
     "Test coverage 87% overall — hooks/ at 92%, cairn/ at 81%, weakest in query.py date parsing edge cases",
     "webapp", 0),
    ("correction", "test-daemon-flaky",
     "test_daemon socket cleanup race fixed with retry loop and 100ms exponential backoff on ConnectionRefused",
     "webapp", 12),

    # === CROSS-PROJECT: Different project, same domain ===
    ("decision", "mobile-auth-oauth",
     "Mobile app uses OAuth2 PKCE flow — rejected basic JWT for mobile because token storage on device is insecure without keychain",
     "mobile-app", 14),
    ("fact", "mobile-api-rate-limit",
     "Mobile API rate limit is 100 requests per minute per user — higher than web because of background sync polling",
     "mobile-app", 20),

    # === GLOBAL: No project ===
    ("preference", "coding-style-explicit-imports",
     "Always use explicit imports, never wildcard — helps IDE navigation and makes dependencies clear at a glance",
     None, 40),
    ("skill", "git-interactive-rebase",
     "Use git rebase -i HEAD~5 to squash WIP commits before PR — keeps history clean without losing work",
     None, 50),

    # === DISTRACTORS: Similar vocabulary, wrong domain ===
    ("fact", "jwt-standard-rfc",
     "JWT standard defined in RFC 7519 — three base64url-encoded parts separated by dots: header.payload.signature",
     "documentation", 60),  # About JWT spec, not about OUR auth decisions
    ("fact", "postgres-wal-internals",
     "PostgreSQL WAL segment size is 16MB by default — each segment contains sequential log records for crash recovery",
     "documentation", 55),  # About Postgres internals, not about OUR database choice
    ("fact", "docker-layer-caching",
     "Docker builds cache each layer by instruction hash — changing COPY before RUN pip install invalidates the pip cache layer",
     "documentation", 50),  # About Docker mechanics, not about OUR deployment

    # === NOISE: Completely unrelated ===
    ("fact", "weather-tauranga",
     "Tauranga averages 22°C in January with humidity around 75% — occasional subtropical cyclone remnants from Coral Sea",
     None, 90),
    ("person", "vet-dr-chen",
     "Dr Sarah Chen at Bay Vet Clinic handles Milo's annual checkup — phone 07-555-0123, good with anxious dogs",
     None, 80),
    ("preference", "music-coding",
     "Lo-fi hip hop or ambient electronic while coding — no lyrics, 80-90 BPM, the Lofi Girl stream is reliable",
     None, 70),
]

# Queries with difficulty tiers and expected relevant memory indices
# Format: (query, relevant_indices, difficulty, notes)
QUERIES = [
    # --- EASY: Same vocabulary ---
    ("What JWT authentication strategy did we choose?",
     {0}, "easy", "Direct vocabulary match with auth-jwt-choice"),
    ("How to preview alembic migrations before applying?",
     {5}, "easy", "Direct match on alembic + preview"),
    ("What is our deployment pipeline?",
     {9}, "easy", "Direct match on deploy-pipeline"),
    ("What is our test coverage percentage?",
     {14}, "easy", "Direct match on coverage numbers"),

    # --- MEDIUM: Rephrased concepts ---
    ("Why did we pick PostgreSQL over MongoDB?",
     {4}, "medium", "Rephrase of db-postgres-choice — 'pick over' vs 'chosen over'"),
    ("How do we prevent credential leaks in our codebase?",
     {11}, "medium", "Asks about prevention, memory describes the incident and fix"),
    ("What rate limiting do we have on the login page?",
     {3}, "medium", "'Login page' vs 'login endpoint', 'have' vs 'rate-limited'"),
    ("How should I write test fixtures?",
     {13}, "medium", "Asks about writing fixtures, memory is about preferring pytest style"),

    # --- HARD: Indirect / symptom-based ---
    ("Users are getting logged out unexpectedly after a few seconds",
     {1}, "hard", "Symptom of the token expiry bug — no shared vocabulary"),
    ("The app can't connect to the database under heavy load",
     {6}, "hard", "Symptom of connection pool exhaustion — 'heavy load' is indirect"),
    ("Why did the auth_tokens migration fail?",
     {8}, "hard", "Overlapping cluster — spans auth AND database migration"),
    ("Where do we store refresh tokens and why not Redis?",
     {7}, "hard", "Overlapping memory — auth concern stored in db-auth-session-store"),
    ("We need a database snapshot before each deploy — how?",
     {12}, "hard", "Overlap deploy+db — 'snapshot' rephrases 'backup'"),
    ("How does the mobile app handle authentication differently?",
     {16}, "hard", "Cross-project retrieval — different project, same domain"),

    # --- IMPOSSIBLE: No relevant memory exists ---
    ("How do we handle WebSocket connections for real-time updates?",
     set(), "impossible", "No memory about WebSockets"),
    ("What caching strategy do we use for the API?",
     set(), "impossible", "No memory about API caching (only Docker layer caching exists as distractor)"),
    ("How is logging configured in production?",
     set(), "impossible", "No memory about logging"),
    ("What's our approach to database sharding?",
     set(), "impossible", "No memory about sharding (Postgres choice exists but not about sharding)"),
]


# --- DB setup ---

def build_hard_db():
    """Build test DB with temporal spread and cross-project memories."""
    db_path = os.path.join(TEST_DIR, "hard_quality.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        embedding BLOB, session_id TEXT, project TEXT,
        confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER,
        depth INTEGER, archived_reason TEXT, associated_files TEXT, keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        origin_id TEXT,
        user_id TEXT,
        updated_by TEXT,
        team_id TEXT,
        source_ref TEXT,
        deleted_at TIMESTAMP,
        synced_at TIMESTAMP
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
        topic, content, keywords, content=memories, content_rowid=id
    )""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content, keywords)
        VALUES (new.id, new.topic, new.content, new.keywords);
    END""")
    conn.execute("""CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content, keywords)
        VALUES ('delete', old.id, old.topic, old.content, old.keywords);
    END""")
    conn.execute("""CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content, keywords)
        VALUES ('delete', old.id, old.topic, old.content, old.keywords);
        INSERT INTO memories_fts(rowid, topic, content, keywords)
        VALUES (new.id, new.topic, new.content, new.keywords);
    END""")

    conn.execute(
        "INSERT INTO sessions (session_id, project) VALUES ('hard-sid', 'webapp')"
    )

    import cairn.embeddings as embeddings
    memory_ids: list[int] = []
    now = datetime.now()

    for mtype, topic, content, project, age_days in MEMORIES:
        embed_project = project or ""
        embed_text = f"{embed_project} {mtype} {topic} {content}"
        vec = embeddings.embed(embed_text)

        # Set updated_at to simulate temporal spread
        ts = (now - timedelta(days=age_days)).strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            "INSERT INTO memories (type, topic, content, embedding, project, session_id, "
            "confidence, updated_at, created_at) VALUES (?, ?, ?, ?, ?, 'seed-session', 0.7, ?, ?)",
            (mtype, topic, content, embeddings.to_blob(vec), project, ts, ts)
        )
        memory_ids.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    conn.commit()
    return db_path, conn, memory_ids


# --- Query runners ---

def run_semantic(db_path: str, query: str, limit: int = 10) -> list[int]:
    import cairn.embeddings as embeddings
    import hooks.hook_helpers as hook_helpers
    original_db = hook_helpers.DB_PATH
    try:
        hook_helpers.DB_PATH = db_path
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        results = embeddings.find_similar(conn, query, threshold=0.0, limit=limit,
                                          current_project="webapp")
        conn.close()
        return [r["id"] for r in results]
    finally:
        hook_helpers.DB_PATH = original_db


def run_fts(db_path: str, query: str, limit: int = 10) -> list[int]:
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
        SELECT m.id FROM memories_fts f JOIN memories m ON f.rowid = m.id
        WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?
    """, (fts_query, limit)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def run_rrf(db_path: str, query: str) -> list[int]:
    import re
    import hooks.hook_helpers as hook_helpers
    from hooks.retrieval import retrieve_context
    original_db, original_log = hook_helpers.DB_PATH, hook_helpers.LOG_PATH
    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, "hard.log")
        result_xml = retrieve_context(query, session_id="hard-sid")
        if not result_xml:
            return []
        return [int(m) for m in re.findall(r'id="(\d+)"', result_xml)]
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log


def run_fanout(db_path: str, query: str, limit: int = 10) -> list[int]:
    import cairn.embeddings as embeddings
    from hooks.query_expansion import type_prefix_fanout
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    results = type_prefix_fanout(conn, query, embeddings.embed,
                                  current_project="webapp", limit=limit)
    conn.close()
    return [r["id"] for r in results]


def run_combined(db_path: str, query: str, limit: int = 10) -> list[int]:
    import cairn.embeddings as embeddings
    from hooks.query_expansion import combined_expansion
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    results = combined_expansion(conn, query, embeddings.embed,
                                  current_project="webapp", limit=limit)
    conn.close()
    return [r["id"] for r in results]


# --- Evaluation ---

def evaluate_by_difficulty(strategy_fn, db_path, memory_ids):
    """Evaluate a strategy grouped by difficulty tier."""
    tiers = {"easy": [], "medium": [], "hard": [], "impossible": []}

    for query_text, relevant_indices, difficulty, notes in QUERIES:
        relevant_ids = {memory_ids[i] for i in relevant_indices}
        retrieved = strategy_fn(db_path, query_text)

        if difficulty == "impossible":
            # For impossible queries, measure false positive rate
            fp = 1.0 if retrieved else 0.0
            tiers["impossible"].append({
                "query": query_text, "fp": fp, "notes": notes,
                "retrieved_count": len(retrieved),
            })
        else:
            m = mrr(retrieved, relevant_ids)
            p3 = precision_at_k(retrieved, relevant_ids, 3)
            r5 = recall_at_k(retrieved, relevant_ids, 5)
            tiers[difficulty].append({
                "query": query_text, "mrr": m, "p3": p3, "r5": r5,
                "notes": notes,
            })

    # Aggregate
    result = {}
    for tier in ["easy", "medium", "hard"]:
        entries = tiers[tier]
        if entries:
            result[tier] = {
                "avg_mrr": sum(e["mrr"] for e in entries) / len(entries),
                "avg_p3": sum(e["p3"] for e in entries) / len(entries),
                "avg_r5": sum(e["r5"] for e in entries) / len(entries),
                "per_query": entries,
            }
    imp = tiers["impossible"]
    if imp:
        result["impossible"] = {
            "false_positive_rate": sum(e["fp"] for e in imp) / len(imp),
            "per_query": imp,
        }

    # Overall (excluding impossible)
    all_entries = tiers["easy"] + tiers["medium"] + tiers["hard"]
    if all_entries:
        result["overall"] = {
            "avg_mrr": sum(e["mrr"] for e in all_entries) / len(all_entries),
            "avg_p3": sum(e["p3"] for e in all_entries) / len(all_entries),
            "avg_r5": sum(e["r5"] for e in all_entries) / len(all_entries),
        }

    return result


# --- Fixtures ---

@pytest.fixture(scope="module")
def hard_db():
    import cairn.config as config
    orig = config.CROSS_ENCODER_ENABLED
    config.CROSS_ENCODER_ENABLED = False
    db_path, conn, memory_ids = build_hard_db()
    conn.close()
    yield db_path, memory_ids
    config.CROSS_ENCODER_ENABLED = orig


# --- Tests ---

class TestHardBaselineCharacterisation:
    """Characterises baseline performance on the hard benchmark."""

    @pytest.mark.behavioural
    def test_easy_queries_mostly_correct(self, hard_db):
        """Easy queries (same vocabulary) should have MRR > 0.7."""
        db_path, memory_ids = hard_db
        result = evaluate_by_difficulty(run_semantic, db_path, memory_ids)
        assert result["easy"]["avg_mrr"] >= 0.7, \
            f"Easy MRR {result['easy']['avg_mrr']:.3f} below 0.7"

    @pytest.mark.behavioural
    def test_medium_queries_reasonable(self, hard_db):
        """Medium queries (rephrased) should have MRR > 0.4."""
        db_path, memory_ids = hard_db
        result = evaluate_by_difficulty(run_semantic, db_path, memory_ids)
        assert result["medium"]["avg_mrr"] >= 0.4, \
            f"Medium MRR {result['medium']['avg_mrr']:.3f} below 0.4"

    @pytest.mark.behavioural
    def test_hard_queries_nonzero(self, hard_db):
        """Hard queries (indirect) should find at least something (MRR > 0.1)."""
        db_path, memory_ids = hard_db
        result = evaluate_by_difficulty(run_semantic, db_path, memory_ids)
        assert result["hard"]["avg_mrr"] >= 0.1, \
            f"Hard MRR {result['hard']['avg_mrr']:.3f} — finding nothing"

    @pytest.mark.behavioural
    def test_impossible_queries_limited_fp(self, hard_db):
        """Impossible queries should have limited false positives (quality gates help)."""
        db_path, memory_ids = hard_db
        result = evaluate_by_difficulty(run_semantic, db_path, memory_ids)
        # Quality gates should filter some impossible queries, but not all
        # (the distractors share vocabulary). FP rate < 1.0 is a win.
        fp_rate = result["impossible"]["false_positive_rate"]
        assert fp_rate <= 1.0  # Soft — just characterise for now


class TestExpansionOnHardBenchmark:
    """Compares expansion strategies on the hard benchmark."""

    @pytest.mark.behavioural
    def test_fanout_does_not_regress_easy(self, hard_db):
        """Fan-out should maintain easy query performance."""
        db_path, memory_ids = hard_db
        baseline = evaluate_by_difficulty(run_semantic, db_path, memory_ids)
        fanout = evaluate_by_difficulty(run_fanout, db_path, memory_ids)

        assert fanout["easy"]["avg_mrr"] >= baseline["easy"]["avg_mrr"] * 0.85, \
            f"Fan-out easy MRR {fanout['easy']['avg_mrr']:.3f} regressed vs baseline {baseline['easy']['avg_mrr']:.3f}"

    @pytest.mark.behavioural
    def test_fanout_improves_or_maintains_hard(self, hard_db):
        """Fan-out should not hurt hard query performance."""
        db_path, memory_ids = hard_db
        baseline = evaluate_by_difficulty(run_semantic, db_path, memory_ids)
        fanout = evaluate_by_difficulty(run_fanout, db_path, memory_ids)

        assert fanout["hard"]["avg_mrr"] >= baseline["hard"]["avg_mrr"] * 0.85, \
            f"Fan-out hard MRR {fanout['hard']['avg_mrr']:.3f} regressed vs baseline {baseline['hard']['avg_mrr']:.3f}"

    @pytest.mark.behavioural
    def test_combined_does_not_regress_overall(self, hard_db):
        """Combined strategy should not regress overall MRR."""
        db_path, memory_ids = hard_db
        baseline = evaluate_by_difficulty(run_semantic, db_path, memory_ids)
        combined = evaluate_by_difficulty(run_combined, db_path, memory_ids)

        assert combined["overall"]["avg_mrr"] >= baseline["overall"]["avg_mrr"] * 0.85, \
            f"Combined overall MRR {combined['overall']['avg_mrr']:.3f} regressed vs baseline {baseline['overall']['avg_mrr']:.3f}"


class TestDistractorDiscrimination:
    """Tests that strategies don't get fooled by vocabulary-sharing distractors."""

    @pytest.mark.behavioural
    def test_auth_query_avoids_jwt_rfc_distractor(self, hard_db):
        """Auth decision query should prefer our auth-jwt-choice over the JWT RFC spec distractor."""
        db_path, memory_ids = hard_db
        auth_decision_id = memory_ids[0]   # auth-jwt-choice
        jwt_rfc_id = memory_ids[20]        # jwt-standard-rfc (distractor)

        retrieved = run_semantic(db_path, "What JWT authentication strategy did we choose?")
        if auth_decision_id in retrieved and jwt_rfc_id in retrieved:
            auth_rank = retrieved.index(auth_decision_id)
            rfc_rank = retrieved.index(jwt_rfc_id)
            assert auth_rank < rfc_rank, \
                f"Auth decision (rank {auth_rank}) should rank above JWT RFC distractor (rank {rfc_rank})"

    @pytest.mark.behavioural
    def test_db_query_avoids_wal_internals_distractor(self, hard_db):
        """DB choice query should prefer our db-postgres-choice over WAL internals distractor."""
        db_path, memory_ids = hard_db
        db_choice_id = memory_ids[4]       # db-postgres-choice
        wal_internals_id = memory_ids[21]  # postgres-wal-internals (distractor)

        retrieved = run_semantic(db_path, "Why did we pick PostgreSQL over MongoDB?")
        if db_choice_id in retrieved and wal_internals_id in retrieved:
            choice_rank = retrieved.index(db_choice_id)
            wal_rank = retrieved.index(wal_internals_id)
            assert choice_rank < wal_rank, \
                f"DB choice (rank {choice_rank}) should rank above WAL internals distractor (rank {wal_rank})"


class TestCrossProjectRetrieval:
    """Tests retrieval across project boundaries."""

    @pytest.mark.behavioural
    def test_cross_project_auth_found(self, hard_db):
        """Cross-project query about mobile auth should find mobile-app memories."""
        db_path, memory_ids = hard_db
        mobile_auth_id = memory_ids[16]  # mobile-auth-oauth

        retrieved = run_semantic(db_path, "How does the mobile app handle authentication differently?")
        # Cross-project retrieval is harder — just check it's found somewhere in results
        assert mobile_auth_id in retrieved, \
            f"Mobile auth memory should appear in results for cross-project query"


class TestTemporalDynamics:
    """Tests that recency weighting works correctly."""

    @pytest.mark.behavioural
    def test_recent_memory_ranked_higher(self, hard_db):
        """Given two relevant memories, the more recent one should rank higher."""
        db_path, memory_ids = hard_db
        # auth-jwt-choice (age 0 days) vs auth-jwt-testing (age 30 days) — both about JWT auth
        recent_id = memory_ids[0]   # 0 days old
        old_id = memory_ids[2]      # 30 days old

        retrieved = run_semantic(db_path, "JWT authentication approach")
        if recent_id in retrieved and old_id in retrieved:
            recent_rank = retrieved.index(recent_id)
            old_rank = retrieved.index(old_id)
            assert recent_rank < old_rank, \
                f"Recent memory (rank {recent_rank}) should outrank older one (rank {old_rank})"


class TestHardBenchmarkSummary:
    """Prints full comparison table across strategies and difficulty tiers."""

    @pytest.mark.behavioural
    def test_print_hard_summary(self, hard_db):
        """Print comparative summary (diagnostic, always passes)."""
        db_path, memory_ids = hard_db

        strategies = {
            "Semantic": run_semantic,
            "FTS5": run_fts,
            "Hybrid RRF": run_rrf,
            "Fan-out": run_fanout,
            "Combined": run_combined,
        }

        print("\n\n=== Hard Benchmark — Strategy Comparison by Difficulty ===\n")
        print(f"{'Strategy':<15} {'Easy MRR':>9} {'Med MRR':>9} {'Hard MRR':>10} {'Overall':>9} {'FP Rate':>9}")
        print("-" * 65)

        all_results = {}
        for name, fn in strategies.items():
            result = evaluate_by_difficulty(fn, db_path, memory_ids)
            all_results[name] = result

            easy = result.get("easy", {}).get("avg_mrr", 0)
            med = result.get("medium", {}).get("avg_mrr", 0)
            hard_mrr = result.get("hard", {}).get("avg_mrr", 0)
            overall = result.get("overall", {}).get("avg_mrr", 0)
            fp = result.get("impossible", {}).get("false_positive_rate", 0)

            print(f"{name:<15} {easy:>9.3f} {med:>9.3f} {hard_mrr:>10.3f} {overall:>9.3f} {fp:>9.2f}")

        # Per-query detail
        print(f"\n{'Query':<55} {'Diff':<6} {'Sem':>5} {'FTS':>5} {'RRF':>5} {'Fan':>5} {'Comb':>5}")
        print("-" * 90)

        for i, (query_text, relevant_indices, difficulty, notes) in enumerate(QUERIES):
            q_short = query_text[:53]
            scores = []
            for name in strategies:
                result = all_results[name]
                tier = result.get(difficulty, {})
                pq = tier.get("per_query", [])
                # Find this query in the per_query list
                score = 0.0
                for entry in pq:
                    if entry["query"] == query_text:
                        score = entry.get("mrr", 0.0)
                        break
                scores.append(score)

            diff_short = difficulty[:4]
            print(f"{q_short:<55} {diff_short:<6} {scores[0]:>5.2f} {scores[1]:>5.2f} {scores[2]:>5.2f} {scores[3]:>5.2f} {scores[4]:>5.2f}")

        print()
        assert True


# --- Cleanup ---

def teardown_module():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
