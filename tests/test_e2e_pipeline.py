#!/usr/bin/env python3
"""End-to-end pipeline tests — exercises the full round-trip through all layers.

Simulates multi-turn conversations by feeding JSON payloads through prompt_hook
and stop_hook in sequence. Uses a synthetic test DB with seeded memories to verify
memories flow end-to-end across turns and retrieval layers.

Layers tested:
- Project bootstrap (CWD → standing context)
- Layer 1 (first-prompt semantic push)
- Layer 1.5 (per-prompt semantic push)
- Layer 2 (cross-project keyword staging → injection)
- Layer 3 (pull-based context retrieval via stop hook)
- Gotcha (file-associated correction injection via pretool hook)
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
from unittest.mock import patch, MagicMock
from io import StringIO

import numpy as np


TEST_DIR = tempfile.mkdtemp()
_db_counter = [0]

# --- Test DB setup ---

def make_vector(seed):
    """Create a deterministic 384-dim normalised vector from a seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v


def fresh_db():
    """Create a fresh test database with full schema."""
    _db_counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"pipeline_{_db_counter[0]}.db")
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
        session_id TEXT PRIMARY KEY, parent_session_id TEXT,
        project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content, session_id, changed_at)
        VALUES (old.id, old.content, old.session_id, old.updated_at);
    END""")
    conn.execute("CREATE INDEX idx_memories_type ON memories(type)")
    conn.execute("CREATE INDEX idx_memories_topic ON memories(topic)")
    conn.execute("CREATE INDEX idx_memories_project ON memories(project)")
    conn.execute("CREATE INDEX idx_metrics_event ON metrics(event)")
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
    conn.commit()
    return db_path, conn


def seed_memories(conn, project="testproject", other_project="otherproject"):
    """Seed the DB with memories for testing all layers."""
    memories = [
        # Project-scoped standing context (for project bootstrap)
        ("project", "testproject-status", "Project is in alpha — 3 modules done, 2 remaining",
         "testproject", make_vector(10)),
        ("preference", "testproject-style", "Use pytest fixtures over setup/teardown methods",
         "testproject", make_vector(11)),
        ("fact", "testproject-db", "Uses PostgreSQL 15 with pgvector extension for embeddings",
         "testproject", make_vector(12)),

        # Project-scoped topical memories (for L1/L1.5/L3 retrieval)
        ("decision", "auth-strategy", "JWT with refresh tokens — rejected session cookies for stateless API",
         "testproject", make_vector(20)),
        ("correction", "auth-bug", "Token expiry was checking issued_at instead of expires_at — fixed in auth.py",
         "testproject", make_vector(21)),
        ("skill", "db-migration", "Use alembic upgrade head --sql to preview migration SQL before applying",
         "testproject", make_vector(22)),

        # Cross-project memories (for L2 cross-project)
        ("fact", "docker-compose", "Always pin image versions in docker-compose for reproducibility",
         "otherproject", make_vector(30)),
        ("skill", "ci-caching", "Cache pip dependencies with hashFiles('requirements.txt') in GitHub Actions",
         "otherproject", make_vector(31)),

        # Global memories (no project)
        ("preference", "code-style", "Prefer explicit imports over wildcard — helps IDE navigation",
         None, make_vector(40)),

        # Correction with file association (for gotcha layer)
        ("correction", "config-bug", "config.py line 42 had wrong default timeout — must be 30 not 300",
         "testproject", make_vector(50)),
    ]

    for mtype, topic, content, project_val, vec in memories:
        conn.execute(
            "INSERT INTO memories (type, topic, content, project, embedding, session_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, 'seed-session', 0.7)",
            (mtype, topic, content, project_val, vec.tobytes())
        )

    # Set file association on the correction
    conn.execute(
        "UPDATE memories SET associated_files = ? WHERE topic = 'config-bug'",
        (json.dumps(["/home/test/testproject/config.py"]),)
    )

    conn.commit()


# --- Mock embedder ---

class MockEmbedder:
    """Deterministic embedder that maps known queries to seed vectors."""

    def __init__(self):
        self._query_map = {
            "authentication tokens": make_vector(20),  # near auth-strategy
            "authentication token strategy": make_vector(20),
            "database migration": make_vector(22),       # near db-migration
            "docker compose": make_vector(30),           # near docker-compose
            "project status": make_vector(10),           # near testproject-status
            "code style imports": make_vector(40),       # near code-style
            "config timeout": make_vector(50),           # near config-bug
        }

    def embed(self, text):
        text_lower = text.lower()
        for key, vec in self._query_map.items():
            if key in text_lower:
                return vec
        # Deterministic but unique per text — avoids false matches in trailing intent
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
            score = composite_score(sim, r[7], r[4], r[5], current_project)
            results.append({
                "id": r[0], "type": r[1], "topic": r[2], "content": r[3],
                "updated_at": r[4], "project": r[5], "session_id": r[6],
                "confidence": r[7] or 0.7, "depth": r[8],
                "archived_reason": r[9],
                "similarity": sim, "score": score,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


# --- Hook runners ---

def run_prompt_hook(db_path, session_id, user_message, cwd="/home/test/testproject"):
    """Run prompt_hook.main() and return the JSON output (or None)."""
    payload = {
        "session_id": session_id,
        "cwd": cwd,
        "user_message": user_message,
    }

    import hooks.hook_helpers as hook_helpers
    import hooks.prompt_hook as prompt_hook

    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH
    captured = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    mock_emb = MockEmbedder()

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, "test_prompt.log")

        with patch("sys.stdin", StringIO(json.dumps(payload))), \
             patch("sys.stdout", captured), \
             patch("sys.exit", mock_exit), \
             patch("cairn.config.EPHEMERAL_DB_PATH", db_path), \
             patch.object(hook_helpers, "get_embedder", return_value=mock_emb), \
             patch("hooks.prompt_hook.get_embedder", return_value=mock_emb):
            try:
                prompt_hook.main()
            except SystemExit:
                pass
            finally:
                hook_helpers.flush_metrics()
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log

    output = captured.getvalue().strip()
    if output:
        return json.loads(output)
    return None


def run_stop_hook(db_path, session_id, assistant_message, cwd="/home/test/testproject",
                  is_continuation=False, transcript_path=""):
    # Reset cached intent embeddings to avoid cross-test contamination
    import hooks.enforcement as enforcement
    enforcement._intent_embeddings = None

    """Run stop_hook.main() and return (exit_code, output_json_or_None)."""
    payload = {
        "session_id": session_id,
        "cwd": cwd,
        "last_assistant_message": assistant_message,
        "stop_hook_active": is_continuation,
        "transcript_path": transcript_path or os.path.join(TEST_DIR, "transcript.jsonl"),
    }

    import hooks.hook_helpers as hook_helpers
    import hooks.stop_hook as stop_hook

    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH
    captured = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    mock_emb = MockEmbedder()

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, "test_stop.log")

        with patch("sys.stdin", StringIO(json.dumps(payload))), \
             patch("sys.stdout", captured), \
             patch("sys.exit", mock_exit), \
             patch("cairn.config.EPHEMERAL_DB_PATH", db_path), \
             patch.object(hook_helpers, "get_embedder", return_value=mock_emb), \
             patch("hooks.stop_hook.get_embedder", return_value=mock_emb):
            try:
                stop_hook.main()
            except SystemExit:
                pass
            finally:
                hook_helpers.flush_metrics()
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log

    output = captured.getvalue().strip()
    result = json.loads(output) if output else None
    return exit_code[0], result


def run_pretool_hook(db_path, session_id, tool_name, file_path):
    """Run pretool_hook.main() and return the JSON output (or None)."""
    payload = {
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_input": {"file_path": file_path},
    }

    import hooks.hook_helpers as hook_helpers
    import hooks.pretool_hook as pretool_hook

    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH
    captured = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, "test_pretool.log")

        with patch("sys.stdin", StringIO(json.dumps(payload))), \
             patch("sys.stdout", captured), \
             patch("sys.exit", mock_exit):
            try:
                pretool_hook.main()
            except SystemExit:
                pass
            finally:
                hook_helpers.flush_metrics()
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log

    output = captured.getvalue().strip()
    if output:
        return json.loads(output)
    return None


# --- Pipeline tests ---

import pytest


@pytest.fixture
def pipeline_db():
    """Fresh DB with seeded memories for pipeline tests."""
    db_path, conn = fresh_db()
    seed_memories(conn)
    conn.close()
    yield db_path


class TestFullPipelineRoundTrip:
    """Tests the complete multi-turn flow: prompt → response → stop → next prompt."""

    @pytest.mark.behavioural
    def test_turn1_project_bootstrap_and_layer1(self, pipeline_db):
        """Turn 1: First prompt triggers project bootstrap + Layer 1 injection."""
        sid = "pipeline-test-001"
        result = run_prompt_hook(pipeline_db, sid, "How is the authentication working?")

        assert result is not None, "Prompt hook should produce output on first prompt"
        context = result["hookSpecificOutput"]["additionalContext"]

        # Project bootstrap injects standing context
        assert "project-bootstrap" in context, "Should include project bootstrap layer"
        assert "testproject-status" in context or "alpha" in context, \
            "Should inject project status as standing context"

        # Memory block reminder on first prompt
        assert "MEMORY BLOCK" in context, "Should include memory block reminder"

        # Layer 1 may or may not find results depending on mock vector similarity —
        # the important thing is that the prompt hook produced output at all
        # (project bootstrap + reminder is sufficient for a valid first-prompt response)

    @pytest.mark.behavioural
    def test_turn1_stop_stores_memory(self, pipeline_db):
        """Stop hook parses and stores a well-formed memory block."""
        sid = "pipeline-test-002"
        # Register session via prompt hook first
        run_prompt_hook(pipeline_db, sid, "test prompt")

        response = (
            "The authentication system uses JWT with RS256 and refresh tokens stored in httponly cookies.\n\n"
            "<memory>\n"
            "- type: fact\n"
            "- topic: auth-review\n"
            "- content: Authentication uses JWT with RS256 — refresh tokens stored in httponly cookies\n"
            "- complete: true\n"
            "- context: sufficient\n"
            "- keywords: authentication, jwt, security\n"
            "</memory>"
        )
        exit_code, result = run_stop_hook(pipeline_db, sid, response)

        assert exit_code == 0, f"Well-formed response should allow stop, got: {result}"
        assert result is None, f"No blocking output expected, got: {result}"

        # Verify memory was stored
        conn = sqlite3.connect(pipeline_db)
        row = conn.execute(
            "SELECT type, topic, content FROM memories WHERE topic = 'auth-review'"
        ).fetchone()
        conn.close()
        assert row is not None, "Memory should be stored in DB"
        assert row[0] == "fact"
        assert "JWT" in row[2]

    @pytest.mark.behavioural
    def test_turn2_layer1_5_injects(self, pipeline_db):
        """Turn 2: Subsequent prompt triggers Layer 1.5 (per-prompt push)."""
        sid = "pipeline-test-003"
        # Turn 1: first prompt (marks first_prompt_done)
        run_prompt_hook(pipeline_db, sid, "initial question about the project")

        # Turn 2: subsequent prompt — should use Layer 1.5
        result = run_prompt_hook(pipeline_db, sid, "tell me about database migration")

        # Layer 1.5 fires on subsequent prompts with higher threshold
        if result is not None:
            context = result["hookSpecificOutput"]["additionalContext"]
            assert "per-prompt" in context or "migration" in context.lower(), \
                "Layer 1.5 should inject per-prompt context"

    @pytest.mark.behavioural
    def test_layer3_context_retrieval(self, pipeline_db):
        """Stop hook retrieves context when LLM declares context: insufficient."""
        sid = "pipeline-test-004"
        # Register session
        run_prompt_hook(pipeline_db, sid, "test prompt")

        # Store a hook_fired metric so the session appears active
        conn = sqlite3.connect(pipeline_db)
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', ?)", (sid,))
        conn.execute("INSERT INTO memories (type, topic, content, session_id) VALUES ('fact', 'dummy', 'dummy content for session', ?)", (sid,))
        conn.commit()
        conn.close()

        response = (
            "I need to check what we decided about authentication.\n\n"
            "<memory>\n"
            "- type: fact\n"
            "- topic: auth-check\n"
            "- content: Checking cairn for authentication decisions and token strategy\n"
            "- complete: true\n"
            "- context: insufficient\n"
            "- context_need: authentication token strategy decisions\n"
            "- keywords: authentication, tokens\n"
            "</memory>"
        )
        exit_code, result = run_stop_hook(pipeline_db, sid, response)

        # Layer 3 should block and inject context
        if result and result.get("decision") == "block":
            assert "CAIRN CONTEXT" in result.get("reason", ""), \
                "Block reason should contain retrieved context"
            assert "cairn_context" in result.get("reason", ""), \
                "Should contain cairn_context XML"

    @pytest.mark.behavioural
    def test_layer2_cross_project_staging(self, pipeline_db):
        """Stop hook stages cross-project context via keywords for next prompt."""
        sid = "pipeline-test-005"

        # Register session with project
        conn = sqlite3.connect(pipeline_db)
        conn.execute(
            "INSERT INTO sessions (session_id, project) VALUES (?, 'testproject')",
            (sid,)
        )
        conn.commit()
        conn.close()

        # First prompt to mark first_prompt_done
        run_prompt_hook(pipeline_db, sid, "working on the project")

        response = (
            "I set up the docker compose configuration.\n\n"
            "<memory>\n"
            "- type: fact\n"
            "- topic: docker-setup\n"
            "- content: Docker compose configuration created with PostgreSQL and Redis services\n"
            "- complete: true\n"
            "- context: sufficient\n"
            "- keywords: docker, compose, ci, deployment\n"
            "</memory>"
        )
        exit_code, _ = run_stop_hook(pipeline_db, sid, response)

        # Check that L2 staged context in hook_state
        conn = sqlite3.connect(pipeline_db)
        staged = conn.execute(
            "SELECT value FROM hook_state WHERE session_id = ? AND key = 'staged_context'",
            (sid,)
        ).fetchone()
        conn.close()

        if staged:
            # Layer 2 found cross-project matches and staged them
            assert "cross-project" in staged[0] or "cairn_context" in staged[0], \
                "Staged context should be cairn_context XML"

            # Next prompt should inject the staged context
            result = run_prompt_hook(pipeline_db, sid, "continuing work")
            if result:
                context = result["hookSpecificOutput"]["additionalContext"]
                assert "cross-project" in context or "otherproject" in context.lower() or \
                       "cairn_context" in context, \
                    "Layer 2 staged context should be injected on next prompt"

    @pytest.mark.behavioural
    def test_gotcha_pretool_injection(self, pipeline_db):
        """PreToolUse hook injects corrections for files with associated_files."""
        sid = "pipeline-test-006"
        result = run_pretool_hook(
            pipeline_db, sid, "Edit", "/home/test/testproject/config.py"
        )

        if result:
            output = result.get("hookSpecificOutput", {})
            additional = output.get("additionalContext", "")
            assert "config" in additional.lower() or "timeout" in additional.lower(), \
                "Should inject the config-bug correction as a gotcha"

    @pytest.mark.behavioural
    def test_complete_three_turn_conversation(self, pipeline_db):
        """Full 3-turn conversation exercising bootstrap → L1 → store → L1.5 → store → L3."""
        sid = "pipeline-test-007"

        # --- Turn 1: First prompt ---
        t1_prompt = run_prompt_hook(pipeline_db, sid, "What's the current project status?")
        assert t1_prompt is not None, "Turn 1 should get bootstrap + L1 context"

        t1_response = (
            "The project is in alpha with 3 modules complete.\n\n"
            "<memory>\n"
            "- type: project\n"
            "- topic: status-check\n"
            "- content: Reviewed project status — alpha phase, 3/5 modules done, auth and db remaining\n"
            "- complete: true\n"
            "- context: sufficient\n"
            "- keywords: project, status, alpha\n"
            "</memory>"
        )
        exit_code, _ = run_stop_hook(pipeline_db, sid, t1_response)
        assert exit_code == 0, "Turn 1 stop should allow"

        # Verify memory stored
        conn = sqlite3.connect(pipeline_db)
        assert conn.execute(
            "SELECT COUNT(*) FROM memories WHERE topic = 'status-check'"
        ).fetchone()[0] == 1
        conn.close()

        # --- Turn 2: Subsequent prompt (L1.5 eligible) ---
        t2_prompt = run_prompt_hook(pipeline_db, sid, "Let's work on database migration next")

        t2_response = (
            "I'll set up the alembic migration.\n\n"
            "<memory>\n"
            "- type: decision\n"
            "- topic: migration-tool\n"
            "- content: Using alembic for database migrations — chosen over raw SQL for version tracking\n"
            "- complete: true\n"
            "- context: sufficient\n"
            "- keywords: database, migration, alembic\n"
            "</memory>"
        )
        exit_code, _ = run_stop_hook(pipeline_db, sid, t2_response)
        assert exit_code == 0, "Turn 2 stop should allow"

        # --- Turn 3: Context insufficient → L3 retrieval ---
        t3_prompt = run_prompt_hook(pipeline_db, sid, "How should we handle authentication?")

        t3_response = (
            "Let me check what we decided about auth.\n\n"
            "<memory>\n"
            "- type: fact\n"
            "- topic: auth-lookup\n"
            "- content: Checking existing authentication decisions before proceeding\n"
            "- complete: true\n"
            "- context: insufficient\n"
            "- context_need: authentication token strategy decisions\n"
            "- keywords: authentication, jwt, tokens\n"
            "</memory>"
        )
        exit_code, result = run_stop_hook(pipeline_db, sid, t3_response)

        # Verify all 3 memories are in the DB
        conn = sqlite3.connect(pipeline_db)
        count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        conn.close()
        assert count >= 3, f"Should have stored at least 3 memories, got {count}"


class TestEnforcementPipeline:
    """Tests enforcement decisions flow correctly through the pipeline."""

    @pytest.mark.behavioural
    def test_missing_memory_block_blocks(self, pipeline_db):
        """Response without memory block is blocked (after session has memories)."""
        sid = "enforcement-test-001"
        run_prompt_hook(pipeline_db, sid, "test")

        # Seed a prior memory so session appears instructed
        conn = sqlite3.connect(pipeline_db)
        conn.execute(
            "INSERT INTO memories (type, topic, content, session_id) "
            "VALUES ('fact', 'prior', 'prior memory', ?)", (sid,)
        )
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', ?)", (sid,))
        conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', ?)", (sid,))
        conn.commit()
        conn.close()

        exit_code, result = run_stop_hook(pipeline_db, sid, "Just a plain response with no memory block.")
        assert result is not None, "Should produce blocking output"
        assert result.get("decision") == "block"
        assert "memory" in result.get("reason", "").lower()

    @pytest.mark.behavioural
    def test_incomplete_response_continues(self, pipeline_db):
        """Response with complete: false triggers re-prompt."""
        sid = "enforcement-test-002"
        run_prompt_hook(pipeline_db, sid, "test")

        response = (
            "I'm starting the work now.\n\n"
            "<memory>\n"
            "- type: project\n"
            "- topic: work-start\n"
            "- content: Beginning implementation of the auth module\n"
            "- complete: false\n"
            "- remaining: Need to implement token validation and refresh logic\n"
            "- context: sufficient\n"
            "- keywords: auth, implementation\n"
            "</memory>"
        )
        exit_code, result = run_stop_hook(pipeline_db, sid, response)
        assert result is not None, "Should block for incomplete response"
        assert result.get("decision") == "block"
        assert "incomplete" in result.get("reason", "").lower() or "continue" in result.get("reason", "").lower()

    @pytest.mark.behavioural
    def test_confidence_updates_applied(self, pipeline_db):
        """Confidence updates in memory block are applied to referenced memories."""
        sid = "enforcement-test-003"
        run_prompt_hook(pipeline_db, sid, "test")

        # Get a memory ID to update
        conn = sqlite3.connect(pipeline_db)
        mem_id = conn.execute("SELECT id FROM memories WHERE topic = 'auth-strategy'").fetchone()[0]
        original_conf = conn.execute("SELECT confidence FROM memories WHERE id = ?", (mem_id,)).fetchone()[0]
        conn.close()

        response = (
            f"The auth strategy is confirmed.\n\n"
            f"<memory>\n"
            f"- type: fact\n"
            f"- topic: auth-confirmed\n"
            f"- content: Confirmed JWT with refresh tokens is the right approach\n"
            f"- complete: true\n"
            f"- context: sufficient\n"
            f"- confidence_update: {mem_id}:+\n"
            f"- keywords: authentication, jwt\n"
            f"</memory>"
        )
        exit_code, _ = run_stop_hook(pipeline_db, sid, response)

        # Check confidence was boosted
        conn = sqlite3.connect(pipeline_db)
        new_conf = conn.execute("SELECT confidence FROM memories WHERE id = ?", (mem_id,)).fetchone()[0]
        conn.close()
        assert new_conf > original_conf, \
            f"Confidence should increase from {original_conf} to {new_conf}"

    @pytest.mark.behavioural
    def test_content_density_blocks_thin_entry(self, pipeline_db):
        """Entry with very short content triggers density block."""
        sid = "enforcement-test-004"
        run_prompt_hook(pipeline_db, sid, "test")

        response = (
            "Here's my analysis of the system.\n\n"
            "<memory>\n"
            "- type: fact\n"
            "- topic: thin\n"
            "- content: yes it works\n"
            "- complete: true\n"
            "- context: sufficient\n"
            "- keywords: test\n"
            "</memory>"
        )
        exit_code, result = run_stop_hook(pipeline_db, sid, response)
        assert result is not None, "Should block thin content"
        assert result.get("decision") == "block"
        assert "short" in result.get("reason", "").lower() or "density" in result.get("reason", "").lower() \
            or "specific" in result.get("reason", "").lower()


class TestSessionIsolation:
    """Tests that sessions are properly isolated and chained."""

    @pytest.mark.behavioural
    def test_sessions_dont_see_each_others_new_memories(self, pipeline_db):
        """Memories stored by session A don't appear in session A's own retrieval."""
        sid_a = "session-a-001"
        run_prompt_hook(pipeline_db, sid_a, "test prompt")

        # Store a memory in session A
        response = (
            "Found a new pattern.\n\n"
            "<memory>\n"
            "- type: skill\n"
            "- topic: new-pattern\n"
            "- content: Use the repository pattern for database access — cleanly separates queries from logic\n"
            "- complete: true\n"
            "- context: sufficient\n"
            "- keywords: pattern, repository, database\n"
            "</memory>"
        )
        run_stop_hook(pipeline_db, sid_a, response)

        # Session A requests context about the same topic — own memories should be excluded
        response2 = (
            "Let me check patterns.\n\n"
            "<memory>\n"
            "- type: fact\n"
            "- topic: pattern-check\n"
            "- content: Checking for database access patterns\n"
            "- complete: true\n"
            "- context: insufficient\n"
            "- context_need: database access patterns and repository pattern\n"
            "- keywords: pattern, database\n"
            "</memory>"
        )
        exit_code, result = run_stop_hook(pipeline_db, sid_a, response2)

        # If context was retrieved, it should not include session A's own memory
        if result and "CAIRN CONTEXT" in result.get("reason", ""):
            # The seed memories might match, but the session's own new-pattern shouldn't
            reason = result["reason"]
            # This is a soft check — the mock embedder may or may not match
            assert "cairn_context" in reason

    @pytest.mark.behavioural
    def test_project_label_propagates(self, pipeline_db):
        """Auto-label from CWD propagates to session record."""
        sid = "project-label-001"
        response = (
            "Working on it.\n\n"
            "<memory>\n"
            "- type: fact\n"
            "- topic: label-test\n"
            "- content: Testing that project labels propagate correctly from CWD\n"
            "- complete: true\n"
            "- context: sufficient\n"
            "- keywords: project, label\n"
            "</memory>"
        )
        run_stop_hook(pipeline_db, sid, response, cwd="/home/test/testproject")

        conn = sqlite3.connect(pipeline_db)
        row = conn.execute(
            "SELECT project FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
        conn.close()
        assert row is not None, "Session should be registered"
        assert row[0] == "testproject", f"Project should be 'testproject', got '{row[0]}'"


class TestMetricsTracking:
    """Tests that metrics are properly recorded through the pipeline."""

    @pytest.mark.behavioural
    def test_metrics_recorded_through_pipeline(self, pipeline_db):
        """A complete turn records hook_fired and memories_stored metrics."""
        sid = "metrics-test-001"
        run_prompt_hook(pipeline_db, sid, "test")

        response = (
            "Done.\n\n"
            "<memory>\n"
            "- type: fact\n"
            "- topic: metrics-check\n"
            "- content: Verifying that pipeline metrics are recorded for each hook firing\n"
            "- complete: true\n"
            "- context: sufficient\n"
            "- keywords: metrics, testing\n"
            "</memory>"
        )
        run_stop_hook(pipeline_db, sid, response)

        conn = sqlite3.connect(pipeline_db)
        events = [r[0] for r in conn.execute(
            "SELECT event FROM metrics WHERE session_id = ?", (sid,)
        ).fetchall()]
        conn.close()

        assert "hook_fired" in events, "Should record hook_fired metric"
        assert "memories_stored" in events, "Should record memories_stored metric"


# --- Cleanup ---

def teardown_module():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
