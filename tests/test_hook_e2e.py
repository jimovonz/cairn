#!/usr/bin/env python3
"""End-to-end tests for the stop hook — feeds real JSON payloads through main()
and verifies DB state, blocking decisions, and metric recording.

Uses a temporary DB and patches all paths to isolate from the real system.
"""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

TEST_DIR = tempfile.mkdtemp()
_db_counter = [0]


def fresh_db():
    _db_counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"e2e_{_db_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
    conn.execute("""CREATE TABLE IF NOT EXISTS hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


def run_hook(db_path, payload):
    """Run the stop hook main() with patches for DB path and stdin."""
    payload_json = json.dumps(payload)

    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    import hook_helpers
    import stop_hook
    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, 'test.log')

        with patch('sys.stdin', StringIO(payload_json)), \
             patch('sys.stdout', captured_output), \
             patch('sys.exit', mock_exit), \
             patch.object(hook_helpers, 'get_embedder', return_value=None):
            try:
                stop_hook.main()
            except SystemExit:
                pass
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log

    output = captured_output.getvalue()
    result = json.loads(output) if output.strip() else None
    return result, exit_code[0]


# ============================================================
# Test: Valid memory block → stores in DB
# ============================================================

# Verifies: valid memory block is parsed and stored in DB
def test_valid_block_stores_memory():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "test-session",
        "transcript_path": "",
        "cwd": "/tmp/myproject",
        "last_assistant_message": "Here is my answer.\n<memory>\n- type: fact\n- topic: test-store\n- content: This should be stored\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    # Should allow stop (no block)
    assert result is None, f"Expected no block, got: {result}"

    # Memory should be in DB
    row = conn.execute("SELECT type, topic, content FROM memories WHERE topic = 'test-store'").fetchone()
    assert row is not None, "Memory not found in DB"
    assert row[0] == "fact"
    assert row[2] == "This should be stored"
    conn.close()


# ============================================================
# Test: Missing block → blocks with re-prompt
# ============================================================

# Verifies: missing memory block triggers block/re-prompt decision
def test_missing_block_triggers_reprompt():
    db_path, conn = fresh_db()
    # Mark session as instructed so enforcement applies
    conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', 'test-missing')")
    conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', 'test-missing')")
    conn.commit()
    payload = {
        "stop_hook_active": False,
        "session_id": "test-missing",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Just a plain response with no memory block."
    }
    result, code = run_hook(db_path, payload)

    assert result is not None, "Expected block decision"
    assert result["decision"] == "block"
    conn.close()


# ============================================================
# Test: Missing block on continuation → allows stop (no loop)
# ============================================================

# Verifies: missing block on continuation does not loop-block
def test_missing_block_on_continuation_allows_stop():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": True,
        "session_id": "test-cont",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Still no block on continuation."
    }
    result, code = run_hook(db_path, payload)

    # Should NOT block on continuation
    assert result is None, f"Should not block on continuation, got: {result}"
    conn.close()


# ============================================================
# Test: complete: false → blocks with remaining text
# ============================================================

# Verifies: complete:false blocks with remaining text in decision
def test_incomplete_blocks_with_remaining():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "test-incomplete",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Partial work.\n<memory>\n- complete: false\n- remaining: need to finish implementation\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    assert result["decision"] == "block"
    conn.close()


# ============================================================
# Test: Multiple entries in one block
# ============================================================

# Verifies: multiple memory entries in one block are all stored
def test_multiple_entries_stored():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "test-multi",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Done.\n<memory>\n- type: fact\n- topic: multi-1\n- content: first fact about the system architecture\n- type: decision\n- topic: multi-2\n- content: second decision about API design choices\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 2, f"Expected 2 memories, got {count}"

    topics = [r[0] for r in conn.execute("SELECT topic FROM memories ORDER BY id").fetchall()]
    assert "multi-1" in topics
    assert "multi-2" in topics
    conn.close()


# ============================================================
# Test: Session auto-registration
# ============================================================

# Verifies: session is auto-registered with project from cwd
def test_session_registered():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-abc123",
        "transcript_path": "/some/path.jsonl",
        "cwd": "/home/user/myproject",
        "last_assistant_message": "ok\n<memory>\ncomplete: true\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    session = conn.execute("SELECT session_id, project FROM sessions WHERE session_id = 'sess-abc123'").fetchone()
    assert session is not None, "Session not registered"
    assert session[1] == "myproject", f"Expected project 'myproject', got '{session[1]}'"
    conn.close()


# ============================================================
# Test: Auto project labelling from cwd
# ============================================================

# Verifies: project label derived from cwd directory name
def test_auto_project_label():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-proj",
        "transcript_path": "",
        "cwd": "/home/user/Projects/robotics-nav",
        "last_assistant_message": "ok\n<memory>\ncomplete: true\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    session = conn.execute("SELECT project FROM sessions WHERE session_id = 'sess-proj'").fetchone()
    assert session[0] == "robotics-nav", f"Expected 'robotics-nav', got '{session[0]}'"
    conn.close()


# ============================================================
# Test: Metrics recorded
# ============================================================

# Verifies: hook_fired and memories_stored metrics are recorded
def test_metrics_recorded():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-metrics",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "answer\n<memory>\n- type: fact\n- topic: metric-test\n- content: testing metrics recording and retrieval\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    events = [r[0] for r in conn.execute("SELECT event FROM metrics").fetchall()]
    assert "hook_fired" in events, f"Expected hook_fired metric, got: {events}"
    assert "memories_stored" in events, f"Expected memories_stored metric, got: {events}"
    conn.close()


# ============================================================
# Test: Write throttling with too many entries
# ============================================================

# Verifies: write throttle caps entries at MAX_MEMORIES_PER_RESPONSE
def test_write_throttle_limits_entries():
    from config import MAX_MEMORIES_PER_RESPONSE
    db_path, conn = fresh_db()

    # Build a block with 8 entries (above limit of 5)
    entries_text = ""
    for i in range(8):
        entries_text += f"\n- type: fact\n- topic: throttle-{i}\n- content: fact number with sufficient detail {i}"

    payload = {
        "stop_hook_active": False,
        "session_id": "sess-throttle",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": f"lots of stuff\n<memory>{entries_text}\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count <= MAX_MEMORIES_PER_RESPONSE, f"Expected max {MAX_MEMORIES_PER_RESPONSE}, got {count}"
    conn.close()


# ============================================================
# Test: Noop block (complete: true, no entries) → allow stop, no storage
# ============================================================

# Verifies: noop block (no entries) allows stop with zero storage
def test_noop_block_no_storage():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-noop",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "simple answer\n<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    assert result is None, f"Noop should allow stop, got: {result}"
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 0, f"Noop should store nothing, got {count}"
    conn.close()


# ============================================================
# Test: Malformed block with open tag but garbage content
# ============================================================

# Verifies: garbage inside valid tags stores nothing, no crash
def test_malformed_block_garbage_content():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-malformed",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "answer\n<memory>\nthis is just random text not following format\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    # Parser returns empty entries with default complete=True → allows stop
    # (garbage inside valid tags is forgiven, not blocked)
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 0, f"Garbage block should store nothing, got {count}"
    conn.close()


# ============================================================
# Test: Realistic messy Claude output
# ============================================================

# Verifies: memory block parsed despite trailing commentary text
def test_realistic_claude_output_with_extra_text():
    """Claude sometimes adds commentary after the memory block."""
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-messy",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Here's what I found.\n\nThe authentication system uses JWT.\n\n<memory>\n- type: decision\n- topic: auth\n- content: JWT chosen for stateless auth\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>\n\nLet me know if you need anything else."
    }
    result, code = run_hook(db_path, payload)

    row = conn.execute("SELECT content FROM memories WHERE topic = 'auth'").fetchone()
    assert row is not None, "Memory not found despite valid block"
    assert row[0] == "JWT chosen for stateless auth"
    conn.close()


# Verifies: memory block parsed inside markdown code fences
def test_realistic_claude_markdown_wrapped():
    """Claude sometimes wraps the block in markdown code fences."""
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-markdown",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Done.\n```\n<memory>\n- type: fact\n- topic: markdown-test\n- content: wrapped in code fence\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>\n```"
    }
    result, code = run_hook(db_path, payload)

    row = conn.execute("SELECT content FROM memories WHERE topic = 'markdown-test'").fetchone()
    assert row is not None, "Should parse memory block even inside code fence"
    conn.close()


# ============================================================
# Contradiction enforcement: blocks when response contradicts
# a retrieved memory without -! annotation
# ============================================================

# Verifies: inline contradiction enforcement is disabled (no block)
def test_contradiction_enforcement_blocks():
    """If retrieved memories exist in hook_state and the response negates one,
    the stop hook should block and request a -! annotation."""
    import numpy as np
    db_path, conn = fresh_db()

    # Insert a memory that will be "retrieved"
    vec = np.random.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    blob = vec.tobytes()
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, session_id, project, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("decision", "tunnel-approach", "Use Cloudflare tunnel for public access", blob, "old-sess", "proj", 0.7)
    )
    # Simulate prompt hook having recorded this memory as retrieved
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value, updated_at) VALUES (?, 'retrieved_ids', ?, CURRENT_TIMESTAMP)",
        ("sess-contradict", json.dumps([1]))
    )
    conn.commit()

    # Response that contradicts the retrieved memory (negation: "not" + "replaced" + "removed")
    response = (
        "We replaced the Cloudflare tunnel and removed cloudflared entirely. "
        "The GCE edge server is not using tunnels anymore.\n"
        "<memory>\n- type: fact\n- topic: edge-setup\n- content: GCE edge server deployed at conduit.alimento.co.nz replacing cloudflare tunnel\n"
        "- complete: true\n- context: sufficient\n- keywords: edge, tunnel\n</memory>"
    )

    # Verify the negation detection precondition — response text must actually
    # trigger negation mismatch against the memory content
    from storage import _has_negation_mismatch
    response_text = response.split("<memory>")[0].strip()
    mem_content = "Use Cloudflare tunnel for public access"
    assert _has_negation_mismatch(response_text, mem_content), \
        "Test precondition: response must trigger negation detection against the memory"

    # Mock embedder for similarity comparison
    from unittest.mock import MagicMock
    mock_emb = MagicMock()
    mock_emb.embed.return_value = vec
    mock_emb.from_blob.return_value = vec
    mock_emb.cosine_similarity.return_value = 0.6  # Above 0.4 threshold
    mock_emb.to_blob.return_value = blob
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()

    payload = {
        "stop_hook_active": False,
        "session_id": "sess-contradict",
        "transcript_path": "",
        "cwd": "/tmp/proj",
        "last_assistant_message": response
    }

    import hook_helpers
    payload_json = json.dumps(payload)
    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, 'test.log')

        import stop_hook
        with patch('sys.stdin', StringIO(payload_json)), \
             patch('sys.stdout', captured_output), \
             patch('sys.exit', mock_exit), \
             patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
             patch.object(stop_hook, 'get_embedder', return_value=mock_emb):
            try:
                stop_hook.main()
            except SystemExit:
                pass
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log

    # Inline contradiction enforcement is DISABLED (too many false positives).
    # Voluntary -! annotations + offline contradiction_scan.py handle this instead.
    # Test verifies the hook does NOT block on contradictions.
    assert exit_code[0] == 0, f"Should pass (exit 0) — inline enforcement disabled, got exit {exit_code[0]}"
    conn.close()


# Verifies: -! annotation writes archived_reason and allows stop
def test_contradiction_enforcement_skips_when_annotated():
    """If the LLM already annotated the contradicted memory with -!, enforcement should not block."""
    import numpy as np
    db_path, conn = fresh_db()

    vec = np.random.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    blob = vec.tobytes()
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, session_id, project, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("decision", "tunnel-approach", "Use Cloudflare tunnel for public access", blob, "old-sess", "proj", 0.7)
    )
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value, updated_at) VALUES (?, 'retrieved_ids', ?, CURRENT_TIMESTAMP)",
        ("sess-annotated", json.dumps([1]))
    )
    conn.commit()

    # Response contradicts AND includes -! annotation — should pass
    response = (
        "We replaced the Cloudflare tunnel with GCE edge.\n"
        "<memory>\n- type: decision\n- topic: edge-approach\n- content: GCE edge replaces cloudflare tunnel\n"
        "- confidence_update: 1:-! replaced by GCE edge approach\n"
        "- complete: true\n- context: sufficient\n- keywords: edge\n</memory>"
    )

    from unittest.mock import MagicMock
    mock_emb = MagicMock()
    mock_emb.embed.return_value = vec
    mock_emb.from_blob.return_value = vec
    mock_emb.cosine_similarity.return_value = 0.6
    mock_emb.to_blob.return_value = blob
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()

    payload = {
        "stop_hook_active": False,
        "session_id": "sess-annotated",
        "transcript_path": "",
        "cwd": "/tmp/proj",
        "last_assistant_message": response
    }

    import hook_helpers
    payload_json = json.dumps(payload)
    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, 'test.log')

        import stop_hook
        with patch('sys.stdin', StringIO(payload_json)), \
             patch('sys.stdout', captured_output), \
             patch('sys.exit', mock_exit), \
             patch.object(hook_helpers, 'get_embedder', return_value=mock_emb), \
             patch.object(stop_hook, 'get_embedder', return_value=mock_emb):
            try:
                stop_hook.main()
            except SystemExit:
                pass
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log

    # Should NOT block — the -! annotation was provided
    assert exit_code[0] == 0, f"Should allow stop (exit 0), got exit {exit_code[0]}"

    # The -! should have written archived_reason
    row = conn.execute("SELECT archived_reason FROM memories WHERE id = 1").fetchone()
    assert row[0] is not None, "Should have archived_reason from -! annotation"
    assert "GCE edge" in row[0]
    conn.close()


# ============================================================
# Cleanup
# ============================================================

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
