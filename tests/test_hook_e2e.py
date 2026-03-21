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
        source_end INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
    conn.commit()
    return db_path, conn


def run_hook(db_path, payload, cache_path=None, cont_path=None):
    """Run the stop hook main() with patches for DB path and stdin."""
    payload_json = json.dumps(payload)

    # Patch paths
    patches = {
        'stop_hook.DB_PATH': db_path,
        'stop_hook.LOG_PATH': os.path.join(TEST_DIR, 'test.log'),
        'stop_hook.CONTEXT_CACHE_PATH': cache_path or os.path.join(TEST_DIR, f'.cache_{_db_counter[0]}'),
        'stop_hook.CONTINUATION_COUNT_PATH': cont_path or os.path.join(TEST_DIR, f'.cont_{_db_counter[0]}'),
    }

    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    import stop_hook
    original_db = stop_hook.DB_PATH
    original_log = stop_hook.LOG_PATH
    original_cache = stop_hook.CONTEXT_CACHE_PATH
    original_cont = stop_hook.CONTINUATION_COUNT_PATH

    try:
        stop_hook.DB_PATH = db_path
        stop_hook.LOG_PATH = os.path.join(TEST_DIR, 'test.log')
        stop_hook.CONTEXT_CACHE_PATH = cache_path or os.path.join(TEST_DIR, f'.cache_{_db_counter[0]}')
        stop_hook.CONTINUATION_COUNT_PATH = cont_path or os.path.join(TEST_DIR, f'.cont_{_db_counter[0]}')

        with patch('sys.stdin', StringIO(payload_json)), \
             patch('sys.stdout', captured_output), \
             patch('sys.exit', mock_exit), \
             patch.object(stop_hook, 'get_embedder', return_value=None):
            try:
                stop_hook.main()
            except SystemExit:
                pass
    finally:
        stop_hook.DB_PATH = original_db
        stop_hook.LOG_PATH = original_log
        stop_hook.CONTEXT_CACHE_PATH = original_cache
        stop_hook.CONTINUATION_COUNT_PATH = original_cont

    output = captured_output.getvalue()
    result = json.loads(output) if output.strip() else None
    return result, exit_code[0]


# ============================================================
# Test: Valid memory block → stores in DB
# ============================================================

def test_valid_block_stores_memory():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "test-session",
        "transcript_path": "",
        "cwd": "/tmp/myproject",
        "last_assistant_message": "Here is my answer.\n<memory>\n- type: fact\n- topic: test-store\n- content: This should be stored\n- complete: true\n</memory>"
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

def test_missing_block_triggers_reprompt():
    db_path, conn = fresh_db()
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

    assert result is not None
    assert result["decision"] == "block"
    conn.close()


# ============================================================
# Test: Multiple entries in one block
# ============================================================

def test_multiple_entries_stored():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "test-multi",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Done.\n<memory>\n- type: fact\n- topic: multi-1\n- content: first fact\n- type: decision\n- topic: multi-2\n- content: second decision\n- complete: true\n</memory>"
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

def test_metrics_recorded():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-metrics",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "answer\n<memory>\n- type: fact\n- topic: metric-test\n- content: testing metrics\n- complete: true\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    events = [r[0] for r in conn.execute("SELECT event FROM metrics").fetchall()]
    assert "hook_fired" in events, f"Expected hook_fired metric, got: {events}"
    assert "memories_stored" in events, f"Expected memories_stored metric, got: {events}"
    conn.close()


# ============================================================
# Test: Write throttling with too many entries
# ============================================================

def test_write_throttle_limits_entries():
    from config import MAX_MEMORIES_PER_RESPONSE
    db_path, conn = fresh_db()

    # Build a block with 8 entries (above limit of 5)
    entries_text = ""
    for i in range(8):
        entries_text += f"\n- type: fact\n- topic: throttle-{i}\n- content: fact number {i}"

    payload = {
        "stop_hook_active": False,
        "session_id": "sess-throttle",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": f"lots of stuff\n<memory>{entries_text}\n- complete: true\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count <= MAX_MEMORIES_PER_RESPONSE, f"Expected max {MAX_MEMORIES_PER_RESPONSE}, got {count}"
    conn.close()


# ============================================================
# Test: Noop block (complete: true, no entries) → allow stop, no storage
# ============================================================

def test_noop_block_no_storage():
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-noop",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "simple answer\n<memory>\ncomplete: true\n</memory>"
    }
    result, code = run_hook(db_path, payload)

    assert result is None, f"Noop should allow stop, got: {result}"
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 0, f"Noop should store nothing, got {count}"
    conn.close()


# ============================================================
# Test: Malformed block with open tag but garbage content
# ============================================================

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

def test_realistic_claude_output_with_extra_text():
    """Claude sometimes adds commentary after the memory block."""
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-messy",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Here's what I found.\n\nThe authentication system uses JWT.\n\n<memory>\n- type: decision\n- topic: auth\n- content: JWT chosen for stateless auth\n- complete: true\n</memory>\n\nLet me know if you need anything else."
    }
    result, code = run_hook(db_path, payload)

    row = conn.execute("SELECT content FROM memories WHERE topic = 'auth'").fetchone()
    assert row is not None, "Memory not found despite valid block"
    assert row[0] == "JWT chosen for stateless auth"
    conn.close()


def test_realistic_claude_markdown_wrapped():
    """Claude sometimes wraps the block in markdown code fences."""
    db_path, conn = fresh_db()
    payload = {
        "stop_hook_active": False,
        "session_id": "sess-markdown",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "Done.\n```\n<memory>\n- type: fact\n- topic: markdown-test\n- content: wrapped in code fence\n- complete: true\n</memory>\n```"
    }
    result, code = run_hook(db_path, payload)

    row = conn.execute("SELECT content FROM memories WHERE topic = 'markdown-test'").fetchone()
    assert row is not None, "Should parse memory block even inside code fence"
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
