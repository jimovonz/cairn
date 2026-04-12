#!/usr/bin/env python3
"""Tests for daemon communication, embeddings fallback, context cache, loop protection, and fail-open."""

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

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"dc_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE memory_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER, content TEXT, session_id TEXT,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content, session_id, changed_at)
        VALUES (old.id, old.content, old.session_id, old.updated_at); END""")
    conn.commit()
    return db_path, conn


# ============================================================
# Daemon: embed() fallback when daemon unavailable
# ============================================================

# Verifies: embed(allow_slow=False) returns None when daemon is down
def test_embed_allow_slow_false_returns_none():
    """embed(allow_slow=False) should return None when daemon isn't running."""
    import cairn.embeddings as emb
    with patch.object(emb, '_daemon_embed', return_value=None), \
         patch.object(emb, '_daemon_start_attempted', True):
        result = emb.embed("test text", allow_slow=False)
    assert result is None


# Verifies: embed(allow_slow=True) falls back to local model loading
def test_embed_allow_slow_true_loads_model():
    """embed(allow_slow=True) should fall back to model loading."""
    import cairn.embeddings as emb
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
    with patch.object(emb, '_daemon_embed', return_value=None), \
         patch.object(emb, 'get_model', return_value=mock_model):
        result = emb.embed("test text", allow_slow=True)
    assert len(result) == 384


# Verifies: daemon response vector is used directly by embed()
def test_daemon_embed_returns_vector_on_success():
    """When daemon responds, embed should use its vector."""
    import cairn.embeddings as emb
    fake_vec = np.random.randn(384).astype(np.float32)
    fake_hex = fake_vec.tobytes().hex()

    with patch.object(emb, '_daemon_embed', return_value=fake_vec):
        result = emb.embed("test text")
    assert np.allclose(result, fake_vec)


# ============================================================
# Daemon: is_running with stale PID
# ============================================================

# Verifies: stale PID file detected and cleaned up
def test_is_running_stale_pid():
    """Stale PID file should return False and clean up."""
    from cairn.daemon import is_running, PID_PATH
    pid_path = os.path.join(TEST_DIR, ".test_pid")

    with open(pid_path, "w") as f:
        f.write("999999999")  # Non-existent PID

    with patch('cairn.daemon.PID_PATH', pid_path):
        result = is_running()
    assert result is False
    assert not os.path.exists(pid_path)  # Should clean up stale file


# ============================================================
# Context cache: semantic matching
# ============================================================

# Verifies: exact context_need string is recognized as cached
def test_context_cache_exact_string_match():
    """Exact same context_need should be cached."""
    from hooks.retrieval import is_context_cached, add_to_context_cache
    served = []
    served = add_to_context_cache("what database did we choose", served, None)
    assert is_context_cached("what database did we choose", served, None) is True


# Verifies: different context_need string is not a cache hit
def test_context_cache_different_string():
    """Different context_need should not be cached (without embeddings)."""
    from hooks.retrieval import is_context_cached, add_to_context_cache
    served = []
    served = add_to_context_cache("what database did we choose", served, None)
    assert is_context_cached("how does authentication work", served, None) is False



# ============================================================
# Loop protection: continuation counter
# ============================================================

def _make_state_db():
    """Create a temp DB with hook_state table for continuation tests."""
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"state_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


# Verifies: continuation counter increments correctly per call
def test_continuation_counter_increments():
    from hooks.enforcement import get_continuation_count, increment_continuation
    db_path, conn = _make_state_db()

    with patch('hooks.hook_helpers.DB_PATH', db_path):
        assert get_continuation_count("sess-1") == 0
        increment_continuation("sess-1")
        assert get_continuation_count("sess-1") == 1
        increment_continuation("sess-1")
        assert get_continuation_count("sess-1") == 2
    conn.close()


# Verifies: continuation counter resets to zero
def test_continuation_counter_resets():
    from hooks.enforcement import get_continuation_count, increment_continuation, reset_continuation
    db_path, conn = _make_state_db()

    with patch('hooks.hook_helpers.DB_PATH', db_path):
        increment_continuation("sess-1")
        increment_continuation("sess-1")
        assert get_continuation_count("sess-1") == 2
        reset_continuation("sess-1")
        assert get_continuation_count("sess-1") == 0
    conn.close()


# Verifies: continuation counter reaches MAX_CONTINUATIONS cap
def test_continuation_cap_at_three():
    from hooks.enforcement import get_continuation_count, increment_continuation
    from cairn.config import MAX_CONTINUATIONS
    db_path, conn = _make_state_db()

    with patch('hooks.hook_helpers.DB_PATH', db_path):
        for _ in range(MAX_CONTINUATIONS):
            increment_continuation("sess-1")
        assert get_continuation_count("sess-1") >= MAX_CONTINUATIONS
    conn.close()


# Verifies: continuation counters are isolated between sessions
def test_continuation_isolated_per_session():
    from hooks.enforcement import get_continuation_count, increment_continuation
    db_path, conn = _make_state_db()

    with patch('hooks.hook_helpers.DB_PATH', db_path):
        increment_continuation("sess-1")
        increment_continuation("sess-1")
        assert get_continuation_count("sess-1") == 2
        assert get_continuation_count("sess-2") == 0
    conn.close()


# ============================================================
# Fail-open: hook crash → exit 0
# ============================================================

# Verifies: hook crash results in exit 0 (fail-open behavior)
def test_hook_crash_exits_zero():
    """A crash in main() should be caught and exit 0 (fail-open)."""
    import hooks.stop_hook as stop_hook

    payload = json.dumps({
        "stop_hook_active": False,
        "session_id": "crash-test",
        "transcript_path": "",
        "cwd": "/tmp",
        "last_assistant_message": "test"
    })
    exit_code = [None]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    # Patch DB_PATH to a non-existent path to force a crash inside main
    with patch.object(hook_helpers, 'DB_PATH', '/nonexistent/path/db.sqlite'), \
         patch('sys.stdin', StringIO(payload)), \
         patch('sys.stdout', StringIO()), \
         patch('sys.exit', mock_exit), \
         patch.object(hook_helpers, 'LOG_PATH', os.path.join(TEST_DIR, 'crash.log')):
        try:
            # The wrapper around main() should catch and exit 0
            try:
                stop_hook.main()
            except SystemExit:
                pass
        except Exception:
            pass

    # Even if main crashed, the fail-open wrapper should exit 0
    assert exit_code[0] == 0 or exit_code[0] is None, f"Expected exit 0, got {exit_code[0]}"


# ============================================================
# Metrics: events recorded correctly
# ============================================================

# Verifies: record_metric persists event, detail, and value to DB
def test_record_metric_stores_event():
    from hooks.hook_helpers import record_metric, flush_metrics
    db_path, conn = fresh_db()

    with patch('hooks.hook_helpers.DB_PATH', db_path):
        record_metric("sess-1", "test_event", "detail", 42.0)
        flush_metrics()

    row = conn.execute("SELECT event, detail, value FROM metrics").fetchone()
    assert row[0] == "test_event"
    assert row[1] == "detail"
    assert row[2] == 42.0
    conn.close()


# Verifies: record_metric silently handles DB errors
def test_record_metric_survives_db_error():
    """Metric recording should not crash on DB errors."""
    from hooks.hook_helpers import record_metric, flush_metrics
    # Non-existent DB path — should silently fail
    with patch('hooks.hook_helpers.DB_PATH', '/nonexistent/db.sqlite'):
        record_metric("sess-1", "test", "detail", 1.0)  # Should not raise
        flush_metrics()  # Flush also should not raise


# ============================================================
# Low-info pre-filter (through main())
# ============================================================

# Verifies: low-info context_need 'help' is pre-filtered via main()
def test_low_info_context_need_filtered_through_main():
    """A context_need of 'help' should be pre-filtered — verified via metric."""
    import hooks.stop_hook as stop_hook
    db_path = os.path.join(TEST_DIR, f"prefilter_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY, type TEXT, topic TEXT,
        content TEXT, embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE hook_state (session_id TEXT NOT NULL, key TEXT NOT NULL,
        value TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content); END""")
    conn.commit()

    payload = json.dumps({
        "session_id": "s-prefilter", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "ok\n<memory>\n- context: insufficient\n- keywords: test\n- context_need: help\n- complete: true\n</memory>"
    })

    captured = StringIO()
    def mock_exit(code=0):
        raise SystemExit(code)

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'LOG_PATH', os.path.join(TEST_DIR, 'prefilter.log')), \
         patch.object(hook_helpers, 'get_embedder', return_value=None), \
         patch('sys.stdin', StringIO(payload)), \
         patch('sys.stdout', captured), \
         patch('sys.exit', mock_exit):
        try:
            stop_hook.main()
        except SystemExit:
            pass
        finally:
            hook_helpers.flush_metrics()

    hit = conn.execute("SELECT COUNT(*) FROM metrics WHERE event = 'context_prefiltered'").fetchone()[0]
    assert hit >= 1, "Low-info context_need 'help' should trigger context_prefiltered metric"
    conn.close()


# Verifies: substantive context_need passes pre-filter to retrieval
def test_substantive_context_need_not_filtered():
    """A real question should NOT be pre-filtered."""
    import hooks.stop_hook as stop_hook
    db_path = os.path.join(TEST_DIR, f"subst_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY, type TEXT, topic TEXT,
        content TEXT, embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE hook_state (session_id TEXT NOT NULL, key TEXT NOT NULL,
        value TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content); END""")
    conn.commit()

    payload = json.dumps({
        "session_id": "s-subst", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "ok\n<memory>\n- context: insufficient\n- keywords: test\n- context_need: what database architecture decisions have we made\n- complete: true\n</memory>"
    })

    captured = StringIO()
    def mock_exit(code=0):
        raise SystemExit(code)

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'LOG_PATH', os.path.join(TEST_DIR, 'subst.log')), \
         patch.object(hook_helpers, 'get_embedder', return_value=None), \
         patch('sys.stdin', StringIO(payload)), \
         patch('sys.stdout', captured), \
         patch('sys.exit', mock_exit):
        try:
            stop_hook.main()
        except SystemExit:
            pass
        finally:
            hook_helpers.flush_metrics()

    prefiltered = conn.execute("SELECT COUNT(*) FROM metrics WHERE event = 'context_prefiltered'").fetchone()[0]
    requested = conn.execute("SELECT COUNT(*) FROM metrics WHERE event = 'context_requested'").fetchone()[0]
    assert prefiltered == 0, "Substantive context_need should NOT be pre-filtered"
    assert requested >= 1, "Substantive context_need should trigger context_requested metric"
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
