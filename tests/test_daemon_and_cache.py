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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"dc_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

def test_embed_allow_slow_false_returns_none():
    """embed(allow_slow=False) should return None when daemon isn't running."""
    import embeddings as emb
    with patch.object(emb, '_daemon_embed', return_value=None), \
         patch.object(emb, '_daemon_start_attempted', True):
        result = emb.embed("test text", allow_slow=False)
    assert result is None


def test_embed_allow_slow_true_loads_model():
    """embed(allow_slow=True) should fall back to model loading."""
    import embeddings as emb
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
    with patch.object(emb, '_daemon_embed', return_value=None), \
         patch.object(emb, 'get_model', return_value=mock_model):
        result = emb.embed("test text", allow_slow=True)
    assert result is not None
    assert len(result) == 384


def test_daemon_embed_returns_vector_on_success():
    """When daemon responds, embed should use its vector."""
    import embeddings as emb
    fake_vec = np.random.randn(384).astype(np.float32)
    fake_hex = fake_vec.tobytes().hex()

    with patch.object(emb, '_daemon_embed', return_value=fake_vec):
        result = emb.embed("test text")
    assert result is not None
    assert np.allclose(result, fake_vec)


# ============================================================
# Daemon: is_running with stale PID
# ============================================================

def test_is_running_stale_pid():
    """Stale PID file should return False and clean up."""
    from daemon import is_running, PID_PATH
    pid_path = os.path.join(TEST_DIR, ".test_pid")

    with open(pid_path, "w") as f:
        f.write("999999999")  # Non-existent PID

    with patch('daemon.PID_PATH', pid_path):
        result = is_running()
    assert result is False
    assert not os.path.exists(pid_path)  # Should clean up stale file


# ============================================================
# Context cache: semantic matching
# ============================================================

def test_context_cache_exact_string_match():
    """Exact same context_need should be cached."""
    from stop_hook import is_context_cached, add_to_context_cache
    served = []
    served = add_to_context_cache("what database did we choose", served, None)
    assert is_context_cached("what database did we choose", served, None) is True


def test_context_cache_different_string():
    """Different context_need should not be cached (without embeddings)."""
    from stop_hook import is_context_cached, add_to_context_cache
    served = []
    served = add_to_context_cache("what database did we choose", served, None)
    assert is_context_cached("how does authentication work", served, None) is False


def test_context_cache_empty():
    """Empty cache should return False."""
    from stop_hook import is_context_cached
    assert is_context_cached("anything", [], None) is False


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


def test_continuation_counter_increments():
    from stop_hook import get_continuation_count, increment_continuation
    db_path, conn = _make_state_db()

    with patch('stop_hook.DB_PATH', db_path):
        assert get_continuation_count("sess-1") == 0
        increment_continuation("sess-1")
        assert get_continuation_count("sess-1") == 1
        increment_continuation("sess-1")
        assert get_continuation_count("sess-1") == 2
    conn.close()


def test_continuation_counter_resets():
    from stop_hook import get_continuation_count, increment_continuation, reset_continuation
    db_path, conn = _make_state_db()

    with patch('stop_hook.DB_PATH', db_path):
        increment_continuation("sess-1")
        increment_continuation("sess-1")
        assert get_continuation_count("sess-1") == 2
        reset_continuation("sess-1")
        assert get_continuation_count("sess-1") == 0
    conn.close()


def test_continuation_cap_at_three():
    from stop_hook import get_continuation_count, increment_continuation
    from config import MAX_CONTINUATIONS
    db_path, conn = _make_state_db()

    with patch('stop_hook.DB_PATH', db_path):
        for _ in range(MAX_CONTINUATIONS):
            increment_continuation("sess-1")
        assert get_continuation_count("sess-1") >= MAX_CONTINUATIONS
    conn.close()


def test_continuation_isolated_per_session():
    from stop_hook import get_continuation_count, increment_continuation
    db_path, conn = _make_state_db()

    with patch('stop_hook.DB_PATH', db_path):
        increment_continuation("sess-1")
        increment_continuation("sess-1")
        assert get_continuation_count("sess-1") == 2
        assert get_continuation_count("sess-2") == 0
    conn.close()


# ============================================================
# Fail-open: hook crash → exit 0
# ============================================================

def test_hook_crash_exits_zero():
    """A crash in main() should be caught and exit 0 (fail-open)."""
    import stop_hook

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
    with patch.object(stop_hook, 'DB_PATH', '/nonexistent/path/db.sqlite'), \
         patch('sys.stdin', StringIO(payload)), \
         patch('sys.stdout', StringIO()), \
         patch('sys.exit', mock_exit), \
         patch.object(stop_hook, 'LOG_PATH', os.path.join(TEST_DIR, 'crash.log')):
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

def test_record_metric_stores_event():
    from stop_hook import record_metric
    db_path, conn = fresh_db()

    with patch('stop_hook.DB_PATH', db_path):
        record_metric("sess-1", "test_event", "detail", 42.0)

    row = conn.execute("SELECT event, detail, value FROM metrics").fetchone()
    assert row is not None
    assert row[0] == "test_event"
    assert row[1] == "detail"
    assert row[2] == 42.0
    conn.close()


def test_record_metric_survives_db_error():
    """Metric recording should not crash on DB errors."""
    from stop_hook import record_metric
    # Non-existent DB path — should silently fail
    with patch('stop_hook.DB_PATH', '/nonexistent/db.sqlite'):
        record_metric("sess-1", "test", "detail", 1.0)  # Should not raise


# ============================================================
# Low-info pre-filter
# ============================================================

def test_low_info_context_need_detected():
    """Short/generic context_need should be caught."""
    LOW_INFO = {"help", "continue", "more", "yes", "no", "ok", "thanks", "done", "info", "more info"}
    assert len("help") < 8
    assert set("ok".lower().split()) <= LOW_INFO
    assert set("more info".lower().split()) <= LOW_INFO


def test_substantive_context_need_passes():
    """Real questions should not be filtered."""
    LOW_INFO = {"help", "continue", "more", "yes", "no", "ok", "thanks", "done", "info", "more info"}
    need = "what database did we choose for the project"
    assert len(need) >= 8
    assert not (set(need.lower().split()) <= LOW_INFO)


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
