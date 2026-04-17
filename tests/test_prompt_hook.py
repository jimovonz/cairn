#!/usr/bin/env python3
"""Tests for prompt_hook.py — Layer 1 (first-prompt) and Layer 2 (staged cross-project)."""

import sys
import os
import json
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import tempfile
import shutil
from unittest.mock import patch
from io import StringIO


import hooks.hook_helpers as hook_helpers

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_env():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"prompt_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, keywords TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        origin_id TEXT,
        user_id TEXT,
        updated_by TEXT,
        team_id TEXT,
        source_ref TEXT,
        deleted_at TIMESTAMP,
        synced_at TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


def run_prompt_hook(db_path, payload):
    import hooks.prompt_hook as prompt_hook
    payload_json = json.dumps(payload)
    captured = StringIO()
    exit_code = [0]

    orig_db = prompt_hook.DB_PATH
    orig_log = prompt_hook.LOG_PATH
    orig_hh_db = hook_helpers.DB_PATH

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    try:
        prompt_hook.DB_PATH = db_path
        prompt_hook.LOG_PATH = os.path.join(TEST_DIR, "prompt.log")
        hook_helpers.DB_PATH = db_path  # Ensure get_conn() uses test DB

        with patch('sys.stdin', StringIO(payload_json)), \
             patch('sys.stdout', captured), \
             patch('sys.exit', mock_exit):
            try:
                prompt_hook.main()
            except SystemExit:
                pass
    finally:
        prompt_hook.DB_PATH = orig_db
        prompt_hook.LOG_PATH = orig_log
        hook_helpers.DB_PATH = orig_hh_db

    output = captured.getvalue()
    return json.loads(output) if output.strip() else None


# ============================================================
# Layer 1: First-prompt detection
# ============================================================

# Verifies: new session is detected as first prompt
def test_first_prompt_detected():
    """First prompt for a session should be detected."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        assert prompt_hook.is_first_prompt("new-session") is True
    conn.close()


# Verifies: second prompt after mark_done is not treated as first
def test_second_prompt_not_first():
    """After marking done, second prompt should not be first."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        prompt_hook.mark_first_prompt_done("sess-1")
        assert prompt_hook.is_first_prompt("sess-1") is False
    conn.close()


# Verifies: first-prompt state is isolated per session
def test_different_session_still_first():
    """Marking one session done should not affect another."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        prompt_hook.mark_first_prompt_done("sess-1")
        assert prompt_hook.is_first_prompt("sess-2") is True
    conn.close()


# ============================================================
# Layer 2: Staged context
# ============================================================

# Verifies: staged context is loaded from hook_state DB
def test_staged_context_loaded():
    """Staged context should be loaded and consumed."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    # Stage via DB (as stop_hook would)
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'staged_context', ?)",
        ("sess-1", "<cairn_context>test data</cairn_context>")
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        data = prompt_hook.load_staged_context("sess-1")
    assert data == "<cairn_context>test data</cairn_context>"
    conn.close()


# Verifies: loaded staged context is deleted, other sessions untouched
def test_staged_context_consumed():
    """After loading, staged context should be removed for that session."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'staged_context', ?)",
        ("sess-1", "data1")
    )
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'staged_context', ?)",
        ("sess-2", "data2")
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        prompt_hook.load_staged_context("sess-1")

    # sess-1 should be consumed, sess-2 should remain
    row1 = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 'sess-1' AND key = 'staged_context'"
    ).fetchone()
    row2 = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 'sess-2' AND key = 'staged_context'"
    ).fetchone()
    assert row1 is None, "sess-1 staged context should be consumed"
    assert row2 is not None, "sess-2 staged context should remain"
    conn.close()


# Verifies: missing staged context returns None without error
def test_no_staged_context():
    """Missing staged data should return None gracefully."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        data = prompt_hook.load_staged_context("sess-nonexistent")
    assert data is None
    conn.close()


# ============================================================
# Short message handling
# ============================================================

# Verifies: very short messages are skipped (no injection)
def test_short_message_skipped():
    """Messages < 3 chars should be skipped."""
    db_path, conn = fresh_env()
    result = run_prompt_hook(db_path, {
        "session_id": "sess-short",
        "user_message": "hi"
    })
    assert result is None
    conn.close()


# Verifies: empty database produces no context injection
def test_empty_db_no_injection():
    """Empty DB should only inject the memory block reminder, no actual memories."""
    db_path, conn = fresh_env()
    result = run_prompt_hook(db_path, {
        "session_id": "sess-empty",
        "user_message": "what decisions have we made about the authentication system?"
    })
    # No memories in DB → only the memory block reminder is injected (no L1/L2 context)
    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert "MEMORY BLOCK" in ctx
    assert "<cairn_context" not in ctx  # No actual memory entries
    conn.close()


# ============================================================
# Retrieved ID tracking for contradiction enforcement
# ============================================================

# Verifies: injected entry IDs are stored for contradiction enforcement
def test_injected_ids_stored_in_hook_state():
    """When context is injected containing entry IDs, prompt hook should
    write those IDs to hook_state for the stop hook's contradiction enforcement."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    # Stage Layer 2 context containing entry IDs (as stop hook would after keyword search)
    staged_context = (
        '<cairn_context query="test" current_project="proj">\n'
        '  <scope level="global" weight="low">\n'
        '    <entry id="42" type="fact" topic="auth" reliability="strong" days="2">Use JWT tokens</entry>\n'
        '    <entry id="17" type="decision" topic="db" reliability="moderate" days="5">PostgreSQL chosen</entry>\n'
        '  </scope>\n'
        '</cairn_context>'
    )
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'staged_context', ?)",
        ("sess-ids", staged_context)
    )
    # Mark first prompt as done so Layer 1 doesn't run
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'first_prompt_done', '1')",
        ("sess-ids",)
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        result = run_prompt_hook(db_path, {
            "session_id": "sess-ids",
            "user_message": "tell me about the auth decisions"
        })

    # Context should have been injected — result is a dict with hookSpecificOutput
    assert isinstance(result, dict) and "hookSpecificOutput" in result

    # Retrieved IDs should be stored in hook_state
    ids_row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = 'sess-ids' AND key = 'retrieved_ids'"
    ).fetchone()
    assert ids_row is not None, "Expected retrieved_ids in hook_state"
    ids = json.loads(ids_row[0])
    assert 42 in ids
    assert 17 in ids
    assert len(ids) == 2
    conn.close()


# ============================================================
# Layer 1.5: Per-prompt injection
# ============================================================

# Verifies: Layer 1.5 returns None when disabled (default)
def test_layer1_5_disabled_by_default():
    import hooks.prompt_hook as prompt_hook
    import cairn.config as config
    original = config.L1_5_ENABLED
    try:
        config.L1_5_ENABLED = False
        db_path, conn = fresh_env()
        conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
        conn.commit()
        with patch.object(hook_helpers, 'DB_PATH', db_path):
            result = prompt_hook.layer1_5_search("what is the auth approach", "s1")
        assert result is None
        conn.close()
    finally:
        config.L1_5_ENABLED = original


# Verifies: Layer 1.5 fires on subsequent prompts when enabled with matching results
def test_layer1_5_injects_on_subsequent_prompt():
    import hooks.prompt_hook as prompt_hook
    import cairn.config as config
    import numpy as np
    from unittest.mock import MagicMock
    original = config.L1_5_ENABLED
    original_thresh = config.L1_5_SIM_THRESHOLD
    try:
        config.L1_5_ENABLED = True
        config.L1_5_SIM_THRESHOLD = 0.50
        db_path, conn = fresh_env()
        conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
        conn.execute(
            "INSERT INTO memories (id, type, topic, content, embedding, project, session_id) VALUES (99, 'decision', 'auth', 'Use JWT', X'00', 'proj', 'other')"
        )
        conn.commit()

        mock_emb = MagicMock()
        mock_emb.find_similar.return_value = [{
            "id": 99, "type": "decision", "topic": "auth",
            "content": "Use JWT for stateless auth",
            "updated_at": "2026-04-01 10:00:00", "project": "proj",
            "confidence": 0.8, "session_id": "other", "depth": None,
            "archived_reason": None, "similarity": 0.75, "score": 0.55,
        }]

        with patch.object(hook_helpers, 'DB_PATH', db_path), \
             patch('hooks.prompt_hook.get_embedder', return_value=mock_emb):
            result = prompt_hook.layer1_5_search("auth approach JWT", "s1")

        assert result is not None
        assert "per-prompt" in result
        assert "JWT" in result
        conn.close()
    finally:
        config.L1_5_ENABLED = original
        config.L1_5_SIM_THRESHOLD = original_thresh


# Verifies: Central dedup gate strips already-injected IDs from output
def test_layer1_5_skips_already_injected():
    import hooks.prompt_hook as prompt_hook
    import cairn.config as config
    import json as _json
    from unittest.mock import MagicMock
    original = config.L1_5_ENABLED
    original_thresh = config.L1_5_SIM_THRESHOLD
    try:
        config.L1_5_ENABLED = True
        config.L1_5_SIM_THRESHOLD = 0.50
        db_path, conn = fresh_env()
        conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
        conn.execute(
            "INSERT INTO memories (id, type, topic, content, embedding, project, session_id) VALUES (99, 'fact', 'test', 'already seen', X'00', 'proj', 'other')"
        )
        # Mark memory 99 as already injected
        conn.execute(
            "INSERT INTO hook_state (session_id, key, value) VALUES ('s1', 'retrieved_ids', ?)",
            (_json.dumps([99]),)
        )
        conn.commit()

        mock_emb = MagicMock()
        mock_emb.find_similar.return_value = [{
            "id": 99, "type": "fact", "topic": "test",
            "content": "already seen this",
            "updated_at": "2026-04-01 10:00:00", "project": "proj",
            "confidence": 0.8, "session_id": "other", "depth": None,
            "archived_reason": None, "similarity": 0.80, "score": 0.60,
        }]

        with patch.object(hook_helpers, 'DB_PATH', db_path), \
             patch('hooks.prompt_hook.get_embedder', return_value=mock_emb):
            # L1.5 returns XML (dedup moved to central gate)
            result = prompt_hook.layer1_5_search("some query", "s1")
            assert result is not None  # L1.5 builds XML without filtering

            # Central gate strips already-seen entries
            from hooks.hook_helpers import strip_seen_entries
            stripped = strip_seen_entries(result, "s1")
            assert stripped is None  # All entries were already injected

        conn.close()
    finally:
        config.L1_5_ENABLED = original
        config.L1_5_SIM_THRESHOLD = original_thresh


# Verifies: bool config override via environment variable
def test_l1_5_enabled_env_override():
    import importlib
    import os
    env_backup = os.environ.copy()
    try:
        os.environ["CAIRN_L1_5_ENABLED"] = "1"
        import cairn.config as config
        importlib.reload(config)
        assert config.L1_5_ENABLED is True
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
        importlib.reload(config)


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
