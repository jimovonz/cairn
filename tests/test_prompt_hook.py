#!/usr/bin/env python3
"""Tests for prompt_hook.py — Layer 1 (first-prompt) and Layer 2 (staged cross-project)."""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_env():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"prompt_{_counter[0]}.db")
    staged_path = os.path.join(TEST_DIR, f".staged_{_counter[0]}")
    first_path = os.path.join(TEST_DIR, f".first_{_counter[0]}")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    return db_path, conn, staged_path, first_path


def run_prompt_hook(db_path, staged_path, first_path, payload):
    import prompt_hook
    payload_json = json.dumps(payload)
    captured = StringIO()
    exit_code = [0]

    orig_db = prompt_hook.DB_PATH
    orig_staged = prompt_hook.STAGED_PATH
    orig_first = prompt_hook.FIRST_PROMPT_PATH
    orig_log = prompt_hook.LOG_PATH

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    try:
        prompt_hook.DB_PATH = db_path
        prompt_hook.STAGED_PATH = staged_path
        prompt_hook.FIRST_PROMPT_PATH = first_path
        prompt_hook.LOG_PATH = os.path.join(TEST_DIR, "prompt.log")

        with patch('sys.stdin', StringIO(payload_json)), \
             patch('sys.stdout', captured), \
             patch('sys.exit', mock_exit):
            try:
                prompt_hook.main()
            except SystemExit:
                pass
    finally:
        prompt_hook.DB_PATH = orig_db
        prompt_hook.STAGED_PATH = orig_staged
        prompt_hook.FIRST_PROMPT_PATH = orig_first
        prompt_hook.LOG_PATH = orig_log

    output = captured.getvalue()
    return json.loads(output) if output.strip() else None


# ============================================================
# Layer 1: First-prompt detection
# ============================================================

def test_first_prompt_detected():
    """First prompt for a session should be detected."""
    import prompt_hook
    db_path, conn, staged_path, first_path = fresh_env()

    with patch.object(prompt_hook, 'FIRST_PROMPT_PATH', first_path):
        assert prompt_hook.is_first_prompt("new-session") is True


def test_second_prompt_not_first():
    """After marking done, second prompt should not be first."""
    import prompt_hook
    db_path, conn, staged_path, first_path = fresh_env()

    with patch.object(prompt_hook, 'FIRST_PROMPT_PATH', first_path):
        prompt_hook.mark_first_prompt_done("sess-1")
        assert prompt_hook.is_first_prompt("sess-1") is False


def test_different_session_still_first():
    """Marking one session done should not affect another."""
    import prompt_hook
    db_path, conn, staged_path, first_path = fresh_env()

    with patch.object(prompt_hook, 'FIRST_PROMPT_PATH', first_path):
        prompt_hook.mark_first_prompt_done("sess-1")
        assert prompt_hook.is_first_prompt("sess-2") is True


# ============================================================
# Layer 2: Staged context
# ============================================================

def test_staged_context_loaded():
    """Staged context should be loaded and consumed."""
    import prompt_hook
    db_path, conn, staged_path, first_path = fresh_env()

    staged_data = {"sess-1": "<brain_context>test data</brain_context>"}
    with open(staged_path, "w") as f:
        json.dump(staged_data, f)

    with patch.object(prompt_hook, 'STAGED_PATH', staged_path):
        data = prompt_hook.load_staged_context("sess-1")
    assert data is not None
    assert "test data" in data


def test_staged_context_consumed():
    """After loading, staged context should be removed for that session."""
    import prompt_hook
    db_path, conn, staged_path, first_path = fresh_env()

    staged_data = {"sess-1": "data1", "sess-2": "data2"}
    with open(staged_path, "w") as f:
        json.dump(staged_data, f)

    with patch.object(prompt_hook, 'STAGED_PATH', staged_path):
        prompt_hook.load_staged_context("sess-1")

    with open(staged_path) as f:
        remaining = json.load(f)
    assert "sess-1" not in remaining
    assert "sess-2" in remaining


def test_no_staged_context():
    """Missing staged file should return None gracefully."""
    import prompt_hook
    db_path, conn, staged_path, first_path = fresh_env()

    with patch.object(prompt_hook, 'STAGED_PATH', staged_path):
        data = prompt_hook.load_staged_context("sess-nonexistent")
    assert data is None


# ============================================================
# Short message handling
# ============================================================

def test_short_message_skipped():
    """Messages < 3 chars should be skipped."""
    db_path, conn, staged_path, first_path = fresh_env()
    result = run_prompt_hook(db_path, staged_path, first_path, {
        "session_id": "sess-short",
        "user_message": "hi"
    })
    assert result is None


def test_empty_db_no_injection():
    """Empty DB should not inject anything."""
    db_path, conn, staged_path, first_path = fresh_env()
    result = run_prompt_hook(db_path, staged_path, first_path, {
        "session_id": "sess-empty",
        "user_message": "what decisions have we made about the authentication system?"
    })
    # No memories in DB → nothing to inject (Layer 1 finds nothing)
    assert result is None


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
