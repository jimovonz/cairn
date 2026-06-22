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
        synced_at TIMESTAMP,
        facts TEXT,
        topic_embedding BLOB)""")
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
             patch("sys.exit", mock_exit), patch("cairn.config.EPHEMERAL_DB_PATH", db_path):
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path):
        assert prompt_hook.is_first_prompt("new-session") is True
    conn.close()


# Verifies: second prompt after mark_done is not treated as first
def test_second_prompt_not_first():
    """After marking done, second prompt should not be first."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path):
        prompt_hook.mark_first_prompt_done("sess-1")
        assert prompt_hook.is_first_prompt("sess-1") is False
    conn.close()


# Verifies: first-prompt state is isolated per session
def test_different_session_still_first():
    """Marking one session done should not affect another."""
    import hooks.prompt_hook as prompt_hook
    db_path, conn = fresh_env()

    with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path):
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path):
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path):
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path):
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

    with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path):
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
        with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path):
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

        with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path), \
             patch('hooks.hook_helpers.get_embedder', return_value=mock_emb):
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

        with patch.object(hook_helpers, 'DB_PATH', db_path), patch('cairn.config.EPHEMERAL_DB_PATH', db_path), \
             patch('hooks.hook_helpers.get_embedder', return_value=mock_emb):
            # SQL-level dedup: hybrid_search receives exclude_ids={99} and
            # filters before L1.5 ever builds XML. L1.5 returns None directly.
            result = prompt_hook.layer1_5_search("some query", "s1")
            assert result is None  # SQL gate excluded the only candidate

            # Belt-and-braces: strip_seen_entries still works on XML if it
            # somehow contained id=99 (e.g. format-mismatch leak in future).
            from hooks.hook_helpers import strip_seen_entries
            synthetic = '<cairn_context query="x"><entry id="99" sim="0.5">x</entry></cairn_context>'
            stripped = strip_seen_entries(synthetic, "s1")
            assert stripped is None

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


def test_is_action_prompt_true_cases():
    """Short confirmation/action prompts must be detected so L1.5 retrieval is skipped."""
    from hooks.prompt_hook import _is_action_prompt
    positives = [
        "yes", "Yes", "YES", "yep", "yeah",
        "ok", "OK", "okay", "Okay",
        "sure", "do it", "Do it",
        "go", "go ahead", "Go ahead",
        "proceed", "continue",
        "lgtm", "LGTM", "ship it",
        "sounds good", "perfect", "correct", "right", "exactly",
        "please", "thanks",
        "done", "next", "yup", "ack", "k",
        # Trailing punctuation variants — must all be skipped
        "yes.", "yes!", "ok!!", "ok!!!", "do it!!", "yes?", "ok!?",
        # Whitespace tolerance
        "  yes  ", "\nok\n",
        # Emoji confirmations
        "👍", "✅",
    ]
    for msg in positives:
        assert _is_action_prompt(msg), f"expected action prompt: {msg!r}"


def test_is_action_prompt_false_cases():
    """Substantive prompts must NOT match — false positives silently disable retrieval."""
    from hooks.prompt_hook import _is_action_prompt
    negatives = [
        # Action stems with real content following
        "yes do that and also refactor the parser",
        "ok but first explain why",
        "continue with the refactor",
        "do it but only for the new files",
        "go ahead and update the docs too",
        "sure, what about edge cases?",
        # Substantive questions
        "what does the action prompt regex match?",
        "tell me about the keyword overlap change",
        # Long messages — over 80 char cap
        "ok " + ("x" * 100),
        # Empty / whitespace
        "",
        "   ",
        # Action-adjacent but not a confirmation
        "yesterday I changed the threshold",
        "okra is a vegetable",
        "rightly or wrongly",
    ]
    for msg in negatives:
        assert not _is_action_prompt(msg), f"expected non-action: {msg!r}"


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


# ============================================================
# Format-spec gating (v0.14 token trim)
# ============================================================

#TAG: [FS01] 2026-06-12
# Verifies: first prompt of a Claude Code session (transcript under
# ~/.claude/projects/) does NOT receive the ~600-token MEMORY_FORMAT_SPEC —
# the rules file in the system prompt already carries it
def test_main_behavioural_format_spec_gated_for_claude_code():
    db_path, conn = fresh_env()
    conn.close()
    result = run_prompt_hook(db_path, {
        "session_id": "fs-cc-1",
        "prompt": "tell me about the architecture of this project",
        "transcript_path": "/home/user/.claude/projects/-home-user-proj/abc.jsonl",
    })
    if result is not None:
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert ctx.count("MEMORY BLOCK (REQUIRED)") == 0
    # The session must still be marked instructed for stop-hook enforcement
    check = sqlite3.connect(db_path)
    flag = check.execute(
        "SELECT value FROM hook_state WHERE session_id = 'fs-cc-1' AND key = 'format_spec_injected'"
    ).fetchone()
    check.close()
    assert flag == ("1",)


#TAG: [FS02] 2026-06-12
# Verifies: a non-Claude-Code session (Copilot-shaped transcript path) still
# receives the format spec — it has no rules file to learn the format from
def test_main_behavioural_format_spec_kept_for_copilot():
    db_path, conn = fresh_env()
    conn.close()
    result = run_prompt_hook(db_path, {
        "session_id": "fs-cp-1",
        "prompt": "tell me about the architecture of this project",
        "transcript_path": "/home/user/.vscode/copilot/chat/session.jsonl",
    })
    assert result is not None
    ctx = result["hookSpecificOutput"]["additionalContext"]
    assert ctx.count("MEMORY BLOCK (REQUIRED)") == 1


# ---------- refresh_graph_on_head_change (branch-switch freshness) ----------

def test_refresh_graph_on_head_change_behavioural():
    """HEAD moved since last prompt → persist new HEAD and record metric."""
    import hooks.prompt_hook as prompt_hook
    import cairn.repo_discovery as repo_discovery
    cwd = "/repo/x"
    with patch.object(prompt_hook, "load_hook_state", return_value="old-sha"), \
         patch.object(prompt_hook, "save_hook_state") as mock_save, \
         patch.object(prompt_hook, "record_metric") as mock_metric, \
         patch.object(repo_discovery, "kick_graph_update_if_head_changed",
                      return_value=(True, "new-sha")):
        prompt_hook.refresh_graph_on_head_change("sess1", cwd)
    saved = mock_save.call_args[0]
    assert saved[0] == prompt_hook._GRAPH_HEAD_STATE_SESSION
    assert saved[1] == prompt_hook._graph_head_key(cwd)
    assert saved[2] == "new-sha"
    assert mock_metric.call_count == 1


def test_refresh_graph_on_head_change_edge():
    """HEAD unchanged → no persist, no metric (avoid redundant updates)."""
    import hooks.prompt_hook as prompt_hook
    import cairn.repo_discovery as repo_discovery
    with patch.object(prompt_hook, "load_hook_state", return_value="same-sha"), \
         patch.object(prompt_hook, "save_hook_state") as mock_save, \
         patch.object(prompt_hook, "record_metric") as mock_metric, \
         patch.object(repo_discovery, "kick_graph_update_if_head_changed",
                      return_value=(False, "same-sha")):
        prompt_hook.refresh_graph_on_head_change("sess1", "/repo/x")
    assert mock_save.call_count == 0
    assert mock_metric.call_count == 0


def test_refresh_graph_on_head_change_edge_empty_cwd():
    """Empty cwd is a no-op — never touches state."""
    import hooks.prompt_hook as prompt_hook
    with patch.object(prompt_hook, "load_hook_state") as mock_load, \
         patch.object(prompt_hook, "save_hook_state") as mock_save:
        prompt_hook.refresh_graph_on_head_change("sess1", "")
    assert mock_load.call_count == 0
    assert mock_save.call_count == 0


def test_refresh_graph_on_head_change_error():
    """Underlying failure must fail open — no exception escapes the hook."""
    import hooks.prompt_hook as prompt_hook
    import cairn.repo_discovery as repo_discovery
    with patch.object(prompt_hook, "load_hook_state", side_effect=RuntimeError("boom")), \
         patch.object(prompt_hook, "save_hook_state") as mock_save:
        prompt_hook.refresh_graph_on_head_change("sess1", "/repo/x")  # must not raise
    assert mock_save.call_count == 0
