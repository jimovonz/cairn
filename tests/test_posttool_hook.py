#!/usr/bin/env python3
"""Tests for hooks/posttool_hook.py — signal detection, cooldown gating, and main().

Tests the PostToolUse checkpoint hook that nudges the LLM to emit <memory_note>
tags after high-signal tool calls."""

import sys
import os
import json
import sqlite3
import tempfile
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

TEST_DIR = tempfile.mkdtemp()
_db_counter = [0]


def fresh_db():
    _db_counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"posttool_{_db_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


def run_hook(db_path, payload):
    """Run posttool_hook.main() with patches. Returns (stdout_json_or_none, exit_code)."""
    import hooks.hook_helpers as hook_helpers
    import hooks.posttool_hook as posttool_hook

    payload_json = json.dumps(payload)
    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    original_db = hook_helpers.DB_PATH
    try:
        hook_helpers.DB_PATH = db_path
        with patch("sys.stdin", StringIO(payload_json)), \
             patch("sys.stdout", captured_output), \
             patch("sys.exit", mock_exit), \
             patch.object(hook_helpers, "log", lambda msg: None):
            try:
                posttool_hook.main()
            except SystemExit:
                pass
            finally:
                hook_helpers.flush_metrics()
    finally:
        hook_helpers.DB_PATH = original_db

    output = captured_output.getvalue().strip()
    parsed = json.loads(output) if output else None
    return parsed, exit_code[0]


# === Signal detection tests ===

class TestBashSignalDetection:
    def test_nonzero_exit_is_high_signal(self):
        from hooks.posttool_hook import _is_high_signal_bash
        high, reason = _is_high_signal_bash({}, {"exitCode": 1, "stdout": ""})
        assert high is True
        assert "non-zero" in reason

    def test_zero_exit_small_output_not_signal(self):
        from hooks.posttool_hook import _is_high_signal_bash
        high, _ = _is_high_signal_bash({}, {"exitCode": 0, "stdout": "ok\n"})
        assert high is False

    def test_error_pattern_in_stdout(self):
        from hooks.posttool_hook import _is_high_signal_bash
        high, reason = _is_high_signal_bash({}, {"exitCode": 0, "stdout": "Traceback (most recent call last):\n  File ..."})
        assert high is True
        assert "traceback" in reason

    def test_error_pattern_in_stderr(self):
        from hooks.posttool_hook import _is_high_signal_bash
        high, reason = _is_high_signal_bash({}, {"exitCode": 0, "stdout": "", "stderr": "Permission denied"})
        assert high is True
        assert "denied" in reason

    def test_large_output_triggers(self):
        from hooks.posttool_hook import _is_high_signal_bash
        big_output = "\n".join(f"line {i}" for i in range(50))
        high, reason = _is_high_signal_bash({}, {"exitCode": 0, "stdout": big_output})
        assert high is True
        assert "large output" in reason

    def test_moderate_output_no_errors_not_signal(self):
        from hooks.posttool_hook import _is_high_signal_bash
        output = "\n".join(f"line {i}" for i in range(10))
        high, _ = _is_high_signal_bash({}, {"exitCode": 0, "stdout": output})
        assert high is False


class TestEditSignalDetection:
    def test_edit_always_high_signal(self):
        from hooks.posttool_hook import _is_high_signal_edit
        high, reason = _is_high_signal_edit({"file_path": "/foo/bar.py"}, {})
        assert high is True
        assert "bar.py" in reason

    def test_edit_with_filepath_key(self):
        from hooks.posttool_hook import _is_high_signal_edit
        high, reason = _is_high_signal_edit({"filePath": "/a/b/config.json"}, {})
        assert high is True
        assert "config.json" in reason


# === Cooldown and gating tests ===

class TestCooldownGating:
    def test_first_high_signal_fires_nudge(self):
        db_path, conn = fresh_db()
        payload = {
            "tool_name": "Bash",
            "session_id": "sess-1",
            "tool_input": {"command": "make test"},
            "tool_output": {"exitCode": 1, "stdout": "FAILED"},
        }
        result, exit_code = run_hook(db_path, payload)
        assert result is not None
        assert "MEMORY CHECKPOINT" in result["hookSpecificOutput"]["additionalContext"]

    def test_cooldown_suppresses_immediate_repeat(self):
        db_path, conn = fresh_db()
        # First nudge
        payload = {
            "tool_name": "Bash",
            "session_id": "sess-2",
            "tool_input": {},
            "tool_output": {"exitCode": 1, "stdout": "error"},
        }
        run_hook(db_path, payload)

        # Second high-signal call immediately after — should be suppressed
        result, _ = run_hook(db_path, payload)
        assert result is None  # No nudge output

    def test_nudge_fires_after_cooldown_expires(self):
        db_path, conn = fresh_db()
        session_id = "sess-3"

        # First nudge
        payload = {
            "tool_name": "Bash",
            "session_id": session_id,
            "tool_input": {},
            "tool_output": {"exitCode": 1, "stdout": "error"},
        }
        run_hook(db_path, payload)

        # Burn through cooldown with non-signal calls (exit 0, small output)
        for _ in range(3):
            boring = {
                "tool_name": "Bash",
                "session_id": session_id,
                "tool_input": {},
                "tool_output": {"exitCode": 0, "stdout": "ok"},
            }
            run_hook(db_path, boring)

        # Now another high-signal call should fire
        result, _ = run_hook(db_path, payload)
        assert result is not None
        assert "MEMORY CHECKPOINT" in result["hookSpecificOutput"]["additionalContext"]

    def test_non_whitelisted_tool_ignored(self):
        db_path, _ = fresh_db()
        payload = {
            "tool_name": "Read",
            "session_id": "sess-4",
            "tool_input": {"file_path": "/foo.py"},
            "tool_output": {"content": "file contents"},
        }
        result, _ = run_hook(db_path, payload)
        assert result is None

    def test_subagent_skipped(self):
        db_path, _ = fresh_db()
        payload = {
            "tool_name": "Bash",
            "session_id": "sess-5",
            "agent_id": "subagent-1",
            "tool_input": {},
            "tool_output": {"exitCode": 1, "stdout": "error"},
        }
        result, _ = run_hook(db_path, payload)
        assert result is None

    def test_edit_fires_nudge(self):
        db_path, _ = fresh_db()
        payload = {
            "tool_name": "Edit",
            "session_id": "sess-6",
            "tool_input": {"file_path": "/src/main.py"},
            "tool_output": {},
        }
        result, _ = run_hook(db_path, payload)
        assert result is not None
        assert "MEMORY CHECKPOINT" in result["hookSpecificOutput"]["additionalContext"]

    def test_string_tool_output_handled(self):
        db_path, _ = fresh_db()
        payload = {
            "tool_name": "Bash",
            "session_id": "sess-7",
            "tool_input": {},
            "tool_output": "fatal error occurred",
        }
        result, _ = run_hook(db_path, payload)
        assert result is not None


# === Memory note parser tests ===

class TestMemoryNoteParser:
    def test_parse_single_note(self):
        from hooks.parser import parse_memory_notes
        text = 'Some text <memory_note>fact/config-path: Actual path is /etc/app.json not ~/.config</memory_note> more text'
        notes = parse_memory_notes(text)
        assert len(notes) == 1
        assert notes[0]["type"] == "fact"
        assert notes[0]["topic"] == "config-path"
        assert "/etc/app.json" in notes[0]["content"]

    def test_parse_multiple_notes(self):
        from hooks.parser import parse_memory_notes
        text = (
            '<memory_note>correction/api-url: endpoint moved to /v2/api</memory_note>'
            ' text in between '
            '<memory_note>skill/grep-trick: use rg --type py for Python-only search</memory_note>'
        )
        notes = parse_memory_notes(text)
        assert len(notes) == 2
        assert notes[0]["type"] == "correction"
        assert notes[1]["type"] == "skill"

    def test_malformed_note_skipped(self):
        from hooks.parser import parse_memory_notes
        text = '<memory_note>this is not formatted correctly</memory_note>'
        notes = parse_memory_notes(text)
        assert len(notes) == 0

    def test_empty_text_returns_nothing(self):
        from hooks.parser import parse_memory_notes
        assert parse_memory_notes("") == []
        assert parse_memory_notes("no notes here") == []

    def test_note_with_special_characters(self):
        from hooks.parser import parse_memory_notes
        text = '<memory_note>fact/db-schema: table has columns (id, name, email) — not (id, username)</memory_note>'
        notes = parse_memory_notes(text)
        assert len(notes) == 1
        assert "columns" in notes[0]["content"]


# === Memory note collection in stop hook ===

class TestMemoryNoteCollection:
    def _make_transcript(self, messages):
        """Create a temp transcript JSONL file from a list of (role, text) tuples."""
        path = os.path.join(TEST_DIR, f"transcript_{_db_counter[0]}.jsonl")
        _db_counter[0] += 1
        with open(path, "w") as f:
            for role, text in messages:
                entry = {
                    "type": role,
                    "message": {
                        "role": role,
                        "content": [{"type": "text", "text": text}],
                    },
                }
                f.write(json.dumps(entry) + "\n")
        return path

    def _make_full_db(self):
        """Create a DB with all tables needed for stop hook memory collection."""
        _db_counter[0] += 1
        db_path = os.path.join(TEST_DIR, f"collect_{_db_counter[0]}.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
            project TEXT, confidence REAL DEFAULT 0.7, associated_files TEXT, keywords TEXT,
            depth INTEGER, archived_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TABLE memory_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER, content TEXT, session_id TEXT,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
            INSERT INTO memory_history (memory_id, content, session_id, changed_at)
            VALUES (old.id, old.content, old.session_id, old.updated_at); END""")
        conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
            parent_session_id TEXT, project TEXT, transcript_path TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT, session_id TEXT, detail TEXT, value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TABLE hook_state (
            session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, key))""")
        conn.commit()
        return db_path, conn

    def test_collects_notes_from_transcript(self):
        db_path, conn = self._make_full_db()
        transcript = self._make_transcript([
            ("assistant", "Checking... <memory_note>fact/port: service runs on port 8080 not 3000</memory_note>"),
            ("user", "ok"),
            ("assistant", "Done.\n<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"),
        ])

        import hooks.hook_helpers as hook_helpers
        from hooks.stop_hook import collect_memory_notes
        original_db = hook_helpers.DB_PATH
        try:
            hook_helpers.DB_PATH = db_path
            with patch.object(hook_helpers, "log", lambda msg: None), \
                 patch("hooks.hook_helpers.get_embedder", return_value=None):
                count = collect_memory_notes(transcript, "sess-collect-1", [])
        finally:
            hook_helpers.DB_PATH = original_db

        assert count == 1
        row = conn.execute("SELECT type, topic, content FROM memories").fetchone()
        assert row[0] == "fact"
        assert row[1] == "port"
        assert "8080" in row[2]

    def test_deduplicates_against_final_block(self):
        db_path, conn = self._make_full_db()
        transcript = self._make_transcript([
            ("assistant", "<memory_note>fact/port: service runs on port 8080</memory_note>"),
            ("assistant", "Final response"),
        ])

        # Final block already has a fact/port entry
        final_entries = [{"type": "fact", "topic": "port", "content": "port is 8080"}]

        import hooks.hook_helpers as hook_helpers
        from hooks.stop_hook import collect_memory_notes
        original_db = hook_helpers.DB_PATH
        try:
            hook_helpers.DB_PATH = db_path
            with patch.object(hook_helpers, "log", lambda msg: None), \
                 patch("hooks.hook_helpers.get_embedder", return_value=None):
                count = collect_memory_notes(transcript, "sess-collect-2", final_entries)
        finally:
            hook_helpers.DB_PATH = original_db

        assert count == 0

    def test_respects_session_cap(self):
        db_path, conn = self._make_full_db()
        # Pre-set the count near the cap
        conn.execute(
            "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
            ("sess-collect-3", "memory_notes_stored", "20"),
        )
        conn.commit()

        transcript = self._make_transcript([
            ("assistant", "<memory_note>fact/new: something new</memory_note>"),
        ])

        import hooks.hook_helpers as hook_helpers
        from hooks.stop_hook import collect_memory_notes
        original_db = hook_helpers.DB_PATH
        try:
            hook_helpers.DB_PATH = db_path
            with patch.object(hook_helpers, "log", lambda msg: None), \
                 patch("hooks.hook_helpers.get_embedder", return_value=None):
                count = collect_memory_notes(transcript, "sess-collect-3", [])
        finally:
            hook_helpers.DB_PATH = original_db

        assert count == 0

    def test_no_transcript_returns_zero(self):
        from hooks.stop_hook import collect_memory_notes
        with patch("hooks.hook_helpers.log", lambda msg: None):
            assert collect_memory_notes("", "sess", []) == 0
            assert collect_memory_notes("/nonexistent/path", "sess", []) == 0
