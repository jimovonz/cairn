#!/usr/bin/env python3
"""Tests for hooks/pretool_hook.py — find_memories_for_file and main().

Uses file-based SQLite temp DBs and patches DB_PATH/record_metric/log to isolate
from the real system. Real query logic runs unmodified."""

import sys
import os
import json
import sqlite3
import tempfile
import io
import pytest
from unittest.mock import patch, MagicMock


import hooks.hook_helpers as hook_helpers
import hooks.pretool_hook as pretool_hook
from hooks.pretool_hook import find_memories_for_file

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    """Create a temp file-based SQLite DB with minimal schema for pretool hook."""
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"pretool_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        topic TEXT NOT NULL,
        content TEXT NOT NULL,
        associated_files TEXT,
        keywords TEXT,
        confidence REAL DEFAULT 0.7,
        archived_reason TEXT
    )""")
    conn.execute("""CREATE TABLE metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT,
        session_id TEXT,
        detail TEXT,
        value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    return db_path, conn


def run_main(hook_input_dict, db_path):
    """Run pretool_hook.main() with given hook payload. Returns (stdout, exit_code)."""
    stdin_data = json.dumps(hook_input_dict)
    captured_output = []

    def fake_print(*args, **kwargs):
        captured_output.append(" ".join(str(a) for a in args))

    exit_code = None
    with patch("sys.stdin", io.StringIO(stdin_data)), \
         patch.object(hook_helpers, "DB_PATH", db_path), \
         patch("hooks.pretool_hook.record_metric"), \
         patch("hooks.pretool_hook.log"), \
         patch("builtins.print", side_effect=fake_print):
        try:
            pretool_hook.main()
        except SystemExit as e:
            exit_code = e.code

    return "\n".join(captured_output), exit_code


# ============================================================
# find_memories_for_file — 4 tests
# ============================================================

#TAG: [9C6F] 2026-04-05
# Verifies: exact path match with corrections_only=True returns the correction memory with all fields populated
@pytest.mark.behavioural
def test_find_memories_for_file_behavioural():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('correction', 'null-guard', 'Always check None before indexing', ?, 0.9, NULL)",
        (json.dumps(["/src/auth.py", "/src/utils.py"]),)
    )
    # non-correction must be excluded when corrections_only=True
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('fact', 'schema', 'Flat schema used', ?, 0.8, NULL)",
        (json.dumps(["/src/auth.py"]),)
    )
    conn.commit()
    conn.close()

    with patch.object(hook_helpers, "DB_PATH", db_path), \
         patch("hooks.pretool_hook.log"):
        results = find_memories_for_file("/src/auth.py", corrections_only=True)

    assert len(results) == 1
    assert results[0]["type"] == "correction"
    assert results[0]["topic"] == "null-guard"
    assert results[0]["content"] == "Always check None before indexing"
    assert results[0]["confidence"] == 0.9


#TAG: [970A] 2026-04-05
# Verifies: basename match works across different absolute paths; archived memories are excluded
@pytest.mark.edge
def test_find_memories_for_file_edge():
    db_path, conn = fresh_db()
    # stored with old path — should match new path by basename "storage.py"
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('fact', 'wal-mode', 'Uses WAL mode for concurrency', ?, 0.8, NULL)",
        (json.dumps(["/old/project/storage.py"]),)
    )
    # archived — must be excluded regardless of path match
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('fact', 'stale', 'Old outdated fact', ?, 0.7, 'superseded by new approach')",
        (json.dumps(["/old/project/storage.py"]),)
    )
    conn.commit()
    conn.close()

    with patch.object(hook_helpers, "DB_PATH", db_path), \
         patch("hooks.pretool_hook.log"):
        results = find_memories_for_file("/new/project/storage.py", corrections_only=False)

    assert len(results) == 1
    assert results[0]["topic"] == "wal-mode"
    assert results[0]["content"] == "Uses WAL mode for concurrency"


#TAG: [F91F] 2026-04-05
# Verifies: sqlite3.Error during query is caught, logs error message, and returns empty list
@pytest.mark.error
def test_find_memories_for_file_error():
    # DB with no memories table — conn.execute raises OperationalError
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"pretool_err_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.commit()
    conn.close()  # memories table deliberately not created

    with patch.object(hook_helpers, "DB_PATH", db_path), \
         patch("hooks.pretool_hook.log") as mock_log:
        results = find_memories_for_file("/any/file.py", corrections_only=False)

    assert results == []
    assert mock_log.called
    logged_msg = mock_log.call_args[0][0]
    assert logged_msg.startswith("File context query error:")


#TAG: [6DC1] 2026-04-05
# Verifies: rows with malformed associated_files JSON are silently skipped; valid rows still matched
@pytest.mark.adversarial
def test_find_memories_for_file_adversarial():
    db_path, conn = fresh_db()
    # malformed JSON — must be silently skipped via JSONDecodeError handler
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('fact', 'bad-json', 'Content with broken files list', 'NOT_VALID_JSON[[[', 0.7, NULL)"
    )
    # valid row — must still be returned
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('fact', 'good', 'Good valid memory', ?, 0.8, NULL)",
        (json.dumps(["/target/file.py"]),)
    )
    conn.commit()
    conn.close()

    with patch.object(hook_helpers, "DB_PATH", db_path), \
         patch("hooks.pretool_hook.log"):
        results = find_memories_for_file("/target/file.py", corrections_only=False)

    assert len(results) == 1
    assert results[0]["topic"] == "good"
    assert results[0]["content"] == "Good valid memory"


# ============================================================
# main — 4 tests
# ============================================================

#TAG: [1722] 2026-04-05
# Verifies: Read tool with a correction in DB produces hookSpecificOutput JSON with CAIRN GOTCHA section
@pytest.mark.behavioural
def test_main_behavioural():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('correction', 'off-by-one', 'Range end is exclusive — use len(x) not len(x)-1', ?, 0.95, NULL)",
        (json.dumps(["/src/parser.py"]),)
    )
    conn.commit()
    conn.close()

    hook_input = {
        "tool_name": "Read",
        "session_id": "test-sess",
        "tool_input": {"file_path": "/src/parser.py"},
    }
    output_text, exit_code = run_main(hook_input, db_path)

    assert exit_code == 0
    assert output_text != "", "Expected JSON output for file with corrections"
    parsed = json.loads(output_text)
    hook_out = parsed["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "PreToolUse"
    additional = hook_out["additionalContext"]
    lines = additional.split("\n")
    assert lines[0] == "CAIRN GOTCHA for parser.py:"
    assert lines[1] == "- [off-by-one] Range end is exclusive — use len(x) not len(x)-1"
    assert lines[2].startswith("Sources: ")


#TAG: [693F] 2026-04-05
# Verifies: Bash tool (not in allowed set) causes silent sys.exit(0) with no stdout output
@pytest.mark.edge
def test_main_edge():
    db_path, conn = fresh_db()
    conn.commit()
    conn.close()

    hook_input = {
        "tool_name": "Bash",
        "session_id": "sess1",
        "tool_input": {"command": "ls -la"},
    }
    output_text, exit_code = run_main(hook_input, db_path)

    assert exit_code == 0
    assert output_text == "", f"Expected no output for Bash tool, got: {output_text!r}"


#TAG: [6CA5] 2026-04-05
# Verifies: Write tool with empty tool_input (no file_path or filePath) causes silent sys.exit(0)
@pytest.mark.error
def test_main_error():
    db_path, conn = fresh_db()
    conn.commit()
    conn.close()

    hook_input = {
        "tool_name": "Write",
        "session_id": "sess2",
        "tool_input": {},  # No file_path or filePath present
    }
    output_text, exit_code = run_main(hook_input, db_path)

    assert exit_code == 0
    assert output_text == "", f"Expected no output when file_path is absent, got: {output_text!r}"


#TAG: [A5E5] 2026-04-05
# Verifies: alternate key names ("input" + "filePath") are resolved and produce CAIRN CONTEXT output
@pytest.mark.adversarial
def test_main_adversarial():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('decision', 'schema-choice', 'Chose flat schema for query speed over nested', ?, 0.85, NULL)",
        (json.dumps(["/project/models.py"]),)
    )
    conn.commit()
    conn.close()

    # Uses "input" (not "tool_input") and "filePath" (not "file_path")
    hook_input = {
        "tool_name": "Edit",
        "session_id": "sess3",
        "input": {"filePath": "/project/models.py"},
    }
    output_text, exit_code = run_main(hook_input, db_path)

    assert exit_code == 0
    assert output_text != "", "Expected output when using alternate input/filePath keys"
    parsed = json.loads(output_text)
    additional = parsed["hookSpecificOutput"]["additionalContext"]
    lines = additional.split("\n")
    assert lines[0] == "CAIRN CONTEXT for models.py:"
    assert lines[1] == "- [decision/schema-choice] Chose flat schema for query speed over nested"
    assert lines[2].startswith("Sources: ")
