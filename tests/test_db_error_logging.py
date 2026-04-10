#!/usr/bin/env python3
"""Tests for DB error logging — replaces the prior silent exception swallowing
in record_metric/save_hook_state/delete_hook_state/load_hook_state with loud
logging that surfaces corruption separately."""

import os
import sqlite3
import tempfile
from unittest.mock import patch

from hooks.hook_helpers import _is_corruption_error, _log_db_error


def test_is_corruption_error_detects_malformed():
    """sqlite3 'database disk image is malformed' → corruption."""
    exc = sqlite3.DatabaseError("database disk image is malformed")
    assert _is_corruption_error(exc) is True


def test_is_corruption_error_detects_not_a_database():
    """sqlite3 'file is not a database' → corruption."""
    exc = sqlite3.DatabaseError("file is not a database")
    assert _is_corruption_error(exc) is True


def test_is_corruption_error_detects_no_such_table():
    """sqlite3 'no such table' → structural drift, counted as corruption."""
    exc = sqlite3.OperationalError("no such table: memories_fts")
    assert _is_corruption_error(exc) is True


def test_is_corruption_error_ignores_busy():
    """sqlite3 'database is locked' → contention, NOT corruption."""
    exc = sqlite3.OperationalError("database is locked")
    assert _is_corruption_error(exc) is False


def test_is_corruption_error_ignores_syntax():
    """sqlite3 syntax error → bug, not corruption."""
    exc = sqlite3.OperationalError("near \"FOO\": syntax error")
    assert _is_corruption_error(exc) is False


def test_log_db_error_writes_corruption_log_on_corruption():
    """Corruption errors are logged to corruption.log in addition to the main log."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "hook.log")
        corruption_path = os.path.join(tmp, "corruption.log")
        with patch("hooks.hook_helpers.LOG_PATH", log_path):
            exc = sqlite3.DatabaseError("database disk image is malformed")
            _log_db_error("test_context", exc)
        assert os.path.exists(corruption_path), "corruption.log should be created"
        with open(corruption_path) as f:
            content = f.read()
        assert "test_context" in content
        assert "malformed" in content


def test_log_db_error_no_corruption_log_on_non_corruption():
    """Non-corruption errors do not write to corruption.log."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "hook.log")
        corruption_path = os.path.join(tmp, "corruption.log")
        with patch("hooks.hook_helpers.LOG_PATH", log_path):
            exc = sqlite3.OperationalError("database is locked")
            _log_db_error("test_context", exc)
        assert not os.path.exists(corruption_path), "corruption.log should NOT be created for lock errors"


def test_record_metric_does_not_raise_on_corruption():
    """record_metric should log but never raise — hooks must keep running."""
    from hooks import hook_helpers
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "hook.log")
        # Point at a non-existent DB path to force an error
        bad_db = os.path.join(tmp, "does_not_exist", "nope.db")
        with patch.object(hook_helpers, "LOG_PATH", log_path), \
             patch.object(hook_helpers, "DB_PATH", bad_db):
            # Should not raise
            hook_helpers.record_metric("test-sess", "test-event")
        # Log file should exist with an entry
        if os.path.exists(log_path):
            with open(log_path) as f:
                assert "DB ERROR" in f.read()


def test_save_hook_state_does_not_raise_on_corruption():
    """save_hook_state should log but never raise."""
    from hooks import hook_helpers
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "hook.log")
        bad_db = os.path.join(tmp, "does_not_exist", "nope.db")
        with patch.object(hook_helpers, "LOG_PATH", log_path), \
             patch.object(hook_helpers, "DB_PATH", bad_db):
            hook_helpers.save_hook_state("test-sess", "test-key", "test-value")
        if os.path.exists(log_path):
            with open(log_path) as f:
                assert "DB ERROR" in f.read()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
