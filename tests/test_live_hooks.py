#!/usr/bin/env python3
"""Live integration test — sends a real prompt through claude -p and verifies
the hook pipeline works end-to-end.

NOT portable. Requires:
- claude CLI installed and authenticated
- Cairn installed (install.sh has been run)
- Embedding daemon running

Run manually: python3 tests/test_live_hooks.py
Not included in pytest suite (no test_ prefix pattern match by default).
"""

import shutil
import subprocess
import json
import time
import os
import sys
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3

import pytest

# Skip entire module if claude CLI is not installed
pytestmark = pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="Requires claude CLI installed and authenticated",
)

CAIRN_DB = os.path.join(os.path.dirname(__file__), "..", "cairn", "cairn.db")
HOOK_LOG = os.path.join(os.path.dirname(__file__), "..", "cairn", "hook.log")


def get_log_size():
    try:
        return os.path.getsize(HOOK_LOG)
    except FileNotFoundError:
        return 0


def get_log_tail(from_pos):
    try:
        with open(HOOK_LOG, "r") as f:
            f.seek(from_pos)
            return f.read()
    except FileNotFoundError:
        return ""


def get_memory_count():
    conn = sqlite3.connect(CAIRN_DB)
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    return count


def run_claude(prompt, timeout=60):
    """Send a prompt through claude -p and return the result."""
    result = subprocess.run(
        ["claude", "-p", "--output-format", "json", prompt],
        capture_output=True, text=True, timeout=timeout
    )
    return result


# Verifies: both hooks fire and read input fields correctly
def test_hooks_fire_and_fields_valid():
    """Send a prompt through claude -p, verify both hooks fire and read their
    input fields correctly. This catches field name renames in Claude Code updates."""
    log_before = get_log_size()
    count_before = get_memory_count()

    result = run_claude(
        "Store a test memory with type: fact, topic: cairn-smoke-test, "
        "content: live hook integration test verifying hook pipeline. Reply briefly."
    )

    if result.returncode != 0:
        print(f"FAIL: claude -p returned {result.returncode}")
        print(f"  stderr: {result.stderr[:200]}")
        return False

    time.sleep(1)
    new_log = get_log_tail(log_before)

    # Stop hook must have fired
    if "Hook fired" not in new_log:
        print("FAIL: Stop hook did not fire")
        return False

    # Check for field name warnings — the critical check
    if "No text found in hook input" in new_log:
        print("FAIL: Stop hook couldn't read response text — field name changed")
        print(f"  Log keys line: {[l for l in new_log.split(chr(10)) if 'Keys:' in l]}")
        return False

    if "No user message found in hook input" in new_log:
        print("FAIL: Prompt hook couldn't read user message — field name changed")
        print(f"  Log keys line: {[l for l in new_log.split(chr(10)) if 'Keys:' in l]}")
        return False

    # Memory should have been stored (proves stop hook parsed the response)
    count_after = get_memory_count()
    if count_after <= count_before:
        # Not a hard failure — LLM might not have complied, but log the concern
        print(f"WARN: No new memory stored (before={count_before}, after={count_after})")

    print("PASS: hooks_fire_and_fields_valid")
    return True


def cleanup():
    """Remove smoke test memories."""
    conn = sqlite3.connect(CAIRN_DB)
    deleted = conn.execute(
        "DELETE FROM memories WHERE topic = 'cairn-smoke-test'"
    ).rowcount
    conn.commit()
    conn.close()
    if deleted:
        print(f"Cleaned up {deleted} smoke test memories")


if __name__ == "__main__":
    print("=== Cairn Live Hook Smoke Test ===\n")

    tests = [test_hooks_fire_and_fields_valid]
    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            failed += 1

    cleanup()

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
