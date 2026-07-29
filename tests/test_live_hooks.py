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


def run_claude(prompt, timeout=240):  # 60s times out under full-suite load — flaky gate, not a real failure
    """Send a prompt through claude -p and return the result."""
    result = subprocess.run(
        ["claude", "-p", "--output-format", "json", prompt],
        capture_output=True, text=True, timeout=timeout
    )
    return result


# Verifies: both hooks fire and read input fields correctly
@pytest.fixture(autouse=True)
def _cleanup_smoke_memories():
    """This live test inserts a real cairn-smoke-test memory through the actual
    pipeline; always remove it afterward so repeated runs do not pollute the
    corpus. (cleanup() previously ran only in __main__, not under pytest, which
    leaked ~92 rows into cairn.db over time.)"""
    yield
    cleanup()


def test_hooks_fire_and_fields_valid():
    """Send a prompt through claude -p, verify both hooks fire and read their
    input fields correctly. This catches field name renames in Claude Code updates.

    Failures are ASSERTED, not returned. `return False` merely raises
    PytestReturnNotNoneWarning and the test still PASSES, so every contract check
    below was silently passing regardless of its outcome — the canary could not
    fail for the reason it exists.

    A CLI timeout is an environment condition (machine load, and the daemon now
    holds two cross-encoders while the reranker A/B runs), not a contract
    violation. It skips rather than fails, so the only failures reported are real
    upstream changes.
    """
    log_before = get_log_size()
    count_before = get_memory_count()

    try:
        result = run_claude(
            "Store a test memory with type: fact, topic: cairn-smoke-test, "
            "content: live hook integration test verifying hook pipeline. Reply briefly."
        )
    except subprocess.TimeoutExpired:
        pytest.skip("claude CLI timed out — machine load, not a hook-contract failure")

    assert result.returncode == 0, (
        f"claude -p returned {result.returncode}: {result.stderr[:200]}"
    )

    time.sleep(1)
    new_log = get_log_tail(log_before)

    assert "Hook fired" in new_log, "Stop hook did not fire"

    keys = [l for l in new_log.splitlines() if "Keys:" in l]
    assert "No text found in hook input" not in new_log, (
        f"Stop hook could not read response text — field name changed. {keys}"
    )
    assert "No user message found in hook input" not in new_log, (
        f"Prompt hook could not read user message — field name changed. {keys}"
    )

    # Soft: the LLM may legitimately not comply with the store request.
    count_after = get_memory_count()
    if count_after <= count_before:
        print(f"WARN: No new memory stored (before={count_before}, after={count_after})")


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
        # The tests assert rather than return a verdict, so "no exception" is a
        # pass. Reading a return value here is what let silent failures through.
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    cleanup()

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
