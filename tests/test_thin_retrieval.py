#!/usr/bin/env python3
"""Tests for L5 thin-retrieval escalation enforcement."""

import json
import os
import tempfile

from hooks.retrieval import _is_thin_retrieval
from hooks.hook_helpers import query_py_invoked_since


# === _is_thin_retrieval ===


# Verifies: empty entry list flagged as thin
def test_thin_retrieval_empty():
    """No entries → thin (reason=empty)."""
    is_thin, diag = _is_thin_retrieval([])
    assert is_thin is True
    assert diag["reason"] == "empty"
    assert diag["count"] == 0


# Verifies: single entry flagged as thin (below MIN_ENTRIES=3)
def test_thin_retrieval_single_entry():
    """One entry → thin (reason=too_few)."""
    entries = [{"similarity": 0.9, "id": 1}]
    is_thin, diag = _is_thin_retrieval(entries)
    assert is_thin is True
    assert diag["reason"] == "too_few"
    assert diag["count"] == 1


# Verifies: two entries flagged as thin (still below MIN_ENTRIES=3)
def test_thin_retrieval_two_entries():
    """Two entries → thin (still below threshold)."""
    entries = [{"similarity": 0.9, "id": 1}, {"similarity": 0.8, "id": 2}]
    is_thin, diag = _is_thin_retrieval(entries)
    assert is_thin is True
    assert diag["reason"] == "too_few"


# Verifies: three weak entries flagged as thin (max_sim < threshold)
def test_thin_retrieval_three_weak_entries():
    """Three entries but all below similarity threshold → thin."""
    entries = [
        {"similarity": 0.30, "id": 1},
        {"similarity": 0.25, "id": 2},
        {"similarity": 0.28, "id": 3},
    ]
    is_thin, diag = _is_thin_retrieval(entries)
    assert is_thin is True
    assert diag["reason"] == "top_too_weak"
    assert diag["max_sim"] == 0.30


# Verifies: three strong entries pass the gate
def test_thin_retrieval_three_strong_entries():
    """Three entries with strong similarity → not thin."""
    entries = [
        {"similarity": 0.70, "id": 1},
        {"similarity": 0.55, "id": 2},
        {"similarity": 0.50, "id": 3},
    ]
    is_thin, diag = _is_thin_retrieval(entries)
    assert is_thin is False
    assert diag["reason"] == "ok"
    assert diag["max_sim"] == 0.70


# Verifies: mixed entries pass if max similarity exceeds threshold
def test_thin_retrieval_mixed_entries_max_strong_passes():
    """If even one entry is above the similarity threshold, retrieval is not thin."""
    entries = [
        {"similarity": 0.55, "id": 1},
        {"similarity": 0.30, "id": 2},
        {"similarity": 0.20, "id": 3},
    ]
    is_thin, diag = _is_thin_retrieval(entries)
    assert is_thin is False
    assert diag["max_sim"] == 0.55


# Verifies: missing similarity field treated as zero
def test_thin_retrieval_missing_similarity_treated_as_zero():
    """Entries without a similarity field default to 0 (still thin)."""
    entries = [{"id": 1}, {"id": 2}, {"id": 3}]
    is_thin, diag = _is_thin_retrieval(entries)
    assert is_thin is True
    assert diag["reason"] == "top_too_weak"
    assert diag["max_sim"] == 0.0


# === query_py_invoked_since ===


def _make_transcript(turns: list[dict]) -> str:
    """Build a temp transcript JSONL file matching Claude Code format."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")
    return path


# Verifies: missing transcript path returns False
def test_query_py_invoked_missing_path():
    """Non-existent transcript path returns False."""
    assert query_py_invoked_since("/nonexistent/path.jsonl", "2026-01-01T00:00:00") is False


# Verifies: empty path returns False
def test_query_py_invoked_empty_path():
    """Empty transcript path returns False."""
    assert query_py_invoked_since("", "2026-01-01T00:00:00") is False


# Verifies: substantive query.py call after timestamp returns True
def test_query_py_invoked_substantive_after():
    """Bash call with query.py --semantic after the cutoff timestamp counts."""
    path = _make_transcript([
        {
            "timestamp": "2026-04-09T12:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": "python3 cairn/query.py --semantic 'James role'"},
                }],
            },
        },
    ])
    try:
        assert query_py_invoked_since(path, "2026-04-09T11:00:00") is True
    finally:
        os.remove(path)


# Verifies: trivial query.py invocation does NOT satisfy escalation
def test_query_py_invoked_trivial_does_not_count():
    """query.py --stats is not a substantive search."""
    path = _make_transcript([
        {
            "timestamp": "2026-04-09T12:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": "python3 cairn/query.py --stats"},
                }],
            },
        },
    ])
    try:
        assert query_py_invoked_since(path, "2026-04-09T11:00:00") is False
    finally:
        os.remove(path)


# Verifies: query.py call BEFORE the cutoff doesn't count
def test_query_py_invoked_before_cutoff_excluded():
    """Earlier query.py invocations don't satisfy a later escalation flag."""
    path = _make_transcript([
        {
            "timestamp": "2026-04-09T10:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": "python3 cairn/query.py --semantic 'old query'"},
                }],
            },
        },
    ])
    try:
        assert query_py_invoked_since(path, "2026-04-09T11:00:00") is False
    finally:
        os.remove(path)


# Verifies: bare positional search (query.py "term") counts as substantive
def test_query_py_invoked_bare_positional_counts():
    """query.py with positional FTS arg (no --flag) counts as substantive."""
    path = _make_transcript([
        {
            "timestamp": "2026-04-09T12:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": "python3 cairn/query.py 'brother'"},
                }],
            },
        },
    ])
    try:
        assert query_py_invoked_since(path, "2026-04-09T11:00:00") is True
    finally:
        os.remove(path)


# Verifies: malformed JSONL lines are skipped silently
def test_query_py_invoked_malformed_lines_skipped():
    """Garbage lines in the transcript don't crash the scanner."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        f.write("not valid json\n")
        f.write(json.dumps({
            "timestamp": "2026-04-09T12:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": "python3 cairn/query.py --semantic 'test'"},
                }],
            },
        }) + "\n")
        f.write("more garbage\n")
    try:
        assert query_py_invoked_since(path, "2026-04-09T11:00:00") is True
    finally:
        os.remove(path)


# Verifies: non-Bash tool calls are ignored
def test_query_py_invoked_non_bash_ignored():
    """A Read tool call mentioning query.py doesn't count — only Bash counts."""
    path = _make_transcript([
        {
            "timestamp": "2026-04-09T12:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "name": "Read",
                    "input": {"file_path": "/home/james/cairn/query.py"},
                }],
            },
        },
    ])
    try:
        assert query_py_invoked_since(path, "2026-04-09T11:00:00") is False
    finally:
        os.remove(path)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
