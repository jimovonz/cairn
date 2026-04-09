#!/usr/bin/env python3
"""Tests for L2 query-quality enforcement and last_user_message helper."""

import json
import os
import tempfile

from hooks.hook_helpers import last_user_message
from hooks.stop_hook import _is_phoned_in_context_need


def _make_transcript(turns: list[dict]) -> str:
    """Build a temp transcript JSONL file matching Claude Code format."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")
    return path


# === last_user_message ===


# Verifies: returns most recent user message text
def test_last_user_message_basic():
    """Returns the most recent user-role message string content."""
    path = _make_transcript([
        {"message": {"role": "user", "content": "first"}},
        {"message": {"role": "assistant", "content": "response"}},
        {"message": {"role": "user", "content": "second"}},
    ])
    try:
        assert last_user_message(path) == "second"
    finally:
        os.remove(path)


# Verifies: returns empty string for missing transcript
def test_last_user_message_missing_path():
    """Non-existent transcript returns empty string."""
    assert last_user_message("/nonexistent/path.jsonl") == ""


# Verifies: skips command messages (slash commands)
def test_last_user_message_skips_command_messages():
    """Slash command wrappers are skipped."""
    path = _make_transcript([
        {"message": {"role": "user", "content": "real question"}},
        {"message": {"role": "user", "content": "<command-message>cairn</command-message>"}},
    ])
    try:
        assert last_user_message(path) == "real question"
    finally:
        os.remove(path)


# Verifies: handles list-of-blocks content format
def test_last_user_message_list_content():
    """Content as a list of blocks gets text extracted from text-typed blocks."""
    path = _make_transcript([
        {"message": {"role": "user", "content": [
            {"type": "text", "text": "what is the answer"},
        ]}},
    ])
    try:
        assert last_user_message(path) == "what is the answer"
    finally:
        os.remove(path)


# === _is_phoned_in_context_need ===


# Verifies: context_need with overlapping terms is NOT phoned in
def test_phoned_in_overlap_pass():
    """When context_need shares substantive words with user message, not phoned in."""
    path = _make_transcript([
        {"message": {"role": "user", "content": "what is my brother's profession"}},
    ])
    try:
        assert _is_phoned_in_context_need("user's brother profession", path) is False
    finally:
        os.remove(path)


# Verifies: generic context_need with no overlap IS phoned in
def test_phoned_in_no_overlap_flagged():
    """Generic context_need with zero overlap to user message is phoned in."""
    path = _make_transcript([
        {"message": {"role": "user", "content": "what is my brother's profession"}},
    ])
    try:
        assert _is_phoned_in_context_need("general project standing context", path) is True
    finally:
        os.remove(path)


# Verifies: missing user message returns False (give benefit of doubt)
def test_phoned_in_no_user_message():
    """Empty transcript: assume not phoned in to avoid false positives."""
    path = _make_transcript([])
    try:
        assert _is_phoned_in_context_need("anything", path) is False
    finally:
        os.remove(path)


# Verifies: stopwords are excluded from overlap calculation
def test_phoned_in_stopwords_excluded():
    """Common words like 'context', 'information', 'general' don't count as overlap."""
    path = _make_transcript([
        {"message": {"role": "user", "content": "what is my brother's profession"}},
    ])
    try:
        # Both contain "context information" which are stopwords — no real overlap
        assert _is_phoned_in_context_need("general context information", path) is True
    finally:
        os.remove(path)


# Verifies: short words (<4 chars) don't count
def test_phoned_in_short_words_excluded():
    """Words shorter than 4 chars don't count even if they overlap."""
    path = _make_transcript([
        {"message": {"role": "user", "content": "go to bed now"}},
    ])
    try:
        # 'go', 'to', 'bed', 'now' all short or stopwords
        assert _is_phoned_in_context_need("hi yo go", path) is False  # nothing substantive
    finally:
        os.remove(path)


# Verifies: multi-dimensional question with at least one keyword overlap passes
def test_phoned_in_multi_dimension_partial_overlap_passes():
    """Even partial overlap (one substantive term) is enough — not phoned in."""
    path = _make_transcript([
        {"message": {"role": "user", "content": "tell me about my brother and my job"}},
    ])
    try:
        # Mentions 'brother' but not 'job'
        assert _is_phoned_in_context_need("user's brother family", path) is False
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
