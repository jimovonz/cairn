#!/usr/bin/env python3
"""Tests for memory block parser robustness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))

from stop_hook import parse_memory_block


def test_valid_single_entry():
    text = 'response\n<memory>\n- type: fact\n- topic: test\n- content: hello world\n- complete: true\n</memory>'
    entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["type"] == "fact"
    assert entries[0]["topic"] == "test"
    assert entries[0]["content"] == "hello world"
    assert complete is True


def test_noop_block():
    text = 'response\n<memory>\ncomplete: true\n</memory>'
    entries, complete, *_ = parse_memory_block(text)
    assert entries == []
    assert complete is True


def test_missing_block():
    text = 'response with no memory block'
    result = parse_memory_block(text)
    assert result[0] is None  # entries
    assert result[1] is None  # complete


def test_unclosed_tag():
    text = 'response\n<memory>\n- type: fact\n- topic: test\n- content: unclosed'
    entries, complete, *_ = parse_memory_block(text)
    assert entries is not None
    assert len(entries) == 1
    assert entries[0]["content"] == "unclosed"


def test_multiple_entries():
    text = '''response
<memory>
- type: fact
- topic: first
- content: entry one
- type: decision
- topic: second
- content: entry two
- complete: true
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 2
    assert entries[0]["type"] == "fact"
    assert entries[1]["type"] == "decision"
    assert complete is True


def test_source_messages_after_content():
    text = '''response
<memory>
- type: fact
- topic: test
- content: with source
- source_messages: 42-55
- complete: true
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["source_start"] == 42
    assert entries[0]["source_end"] == 55


def test_source_messages_before_content():
    text = '''response
<memory>
- type: fact
- topic: test
- source_messages: 10-20
- content: with source before
- complete: true
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["source_start"] == 10
    assert entries[0]["source_end"] == 20


def test_confidence_updates():
    text = '''response
<memory>
- confidence_update: 42:+
- confidence_update: 17:-
- complete: true
</memory>'''
    entries, complete, remaining, context, context_need, conf_updates, *_ = parse_memory_block(text)
    assert len(conf_updates) == 2
    assert conf_updates[0] == (42, "+")
    assert conf_updates[1] == (17, "-")


def test_context_insufficient():
    text = '''response
<memory>
- context: insufficient
- context_need: what decisions were made about auth
- complete: true
</memory>'''
    entries, complete, remaining, context, context_need, *_ = parse_memory_block(text)
    assert context == "insufficient"
    assert context_need == "what decisions were made about auth"


def test_keywords():
    text = '''response
<memory>
- type: fact
- topic: test
- content: keyword test
- keywords: auth, JWT, session tokens
- complete: true
</memory>'''
    *_, keywords = parse_memory_block(text)
    assert keywords == ["auth", "JWT", "session tokens"]


def test_unknown_type_accepted():
    text = '''response
<memory>
- type: goal
- topic: test
- content: unknown type accepted
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["type"] == "goal"


def test_unknown_fields_ignored():
    text = '''response
<memory>
- type: fact
- topic: test
- content: extra fields
- priority: high
- tags: important
- complete: true
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert complete is True


def test_retrieval_outcome():
    text = '''response
<memory>
- retrieval_outcome: harmful
- complete: true
</memory>'''
    entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords = parse_memory_block(text)
    assert retrieval_outcome == "harmful"


def test_incomplete_with_remaining():
    text = '''response
<memory>
- type: fact
- topic: test
- content: partial work
- complete: false
- remaining: finish the implementation
</memory>'''
    entries, complete, remaining, *_ = parse_memory_block(text)
    assert complete is False
    assert remaining == "finish the implementation"


def test_partial_entry_no_content():
    """Entry with type+topic but missing content should use topic as content."""
    text = '''response
<memory>
- type: fact
- topic: test-topic
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["content"] == "test-topic"


def test_empty_block():
    text = 'response\n<memory>\n</memory>'
    entries, complete, *_ = parse_memory_block(text)
    assert entries == []
    assert complete is True  # default


def test_multiple_memory_blocks_uses_last():
    text = '''response
<memory>
- type: fact
- topic: first-block
- content: should be ignored
- complete: true
</memory>
more text
<memory>
- type: decision
- topic: second-block
- content: should be used
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["topic"] == "second-block"


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
        except Exception as e:
            failed += 1
            print(f"  ERROR: {test.__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
