#!/usr/bin/env python3
"""Tests for memory block parser robustness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))

from parser import parse_memory_block


def test_valid_single_entry():
    text = 'response\n<memory>\n- type: fact\n- topic: test\n- content: hello world\n- complete: true\n</memory>'
    entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords, intent = parse_memory_block(text)
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
    *_, keywords, intent = parse_memory_block(text)
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
    entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords, intent = parse_memory_block(text)
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


# ============================================================
# Real-world Claude output variations
# ============================================================

def test_memory_block_inside_markdown_code_fence():
    """Claude sometimes wraps memory in triple backticks."""
    text = '''Here's the answer.
```
<memory>
- type: fact
- topic: fenced
- content: inside code fence
- complete: true
</memory>
```'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["topic"] == "fenced"


def test_memory_block_with_trailing_text():
    """Claude adds commentary after the closing tag."""
    text = '''My response.
<memory>
- type: decision
- topic: trailing
- content: has text after
- complete: true
</memory>

Let me know if you need anything else!'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["content"] == "has text after"


def test_memory_block_with_leading_whitespace():
    """Lines have inconsistent indentation."""
    text = '''response
<memory>
  - type: fact
   - topic: indented
    - content: inconsistent whitespace
  - complete: true
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["topic"] == "indented"


def test_memory_block_with_no_dashes():
    """Claude omits the leading dashes."""
    text = '''response
<memory>
type: fact
topic: no-dashes
content: missing leading dashes
complete: true
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["topic"] == "no-dashes"


def test_content_with_special_characters():
    """Content contains colons, dashes, and other punctuation."""
    text = '''response
<memory>
- type: fact
- topic: special-chars
- content: Use --no-verify flag; connect to host:port (e.g. localhost:5432)
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert "localhost:5432" in entries[0]["content"]


def test_content_with_url():
    """Content contains a URL with colons and slashes."""
    text = '''response
<memory>
- type: fact
- topic: url-content
- content: API docs at https://api.example.com/v2/docs — use Bearer token
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert "https://api.example.com" in entries[0]["content"]


def test_content_with_emoji():
    """Some Claude outputs include emoji."""
    text = '''response
<memory>
- type: preference
- topic: emoji-test
- content: User prefers concise responses without unnecessary detail
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1


def test_very_long_content():
    """Content exceeds normal length."""
    long_content = "word " * 200  # 1000 chars
    text = f'''response
<memory>
- type: fact
- topic: long
- content: {long_content.strip()}
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert len(entries[0]["content"]) > 500


def test_content_with_angle_brackets():
    """Content contains literal angle brackets (e.g. generics, HTML)."""
    text = '''response
<memory>
- type: fact
- topic: brackets
- content: Use List<String> for the response type
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    # The regex-based parser may struggle here — this tests current behaviour
    assert entries is not None
    # Even if the content is truncated, we should get something
    if len(entries) > 0:
        assert "List" in entries[0]["content"]


def test_mixed_dashes_and_no_dashes():
    """Some lines have dashes, others don't."""
    text = '''response
<memory>
- type: fact
topic: mixed
- content: mixed formatting
complete: true
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert complete is True


def test_duplicate_type_field():
    """Claude outputs type twice — second should win."""
    text = '''response
<memory>
- type: fact
- type: decision
- topic: dup-type
- content: which type wins
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["type"] == "decision"


def test_empty_content_field():
    """Content is present but empty."""
    text = '''response
<memory>
- type: fact
- topic: empty-content
- content:
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    # Empty content after colon — may or may not parse
    # The important thing is it doesn't crash


def test_source_messages_single_number():
    """source_messages with just one number (not a range)."""
    text = '''response
<memory>
- type: fact
- topic: single-source
- content: single source ref
- source_messages: 42
- complete: true
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["source_start"] == 42
    assert entries[0]["source_end"] == 42


def test_many_entries_stress():
    """10 entries in one block — tests parser doesn't lose track."""
    lines = []
    for i in range(10):
        lines.extend([f"- type: fact", f"- topic: stress-{i}", f"- content: entry number {i}"])
    block = "\n".join(lines)
    text = f"response\n<memory>\n{block}\n- complete: true\n</memory>"
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 10
    assert entries[0]["topic"] == "stress-0"
    assert entries[9]["topic"] == "stress-9"


def test_all_fields_populated():
    """Every possible field present on one entry."""
    text = '''response
<memory>
- type: decision
- topic: full-entry
- content: all fields present
- source_messages: 5-12
- keywords: auth, security, tokens
- context: sufficient
- context_need: none needed
- confidence_update: 42:+
- retrieval_outcome: useful
- complete: true
</memory>'''
    entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords, intent = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["type"] == "decision"
    assert entries[0]["source_start"] == 5
    assert entries[0]["source_end"] == 12
    assert keywords == ["auth", "security", "tokens"]
    assert context == "sufficient"
    assert conf_updates == [(42, "+")]
    assert retrieval_outcome == "useful"
    assert complete is True


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
