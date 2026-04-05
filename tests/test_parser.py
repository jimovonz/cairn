#!/usr/bin/env python3
"""Tests for memory block parser robustness."""

import sys
import os

from hooks.parser import parse_memory_block


# Verifies: single well-formed entry parses all fields correctly
def test_valid_single_entry():
    text = 'response\n<memory>\n- type: fact\n- topic: test\n- content: hello world\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>'
    parsed = parse_memory_block(text); entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords, intent = parsed.entries, parsed.complete, parsed.remaining, parsed.context, parsed.context_need, parsed.confidence_updates, parsed.retrieval_outcome, parsed.keywords, parsed.intent
    assert len(entries) == 1
    assert entries[0]["type"] == "fact"
    assert entries[0]["topic"] == "test"
    assert entries[0]["content"] == "hello world"
    assert complete is True


# Verifies: block with only complete flag yields empty entries
def test_noop_block():
    text = 'response\n<memory>\ncomplete: true\n</memory>'
    entries, complete, *_ = parse_memory_block(text)
    assert entries == []
    assert complete is True


# Verifies: response without <memory> tag returns None values
def test_missing_block():
    text = 'response with no memory block'
    result = parse_memory_block(text)
    assert result[0] is None  # entries
    assert result[1] is None  # complete


# Verifies: unclosed <memory> tag still parses entries
def test_unclosed_tag():
    text = 'response\n<memory>\n- type: fact\n- topic: test\n- content: unclosed'
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["content"] == "unclosed"


# Verifies: two entries in one block are split correctly
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
- context: sufficient
- keywords: test
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 2
    assert entries[0]["type"] == "fact"
    assert entries[1]["type"] == "decision"
    assert complete is True


# Verifies: depth field parses as int when after content
def test_depth_after_content():
    text = '''response
<memory>
- type: fact
- topic: test
- content: with depth
- depth: 3
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["depth"] == 3


# Verifies: depth field parses as int when before content
def test_depth_before_content():
    text = '''response
<memory>
- type: fact
- topic: test
- depth: 5
- content: with depth before
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["depth"] == 5


# Verifies: confidence_update lines parse id, direction, reason
def test_confidence_updates():
    text = '''response
<memory>
- confidence_update: 42:+
- confidence_update: 17:-
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, complete, remaining, context, context_need, conf_updates, *_ = parse_memory_block(text)
    assert len(conf_updates) == 2
    assert conf_updates[0] == (42, "+", None)
    assert conf_updates[1] == (17, "-", None)


# Verifies: context insufficient with context_need is extracted
def test_context_insufficient():
    text = '''response
<memory>
- context: insufficient
- context_need: what decisions were made about auth
- keywords: auth, decisions
- complete: true
</memory>'''
    entries, complete, remaining, context, context_need, *_ = parse_memory_block(text)
    assert context == "insufficient"
    assert context_need == "what decisions were made about auth"


# Verifies: comma-separated keywords parse into list
def test_keywords():
    text = '''response
<memory>
- type: fact
- topic: test
- content: keyword test
- keywords: auth, JWT, session tokens
- complete: true
- context: sufficient
</memory>'''
    parsed = parse_memory_block(text)
    assert parsed.keywords == ["auth", "JWT", "session tokens"]


# Verifies: non-standard type values are accepted without error
def test_unknown_type_accepted():
    text = '''response
<memory>
- type: goal
- topic: test
- content: unknown type accepted
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["type"] == "goal"


# Verifies: unrecognized fields do not cause parse failure
def test_unknown_fields_ignored():
    text = '''response
<memory>
- type: fact
- topic: test
- content: extra fields
- priority: high
- tags: important
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert complete is True


# Verifies: retrieval_outcome field is extracted correctly
def test_retrieval_outcome():
    text = '''response
<memory>
- retrieval_outcome: harmful
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    parsed = parse_memory_block(text); entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords, intent = parsed.entries, parsed.complete, parsed.remaining, parsed.context, parsed.context_need, parsed.confidence_updates, parsed.retrieval_outcome, parsed.keywords, parsed.intent
    assert retrieval_outcome == "harmful"


# Verifies: complete=false with remaining text is extracted
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


# Verifies: missing content field falls back to topic value
def test_partial_entry_no_content():
    """Entry with type+topic but missing content should use topic as content."""
    text = '''response
<memory>
- type: fact
- topic: test-topic
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["content"] == "test-topic"


# Verifies: empty memory block yields no entries and None complete
def test_empty_block():
    text = 'response\n<memory>\n</memory>'
    entries, complete, *_ = parse_memory_block(text)
    assert entries == []
    assert complete is None  # omitted complete field defaults to None (incomplete)


# Verifies: when multiple memory blocks exist, last one wins
def test_multiple_memory_blocks_uses_last():
    text = '''response
<memory>
- type: fact
- topic: first-block
- content: should be ignored
- complete: true
- context: sufficient
- keywords: test
</memory>
more text
<memory>
- type: decision
- topic: second-block
- content: should be used
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["topic"] == "second-block"


# ============================================================
# Real-world Claude output variations
# ============================================================

# Verifies: memory block inside markdown code fence is parsed
def test_memory_block_inside_markdown_code_fence():
    """Claude sometimes wraps memory in triple backticks."""
    text = '''Here's the answer.
```
<memory>
- type: fact
- topic: fenced
- content: inside code fence
- complete: true
- context: sufficient
- keywords: test
</memory>
```'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["topic"] == "fenced"


# Verifies: text after closing </memory> tag is ignored
def test_memory_block_with_trailing_text():
    """Claude adds commentary after the closing tag."""
    text = '''My response.
<memory>
- type: decision
- topic: trailing
- content: has text after
- complete: true
- context: sufficient
- keywords: test
</memory>

Let me know if you need anything else!'''
    entries, complete, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["content"] == "has text after"


# Verifies: inconsistent line indentation is handled gracefully
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


# Verifies: fields without leading dashes still parse
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


# Verifies: colons, dashes, and punctuation in content preserved
def test_content_with_special_characters():
    """Content contains colons, dashes, and other punctuation."""
    text = '''response
<memory>
- type: fact
- topic: special-chars
- content: Use --no-verify flag; connect to host:port (e.g. localhost:5432)
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert "localhost:5432" in entries[0]["content"]


# Verifies: URLs with colons and slashes in content preserved
def test_content_with_url():
    """Content contains a URL with colons and slashes."""
    text = '''response
<memory>
- type: fact
- topic: url-content
- content: API docs at https://api.example.com/v2/docs — use Bearer token
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert "https://api.example.com" in entries[0]["content"]


# Verifies: emoji characters in content do not break parsing
def test_content_with_emoji():
    """Some Claude outputs include emoji."""
    text = '''response
<memory>
- type: preference
- topic: emoji-test
- content: User prefers concise responses without unnecessary detail
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1


# Verifies: very long content (1000+ chars) parses without truncation
def test_very_long_content():
    """Content exceeds normal length."""
    long_content = "word " * 200  # 1000 chars
    text = f'''response
<memory>
- type: fact
- topic: long
- content: {long_content.strip()}
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert len(entries[0]["content"]) > 500


# Verifies: angle brackets in content do not break tag parsing
def test_content_with_angle_brackets():
    """Content contains literal angle brackets (e.g. generics, HTML)."""
    text = '''response
<memory>
- type: fact
- topic: brackets
- content: Use List<String> for the response type
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    # The regex-based parser may struggle here — this tests current behaviour
    assert len(entries) >= 1
    assert "List" in entries[0]["content"]


# Verifies: mixed dash/no-dash formatting parses correctly
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


# Verifies: duplicate type field uses last value
def test_duplicate_type_field():
    """Claude outputs type twice — second should win."""
    text = '''response
<memory>
- type: fact
- type: decision
- topic: dup-type
- content: which type wins
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["type"] == "decision"


# Verifies: empty content value does not crash parser
def test_empty_content_field():
    """Content is present but empty."""
    text = '''response
<memory>
- type: fact
- topic: empty-content
- content:
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    # Empty content after colon — may or may not parse
    # The important thing is it doesn't crash


# Verifies: depth value of 1 parses correctly
def test_depth_single_turn():
    """depth with value of 1."""
    text = '''response
<memory>
- type: fact
- topic: single-depth
- content: single turn depth
- depth: 1
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 1
    assert entries[0]["depth"] == 1


# Verifies: 10 entries in one block all parse without loss
def test_many_entries_stress():
    """10 entries in one block — tests parser doesn't lose track."""
    lines = []
    for i in range(10):
        lines.extend([f"- type: fact", f"- topic: stress-{i}", f"- content: entry number {i}"])
    block = "\n".join(lines)
    text = f"response\n<memory>\n{block}\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    entries, *_ = parse_memory_block(text)
    assert len(entries) == 10
    assert entries[0]["topic"] == "stress-0"
    assert entries[9]["topic"] == "stress-9"


# Verifies: all possible fields present parse correctly together
def test_all_fields_populated():
    """Every possible field present on one entry."""
    text = '''response
<memory>
- type: decision
- topic: full-entry
- content: all fields present
- depth: 4
- keywords: auth, security, tokens
- context: sufficient
- context_need: none needed
- confidence_update: 42:+
- retrieval_outcome: useful
- complete: true
</memory>'''
    parsed = parse_memory_block(text); entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords, intent = parsed.entries, parsed.complete, parsed.remaining, parsed.context, parsed.context_need, parsed.confidence_updates, parsed.retrieval_outcome, parsed.keywords, parsed.intent
    assert len(entries) == 1
    assert entries[0]["type"] == "decision"
    assert entries[0]["depth"] == 4
    assert keywords == ["auth", "security", "tokens"]
    assert context == "sufficient"
    assert conf_updates == [(42, "+", None)]
    assert retrieval_outcome == "useful"
    assert complete is True


# Verifies: -! syntax extracts direction and reason separately
def test_contradiction_annotation_parsing():
    """The -! syntax should extract direction and reason as separate fields."""
    text = '''response
<memory>
- confidence_update: 42:-! replaced by GCE edge approach
- confidence_update: 17:-! no longer valid since migration
- confidence_update: 5:+
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    parsed = parse_memory_block(text)
    conf_updates = parsed.confidence_updates
    assert len(conf_updates) == 3
    assert conf_updates[0] == (42, "-!", "replaced by GCE edge approach")
    assert conf_updates[1] == (17, "-!", "no longer valid since migration")
    assert conf_updates[2] == (5, "+", None)


# Verifies: -! without reason text parses with None reason
def test_contradiction_annotation_without_reason():
    """-! with no reason should still parse, with None reason."""
    text = '''response
<memory>
- confidence_update: 42:-!
- complete: true
- context: sufficient
- keywords: test
</memory>'''
    parsed = parse_memory_block(text)
    conf_updates = parsed.confidence_updates
    assert len(conf_updates) == 1
    assert conf_updates[0][0] == 42
    assert conf_updates[0][1] == "-!"
    assert conf_updates[0][2] is None or conf_updates[0][2] == ""


# ============================================================
# Compact format tests
# ============================================================

# Verifies: compact type/topic: content [k:] format parses fully
def test_compact_single_entry():
    text = 'Some response.\n<memory>\nfact/test-topic: This is the content [k: testing, parser]\n+ c h:34\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert len(parsed.entries) == 1
    assert parsed.entries[0]["type"] == "fact"
    assert parsed.entries[0]["topic"] == "test-topic"
    assert parsed.entries[0]["content"] == "This is the content"
    assert parsed.keywords == ["testing", "parser"]
    assert parsed.complete is True
    assert parsed.context == "sufficient"
    assert parsed.hash_claimed == 0x34  # hex parsed


# Verifies: compact noop "." with hash yields empty entries
def test_compact_noop():
    text = 'Merged to main. Deployment triggered.\n<memory>\n. h:11\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert parsed.entries == []
    assert parsed.complete is True
    assert parsed.hash_claimed == 0x11  # hex 11 = decimal 17


# Verifies: bare compact noop "." without hash parses cleanly
def test_compact_noop_bare():
    text = 'Done.\n<memory>\n.\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert parsed.entries == []
    assert parsed.complete is True
    assert parsed.hash_claimed is None


# Verifies: compact "- :remaining" sets incomplete with reason
def test_compact_incomplete():
    text = 'Found the bug.\n<memory>\nfact/redis-oom: Redis container OOM from leaked connection pools [k: testing, redis]\n- :fix test fixture teardown\nh:2E\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert len(parsed.entries) == 1
    assert parsed.complete is False
    assert parsed.remaining == "fix test fixture teardown"
    assert parsed.hash_claimed == 0x2E  # hex 2E = decimal 46


# Verifies: compact "c?:query" sets context insufficient
def test_compact_context_insufficient():
    text = 'Need more info.\n<memory>\n.\nc?:what was decided about auth middleware\nh:E\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert parsed.context == "insufficient"
    assert parsed.context_need == "what was decided about auth middleware"
    assert parsed.hash_claimed == 0xE  # hex E = decimal 14


# Verifies: confidence_update lines work in compact format
def test_compact_confidence_updates():
    text = 'Response.\n<memory>\nfact/test: Some fact [k: test]\nconfidence_update: 42:+\nconfidence_update: 17:-! no longer valid\n+ c h:12\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert len(parsed.confidence_updates) == 2
    assert parsed.confidence_updates[0] == (42, "+", None)
    assert parsed.confidence_updates[1] == (17, "-!", "no longer valid")


# Verifies: multiple compact entries in one block split correctly
def test_compact_multiple_entries():
    text = '''Response text.
<memory>
fact/first-thing: This is the first entry [k: one]
decision/second-thing: This is the second entry [k: two]
+ c h:12
</memory>'''
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert len(parsed.entries) == 2
    assert parsed.entries[0]["type"] == "fact"
    assert parsed.entries[0]["topic"] == "first-thing"
    assert parsed.entries[1]["type"] == "decision"
    assert parsed.entries[1]["topic"] == "second-thing"


# Verifies: retrieval_outcome works in compact format
def test_compact_with_retrieval_outcome():
    text = 'Response.\n<memory>\nfact/test: Content [k: test]\nretrieval_outcome: useful\n+ c h:12\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert parsed.retrieval_outcome == "useful"


# Verifies: intent field works in compact format
def test_compact_with_intent():
    text = 'Response.\n<memory>\nfact/test: Content [k: test]\nintent: resolved\n+ c h:12\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.is_compact is True
    assert parsed.intent == "resolved"


# Verifies: colons in compact content do not break parsing
def test_compact_content_with_colons():
    """Compact content containing colons should not break parsing."""
    text = 'Response.\n<memory>\nfact/api-endpoint: Connect to host:5432 using SSL [k: database]\n+ c h:12\n</memory>'
    parsed = parse_memory_block(text)
    assert len(parsed.entries) == 1
    assert "host:5432" in parsed.entries[0]["content"]


# Verifies: URLs in compact content are preserved intact
def test_compact_content_with_url():
    """Compact content containing URL with slashes."""
    text = 'Response.\n<memory>\nfact/docs: API at https://api.example.com/v2 [k: api]\n+ c h:12\n</memory>'
    parsed = parse_memory_block(text)
    assert len(parsed.entries) == 1
    assert "https://api.example.com/v2" in parsed.entries[0]["content"]


# Verifies: c?:query on control line sets context insufficient
def test_compact_context_insufficient_on_control_line():
    """Context insufficient declared on the control line."""
    text = 'Response.\n<memory>\nfact/test: Content [k: test]\n+ c?:auth decisions h:12\n</memory>'
    parsed = parse_memory_block(text)
    assert parsed.context == "insufficient"
    assert parsed.context_need == "auth decisions"
    assert parsed.hash_claimed == 0x12  # hex 12 = decimal 18


# Verifies: verbose format still works with compact parser present
def test_verbose_format_unchanged():
    """Verbose format should still work exactly as before."""
    text = '''response
<memory>
- type: fact
- topic: test
- content: verbose still works
- complete: true
- context: sufficient
- keywords: backwards, compat
</memory>'''
    parsed = parse_memory_block(text)
    assert parsed.is_compact is False
    assert len(parsed.entries) == 1
    assert parsed.entries[0]["content"] == "verbose still works"
    assert parsed.keywords == ["backwards", "compat"]
    assert parsed.hash_claimed is None


# Verifies: compact entry without [k:] has empty keywords list
def test_compact_no_keywords():
    """Compact entry without [k: ...] suffix."""
    text = 'Response.\n<memory>\nfact/test: Content without keywords\n+ c h:12\n</memory>'
    parsed = parse_memory_block(text)
    assert len(parsed.entries) == 1
    assert parsed.entries[0]["content"] == "Content without keywords"
    assert parsed.keywords == []
    assert parsed.keywords_explicit is False


# ============================================================
# Hash computation tests
# ============================================================

from hooks.hash_verify import compute_response_hash, verify_hash


# Verifies: hash sums first-char values of each sentence
def test_hash_simple_response():
    """Hash of a simple multi-sentence response."""
    text = "The fix is ready. Tests pass."
    # t=20, t=20 → 40
    assert compute_response_hash(text) == 40


# Verifies: memory block is excluded from hash computation
def test_hash_strips_memory_block():
    """Memory block should be excluded from hash."""
    text = "Found it. Applied the fix.\n<memory>\nfact/fix: details [k: test]\n+ c h:15\n</memory>"
    # f=6, a=1 → 7 (memory block stripped)
    assert compute_response_hash(text) == 7


# Verifies: fenced code blocks are excluded from hash
def test_hash_strips_code_blocks():
    """Fenced code blocks should be excluded."""
    text = "Here's the change:\n\n```python\ndef foo():\n    pass\n```\n\nThis should work."
    # h=8, t=20 → 28
    h = compute_response_hash(text)
    assert h == 28


# Verifies: double newlines act as sentence boundaries for hash
def test_hash_paragraph_breaks():
    """Double newlines treated as sentence boundaries."""
    text = "First paragraph here.\n\nSecond paragraph starts."
    # f=6, s=19 → 25
    assert compute_response_hash(text) == 25


# Verifies: empty or memory-only response hashes to 0
def test_hash_empty_response():
    """Empty response returns 0."""
    assert compute_response_hash("") == 0
    assert compute_response_hash("<memory>\n.\n</memory>") == 0


# Verifies: single sentence hashes to its first char value
def test_hash_single_sentence():
    """Single sentence response."""
    text = "Done."
    # d=4
    assert compute_response_hash(text) == 4


# Verifies: verify_hash returns True when claimed hash matches
def test_hash_verify_match():
    """Verify returns True on correct hash."""
    text = "The fix is ready. Tests pass.\n<memory>\n. h:28\n</memory>"
    match, actual = verify_hash(text, 40)  # 40 decimal, 0x28 hex
    assert match is True
    assert actual == 40


# Verifies: verify_hash returns False when claimed hash is wrong
def test_hash_verify_mismatch():
    """Verify returns False on incorrect hash."""
    text = "The fix is ready. Tests pass.\n<memory>\n. h:63\n</memory>"
    match, actual = verify_hash(text, 99)  # 99 decimal, 0x63 hex
    assert match is False
    assert actual == 40


# Verifies: ! and ? are treated as sentence boundaries
def test_hash_exclamation_and_question():
    """Sentences ending with ! and ? are boundaries too."""
    text = "What happened? Something broke! Fix it now."
    # w=23, s=19, f=6 → 48
    assert compute_response_hash(text) == 48


# Verifies: code block fences act as sentence breaks for hash
def test_hash_multiline_code_block_boundary():
    """Code block boundaries act as sentence breaks."""
    text = "Before code.\n\n```js\nconsole.log('hi')\n```\n\nAfter code."
    # b=2, a=1 → 3
    h = compute_response_hash(text)
    assert h == 3


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
