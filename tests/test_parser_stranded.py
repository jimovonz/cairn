#!/usr/bin/env python3
"""Stranded tests for hooks/parser.py — 4-strand format."""

import sys
import os

import pytest
from hooks.parser import parse_memory_block, NO_BLOCK, NOOP_BLOCK, ParseResult


# ---------------------------------------------------------------------------
# 1. parse_memory_block — format detection and tag extraction
# ---------------------------------------------------------------------------

#TAG: [8C9C] 2026-04-05
# Verifies: closed <memory> tags are found and the last block is used when multiple exist
@pytest.mark.behavioural
def test_parse_memory_block_uses_last_closed_tag():
    text = (
        "<memory>\n- type: fact\n- topic: first\n- content: ignore me\n"
        "- complete: true\n- context: sufficient\n- keywords: x\n</memory>\n"
        "Some text\n"
        "<memory>\n- type: decision\n- topic: second\n- content: keep me\n"
        "- complete: false\n- remaining: more\n- context: sufficient\n- keywords: y\n</memory>"
    )
    result = parse_memory_block(text)
    assert len(result.entries) == 1
    assert result.entries[0]["type"] == "decision"
    assert result.entries[0]["topic"] == "second"
    assert result.entries[0]["content"] == "keep me"
    assert result.complete is False


#TAG: [93FF] 2026-04-05
# Verifies: unclosed <memory> tag is parsed as fallback when no closing tag exists
@pytest.mark.edge
def test_parse_memory_block_unclosed_tag():
    text = "response text\n<memory>\n- type: fact\n- topic: orphan\n- content: data\n- complete: true\n- context: sufficient\n- keywords: z"
    result = parse_memory_block(text)
    assert len(result.entries) == 1
    assert result.entries[0]["type"] == "fact"
    assert result.entries[0]["topic"] == "orphan"
    assert result.entries[0]["content"] == "data"
    assert result.is_compact is False


#TAG: [8287] 2026-04-05
# Verifies: text with no <memory> tag returns NO_BLOCK sentinel
@pytest.mark.error
def test_parse_memory_block_no_tag_returns_no_block():
    result = parse_memory_block("just plain text with no memory tags at all")
    assert result is NO_BLOCK
    assert result.entries is None
    assert result.complete is None
    assert result.confidence_updates == []


#TAG: [8277] 2026-04-05
# Verifies: compact format is detected via type/topic: pattern and dispatched correctly
@pytest.mark.adversarial
def test_parse_memory_block_compact_detection():
    # Compact format: type/topic: content
    text = "<memory>\nfact/test-topic: some content [k: kw1]\n+ c\n</memory>"
    result = parse_memory_block(text)
    assert result.is_compact is True
    assert len(result.entries) == 1
    assert result.entries[0]["type"] == "fact"
    # Verbose format: - type: value
    text2 = "<memory>\n- type: fact\n- topic: test\n- content: hello\n- complete: true\n- context: sufficient\n- keywords: kw1\n</memory>"
    result2 = parse_memory_block(text2)
    assert result2.is_compact is False


# ---------------------------------------------------------------------------
# 2. Compact entry parsing (type/topic: content [k: keywords])
# ---------------------------------------------------------------------------

#TAG: [50C4] 2026-04-05
# Verifies: compact entry with keywords extracts type, topic, content, and keyword list correctly
@pytest.mark.behavioural
def test_compact_entry_with_keywords():
    text = "<memory>\ndecision/db-choice: Use SQLite over PostgreSQL [k: database, storage, sqlite]\n+ c\n</memory>"
    result = parse_memory_block(text)
    assert result.entries[0]["type"] == "decision"
    assert result.entries[0]["topic"] == "db-choice"
    assert result.entries[0]["content"] == "Use SQLite over PostgreSQL"
    assert result.keywords == ["database", "storage", "sqlite"]
    assert result.keywords_explicit is True


#TAG: [8DAC] 2026-04-05
# Verifies: compact entry without [k: ...] suffix keeps full content and keywords remain empty
@pytest.mark.edge
def test_compact_entry_no_keywords():
    text = "<memory>\nfact/info: plain content without keywords\n+ c\n</memory>"
    result = parse_memory_block(text)
    assert result.entries[0]["content"] == "plain content without keywords"
    assert result.keywords == []
    assert result.keywords_explicit is False


#TAG: [1E82] 2026-04-05
# Verifies: compact entry with colons in content is parsed correctly (colon after topic delimiter)
@pytest.mark.error
def test_compact_entry_colons_in_content():
    text = "<memory>\nskill/docker: run with: docker run -p 8080:80 image:latest [k: docker]\n+ c\n</memory>"
    result = parse_memory_block(text)
    assert result.entries[0]["topic"] == "docker"
    assert result.entries[0]["content"] == "run with: docker run -p 8080:80 image:latest"


#TAG: [4227] 2026-04-05
# Verifies: multiple compact entries accumulate and last entry's keywords win
@pytest.mark.adversarial
def test_compact_multiple_entries_keywords_override():
    text = (
        "<memory>\n"
        "fact/first: content one [k: alpha, beta]\n"
        "decision/second: content two [k: gamma]\n"
        "+ c\n"
        "</memory>"
    )
    result = parse_memory_block(text)
    assert len(result.entries) == 2
    assert result.entries[0]["type"] == "fact"
    assert result.entries[1]["type"] == "decision"
    # Keywords from last entry override previous
    assert result.keywords == ["gamma"]


# ---------------------------------------------------------------------------
# 3. Compact control lines (complete/context/hash/noop/incomplete)
# ---------------------------------------------------------------------------

#TAG: [189A] 2026-04-05
# Verifies: control line "+ c h:1A" sets complete=True, context=sufficient, hash=0x1A
@pytest.mark.behavioural
def test_compact_control_line_positive():
    text = "<memory>\nfact/x: data [k: test]\n+ c h:1A\n</memory>"
    result = parse_memory_block(text)
    assert result.complete is True
    assert result.complete_explicit is True
    assert result.context == "sufficient"
    assert result.hash_claimed == 0x1A


#TAG: [2ED1] 2026-04-05
# Verifies: noop "." block with hash is parsed; entries list is empty
@pytest.mark.edge
def test_compact_noop_with_hash():
    text = "<memory>\n. h:FF\n</memory>"
    result = parse_memory_block(text)
    assert result.entries == []
    assert result.hash_claimed == 0xFF
    assert result.is_compact is True


#TAG: [2DEA] 2026-04-05
# Verifies: "- :remaining text" sets complete=False and captures remaining
@pytest.mark.error
def test_compact_incomplete_with_remaining():
    text = "<memory>\nfact/wip: started work [k: progress]\n- :need to finish the migration\n</memory>"
    result = parse_memory_block(text)
    assert result.complete is False
    assert result.remaining == "need to finish the migration"


#TAG: [4AD9] 2026-04-05
# Verifies: control line "- c?:query text h:AB" sets incomplete, insufficient context, and hash
@pytest.mark.adversarial
def test_compact_control_line_all_fields():
    text = "<memory>\nfact/q: question [k: q]\n- c?:what about X h:AB\n</memory>"
    result = parse_memory_block(text)
    assert result.complete is False
    assert result.context == "insufficient"
    assert result.context_need == "what about X"
    assert result.hash_claimed == 0xAB


# ---------------------------------------------------------------------------
# 4. Verbose entry + field parsing
# ---------------------------------------------------------------------------

#TAG: [E71E] 2026-04-05
# Verifies: verbose format parses all standard fields (type, topic, content, complete, context, keywords)
@pytest.mark.behavioural
def test_parse_memory_block_verbose_full_entry():
    text = (
        "<memory>\n"
        "- type: correction\n- topic: bug-fix\n- content: Off-by-one in loop\n"
        "- complete: true\n- context: sufficient\n- keywords: bugfix, loop\n"
        "- retrieval_outcome: useful\n- intent: resolved\n"
        "</memory>"
    )
    result = parse_memory_block(text)
    assert result.entries[0] == {"type": "correction", "topic": "bug-fix", "content": "Off-by-one in loop"}
    assert result.complete is True
    assert result.complete_explicit is True
    assert result.context == "sufficient"
    assert result.context_explicit is True
    assert result.keywords == ["bugfix", "loop"]
    assert result.retrieval_outcome == "useful"
    assert result.intent == "resolved"
    assert result.is_compact is False


#TAG: [75F8] 2026-04-05
# Verifies: verbose block with no complete/context fields has explicit flags as False
@pytest.mark.edge
def test_parse_memory_block_verbose_missing_fields():
    text = "<memory>\n- type: fact\n- topic: test\n- content: data\n</memory>"
    result = parse_memory_block(text)
    assert result.complete is None
    assert result.complete_explicit is False
    assert result.context_explicit is False
    assert result.keywords_explicit is False


#TAG: [433A] 2026-04-05
# Verifies: verbose block with "complete: false" and remaining text is parsed correctly
@pytest.mark.error
def test_parse_memory_block_verbose_incomplete_remaining():
    text = (
        "<memory>\n"
        "- type: project\n- topic: migration\n- content: started\n"
        "- complete: false\n- remaining: need to run tests\n"
        "- context: sufficient\n- keywords: migration\n"
        "</memory>"
    )
    result = parse_memory_block(text)
    assert result.complete is False
    assert result.remaining == "need to run tests"


#TAG: [F58A] 2026-04-05
# Verifies: verbose block with bare "- " line separators and no dashes on key lines still parses
@pytest.mark.adversarial
def test_parse_memory_block_verbose_mixed_dash():
    text = (
        "<memory>\n"
        "type: fact\n"
        "topic: nodash\n"
        "content: works without dashes\n"
        "complete: true\n"
        "context: sufficient\n"
        "keywords: test\n"
        "</memory>"
    )
    result = parse_memory_block(text)
    assert len(result.entries) == 1
    assert result.entries[0]["content"] == "works without dashes"
    assert result.complete is True


# ---------------------------------------------------------------------------
# 5. Verbose multi-entry and partial entry handling
# ---------------------------------------------------------------------------

#TAG: [EE1C] 2026-04-05
# Verifies: two consecutive verbose entries with repeated "type" key triggers entry boundary
@pytest.mark.behavioural
def test_parse_memory_block_verbose_multi_entry():
    text = (
        "<memory>\n"
        "- type: fact\n- topic: first\n- content: entry one\n"
        "- type: decision\n- topic: second\n- content: entry two\n"
        "- complete: true\n- context: sufficient\n- keywords: multi\n"
        "</memory>"
    )
    result = parse_memory_block(text)
    assert len(result.entries) == 2
    assert result.entries[0]["type"] == "fact"
    assert result.entries[0]["content"] == "entry one"
    assert result.entries[1]["type"] == "decision"
    assert result.entries[1]["content"] == "entry two"


#TAG: [BCB5] 2026-04-05
# Verifies: partial entry (type+topic, no content) defaults content to topic value
@pytest.mark.edge
def test_parse_memory_block_verbose_partial_defaults():
    text = (
        "<memory>\n"
        "- type: fact\n- topic: orphan-topic\n"
        "- complete: true\n- context: sufficient\n- keywords: partial\n"
        "</memory>"
    )
    result = parse_memory_block(text)
    assert len(result.entries) == 1
    assert result.entries[0]["content"] == "orphan-topic"


#TAG: [C93F] 2026-04-05
# Verifies: entry with only type (no topic) is not added to entries
@pytest.mark.error
def test_parse_memory_block_verbose_type_only():
    text = (
        "<memory>\n"
        "- type: fact\n"
        "- complete: true\n- context: sufficient\n- keywords: test\n"
        "</memory>"
    )
    result = parse_memory_block(text)
    assert len(result.entries) == 0


#TAG: [CC8A] 2026-04-05
# Verifies: confidence_update in verbose format parses id, direction, and optional reason
@pytest.mark.adversarial
def test_parse_memory_block_verbose_confidence_updates():
    text = (
        "<memory>\n"
        "- type: fact\n- topic: cu\n- content: testing updates\n"
        "- confidence_update: 42:+\n"
        "- confidence_update: 17:-\n"
        "- confidence_update: 99:-! superseded by new approach\n"
        "- complete: true\n- context: sufficient\n- keywords: confidence\n"
        "</memory>"
    )
    result = parse_memory_block(text)
    assert len(result.confidence_updates) == 3
    assert result.confidence_updates[0] == (42, "+", None)
    assert result.confidence_updates[1] == (17, "-", None)
    assert result.confidence_updates[2] == (99, "-!", "superseded by new approach")


# ---------------------------------------------------------------------------
# Sentinel constants (lightweight)
# ---------------------------------------------------------------------------

#TAG: [FAC9] 2026-04-05
# Verifies: NO_BLOCK sentinel has entries=None, complete=None, and all explicit flags False
@pytest.mark.behavioural
def test_parse_memory_block_no_block_sentinel():
    assert NO_BLOCK.entries is None
    assert NO_BLOCK.complete is None
    assert NO_BLOCK.remaining is None
    assert NO_BLOCK.context is None
    assert NO_BLOCK.complete_explicit is False
    assert NO_BLOCK.context_explicit is False
    assert NO_BLOCK.is_compact is False


#TAG: [1B1C] 2026-04-05
# Verifies: NOOP_BLOCK sentinel has empty entries list, complete=True, context=sufficient
@pytest.mark.behavioural
def test_parse_memory_block_noop_sentinel():
    assert NOOP_BLOCK.entries == []
    assert NOOP_BLOCK.complete is True
    assert NOOP_BLOCK.context == "sufficient"
    assert NOOP_BLOCK.complete_explicit is True
    assert NOOP_BLOCK.context_explicit is True
    assert NOOP_BLOCK.is_compact is False
