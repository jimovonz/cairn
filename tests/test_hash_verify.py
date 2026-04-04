"""Tests for hooks/hash_verify.py — response hash computation and verification."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hooks.hash_verify import compute_response_hash, verify_hash


# ═══════════════════════════════════════════════════════════════
# compute_response_hash — core 4 categories
# ═══════════════════════════════════════════════════════════════

#TAG: [5E23] 2026-04-05
# Verifies: basic single-sentence hash equals first letter value (h=8)
@pytest.mark.behavioural
def test_compute_response_hash_single_sentence():
    result = compute_response_hash("Hello world.")
    assert result == 8


#TAG: [5D8C] 2026-04-05
# Verifies: empty string returns 0
@pytest.mark.edge
def test_compute_response_hash_empty():
    assert compute_response_hash("") == 0


#TAG: [5996] 2026-04-05
# Verifies: text that is only a memory block returns 0 after stripping
@pytest.mark.error
def test_compute_response_hash_only_memory_block():
    text = "<memory>Some memory content here.</memory>"
    assert compute_response_hash(text) == 0


#TAG: [440A] 2026-04-05
# Verifies: nested memory tags — non-greedy regex strips first match, rest leaks through with computable hash
@pytest.mark.adversarial
def test_compute_response_hash_nested_memory_tags():
    text = "Alpha. <memory>Beta. <memory>Gamma.</memory> Delta.</memory> Echo."
    result = compute_response_hash(text)
    # Non-greedy strips <memory>Beta. <memory>Gamma.</memory> first
    # Leaves: "Alpha.  Delta.</memory> Echo."
    # Sentence split on (?<=[.!?])\s+(?=[A-Z]): ["Alpha.", "Delta.</memory> Echo."]
    # a=1, d=4 => 5
    assert result == 5


# ═══════════════════════════════════════════════════════════════
# compute_response_hash — additional behavioural tests
# ═══════════════════════════════════════════════════════════════

#TAG: [827B] 2026-04-05
# Verifies: memory block stripped, surrounding text on both sides contributes to hash
@pytest.mark.behavioural
def test_compute_response_hash_strips_memory():
    text = "Hello world. <memory>Ignore this.</memory> Zulu time."
    result = compute_response_hash(text)
    # h=8, z=26 => 34
    assert result == 34


#TAG: [8204] 2026-04-05
# Verifies: fenced code blocks replaced with sentence boundary, content inside not hashed
@pytest.mark.behavioural
def test_compute_response_hash_strips_code_blocks():
    text = "Alpha.\n```python\nBeta code here\n```\nCharlie."
    result = compute_response_hash(text)
    # Code block -> ". " => "Alpha.\n. \nCharlie."
    # Split finds ". " boundary before "Charlie"
    # a=1, c=3 => 4
    assert result == 4


#TAG: [C464] 2026-04-05
# Verifies: double-newline paragraph breaks become sentence boundaries
@pytest.mark.behavioural
def test_compute_response_hash_paragraph_breaks():
    text = "Alpha sentence\n\nBeta sentence"
    result = compute_response_hash(text)
    # \n\n -> ". " => "Alpha sentence. Beta sentence"
    # a=1, b=2 => 3
    assert result == 3


#TAG: [7563] 2026-04-05
# Verifies: all three punctuation types (. ! ?) act as sentence boundaries
@pytest.mark.behavioural
def test_compute_response_hash_multiple_punctuation_types():
    text = "Alpha! Beta? Charlie."
    result = compute_response_hash(text)
    # a=1, b=2, c=3 => 6
    assert result == 6


#TAG: [E433] 2026-04-05
# Verifies: non-alpha first character contributes 0, only alpha chars add to sum
@pytest.mark.behavioural
def test_compute_response_hash_non_alpha_start():
    text = "123 numeric start. Alpha."
    result = compute_response_hash(text)
    # '1' not alpha -> 0, a=1 => 1
    assert result == 1


#TAG: [2089] 2026-04-05
# Verifies: uppercase Z produces same value as lowercase z (z=26)
@pytest.mark.behavioural
def test_compute_response_hash_case_insensitive():
    result = compute_response_hash("Zoo animals.")
    assert result == 26


# ═══════════════════════════════════════════════════════════════
# verify_hash — core 4 categories
# ═══════════════════════════════════════════════════════════════

#TAG: [1C36] 2026-04-05
# Verifies: matching claimed hash returns (True, actual) with correct actual value
@pytest.mark.behavioural
def test_verify_hash_match():
    text = "Hello world."
    match, actual = verify_hash(text, 8)
    assert match is True
    assert actual == 8


#TAG: [AA3D] 2026-04-05
# Verifies: empty text with claimed=0 returns (True, 0) — boundary match
@pytest.mark.edge
def test_verify_hash_empty_text():
    match, actual = verify_hash("", 0)
    assert match is True
    assert actual == 0


#TAG: [7C52] 2026-04-05
# Verifies: incorrect claimed value returns (False, correct_actual) so caller can see difference
@pytest.mark.error
def test_verify_hash_mismatch():
    text = "Hello world."
    match, actual = verify_hash(text, 999)
    assert match is False
    assert actual == 8


#TAG: [98C5] 2026-04-05
# Verifies: negative claimed hash never matches since computed hash is always >= 0
@pytest.mark.adversarial
def test_verify_hash_negative_claimed():
    text = "Alpha."
    match, actual = verify_hash(text, -1)
    assert match is False
    assert actual == 1


#TAG: [C73B] 2026-04-05
# Verifies: multi-sentence verify returns correct actual on mismatch for debugging
@pytest.mark.behavioural
def test_verify_hash_multi_sentence_mismatch():
    text = "Alpha! Beta? Charlie."
    match, actual = verify_hash(text, 5)
    assert match is False
    assert actual == 6
