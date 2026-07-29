"""Multi-line tolerance and accurate diagnosis for [cm] blocks.

A block that wrapped across lines is invalid JSON — strings cannot contain a
literal newline — even though its content is perfectly good. Rejecting it cost a
full inference turn, and the error named bad JSON, which sent several debugging
attempts chasing quoting bugs that did not exist. If a failure is mechanically
recoverable, recover it; if it is not, say what actually broke.
"""
from hooks.parser import (extract_linkdef_raw, linkdef_diagnosis,
                          parse_memory_block)

GOOD = ('[cm]: # \'{"e":[{"t":"fact","to":"topic","c":"content here"}],'
        '"ok":true,"ctx":"s","kw":["a","b"]}\'')


def test_single_line_block_needs_no_recovery():
    raw, recovery = extract_linkdef_raw(GOOD)
    assert recovery is None
    assert raw.startswith("{")


def test_wrapped_block_is_recovered():
    wrapped = GOOD.replace('"content here"', '"content\n    here"')
    raw, recovery = extract_linkdef_raw(wrapped)
    assert recovery == "multiline"
    result = parse_memory_block(wrapped)
    assert len(result.entries) == 1
    assert result.entries[0]["content"] == "content here"


def test_recovery_does_not_fuse_two_blocks_from_history():
    """The hazard: a greedy match across newlines can run from an EARLIER block
    in the conversation to this one's closing quote, silently merging them."""
    earlier = ('[cm]: # \'{"e":[{"t":"fact","to":"old","c":"old content"}],'
               '"ok":true,"ctx":"s","kw":["x"]}\'')
    later = GOOD.replace('"content here"', '"new\n    content"')
    text = f"{earlier}\n\nsome prose in between\n\n{later}"
    result = parse_memory_block(text)
    assert len(result.entries) == 1
    assert result.entries[0]["topic"] == "topic"
    assert result.entries[0]["content"] == "new content"


def test_no_block_yields_no_diagnosis():
    assert linkdef_diagnosis("just prose, no block at all") is None


def test_valid_block_yields_no_diagnosis():
    assert linkdef_diagnosis(GOOD) is None


def test_recoverable_block_yields_no_diagnosis():
    """Recovered means not a failure — it must not be reported as one."""
    wrapped = GOOD.replace('"content here"', '"content\n    here"')
    assert linkdef_diagnosis(wrapped) is None


def test_salvageable_inner_quote_yields_no_diagnosis():
    """Existing salvage handles unescaped inner quotes; diagnosis must agree
    with it rather than reporting a failure the parser already absorbed."""
    inner = GOOD.replace('"content here"', '"he said "hello" loudly"')
    assert linkdef_diagnosis(inner) is None


def test_unrecoverable_block_names_the_real_problem():
    broken = "[cm]: # 'this is not json at all"
    d = linkdef_diagnosis(broken)
    assert d is not None
    assert "single quotes" in d or "could not be located" in d


def test_marker_without_payload_is_diagnosed_not_silently_ignored():
    d = linkdef_diagnosis("[cm]: # ")
    assert d is not None
