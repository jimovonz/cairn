"""Canaries for undocumented Claude Code internals Cairn depends on (spec 1.6).

Cairn reads a transcript JSONL format and a hook payload shape that Anthropic
can change without notice. That dependency cannot be removed, but its failure
mode can: without these, upstream drift degrades capture silently — memories
simply stop being extracted — and the resulting gap is indistinguishable from a
quiet period. These convert that into a named failure at a known place.

Ordered before the Stage 2 measurement work deliberately: drift during a
measurement campaign would silently invalidate every number it collects.
"""
import glob
import json
import os

import pytest

import cairn.session_extract as se

TRANSCRIPT_ROOT = os.path.expanduser("~/.claude/projects")

# Fields load_turns() reads. Changing this set is a deliberate act — it means
# Cairn's contract with the transcript format has moved.
REQUIRED_ENTRY_FIELDS = {"type", "message"}
RECOGNISED_ROLE_TYPES = {"user", "human", "assistant"}


def _newest_transcripts(n=3):
    paths = glob.glob(os.path.join(TRANSCRIPT_ROOT, "*", "*.jsonl"))
    return sorted(paths, key=os.path.getmtime, reverse=True)[:n]


requires_transcripts = pytest.mark.skipif(
    not _newest_transcripts(),
    reason="no Claude Code transcripts on this machine — canary is a no-op in CI",
)


@requires_transcripts
def test_transcript_lines_are_json_objects():
    for path in _newest_transcripts():
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip() or i > 200:
                    break
                obj = json.loads(line)
                assert isinstance(obj, dict), f"{path}:{i} is not a JSON object"


@requires_transcripts
def test_conversational_entries_still_carry_type_and_message():
    """The two fields load_turns() keys on. If either disappears or is renamed,
    capture silently yields zero turns."""
    seen_roles = set()
    for path in _newest_transcripts():
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip() or i > 400:
                    break
                obj = json.loads(line)
                etype = obj.get("type")
                if etype in RECOGNISED_ROLE_TYPES:
                    missing = REQUIRED_ENTRY_FIELDS - set(obj)
                    assert not missing, f"{path}:{i} missing {missing}"
                    seen_roles.add(etype)
    assert seen_roles, (
        "no user/human/assistant entries found in recent transcripts — the "
        "conversational entry type has probably been renamed upstream"
    )


@requires_transcripts
def test_message_content_is_string_or_block_list():
    """_content_text() handles exactly these two shapes; a third would be
    silently rendered as empty text."""
    for path in _newest_transcripts():
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip() or i > 400:
                    break
                obj = json.loads(line)
                if obj.get("type") not in RECOGNISED_ROLE_TYPES:
                    continue
                content = obj.get("message", {}).get("content", "")
                assert isinstance(content, (str, list)), (
                    f"{path}:{i} message.content is {type(content).__name__}"
                )


@requires_transcripts
def test_extractor_still_produces_turns_from_a_real_transcript():
    """End-to-end canary. Every field assertion above can pass while the
    extractor still yields nothing, so assert the actual outcome Cairn needs."""
    for path in _newest_transcripts():
        turns = se.load_turns(path)
        if turns:
            assert any(t["role"] == "assistant" for t in turns)
            assert any(t["text"] for t in turns), "all turns extracted as empty text"
            return
    pytest.fail("no recent transcript yielded any turns — transcript layout may have changed")


def test_content_block_shape_is_still_dict_with_type():
    """Pinned against a golden sample rather than live data, so it fails even on
    a machine with no transcripts."""
    blocks = [{"type": "text", "text": "hello"},
              {"type": "tool_use", "name": "Bash", "input": {}},
              {"type": "tool_result", "content": "out"}]
    assert se._content_text(blocks).strip() == "hello"
    assert se._has_tool_blocks(blocks) is True


def test_plain_string_content_is_still_supported():
    assert se._content_text("plain text") == "plain text"
