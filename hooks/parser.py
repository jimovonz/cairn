"""Memory block parser for Cairn stop hook."""

from __future__ import annotations

import re
from typing import NamedTuple, Optional

from hook_helpers import log


class ParseResult(NamedTuple):
    """Structured result from parsing a <memory> block."""
    entries: Optional[list[dict[str, str]]]
    complete: Optional[bool]
    remaining: Optional[str]
    context: Optional[str]
    context_need: Optional[str]
    confidence_updates: list[tuple[int, str]]
    retrieval_outcome: Optional[str]
    keywords: list[str]
    intent: Optional[str]


# Sentinel for "no memory block found"
NO_BLOCK = ParseResult(None, None, None, None, None, [], None, [], None)
NOOP_BLOCK = ParseResult([], True, None, "sufficient", None, [], None, [], None)


def parse_memory_block(text: str) -> ParseResult:
    """Extract memory entries, completeness, and context needs from a <memory> block.

    Robust parser that handles:
    - Malformed/unclosed tags (tries to find content after last <memory>)
    - Unknown fields (ignored gracefully)
    - Unknown types (accepted as-is)
    - Extra whitespace, missing dashes, inconsistent formatting
    """
    # Try closed tags first, fall back to unclosed
    pattern = r"<memory>(.*?)</memory>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        # Try unclosed tag — grab everything after last <memory>
        unclosed = re.search(r"<memory>(.*?)$", text, re.DOTALL)
        if unclosed:
            matches = [unclosed.group(1)]
            log("Warning: unclosed <memory> tag — parsed anyway")
        else:
            return NO_BLOCK

    block = matches[-1].strip()

    # Check for no-op block
    if block in ("complete: true", "- complete: true"):
        return NOOP_BLOCK

    # Parse entries
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    complete: Optional[bool] = None
    remaining: Optional[str] = None
    context: str = "sufficient"
    context_need: Optional[str] = None
    retrieval_outcome: Optional[str] = None
    keywords: list[str] = []
    confidence_updates: list[tuple[int, str]] = []
    intent: Optional[str] = None

    for line in block.split("\n"):
        line = line.strip()
        if not line or line == "-":
            continue

        # Parse confidence_update: <id>:+ or <id>:-
        conf_match = re.match(r"^-?\s*confidence_update:\s*(\d+)\s*:\s*([+-])", line)
        if conf_match:
            confidence_updates.append((int(conf_match.group(1)), conf_match.group(2)))
            continue

        # Parse any "- key: value" or "key: value" pattern
        match = re.match(r"^-?\s*(\w+):\s*(.+)$", line)
        if not match:
            continue

        key, value = match.group(1).strip(), match.group(2).strip()

        if key == "complete":
            complete = value.lower() == "true"
        elif key == "remaining":
            remaining = value
        elif key == "context":
            context = value.lower()
        elif key == "context_need":
            context_need = value
        elif key == "retrieval_outcome":
            retrieval_outcome = value.lower()
        elif key == "keywords":
            keywords = [k.strip() for k in value.split(",") if k.strip()]
        elif key == "intent":
            intent = value.lower()
        elif key == "source_messages":
            try:
                if "-" in value:
                    parts = value.split("-")
                    current["source_start"] = int(parts[0].strip())
                    current["source_end"] = int(parts[1].strip())
                else:
                    current["source_start"] = int(value.strip())
                    current["source_end"] = int(value.strip())
            except (ValueError, IndexError):
                pass
        elif key in ("type", "topic", "content"):
            if key == "type" and "type" in current and "topic" in current and "content" in current:
                entries.append(current.copy())
                current = {}
            current[key] = value

    # Handle partial entry (has type+topic but missing content)
    if current and "type" in current and "topic" in current:
        current.setdefault("content", current.get("topic", ""))
        entries.append(current.copy())

    return ParseResult(entries, complete, remaining, context, context_need,
                       confidence_updates, retrieval_outcome, keywords, intent)
