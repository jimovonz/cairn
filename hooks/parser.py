"""Memory block parser for Cairn stop hook.

Supports three memory block formats:
1. Verbose: <memory>- type: fact\n- topic: ...\n- complete: true\n...</memory>
2. Compact: <memory>fact/topic: content [k: kw1]\n+ c h:52</memory>
3. Link-definition (JSON): [cairn-memory]: # '{"entries":[...],"complete":true,...}'
   Invisible in Copilot chat panel (markdown link defs produce no rendered output).
   Uses single-quoted title with greedy match to last quote on line.
"""

from __future__ import annotations

import json as _json
import re
from typing import NamedTuple, Optional

from hooks.hook_helpers import log


class ParseResult(NamedTuple):
    """Structured result from parsing a <memory> block."""
    entries: Optional[list[dict[str, str]]]
    complete: Optional[bool]
    remaining: Optional[str]
    context: Optional[str]
    context_need: Optional[str]
    confidence_updates: list[tuple[int, str, Optional[str]]]  # (id, direction, reason)
    retrieval_outcome: Optional[str]
    keywords: list[str]
    intent: Optional[str]
    complete_explicit: bool = False   # True if 'complete' was explicitly declared
    context_explicit: bool = False    # True if 'context' was explicitly declared
    keywords_explicit: bool = False   # True if 'keywords' was explicitly declared
    hash_claimed: Optional[int] = None  # Claimed response hash (h:NNN)
    is_compact: bool = False         # True if compact format was used


# Sentinel for "no memory block found"
NO_BLOCK = ParseResult(None, None, None, None, None, [], None, [], None, False, False, False, None, False)
NOOP_BLOCK = ParseResult([], True, None, "sufficient", None, [], None, [], None, True, True, False, None, False)


def _g(d: dict, *keys, default=None):
    """Get first matching key from dict — supports short/long key aliases."""
    for k in keys:
        if k in d:
            return d[k]
    return default


# Context value shortcuts: "s" → "sufficient", "i" → "insufficient"
_CTX_EXPAND = {"s": "sufficient", "i": "insufficient"}


def _parse_linkdef(text: str) -> Optional[ParseResult]:
    """Try to parse a link-definition JSON memory block from text.

    Format: [cm]: # '{"e":[{"t":"fact","to":"...","c":"..."}],"ok":true,"ctx":"s","kw":["k1"]}'

    Accepts both short and long JSON keys for token economy:
        e/entries, t/type, to/topic, c/content, kw/keywords,
        ok/complete, ctx/context, cn/context_need, rem/remaining,
        cu/confidence_updates, ro/retrieval_outcome, int/intent, tr/trigger, d/depth

    Uses greedy match to the last single-quote on the line so that JSON
    content containing literal single quotes (e.g. "it's") is handled
    without escaping. Returns None if no link-definition block found.

    Accepts [cairn-memory] or [cm] as the link label.
    """
    # Scan for the link definition line(s) — take the last one if multiple
    matches = re.findall(
        r"^\[(?:cairn-memory|cm)\]:\s*#\s*'(.+)'\s*$",
        text, re.MULTILINE
    )
    if not matches:
        return None

    raw_json = matches[-1]
    try:
        data = _json.loads(raw_json)
    except (_json.JSONDecodeError, ValueError) as e:
        log(f"Link-def JSON parse error: {e} — raw: {raw_json[:120]}")
        return None

    if not isinstance(data, dict):
        log(f"Link-def JSON is not a dict: {type(data)}")
        return None

    # Extract entries (e or entries)
    raw_entries = _g(data, "e", "entries", default=[])
    entries: list[dict[str, str]] = []
    if isinstance(raw_entries, list):
        for e in raw_entries:
            if not isinstance(e, dict):
                continue
            e_type = _g(e, "t", "type")
            e_topic = _g(e, "to", "topic")
            if not e_type or not e_topic:
                continue
            entry = {
                "type": str(e_type),
                "topic": str(e_topic),
                "content": str(_g(e, "c", "content", default=e_topic)),
            }
            kw = _g(e, "kw", "keywords")
            if kw is not None:
                entry["keywords"] = kw if isinstance(kw, list) else [k.strip() for k in str(kw).split(",")]
            trigger = _g(e, "tr", "trigger")
            if trigger:
                entry["trigger"] = str(trigger)
            depth = _g(e, "d", "depth")
            if depth is not None:
                try:
                    entry["depth"] = int(depth)
                except (ValueError, TypeError):
                    pass
            entries.append(entry)

    # Extract metadata (short/long keys)
    complete = _g(data, "ok", "complete", default=True)
    if isinstance(complete, str):
        complete = complete.lower() == "true"
    remaining = _g(data, "rem", "remaining")
    raw_ctx = str(_g(data, "ctx", "context", default="sufficient")).lower()
    context = _CTX_EXPAND.get(raw_ctx, raw_ctx)
    context_need = _g(data, "cn", "context_need")
    retrieval_outcome = _g(data, "ro", "retrieval_outcome")
    if retrieval_outcome:
        retrieval_outcome = str(retrieval_outcome).lower()
    intent = _g(data, "int", "intent")
    if intent:
        intent = str(intent).lower()

    # Keywords — block-level (kw or keywords)
    raw_kw = _g(data, "kw", "keywords", default=[])
    if isinstance(raw_kw, list):
        keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
    elif isinstance(raw_kw, str):
        keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
    else:
        keywords = []

    # Attach block-level keywords to entries that don't have their own
    if keywords:
        for entry in entries:
            if "keywords" not in entry:
                entry["keywords"] = keywords

    # Confidence updates — accept "42:+" or "17:-! reason" strings (cu or confidence_updates)
    confidence_updates: list[tuple[int, str, Optional[str]]] = []
    for cu in _g(data, "cu", "confidence_updates", default=[]):
        if isinstance(cu, str):
            m = re.match(r"(\d+)\s*:\s*(-!|[+-])\s*(.*)?$", cu)
            if m:
                reason = m.group(3).strip() if m.group(3) else None
                confidence_updates.append((int(m.group(1)), m.group(2), reason))

    return ParseResult(
        entries=entries,
        complete=complete,
        remaining=remaining,
        context=context,
        context_need=context_need,
        confidence_updates=confidence_updates,
        retrieval_outcome=retrieval_outcome,
        keywords=keywords,
        intent=intent,
        complete_explicit=True,
        context_explicit=True,
        keywords_explicit=bool(keywords),
        hash_claimed=None,
        is_compact=False,
    )


def parse_memory_block(text: str) -> ParseResult:
    """Extract memory entries, completeness, and context needs from a memory block.

    Tries formats in order:
    1. Link-definition JSON ([cairn-memory]: # '{...}') — invisible in Copilot
    2. <memory> block (verbose or compact) — the original format
    """
    # Try link-definition format first (invisible in Copilot chat panel)
    linkdef_result = _parse_linkdef(text)
    if linkdef_result is not None:
        return linkdef_result

    # Fall back to <memory> block parsing
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

    # Detect compact format: contains type/topic: entry, or starts with "." no-op,
    # or has standalone h:NNN line, or compact control line (+ c or - c with h:)
    is_compact = (bool(re.search(r'^\w+/[^:]+:', block, re.MULTILINE))
                  or block.startswith(".")
                  or bool(re.search(r'^h:[0-9A-Fa-f]+$', block, re.MULTILINE))
                  or bool(re.search(r'^[+-] c[? ]', block, re.MULTILINE))
                  or bool(re.search(r'^[+-] c$', block, re.MULTILINE)))

    if is_compact:
        return _parse_compact(block)
    else:
        return _parse_verbose(block)


def _parse_compact(block: str) -> ParseResult:
    """Parse compact memory format.

    Compact format examples:
        fact/topic: content [k: kw1, kw2]
        + c h:52

        .
        h:24

        fact/topic: content [k: kw1]
        - :still need to finish
        c?:what was decided about X
        h:46
    """
    entries: list[dict[str, str]] = []
    complete: Optional[bool] = True  # Default true in compact format
    complete_set: bool = True
    remaining: Optional[str] = None
    context: str = "sufficient"
    context_set: bool = True
    context_need: Optional[str] = None
    retrieval_outcome: Optional[str] = None
    keywords: list[str] = []
    keywords_set: bool = False
    confidence_updates: list[tuple[int, str, Optional[str]]] = []
    intent: Optional[str] = None
    hash_claimed: Optional[int] = None
    noop: bool = False

    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue

        # No-op line: "." or ". h:NNN"
        if line.startswith("."):
            noop = True
            hm = re.search(r'h:([0-9A-Fa-f]+)', line)
            if hm:
                hash_claimed = int(hm.group(1), 16)
            continue

        # Compact entry: type/topic: content [k: keywords]
        entry_match = re.match(r'^(\w+)/([^:]+):\s*(.+)$', line)
        if entry_match:
            entry_type = entry_match.group(1)
            topic = entry_match.group(2).strip()
            content = entry_match.group(3).strip()
            # Extract [k: ...] keywords suffix
            entry_keywords: list[str] = []
            km = re.search(r'\[k:\s*([^\]]+)\]', content)
            if km:
                entry_keywords = [k.strip() for k in km.group(1).split(",") if k.strip()]
                keywords = entry_keywords  # block-level for L2 search
                keywords_set = True
                content = re.sub(r'\s*\[k:[^\]]+\]', '', content).strip()
            # Extract [t: ...] trigger suffix (corrections only)
            entry_trigger: str = ""
            tm = re.search(r'\[t:\s*([^\]]+)\]', content)
            if tm:
                entry_trigger = tm.group(1).strip()
                content = re.sub(r'\s*\[t:[^\]]+\]', '', content).strip()
            entry_dict = {"type": entry_type, "topic": topic, "content": content, "keywords": entry_keywords}
            if entry_trigger:
                entry_dict["trigger"] = entry_trigger
            entries.append(entry_dict)
            continue

        # Standalone incomplete: - :remaining text or -:remaining text
        # Must check before control line since both start with -
        inc_match = re.match(r'^-\s*:\s*(.+)$', line)
        if inc_match:
            complete = False
            complete_set = True
            remaining = inc_match.group(1).strip()
            continue

        # Control line: + c h:NNN or variants
        # Complete flag: + or -
        ctrl_match = re.match(r'^([+-])\s+', line)
        if ctrl_match:
            complete = ctrl_match.group(1) == "+"
            complete_set = True
            # Context: c or c?
            if 'c?' in line:
                context = "insufficient"
                cn = re.search(r'c\?:(\S.*?)(?=\s+h:|\s*$)', line)
                if cn:
                    context_need = cn.group(1).strip()
            # Hash
            hm = re.search(r'h:([0-9A-Fa-f]+)', line)
            if hm:
                hash_claimed = int(hm.group(1), 16)
            continue

        # Standalone context need: c?:text
        if line.startswith("c?:"):
            context = "insufficient"
            context_set = True
            context_need = line[3:].strip()
            continue

        # Standalone hash: h:NNN
        hm = re.match(r'^h:([0-9A-Fa-f]+)$', line)
        if hm:
            hash_claimed = int(hm.group(1), 16)
            continue

        # Confidence updates: same format as verbose
        conf_match = re.match(r"^-?\s*confidence_update:\s*(\d+)\s*:\s*(-!|[+-])\s*(.*)?$", line)
        if conf_match:
            direction = conf_match.group(2)
            reason = conf_match.group(3).strip() if conf_match.group(3) else None
            confidence_updates.append((int(conf_match.group(1)), direction, reason))
            continue

        # Retrieval outcome
        ro_match = re.match(r'^retrieval_outcome:\s*(\w+)', line)
        if ro_match:
            retrieval_outcome = ro_match.group(1).lower()
            continue

        # Intent
        intent_match = re.match(r'^intent:\s*(\w+)', line)
        if intent_match:
            intent = intent_match.group(1).lower()
            continue

    return ParseResult(entries, complete, remaining, context, context_need,
                       confidence_updates, retrieval_outcome, keywords, intent,
                       complete_set, context_set, keywords_set, hash_claimed, True)


def _parse_verbose(block: str) -> ParseResult:
    """Parse verbose (current/legacy) memory format."""
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    complete: Optional[bool] = None
    complete_set: bool = False
    remaining: Optional[str] = None
    context: str = "sufficient"
    context_set: bool = False
    context_need: Optional[str] = None
    retrieval_outcome: Optional[str] = None
    keywords: list[str] = []
    keywords_set: bool = False
    confidence_updates: list[tuple[int, str, Optional[str]]] = []
    intent: Optional[str] = None
    hash_claimed: Optional[int] = None

    for line in block.split("\n"):
        line = line.strip()
        if not line or line == "-":
            continue

        # Parse confidence_update: <id>:+ or <id>:- or <id>:-! reason
        conf_match = re.match(r"^-?\s*confidence_update:\s*(\d+)\s*:\s*(-!|[+-])\s*(.*)?$", line)
        if conf_match:
            direction = conf_match.group(2)
            reason = conf_match.group(3).strip() if conf_match.group(3) else None
            confidence_updates.append((int(conf_match.group(1)), direction, reason))
            continue

        # Parse any "- key: value" or "key: value" pattern
        match = re.match(r"^-?\s*(\w+):\s*(.+)$", line)
        if not match:
            continue

        key, value = match.group(1).strip(), match.group(2).strip()

        if key == "complete":
            complete = value.lower() == "true"
            complete_set = True
        elif key == "remaining":
            remaining = value
        elif key == "context":
            context = value.lower()
            context_set = True
        elif key == "context_need":
            context_need = value
        elif key == "retrieval_outcome":
            retrieval_outcome = value.lower()
        elif key == "keywords":
            keywords = [k.strip() for k in value.split(",") if k.strip()]
            keywords_set = True
        elif key == "intent":
            intent = value.lower()
        elif key == "trigger":
            current["trigger"] = value
        elif key == "depth":
            try:
                current["depth"] = int(value.strip())
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

    # Attach block-level keywords to all entries that don't have their own
    if keywords:
        for entry in entries:
            if "keywords" not in entry:
                entry["keywords"] = keywords

    return ParseResult(entries, complete, remaining, context, context_need,
                       confidence_updates, retrieval_outcome, keywords, intent,
                       complete_set, context_set, keywords_set, hash_claimed, False)


def parse_memory_notes(text: str) -> list[dict[str, str]]:
    """Extract memory notes from text.

    Accepts two formats:
    1. <memory_note>type/topic: content</memory_note>  (original, visible in Copilot)
    2. [cairn-note]: # '{"type":"...","topic":"...","content":"..."}'  (invisible in Copilot)

    Returns a list of dicts with type, topic, content keys.
    Silently skips malformed notes.
    """
    notes: list[dict[str, str]] = []

    # Format 1: <memory_note> tags
    for match in re.finditer(r"<memory_note>(.*?)</memory_note>", text, re.DOTALL):
        body = match.group(1).strip()
        entry_match = re.match(r"^(\w+)/([^:]+):\s*(.+)$", body, re.DOTALL)
        if entry_match:
            notes.append({
                "type": entry_match.group(1).strip(),
                "topic": entry_match.group(2).strip(),
                "content": entry_match.group(3).strip(),
            })
        else:
            log(f"Skipping malformed memory_note: {body[:60]}")

    # Format 2: [cairn-note] link definitions (invisible in Copilot)
    for match in re.finditer(r"^\[(?:cairn-note|cn)\]:\s*#\s*'(.+)'\s*$", text, re.MULTILINE):
        raw = match.group(1)
        try:
            data = _json.loads(raw)
            if isinstance(data, dict) and "type" in data and "topic" in data:
                notes.append({
                    "type": str(data["type"]),
                    "topic": str(data["topic"]),
                    "content": str(data.get("content", data.get("topic", ""))),
                })
        except (_json.JSONDecodeError, ValueError):
            log(f"Skipping malformed cairn-note JSON: {raw[:60]}")

    return notes
