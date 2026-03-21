"""Memory block parser for Cairn stop hook."""

import re
from hook_helpers import log


def parse_memory_block(text):
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
            return None, None, None, None, None, [], None, [], None

    block = matches[-1].strip()

    # Check for no-op block
    if block in ("complete: true", "- complete: true"):
        return [], True, None, "sufficient", None, [], None, [], None

    # Parse entries
    entries = []
    current = {}
    complete = True
    remaining = None
    context = "sufficient"
    context_need = None
    retrieval_outcome = None  # useful | neutral | harmful
    keywords = []  # topic keywords for Layer 2 cross-project search
    confidence_updates = []  # list of (memory_id, direction)
    intent = None  # "resolved" when LLM confirms no further action needed

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
            # Parse "12-18" or "5" into start, end
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
            # If starting a new entry (type seen) and current entry is complete, commit it
            if key == "type" and "type" in current and "topic" in current and "content" in current:
                entries.append(current.copy())
                current = {}
            current[key] = value
        # Unknown fields are silently ignored

    # Handle partial entry (has type+topic but missing content)
    if current and "type" in current and "topic" in current:
        current.setdefault("content", current.get("topic", ""))
        entries.append(current.copy())

    return entries, complete, remaining, context, context_need, confidence_updates, retrieval_outcome, keywords, intent
