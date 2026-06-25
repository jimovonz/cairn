"""Cache-safe request rewriting for the cairn proxy.

Three pure transforms over the Anthropic request body (a parsed ``dict``):

* :func:`reinject_cm` — append each captured verbatim ``[cm]`` block back onto
  the assistant turn that generated it, so the wire bytes match what the model
  produced (maximising any prompt-cache coverage that extends into the
  assistant turns). Matched by SHA-256 of the stripped assistant text.
* :func:`inject_bootstrap` — add Cairn's standing bootstrap as a ``system``
  block and relocate the trailing ``cache_control`` breakpoint onto it. The
  bootstrap is byte-identical per session, so turn ≥2 gets ``cache_read`` over
  ``system + bootstrap`` — no prefix invalidation.
* :func:`inject_prompt_context` — append the volatile per-prompt retrieval as a
  text block on the **last user message**, i.e. *after* every cache breakpoint,
  so it never invalidates the cached prefix.

All transforms mutate and return ``data``; each is idempotent (re-applying the
same payload does not duplicate it), which keeps the per-turn wire bytes stable.
"""

from __future__ import annotations

import hashlib


def _as_block_list(value):
    """Normalise a string-or-list system/content field to a block list."""
    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    if isinstance(value, list):
        return value
    return []


def _assistant_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def reinject_cm(data: dict, sha_to_cm: dict) -> dict:
    """Append verbatim [cm] to each assistant turn keyed by stripped-text SHA."""
    if not sha_to_cm:
        return data
    for msg in data.get("messages", []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        text = _assistant_text(content)
        if not text:
            continue
        cm = sha_to_cm.get(hashlib.sha256(text.encode("utf-8")).hexdigest())
        if not cm or text.endswith(cm):
            continue
        if isinstance(content, str):
            msg["content"] = content + cm
        elif isinstance(content, list):
            # append to the last text block, else add one
            for b in reversed(content):
                if isinstance(b, dict) and b.get("type") == "text":
                    b["text"] = b.get("text", "") + cm
                    break
            else:
                content.append({"type": "text", "text": cm})
    return data


_BOOTSTRAP_SENTINEL = "<!--cairn-bootstrap-->"


def inject_bootstrap(data: dict, bootstrap_text: str, move_breakpoint: bool = True) -> dict:
    """Add bootstrap as a system block, moving the trailing cache breakpoint onto it.

    ``move_breakpoint`` (default True) relocates the last ``cache_control`` onto
    the bootstrap so the cached prefix grows to include it — correct only while
    the bootstrap is byte-stable across turns. When the caller has detected the
    bootstrap changed mid-session, it passes ``move_breakpoint=False`` so the
    bootstrap is appended *after* the existing breakpoint: uncached (re-billed
    each turn) but it can no longer invalidate the stable upstream prefix.
    """
    if not bootstrap_text:
        return data
    blocks = _as_block_list(data.get("system", []))
    payload = _BOOTSTRAP_SENTINEL + "\n" + bootstrap_text
    if any(isinstance(b, dict) and b.get("text", "").startswith(_BOOTSTRAP_SENTINEL)
           for b in blocks):
        return data  # already injected this turn
    new_block = {"type": "text", "text": payload}
    if move_breakpoint:
        # Relocate the last cache_control breakpoint (the prefix boundary) onto
        # the new block so the cached prefix grows to include the stable bootstrap.
        for b in reversed(blocks):
            if isinstance(b, dict) and "cache_control" in b:
                new_block["cache_control"] = b.pop("cache_control")
                break
    blocks.append(new_block)
    data["system"] = blocks
    return data


_CTX_SENTINEL = "<!--cairn-context-->"


def inject_prompt_context(data: dict, context_text: str) -> dict:
    """Append volatile per-prompt context to the last user message (post-cache)."""
    if not context_text:
        return data
    messages = data.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            payload = _CTX_SENTINEL + "\n" + context_text
            content = msg.get("content")
            if isinstance(content, str):
                if _CTX_SENTINEL in content:
                    return data
                msg["content"] = content + "\n\n" + payload
            elif isinstance(content, list):
                if any(isinstance(b, dict) and _CTX_SENTINEL in b.get("text", "")
                       for b in content):
                    return data
                content.append({"type": "text", "text": payload})
            return data
    return data


def sanitize_empty_text_blocks(data: dict) -> dict:
    """Drop empty/whitespace-only text content blocks from messages.

    The Anthropic API rejects a request with HTTP 400 "messages: text content
    blocks must be non-empty" if any message carries a ``{"type":"text","text":""}``
    block. Claude Code can hold such a block in its in-memory conversation (e.g. an
    assistant turn that streamed thinking + an empty text + a tool_use), and it is
    reconstructed into every subsequent request — wedging the session in a 400 loop
    that editing the on-disk transcript cannot clear (the live process never re-reads
    it). As the request-rewrite layer we strip these blocks unconditionally so the
    session self-heals on its next turn. Fail-open: never raise.

    If filtering would empty a message entirely, a single-space placeholder is left
    so the message stays valid (space is non-empty, so the API accepts it) and any
    tool_use/tool_result pairing across messages is preserved.
    """
    try:
        for msg in data.get("messages", []):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            kept = [
                b for b in content
                if not (isinstance(b, dict) and b.get("type") == "text"
                        and not b.get("text", "").strip())
            ]
            if len(kept) == len(content):
                continue
            msg["content"] = kept if kept else [{"type": "text", "text": " "}]
    except Exception:
        pass
    return data
