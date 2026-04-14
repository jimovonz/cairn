"""Enforcement mechanisms — trailing intent, declined-without-trying, and continuation counting."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import hooks.hook_helpers as hook_helpers
from hooks.hook_helpers import log, get_conn
from cairn.config import TRAILING_INTENT_SIM_THRESHOLD

import numpy as np

TRAILING_INTENT_REFS: list[str] = [
    "let me test that now",
    "let me check that",
    "let me investigate",
    "let me run the tests",
    "let me look into this",
    "let me fix that",
    "I'll check this next",
    "I'll investigate that",
    "I'll run the tests now",
    "I'll look into it",
    "I'm going to test this",
    "I'm going to check",
    # Implementation/creation intent
    "let me implement that",
    "I'll add that now",
    "let me create that",
    "I'll build that next",
    # Review/analysis intent
    "let me review that",
    "I'll examine the code",
    "let me inspect that",
    "let me analyze this",
    # Debug/verify intent
    "let me debug this",
    "I'll verify that works",
    "let me try that",
    "I'll take a look",
    # Refactor/cleanup intent
    "let me refactor that",
    "I'll clean that up",
]

_intent_embeddings: Optional[list[tuple[str, np.ndarray]]] = None


def _get_intent_embeddings() -> Optional[list[tuple[str, np.ndarray]]]:
    """Lazy-load and cache reference intent embeddings."""
    global _intent_embeddings
    if _intent_embeddings is not None:
        return _intent_embeddings
    emb = hook_helpers.get_embedder()
    if not emb:
        return None
    _intent_embeddings = [(ref, emb.embed(ref)) for ref in TRAILING_INTENT_REFS]
    _intent_embeddings = [(ref, vec) for ref, vec in _intent_embeddings if vec is not None]
    return _intent_embeddings if _intent_embeddings else None


def _extract_last_sentence(text: str) -> Optional[str]:
    """Extract the last non-empty, non-memory-block sentence from the response."""
    cleaned = re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    sentences = re.split(r"[.!?\n]", cleaned)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return sentences[-1] if sentences else None


def check_trailing_intent(text: str, session_id: str = "") -> Optional[str]:
    """Check if response ends with unfulfilled action intent.

    Returns the matched trailing sentence if intent detected, None otherwise.
    Records metrics for all outcomes (detected, clear, skipped).
    """
    from hooks.hook_helpers import record_metric

    # Questions are not intent — check before extracting
    cleaned = re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL).strip()
    if cleaned.rstrip().endswith("?"):
        return None

    last = _extract_last_sentence(text)
    if not last:
        return None

    refs = _get_intent_embeddings()
    if not refs:
        log("Trailing intent: embedder unavailable — skipping check")
        record_metric(session_id, "trailing_intent_skipped", "embedder_unavailable")
        return None

    emb = hook_helpers.get_embedder()
    if not emb:
        log("Trailing intent: embedder unavailable for last sentence — skipping check")
        record_metric(session_id, "trailing_intent_skipped", "embedder_unavailable")
        return None

    last_vec = emb.embed(last)
    if last_vec is None:
        log("Trailing intent: embedding failed for last sentence — skipping check")
        record_metric(session_id, "trailing_intent_skipped", "embedding_failed")
        return None

    max_sim = 0.0
    for ref_text, ref_vec in refs:
        sim = emb.cosine_similarity(last_vec, ref_vec)
        if sim > max_sim:
            max_sim = sim

    if max_sim > TRAILING_INTENT_SIM_THRESHOLD:
        log(f"Trailing intent match: sim={max_sim:.3f} last='{last[:60]}'")
        record_metric(session_id, "trailing_intent_detected", f"sim={max_sim:.3f}", max_sim)
        return last[:100]

    record_metric(session_id, "trailing_intent_clear", f"sim={max_sim:.3f}", max_sim)
    return None


# --- Continuation counting (SQLite-backed) ---

def get_continuation_count(session_id: str) -> int:
    """Get how many times we've re-prompted this session."""
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'continuation_count'",
        (session_id,)
    ).fetchone()
    conn.close()
    return int(row[0]) if row and row[0] else 0


def increment_continuation(session_id: str) -> int:
    """Increment and return the continuation count."""
    conn = get_conn()
    current = get_continuation_count(session_id)
    new_count = current + 1
    conn.execute(
        "INSERT OR REPLACE INTO hook_state (session_id, key, value) VALUES (?, 'continuation_count', ?)",
        (session_id, str(new_count))
    )
    conn.commit()
    conn.close()
    return new_count


def reset_continuation(session_id: str) -> None:
    """Reset continuation count (called when a response completes normally)."""
    conn = get_conn()
    conn.execute(
        "DELETE FROM hook_state WHERE session_id = ? AND key = 'continuation_count'",
        (session_id,)
    )
    conn.commit()
    conn.close()


# --- Declined-without-trying detection ---

# Patterns that indicate the LLM is pushing work back to the user
_DECLINE_PATTERNS: list[re.Pattern] = [
    re.compile(r"you.ll need to .{3,40}yourself", re.IGNORECASE),
    re.compile(r"you.ll have to .{3,40}yourself", re.IGNORECASE),
    re.compile(r"you.?ll need to .{0,15}(run|do|try|execute|restart|start|launch)", re.IGNORECASE),
    re.compile(r"you.?d need to .{0,15}(run|do|try|execute|restart|start|contact)", re.IGNORECASE),
    re.compile(r"i can.t .{0,30}(from here|from this)", re.IGNORECASE),
    re.compile(r"i.m (not able|unable) to .{3,30}(from|in this)", re.IGNORECASE),
    re.compile(r"i don.t have (access|the ability|permission) to", re.IGNORECASE),
    re.compile(r"you should .{0,10}(run|do|try|execute|restart) .{3,30} yourself", re.IGNORECASE),
]

# Action tools that indicate the LLM actually tried something
_ACTION_TOOLS: set[str] = {"Bash", "Edit", "Write", "MultiEdit", "Agent"}


def _response_has_tool_calls(transcript_path: str) -> bool:
    """Check if the current (last) assistant turn includes action tool calls."""
    if not transcript_path or not os.path.exists(transcript_path):
        return False
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Scan backwards from end — find tool calls after the last user message
        found_assistant = False
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            # Stop at the last user message — everything after is current turn
            msg = entry.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "user":
                break
            tool_name = entry.get("tool_name", "")
            if tool_name in _ACTION_TOOLS:
                return True
            # Also check for tool_use in assistant message content blocks
            if entry.get("type") == "assistant" and isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("name", "") in _ACTION_TOOLS:
                            return True
        return found_assistant  # False — no action tools found
    except (IOError, OSError):
        return False


def check_declined_without_trying(text: str, transcript_path: str,
                                   session_id: str = "") -> Optional[str]:
    """Detect when the LLM declines to do something without attempting it first.

    Returns the matched decline phrase if detected, None otherwise.
    Only fires when: (1) response contains declining language, AND
    (2) transcript shows no action tool calls in current turn.

    Cost of false positive: one extra turn where LLM clarifies.
    Cost of false negative: user has to manually correct an imagined barrier.
    """
    from hooks.hook_helpers import record_metric

    # Strip memory block and code blocks before checking
    cleaned = re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`[^`]+`", "", cleaned)
    cleaned = cleaned.strip()

    # Check for decline patterns
    matched_phrase = None
    for pat in _DECLINE_PATTERNS:
        m = pat.search(cleaned)
        if m:
            matched_phrase = m.group()
            break

    if not matched_phrase:
        return None

    # Decline language found — did the LLM try anything first?
    if _response_has_tool_calls(transcript_path):
        log(f"Decline detected but tool calls present — allowing: '{matched_phrase[:60]}'")
        record_metric(session_id, "decline_with_tools", matched_phrase[:80])
        return None

    # Declined without trying
    log(f"Declined without trying: '{matched_phrase[:60]}'")
    record_metric(session_id, "declined_without_trying", matched_phrase[:80])
    return matched_phrase


# --- Correction trigger matching ---

def check_correction_triggers(text: str, session_id: str = "") -> Optional[tuple[str, str]]:
    """Check if response matches any stored correction triggers.

    Compares the response text against embedded correction triggers.
    Returns (trigger_text, correction_content) if matched, None otherwise.

    Triggers are phrases describing what the bad response looks like,
    written by the LLM at correction time. Each new correction with a trigger
    automatically expands detection coverage.
    """
    from cairn.config import CORRECTION_TRIGGER_ENABLED, CORRECTION_TRIGGER_SIM_THRESHOLD
    from hooks.hook_helpers import record_metric

    if not CORRECTION_TRIGGER_ENABLED:
        return None

    emb = hook_helpers.get_embedder()
    if not emb:
        return None

    # Strip memory/code blocks — match against actual response content
    cleaned = re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`[^`]+`", "", cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) < 30:
        return None

    # Embed the response tail (last ~500 chars — where decline/mistake language tends to be)
    response_tail = cleaned[-500:] if len(cleaned) > 500 else cleaned
    try:
        response_vec = emb.embed(response_tail, allow_slow=False)
        if response_vec is None:
            return None
    except Exception:
        return None

    # Load all correction triggers with embeddings
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT ct.trigger, ct.embedding, m.content, m.id
            FROM correction_triggers ct
            JOIN memories m ON ct.memory_id = m.id
            WHERE ct.embedding IS NOT NULL
            AND (m.archived_reason IS NULL OR m.archived_reason = '')
        """).fetchall()
    except Exception as e:
        log(f"Correction trigger query error: {e}")
        conn.close()
        return None
    conn.close()

    if not rows:
        return None

    # Compare response against each trigger
    best_sim = 0.0
    best_trigger = ""
    best_correction = ""
    best_id = 0

    for trigger_text, trigger_blob, correction_content, memory_id in rows:
        try:
            trigger_vec = emb.from_blob(trigger_blob)
            sim = float(emb.cosine_similarity(response_vec, trigger_vec))
            if sim > best_sim:
                best_sim = sim
                best_trigger = trigger_text
                best_correction = correction_content
                best_id = memory_id
        except Exception:
            continue

    if best_sim >= CORRECTION_TRIGGER_SIM_THRESHOLD:
        log(f"Correction trigger matched: sim={best_sim:.3f} trigger='{best_trigger[:60]}' memory={best_id}")
        record_metric(session_id, "correction_trigger_matched",
                      f"sim={best_sim:.3f} id={best_id} trigger={best_trigger[:60]}", best_sim)
        return (best_trigger, best_correction)

    if best_sim > 0.3:
        record_metric(session_id, "correction_trigger_near_miss",
                      f"sim={best_sim:.3f} id={best_id}", best_sim)

    return None
