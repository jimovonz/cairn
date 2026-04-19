"""Enforcement mechanisms — trailing intent, deferral detection, declined-without-trying,
correction triggers, and continuation counting."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import hooks.hook_helpers as hook_helpers
from hooks.hook_helpers import log, get_conn, get_ephemeral_conn
from cairn.config import TRAILING_INTENT_SIM_THRESHOLD

import numpy as np


# --- Deferral detection ---
# Catches the "I did part of it, the rest is for later" pattern where the LLM
# fabricates session/scope boundaries to justify stopping. This is INDEPENDENT
# of trailing-intent (which catches "let me do X" without doing it) — deferral
# catches "I already did X, Y is multi-session scope" which can co-occur with
# ok:true and tool calls.

DEFERRAL_REFS: list[str] = [
    # Session-boundary fabrication
    "multi-session scope",
    "for a future session",
    "deferred to next session",
    "next session will pick up",
    "this is a multi-session effort",
    "needs a dedicated session",
    "scoped to this session",
    "end of session checkpoint",
    "pause here and continue later",
    "beyond the scope of this session",
    "that's another full chunk",
    "remaining work is substantial enough to warrant",
    "for the next session",
    "in a follow-up session",
    "separate session for that",
    "pick this up in a new session",
    "wrap up this session",
    "good stopping point",
    "natural break point to stop",
    "continue in another session",
    "this session has covered enough",
    "defer the rest",
    # "Still needs" / incomplete-but-marked-done pattern
    "still needs to be done",
    "remaining items that need attention",
    "requires hardware not code",
    "requires manual testing",
    "TODO items remaining",
    "left as an exercise",
    "what still needs doing",
    "outstanding items for later",
    "not yet implemented but straightforward",
    "the following still need to be addressed",
]

_deferral_embeddings: Optional[list[tuple[str, np.ndarray]]] = None


def _get_deferral_embeddings() -> Optional[list[tuple[str, np.ndarray]]]:
    """Lazy-load and cache deferral reference embeddings."""
    global _deferral_embeddings
    if _deferral_embeddings is not None:
        return _deferral_embeddings
    emb = hook_helpers.get_embedder()
    if not emb:
        return None
    _deferral_embeddings = [(ref, emb.embed(ref)) for ref in DEFERRAL_REFS]
    _deferral_embeddings = [(ref, vec) for ref, vec in _deferral_embeddings if vec is not None]
    return _deferral_embeddings if _deferral_embeddings else None


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


def _extract_tail_sentences(text: str, n: int = 3) -> list[str]:
    """Extract the last *n* non-empty, non-memory-block sentences from the response."""
    cleaned = re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^\[(?:cm|cairn-memory)\]:\s*#\s*'.*'$", "", cleaned, flags=re.MULTILINE).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    sentences = re.split(r"[.!?\n]", cleaned)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return sentences[-n:] if sentences else []


def _strip_memory_and_code(text: str) -> str:
    """Strip memory blocks and code blocks from response text for semantic analysis."""
    cleaned = re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^\[(?:cm|cairn-memory)\]:\s*#\s*'.*'$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`[^`]+`", "", cleaned)
    return cleaned.strip()


def _is_permission_seeking(text: str) -> bool:
    """Detect questions that ask the user for permission to proceed — covert exit ramps."""
    return bool(re.search(
        r"want me to|shall I|should I|would you like me to|ready to proceed|"
        r"would you like to|do you want me|if you.{0,10}like.{0,10}I can|"
        r"want me to pick|want me to start|want me to continue|"
        r"I can .{0,30}\?$",
        text, re.IGNORECASE | re.MULTILINE
    ))


def _has_deferral_language(text: str) -> bool:
    """Quick regex pre-check for deferral-adjacent keywords before running embeddings."""
    return bool(re.search(
        r"session|multi.?session|defer|next time|follow.?up|wrap up|stopping point|"
        r"beyond.{0,20}scope|another.{0,10}chunk|pause here|pick.{0,10}up.{0,10}later|"
        r"still needs|remaining.{0,10}items|requires hardware|TODO.{0,10}remain|"
        r"left as.{0,10}exercise|outstanding.{0,10}later|not yet implemented",
        text, re.IGNORECASE
    ))


def check_deferral(text: str, complete: bool, session_id: str = "") -> Optional[str]:
    """Detect scope-deferral language in the response.

    Catches the pattern where the LLM fabricates session/scope boundaries to
    justify stopping ("this is multi-session scope", "defer to next session").
    Runs INDEPENDENTLY of ok:true — deferral in the body is suspicious even
    when (especially when) the memory block claims complete.

    Checks the last ~500 chars of cleaned response against DEFERRAL_REFS
    embeddings. Returns the matched deferral text if detected, None otherwise.
    """
    from hooks.hook_helpers import record_metric
    from cairn.config import TRAILING_INTENT_SIM_THRESHOLD

    cleaned = _strip_memory_and_code(text)
    if len(cleaned) < 30:
        return None

    # Quick keyword pre-check — skip embedding if no deferral-adjacent words
    if not _has_deferral_language(cleaned):
        return None

    refs = _get_deferral_embeddings()
    if not refs:
        return None

    emb = hook_helpers.get_embedder()
    if not emb:
        return None

    # Check last ~500 chars — deferral language tends to be in summaries at the end
    tail = cleaned[-500:]
    tail_vec = emb.embed(tail)
    if tail_vec is None:
        return None

    max_sim = 0.0
    best_ref = ""
    for ref_text, ref_vec in refs:
        sim = emb.cosine_similarity(tail_vec, ref_vec)
        if sim > max_sim:
            max_sim = sim
            best_ref = ref_text

    # Use same threshold as trailing intent — deferral is equally actionable
    if max_sim > TRAILING_INTENT_SIM_THRESHOLD:
        extra = " (ok:true — body contradicts completeness flag)" if complete else ""
        log(f"Deferral detected: sim={max_sim:.3f} ref='{best_ref}'{extra}")
        record_metric(session_id, "deferral_detected", f"sim={max_sim:.3f} ref={best_ref[:40]}", max_sim)
        return tail[:200]

    record_metric(session_id, "deferral_clear", f"sim={max_sim:.3f}", max_sim)
    return None


def check_trailing_intent(text: str, session_id: str = "") -> Optional[str]:
    """Check if response ends with unfulfilled action intent.

    Returns the matched trailing sentence if intent detected, None otherwise.
    Records metrics for all outcomes (detected, clear, skipped).
    """
    from hooks.hook_helpers import record_metric

    cleaned = _strip_memory_and_code(text)

    # Questions bypass trailing-intent UNLESS deferral or permission-seeking language present.
    # "Want me to continue?" / "Shall I proceed?" are covert exit ramps, not genuine questions.
    if cleaned.rstrip().endswith("?"):
        if not _has_deferral_language(cleaned) and not _is_permission_seeking(cleaned):
            return None

    tail = _extract_tail_sentences(text, n=3)
    if not tail:
        return None

    refs = _get_intent_embeddings()
    if not refs:
        log("Trailing intent: embedder unavailable — skipping check")
        record_metric(session_id, "trailing_intent_skipped", "embedder_unavailable")
        return None

    emb = hook_helpers.get_embedder()
    if not emb:
        log("Trailing intent: embedder unavailable for tail sentences — skipping check")
        record_metric(session_id, "trailing_intent_skipped", "embedder_unavailable")
        return None

    # Fast keyword pre-check: sentences starting with intent markers are
    # flagged directly — semantic similarity misses content-heavy sentences
    # like "Let me revise the plan to strip out the old code" (max sim 0.43).
    _INTENT_PREFIX = re.compile(
        r"^(?:let me|i'll|i will|let's|i'm going to)\b", re.IGNORECASE
    )
    for sentence in tail:
        if _INTENT_PREFIX.match(sentence):
            log(f"Trailing intent prefix match: '{sentence[:60]}'")
            record_metric(session_id, "trailing_intent_detected", f"prefix_match", 1.0)
            return sentence[:100]

    best_sim = 0.0
    best_sentence = tail[-1]
    for sentence in tail:
        sent_vec = emb.embed(sentence)
        if sent_vec is None:
            continue
        for _ref_text, ref_vec in refs:
            sim = emb.cosine_similarity(sent_vec, ref_vec)
            if sim > best_sim:
                best_sim = sim
                best_sentence = sentence

    if best_sim > TRAILING_INTENT_SIM_THRESHOLD:
        log(f"Trailing intent match: sim={best_sim:.3f} sent='{best_sentence[:60]}'")
        record_metric(session_id, "trailing_intent_detected", f"sim={best_sim:.3f}", best_sim)
        return best_sentence[:100]

    record_metric(session_id, "trailing_intent_clear", f"sim={best_sim:.3f}", best_sim)
    return None


# --- Continuation counting (SQLite-backed) ---

def get_continuation_count(session_id: str) -> int:
    """Get how many times we've re-prompted this session."""
    conn = get_ephemeral_conn()
    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'continuation_count'",
        (session_id,)
    ).fetchone()
    conn.close()
    return int(row[0]) if row and row[0] else 0


def increment_continuation(session_id: str) -> int:
    """Increment and return the continuation count."""
    conn = get_ephemeral_conn()
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
    conn = get_ephemeral_conn()
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
    from hooks.transcript_adapter import iter_normalized_entries
    try:
        entries = list(iter_normalized_entries(transcript_path))
        # Scan backwards from end — find tool calls after the last user message
        for entry in reversed(entries):
            msg = entry.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "user":
                break
            tool_name = entry.get("tool_name", "")
            if tool_name in _ACTION_TOOLS:
                return True
            if entry.get("type") == "assistant" and isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("name", "") in _ACTION_TOOLS:
                            return True
        return False  # no action tools found in current turn
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
    cleaned = re.sub(r"^\[(?:cm|cairn-memory)\]:\s*#\s*'.*'$", "", cleaned, flags=re.MULTILINE)
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
    cleaned = re.sub(r"^\[(?:cm|cairn-memory)\]:\s*#\s*'.*'$", "", cleaned, flags=re.MULTILINE)
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
