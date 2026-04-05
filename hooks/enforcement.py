"""Enforcement mechanisms — trailing intent detection and continuation counting."""

from __future__ import annotations

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


def check_trailing_intent(text: str) -> Optional[str]:
    """Check if response ends with unfulfilled action intent.

    Returns the matched trailing sentence if intent detected, None otherwise.
    """
    # Questions are not intent — check before extracting
    cleaned = re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL).strip()
    if cleaned.rstrip().endswith("?"):
        return None

    last = _extract_last_sentence(text)
    if not last:
        return None

    refs = _get_intent_embeddings()
    if not refs:
        return None

    emb = hook_helpers.get_embedder()
    last_vec = emb.embed(last)
    if last_vec is None:
        return None

    max_sim = 0.0
    for ref_text, ref_vec in refs:
        sim = emb.cosine_similarity(last_vec, ref_vec)
        if sim > max_sim:
            max_sim = sim

    if max_sim > TRAILING_INTENT_SIM_THRESHOLD:
        log(f"Trailing intent match: sim={max_sim:.3f} last='{last[:60]}'")
        return last[:100]

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
