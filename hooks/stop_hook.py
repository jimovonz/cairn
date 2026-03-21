#!/usr/bin/env python3
"""
Claude Code Stop Hook for Cairn.

Reads the hook input from stdin, parses <memory> blocks from the transcript,
inserts new memories into the database, and blocks stopping if complete: false.

Exit codes:
  0 = allow stop
  2 = block stop (force continuation)
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.dirname(__file__))

from hook_helpers import log, get_conn, record_metric, get_embedder, get_session_project, DB_PATH
from parser import parse_memory_block
from storage import apply_confidence_updates, insert_memories
from enforcement import check_trailing_intent, get_continuation_count, increment_continuation, reset_continuation
from retrieval import (retrieve_context, layer2_cross_project_search,
                        load_context_cache, save_context_cache, is_context_cached, add_to_context_cache,
                        CONTEXT_CACHE_SIM_THRESHOLD)
from config import MAX_CONTINUATIONS, WEAK_ENTRY_SCORE_FLOOR


def register_session(session_id: str, transcript_path: str) -> None:
    """Register this session in the sessions table, extracting parent if available."""
    if not session_id:
        return
    conn = get_conn()
    existing = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if existing:
        conn.close()
        return

    # Extract parent session from first user message in transcript
    parent_session_id: Optional[str] = None
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") in ("user", "assistant"):
                    entry_session = entry.get("sessionId", "")
                    if entry_session and entry_session != session_id:
                        parent_session_id = entry_session
                    break
    except (FileNotFoundError, PermissionError):
        pass

    # Inherit project label from parent session
    project: Optional[str] = None
    if parent_session_id:
        row = conn.execute(
            "SELECT project FROM sessions WHERE session_id = ?", (parent_session_id,)
        ).fetchone()
        if row:
            project = row[0]

    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, parent_session_id, project, transcript_path) VALUES (?, ?, ?, ?)",
        (session_id, parent_session_id, project, transcript_path)
    )
    conn.commit()
    conn.close()
    if parent_session_id:
        log(f"Session {session_id[:8]}... parent: {parent_session_id[:8]}... project: {project}")
    else:
        log(f"Session {session_id[:8]}... (root)")


def auto_label_project(session_id: str, cwd: str) -> None:
    """Heuristically label a session's project based on the working directory."""
    if not session_id or not cwd:
        return
    conn = get_conn()
    row = conn.execute("SELECT project FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row and row[0]:
        conn.close()
        return

    project_name: str = os.path.basename(cwd.rstrip("/")).lower()
    if not project_name or project_name in (".", "/", "home"):
        conn.close()
        return

    conn.execute("UPDATE sessions SET project = ? WHERE session_id = ?", (project_name, session_id))
    conn.commit()
    conn.close()
    log(f"Auto-labelled project: {project_name} (from cwd: {cwd})")


def main() -> None:
    raw: str = sys.stdin.read()
    log(f"--- Hook fired ---")
    hook_input: dict = json.loads(raw)

    is_continuation: bool = hook_input.get("stop_hook_active", False)
    transcript_path: str = hook_input.get("transcript_path", "")
    session_id: str = hook_input.get("session_id", "")
    cwd: str = hook_input.get("cwd", "")

    # Register session and track parent chain
    register_session(session_id, transcript_path)

    # Auto-label project from working directory
    auto_label_project(session_id, cwd)

    # Check continuation cap
    if is_continuation:
        count: int = get_continuation_count(session_id)
        if count >= MAX_CONTINUATIONS:
            log(f"Continuation cap reached ({count}/{MAX_CONTINUATIONS}) — forcing stop")
            record_metric(session_id, "continuation_cap_hit", None, count)
            reset_continuation(session_id)
            sys.exit(0)

    # Use last_assistant_message — this is the current response
    text: str = hook_input.get("last_assistant_message", "")

    if not text:
        log("No text found, allowing stop")
        sys.exit(0)

    log(f"Text length: {len(text)}, has <memory>: {'<memory>' in text}, continuation: {is_continuation}")

    # Parse memory block
    entries, complete, remaining, context, context_need, confidence_updates, retrieval_outcome, keywords, intent = parse_memory_block(text)

    # No memory block found
    if entries is None and complete is None:
        record_metric(session_id, "missing_memory_block", None, 1 if is_continuation else 0)
        if is_continuation:
            log("Missing memory block on continuation — allowing stop to prevent loop")
            reset_continuation(session_id)
            sys.exit(0)
        increment_continuation(session_id)

        has_open_tag: bool = "<memory>" in text
        has_close_tag: bool = "</memory>" in text
        if has_open_tag:
            record_metric(session_id, "malformed_memory_block")
            hint: str = "Your <memory> block could not be parsed. "
            if not has_close_tag:
                hint += "Missing closing </memory> tag. "
            hint += "Use this exact format:\n<memory>\n- type: fact\n- topic: example\n- content: one line description\n- complete: true\n</memory>"
            result: dict = {"decision": "block", "reason": hint}
        else:
            result = {
                "decision": "block",
                "reason": "Response missing required <memory> block. Add a <memory> block with at least complete: true before finishing."
            }
        print(json.dumps(result))
        sys.exit(0)

    log(f"Parsed: entries={len(entries) if entries else 0}, complete={complete}, remaining={remaining}, context={context}, context_need={context_need}, conf_updates={len(confidence_updates)}")

    # Apply confidence updates
    if confidence_updates:
        applied: int = apply_confidence_updates(confidence_updates, session_id=session_id)
        record_metric(session_id, "confidence_updates", None, applied)

    # Record retrieval outcome (system-level learning signal)
    if retrieval_outcome:
        record_metric(session_id, f"retrieval_{retrieval_outcome}", context_need[:100] if context_need else None)
        log(f"Retrieval outcome: {retrieval_outcome}")

    # Insert memories into DB
    if entries:
        count = insert_memories(entries, session_id=session_id)
        record_metric(session_id, "memories_stored", None, count)
        log(f"Stored {count} memories (session: {session_id[:8]}...)" if session_id else f"Stored {count} memories")

    # Record dedup stats
    record_metric(session_id, "hook_fired", f"entries={len(entries) if entries else 0}")

    # Layer 2: cross-project keyword search (stages for next prompt, doesn't block)
    if keywords and not is_continuation:
        layer2_cross_project_search(keywords, session_id=session_id)

    # Check context sufficiency — retrieve and inject if insufficient
    LOW_INFO_STOPLIST: set[str] = {"help", "continue", "more", "yes", "no", "ok", "thanks", "done", "info", "more info"}
    if context == "insufficient" and context_need and not is_continuation:
        need_words: set[str] = set(context_need.lower().split())
        if len(context_need) < 8 or need_words <= LOW_INFO_STOPLIST:
            log(f"Pre-filter: skipping low-info context_need: {context_need}")
            record_metric(session_id, "context_prefiltered", context_need[:100])
        else:
            record_metric(session_id, "context_requested", context_need[:100])
            emb = get_embedder()
            served: list = load_context_cache(session_id)
            if not is_context_cached(context_need, served, emb):
                retrieved: Optional[str] = retrieve_context(context_need, session_id=session_id)
                if retrieved:
                    import re as _re
                    score_match: Optional[re.Match[str]] = _re.search(r'score="([0-9.]+)"', retrieved)
                    top_score: float = float(score_match.group(1)) if score_match else 1.0
                    if top_score < WEAK_ENTRY_SCORE_FLOOR:
                        log(f"Weak-entry suppression: top score {top_score:.2f} — skipping injection")
                        record_metric(session_id, "context_weak_suppressed", context_need[:100])
                    else:
                        served = add_to_context_cache(context_need, served, emb)
                        save_context_cache(session_id, served)
                        record_metric(session_id, "context_served", context_need[:100])
                        log(f"Context retrieval for: {context_need[:50]}...")
                        increment_continuation(session_id)
                        result = {
                            "decision": "block",
                            "reason": f"CAIRN CONTEXT:\n{retrieved}"
                        }
                        print(json.dumps(result))
                        sys.exit(0)
                else:
                    log(f"No context found for: {context_need}")
            else:
                record_metric(session_id, "context_cache_hit", context_need[:100])
                log(f"Context already served (semantic match) for: {context_need[:50]}... — skipping")

    # Check completeness
    if complete is False:
        count = get_continuation_count(session_id)
        if count >= MAX_CONTINUATIONS:
            log(f"Completeness re-prompt cap reached ({count}/{MAX_CONTINUATIONS}) — forcing stop")
            record_metric(session_id, "completeness_cap_hit", remaining, count)
            reset_continuation(session_id)
            sys.exit(0)
        increment_continuation(session_id)
        llm_reason: str = f"Response marked incomplete. Continue with: {remaining}" if remaining else "Response marked incomplete. Continue."
        result = {
            "decision": "block",
            "reason": llm_reason
        }
        print(json.dumps(result))
        sys.exit(0)

    # Trailing intent detection — block if response ends with unfulfilled action intent
    if intent == "resolved":
        log("Intent explicitly resolved via memory block — skipping trailing intent check")
    elif not is_continuation:
        intent_result: Optional[str] = check_trailing_intent(text)
        if intent_result:
            log(f"Trailing intent detected: {intent_result}")
            record_metric(session_id, "trailing_intent_blocked", intent_result)
            increment_continuation(session_id)
            result = {
                "decision": "block",
                "reason": (
                    f"Your response ends with a stated intent to act: \"{intent_result}\". "
                    "Either follow through now, or remove the promise. "
                    "If you genuinely have nothing more to do, add '- intent: resolved' to your <memory> block."
                )
            }
            print(json.dumps(result))
            sys.exit(0)

    # All good — reset continuation counter and allow stop
    reset_continuation(session_id)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail open — never block the user due to a hook bug
        try:
            log(f"HOOK CRASH: {e}")
            record_metric("", "hook_crash", str(e))
        except Exception:
            pass
        sys.exit(0)
