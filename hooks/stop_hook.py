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

from hooks.hook_helpers import log, get_conn, record_metric, get_embedder, get_session_project, DB_PATH, strip_memory_block, strip_seen_entries, save_injected_ids, record_layer_delivery
from hooks.parser import parse_memory_block
from hooks.hash_verify import compute_response_hash
from hooks.storage import apply_confidence_updates, insert_memories
from hooks.enforcement import check_trailing_intent, get_continuation_count, increment_continuation, reset_continuation

# Appended to block reasons that are purely about memory format — the user already saw the
# response in interactive mode, so restating it is wasteful.  Only used for format/density
# blocks, NOT for behavioural blocks (incomplete work, trailing intent, context retrieval).
AMEND_ONLY_SUFFIX = (
    "\n\nIMPORTANT: The user has already seen your previous response. "
    "Do NOT restate or repeat it. Just output a single short line like "
    "\"Memory block amended.\" followed by a corrected <memory> block."
)
from hooks.retrieval import (retrieve_context, layer2_cross_project_search,
                        load_context_cache, save_context_cache, is_context_cached, add_to_context_cache,
                        CONTEXT_CACHE_SIM_THRESHOLD)
from cairn.config import MAX_CONTINUATIONS, WEAK_ENTRY_SCORE_FLOOR, CONTEXT_BOOTSTRAP_INTERVAL


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
    is_subagent: bool = bool(hook_input.get("agent_id"))

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
        log(f"No text found in hook input. Keys: {list(hook_input.keys())}")
        sys.exit(0)

    # Headless assessment sessions (e.g. contradiction scanner) — skip all enforcement
    if os.environ.get("CAIRN_HEADLESS"):
        log("Headless mode — skipping enforcement")
        sys.exit(0)

    log(f"Text length: {len(text)}, has <memory>: {'<memory>' in text}, continuation: {is_continuation}, subagent: {is_subagent}")

    # Parse memory block
    parsed = parse_memory_block(text)
    entries, complete, remaining = parsed.entries, parsed.complete, parsed.remaining
    context, context_need = parsed.context, parsed.context_need
    confidence_updates, retrieval_outcome = parsed.confidence_updates, parsed.retrieval_outcome
    keywords, intent = parsed.keywords, parsed.intent
    hash_claimed = parsed.hash_claimed

    # Subagent mode: opportunistic — store what's volunteered, skip enforcement
    if is_subagent:
        if confidence_updates:
            apply_confidence_updates(confidence_updates, session_id=session_id)
        if retrieval_outcome:
            record_metric(session_id, f"retrieval_{retrieval_outcome}", context_need[:100] if context_need else None)
        if entries:
            count = insert_memories(entries, session_id=session_id, transcript_path=transcript_path)
            record_metric(session_id, "memories_stored", None, count)
            log(f"Subagent: stored {count} memories opportunistically")
        record_metric(session_id, "hook_fired", f"subagent,entries={len(entries) if entries else 0}")
        sys.exit(0)

    # No memory block found
    if entries is None and complete is None:
        record_metric(session_id, "missing_memory_block", None, 1 if is_continuation else 0)
        if is_continuation:
            log("Missing memory block on continuation — allowing stop to prevent loop")
            reset_continuation(session_id)
            sys.exit(0)

        # Check if this session has ever produced a memory block. If not, the LLM
        # likely doesn't have the rules loaded (e.g. hooks activated mid-session
        # before restart). Fail open rather than enforcing on an uninstructed LLM.
        conn = get_conn()
        session_has_memories = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id = ?", (session_id,)
        ).fetchone()[0] > 0
        # Also check if we've seen any hook_fired metric for this session
        session_hook_count = conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE session_id = ? AND event = 'hook_fired'", (session_id,)
        ).fetchone()[0]
        conn.close()

        if not session_has_memories and session_hook_count <= 1:
            log(f"No prior memories for session {session_id[:8]}... — LLM may lack rules, allowing stop")
            record_metric(session_id, "uninstructed_session_skip")
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
            result: dict = {"decision": "block", "reason": hint + AMEND_ONLY_SUFFIX}
        else:
            result = {
                "decision": "block",
                "reason": "Response missing required <memory> block. Add a <memory> block with at least complete: true before finishing." + AMEND_ONLY_SUFFIX
            }
        print(json.dumps(result))
        sys.exit(0)

    log(f"Parsed: entries={len(entries) if entries else 0}, complete={complete}, remaining={remaining}, context={context}, context_need={context_need}, conf_updates={len(confidence_updates)}")

    # Strict field validation — enforce explicit declaration of all required fields
    missing_fields: list[str] = []
    if not parsed.is_compact:
        # Verbose format: require explicit complete, context, keywords
        if not parsed.complete_explicit:
            missing_fields.append("complete: [true|false]")
        if not parsed.context_explicit:
            missing_fields.append("context: [sufficient|insufficient]")
        if not parsed.keywords_explicit:
            missing_fields.append("keywords: [comma-separated topic keywords]")
    else:
        # Compact format: keywords required on entries via [k: ...], not on no-ops
        if entries and not parsed.keywords_explicit:
            missing_fields.append("[k: keywords] on entry line")
    if complete is False and not remaining:
        missing_fields.append("remaining: [what still needs doing]" if not parsed.is_compact else "- :what still needs doing")
    if context == "insufficient" and not context_need:
        missing_fields.append("context_need: [what context is missing]" if not parsed.is_compact else "c?:what context is missing")

    # Validate entry completeness — every entry needs type, topic, content
    incomplete_entries: list[str] = []
    if entries:
        for i, entry in enumerate(entries):
            entry_missing = [f for f in ("type", "topic", "content") if f not in entry]
            if entry_missing:
                incomplete_entries.append(f"entry {i+1} missing: {', '.join(entry_missing)}")

    if (missing_fields or incomplete_entries) and not is_continuation:
        hints: list[str] = []
        if missing_fields:
            hints.append(f"Memory block is missing: {', '.join(missing_fields)}.")
        if incomplete_entries:
            hints.append(f"Incomplete entries: {'; '.join(incomplete_entries)}.")
        if parsed.is_compact:
            hint_text = " ".join(hints) + " Use this format:\n<memory>\ntype/topic: content [k: keyword1, keyword2]\n+ c h:NNN\n</memory>"
        else:
            hint_text = " ".join(hints) + " All fields are required. Use this format:\n<memory>\n- type: fact\n- topic: short key\n- content: one line\n- complete: true\n- context: sufficient\n- keywords: relevant, topic, words\n</memory>"
        log(f"Strict validation failed: {hint_text[:200]}")
        record_metric(session_id, "strict_validation_failed", hint_text[:100])
        increment_continuation(session_id)
        print(json.dumps({"decision": "block", "reason": hint_text + AMEND_ONLY_SUFFIX}))
        sys.exit(0)

    # Hash verification — optional, log-only (not blocking)
    if hash_claimed is not None:
        from hooks.hash_verify import verify_hash
        match, actual = verify_hash(text, hash_claimed)
        if match:
            log(f"Hash verified: {actual:X}")
            record_metric(session_id, "hash_verified", None, actual)
        else:
            log(f"Hash mismatch (non-blocking): claimed={hash_claimed:X}, actual={actual:X}")
            record_metric(session_id, "hash_mismatch", f"claimed={hash_claimed:X} actual={actual:X}")

    # Content density validation — reject lazy/thin entries
    density_issues: list[str] = []
    if entries:
        for i, entry in enumerate(entries):
            content = entry.get("content", "")
            if len(content) < 20:
                density_issues.append(f"entry {i+1} content too short ({len(content)} chars) — be more specific")
    else:
        # No entries — check if the response was substantive enough to warrant a memory
        stripped = strip_memory_block(text)
        if len(stripped) > 1000 and not is_continuation:
            density_issues.append(
                "Substantive response (>1000 chars) with no memory entries. "
                "Capture what was discussed, decided, or learned."
            )

    if density_issues and not is_continuation:
        hint_text = " ".join(density_issues)
        log(f"Content density check failed: {hint_text[:200]}")
        record_metric(session_id, "content_density_failed", hint_text[:100])
        increment_continuation(session_id)
        print(json.dumps({"decision": "block", "reason": hint_text + AMEND_ONLY_SUFFIX}))
        sys.exit(0)

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
        count = insert_memories(entries, session_id=session_id, transcript_path=transcript_path)
        record_metric(session_id, "memories_stored", None, count)
        log(f"Stored {count} memories (session: {session_id[:8]}...)" if session_id else f"Stored {count} memories")

    # Record dedup stats
    record_metric(session_id, "hook_fired", f"entries={len(entries) if entries else 0}")

    # Layer 2: cross-project keyword search (stages for next prompt, doesn't block)
    if keywords and not is_continuation:
        layer2_cross_project_search(keywords, session_id=session_id)

    # Check context sufficiency — retrieve and inject if insufficient
    LOW_INFO_STOPLIST: set[str] = {"help", "continue", "more", "yes", "no", "ok", "thanks", "done", "info", "more info"}
    if context == "insufficient" and context_need:
        # Record that the LLM declared insufficient — resets the bootstrap counter
        # even on continuations (where the bootstrap forced this declaration)
        record_metric(session_id, "context_requested", context_need[:100])

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
                # Check if this retrieval was triggered by bootstrap — apply tighter cap
                _is_bootstrap = False
                try:
                    _bc = get_conn()
                    _brow = _bc.execute(
                        "SELECT value FROM metrics WHERE session_id = ? AND event = 'context_bootstrap_triggered' "
                        "ORDER BY created_at DESC LIMIT 1", (session_id,)
                    ).fetchone()
                    _bc.close()
                    if _brow:
                        _is_bootstrap = True
                except Exception:
                    pass
                from cairn.config import BOOTSTRAP_MAX_PER_SCOPE
                _max_scope = BOOTSTRAP_MAX_PER_SCOPE if _is_bootstrap else None
                retrieved: Optional[str] = retrieve_context(context_need, session_id=session_id, max_per_scope=_max_scope)
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

                        # Central dedup gate — strip entries already injected
                        retrieved = strip_seen_entries(retrieved, session_id) or ""
                        if not retrieved:
                            log(f"All entries already seen for: {context_need[:50]}...")
                        else:
                            # Track newly injected IDs
                            injected_ids = [int(i) for i in re.findall(r'id="(\d+)"', retrieved)]
                            layer_name = "L3-bootstrap" if _is_bootstrap else "L3"
                            record_layer_delivery(session_id, layer_name, injected_ids)
                            save_injected_ids(session_id, injected_ids)

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

    # Context bootstrapping — force a context: insufficient declaration if the LLM
    # hasn't used layer 3 in CONTEXT_BOOTSTRAP_INTERVAL turns. Builds the habit
    # through demonstrated value rather than rules alone.
    if not is_continuation and context != "insufficient" and CONTEXT_BOOTSTRAP_INTERVAL > 0:
        conn = get_conn()
        # Count hook firings since last context_requested
        last_request = conn.execute(
            "SELECT MAX(created_at) FROM metrics WHERE session_id = ? AND event = 'context_requested'",
            (session_id,)
        ).fetchone()[0]
        if last_request:
            turns_since = conn.execute(
                "SELECT COUNT(*) FROM metrics WHERE session_id = ? AND event = 'hook_fired' AND created_at > ?",
                (session_id, last_request)
            ).fetchone()[0]
        else:
            turns_since = conn.execute(
                "SELECT COUNT(*) FROM metrics WHERE session_id = ? AND event = 'hook_fired'",
                (session_id,)
            ).fetchone()[0]
        conn.close()

        # Use shorter interval for first bootstrap in session, then standard interval
        from cairn.config import CONTEXT_BOOTSTRAP_FIRST_INTERVAL
        conn_check = get_conn()
        _prior_bootstrap = conn_check.execute(
            "SELECT COUNT(*) FROM metrics WHERE session_id = ? AND event = 'context_bootstrap_triggered'",
            (session_id,)
        ).fetchone()[0]
        conn_check.close()
        effective_interval = CONTEXT_BOOTSTRAP_FIRST_INTERVAL if _prior_bootstrap == 0 else CONTEXT_BOOTSTRAP_INTERVAL

        if turns_since >= effective_interval:
            record_metric(session_id, "context_bootstrap_triggered", None, turns_since)
            record_metric(session_id, "context_requested", "bootstrap_forced")

            bootstrap_reminder = (
                f"You have not checked cairn context in {turns_since} turns. "
                "In your memory block, declare context: insufficient with a context_need relevant to what you are "
                "currently discussing. Answer the user's question normally — the context declaration goes in "
                "the memory block only, not in place of your response."
            )

            # Check response length — block immediately if short, defer if substantive
            response_stripped = strip_memory_block(text)
            if len(response_stripped) < 200:
                # Short/empty response — safe to block now
                log(f"Context bootstrap: {turns_since} turns without layer 3 — blocking (response {len(response_stripped)} chars)")
                increment_continuation(session_id)
                print(json.dumps({"decision": "block", "reason": bootstrap_reminder}))
                sys.exit(2)
            else:
                # Substantive response — defer to next turn to avoid eating it
                log(f"Context bootstrap: {turns_since} turns without layer 3 — deferring (response {len(response_stripped)} chars)")
                try:
                    staged_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".staged_context")
                    os.makedirs(staged_dir, exist_ok=True)
                    staged_file = os.path.join(staged_dir, f"{session_id}_bootstrap.txt")
                    with open(staged_file, "w") as f:
                        f.write(bootstrap_reminder)
                    log(f"Bootstrap reminder staged for next prompt")
                except Exception as e:
                    log(f"Failed to stage bootstrap reminder: {e}")

    # Question-before-cairn enforcement — if the LLM is asking the user a question
    # but hasn't declared context: insufficient, it should check cairn first.
    # Deferred (not blocking) to avoid response double-up — the user already sees
    # the response before the stop hook fires, so blocking + "restate" causes duplicates.
    # Instead, stage a reminder for the next prompt, same pattern as bootstrap.
    if not is_continuation and context != "insufficient":
        response_stripped = strip_memory_block(text)
        # Strip code blocks and quoted strings to avoid false positives
        response_no_code = re.sub(r"```[\s\S]*?```", "", response_stripped)
        response_no_quotes = re.sub(r'"[^"]*\?"', "", response_no_code)
        response_no_quotes = re.sub(r"'[^']*\?'", "", response_no_quotes)
        # Check last 3 sentences for question marks (directed at user)
        sentences = [s.strip() for s in re.split(r'[.\n]', response_no_quotes) if s.strip()]
        tail = sentences[-3:] if len(sentences) >= 3 else sentences
        has_question = any("?" in s for s in tail)
        if has_question:
            log(f"Question-before-cairn: deferring reminder to next prompt (avoiding response double-up)")
            record_metric(session_id, "question_before_cairn")
            try:
                staged_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".staged_context")
                os.makedirs(staged_dir, exist_ok=True)
                staged_file = os.path.join(staged_dir, f"{session_id}_question_cairn.txt")
                with open(staged_file, "w") as f:
                    f.write(
                        "You asked the user a question without checking cairn for relevant context. "
                        "In your memory block, declare context: insufficient with a context_need matching "
                        "your question — the cairn may already have the answer from a previous session. "
                        "Answer the user's question normally — the context declaration goes in the memory block "
                        "only, not in place of your response."
                    )
                log(f"Question-before-cairn reminder staged for next prompt")
            except Exception as e:
                log(f"Failed to stage question-before-cairn reminder: {e}")

    # Inline contradiction enforcement — DISABLED
    # False positive rate too high despite sentence-level fix, quote stripping, threshold tuning.
    # Causes re-prompt loops that block real work. Voluntary -! annotations plus the offline
    # contradiction_scan.py provide the same safety net without blocking.
    # See memories #959, #928, #888, #897 for the full history.
    # The retrieved_ids tracking is kept for future use if a better heuristic is found.

    # Check completeness — complete must be explicitly True to pass.
    # If omitted (None) or False, treat as incomplete.
    if complete is not True:
        count = get_continuation_count(session_id)
        if count >= MAX_CONTINUATIONS:
            log(f"Completeness re-prompt cap reached ({count}/{MAX_CONTINUATIONS}) — forcing stop")
            record_metric(session_id, "completeness_cap_hit", remaining, count)
            reset_continuation(session_id)
            sys.exit(0)
        increment_continuation(session_id)
        if complete is None:
            llm_reason: str = (
                "Memory block is missing completeness declaration. "
                "Compact format: add a control line '+ c h:NNN' (where NNN is your response hash). "
                "Verbose format: add '- complete: true'. "
                "See the Response Hash section in your rules for how to compute h:NNN."
            ) + AMEND_ONLY_SUFFIX
        else:
            llm_reason = f"Response marked incomplete. Continue with: {remaining}" if remaining else "Response marked incomplete. Continue."
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
                    "If you genuinely have nothing more to do, add 'intent: resolved' to your <memory> block."
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
