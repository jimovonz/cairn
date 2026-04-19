#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit Hook for Cairn.

Handles two retrieval layers:
- Layer 1: First-prompt push — searches cairn using user's message on first prompt of session
- Layer 2: Cross-project injection — injects staged data from previous stop hook keyword search

Both inject via additionalContext (supported by UserPromptSubmit hooks).
"""

from __future__ import annotations

import json
import re
import sys
import os
from typing import Any, Optional

from hooks.hook_helpers import (
    get_conn, get_embedder, get_session_project, record_metric, flush_metrics,
    DB_PATH, LOG_PATH, strip_seen_entries, save_injected_ids,
    format_entry, split_by_scope, build_context_xml,
    record_layer_delivery, load_hook_state, save_hook_state, delete_hook_state,
    log as _base_log,
)


def log(msg: str) -> None:
    _base_log(f"[prompt] {msg}")


def is_first_prompt(session_id: str) -> bool:
    """Check if this is the first prompt of the session."""
    return load_hook_state(session_id, "first_prompt_done") is None


def mark_first_prompt_done(session_id: str) -> None:
    """Mark that the first prompt has been processed for this session."""
    save_hook_state(session_id, "first_prompt_done", "1")


def load_staged_context(session_id: str) -> Optional[str]:
    """Load and consume cross-project context staged by the stop hook."""
    raw = load_hook_state(session_id, "staged_context")
    if raw:
        delete_hook_state(session_id, "staged_context")
    return raw


def layer1_5_search(user_message: str, session_id: str) -> Optional[str]:
    """Layer 1.5: Per-prompt hybrid injection for subsequent prompts.

    Fires on every message after the first. Higher threshold than Layer 1 (0.55 vs 0.30)
    to avoid mid-session noise. Uses full hybrid search (semantic + FTS5 + RRF).
    """
    from cairn.config import L1_5_ENABLED, L1_5_SIM_THRESHOLD, L1_5_MAX_RESULTS
    from hooks.retrieval import hybrid_search

    if not L1_5_ENABLED:
        return None

    try:
        conn = get_conn()
        project = get_session_project(conn, session_id)
        count = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL AND deleted_at IS NULL").fetchone()[0]
        if count == 0:
            conn.close()
            return None

        project_results, global_results, _ = hybrid_search(
            user_message, conn, project=project, session_id=session_id,
            threshold=L1_5_SIM_THRESHOLD, limit=L1_5_MAX_RESULTS,
        )
        conn.close()
    except Exception as e:
        log(f"Layer 1.5 error: {e}")
        return None

    if not project_results and not global_results:
        record_metric(session_id, "layer1_5_no_match", user_message[:80])
        return None

    result_ids = [r["id"] for r in project_results + global_results]
    record_metric(session_id, "layer1_5_injected", user_message[:80], len(result_ids))
    record_layer_delivery(session_id, "L1.5", result_ids)
    return build_context_xml(user_message, project, "per-prompt", project_results, global_results)


_KNOWLEDGE_QUESTION_PATTERNS = (
    # "what did/do/does ... about/with/for X" — past-context probes
    r"\bwhat (?:did|do|does|have|are) (?:we|i|you|they) ",
    # "do/did/does/have ... told/said/discussed/mentioned" — recall probes
    r"\b(?:do|did|does|have) (?:we|i|you) (?:tell|told|say|said|discuss|mention|remember|recall) ",
    # "what's my/your/our X" — possessive identity probes
    r"\bwhat'?s (?:my|your|our|the) ",
    # "remind me ... about" — explicit recall request
    r"\bremind me ",
    # "remember when" — episodic recall
    r"\bremember (?:when|that|how|why|the time)",
    # "have we ever" — historical existence probes
    r"\bhave (?:we|i|you) ever ",
    # "who is/was X" — person identity
    r"\bwho (?:is|was|are|were) ",
    # "where (do|did) ... live/work/come from" — biographical
    r"\bwhere (?:do|did|does) (?:i|we|you|they|he|she) (?:live|work|come|go)",
    # "what aspect/part of (my|your) X" — personal-context probes (caught the brother/job query)
    r"\bwhat (?:aspect|part|kind|type) of (?:my|your|our) ",
    # "tell me about my X" — direct biographical probe
    r"\btell me about (?:my|your|our|the) ",
)


def _is_knowledge_question(user_message: str) -> bool:
    """Detect if a user prompt is asking about prior knowledge that should be in cairn.

    Pattern-based heuristic — matches common phrasings of "what do you know about X"
    questions. False positives are cheap (LLM gets a reminder it can ignore); false
    negatives miss the active trigger but the timer-based bootstrap still fires.
    """
    msg_lower = user_message.lower()
    return any(re.search(p, msg_lower) for p in _KNOWLEDGE_QUESTION_PATTERNS)


def _check_db_integrity(session_id: str) -> Optional[str]:
    """Lightweight corruption check — runs once per session on first prompt.

    Uses PRAGMA quick_check (faster than full integrity_check — samples pages
    rather than scanning every btree). If corruption is detected, warns the
    user via additionalContext and records a metric.
    """
    try:
        conn = get_conn()
        result = conn.execute("PRAGMA quick_check").fetchone()
        conn.close()
        if result and result[0] == "ok":
            return None
        # Corruption detected
        detail = result[0][:200] if result else "unknown"
        log(f"DB CORRUPTION DETECTED: {detail}")
        record_metric(session_id, "db_corruption_detected", detail)
        return (
            "WARNING — Cairn database corruption detected. Memories may be incomplete or unreliable. "
            "Run `python3 $CAIRN_HOME/cairn/recover.py` to repair. "
            "Until repaired, treat retrieved memories with extra caution."
        )
    except Exception as e:
        log(f"Integrity check error: {e}")
        return None


def project_bootstrap(session_id: str, cwd: str, transcript_path: str = "") -> Optional[str]:
    """Project bootstrap: inject standing-context memories for the CWD project.

    Queries directly by project name + type filter — no semantic search needed.
    Gives Claude project awareness from CWD alone, independent of prompt content.
    """
    from cairn.config import PROJECT_BOOTSTRAP_ENABLED, PROJECT_BOOTSTRAP_MAX, PROJECT_BOOTSTRAP_TYPES
    from hooks.hook_helpers import recency_days, reliability_label

    if not PROJECT_BOOTSTRAP_ENABLED or not cwd:
        return None

    from hooks.hook_helpers import resolve_project
    project_name = resolve_project(cwd, transcript_path)
    if not project_name or project_name in (".", "/", "home", "tmp", "temp"):
        return None

    types = [t.strip() for t in PROJECT_BOOTSTRAP_TYPES.split(",")]
    placeholders = ",".join("?" * len(types))

    try:
        conn = get_conn()
        rows = conn.execute(f"""
            SELECT id, type, topic, content, updated_at, project, confidence, archived_reason
            FROM memories
            WHERE project = ? AND type IN ({placeholders})
            AND (archived_reason IS NULL OR archived_reason = '')
            AND deleted_at IS NULL
            ORDER BY updated_at DESC
            LIMIT ?
        """, (project_name, *types, PROJECT_BOOTSTRAP_MAX)).fetchall()
        conn.close()
    except Exception as e:
        log(f"Project bootstrap error: {e}")
        return None

    if not rows:
        return None

    # Convert rows to dicts for format_entry
    results = []
    for r in rows:
        mem_id, mem_type, topic, content, updated_at, project, confidence, archived_reason = r
        results.append({
            "id": mem_id, "type": mem_type, "topic": topic, "content": content,
            "updated_at": updated_at, "project": project_name,
            "confidence": confidence if confidence is not None else 0.7,
            "score": confidence if confidence is not None else 0.7,
            "similarity": 0, "archived_reason": archived_reason,
        })

    result_ids = [r["id"] for r in results]
    record_metric(session_id, "project_bootstrap_injected", project_name, len(rows))
    record_layer_delivery(session_id, "bootstrap", result_ids)
    log(f"Project bootstrap: injected {len(rows)} standing-context entries for {project_name}")

    instruction = ("These are standing-context memories for this project — "
                   "decisions, preferences, and facts that apply regardless of the current task.")
    return build_context_xml("project standing context", project_name, "project-bootstrap",
                             results, [], instruction=instruction)


def correction_bootstrap(session_id: str) -> Optional[str]:
    """Inject top behavioural corrections into every session.

    Corrections are about patterns of behaviour — they don't match specific queries
    semantically. Without unconditional injection, 96% of corrections never surface
    cross-session. This ensures the highest-value corrections are always present.
    """
    from cairn.config import CORRECTION_BOOTSTRAP_MAX
    from hooks.hook_helpers import recency_days, reliability_label

    try:
        conn = get_conn()
        # Pull top corrections by confidence (most corroborated = most validated)
        # then recency as tiebreaker. Exclude archived.
        rows = conn.execute("""
            SELECT id, type, topic, content, updated_at, project, confidence, archived_reason
            FROM memories
            WHERE type = 'correction'
            AND (archived_reason IS NULL OR archived_reason = '')
            AND deleted_at IS NULL
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
        """, (CORRECTION_BOOTSTRAP_MAX,)).fetchall()
        conn.close()
    except Exception as e:
        log(f"Correction bootstrap error: {e}")
        return None

    if not rows:
        return None

    results = []
    for r in rows:
        mem_id, mem_type, topic, content, updated_at, project, confidence, archived_reason = r
        results.append({
            "id": mem_id, "type": mem_type, "topic": topic, "content": content,
            "updated_at": updated_at, "project": project or "",
            "confidence": confidence if confidence is not None else 0.7,
            "score": confidence if confidence is not None else 0.7,
            "similarity": 0, "archived_reason": archived_reason,
        })

    result_ids = [r["id"] for r in results]
    record_metric(session_id, "correction_bootstrap_injected", None, len(rows))
    record_layer_delivery(session_id, "correction-bootstrap", result_ids)
    log(f"Correction bootstrap: injected {len(rows)} corrections")

    instruction = ("These are behavioural corrections from past sessions — mistakes made and lessons learned. "
                   "Apply these to avoid repeating the same errors.")
    return build_context_xml("behavioural corrections", None, "correction-bootstrap",
                             [], results, instruction=instruction)


def layer1_search(user_message: str, session_id: str) -> Optional[str]:
    """Layer 1: Hybrid search using user's first message (semantic + FTS5 + RRF)."""
    from cairn.config import L1_SIM_THRESHOLD, L1_MAX_RESULTS
    from hooks.retrieval import hybrid_search

    try:
        conn = get_conn()
        project = get_session_project(conn, session_id)
        count = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL AND deleted_at IS NULL").fetchone()[0]
        if count == 0:
            conn.close()
            return None

        project_results, global_results, _ = hybrid_search(
            user_message, conn, project=project, session_id=session_id,
            threshold=L1_SIM_THRESHOLD, limit=L1_MAX_RESULTS,
        )
        conn.close()
    except Exception as e:
        log(f"Layer 1 error: {e}")
        return None

    if not project_results and not global_results:
        return None

    result_ids = [r["id"] for r in project_results + global_results]
    record_layer_delivery(session_id, "L1", result_ids)
    return build_context_xml(user_message, project, "first-prompt", project_results, global_results)


def main() -> None:
    raw = sys.stdin.read()
    hook_input = json.loads(raw)
    session_id = hook_input.get("session_id", "") or hook_input.get("sessionId", "")
    cwd = hook_input.get("cwd", "")
    transcript_path = hook_input.get("transcript_path", "")
    user_message = hook_input.get("user_message") or hook_input.get("prompt", "")

    is_subagent = bool(hook_input.get("agent_id"))

    if not user_message or len(user_message) < 3:
        if not user_message:
            log(f"No user message found in hook input. Keys: {list(hook_input.keys())}")
        sys.exit(0)

    context_parts: list[str] = []

    # Layer 1: First-prompt push (+ bootstrap)
    if is_first_prompt(session_id):
        mark_first_prompt_done(session_id)

        # Lightweight corruption check on first prompt of each session
        corruption_warning = _check_db_integrity(session_id)
        if corruption_warning:
            context_parts.append(corruption_warning)

        # Project bootstrap: inject standing context from CWD-matched project
        pb_context = project_bootstrap(session_id, cwd, transcript_path)
        if pb_context:
            context_parts.append(pb_context)

        l1_context = layer1_search(user_message, session_id)
        if l1_context:
            context_parts.append(l1_context)
            log(f"Layer 1: injected context for: {user_message[:50]}...")
        if not is_subagent:
            # Memory block reminder on first prompt (not needed for subagents)
            context_parts.append(
                "MEMORY BLOCK: End every response with a <memory> block — entries, control signals, and confidence feedback."
            )

    elif not is_subagent:
        # Layer 1.5: Per-prompt semantic injection for subsequent prompts
        # Skipped for subagents — short-lived, adds latency
        l1_5_context = layer1_5_search(user_message, session_id)
        if l1_5_context:
            context_parts.append(l1_5_context)
            log(f"Layer 1.5: injected per-prompt context for: {user_message[:50]}...")

        # Active bootstrap trigger: detect knowledge questions in real time and
        # nudge the LLM to declare context: insufficient. Closes the gap between
        # CONTEXT_BOOTSTRAP_INTERVAL timer fires (every 20 turns) by reacting to
        # specific question shapes that should always trigger a cairn lookup.
        if _is_knowledge_question(user_message):
            context_parts.append(
                "KNOWLEDGE QUESTION DETECTED: this prompt asks about prior context, "
                "preferences, or facts that may already be stored in cairn. In your "
                "memory block, declare context: insufficient with a context_need that "
                "references the substantive nouns/entities from the question. For "
                "multi-dimensional questions use the | separator in your context_need."
            )
            record_metric(session_id, "active_bootstrap_triggered", user_message[:100])
            log(f"Active bootstrap triggered for knowledge question: {user_message[:80]}")

    if not is_subagent:
        # Clean up stale staged context (older than 7 days — sessions unlikely to resume)
        try:
            from hooks.hook_helpers import get_ephemeral_conn
            cleanup_conn = get_ephemeral_conn()
            from cairn.config import STAGED_CONTEXT_RETENTION_DAYS
            cleanup_conn.execute(
                "DELETE FROM hook_state WHERE key = 'staged_context' AND updated_at < datetime('now', ?)",
                (f"-{STAGED_CONTEXT_RETENTION_DAYS} days",)
            )
            cleanup_conn.commit()
            cleanup_conn.close()
        except Exception as e:
            log(f"Staged context cleanup failed: {e}")

        # Layer 2: Staged cross-project context from previous stop hook
        staged = load_staged_context(session_id)
        if staged:
            context_parts.append(staged)
            log(f"Layer 2: injected staged cross-project context")

    # Bootstrap reminder (deferred from previous stop hook — non-blocking)
    staged_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".staged_context")
    bootstrap_file = os.path.join(staged_dir, f"{session_id}_bootstrap.txt")
    if os.path.exists(bootstrap_file):
        try:
            with open(bootstrap_file, "r") as f:
                bootstrap_text = f.read().strip()
            os.remove(bootstrap_file)
            if bootstrap_text:
                context_parts.append(bootstrap_text)
                log(f"Bootstrap reminder injected (deferred)")
        except Exception as e:
            log(f"Failed to load bootstrap reminder: {e}")

    # Thin-retrieval escalation reminder (deferred from previous stop hook)
    thin_escalation_file = os.path.join(staged_dir, f"{session_id}_thin_escalation.txt")
    if os.path.exists(thin_escalation_file):
        try:
            with open(thin_escalation_file, "r") as f:
                escalation_text = f.read().strip()
            os.remove(thin_escalation_file)
            if escalation_text:
                context_parts.append(escalation_text)
                log(f"Thin-retrieval escalation reminder injected (deferred)")
        except Exception as e:
            log(f"Failed to load thin-retrieval escalation reminder: {e}")

    # Query-quality reminder (deferred from previous stop hook — phoned-in context_need)
    query_quality_file = os.path.join(staged_dir, f"{session_id}_query_quality.txt")
    if os.path.exists(query_quality_file):
        try:
            with open(query_quality_file, "r") as f:
                qq_text = f.read().strip()
            os.remove(query_quality_file)
            if qq_text:
                context_parts.append(qq_text)
                log(f"Query-quality reminder injected (deferred)")
        except Exception as e:
            log(f"Failed to load query-quality reminder: {e}")

    # Question-before-cairn reminder (deferred from previous stop hook)
    question_cairn_file = os.path.join(staged_dir, f"{session_id}_question_cairn.txt")
    if os.path.exists(question_cairn_file):
        try:
            with open(question_cairn_file, "r") as f:
                qc_text = f.read().strip()
            os.remove(question_cairn_file)
            if qc_text:
                context_parts.append(qc_text)
                log(f"Question-before-cairn reminder injected (deferred)")
        except Exception as e:
            log(f"Failed to load question-before-cairn reminder: {e}")

    if not context_parts:
        sys.exit(0)

    combined = "\n\n".join(context_parts)

    # Central dedup gate — strip entries already injected this session
    combined = strip_seen_entries(combined, session_id) or ""
    if not combined:
        sys.exit(0)

    # Track newly injected IDs for downstream dedup
    injected_ids = [int(i) for i in re.findall(r'id="(\d+)"', combined)]
    save_injected_ids(session_id, injected_ids)

    # Record original user message and injection size for benchmark data collection
    record_metric(session_id, "retrieval_query", user_message[:200])
    record_metric(session_id, "retrieval_tokens_est", None, len(combined) // 4)

    output: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": f"CAIRN CONTEXT (proactive retrieval — interpret per .claude/rules/memory-system.md):\n\n{combined}"
        }
    }
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            log(f"PROMPT HOOK CRASH: {e}")
        except Exception:
            pass
        sys.exit(0)
