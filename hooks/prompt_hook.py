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
    get_conn, get_embedder, get_session_project, record_metric,
    DB_PATH, LOG_PATH, strip_seen_entries, save_injected_ids,
    format_entry, split_by_scope, build_context_xml,
    record_layer_delivery, load_hook_state, save_hook_state, delete_hook_state,
)


def log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[prompt] {msg}\n")


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
    """Layer 1.5: Per-prompt semantic injection for subsequent prompts.

    Fires on every message after the first. Higher threshold than Layer 1 (0.55 vs 0.30)
    to avoid mid-session noise.
    """
    from cairn.config import L1_5_ENABLED, L1_5_SIM_THRESHOLD, L1_5_MAX_RESULTS, MIN_INJECTION_SIMILARITY

    if not L1_5_ENABLED:
        return None

    emb = get_embedder()
    if not emb:
        return None

    try:
        conn = get_conn()
        project = get_session_project(conn, session_id)
        count = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL").fetchone()[0]
        if count == 0:
            conn.close()
            return None

        results = emb.find_similar(conn, user_message, threshold=L1_5_SIM_THRESHOLD,
                                   limit=L1_5_MAX_RESULTS, current_project=project)
        conn.close()
    except Exception as e:
        log(f"Layer 1.5 error: {e}")
        return None

    if not results or results[0]["similarity"] < L1_5_SIM_THRESHOLD:
        record_metric(session_id, "layer1_5_no_match", user_message[:80])
        return None

    # Skip memories produced in this session (central gate handles cross-layer dedup)
    results = [r for r in results if r.get("session_id") != session_id]
    if not results:
        record_metric(session_id, "layer1_5_skipped_all_seen", user_message[:80], len(results))
        return None

    project_results, global_results = split_by_scope(results, project)
    result_ids = [r["id"] for r in project_results + global_results]
    record_metric(session_id, "layer1_5_injected", user_message[:80], len(results))
    record_layer_delivery(session_id, "L1.5", result_ids)
    return build_context_xml(user_message, project, "per-prompt", project_results, global_results)


def project_bootstrap(session_id: str, cwd: str) -> Optional[str]:
    """Project bootstrap: inject standing-context memories for the CWD project.

    Queries directly by project name + type filter — no semantic search needed.
    Gives Claude project awareness from CWD alone, independent of prompt content.
    """
    from cairn.config import PROJECT_BOOTSTRAP_ENABLED, PROJECT_BOOTSTRAP_MAX, PROJECT_BOOTSTRAP_TYPES
    from hooks.hook_helpers import recency_days, reliability_label

    if not PROJECT_BOOTSTRAP_ENABLED or not cwd:
        return None

    from hooks.hook_helpers import resolve_project
    project_name = resolve_project(cwd)
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


def layer1_search(user_message: str, session_id: str) -> Optional[str]:
    """Layer 1: Search cairn using user's first message."""
    from cairn.config import L1_SIM_THRESHOLD, L1_MAX_RESULTS, MIN_INJECTION_SIMILARITY

    emb = get_embedder()
    if not emb:
        return None

    try:
        conn = get_conn()
        project = get_session_project(conn, session_id)
        count = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL").fetchone()[0]
        if count == 0:
            conn.close()
            return None

        results = emb.find_similar(conn, user_message, threshold=L1_SIM_THRESHOLD,
                                   limit=L1_MAX_RESULTS, current_project=project)
        conn.close()
    except Exception as e:
        log(f"Layer 1 error: {e}")
        return None

    if not results or results[0]["similarity"] < MIN_INJECTION_SIMILARITY:
        return None

    project_results, global_results = split_by_scope(results, project)
    result_ids = [r["id"] for r in project_results + global_results]
    record_layer_delivery(session_id, "L1", result_ids)
    return build_context_xml(user_message, project, "first-prompt", project_results, global_results)


def main() -> None:
    raw = sys.stdin.read()
    hook_input = json.loads(raw)
    session_id = hook_input.get("session_id", "")
    cwd = hook_input.get("cwd", "")
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

        # Project bootstrap: inject standing context from CWD-matched project
        pb_context = project_bootstrap(session_id, cwd)
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

    if not is_subagent:
        # Clean up stale staged context (older than 7 days — sessions unlikely to resume)
        try:
            cleanup_conn = get_conn()
            from cairn.config import STAGED_CONTEXT_RETENTION_DAYS
            cleanup_conn.execute(
                "DELETE FROM hook_state WHERE key = 'staged_context' AND updated_at < datetime('now', ?)",
                (f"-{STAGED_CONTEXT_RETENTION_DAYS} days",)
            )
            cleanup_conn.commit()
            cleanup_conn.close()
        except Exception:
            pass

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
