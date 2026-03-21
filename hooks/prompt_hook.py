#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit Hook for Cairn.

Handles two retrieval layers:
- Layer 1: First-prompt push — searches cairn using user's message on first prompt of session
- Layer 2: Cross-project injection — injects staged data from previous stop hook keyword search

Both inject via additionalContext (supported by UserPromptSubmit hooks).
"""

import json
import sqlite3
import sys
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", "cairn.db")
CAIRN_DIR = os.path.join(os.path.dirname(__file__), "..", "cairn")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", "hook.log")
STAGED_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", ".staged_context")
FIRST_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "cairn", ".first_prompt_done")

sys.path.insert(0, CAIRN_DIR)


def log(msg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[prompt] {msg}\n")


def get_embedder():
    try:
        import embeddings
        return embeddings
    except ImportError:
        return None


def get_session_project(session_id):
    if not session_id:
        return None
    try:
        conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
        row = conn.execute("SELECT project FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def is_first_prompt(session_id):
    """Check if this is the first prompt of the session."""
    try:
        with open(FIRST_PROMPT_PATH, "r", encoding="utf-8") as f:
            done = json.load(f)
        return session_id not in done
    except (FileNotFoundError, json.JSONDecodeError):
        return True


def mark_first_prompt_done(session_id):
    """Mark that the first prompt has been processed for this session."""
    try:
        with open(FIRST_PROMPT_PATH, "r", encoding="utf-8") as f:
            done = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        done = {}
    done[session_id] = True
    with open(FIRST_PROMPT_PATH, "w", encoding="utf-8") as f:
        json.dump(done, f)


def load_staged_context(session_id):
    """Load cross-project context staged by the stop hook."""
    try:
        with open(STAGED_PATH, "r", encoding="utf-8") as f:
            staged = json.load(f)
        data = staged.pop(session_id, None)
        # Clean up consumed data
        with open(STAGED_PATH, "w", encoding="utf-8") as f:
            json.dump(staged, f)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def format_entry(r):
    """Format a memory entry for injection."""
    from config import MIN_INJECTION_SIMILARITY
    sim = r.get("similarity", 0)
    conf = r.get("confidence", 0.7)
    score = r.get("score", conf)
    proj = r.get("project") or "global"
    rel = "strong" if score >= 0.6 else "moderate" if score >= 0.4 else "weak"
    days = 0
    try:
        updated = datetime.strptime(r["updated_at"][:19], "%Y-%m-%d %H:%M:%S")
        days = max(0, (datetime.now() - updated).days)
    except Exception:
        pass
    return (
        f'  <entry id="{r["id"]}" type="{r["type"]}" topic="{r["topic"]}" '
        f'project="{proj}" date="{r["updated_at"]}" confidence="{conf:.2f}" '
        f'score="{score:.2f}" recency_days="{days}" reliability="{rel}" similarity="{sim:.2f}">'
        f'{r["content"]}</entry>'
    )


def layer1_search(user_message, session_id):
    """Layer 1: Search cairn using user's first message."""
    from config import L1_SIM_THRESHOLD, L1_MAX_RESULTS, MIN_INJECTION_SIMILARITY

    emb = get_embedder()
    if not emb:
        return None

    project = get_session_project(session_id)

    try:
        conn = sqlite3.connect(DB_PATH); conn.execute("PRAGMA busy_timeout=5000")
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

    # Split into project and global
    project_results = [r for r in results if project and r.get("project") == project]
    global_results = [r for r in results if not project or r.get("project") != project]

    lines = [f'<cairn_context query="{user_message[:80]}" current_project="{project or "none"}" layer="first-prompt">']

    if project_results:
        lines.append(f'  <scope level="project" name="{project}" weight="high">')
        for r in project_results:
            lines.append("  " + format_entry(r))
        lines.append("  </scope>")

    if global_results:
        lines.append('  <scope level="global" weight="low">')
        for r in global_results:
            lines.append("  " + format_entry(r))
        lines.append("  </scope>")

    lines.append("</cairn_context>")
    return "\n".join(lines)


def main():
    raw = sys.stdin.read()
    hook_input = json.loads(raw)
    session_id = hook_input.get("session_id", "")
    user_message = hook_input.get("user_message", "")

    if not user_message or len(user_message) < 3:
        sys.exit(0)

    context_parts = []

    # Layer 1: First-prompt push
    if is_first_prompt(session_id):
        mark_first_prompt_done(session_id)
        l1_context = layer1_search(user_message, session_id)
        if l1_context:
            context_parts.append(l1_context)
            log(f"Layer 1: injected context for: {user_message[:50]}...")

    # Layer 2: Staged cross-project context from previous stop hook
    staged = load_staged_context(session_id)
    if staged:
        context_parts.append(staged)
        log(f"Layer 2: injected staged cross-project context")

    if not context_parts:
        sys.exit(0)

    combined = "\n\n".join(context_parts)
    output = {
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
