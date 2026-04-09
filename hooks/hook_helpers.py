"""Shared helpers for Cairn hooks — DB access, logging, metrics, embedder, XML formatting."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from types import ModuleType
from typing import Any, Optional

CAIRN_DIR = os.path.join(os.path.dirname(__file__), "..", "cairn")
DB_PATH = os.environ.get("CAIRN_DB_PATH", os.path.join(CAIRN_DIR, "cairn.db"))
LOG_PATH = os.path.join(CAIRN_DIR, "hook.log")

# cairn package available via pip install -e .


def log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def get_conn() -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def resolve_project(cwd: str) -> str:
    """Resolve the project label for a session.

    Precedence: CAIRN_PROJECT environment variable (explicit override) →
    basename of cwd (current behaviour). Set CAIRN_PROJECT before launching
    Claude Code to override the cwd-based default — useful for catch-all
    directories like ~/Projects/temp/ or for benchmark isolation.
    """
    override = os.environ.get("CAIRN_PROJECT", "").strip().lower()
    if override:
        return override
    return os.path.basename(cwd.rstrip("/")).lower() if cwd else ""


def record_metric(session_id: str, event: str, detail: Optional[str] = None,
                  value: Optional[float] = None) -> None:
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO metrics (event, session_id, detail, value) VALUES (?, ?, ?, ?)",
            (event, session_id, detail, value)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_embedder() -> Optional[ModuleType]:
    """Lazy-load the embeddings module."""
    if os.environ.get("CAIRN_SKIP_EMBEDDER"):
        return None
    try:
        from cairn import embeddings
        return embeddings
    except ImportError:
        return None


def get_session_project(conn: sqlite3.Connection, session_id: str) -> Optional[str]:
    """Look up the project label for a session."""
    if not session_id:
        return None
    row = conn.execute("SELECT project FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return row[0] if row else None


# --- Generic hook state helpers ---

def load_hook_state(session_id: str, key: str) -> Optional[str]:
    """Load a raw string value from hook_state."""
    try:
        conn = get_conn()
        row = conn.execute(
            "SELECT value FROM hook_state WHERE session_id = ? AND key = ?",
            (session_id, key)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def save_hook_state(session_id: str, key: str, value: str) -> None:
    """Save a raw string value to hook_state (upsert)."""
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO hook_state (session_id, key, value, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
            (session_id, key, value)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def delete_hook_state(session_id: str, key: str) -> None:
    """Delete a hook_state entry."""
    try:
        conn = get_conn()
        conn.execute(
            "DELETE FROM hook_state WHERE session_id = ? AND key = ?",
            (session_id, key)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# --- Injected ID tracking (built on generic hook state) ---

def load_injected_ids(session_id: str) -> set[int]:
    """Load memory IDs already injected this session."""
    raw = load_hook_state(session_id, "retrieved_ids")
    if raw:
        try:
            return set(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass
    return set()


def save_injected_ids(session_id: str, new_ids: list[int]) -> None:
    """Accumulate newly injected memory IDs into session state."""
    if not new_ids:
        return
    existing = load_injected_ids(session_id)
    merged = list(existing | set(new_ids))
    save_hook_state(session_id, "retrieved_ids", json.dumps(merged))


# --- XML formatting helpers ---

def recency_days(updated_at: str) -> int:
    """Calculate days since a timestamp string (YYYY-MM-DD HH:MM:SS)."""
    try:
        updated = datetime.strptime(updated_at[:19], "%Y-%m-%d %H:%M:%S")
        return max(0, (datetime.now() - updated).days)
    except (ValueError, TypeError, KeyError):
        return 0


def reliability_label(score: float) -> str:
    """Map a composite score to a reliability label."""
    return "strong" if score >= 0.6 else "moderate" if score >= 0.4 else "weak"


def record_layer_delivery(session_id: str, layer: str, ids: list[int]) -> None:
    """Record which memory IDs were delivered via which layer."""
    if ids:
        record_metric(session_id, "layer_delivery", json.dumps({"layer": layer, "ids": ids}))


def strip_memory_block(text: str) -> str:
    """Remove <memory>...</memory> block from response text."""
    return re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL).strip()


def format_entry(r: dict[str, Any]) -> str:
    """Format a memory entry as XML for context injection."""
    sim = r.get("similarity", 0)
    conf = r.get("confidence", 0.7)
    score = r.get("score", conf)
    proj = r.get("project") or "global"
    rel = reliability_label(score)
    days = recency_days(r.get("updated_at", ""))
    reason = r.get("archived_reason")
    if r.get("archived") or reason:
        reason = reason or "unknown"
        return (
            f'  <entry id="{r["id"]}" type="{r["type"]}" topic="{r["topic"]}" '
            f'project="{proj}" superseded="true" reason="{reason}" days="{days}">'
            f'{r["content"]}</entry>'
        )
    return (
        f'  <entry id="{r["id"]}" type="{r["type"]}" topic="{r["topic"]}" '
        f'project="{proj}" date="{r["updated_at"]}" confidence="{conf:.2f}" '
        f'score="{score:.2f}" recency_days="{days}" reliability="{rel}" similarity="{sim:.2f}">'
        f'{r["content"]}</entry>'
    )


def split_by_scope(results: list[dict[str, Any]], project: Optional[str]) -> tuple[list[dict], list[dict]]:
    """Split results into project-scoped and global lists."""
    project_results = [r for r in results if project and r.get("project") == project]
    global_results = [r for r in results if not project or r.get("project") != project]
    return project_results, global_results


def build_context_xml(query: str, project: Optional[str], layer: str,
                      project_results: list[dict[str, Any]],
                      global_results: list[dict[str, Any]],
                      instruction: Optional[str] = None) -> str:
    """Build a complete <cairn_context> XML block."""
    safe_query = query[:80].replace('"', '&quot;')
    lines = [f'<cairn_context query="{safe_query}" current_project="{project or "none"}" layer="{layer}">']
    if instruction is None:
        instruction = ('Before acting on any entry below, run: python3 '
                       '/home/james/Projects/cairn/cairn/query.py --context &lt;id&gt; '
                       'to recover the full conversation behind it.')
    lines.append(f'  <instruction>{instruction}</instruction>')

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


# --- Central dedup gate ---

def strip_seen_entries(xml: str, session_id: str) -> Optional[str]:
    """Remove already-injected memory entries from XML context. Returns None if nothing remains."""
    seen = load_injected_ids(session_id)
    if not seen:
        return xml

    def _filter_entry(m: re.Match) -> str:
        entry_id = int(m.group(1))
        return "" if entry_id in seen else m.group(0)

    filtered = re.sub(
        r'[ \t]*<entry id="(\d+)"[^>]*>.*?</entry>\n?',
        _filter_entry, xml
    )

    # Clean up empty scopes
    filtered = re.sub(
        r'[ \t]*<scope[^>]*>\s*</scope>\n?',
        '', filtered
    )

    # Clean up empty cairn_context blocks
    filtered = re.sub(
        r'[ \t]*<cairn_context[^>]*>\s*(?:<instruction>.*?</instruction>\s*)?</cairn_context>\n?',
        '', filtered, flags=re.DOTALL
    )

    stripped = filtered.strip()
    return stripped if stripped else None
