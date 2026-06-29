"""Shared helpers for Cairn hooks — DB access, logging, metrics, embedder, XML formatting."""

from __future__ import annotations

import html
import json
import logging
import logging.handlers
import os
import re
import sys
from datetime import datetime
from types import ModuleType
from typing import Any, Optional

# Prefer pysqlite3 (ships SQLite 3.51.1) over stdlib sqlite3 (3.37.2 on Ubuntu 22.04).
# The system SQLite has known WAL checkpoint race conditions fixed in 3.39-3.44.
# pysqlite3-binary is a drop-in replacement — same DB format, same API.
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err

CAIRN_DIR = os.path.join(os.path.dirname(__file__), "..", "cairn")
DB_PATH = os.environ.get("CAIRN_DB_PATH", os.path.join(CAIRN_DIR, "cairn.db"))
LOG_PATH = os.path.join(CAIRN_DIR, "hook.log")

# cairn package available via pip install -e .

# --- Structured logging with rotation ---
_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_LOG_BACKUP_COUNT = 3             # Keep 3 rotated backups (20 MB total max)


class _RetargetableFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that re-resolves its target from the module-level
    ``LOG_PATH`` on every emit.

    A plain RotatingFileHandler binds the production ``hook.log`` path at import
    time, so the widespread test pattern of reassigning/patching
    ``hook_helpers.LOG_PATH`` was silently ineffective — every such test was
    really writing into the live ``hook.log`` (e.g. test_db_error_logging's
    fake "database disk image is malformed" lines polluting production), and its
    log-content assertions were vacuous. Re-checking ``LOG_PATH`` per emit makes
    that redirection actually work and stops the pollution, with no churn across
    the ~15 test files that already use the pattern."""

    def emit(self, record):  # type: ignore[override]
        target = os.path.abspath(LOG_PATH)
        if target != self.baseFilename:
            self.baseFilename = target
            if self.stream:
                try:
                    self.stream.close()
                finally:
                    self.stream = None
        super().emit(record)


_logger = logging.getLogger("cairn")
if not _logger.handlers:
    _logger.setLevel(logging.DEBUG)
    _handler = _RetargetableFileHandler(
        LOG_PATH, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _logger.addHandler(_handler)
    _logger.propagate = False


def log(msg: str) -> None:
    _logger.info(msg)


def log_warning(msg: str) -> None:
    _logger.warning(msg)


def log_error(msg: str) -> None:
    _logger.error(msg)


_VEC_LOAD_WARNED = False


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec extension. Without it, writes to memories_vec fail with
    'no such module: vec0', leaving the ANN index empty so find_nearest falls
    back to a Python-side cosine scan over every row — orders of magnitude slower."""
    global _VEC_LOAD_WARNED
    try:
        import sqlite_vec  # type: ignore[import-untyped]
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (ImportError, OSError, AttributeError) as e:
        if not _VEC_LOAD_WARNED:
            log_warning(f"sqlite_vec not loaded ({type(e).__name__}: {e}); memories_vec writes will fail and search falls back to brute force")
            _VEC_LOAD_WARNED = True


def get_conn() -> sqlite3.Connection:
    """Create a SQLite connection to the main (durable) DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    _load_sqlite_vec(conn)
    return conn


def get_ephemeral_conn() -> sqlite3.Connection:
    """Create a SQLite connection to the ephemeral DB (metrics, hook_state,
    pair_assessments). Always uses the dedicated ephemeral file — no fallback
    to main DB. Containing high-frequency writers off the durable file is the
    whole point of the split; falling back defeats it.

    Tests share path by patching cairn.config.EPHEMERAL_DB_PATH to the test
    DB_PATH; auto-init via init_ephemeral handles missing tables on first use."""
    from cairn.config import EPHEMERAL_DB_PATH
    from cairn.init_db import EPHEMERAL_TABLES

    def _open() -> sqlite3.Connection:
        c = sqlite3.connect(EPHEMERAL_DB_PATH)
        c.execute("PRAGMA busy_timeout=5000")
        c.execute("PRAGMA journal_mode=WAL")
        return c

    # Self-heal the ephemeral schema. The old probe checked ONLY `metrics`, so a
    # DB that had `metrics` but was missing `hook_state` (older init, schema
    # drift, or a partial rebuild after a corruption reset) was never repaired —
    # producing the recurring "no such table: hook_state" that silently dropped
    # graph dedup state and the fleet sweep-state write. Probe EVERY ephemeral
    # table (EPHEMERAL_TABLES is the single source of truth shared with
    # init_ephemeral, so the probe can never drift behind a newly-added table —
    # calibration_deliveries was missing from the old hand-maintained list), and
    # rebuild outright if the file itself is corrupt (ephemeral = disposable).
    # The whole open+probe is guarded so corruption surfacing at PRAGMA (e.g. a
    # non-database file) is handled too.
    conn = None
    try:
        conn = _open()
        for _t in EPHEMERAL_TABLES:
            conn.execute(f"SELECT 1 FROM {_t} LIMIT 0")
        return conn
    except sqlite3.DatabaseError as exc:
        msg = str(exc).lower()
        if "no such table" in msg and conn is not None:
            # Missing table(s) — init_ephemeral is idempotent (CREATE IF NOT EXISTS);
            # the existing autocommit conn sees the new tables on its next query.
            from cairn.init_db import init_ephemeral
            init_ephemeral(EPHEMERAL_DB_PATH)
            return conn
        if not _is_corruption_error(exc):
            raise
        # Malformed / unreadable image — the file is unusable; rebuild empty.
        log_error(f"ephemeral DB corrupt ({type(exc).__name__}: {exc}); rebuilding")
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        _reset_ephemeral_db(EPHEMERAL_DB_PATH)
        return _open()


def _reset_ephemeral_db(path: str) -> None:
    """Delete a corrupt ephemeral DB (and its WAL/SHM sidecars) and rebuild the
    empty schema. Ephemeral data (metrics, hook_state, transient queues) is
    disposable by design, so a clean rebuild beats a permanently unreadable file."""
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass
    from cairn.init_db import init_ephemeral
    init_ephemeral(path)


def query_py_invoked_since(transcript_path: str, since_iso: str) -> bool:
    """Scan transcript for substantive Bash tool calls invoking query.py.

    Returns True if any non-trivial query.py call appears after since_iso.
    Trivial subcommands (--stats, --check, --recent, --projects, --today) don't
    count as "actively searching" — they're status queries, not knowledge probes.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return False
    TRIVIAL = ("--stats", "--check", "--recent", "--projects", "--today", "--bootstrap")
    SUBSTANTIVE_FLAGS = ("--semantic", "--type", "--project", "--since", "--context",
                         "--history", "--compact")
    from hooks.transcript_adapter import iter_normalized_entries
    for entry in iter_normalized_entries(transcript_path):
        try:
            ts = entry.get("timestamp", "")
            if not ts or ts < since_iso:
                continue
            msg = entry.get("message", {})
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use" or block.get("name") != "Bash":
                    continue
                cmd = block.get("input", {}).get("command", "")
                if "query.py" not in cmd:
                    continue
                if any(flag in cmd for flag in SUBSTANTIVE_FLAGS):
                    return True
                if not any(t in cmd for t in TRIVIAL):
                    return True
        except (AttributeError, KeyError):
            continue
    return False


def last_user_message(transcript_path: str) -> str:
    """Return the most recent user message text from the transcript JSONL.

    Skips command messages (slash commands) and tool results — only returns
    actual user-typed prompts. Returns empty string if no user message exists
    or transcript is unreadable.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    last = ""
    from hooks.transcript_adapter import iter_normalized_entries
    for entry in iter_normalized_entries(transcript_path):
        try:
            msg = entry.get("message", {})
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            elif isinstance(content, str):
                text = content
            else:
                continue
            if "<command-message>" in text or "<tool_use_id>" in text:
                continue
            text = text.strip()
            if text:
                last = text
        except (AttributeError, KeyError):
            continue
    return last


def _resolve_workspace_from_transcript(transcript_path: str) -> str:
    """Derive workspace folder from VS Code's workspace.json adjacent to a
    Copilot transcript path.

    Copilot transcripts live at:
        <workspaceStorage>/<hash>/GitHub.copilot-chat/transcripts/<session>.jsonl

    The workspace.json at <workspaceStorage>/<hash>/workspace.json maps the
    hash back to the actual folder (e.g. file:///home/user/Projects/myproject).
    """
    if not transcript_path:
        return ""
    try:
        parts = transcript_path.split("/")
        idx = parts.index("GitHub.copilot-chat")
        ws_dir = "/".join(parts[:idx])
        ws_json = os.path.join(ws_dir, "workspace.json")
        if os.path.isfile(ws_json):
            with open(ws_json, encoding="utf-8") as f:
                data = json.loads(f.read())
            folder = data.get("folder", "")
            if folder.startswith("file:///"):
                folder = folder[7:]
            return folder.rstrip("/")
    except (ValueError, OSError, json.JSONDecodeError, KeyError):
        pass
    return ""


def resolve_project(cwd: str, transcript_path: str = "") -> str:
    """Resolve the project label for a session.

    Precedence: CAIRN_PROJECT environment variable (explicit override) →
    basename of cwd (if it's a specific project, not a generic home dir) →
    workspace folder from VS Code workspace.json (Copilot sessions) →
    basename of cwd as final fallback.
    """
    override = os.environ.get("CAIRN_PROJECT", "").strip().lower()
    if override:
        return override

    home = os.path.expanduser("~")
    cwd_clean = cwd.rstrip("/") if cwd else ""

    if cwd_clean and cwd_clean != home:
        return os.path.basename(cwd_clean).lower()

    ws_folder = _resolve_workspace_from_transcript(transcript_path)
    if ws_folder and ws_folder != home:
        return os.path.basename(ws_folder).lower()

    return os.path.basename(cwd_clean).lower() if cwd_clean else ""


def _is_corruption_error(exc: Exception) -> bool:
    """Detect SQLite errors that indicate actual DB damage (as opposed to lock contention)."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "malformed", "disk image", "file is not a database",
        "database is corrupt", "no such table",
    ))


def _log_db_error(where: str, exc: Exception) -> None:
    """Log a DB error loudly. Corruption goes to a dedicated corruption.log so
    it can't be buried inside the normal hook log stream."""
    log_error(f"DB ERROR in {where}: {type(exc).__name__}: {exc}")
    if _is_corruption_error(exc):
        try:
            corruption_log = os.path.join(os.path.dirname(LOG_PATH), "corruption.log")
            with open(corruption_log, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {where}: {exc}\n")
        except OSError:
            pass


# --- Metric batching ---
# Instead of one connection per metric write (caused WAL contention and corruption),
# batch metrics in memory and flush once per hook invocation via flush_metrics().
_metric_buffer: list[tuple[str, str, Optional[str], Optional[float]]] = []


def record_metric(session_id: str, event: str, detail: Optional[str] = None,
                  value: Optional[float] = None) -> None:
    _metric_buffer.append((event, session_id, detail, value))


def flush_metrics() -> None:
    """Write all buffered metrics in a single transaction. Call once at the end of each hook."""
    global _metric_buffer
    if not _metric_buffer:
        return
    batch = _metric_buffer[:]
    _metric_buffer = []
    try:
        conn = get_ephemeral_conn()
        conn.executemany(
            "INSERT INTO metrics (event, session_id, detail, value) VALUES (?, ?, ?, ?)",
            batch
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        _log_db_error(f"flush_metrics({len(batch)} events)", exc)


# Auto-flush: register atexit so metrics flush on process exit (covers sys.exit in hooks)
import atexit as _atexit
_atexit.register(flush_metrics)


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
    """Load a raw string value from hook_state (ephemeral DB)."""
    try:
        conn = get_ephemeral_conn()
        row = conn.execute(
            "SELECT value FROM hook_state WHERE session_id = ? AND key = ?",
            (session_id, key)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as exc:
        _log_db_error(f"load_hook_state({key})", exc)
        return None


def save_hook_state(session_id: str, key: str, value: str) -> None:
    """Save a raw string value to hook_state (ephemeral DB, upsert)."""
    try:
        conn = get_ephemeral_conn()
        conn.execute(
            "INSERT INTO hook_state (session_id, key, value, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
            (session_id, key, value)
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        _log_db_error(f"save_hook_state({key})", exc)


def delete_hook_state(session_id: str, key: str) -> None:
    """Delete a hook_state entry (ephemeral DB)."""
    try:
        conn = get_ephemeral_conn()
        conn.execute(
            "DELETE FROM hook_state WHERE session_id = ? AND key = ?",
            (session_id, key)
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        _log_db_error(f"delete_hook_state({key})", exc)


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
    """Record which memory IDs were delivered via which layer.

    Also bumps per-memory lifetime counters in delivery_counts (ephemeral DB)
    — the signal the per-file injection path uses to retire evergreen entries.
    """
    if not ids:
        return
    record_metric(session_id, "layer_delivery", json.dumps({"layer": layer, "ids": ids}))
    try:
        conn = get_ephemeral_conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS delivery_counts ("
            "memory_id INTEGER PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, "
            "last_delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.executemany(
            "INSERT INTO delivery_counts (memory_id, count) VALUES (?, 1) "
            "ON CONFLICT(memory_id) DO UPDATE SET count = count + 1, "
            "last_delivered_at = CURRENT_TIMESTAMP",
            [(i,) for i in ids]
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        _log_db_error("record_layer_delivery(counts)", exc)


def overdelivered_ids(threshold: int) -> set[int]:
    """Memory IDs delivered at least `threshold` times across all sessions.

    Used by the per-file injection path to retire evergreen entries — a memory
    that has been served this often has had its chance; semantic retrieval
    layers (relevance-driven) are deliberately unaffected.
    """
    try:
        conn = get_ephemeral_conn()
        rows = conn.execute(
            "SELECT memory_id FROM delivery_counts WHERE count >= ?", (threshold,)
        ).fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def strip_memory_block(text: str) -> str:
    """Remove memory blocks from response text (both <memory> tags and [cm] link-defs)."""
    text = re.sub(r"<memory>.*?</memory>", "", text, flags=re.DOTALL)
    text = re.sub(r"^\[(?:cm|cairn-memory)\]:\s*#\s*'.*'$", "", text, flags=re.MULTILINE)
    return text.strip()


def format_entry(r: dict[str, Any]) -> str:
    """Format a memory entry as compact XML for context injection."""
    sim = r.get("similarity", 0)
    days = recency_days(r.get("updated_at", ""))
    reason = r.get("archived_reason")
    content = html.escape(str(r.get("content", "")), quote=True)
    if r.get("archived") or reason:
        reason = html.escape(str(reason or "unknown"), quote=True)
        return (
            f'  <entry id="{r["id"]}" superseded="true" reason="{reason}" days="{days}">'
            f'{content}</entry>'
        )
    # Annotate live org-index location status (DRIFT/MISSING). Imported lazily:
    # retrieval imports hook_helpers at module load, so a top-level import here
    # would be circular. Best-effort — falls back to no annotation on any error.
    loc = ""
    try:
        from hooks.retrieval import _loc_attr
        loc = _loc_attr(r["id"])
    except Exception:
        loc = ""
    return (
        f'  <entry id="{r["id"]}" days="{days}" sim="{sim:.2f}"{loc}>'
        f'{content}</entry>'
    )


def split_by_scope(results: list[dict[str, Any]], project: Optional[str]) -> tuple[list[dict], list[dict]]:
    """Split results into project-scoped and global lists."""
    project_results = [r for r in results if project and r.get("project") == project]
    global_results = [r for r in results if not project or r.get("project") != project]
    return project_results, global_results


def _relevance_prefilter(project_results, global_results, session_id):
    """Drop bucket-4 self-referential meta-memories before injection (gated by
    RELEVANCE_PREFILTER_ENABLED, default off). Corrections are never gated (spec).
    Each drop is audited via the relevance_prefilter_drop metric."""
    try:
        from cairn import config
        if not getattr(config, "RELEVANCE_PREFILTER_ENABLED", False):
            return project_results, global_results
        from cairn.relevance import is_self_referential_meta
    except Exception:
        return project_results, global_results

    def _keep(r):
        if r.get("type") == "correction":
            return True  # corrections surface ungated (spec A.1)
        if is_self_referential_meta(r):
            try:
                record_metric(session_id, "relevance_prefilter_drop",
                              f"id={r.get('id')} {str(r.get('content', ''))[:60]}")
            except Exception:
                pass
            return False
        return True

    return ([r for r in project_results if _keep(r)],
            [r for r in global_results if _keep(r)])


def _suppress_superseded_pairs(project_results, global_results, session_id=None):
    """Drop a superseded entry when the entry that superseded it is ALSO in this
    result set. Memory supersession is recorded in archived_reason as "...(by #N)"
    (consolidate.py), so we parse that id and drop the stale row only when #N is
    present — a lone superseded entry is kept (the deliberate negative-knowledge
    trail). Avoids spending a slot on a near-duplicate sitting next to its
    superseder."""
    all_rows = project_results + global_results
    if len(all_rows) < 2:
        return project_results, global_results
    present = {r.get("id") for r in all_rows}
    dropped = [0]

    def _keep(r):
        m = re.search(r"by #(\d+)", r.get("archived_reason") or "")
        if m and int(m.group(1)) in present:
            dropped[0] += 1
            return False
        return True

    pr = [r for r in project_results if _keep(r)]
    gr = [r for r in global_results if _keep(r)]
    if dropped[0] and session_id:
        record_metric(session_id, "superseded_pair_suppressed", None, dropped[0])
    return pr, gr


def build_context_xml(query: str, project: Optional[str], layer: str,
                      project_results: list[dict[str, Any]],
                      global_results: list[dict[str, Any]],
                      instruction: Optional[str] = None, *,
                      session_id: Optional[str] = None,
                      context_text: Optional[str] = None,
                      context_vec: Optional[bytes] = None) -> str:
    """Build a complete <cairn_context> XML block.

    When `session_id` is given (the semantic-match layers L1/L1.5), this also
    runs the read-side relevance machinery on the injected set: the bucket-4
    prefilter (gated) and a memory_deliveries log keyed by `context_text` (the
    cleaned recent-context window). Bootstrap/correction layers omit session_id
    so they stay ungated and unlogged."""
    # Correctness dedup (all layers): drop a superseded entry when its superseder
    # is already in the same result set — see _suppress_superseded_pairs.
    project_results, global_results = _suppress_superseded_pairs(
        project_results, global_results, session_id)
    if session_id:
        project_results, global_results = _relevance_prefilter(
            project_results, global_results, session_id)
    safe_query = query[:80].replace('"', '&quot;')
    lines = [f'<cairn_context query="{safe_query}" current_project="{project or "none"}" layer="{layer}">']
    if instruction:
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

    # Phase 1 instrument: log each injected (post-filter) memory to memory_deliveries.
    if session_id:
        try:
            from cairn import config
            if getattr(config, "RELEVANCE_LOGGING_ENABLED", True):
                from cairn.relevance import log_memory_deliveries
                cv = context_vec
                if cv is None and (context_text or query):
                    # Record the context embedding (the join key for empirical
                    # context-targeting). Daemon-only (allow_slow=False) so the hot
                    # prompt path stays fast; stays None if the daemon is down.
                    try:
                        emb = get_embedder()
                        if emb is not None:
                            _vec = emb.embed(context_text or query, allow_slow=False)
                            if _vec is not None:
                                cv = emb.to_blob(_vec)
                    except Exception:
                        pass
                log_memory_deliveries(project_results + global_results,
                                      session_id=session_id,
                                      context_text=context_text or query,
                                      context_vec=cv,
                                      layer=layer, project=project)
        except Exception:
            pass
    return "\n".join(lines)


# --- Central dedup gate ---

def strip_seen_entries(xml: str, session_id: str) -> Optional[str]:
    """Remove already-injected memory entries from XML context. Returns None if nothing remains.

    SQL-level retrieval now excludes already-seen IDs in retrieval.py (hybrid_search,
    _keyword_match_search), so this regex gate is belt-and-braces — should normally
    filter 0 entries. If it filters any, record a metric so the SQL-level gap is
    visible.
    """
    seen = load_injected_ids(session_id)
    if not seen:
        return xml

    stripped_count = [0]
    def _filter_entry(m: re.Match) -> str:
        entry_id = int(m.group(1))
        if entry_id in seen:
            stripped_count[0] += 1
            return ""
        return m.group(0)

    filtered = re.sub(
        r'[ \t]*<entry id="(\d+)"[^>]*>.*?</entry>\n?',
        _filter_entry, xml
    )
    if stripped_count[0]:
        record_metric(session_id, "strip_seen_leaked",
                      f"sql-gate missed {stripped_count[0]} entries",
                      stripped_count[0])

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


# --- proxy-aware delivery ----------------------------------------------------
def deliver_additional_context(session_id: str, hook_event_name: str, text: str) -> None:
    """Deliver injected context to the session.

    When the cairn proxy is active (CAIRN_PROXY_ENABLED), stage the payload to a
    sidecar so the proxy injects it into the request under the hood — the user
    never sees it. Otherwise emit the normal visible additionalContext JSON.
    Fail-open: any staging error falls back to visible delivery.
    """
    if not text:
        return
    try:
        from cairn import config
        if getattr(config, "PROXY_ENABLED", False):
            from cairn.proxy import sidecar
            sidecar.append_prompt_context(session_id, text)
            return
    except Exception as exc:  # fail-open to visible delivery
        log(f"deliver_additional_context staging failed, visible fallback: {exc}")
    print(json.dumps({
        "hookSpecificOutput": {"hookEventName": hook_event_name, "additionalContext": text}
    }))


def restore_stripped_cm(session_id: str, text: str) -> str:
    """If the proxy stripped the trailing [cm] block from this response, append
    the captured verbatim block back so the Stop hook's parse/enforcement path
    sees it exactly as if it had been inline. No-op if [cm] is already present
    or no matching capture exists. Fail-open."""
    if re.search(r'^\[(?:cm|cairn-memory)\]:', text, re.MULTILINE):
        return text
    try:
        import hashlib
        from cairn.proxy import sidecar
        rec = sidecar.lookup_capture_by_sha(
            session_id, hashlib.sha256(text.encode("utf-8")).hexdigest())
        if rec and rec.get("cm"):
            return text + rec["cm"]
    except Exception as exc:
        log(f"restore_stripped_cm failed open: {exc}")
    return text
