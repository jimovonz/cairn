"""Phase 7 — one-time CLAUDE.md import.

Scans the user's CLAUDE.md (global + project-level) for first-person
preference statements and seeds calibration_rows with high confidence.
Designed to be idempotent: re-running on the same file does not produce
duplicates. Tracks state in `hook_state` under key=`calibration_md_import`,
session_id=`__global__`, value=JSON {sha256_of_file: True}.

Trigger paths (most are also CLI-callable):
- `cairn-calibration-import-claude-md ~/.claude/CLAUDE.md` — explicit
- one-time on install via install.sh (see Phase 7 spec bullet)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Optional

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


DB_PATH = os.environ.get(
    "CAIRN_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db"),
)
EPH_DB_PATH = os.environ.get(
    "CAIRN_EPHEMERAL_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db"),
)
GLOBAL_STATE_SESSION = "__global__"
IMPORT_STATE_KEY = "calibration_md_import"

# First-person preference markers we'll surface
PREFERENCE_PATTERNS = [
    re.compile(r"^I\s+prefer\s+", re.IGNORECASE),
    re.compile(r"^I\s+(?:like|want|expect|always|never)\s+", re.IGNORECASE),
    re.compile(r"^(?:Always|Never|Stop|Don'?t|Do\s+not)\s+", re.IGNORECASE),
    re.compile(r"^(?:My|Our)\s+(?:preference|style|convention)\s+is\s+",
               re.IGNORECASE),
    re.compile(r"^User\s+(?:prefers|likes|wants|expects)\s+", re.IGNORECASE),
]

# Anti-pattern: lines that look like docs / examples, not preferences
SKIP_MARKERS = (
    "example:", "e.g.", "for instance", "such as", "like this:",
    "```", "<!--", "-->", "---",
)


def _looks_like_preference(line: str) -> bool:
    stripped = line.strip().lstrip("-*").lstrip().lstrip("0123456789.").lstrip()
    if not stripped or len(stripped) < 10:
        return False
    low = stripped.lower()
    if any(low.startswith(m) for m in SKIP_MARKERS):
        return False
    return any(p.match(stripped) for p in PREFERENCE_PATTERNS)


def _normalise(text: str) -> str:
    """Strip leading bullet/list markers and trailing whitespace."""
    s = text.strip()
    s = re.sub(r"^[-*]\s+", "", s)
    s = re.sub(r"^\d+\.\s+", "", s)
    return s.strip()


def extract_preferences(text: str) -> list[str]:
    """Return the list of normalised preference lines from `text`."""
    out = []
    for line in text.splitlines():
        if _looks_like_preference(line):
            out.append(_normalise(line))
    return out


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_imported(sha: str, eph_path: Optional[str] = None) -> bool:
    path = eph_path or EPH_DB_PATH
    if not os.path.exists(path):
        return False
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        row = conn.execute(
            "SELECT value FROM hook_state WHERE session_id = ? AND key = ?",
            (GLOBAL_STATE_SESSION, IMPORT_STATE_KEY),
        ).fetchone()
        if not row or not row[0]:
            return False
        try:
            state = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return False
        return bool(state.get(sha))
    finally:
        conn.close()


def _mark_imported(sha: str, path: str,
                    eph_path: Optional[str] = None) -> None:
    eph = eph_path or EPH_DB_PATH
    conn = sqlite3.connect(eph)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        row = conn.execute(
            "SELECT value FROM hook_state WHERE session_id = ? AND key = ?",
            (GLOBAL_STATE_SESSION, IMPORT_STATE_KEY),
        ).fetchone()
        try:
            state = json.loads(row[0]) if row and row[0] else {}
        except (TypeError, json.JSONDecodeError):
            state = {}
        state[sha] = {"path": path, "imported_at": "CURRENT_TIMESTAMP"}
        conn.execute(
            "INSERT INTO hook_state (session_id, key, value, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(session_id, key) DO UPDATE SET "
            "value = excluded.value, updated_at = CURRENT_TIMESTAMP",
            (GLOBAL_STATE_SESSION, IMPORT_STATE_KEY, json.dumps(state)),
        )
        conn.commit()
    finally:
        conn.close()


def _existing_contents(db_path: Optional[str] = None) -> set[str]:
    path = db_path or DB_PATH
    if not os.path.exists(path):
        return set()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT content FROM calibration_rows WHERE archived_at IS NULL"
        ).fetchall()
        return {r[0].strip().lower() for r in rows if r[0]}
    finally:
        conn.close()


def import_file(path: str, *, force: bool = False,
                db_path: Optional[str] = None,
                eph_path: Optional[str] = None) -> dict:
    """Extract preferences from CLAUDE.md at `path`, insert as
    calibration_rows. Returns a report dict."""
    if not os.path.exists(path):
        return {"path": path, "error": "file not found", "imported": 0}
    sha = _file_sha(path)
    if not force and _already_imported(sha, eph_path=eph_path):
        return {"path": path, "sha": sha, "skipped": "already-imported",
                "imported": 0}
    with open(path) as f:
        text = f.read()
    prefs = extract_preferences(text)
    existing = _existing_contents(db_path=db_path)
    durable = db_path or DB_PATH

    try:
        from cairn import embeddings as emb
    except Exception:
        emb = None

    conn = sqlite3.connect(durable)
    conn.execute("PRAGMA busy_timeout=5000")
    inserted = []
    try:
        for pref in prefs:
            if pref.strip().lower() in existing:
                continue
            blob = None
            if emb is not None:
                try:
                    vec = emb.embed(pref)
                    if vec is not None:
                        blob = emb.to_blob(vec)
                except Exception:
                    blob = None
            # High confidence — these are user-authored CLAUDE.md statements
            conn.execute(
                "INSERT INTO calibration_rows (content, source, confidence, "
                "pinned, layer, embedding) VALUES (?, 'explicit', 0.90, 1, "
                "'general', ?)",
                (pref, blob),
            )
            rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            inserted.append(rid)
            existing.add(pref.strip().lower())
        conn.commit()
    finally:
        conn.close()
    _mark_imported(sha, path, eph_path=eph_path)
    return {"path": path, "sha": sha, "imported": len(inserted),
            "row_ids": inserted, "candidates_found": len(prefs)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cairn-calibration-import-claude-md",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("path", nargs="?",
                    default=os.path.expanduser("~/.claude/CLAUDE.md"))
    p.add_argument("--force", action="store_true",
                    help="Re-import even if file SHA already processed")
    args = p.parse_args(argv)
    report = import_file(args.path, force=args.force)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
