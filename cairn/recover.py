#!/usr/bin/env python3
"""Cairn DB recovery — repairs a corrupted cairn.db using the canonical schema.

Unlike ad-hoc row-copy scripts, this uses init_db.py as the source of truth for
the schema, so recovered databases are guaranteed structurally consistent (all
triggers, FTS5 shape, memories_vec, indexes). Row-copy scripts that skip the
trigger section leave the FTS5 table unsynced on subsequent writes, causing
recurring corruption.

Recovery strategy:
    1. Back up the corrupted DB
    2. Build a new DB via init_db.init() (canonical schema)
    3. Copy each readable table row-by-row from the corrupted DB
    4. For unreadable tables that contain reconstructible state:
       - sessions: rebuild from transcript files on disk
       - metrics/hook_state: start empty (operational state, not durable)
    5. Rebuild FTS5 index from memories via 'rebuild' command
    6. Verify integrity and report

Usage:
    python3 cairn/recover.py              # Recover the production DB
    python3 cairn/recover.py <path>       # Recover a specific DB file
"""

import glob
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime


CAIRN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(CAIRN_DIR, "cairn.db")
TABLES_TO_RECOVER = (
    "memories",
    "memory_history",
    "sessions",
    "metrics",
    "hook_state",
)


def _can_read_table(conn: sqlite3.Connection, table: str) -> bool:
    """Return True if the table can be fully iterated without error."""
    try:
        cur = conn.execute(f"SELECT * FROM {table}")
        for _ in cur:
            pass
        return True
    except sqlite3.DatabaseError:
        return False


def _copy_table(old: sqlite3.Connection, new: sqlite3.Connection, table: str) -> int:
    """Copy all rows from `table` in `old` to `new`. Returns count copied."""
    old.row_factory = sqlite3.Row
    try:
        rows = old.execute(f"SELECT * FROM {table}").fetchall()
    except sqlite3.DatabaseError as exc:
        print(f"  {table}: unreadable ({exc}) — skipping")
        return 0
    if not rows:
        print(f"  {table}: 0 rows")
        return 0
    cols = list(rows[0].keys())
    placeholders = ",".join("?" * len(cols))
    col_names = ",".join(cols)
    copied = 0
    for row in rows:
        try:
            new.execute(
                f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})",
                tuple(row),
            )
            copied += 1
        except sqlite3.Error:
            # Skip corrupted rows silently; count them in the delta
            pass
    new.commit()
    print(f"  {table}: {copied}/{len(rows)} rows")
    return copied


def _reconstruct_sessions_from_transcripts(new: sqlite3.Connection, known_session_ids: set[str]) -> int:
    """Recreate session rows from Claude Code transcript files on disk.

    Only creates rows for sessions that have at least one memory or metric
    pointing to them — avoids resurrecting abandoned sessions.
    """
    transcripts_dir = os.path.expanduser("~/.claude/projects")
    if not os.path.isdir(transcripts_dir):
        print("  sessions: ~/.claude/projects not found — skipping reconstruction")
        return 0

    # Build a map of session_id → transcript path
    transcript_map: dict[str, str] = {}
    for path in glob.glob(os.path.join(transcripts_dir, "*", "*.jsonl")):
        sid = os.path.basename(path).removesuffix(".jsonl")
        transcript_map[sid] = path

    created = 0
    for sid in known_session_ids:
        if not sid:
            continue
        path = transcript_map.get(sid, "")
        # Extract project from the encoded project dir in the transcript path
        project = None
        if path:
            m = re.search(r"/\.claude/projects/-([^/]+)/", path)
            if m:
                parts = m.group(1).split("-")
                project = parts[-1].lower() if parts else None
        try:
            new.execute(
                "INSERT OR IGNORE INTO sessions (session_id, project, transcript_path) VALUES (?, ?, ?)",
                (sid, project, path),
            )
            created += 1
        except sqlite3.Error:
            pass
    new.commit()
    print(f"  sessions: {created} reconstructed from transcripts")
    return created


def _rebuild_fts_and_vec(new: sqlite3.Connection) -> None:
    """Populate FTS5 index and vec index from the memories table."""
    try:
        new.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        new.commit()
        print("  memories_fts: rebuilt from memories")
    except sqlite3.Error as exc:
        print(f"  memories_fts: rebuild failed: {exc}")

    # memories_vec rebuild requires the sqlite-vec extension and re-encoding blobs
    try:
        import sqlite_vec
        new.enable_load_extension(True)
        sqlite_vec.load(new)
        # Copy from memories.embedding into memories_vec
        rows = new.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()
        count = 0
        for mem_id, blob in rows:
            try:
                new.execute(
                    "INSERT OR REPLACE INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (mem_id, blob),
                )
                count += 1
            except sqlite3.Error:
                pass
        new.commit()
        print(f"  memories_vec: {count} entries populated")
    except ImportError:
        print("  memories_vec: sqlite-vec not installed — brute-force search will be used")
    except sqlite3.Error as exc:
        print(f"  memories_vec: rebuild failed: {exc}")


def recover(db_path: str) -> bool:
    """Recover the DB at `db_path`. Returns True on success."""
    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} does not exist")
        return False

    print(f"=== Cairn DB recovery ===")
    print(f"Target: {db_path}")

    # 1. Backup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{db_path}.bak-recover-{timestamp}"
    shutil.copy2(db_path, backup_path)
    print(f"Backup: {backup_path}")

    # 2. Build a new canonical-schema DB
    new_path = f"{db_path}.recover-{timestamp}"
    if os.path.exists(new_path):
        os.remove(new_path)

    # Point init_db at the new path, import and run
    sys.path.insert(0, CAIRN_DIR)
    import init_db
    _original_db_path = init_db.DB_PATH
    try:
        init_db.DB_PATH = new_path
        init_db.init()
    finally:
        init_db.DB_PATH = _original_db_path
    print(f"Built new DB via init_db.init() at {new_path}")

    # 3. Copy readable tables
    old = sqlite3.connect(db_path)
    old.execute("PRAGMA busy_timeout=5000")
    new = sqlite3.connect(new_path)
    new.execute("PRAGMA busy_timeout=5000")

    print("\nCopying tables:")
    known_session_ids: set[str] = set()
    for table in TABLES_TO_RECOVER:
        _copy_table(old, new, table)

    # 4. Gather session IDs from memories and metrics for reconstruction
    for source_table in ("memories", "metrics"):
        try:
            for (sid,) in old.execute(f"SELECT DISTINCT session_id FROM {source_table} WHERE session_id IS NOT NULL"):
                if sid:
                    known_session_ids.add(sid)
        except sqlite3.DatabaseError:
            pass

    # 5. Reconstruct sessions from transcripts for any missing IDs
    existing_sids = {r[0] for r in new.execute("SELECT session_id FROM sessions").fetchall()}
    missing_sids = known_session_ids - existing_sids
    if missing_sids:
        print(f"\nReconstructing {len(missing_sids)} missing sessions from transcripts:")
        _reconstruct_sessions_from_transcripts(new, missing_sids)

    # 6. Rebuild derived tables
    print("\nRebuilding derived indexes:")
    _rebuild_fts_and_vec(new)

    # 7. Verify integrity
    print("\nVerification:")
    integrity = new.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity == "ok":
        print("  integrity: OK")
    else:
        print(f"  integrity: FAILED ({integrity[:100]})")
        new.close()
        old.close()
        return False

    triggers = {r[0] for r in new.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()}
    required = {"memories_ai", "memories_au", "memories_ad", "memories_version"}
    if required.issubset(triggers):
        print(f"  triggers: {len(triggers)} present (required {len(required)})")
    else:
        print(f"  triggers: MISSING {sorted(required - triggers)}")
        new.close()
        old.close()
        return False

    for table in TABLES_TO_RECOVER:
        try:
            c = new.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {c} rows")
        except sqlite3.Error as exc:
            print(f"  {table}: {exc}")

    new.close()
    old.close()

    # 8. Atomic swap — old DB → aside, new DB → production path
    aside_path = f"{db_path}.corrupt-{timestamp}"
    os.rename(db_path, aside_path)
    os.rename(new_path, db_path)
    print(f"\nSwap complete:")
    print(f"  corrupted DB moved to: {aside_path}")
    print(f"  recovered DB now at:   {db_path}")
    print(f"\nBackup retained at:    {backup_path}")
    return True


def main() -> int:
    db_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    if os.path.abspath(db_path) == os.path.abspath(DEFAULT_DB):
        # Safety check on production DB
        print("WARNING: recovering the production DB.")
        print("Stop the daemon and dashboard first, and make sure no Claude Code sessions are active.")
        print("Press Ctrl-C within 5 seconds to abort.")
        import time
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted.")
            return 1
    ok = recover(db_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
