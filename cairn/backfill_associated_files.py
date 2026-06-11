#!/usr/bin/env python3
"""Backfill associated_files for existing memories by scanning session transcripts.

For each memory with associated_files=NULL, finds the session's transcript,
runs extract_associated_files over the full transcript (lookback=0), and
updates all memories in that session with the file list.
"""

import json
import os
import sys

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hooks.storage import extract_associated_files


def backfill(dry_run=False):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")

    rows = conn.execute("""
        SELECT DISTINCT m.session_id, s.transcript_path
        FROM memories m
        LEFT JOIN sessions s ON m.session_id = s.session_id
        WHERE m.associated_files IS NULL AND m.session_id IS NOT NULL
        ORDER BY m.session_id
    """).fetchall()

    print(f"Sessions with NULL associated_files: {len(rows)}")

    updated = 0
    no_transcript = 0
    no_files = 0

    for session_id, transcript_path in rows:
        if not transcript_path or not os.path.exists(transcript_path):
            no_transcript += 1
            continue

        files = extract_associated_files(transcript_path, lookback=0)

        if not files:
            no_files += 1
            continue

        files_json = json.dumps(files)
        mem_count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE session_id = ? AND associated_files IS NULL",
            (session_id,)
        ).fetchone()[0]

        if dry_run:
            print(f"  [DRY] {session_id}: {mem_count} memories -> {len(files)} files")
        else:
            conn.execute(
                "UPDATE memories SET associated_files = ? WHERE session_id = ? AND associated_files IS NULL",
                (files_json, session_id)
            )
            updated += mem_count

    if not dry_run:
        conn.commit()

    conn.close()

    total_sessions = len(rows)
    scanned = total_sessions - no_transcript
    print(f"\nDone. Scanned {scanned}/{total_sessions} sessions ({no_transcript} missing transcripts, {no_files} with no files).")
    if dry_run:
        print("(dry run — no changes written)")
    else:
        print(f"Updated {updated} memories with associated_files.")


def repair(dry_run=False):
    """One-shot hygiene repair of existing associated_files rows.

    Applies the same write-time rules extract_associated_files now enforces:
    junk paths dropped (venvs, logs, DBs, temp output) and the list capped at
    ASSOC_FILES_MAX. Existing rows were written by full-transcript scans
    (avg 25.9 files/memory, max 223) — that bloat is what made generic
    basenames resolve to the same cross-project set everywhere.
    """
    from hooks.storage import _is_junk_path
    from cairn.config import ASSOC_FILES_MAX

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")

    rows = conn.execute(
        "SELECT id, associated_files FROM memories "
        "WHERE associated_files IS NOT NULL AND deleted_at IS NULL"
    ).fetchall()

    changed = 0
    cleared = 0
    before_total = 0
    after_total = 0
    for mid, files_json in rows:
        try:
            files = json.loads(files_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(files, list):
            continue
        before_total += len(files)
        cleaned = [f for f in files if isinstance(f, str) and f and not _is_junk_path(f)]
        cleaned = cleaned[:ASSOC_FILES_MAX]
        after_total += len(cleaned)
        if cleaned == files:
            continue
        changed += 1
        if not cleaned:
            cleared += 1
        if not dry_run:
            conn.execute(
                "UPDATE memories SET associated_files = ? WHERE id = ?",
                (json.dumps(cleaned) if cleaned else None, mid)
            )

    if not dry_run:
        conn.commit()
    conn.close()

    n = len(rows) or 1
    print(f"Repair: {changed}/{len(rows)} rows changed ({cleared} cleared to NULL)")
    print(f"Avg files/row: {before_total/n:.1f} -> {after_total/n:.1f}")
    if dry_run:
        print("(dry run — no changes written)")


def seed_delivery_counts(dry_run=False):
    """Seed delivery_counts (ephemeral DB) from historical layer_delivery metrics
    so over-delivery dampening starts informed rather than from zero."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from hooks.hook_helpers import get_ephemeral_conn

    conn = get_ephemeral_conn()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS delivery_counts ("
        "memory_id INTEGER PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, "
        "last_delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    rows = conn.execute(
        "SELECT detail FROM metrics WHERE event = 'layer_delivery'"
    ).fetchall()
    counts: dict[int, int] = {}
    for (detail,) in rows:
        try:
            ids = json.loads(detail).get("ids") or []
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
        for i in ids:
            if isinstance(i, int):
                counts[i] = counts.get(i, 0) + 1
    if not dry_run and counts:
        conn.executemany(
            "INSERT INTO delivery_counts (memory_id, count) VALUES (?, ?) "
            "ON CONFLICT(memory_id) DO UPDATE SET count = MAX(count, excluded.count)",
            list(counts.items())
        )
        conn.commit()
    conn.close()
    top = sorted(counts.values(), reverse=True)[:5]
    print(f"Seeded delivery_counts: {len(counts)} memories from {len(rows)} events (top counts: {top})")
    if dry_run:
        print("(dry run — no changes written)")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        print("=== DRY RUN ===\n")
    if "--repair" in sys.argv:
        repair(dry_run=dry)
        seed_delivery_counts(dry_run=dry)
    else:
        backfill(dry_run=dry)
