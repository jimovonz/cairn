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


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        print("=== DRY RUN ===\n")
    backfill(dry_run=dry)
