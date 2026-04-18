#!/usr/bin/env python3
"""Backfill ingested memories with provenance data.

Fixes memories from older ingestion runs that are missing:
- Session registration (synthetic session in sessions table)
- Local path in source_ref
- associated_files (if source_files were not captured)

Usage:
    python3 cairn/backfill_ingestion.py                    # dry-run
    python3 cairn/backfill_ingestion.py --execute           # apply changes
    python3 cairn/backfill_ingestion.py --project gnss-sdr  # specific project only
"""

import argparse
import json
import os
import sqlite3
import sys

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")


def backfill(db_path=DB_PATH, execute=False, project=None):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")

    # Find all ingestion sessions
    query = "SELECT DISTINCT session_id, project FROM memories WHERE session_id LIKE 'ingest-%'"
    params = []
    if project:
        query += " AND project = ?"
        params.append(project)
    ingest_sessions = conn.execute(query, params).fetchall()

    if not ingest_sessions:
        print("No ingestion sessions found.")
        conn.close()
        return

    print(f"Found {len(ingest_sessions)} ingestion session(s)\n")

    total_sessions_fixed = 0
    total_sourceref_fixed = 0
    total_path_added = 0

    for sess in ingest_sessions:
        sid = sess["session_id"]
        proj = sess["project"]
        print(f"Session: {sid} (project: {proj})")

        # 1. Register session if missing
        existing = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
        if not existing:
            # Try to find repo path from source_ref
            sr_row = conn.execute(
                "SELECT source_ref FROM memories WHERE session_id = ? AND source_ref IS NOT NULL LIMIT 1",
                (sid,)
            ).fetchone()
            repo_path = ""
            if sr_row and sr_row["source_ref"]:
                try:
                    sr = json.loads(sr_row["source_ref"])
                    repo_path = sr.get("path", "")
                except Exception:
                    pass
            print(f"  [fix] Register session (transcript_path={repo_path})")
            if execute:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, project, transcript_path, started_at) "
                    "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                    (sid, proj, repo_path),
                )
            total_sessions_fixed += 1

        # 2. Fix source_ref — add local path if missing
        memories = conn.execute(
            "SELECT id, source_ref FROM memories WHERE session_id = ? AND source_ref IS NOT NULL",
            (sid,)
        ).fetchall()

        for mem in memories:
            try:
                sr = json.loads(mem["source_ref"])
            except Exception:
                continue

            changed = False

            # Add path if missing but we can infer it from repo URL
            if not sr.get("path"):
                repo_url = sr.get("repo", "")
                if repo_url:
                    # Try common locations
                    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
                    for base in [os.path.expanduser("~/Projects"), os.path.expanduser("~/repos"),
                                 os.path.expanduser("~"), "/opt"]:
                        candidate = os.path.join(base, repo_name)
                        if os.path.isdir(candidate):
                            sr["path"] = candidate
                            changed = True
                            break

            if changed:
                print(f"  [fix] Memory {mem['id']}: add path={sr.get('path')}")
                total_path_added += 1
                if execute:
                    conn.execute(
                        "UPDATE memories SET source_ref = ? WHERE id = ?",
                        (json.dumps(sr), mem["id"]),
                    )
                total_sourceref_fixed += 1

        # 3. Count memories without associated_files
        no_assoc = conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE session_id = ? AND (associated_files IS NULL OR associated_files = '')",
            (sid,)
        ).fetchone()["c"]
        total = len(memories)
        if no_assoc > 0:
            print(f"  {no_assoc}/{total} memories have no associated_files (requires re-ingestion with --distill)")

    if execute:
        conn.commit()

    conn.close()

    print(f"\nSummary:")
    print(f"  Sessions registered: {total_sessions_fixed}")
    print(f"  Source refs with path added: {total_path_added}")
    if not execute:
        print(f"\n  DRY RUN — no changes applied. Use --execute to apply.")


def main():
    parser = argparse.ArgumentParser(description="Backfill ingested memory provenance")
    parser.add_argument("--execute", action="store_true", help="Apply changes (default: dry-run)")
    parser.add_argument("--project", help="Only backfill specific project")
    parser.add_argument("--db", default=DB_PATH, help="Database path")
    args = parser.parse_args()
    backfill(db_path=args.db, execute=args.execute, project=args.project)


if __name__ == "__main__":
    main()
