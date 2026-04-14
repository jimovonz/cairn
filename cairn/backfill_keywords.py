#!/usr/bin/env python3
"""Backfill keywords for existing memories by re-parsing session transcripts.

Scans transcript JSONL files for <memory> blocks, extracts keywords,
and updates memories that have keywords=NULL.
"""

import json
import os
import re
import sys

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")

# Import the parser to reuse its logic
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hooks.parser import parse_memory_block


def extract_memory_blocks_from_transcript(transcript_path):
    """Parse a transcript JSONL and yield (type, topic, content, keywords) tuples."""
    if not os.path.exists(transcript_path):
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") != "assistant":
                continue

            msg = entry.get("message", {})
            if not isinstance(msg, dict):
                continue

            text = ""
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    text = c.get("text", "")
                    break

            if "<memory>" not in text:
                continue

            parsed = parse_memory_block(text)
            if not parsed.entries:
                continue

            # Block-level keywords (verbose format)
            block_keywords = parsed.keywords

            for entry_dict in parsed.entries:
                mem_type = entry_dict.get("type", "")
                topic = entry_dict.get("topic", "")
                content = entry_dict.get("content", "")
                # Per-entry keywords (compact) or block-level (verbose)
                keywords = entry_dict.get("keywords", block_keywords)
                if keywords:
                    yield (mem_type, topic, content, keywords)


def backfill(dry_run=False):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")

    # Get all memories without keywords, grouped by session
    rows = conn.execute("""
        SELECT m.id, m.type, m.topic, m.content, m.session_id, s.transcript_path
        FROM memories m
        LEFT JOIN sessions s ON m.session_id = s.session_id
        WHERE m.keywords IS NULL AND m.session_id IS NOT NULL
        ORDER BY m.session_id
    """).fetchall()

    print(f"Memories without keywords: {len(rows)}")

    # Group by session for efficiency
    by_session = {}
    for row in rows:
        mem_id, mem_type, topic, content, session_id, transcript_path = row
        if session_id not in by_session:
            by_session[session_id] = {"transcript": transcript_path, "memories": []}
        by_session[session_id]["memories"].append({
            "id": mem_id, "type": mem_type, "topic": topic, "content": content
        })

    print(f"Sessions to scan: {len(by_session)}")

    updated = 0
    scanned = 0
    no_transcript = 0

    for session_id, data in by_session.items():
        transcript_path = data["transcript"]
        if not transcript_path or not os.path.exists(transcript_path):
            no_transcript += 1
            continue

        scanned += 1
        # Extract all memory blocks from this transcript
        transcript_entries = list(extract_memory_blocks_from_transcript(transcript_path))
        if not transcript_entries:
            continue

        # Build lookup: (type, topic) -> keywords
        # Use the last occurrence (most recent) for each type+topic
        kw_lookup = {}
        for mem_type, topic, content, keywords in transcript_entries:
            kw_lookup[(mem_type, topic)] = keywords

        # Match and update
        for mem in data["memories"]:
            keywords = kw_lookup.get((mem["type"], mem["topic"]))
            if keywords:
                keywords_csv = ",".join(keywords)
                if dry_run:
                    print(f"  [DRY] #{mem['id']} {mem['type']}/{mem['topic']} -> {keywords_csv}")
                else:
                    conn.execute(
                        "UPDATE memories SET keywords = ? WHERE id = ?",
                        (keywords_csv, mem["id"])
                    )
                updated += 1

    if not dry_run:
        conn.commit()

    conn.close()
    print(f"\nDone. Scanned {scanned} transcripts ({no_transcript} missing).")
    print(f"Updated {updated} memories with keywords.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        print("=== DRY RUN ===\n")
    backfill(dry_run=dry)
