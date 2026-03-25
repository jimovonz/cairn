#!/usr/bin/env python3
"""Scan the memory database for contradictory memories.

Finds pairs of active memories that are semantically similar but contain
negation mismatches — indicating one likely supersedes the other.

Usage:
  python3 contradiction_scan.py              # Report only
  python3 contradiction_scan.py --annotate   # Annotate older memory in each pair
  python3 contradiction_scan.py --since 7d   # Only scan memories from last 7 days
"""

import sys
import os
import sqlite3
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

from hook_helpers import DB_PATH
from storage import _has_negation_mismatch


def parse_since(value: str) -> datetime:
    now = datetime.now()
    if value.endswith("d"):
        return now - timedelta(days=int(value[:-1]))
    elif value.endswith("w"):
        return now - timedelta(weeks=int(value[:-1]))
    elif value.endswith("m"):
        return now - timedelta(days=int(value[:-1]) * 30)
    return datetime.fromisoformat(value)


def scan(since: str | None = None, annotate: bool = False) -> list[dict]:
    try:
        import embeddings as emb
    except ImportError:
        print("Cannot import embeddings module")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 5000")

    # Load all active memories with embeddings
    where = "WHERE embedding IS NOT NULL AND archived_reason IS NULL"
    params: tuple = ()
    if since:
        since_dt = parse_since(since)
        where += " AND created_at >= ?"
        params = (since_dt.strftime("%Y-%m-%d %H:%M:%S"),)

    rows = conn.execute(
        f"SELECT id, type, topic, content, embedding, created_at FROM memories {where} ORDER BY id",
        params
    ).fetchall()

    print(f"Scanning {len(rows)} active memories...")

    # Build vectors
    memories = []
    for row in rows:
        vec = emb.from_blob(row[4])
        memories.append({
            "id": row[0], "type": row[1], "topic": row[2],
            "content": row[3], "vec": vec, "created_at": row[5]
        })

    # Compare pairs — two tiers:
    # 1. Same type+topic: high confidence, lower similarity bar (0.50)
    # 2. Cross-topic: must be very similar (0.80) to avoid noise from incidental negation words
    contradictions = []
    seen = set()

    # Build topic index for fast same-topic lookup
    by_topic: dict[tuple[str, str], list[int]] = {}
    for i, m in enumerate(memories):
        key = (m["type"], m["topic"])
        by_topic.setdefault(key, []).append(i)

    # Tier 1: Same type+topic pairs
    for key, indices in by_topic.items():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                m1, m2 = memories[indices[a]], memories[indices[b]]
                sim = float(emb.cosine_similarity(m1["vec"], m2["vec"]))
                if sim >= 0.50 and _has_negation_mismatch(m1["content"], m2["content"]):
                    older, newer = (m1, m2) if m1["id"] < m2["id"] else (m2, m1)
                    pair_key = (older["id"], newer["id"])
                    if pair_key not in seen:
                        seen.add(pair_key)
                        contradictions.append({
                            "older": older, "newer": newer,
                            "similarity": sim, "same_topic": True
                        })

    # Tier 2: Cross-topic — only very high similarity
    for i, m1 in enumerate(memories):
        for j, m2 in enumerate(memories):
            if j <= i:
                continue
            if m1["type"] == m2["type"] and m1["topic"] == m2["topic"]:
                continue  # Already handled in tier 1
            pair_key = (min(m1["id"], m2["id"]), max(m1["id"], m2["id"]))
            if pair_key in seen:
                continue

            sim = float(emb.cosine_similarity(m1["vec"], m2["vec"]))
            if sim >= 0.80 and _has_negation_mismatch(m1["content"], m2["content"]):
                seen.add(pair_key)
                older, newer = (m1, m2) if m1["id"] < m2["id"] else (m2, m1)
                contradictions.append({
                    "older": older, "newer": newer,
                    "similarity": sim, "same_topic": False
                })

    # Report
    if not contradictions:
        print("No contradictions found.")
        return []

    print(f"\nFound {len(contradictions)} potential contradictions:\n")

    for c in contradictions:
        o, n = c["older"], c["newer"]
        tag = " [SAME TOPIC]" if c.get("same_topic") else ""
        print(f"  #{o['id']} ({o['type']}/{o['topic']}) vs #{n['id']} ({n['type']}/{n['topic']})  sim={c['similarity']:.2f}{tag}")
        print(f"    OLD: {o['content'][:120]}")
        print(f"    NEW: {n['content'][:120]}")

        if annotate:
            conn.execute(
                "UPDATE memories SET archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (f"contradicted by #{n['id']}: {n['content'][:200]}", o["id"])
            )
            print(f"    → Annotated #{o['id']} as superseded by #{n['id']}")
        print()

    if annotate:
        conn.commit()
        print(f"Annotated {len(contradictions)} memories.")

    conn.close()
    return contradictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan for contradictory memories")
    parser.add_argument("--annotate", action="store_true", help="Annotate older memory in each contradictory pair")
    parser.add_argument("--since", type=str, help="Only scan memories from this date (ISO, 7d, 2w, 1m)")
    args = parser.parse_args()

    scan(since=args.since, annotate=args.annotate)
