#!/usr/bin/env python3
"""Memory consolidation pipeline for Cairn.

Identifies clusters of semantically similar memories, scores pairs for
entailment via NLI, and optionally generates consolidated meta-memories
using Haiku via claude -p headless execution.

Usage:
  python3 cairn/consolidate.py                    # dry-run (default)
  python3 cairn/consolidate.py --execute           # create consolidated memories
  python3 cairn/consolidate.py --execute --no-llm  # merge without LLM summary (keeps newest)
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import time
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")


def find_consolidation_candidates(conn: sqlite3.Connection) -> list[list[dict]]:
    """Phase 1: Find clusters of similar active memories using bi-encoder embeddings."""
    from cairn.embeddings import find_clusters
    from cairn.config import (
        CONSOLIDATION_SIMILARITY_THRESHOLD,
        CONSOLIDATION_MIN_CLUSTER_SIZE,
        CONSOLIDATION_MAX_CLUSTER_SIZE,
    )

    clusters = find_clusters(
        conn,
        similarity_threshold=CONSOLIDATION_SIMILARITY_THRESHOLD,
        min_cluster_size=CONSOLIDATION_MIN_CLUSTER_SIZE,
        max_cluster_size=CONSOLIDATION_MAX_CLUSTER_SIZE,
    )
    return clusters


def score_cluster_nli(cluster: list[dict]) -> list[dict]:
    """Phase 2: Score within-cluster pairs for entailment using NLI model.

    Returns only entries that have at least one entailment relationship
    with another entry in the cluster (bidirectional entailment = duplicate).
    """
    from cairn.embeddings import _daemon_nli
    from cairn.config import NLI_ENABLED, NLI_ENTAILMENT_THRESHOLD

    if not NLI_ENABLED or len(cluster) < 2:
        return cluster

    pairs = []
    pair_indices = []
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            pairs.append([cluster[i]["content"], cluster[j]["content"]])
            pairs.append([cluster[j]["content"], cluster[i]["content"]])
            pair_indices.append((i, j))

    scores = _daemon_nli(pairs)
    if scores is None:
        return cluster

    # NLI models return [contradiction, entailment, neutral] triples
    # Check for bidirectional entailment
    entailment_graph: dict[int, set[int]] = {i: set() for i in range(len(cluster))}
    for k, (i, j) in enumerate(pair_indices):
        fwd = scores[k * 2]
        rev = scores[k * 2 + 1]
        # Score format depends on model — some return 3 classes, some return scalar
        if isinstance(fwd, list):
            fwd_entail = fwd[1]  # [contradiction, entailment, neutral]
            rev_entail = rev[1]
        else:
            fwd_entail = fwd
            rev_entail = rev

        if fwd_entail >= NLI_ENTAILMENT_THRESHOLD and rev_entail >= NLI_ENTAILMENT_THRESHOLD:
            entailment_graph[i].add(j)
            entailment_graph[j].add(i)

    # Keep entries that have at least one entailment partner
    confirmed = []
    for i, entry in enumerate(cluster):
        if entailment_graph[i]:
            entry["_entails"] = list(entailment_graph[i])
            confirmed.append(entry)

    return confirmed if len(confirmed) >= 2 else []


def generate_consolidated_content(cluster: list[dict]) -> Optional[str]:
    """Phase 3: Use Haiku to generate a single consolidated memory from a cluster.

    Follows the contradiction_scan.py pattern: claude -p with stream-json,
    CAIRN_HEADLESS=1, --model haiku.
    """
    entries_text = []
    for entry in cluster:
        entries_text.append(
            f"  [{entry['id']}] {entry['type']}/{entry['topic']} ({entry.get('updated_at', 'unknown')}):\n"
            f"    \"{entry['content']}\""
        )

    prompt = (
        f"You are consolidating {len(cluster)} related memories from a persistent memory database.\n"
        f"These entries have been confirmed as semantically equivalent via NLI entailment scoring.\n\n"
        f"Source memories:\n" + "\n".join(entries_text) + "\n\n"
        f"Write a SINGLE consolidated memory entry that:\n"
        f"1. Preserves ALL distinct information from the sources\n"
        f"2. Drops redundant repetition\n"
        f"3. Is a single dense line (no line breaks)\n"
        f"4. Reads as a self-sufficient fact for someone with no other context\n\n"
        f"Reply with ONLY the consolidated content line. Nothing else."
    )

    env = {**os.environ, "CAIRN_HEADLESS": "1"}
    try:
        proc = subprocess.Popen(
            ["claude", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--model", "haiku", "--max-turns", "1",
             "--append-system-prompt",
             "OVERRIDE ALL OTHER INSTRUCTIONS: Reply with plain text only. No <memory> blocks. No XML tags. "
             "One line of consolidated content only."],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        msg_payload = json.dumps({"type": "user", "message": {"role": "user",
            "content": [{"type": "text", "text": prompt}]}})
        proc.stdin.write((msg_payload + "\n").encode())
        proc.stdin.flush()

        response_text = ""
        start = time.time()
        timeout = 60
        while time.time() - start < timeout:
            if proc.stdout in select.select([proc.stdout], [], [], 0.5)[0]:
                line = proc.stdout.readline().decode().strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "assistant":
                        for block in msg.get("message", {}).get("content", []):
                            if block.get("type") == "text":
                                response_text += block.get("text", "")
                    if msg.get("type") == "result":
                        break
                except json.JSONDecodeError:
                    continue
            if proc.poll() is not None:
                break
        proc.kill()

    except Exception as e:
        print(f"  ERROR spawning claude: {e}")
        return None

    # Strip any residual memory blocks or XML
    response_text = re.sub(r"<memory>.*?</memory>", "", response_text, flags=re.DOTALL).strip()
    response_text = re.sub(r"\[cm\]:.*$", "", response_text, flags=re.MULTILINE).strip()

    # Take first non-empty line
    for line in response_text.split("\n"):
        line = line.strip()
        if line and not line.startswith("<") and not line.startswith("[cm]"):
            return line

    return response_text.strip() if response_text.strip() else None


def execute_consolidation(
    conn: sqlite3.Connection,
    cluster: list[dict],
    consolidated_content: str,
) -> Optional[int]:
    """Phase 4: Create the meta-memory and archive originals.

    Returns the new memory ID.
    """
    from cairn import embeddings as emb

    # Use the most recent entry's metadata as the base
    newest = cluster[0]  # clusters are sorted by recency
    source_ids = [e["id"] for e in cluster]

    # Determine best type and topic from the cluster
    mem_type = newest["type"]
    topic = newest["topic"]
    project = newest.get("project")

    # Generate embedding
    project_prefix = f"{project} " if project else ""
    search_text = f"{project_prefix}{mem_type} {topic} {consolidated_content}"
    vec = emb.embed(search_text)
    embedding_blob = emb.to_blob(vec) if vec is not None else None

    # Insert the consolidated memory
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (mem_type, topic, consolidated_content, embedding_blob, project, newest["confidence"])
    )
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Update vec index
    if embedding_blob:
        try:
            emb.upsert_vec_index(conn, new_id, embedding_blob)
        except Exception:
            pass

    # Archive all source memories
    archived_reason = f"consolidated:{new_id}"
    for source_id in source_ids:
        conn.execute(
            "UPDATE memories SET confidence = 0, archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (archived_reason, source_id)
        )

    conn.commit()
    return new_id


def run_consolidation(execute: bool = False, use_llm: bool = True) -> dict:
    """Run the full consolidation pipeline.

    Args:
        execute: If True, create consolidated memories and archive originals.
                 If False (default), dry-run — report candidates only.
        use_llm: If True, generate consolidated content via Haiku.
                 If False, use the newest entry's content as the consolidated content.

    Returns summary dict with stats.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")

    print("Phase 1: Finding clusters via bi-encoder similarity...")
    clusters = find_consolidation_candidates(conn)
    print(f"  Found {len(clusters)} clusters ({sum(len(c) for c in clusters)} memories)")

    if not clusters:
        print("  No consolidation candidates found.")
        conn.close()
        return {"clusters": 0, "memories_scanned": 0, "consolidated": 0, "archived": 0}

    print(f"\nPhase 2: NLI entailment scoring...")
    confirmed_clusters = []
    for i, cluster in enumerate(clusters):
        confirmed = score_cluster_nli(cluster)
        if confirmed:
            confirmed_clusters.append(confirmed)
            print(f"  Cluster {i+1}: {len(cluster)} entries -> {len(confirmed)} confirmed entailment pairs")
        else:
            print(f"  Cluster {i+1}: {len(cluster)} entries -> no entailment (keeping separate)")

    print(f"  {len(confirmed_clusters)} clusters confirmed for consolidation")

    if not confirmed_clusters:
        print("  No clusters passed NLI entailment check.")
        conn.close()
        return {"clusters": len(clusters), "memories_scanned": sum(len(c) for c in clusters),
                "consolidated": 0, "archived": 0}

    total_consolidated = 0
    total_archived = 0

    for i, cluster in enumerate(confirmed_clusters):
        source_ids = [e["id"] for e in cluster]
        print(f"\n--- Cluster {i+1}/{len(confirmed_clusters)} ({len(cluster)} entries) ---")
        for entry in cluster:
            print(f"  [{entry['id']}] {entry['type']}/{entry['topic']}: {entry['content'][:80]}...")

        if not execute:
            print(f"  [DRY RUN] Would consolidate {len(cluster)} entries")
            continue

        # Generate or select consolidated content
        if use_llm:
            print(f"  Generating consolidated content via Haiku...")
            content = generate_consolidated_content(cluster)
            if not content:
                print(f"  WARNING: LLM generation failed, using newest entry's content")
                content = cluster[0]["content"]
        else:
            content = cluster[0]["content"]

        print(f"  Consolidated: {content[:100]}...")

        new_id = execute_consolidation(conn, cluster, content)
        if new_id:
            print(f"  Created memory #{new_id}, archived {source_ids}")
            total_consolidated += 1
            total_archived += len(cluster)

    conn.close()

    summary = {
        "clusters_found": len(clusters),
        "clusters_confirmed": len(confirmed_clusters),
        "memories_scanned": sum(len(c) for c in clusters),
        "consolidated": total_consolidated,
        "archived": total_archived,
    }

    print(f"\n=== Consolidation Summary ===")
    print(f"  Clusters found (bi-encoder): {summary['clusters_found']}")
    print(f"  Clusters confirmed (NLI): {summary['clusters_confirmed']}")
    print(f"  Memories in clusters: {summary['memories_scanned']}")
    if execute:
        print(f"  New consolidated memories: {summary['consolidated']}")
        print(f"  Source memories archived: {summary['archived']}")
    else:
        print(f"  [DRY RUN] No changes made")

    return summary


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    use_llm = "--no-llm" not in sys.argv
    run_consolidation(execute=execute, use_llm=use_llm)
