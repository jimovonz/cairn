#!/usr/bin/env python3
"""Memory consolidation and contradiction detection pipeline for Cairn.

Two modes sharing the same NLI infrastructure:

1. Consolidation (default): Find and merge duplicate memories
   python3 cairn/consolidate.py                    # dry-run
   python3 cairn/consolidate.py --execute           # create consolidated memories

2. Contradiction detection: Find unflagged negations and resolve them
   python3 cairn/consolidate.py --contradictions                # dry-run
   python3 cairn/consolidate.py --contradictions --execute      # auto-archive superseded
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


# ============================================================
# Contradiction detection — find unflagged negations via NLI
# ============================================================


def find_contradiction_pairs(conn: sqlite3.Connection) -> list[dict]:
    """Find candidate pairs of memories that may contradict each other.

    Uses bi-encoder similarity at a lower threshold than consolidation (0.55 vs 0.85)
    since contradicting memories may use different phrasing. Returns pairs sorted by
    similarity descending.
    """
    from cairn.embeddings import from_blob, cosine_similarity
    from cairn.config import CONTRADICTION_SIMILARITY_THRESHOLD, CONTRADICTION_MAX_PAIRS

    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, created_at, project, confidence "
        "FROM memories WHERE embedding IS NOT NULL AND (archived_reason IS NULL OR archived_reason = '')"
    ).fetchall()

    entries = []
    for row in rows:
        vec = from_blob(row[4])
        entries.append({
            "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
            "vec": vec, "created_at": row[5], "project": row[6], "confidence": row[7] or 0.7,
        })

    # Build pairs in two tiers:
    # Tier 1 (always included): Same type+topic pairs — highest signal for contradictions
    # Tier 2 (fill remaining budget): Cross-topic pairs, spread across similarity bands
    same_topic_pairs = []
    cross_topic_pairs = []

    # Build topic index for fast same-topic lookup
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, e in enumerate(entries):
        key = (e["type"], e["topic"])
        by_key.setdefault(key, []).append(i)

    # Tier 1: Same type+topic pairs
    for key, indices in by_key.items():
        if len(indices) < 2:
            continue
        for a_idx in range(len(indices)):
            for b_idx in range(a_idx + 1, len(indices)):
                a, b = entries[indices[a_idx]], entries[indices[b_idx]]
                sim = cosine_similarity(a["vec"], b["vec"])
                if sim >= CONTRADICTION_SIMILARITY_THRESHOLD:
                    older, newer = (a, b) if a["id"] < b["id"] else (b, a)
                    same_topic_pairs.append({
                        "older": older, "newer": newer,
                        "similarity": float(sim), "same_topic": True,
                    })

    # Tier 2: Cross-topic pairs — sample across similarity bands to catch
    # contradictions at lower similarity (not just near-duplicates at 0.95+)
    seen_ids = {(p["older"]["id"], p["newer"]["id"]) for p in same_topic_pairs}
    remaining_budget = CONTRADICTION_MAX_PAIRS - len(same_topic_pairs)

    if remaining_budget > 0:
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                if a["type"] == b["type"] and a["topic"] == b["topic"]:
                    continue
                pair_key = (min(a["id"], b["id"]), max(a["id"], b["id"]))
                if pair_key in seen_ids:
                    continue
                sim = cosine_similarity(a["vec"], b["vec"])
                if sim >= CONTRADICTION_SIMILARITY_THRESHOLD:
                    older, newer = (a, b) if a["id"] < b["id"] else (b, a)
                    cross_topic_pairs.append({
                        "older": older, "newer": newer,
                        "similarity": float(sim), "same_topic": False,
                    })

        # Sort cross-topic by similarity but sample evenly across bands
        cross_topic_pairs.sort(key=lambda p: p["similarity"], reverse=True)
        if len(cross_topic_pairs) > remaining_budget:
            # Take every Nth pair to spread across similarity range
            step = max(1, len(cross_topic_pairs) // remaining_budget)
            cross_topic_pairs = cross_topic_pairs[::step][:remaining_budget]

    all_pairs = same_topic_pairs + cross_topic_pairs
    all_pairs.sort(key=lambda p: (not p["same_topic"], -p["similarity"]))
    return all_pairs[:CONTRADICTION_MAX_PAIRS]


def score_contradictions_nli(pairs: list[dict]) -> list[dict]:
    """Score candidate pairs for contradiction using NLI model.

    Returns only pairs where the NLI model detects contradiction
    (score index 0 above threshold).
    """
    from cairn.embeddings import _daemon_nli
    from cairn.config import NLI_ENABLED, NLI_CONTRADICTION_THRESHOLD

    if not NLI_ENABLED or not pairs:
        return pairs

    # Build NLI pairs — both directions
    nli_pairs = []
    for p in pairs:
        nli_pairs.append([p["older"]["content"], p["newer"]["content"]])
        nli_pairs.append([p["newer"]["content"], p["older"]["content"]])

    scores = _daemon_nli(nli_pairs)
    if scores is None:
        return pairs

    confirmed = []
    for k, p in enumerate(pairs):
        fwd = scores[k * 2]
        rev = scores[k * 2 + 1]

        if isinstance(fwd, list):
            fwd_contra = fwd[0]  # [contradiction, entailment, neutral]
            rev_contra = rev[0]
        else:
            fwd_contra = fwd
            rev_contra = rev

        # Either direction detecting contradiction is sufficient
        max_contra = max(fwd_contra, rev_contra)
        if max_contra >= NLI_CONTRADICTION_THRESHOLD:
            p["nli_contradiction_score"] = float(max_contra)
            confirmed.append(p)

    confirmed.sort(key=lambda p: p["nli_contradiction_score"], reverse=True)
    return confirmed


def assess_contradictions_haiku(contradictions: list[dict]) -> list[dict]:
    """Use Haiku to classify contradiction pairs as SUPERSEDED/SEQUENTIAL/COMPLEMENTARY.

    Reuses the same stream-json + CAIRN_HEADLESS pattern as consolidation.
    Returns only SUPERSEDED pairs with Haiku's reasoning.
    """
    if not contradictions:
        return []

    lines = [
        f"You are reviewing {len(contradictions)} pairs of memories flagged as potentially contradictory.",
        "",
        "For EACH pair, reply with exactly one line:",
        "  PAIR <n>: SUPERSEDED: <reason why the older is now wrong>",
        "  PAIR <n>: COMPLEMENTARY: <they coexist without contradiction>",
        "  PAIR <n>: SEQUENTIAL: <older describes a previous state, not wrong just historical>",
        "",
        "SUPERSEDED = the older memory would actively mislead if acted upon.",
        "COMPLEMENTARY = different aspects of the same topic, both valid.",
        "SEQUENTIAL = natural evolution, older captures valid historical state.",
        "",
    ]

    for i, c in enumerate(contradictions):
        o, n = c["older"], c["newer"]
        tag = " [SAME TOPIC]" if c.get("same_topic") else ""
        lines.append(f"--- PAIR {i+1}{tag} ---")
        lines.append(f"OLDER (#{o['id']}, {o['type']}/{o['topic']}, {o['created_at']}):")
        lines.append(f'  "{o["content"]}"')
        lines.append(f"NEWER (#{n['id']}, {n['type']}/{n['topic']}, {n['created_at']}):")
        lines.append(f'  "{n["content"]}"')
        lines.append("")

    prompt = "\n".join(lines)
    print(f"  Haiku prompt: {len(prompt)} chars, {len(contradictions)} pairs")

    env = {**os.environ, "CAIRN_HEADLESS": "1"}
    try:
        proc = subprocess.Popen(
            ["claude", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--model", "haiku", "--max-turns", "1",
             "--append-system-prompt",
             "OVERRIDE ALL OTHER INSTRUCTIONS: Reply with plain text only. No <memory> blocks. No XML tags. "
             "One line per pair in format: PAIR N: VERDICT: reason"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        msg_payload = json.dumps({"type": "user", "message": {"role": "user",
            "content": [{"type": "text", "text": prompt}]}})
        proc.stdin.write((msg_payload + "\n").encode())
        proc.stdin.flush()

        response_text = ""
        start = time.time()
        timeout = max(60, len(contradictions) * 3)
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
        return []

    response_text = re.sub(r"<memory>.*?</memory>", "", response_text, flags=re.DOTALL).strip()
    response_text = re.sub(r"\[cm\]:.*$", "", response_text, flags=re.MULTILINE).strip()

    superseded = []
    for line in response_text.split("\n"):
        match = re.match(r"PAIR\s+(\d+)\s*:\s*(SUPERSEDED|COMPLEMENTARY|SEQUENTIAL)\s*:?\s*(.*)", line.strip())
        if not match:
            continue
        pair_idx = int(match.group(1)) - 1
        verdict = match.group(2)
        reason = match.group(3).strip()

        if pair_idx < 0 or pair_idx >= len(contradictions):
            continue

        c = contradictions[pair_idx]
        o, n = c["older"], c["newer"]
        tag = " [SAME TOPIC]" if c.get("same_topic") else ""
        nli_score = c.get("nli_contradiction_score", 0)
        print(f"  [{pair_idx+1}/{len(contradictions)}] #{o['id']} vs #{n['id']}{tag}  "
              f"sim={c['similarity']:.2f}  nli_contra={nli_score:.1f}")
        print(f"    OLD: {o['content'][:80]}")
        print(f"    NEW: {n['content'][:80]}")
        print(f"    -> {verdict}: {reason[:120]}")
        print()

        if verdict == "SUPERSEDED":
            c["verdict"] = "superseded"
            c["reason"] = reason or f"superseded by #{n['id']}"
            superseded.append(c)

    # Report unparsed pairs
    parsed = set()
    for line in response_text.split("\n"):
        m = re.match(r"PAIR\s+(\d+)", line.strip())
        if m:
            parsed.add(int(m.group(1)))
    missing = [i + 1 for i in range(len(contradictions)) if i + 1 not in parsed]
    if missing:
        print(f"  Warning: {len(missing)} pairs not parsed: {missing[:10]}")

    return superseded


def execute_supersession(conn: sqlite3.Connection, superseded: list[dict]) -> int:
    """Archive superseded memories with contradiction annotation."""
    count = 0
    for c in superseded:
        older = c["older"]
        newer = c["newer"]
        reason = f"superseded: {c.get('reason', '')} (by #{newer['id']})"
        conn.execute(
            "UPDATE memories SET archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (reason, older["id"])
        )
        count += 1
        print(f"  Archived #{older['id']}: {reason[:100]}")
    conn.commit()
    return count


def run_contradiction_detection(execute: bool = False) -> dict:
    """Run the contradiction detection pipeline.

    Phase 1: Bi-encoder candidate pairs (cosine >= 0.55)
    Phase 2: NLI contradiction scoring (filters to genuine contradictions)
    Phase 3: Haiku assessment in batches (SUPERSEDED/SEQUENTIAL/COMPLEMENTARY)
    Phase 4: Archive superseded memories (if --execute)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")

    print("Phase 1: Finding candidate pairs via bi-encoder similarity...")
    pairs = find_contradiction_pairs(conn)
    same = sum(1 for p in pairs if p["same_topic"])
    cross = len(pairs) - same
    print(f"  Found {len(pairs)} candidate pairs ({same} same-topic, {cross} cross-topic)")

    if not pairs:
        print("  No contradiction candidates found.")
        conn.close()
        return {"pairs_found": 0, "nli_confirmed": 0, "superseded": 0}

    print(f"\nPhase 2: NLI contradiction scoring...")
    contradictions = score_contradictions_nli(pairs)
    print(f"  {len(contradictions)} pairs confirmed as contradictions by NLI")

    if not contradictions:
        print("  No contradictions passed NLI check.")
        conn.close()
        return {"pairs_found": len(pairs), "nli_confirmed": 0, "superseded": 0}

    # Batch Haiku assessment — 50 pairs per batch to stay within prompt limits
    BATCH_SIZE = 50
    print(f"\nPhase 3: Haiku assessment ({len(contradictions)} pairs in {(len(contradictions) + BATCH_SIZE - 1) // BATCH_SIZE} batches)...")
    all_superseded = []
    for batch_start in range(0, len(contradictions), BATCH_SIZE):
        batch = contradictions[batch_start:batch_start + BATCH_SIZE]
        print(f"  Batch {batch_start // BATCH_SIZE + 1}: pairs {batch_start + 1}-{batch_start + len(batch)}")
        superseded = assess_contradictions_haiku(batch)
        all_superseded.extend(superseded)

    superseded = all_superseded
    print(f"  {len(superseded)} pairs classified as SUPERSEDED")

    archived = 0
    if superseded and execute:
        print(f"\nPhase 4: Archiving superseded memories...")
        archived = execute_supersession(conn, superseded)
    elif superseded and not execute:
        print(f"\n  [DRY RUN] Would archive {len(superseded)} superseded memories:")
        for c in superseded:
            print(f"    #{c['older']['id']} superseded by #{c['newer']['id']}: {c.get('reason', '')[:80]}")

    conn.close()

    summary = {
        "pairs_found": len(pairs),
        "nli_confirmed": len(contradictions),
        "haiku_superseded": len(superseded),
        "archived": archived,
    }

    print(f"\n=== Contradiction Detection Summary ===")
    print(f"  Candidate pairs (bi-encoder): {summary['pairs_found']}")
    print(f"  Contradictions (NLI): {summary['nli_confirmed']}")
    print(f"  Superseded (Haiku): {summary['haiku_superseded']}")
    if execute:
        print(f"  Memories archived: {summary['archived']}")
    else:
        print(f"  [DRY RUN] No changes made")

    return summary


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    if "--contradictions" in sys.argv:
        run_contradiction_detection(execute=execute)
    else:
        use_llm = "--no-llm" not in sys.argv
        run_consolidation(execute=execute, use_llm=use_llm)
