#!/usr/bin/env python3
"""Extract retrieval benchmark dataset from session transcripts.

Mines JSONL session files directly, extracting each retrieval event:
  - The user message that triggered the injection
  - The <cairn_context> block (query, memory IDs, similarity scores, layers)
  - The LLM's retrieval_outcome rating from the following memory block
  - Estimated token cost of the injection

Also supplements with metrics DB data where available.

Output modes:
    --stats     Summary statistics (default)
    --tsv       Per-event TSV dataset
    --json      Per-event JSON dataset

Usage:
    python3 cairn/benchmark_extract.py                # Summary stats
    python3 cairn/benchmark_extract.py --tsv > data.tsv
    python3 cairn/benchmark_extract.py --json > data.json
"""

from __future__ import annotations

import json
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict
from typing import Optional


DB_PATH = os.environ.get(
    "CAIRN_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db"),
)

TRANSCRIPT_ROOTS = [
    os.path.expanduser("~/.claude/projects"),
]

# Patterns for extracting data from transcripts
CAIRN_CONTEXT_RE = re.compile(
    r'<cairn_context\s+query="([^"]*)"[^>]*current_project="([^"]*)"[^>]*layer="([^"]*)"[^>]*>'
    r'(.*?)</cairn_context>',
    re.DOTALL,
)
ENTRY_RE = re.compile(
    r'<entry\s+id="(\d+)"[^>]*type="([^"]*)"[^>]*similarity="([^"]*)"[^>]*>([^<]*)</entry>'
)
ENTRY_LOOSE_RE = re.compile(
    r'<entry\s+id="(\d+)"[^>]*>'
)
OUTCOME_RE = re.compile(r'retrieval_outcome:\s*(useful|neutral|harmful)')
MEMORY_BLOCK_RE = re.compile(r'<memory>(.*?)</memory>', re.DOTALL)


def find_transcript_files() -> dict[str, str]:
    """Map session_id → transcript file path."""
    files = {}
    for root in TRANSCRIPT_ROOTS:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "**/*.jsonl"), recursive=True):
            session_id = os.path.splitext(os.path.basename(path))[0]
            files[session_id] = path
    return files


def extract_text(content) -> str:
    """Extract plain text from message content (string or content blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(p.strip() for p in parts if p.strip())
    return ""


def parse_cairn_context(text: str) -> list[dict]:
    """Extract all <cairn_context> blocks from text, returning structured data."""
    contexts = []
    for match in CAIRN_CONTEXT_RE.finditer(text):
        query = match.group(1)
        project = match.group(2)
        layer = match.group(3)
        body = match.group(4)

        # Extract entry IDs and similarity scores
        entries = []
        for entry_match in ENTRY_RE.finditer(body):
            entries.append({
                "id": int(entry_match.group(1)),
                "type": entry_match.group(2),
                "similarity": float(entry_match.group(3)) if entry_match.group(3) else None,
                "content_preview": entry_match.group(4)[:100],
            })

        # Fallback: if strict regex missed entries, try loose ID extraction
        if not entries:
            for loose_match in ENTRY_LOOSE_RE.finditer(body):
                entries.append({"id": int(loose_match.group(1))})

        contexts.append({
            "query": query,
            "project": project,
            "layer": layer,
            "entry_count": len(entries),
            "entries": entries,
            "memory_ids": [e["id"] for e in entries],
            "char_length": len(match.group(0)),
        })
    return contexts


def mine_transcript(path: str, session_id: str) -> list[dict]:
    """Mine a single transcript for retrieval events.

    <cairn_context> blocks appear in user messages (injected by prompt hook).
    retrieval_outcome ratings appear in later assistant <memory> blocks — not
    necessarily the immediate next response. Therefore we track injections
    per-event but link outcomes at session level.

    Returns a list of retrieval events with session-level outcome attached.
    """
    entries = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    # First pass: collect all injection events and all outcomes
    injections = []
    outcomes = []
    last_user_message = None
    last_user_timestamp = None

    for entry in entries:
        etype = entry.get("type", "")
        msg = entry.get("message", {})
        content = msg.get("content", "")
        text = extract_text(content)
        timestamp = entry.get("timestamp", "")

        if etype in ("user", "human"):
            cairn_contexts = parse_cairn_context(text)
            if cairn_contexts:
                injections.append({
                    "contexts": cairn_contexts,
                    "user_message": last_user_message,
                    "timestamp": last_user_timestamp or timestamp,
                })

            # Track actual user message (before hook-injected content)
            clean = text.split("<cairn_context")[0].split("<system-reminder")[0].strip()
            if clean and not clean.startswith("Stop hook"):
                last_user_message = clean[:500]
                last_user_timestamp = timestamp

        elif etype == "assistant":
            for match in OUTCOME_RE.finditer(text):
                outcomes.append(match.group(1))

    if not injections:
        return []

    # Session-level outcome: majority vote across all outcome declarations
    outcome_counts = defaultdict(int)
    for o in outcomes:
        outcome_counts[o] += 1
    session_outcome = max(outcome_counts, key=outcome_counts.get) if outcome_counts else None

    # Build per-injection events with session-level outcome
    events = []
    for inj in injections:
        all_memory_ids = []
        all_layers = set()
        total_chars = 0
        for ctx in inj["contexts"]:
            all_memory_ids.extend(ctx["memory_ids"])
            all_layers.add(ctx["layer"])
            total_chars += ctx["char_length"]

        events.append({
            "session_id": session_id,
            "timestamp": inj["timestamp"],
            "user_message": inj["user_message"],
            "query": inj["contexts"][0]["query"] if inj["contexts"] else None,
            "project": inj["contexts"][0]["project"] if inj["contexts"] else None,
            "layers": sorted(all_layers),
            "memory_ids": sorted(set(all_memory_ids)),
            "memory_count": len(set(all_memory_ids)),
            "token_cost_est": total_chars // 4,
            "outcome": session_outcome,
            "outcome_count": len(outcomes),
            "context_blocks": len(inj["contexts"]),
        })

    return events


def print_stats(events: list[dict]) -> None:
    """Print summary statistics for the extracted dataset."""
    total = len(events)
    sessions = len({e["session_id"] for e in events})
    with_outcome = sum(1 for e in events if e["outcome"])
    with_user_msg = sum(1 for e in events if e["user_message"])

    outcome_dist = defaultdict(int)
    for e in events:
        outcome_dist[e["outcome"] or "unlabelled"] += 1

    layer_dist = defaultdict(int)
    for e in events:
        for l in e["layers"]:
            layer_dist[l] += 1

    total_memories = sum(e["memory_count"] for e in events)
    total_tokens = sum(e["token_cost_est"] for e in events)

    # Per-layer outcome breakdown
    layer_outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in events:
        for l in e["layers"]:
            layer_outcomes[l][e["outcome"] or "unlabelled"] += 1

    print("=" * 60)
    print("CAIRN RETRIEVAL BENCHMARK DATASET")
    print("=" * 60)

    print(f"\nExtracted from session transcripts")
    print(f"{'─' * 40}")
    print(f"  Retrieval events:       {total:>6}")
    print(f"  Across sessions:        {sessions:>6}")
    print(f"  With outcome label:     {with_outcome:>6} ({pct(with_outcome, total)})")
    print(f"  With user message:      {with_user_msg:>6} ({pct(with_user_msg, total)})")

    print(f"\nOutcome distribution")
    print(f"{'─' * 40}")
    for outcome in ["useful", "neutral", "harmful", "unlabelled"]:
        count = outcome_dist.get(outcome, 0)
        bar = "█" * (count * 30 // max(outcome_dist.values())) if outcome_dist else ""
        print(f"  {outcome:>12}: {count:>5} ({pct(count, total)}) {bar}")

    print(f"\nLayer distribution")
    print(f"{'─' * 40}")
    for layer, count in sorted(layer_dist.items(), key=lambda x: -x[1]):
        print(f"  {layer:>15}: {count:>5}")

    print(f"\nRetrieval volume")
    print(f"{'─' * 40}")
    print(f"  Total memories injected:  {total_memories:>8}")
    print(f"  Avg per event:            {total_memories / total:>8.1f}" if total else "")
    print(f"  Total est. tokens:        {total_tokens:>8,}")
    print(f"  Avg tokens per event:     {total_tokens / total:>8,.0f}" if total else "")

    print(f"\nOutcome by layer")
    print(f"{'─' * 40}")
    for layer in sorted(layer_outcomes.keys()):
        outcomes = layer_outcomes[layer]
        useful = outcomes.get("useful", 0)
        neutral = outcomes.get("neutral", 0)
        harmful = outcomes.get("harmful", 0)
        unlabelled = outcomes.get("unlabelled", 0)
        labelled = useful + neutral + harmful
        hit_rate = f"{useful * 100 / labelled:.0f}%" if labelled else "n/a"
        print(f"  {layer:>15}: useful={useful}, neutral={neutral}, harmful={harmful}, "
              f"unlabelled={unlabelled} (hit rate: {hit_rate})")

    # Benchmark readiness assessment
    print(f"\nBenchmark readiness")
    print(f"{'─' * 40}")
    labelled = sum(1 for e in events if e["outcome"])
    print(f"  Labelled events:     {labelled}")
    if labelled >= 200:
        print(f"  Status:              ✓ Sufficient for statistical analysis")
    elif labelled >= 50:
        print(f"  Status:              ~ Marginal — directional signals only")
    else:
        print(f"  Status:              ✗ Insufficient — need {200 - labelled} more labelled events")

    print()


def pct(n: int, total: int) -> str:
    return f"{n * 100 / total:.0f}%" if total else "0%"


def print_tsv(events: list[dict]) -> None:
    """Print dataset as TSV."""
    headers = [
        "session_id", "timestamp", "user_message", "query",
        "project", "layers", "memory_count", "memory_ids",
        "token_cost_est", "outcome",
    ]
    print("\t".join(headers))
    for e in events:
        print("\t".join([
            e["session_id"],
            str(e["timestamp"] or ""),
            (e["user_message"] or "")[:200].replace("\t", " ").replace("\n", " "),
            (e["query"] or "")[:200].replace("\t", " ").replace("\n", " "),
            e["project"] or "",
            ",".join(e["layers"]),
            str(e["memory_count"]),
            ",".join(str(i) for i in e["memory_ids"][:30]),
            str(e["token_cost_est"]),
            e["outcome"] or "",
        ]))


def main():
    mode = "stats"
    if "--tsv" in sys.argv:
        mode = "tsv"
    elif "--json" in sys.argv:
        mode = "json"

    # Find and mine all transcripts
    transcripts = find_transcript_files()
    all_events = []

    for session_id, path in sorted(transcripts.items()):
        events = mine_transcript(path, session_id)
        all_events.extend(events)

    # Sort by timestamp
    all_events.sort(key=lambda e: e.get("timestamp") or "")

    if mode == "stats":
        print_stats(all_events)
    elif mode == "json":
        print(json.dumps(all_events, indent=2))
    else:
        print_tsv(all_events)


if __name__ == "__main__":
    main()
