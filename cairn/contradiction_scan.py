#!/usr/bin/env python3
"""Scan the memory database for contradictory memories.

Finds pairs of active memories that are semantically similar but contain
negation mismatches — indicating one likely supersedes the other.

Usage:
  python3 contradiction_scan.py              # Report only
  python3 contradiction_scan.py --annotate   # Annotate older memory in each pair
  python3 contradiction_scan.py --assess     # Use Haiku to judge each pair, auto-annotate confirmed
  python3 contradiction_scan.py --since 7d   # Only scan memories from last 7 days
"""

import sys
import os
import json
import sqlite3
import argparse
from datetime import datetime, timedelta
from typing import Optional

from hooks.hook_helpers import DB_PATH
from hooks.storage import _has_negation_mismatch


def parse_since(value: str) -> datetime:
    now = datetime.utcnow()
    if value.endswith("h"):
        return now - timedelta(hours=int(value[:-1]))
    elif value.endswith("d"):
        return now - timedelta(days=int(value[:-1]))
    elif value.endswith("w"):
        return now - timedelta(weeks=int(value[:-1]))
    elif value.endswith("m"):
        return now - timedelta(days=int(value[:-1]) * 30)
    return datetime.fromisoformat(value)


def scan(since: str | None = None, annotate: bool = False) -> list[dict]:
    try:
        from cairn import embeddings as emb
    except ImportError:
        print("Cannot import embeddings module")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 5000")

    # Load all active memories with embeddings
    base_where = "WHERE embedding IS NOT NULL AND archived_reason IS NULL"

    rows = conn.execute(
        f"SELECT id, type, topic, content, embedding, created_at FROM memories {base_where} ORDER BY id"
    ).fetchall()

    # When --since is used, determine which memory IDs are "recent" so we only
    # form pairs where at least one member is recent (incremental scan).
    recent_ids: set | None = None
    if since:
        since_dt = parse_since(since)
        since_str = since_dt.strftime("%Y-%m-%d %H:%M:%S")
        recent_ids = {
            r[0] for r in conn.execute(
                f"SELECT id FROM memories {base_where} AND created_at >= ?",
                (since_str,)
            ).fetchall()
        }
        print(f"Scanning {len(rows)} active memories ({len(recent_ids)} recent since {since_str})..."
        )
    else:
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

    def _is_recent_pair(m1, m2) -> bool:
        """When doing incremental scan, at least one member must be recent."""
        if recent_ids is None:
            return True
        return m1["id"] in recent_ids or m2["id"] in recent_ids

    # Tier 1: Same type+topic pairs
    for key, indices in by_topic.items():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                m1, m2 = memories[indices[a]], memories[indices[b]]
                if not _is_recent_pair(m1, m2):
                    continue
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
            if not _is_recent_pair(m1, m2):
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


# ============================================================
# Context recovery — get conversation context for a memory
# ============================================================

def recover_context(memory_id: int, margin: int = 3) -> Optional[str]:
    """Recover conversation context around a memory from its session transcript.
    Returns a condensed text excerpt, or None if transcript unavailable."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 5000")
    row = conn.execute(
        "SELECT session_id, created_at, depth FROM memories WHERE id = ?",
        (memory_id,)
    ).fetchone()
    if not row:
        conn.close()
        return None

    session_id, created_at, depth = row
    session = conn.execute(
        "SELECT transcript_path FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()

    if not session or not session[0] or not os.path.exists(session[0]):
        return None

    try:
        mem_time = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None

    lookback = depth if depth and depth > 0 else margin

    messages = []
    with open(session[0], encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                msg = entry.get("message", entry)
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                ts_str = entry.get("timestamp", "")
                if content.strip():
                    messages.append({"role": role, "text": content.strip()[:500], "ts": ts_str})
            except (json.JSONDecodeError, KeyError):
                continue

    # Find messages around the memory's creation time
    target_idx = len(messages) - 1
    for i, m in enumerate(messages):
        try:
            msg_time = datetime.strptime(m["ts"][:19], "%Y-%m-%dT%H:%M:%S")
            if msg_time >= mem_time:
                target_idx = i
                break
        except (ValueError, TypeError):
            continue

    start = max(0, target_idx - lookback)
    end = min(len(messages), target_idx + 2)
    excerpt = messages[start:end]

    if not excerpt:
        return None

    lines = []
    for m in excerpt:
        prefix = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"[{prefix}] {m['text'][:300]}")
    return "\n".join(lines)


# ============================================================
# Haiku assessment — use Claude Haiku to judge contradictions
# ============================================================

def assess_with_haiku(contradictions: list[dict]) -> list[dict]:
    """Send all contradiction pairs to Haiku in a single batched call.

    Uses one claude process via stream-json — dramatically cheaper and faster
    than per-pair spawning (~30s vs ~5min for 52 pairs).
    Returns list of confirmed contradictions with Haiku's reason.
    """
    import subprocess, re as _re, select, time as _time

    # Build batched prompt with all pairs
    lines = [
        "You are reviewing a memory database for contradictions.",
        f"Below are {len(contradictions)} pairs of memories flagged as potentially contradictory.",
        "",
        "For EACH pair, reply with exactly one line in this format:",
        "  PAIR <n>: SUPERSEDED: <reason>",
        "  PAIR <n>: COMPLEMENTARY",
        "  PAIR <n>: SEQUENTIAL",
        "",
        "Only say SUPERSEDED if the older memory would actively mislead a future session that acts on it.",
        "COMPLEMENTARY means they coexist without contradiction.",
        "SEQUENTIAL means the older describes a previous state that evolved — not wrong, just historical.",
        "",
    ]

    for i, c in enumerate(contradictions):
        o, n = c["older"], c["newer"]
        ctx_old = recover_context(o["id"]) or "(no transcript)"
        ctx_new = recover_context(n["id"]) or "(no transcript)"
        lines.append(f"--- PAIR {i+1} ---")
        lines.append(f"OLDER (#{o['id']}, {o['type']}/{o['topic']}, {o['created_at']}):")
        lines.append(f'  "{o["content"]}"')
        if ctx_old != "(no transcript)":
            lines.append(f"  Context: {ctx_old[:400]}")
        lines.append(f"NEWER (#{n['id']}, {n['type']}/{n['topic']}, {n['created_at']}):")
        lines.append(f'  "{n["content"]}"')
        if ctx_new != "(no transcript)":
            lines.append(f"  Context: {ctx_new[:400]}")
        lines.append("")

    prompt = "\n".join(lines)
    print(f"  Prompt: {len(prompt)} chars, {len(contradictions)} pairs")

    # Single claude call via stream-json
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

        # Read stream
        response_text = ""
        start = _time.time()
        timeout = max(60, len(contradictions) * 3)  # Scale timeout with pair count
        while _time.time() - start < timeout:
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

    # Strip memory blocks
    response_text = _re.sub(r"<memory>.*?</memory>", "", response_text, flags=_re.DOTALL).strip()

    # Parse verdicts — one per line matching "PAIR N: VERDICT"
    confirmed = []
    for line in response_text.split("\n"):
        match = _re.match(r"PAIR\s+(\d+)\s*:\s*(SUPERSEDED|COMPLEMENTARY|SEQUENTIAL)\s*:?\s*(.*)", line.strip())
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
        print(f"  [{pair_idx+1}/{len(contradictions)}] #{o['id']} vs #{n['id']}{tag}  sim={c['similarity']:.2f}")
        print(f"    OLD: {o['content'][:80]}")
        print(f"    NEW: {n['content'][:80]}")
        print(f"    -> {verdict}: {reason[:120]}")
        print()

        c["verdict"] = verdict.lower()
        if verdict == "SUPERSEDED":
            c["reason"] = reason or f"superseded by #{n['id']}"
            confirmed.append(c)

    # Report unparsed pairs
    parsed_indices = set()
    for line in response_text.split("\n"):
        m = _re.match(r"PAIR\s+(\d+)", line.strip())
        if m:
            parsed_indices.add(int(m.group(1)))
    missing = [i+1 for i in range(len(contradictions)) if i+1 not in parsed_indices]
    if missing:
        print(f"  Warning: {len(missing)} pairs not parsed: {missing[:10]}")

    return confirmed


def assess_and_annotate(since: str | None = None):
    """Full pipeline: scan → assess with Haiku → annotate confirmed contradictions."""
    contradictions = scan(since=since)
    if not contradictions:
        return

    print(f"\n{'='*60}")
    print(f"Assessing {len(contradictions)} candidates with Haiku...\n")

    confirmed = assess_with_haiku(contradictions)

    if not confirmed:
        print("No confirmed contradictions.")
        return

    print(f"\n{'='*60}")
    print(f"{len(confirmed)} confirmed supersessions out of {len(contradictions)} candidates.\n")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 5000")
    for c in confirmed:
        o, n = c["older"], c["newer"]
        reason = c.get("reason", f"superseded by #{n['id']}")
        conn.execute(
            "UPDATE memories SET archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (reason, o["id"])
        )
        print(f"  Annotated #{o['id']}: {reason}")
    conn.commit()
    conn.close()

    print(f"\nDone. {len(confirmed)} memories annotated as superseded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan for contradictory memories")
    parser.add_argument("--annotate", action="store_true", help="Annotate older memory in each contradictory pair (no AI judgement)")
    parser.add_argument("--assess", action="store_true", help="Use Haiku to judge each pair, auto-annotate confirmed contradictions")
    parser.add_argument("--since", type=str, help="Only scan memories from this date (ISO, 7d, 2w, 1m)")
    args = parser.parse_args()

    if args.assess:
        assess_and_annotate(since=args.since)
    else:
        scan(since=args.since, annotate=args.annotate)
