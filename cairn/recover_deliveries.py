"""Recover layer_delivery metrics from session JSONL transcripts.

Scans ~/.claude/projects/*/*.jsonl for <cairn_context> blocks containing
<entry id="..."> memory IDs and reconstructs the per-record delivery
events that would have been emitted into the ephemeral metrics table at
injection time. Inserts them as synthetic `layer_delivery` rows so the
dashboard's _memory_served_counts (cairn/dashboard.py:204) picks them up
without code changes.

Idempotent — each recovered event includes its source JSONL record uuid
in the metric detail; re-runs skip already-imported records.

Usage:
    python3 cairn/recover_deliveries.py                # dry-run summary
    python3 cairn/recover_deliveries.py --commit       # write to ephemeral DB
    python3 cairn/recover_deliveries.py --since 2026-05-01 --commit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Iterator

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3


_ID_RE = re.compile(r'<entry\s+id="([0-9]+)"')

DEFAULT_PROJECTS_GLOB = os.path.expanduser("~/.claude/projects/*/*.jsonl")
EPH_DB_PATH = os.environ.get(
    "CAIRN_EPHEMERAL_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db"),
)


def _iter_strings(o) -> Iterator[str]:
    if isinstance(o, dict):
        for v in o.values():
            yield from _iter_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _iter_strings(v)
    elif isinstance(o, str):
        yield o


def scan_jsonls(paths: list[str], since: str | None = None) -> list[dict]:
    """Return list of recovered events. Each event = one transcript
    record carrying >=1 <entry id="..."> memory IDs."""
    events = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "<entry" not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("timestamp")
                    if since and ts and ts < since:
                        continue
                    uuid = rec.get("uuid")
                    if not uuid:
                        continue
                    session_id = rec.get("sessionId") or ""
                    ids: list[int] = []
                    seen: set[str] = set()
                    for s in _iter_strings(rec):
                        if "<entry" not in s:
                            continue
                        for mid in _ID_RE.findall(s):
                            if mid in seen:
                                continue
                            seen.add(mid)
                            try:
                                ids.append(int(mid))
                            except ValueError:
                                pass
                    if not ids:
                        continue
                    events.append({
                        "timestamp": ts,
                        "session_id": session_id,
                        "uuid": uuid,
                        "ids": ids,
                        "source_jsonl": os.path.basename(p),
                    })
        except OSError:
            continue
    return events


def existing_recovered_uuids(conn) -> set[str]:
    """UUIDs already imported on a previous run (idempotency)."""
    out: set[str] = set()
    rows = conn.execute(
        "SELECT detail FROM metrics "
        "WHERE event = 'layer_delivery' "
        "AND detail LIKE '%\"recovered_from\"%'"
    ).fetchall()
    for (detail,) in rows:
        try:
            d = json.loads(detail)
        except (TypeError, ValueError):
            continue
        u = d.get("recovered_from")
        if u:
            out.add(u)
    return out


def commit(events: list[dict], eph_path: str) -> int:
    """Insert events into ephemeral metrics as layer_delivery rows.

    Returns count of newly-inserted rows (skips dups).
    """
    conn = sqlite3.connect(eph_path)
    conn.execute("PRAGMA busy_timeout=5000")
    inserted = 0
    try:
        seen = existing_recovered_uuids(conn)
        for ev in events:
            if ev["uuid"] in seen:
                continue
            detail = json.dumps({
                "layer": "recovered",
                "ids": ev["ids"],
                "recovered_from": ev["uuid"],
                "source_jsonl": ev["source_jsonl"],
            })
            conn.execute(
                "INSERT INTO metrics (event, session_id, detail, created_at) "
                "VALUES ('layer_delivery', ?, ?, ?)",
                (ev["session_id"], detail, ev["timestamp"]),
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted


def summarise(events: list[dict]) -> None:
    if not events:
        print("no events recovered")
        return
    deliveries: dict[int, int] = defaultdict(int)
    sess_count: dict[int, set[str]] = defaultdict(set)
    timestamps = sorted(e["timestamp"] for e in events if e["timestamp"])
    for ev in events:
        for mid in ev["ids"]:
            deliveries[mid] += 1
            sess_count[mid].add(ev["session_id"])
    total_hits = sum(deliveries.values())
    print(f"events:                  {len(events)}")
    print(f"unique memory IDs:       {len(deliveries)}")
    print(f"total delivery hits:     {total_hits}")
    print(f"date range:              {timestamps[0]}  →  {timestamps[-1]}")
    print(f"sessions touched:        "
          f"{len({e['session_id'] for e in events})}")
    print(f"\ntop 10 most-delivered memory IDs:")
    for mid, cnt in sorted(deliveries.items(), key=lambda x: -x[1])[:10]:
        print(f"  id={mid:>15} delivered {cnt:>4}x "
              f"across {len(sess_count[mid])} sessions")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glob", default=DEFAULT_PROJECTS_GLOB,
                    help="JSONL glob (default: %(default)s)")
    ap.add_argument("--since", help="ISO timestamp lower bound (e.g. 2026-05-01)")
    ap.add_argument("--commit", action="store_true",
                    help="Write recovered events to ephemeral metrics table")
    ap.add_argument("--eph-db", default=EPH_DB_PATH,
                    help="Ephemeral DB path (default: %(default)s)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    print(f"scanning {len(paths)} JSONLs from {args.glob}"
          + (f" since={args.since}" if args.since else ""))

    events = scan_jsonls(paths, since=args.since)
    summarise(events)

    if args.commit:
        if not os.path.exists(args.eph_db):
            print(f"\nERROR: ephemeral DB not found at {args.eph_db}", file=sys.stderr)
            sys.exit(2)
        n = commit(events, args.eph_db)
        print(f"\ninserted {n} new layer_delivery rows into {args.eph_db}")
        skipped = len(events) - n
        if skipped > 0:
            print(f"skipped {skipped} already-imported records")
    else:
        print("\n(dry-run; pass --commit to write to ephemeral metrics)")


if __name__ == "__main__":
    main()
