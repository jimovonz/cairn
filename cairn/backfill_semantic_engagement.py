#!/usr/bin/env python3
"""Retro-score historical deliveries with the semantic engagement second chance.

Lexical term-overlap under-detects a response that APPLIED a memory while
paraphrasing it, so every delivery scored before the semantic pass shipped
carries a verdict biased toward "not engaged". `apply_engagement` only touches
rows where `engaged_score IS NULL`, so history is never revisited on its own —
without this backfill the pre- and post-change eras sit on different
measurement bases and any rate spanning them is invalid.

Response text is not stored on the delivery row, so it is recovered from the
session transcript: the first assistant message at or after `delivered_at`.

Idempotent — rows already carrying an `engaged_method` tag are skipped.

    .venv/bin/python cairn/backfill_semantic_engagement.py --dry-run
    .venv/bin/python cairn/backfill_semantic_engagement.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pysqlite3 as sqlite3
except ImportError:  # pragma: no cover - guarded by tests/test_sqlite_guard.py
    if os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") != "1":
        raise
    import sqlite3

MAX_CHARS = 4000  # must match apply_engagement, or scores are not comparable


def _parse_ts(raw):
    """Parse either a SQLite UTC stamp or a transcript ISO stamp to naive UTC."""
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "").replace("T", " ")
    if "+" in s:
        s = s.split("+")[0]
    s = s.split(".")[0]
    try:
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _assistant_turns(transcript_path):
    """-> sorted [(ts, text)] of assistant messages carrying text."""
    from hooks.transcript_adapter import iter_normalized_entries

    out = []
    try:
        for entry in iter_normalized_entries(transcript_path):
            msg = entry.get("message", {})
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text")
            else:
                text = content if isinstance(content, str) else ""
            text = (text or "").strip()
            if not text:
                continue
            ts = _parse_ts(entry.get("timestamp", ""))
            if ts:
                out.append((ts, text))
    except Exception:
        return []
    out.sort(key=lambda r: r[0])
    return out


def _response_for(turns, delivered_at):
    """First assistant response at or after the delivery — the turn it landed in."""
    ts = _parse_ts(delivered_at)
    if ts is None:
        return None
    for t, text in turns:
        if t >= ts:
            return text
    return None


def backfill(dry_run=False, limit=None, eph_path=None, durable_path=None, verbose=False):
    from cairn.relevance import _eph_path, _durable_path, semantic_engaged, _cosine
    from cairn.embeddings import embed, from_blob

    econn = sqlite3.connect(_eph_path(eph_path))
    econn.execute("PRAGMA busy_timeout=10000")
    dconn = sqlite3.connect(_durable_path(durable_path))

    rows = econn.execute(
        "SELECT id, session_id, memory_id, context_text, delivered_at "
        "FROM memory_deliveries "
        "WHERE engaged IS NOT NULL AND engaged != 1 AND engaged_method IS NULL "
        "ORDER BY session_id, delivered_at"
    ).fetchall()
    if limit:
        rows = rows[:limit]

    stats = {"examined": 0, "rescued": 0, "no_response": 0,
             "no_vector": 0, "no_transcript": 0}
    if not rows:
        return stats

    mem_ids = sorted({int(r[2]) for r in rows})
    qm = ",".join("?" * len(mem_ids))
    mem_vec = {}
    for mid, emb in dconn.execute(
            f"SELECT id, embedding FROM memories WHERE id IN ({qm})", mem_ids):
        if emb:
            try:
                mem_vec[int(mid)] = from_blob(emb)
            except Exception:
                pass

    paths = dict(dconn.execute("SELECT session_id, transcript_path FROM sessions"))
    vec_cache, updates = {}, []

    def _vec(text):
        key = text[:MAX_CHARS]
        if key not in vec_cache:
            vec_cache[key] = embed(key, allow_slow=True)
        return vec_cache[key]

    by_session = {}
    for r in rows:
        by_session.setdefault(r[1], []).append(r)

    for sid, srows in by_session.items():
        tpath = paths.get(sid)
        if not tpath or not os.path.exists(tpath):
            stats["no_transcript"] += len(srows)
            continue
        turns = _assistant_turns(tpath)
        if not turns:
            stats["no_transcript"] += len(srows)
            continue

        for row_id, _sid, mid, ctx, delivered_at in srows:
            stats["examined"] += 1
            mv = mem_vec.get(int(mid))
            if mv is None:
                stats["no_vector"] += 1
                continue
            response = _response_for(turns, delivered_at)
            if not response:
                stats["no_response"] += 1
                continue
            rv = _vec(response)
            if rv is None:
                stats["no_response"] += 1
                continue
            crm = _cosine(rv, mv)
            ccm = _cosine(_vec(ctx) if ctx else None, mv)
            if semantic_engaged(crm, ccm):
                updates.append((1, round(crm, 4), "semantic-backfill", row_id))
                stats["rescued"] += 1
                if verbose:
                    print(f"  rescued delivery {row_id} (mem {mid}): "
                          f"cos_resp={crm:.3f} cos_ctx={ccm:.3f}", file=sys.stderr)
            else:
                # Verdict stands, but now confirmed under the semantic rule too.
                updates.append((None, None, "lexical", row_id))

    if not dry_run and updates:
        for engaged, score, method, row_id in updates:
            if engaged is None:
                econn.execute(
                    "UPDATE memory_deliveries SET engaged_method = ? WHERE id = ?",
                    (method, row_id))
            else:
                econn.execute(
                    "UPDATE memory_deliveries SET engaged = ?, engaged_score = ?, "
                    "engaged_method = ? WHERE id = ?", (engaged, score, method, row_id))
        econn.commit()

    econn.close()
    dconn.close()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="print each rescue")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    stats = backfill(dry_run=args.dry_run, limit=args.limit, verbose=args.verbose)
    if args.json:
        print(json.dumps(stats))
        return
    ex, res = stats["examined"], stats["rescued"]
    rate = f"{100 * res / ex:.1f}%" if ex else "n/a"
    print(f"{'DRY RUN — ' if args.dry_run else ''}examined {ex} lexically-unengaged deliveries")
    print(f"  rescued as semantically engaged : {res}  ({rate})")
    print(f"  no recoverable response         : {stats['no_response']}")
    print(f"  memory vector missing           : {stats['no_vector']}")
    print(f"  transcript unavailable          : {stats['no_transcript']}")


if __name__ == "__main__":
    main()
