#!/usr/bin/env python3
"""Shadow cross-encoder scoring for unreranked deliveries.

WHY: ~93% of injected memories arrive via layers that bypass the cross-encoder
(project-bootstrap, first-prompt, correction-bootstrap). Those layers currently
out-engage the reranked per-prompt layer, so gating them blind would probably
destroy value — but their label coverage is 0.35%, so "probably" is all anyone
can say. This scores them WITHOUT acting on the score, so a floor can later be
calibrated from evidence the same way the 0.10 CUDA floor was calibrated from
9k deliveries on 2026-07-02.

DELIBERATELY NOT INLINE. Scoring in the prompt hook would add CE latency to
every turn for zero behavioural benefit, and the hot hook path is specifically
kept free of torch. Every input is already persisted per delivery
(context_text + memory_id), so this runs offline over rows already written.

Reads and writes only `memory_deliveries.shadow_ce_score` (added here if
missing). Nothing consumes that column — no retrieval path changes, no
filtering, no archiving. It exists to answer one question later: at floor X,
how many ENGAGED deliveries would have been dropped?

NOTE: ce_score on an unreranked row is NOT a cross-encoder score — log_memory_
deliveries falls back to the composite score when no reranker ran. That is why
this needs its own column rather than backfilling ce_score.

Usage:
  python3 cairn/shadow_ce_backfill.py [--limit N] [--layer L] [--dry-run] [--stats]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cairn.ingest import sqlite3  # pysqlite3 guard — never stdlib on a cairn WAL DB
from cairn import config

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db")
EPH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db")
BATCH = 256


def ensure_column(conn) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_deliveries)")}
    if "shadow_ce_score" not in cols:
        conn.execute("ALTER TABLE memory_deliveries ADD COLUMN shadow_ce_score REAL")
        conn.commit()


def load_encoder():
    """Mirror daemon._get_cross_encoder's device-aware selection.

    Standalone batch job, so importing torch here is fine — the constraint is
    that the *hook* path stays torch-free, not this one.
    """
    if not config.CROSS_ENCODER_ENABLED:
        return None, None
    from sentence_transformers import CrossEncoder
    name, floor = config.resolve_reranker()
    return CrossEncoder(name), name


def pending(conn, limit, layer=None):
    q = ("SELECT id, memory_id, context_text FROM memory_deliveries "
         "WHERE (reranker_model IS NULL OR reranker_model = '') "
         "AND shadow_ce_score IS NULL AND context_text IS NOT NULL AND context_text != ''")
    args = []
    if layer:
        q += " AND layer = ?"
        args.append(layer)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    return conn.execute(q, args).fetchall()


def memory_texts(ids):
    """Fetch topic+content for the delivered memories, from the durable DB."""
    if not ids:
        return {}
    dur = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        out = {}
        ids = list(ids)
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            ph = ",".join("?" * len(chunk))
            for mid, topic, content in dur.execute(
                    f"SELECT id, topic, content FROM memories WHERE id IN ({ph})", chunk):
                out[mid] = f"{topic or ''} {content or ''}".strip()
        return out
    finally:
        dur.close()


def stats(conn):
    row = conn.execute(
        "SELECT COUNT(*), SUM(shadow_ce_score IS NOT NULL) FROM memory_deliveries "
        "WHERE reranker_model IS NULL OR reranker_model = ''").fetchone()
    print(f"unreranked deliveries: {row[0]}   shadow-scored: {row[1] or 0}")
    print("\nby layer — shadow score vs engagement (the calibration input):")
    print(f"  {'layer':24s} {'scored':>7} {'eng':>6} {'notEng':>7} {'medEng':>7} {'medNot':>7}")
    for layer, n, e, ne, me, mn in conn.execute("""
        SELECT COALESCE(layer,'?'), COUNT(*),
               SUM(engaged = 1), SUM(engaged = 0),
               ROUND(AVG(CASE WHEN engaged = 1 THEN shadow_ce_score END), 4),
               ROUND(AVG(CASE WHEN engaged = 0 THEN shadow_ce_score END), 4)
        FROM memory_deliveries
        WHERE shadow_ce_score IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC"""):
        print(f"  {layer:24s} {n:>7} {e or 0:>6} {ne or 0:>7} "
              f"{str(me if me is not None else '-'):>7} {str(mn if mn is not None else '-'):>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--layer", default=None, help="restrict to one retrieval layer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    conn = sqlite3.connect(EPH_PATH)
    conn.execute("PRAGMA busy_timeout=8000")
    ensure_column(conn)

    if a.stats:
        stats(conn)
        conn.close()
        return

    rows = pending(conn, a.limit, a.layer)
    if not rows:
        print("nothing to score")
        conn.close()
        return
    texts = memory_texts({r[1] for r in rows})
    pairs, keep = [], []
    for did, mid, ctx in rows:
        t = texts.get(mid)
        if not t:
            continue
        pairs.append((ctx, t))
        keep.append(did)
    if not pairs:
        print(f"{len(rows)} pending, none resolvable to memory text")
        conn.close()
        return
    if a.dry_run:
        print(f"DRY: would score {len(pairs)} deliveries")
        conn.close()
        return

    enc, name = load_encoder()
    if enc is None:
        print("cross-encoder unavailable (CROSS_ENCODER_ENABLED off?)")
        conn.close()
        return
    written = 0
    for i in range(0, len(pairs), BATCH):
        scores = enc.predict(pairs[i:i + BATCH])
        conn.executemany("UPDATE memory_deliveries SET shadow_ce_score = ? WHERE id = ?",
                         [(float(s), d) for s, d in zip(scores, keep[i:i + BATCH])])
        conn.commit()
        written += len(scores)
    print(f"shadow-scored {written} deliveries with {name}")
    conn.close()


if __name__ == "__main__":
    main()
