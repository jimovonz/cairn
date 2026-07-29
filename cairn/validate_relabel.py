#!/usr/bin/env python3
"""Validation-slice re-label: does ENRICHING the reranker passage (add keywords+facts)
shift Opus grades vs the stored labels?

Samples existing labelled pairs (oversampling long-content memories, where enrichment
most likely reveals content the 600-char truncation hid), re-judges each with the
ENRICHED passage using the SAME judge rubric, and reports grade drift. This is the
cheap check that tells us whether the existing 6.4k grades survive the input change:
  LOW drift  -> reuse the corpus, just retrain on the enriched passage (free).
  HIGH drift -> the grades are representation-biased; re-label the corpus enriched.

  .venv/bin/python -m cairn.validate_relabel --n 150
"""
from __future__ import annotations
import argparse, json, os, sys, random, statistics
import pysqlite3 as sqlite3
import cairn.query as q
from cairn.label_relevance import judge_batch, _memtext
DEFAULT_LABELS = os.path.join(os.path.dirname(__file__), "training_data", "relevance_silver.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--control", action="store_true", help="re-judge the UNENRICHED passage = judge test-retest noise floor")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "training_data", "relabel_drift.jsonl"))
    args = ap.parse_args()

    rows = []
    for line in open(args.labels):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("query") and d.get("grade") is not None and d.get("memory_id"):
            rows.append(d)

    db = sqlite3.connect(q.DB_PATH)
    for r in rows:
        row = db.execute("SELECT length(content) FROM memories WHERE id=?", (r["memory_id"],)).fetchone()
        r["_clen"] = row[0] if row else 0
    rng = random.Random(0); rng.shuffle(rows)
    longr = [r for r in rows if r["_clen"] > 600]
    shortr = [r for r in rows if 0 < r["_clen"] <= 600]
    half = args.n // 2
    sample = (longr[:half] + shortr[:args.n - half])[:args.n]
    rng.shuffle(sample)

    items = []
    for r in sample:
        mem = _memtext(db, r["memory_id"], enrich=not args.control)
        if not mem:
            continue
        items.append({"memory_id": r["memory_id"], "query": r["query"], "mem": mem,
                      "old": int(r["grade"]), "_clen": r["_clen"]})
    print(f"validating {len(items)} enriched pairs vs stored grades "
          f"(long-content {sum(1 for i in items if i['_clen']>600)})", file=sys.stderr)

    results = []
    for off in range(0, len(items), args.batch):
        batch = items[off:off + args.batch]
        grades, ok, err = judge_batch(batch, args.model, off)
        if not ok:
            print(f"  batch {off//args.batch+1} FAILED: {err}", file=sys.stderr)
            continue
        for i, it in enumerate(batch):
            g = grades.get(off + i + 1)
            if g is None:
                continue
            it["new"] = g; results.append(it)
        print(f"  batch {off//args.batch+1}: total {len(results)}", file=sys.stderr)

    if not results:
        sys.exit("no results (judge unavailable / quota?)")

    exact = sum(r["new"] == r["old"] for r in results) / len(results)
    within1 = sum(abs(r["new"] - r["old"]) <= 1 for r in results) / len(results)
    signed = statistics.mean(r["new"] - r["old"] for r in results)
    absmean = statistics.mean(abs(r["new"] - r["old"]) for r in results)
    up = sum(r["new"] > r["old"] for r in results) / len(results)
    down = sum(r["new"] < r["old"] for r in results) / len(results)
    lr = [r for r in results if r["_clen"] > 600]
    long_up = (sum(r["new"] > r["old"] for r in lr) / len(lr)) if lr else 0.0

    print("\n=== enriched-passage grade drift vs stored labels ===")
    print(f"n={len(results)} (long-content {len(lr)})")
    print(f"exact match:        {exact*100:4.0f}%")
    print(f"within +-1:         {within1*100:4.0f}%")
    print(f"mean signed drift:  {signed:+.2f}   (positive = enrichment RAISED the grade)")
    print(f"mean abs drift:     {absmean:.2f}")
    print(f"grade up: {up*100:.0f}%   down: {down*100:.0f}%   "
          f"(up on long/truncated content: {long_up*100:.0f}%)")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps({k: r[k] for k in ("memory_id", "old", "new", "_clen")}) + "\n")

    if absmean < 0.35 and abs(signed) < 0.25:
        print("\nVERDICT: LOW drift -> reuse existing grades, retrain on the enriched passage (free, no re-label).")
    else:
        print("\nVERDICT: MATERIAL drift -> grades shift with enrichment; re-label the corpus with the enriched passage.")


if __name__ == "__main__":
    main()
