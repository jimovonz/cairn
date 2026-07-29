#!/usr/bin/env python3
"""Multi-vote label denoising: re-grade each (query, memory) pair K times and take the
MEDIAN, to cut the ~0.81 Opus test-retest noise that ceilings the reranker student
(validate_relabel measured that floor; enrichment/more-data could not beat it).

The stored relevance_silver grade is vote 1; this adds votes 2..K. Resumable: extra
votes accumulate in denoise_votes.jsonl keyed by (memory_id, qhash), so quota walls just
pause it. Emits denoised_silver.jsonl (median grade + raw votes) for retraining.

  .venv/bin/python -m cairn.multi_vote_label --votes 3
"""
from __future__ import annotations
import argparse, json, os, sys, statistics
from cairn.label_relevance import judge_batch, qhash

HERE = os.path.dirname(__file__)
SILVER = os.path.join(HERE, "training_data", "relevance_silver.jsonl")
VOTES = os.path.join(HERE, "training_data", "denoise_votes.jsonl")
OUT = os.path.join(HERE, "training_data", "denoised_silver.jsonl")


def _key(mid, query):
    return f"{mid}:{qhash(query)}"


def _emit(base, extra, votes_target):
    n = changed = full = 0
    with open(OUT, "w") as f:
        for k, b in base.items():
            votes = [b["v1"]] + extra.get(k, [])
            med = round(statistics.median(votes))
            changed += (med != b["v1"])
            full += (len(votes) >= votes_target)
            f.write(json.dumps({"memory_id": b["memory_id"], "query": b["query"],
                                "mem": b["mem"], "grade": med, "votes": votes}) + "\n")
            n += 1
    print(f"\nemitted {n} denoised labels -> {OUT}  ({changed} changed from vote-1, "
          f"{full} fully {votes_target}-voted)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", type=int, default=3, help="total votes per pair incl the original")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--max-empty", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="cap pairs (0=all) for a smoke test")
    ap.add_argument("--silver", default=SILVER)
    args = ap.parse_args()

    base = {}
    for line in open(args.silver):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("query") and d.get("grade") is not None and d.get("memory_id") is not None and d.get("mem"):
            base[_key(d["memory_id"], d["query"])] = {
                "memory_id": d["memory_id"], "query": d["query"], "mem": d["mem"], "v1": int(d["grade"])}

    extra = {}
    if os.path.exists(VOTES):
        for line in open(VOTES):
            try:
                d = json.loads(line)
            except Exception:
                continue
            extra.setdefault(d["key"], []).extend(d.get("votes", []))

    keys = list(base.keys())
    if args.limit:
        keys = keys[:args.limit]
    empty = 0
    vf = open(VOTES, "a")
    while True:
        need = [k for k in keys if 1 + len(extra.get(k, [])) < args.votes]
        if not need:
            break
        print(f"{len(need)} pairs need more votes (target {args.votes})", file=sys.stderr)
        got_any = False
        for off in range(0, len(need), args.batch):
            bk = need[off:off + args.batch]
            batch = [{"memory_id": base[k]["memory_id"], "query": base[k]["query"],
                      "mem": base[k]["mem"], "_k": k} for k in bk]
            grades, ok, err = judge_batch(batch, args.model, off)
            if not ok:
                empty += 1
                print(f"  batch FAILED ({empty}/{args.max_empty}): {err}", file=sys.stderr)
                if empty >= args.max_empty:
                    print("aborting (quota?) — re-run to resume.", file=sys.stderr)
                    vf.close(); _emit(base, extra, args.votes); return
                continue
            empty = 0; got = 0
            for i, it in enumerate(batch):
                g = grades.get(off + i + 1)
                if g is None:
                    continue
                extra.setdefault(it["_k"], []).append(g)
                vf.write(json.dumps({"key": it["_k"], "votes": [g]}) + "\n"); vf.flush()
                got += 1; got_any = True
            print(f"  batch {off//args.batch+1}: +{got}", file=sys.stderr)
        if not got_any:
            break
    vf.close()
    _emit(base, extra, args.votes)


if __name__ == "__main__":
    main()
