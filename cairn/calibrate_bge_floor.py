#!/usr/bin/env python3
"""Calibrate the bge-reranker-base suppression floor from Opus rg-grade labels.

The live floor CROSS_ENCODER_SCORE_FLOOR_CUDA (config) was set from ENGAGEMENT
(a weak proxy). The relevance_silver.jsonl labels are a stronger signal: Opus
graded (query, memory) pairs 0-3. We run bge on those exact pairs and pick the
floor that DROPS noise (grade 0) while never dropping load-bearing memories
(grade 3) — a suppression floor must not reject real hits.

CPU-forced (bge calibration is an offline batch job; the daemon runs bge on the
4070 at inference time — same model, so the floor transfers).

  .venv/bin/python -m cairn.calibrate_bge_floor                 # report only
  .venv/bin/python -m cairn.calibrate_bge_floor --write-config  # patch config floor
"""
from __future__ import annotations
import argparse, json, os, sys

DEFAULT_LABELS = os.path.join(os.path.dirname(__file__), "training_data", "relevance_silver.jsonl")


def load_labels(path):
    pairs, grades = [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("query") and d.get("mem") and d.get("grade") is not None:
                pairs.append((d["query"], d["mem"])); grades.append(int(d["grade"]))
    return pairs, grades


def bge_scores(pairs, device, batch=64):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("BAAI/bge-reranker-base", device=device)
    return list(ce.predict(pairs, batch_size=batch, show_progress_bar=True))


def evaluate_floor(scores, grades, t):
    """At floor t (keep score>=t): recall of relevant(>=2), rejection of noise(0),
    and — the hard constraint — retention of load-bearing(3)."""
    pos = [s for s, g in zip(scores, grades) if g >= 2]
    neg = [s for s, g in zip(scores, grades) if g == 0]
    lb = [s for s, g in zip(scores, grades) if g == 3]
    keep_pos = sum(s >= t for s in pos) / max(1, len(pos))
    drop_neg = sum(s < t for s in neg) / max(1, len(neg))
    keep_lb = sum(s >= t for s in lb) / max(1, len(lb))
    return keep_pos, drop_neg, keep_lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lb-retain", type=float, default=0.97,
                    help="min fraction of grade-3 to retain (suppression floor must not drop hits)")
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args()

    pairs, grades = load_labels(args.labels)
    if not pairs:
        sys.exit(f"no labels in {args.labels}")
    n3 = sum(g == 3 for g in grades); n0 = sum(g == 0 for g in grades)
    print(f"labels: {len(pairs)}  (grade0={n0} grade3={n3} rel>=2={sum(g>=2 for g in grades)})",
          file=sys.stderr)

    scores = bge_scores(pairs, args.device)

    # sweep candidate floors
    cand = [i / 200 for i in range(0, 200)]  # 0.000 .. 0.995 step 0.005
    rows = [(t, *evaluate_floor(scores, grades, t)) for t in cand]

    # recommended: highest floor that still retains >= lb-retain of grade-3,
    # i.e. maximise noise rejection subject to not dropping load-bearing hits.
    feasible = [r for r in rows if r[3] >= args.lb_retain]
    rec = max(feasible, key=lambda r: r[2]) if feasible else rows[0]
    # youden-optimal (unconstrained) for reference
    youden = max(rows, key=lambda r: r[1] + r[2] - 1)

    from cairn.config import CROSS_ENCODER_SCORE_FLOOR_CUDA as cur
    cur_eval = evaluate_floor(scores, grades, cur)

    def fmt(r):
        return (f"floor={r[0]:.3f}  keep_rel={r[1]*100:4.0f}%  drop_noise={r[2]*100:4.0f}%  "
                f"keep_loadbearing={r[3]*100:4.0f}%")
    print("\n=== bge floor calibration (from rg grades) ===")
    print(f"CURRENT (engagement-set)  {fmt((cur,)+cur_eval)}")
    print(f"RECOMMENDED (>= {args.lb_retain*100:.0f}% grade-3 kept)  {fmt(rec)}")
    print(f"youden-optimal (ref)      {fmt(youden)}")

    if args.write_config:
        import re
        cfg = os.path.join(os.path.dirname(__file__), "config.py")
        src = open(cfg).read()
        new = re.sub(r"(CROSS_ENCODER_SCORE_FLOOR_CUDA\s*=\s*)[\d.]+",
                     rf"\g<1>{rec[0]:.3f}", src, count=1)
        if new != src:
            open(cfg, "w").write(new)
            print(f"\npatched config: CROSS_ENCODER_SCORE_FLOOR_CUDA {cur} -> {rec[0]:.3f}")
        else:
            print("\nconfig unchanged (pattern not found)")
    else:
        print(f"\n(dry-run) to apply: CROSS_ENCODER_SCORE_FLOOR_CUDA = {rec[0]:.3f}  (--write-config)")


if __name__ == "__main__":
    main()
