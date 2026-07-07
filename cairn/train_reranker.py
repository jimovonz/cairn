#!/usr/bin/env python3
"""Pairwise fine-tune bge-reranker-base on Opus rg-grade labels (Phase 3 student).

Design (per docs/spec-memory-relevance-grading.md):
- PAIRWISE, not absolute: within a query's shortlist, teacher grades induce order
  (grade_a > grade_b => a should rank above b). The agent is reliable on relative
  order, noisy on absolute value — so we train on the ordering, via MarginRankingLoss
  over the cross-encoder's (query, memory) logits. Trains hard on the extremes.
- Fine-tune the PRETRAINED base (BAAI/bge-reranker-base), never from scratch. The
  bi-encoder recall stays untrained; the student only re-ranks within the shortlist,
  so exposure bias is bounded.
- Split held-out BY QUERY (a whole query-group is train xor eval) — the student must
  generalise to unseen shortlists. For a cross-encoder the scored unit is the PAIR,
  so a memory recurring under a different query in the other split is a distinct input.
- Promotion gate: student-vs-teacher PAIRWISE agreement on held-out must clear
  --gate (default 0.90) AND beat the pretrained baseline, else DO NOT deploy.

CPU by default (CUDA is currently erroring; this is an offline pass). Small caps keep
a proof run tractable; raise for a real run.

  .venv/bin/python -m cairn.train_reranker --epochs 2 --max-pairs 3000
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, itertools, random

DEFAULT_LABELS = os.path.join(os.path.dirname(__file__), "training_data", "relevance_silver.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "training_data", "reranker-student")


def qhash(q): return hashlib.sha1(q.encode("utf-8", "ignore")).hexdigest()[:12]


def load_groups(path):
    """-> {qhash: [(query, mem, grade), ...]} grouped by query."""
    groups = {}
    with open(path) as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            if not (d.get("query") and d.get("mem") and d.get("grade") is not None):
                continue
            groups.setdefault(qhash(d["query"]), []).append(
                (d["query"], d["mem"], int(d["grade"])))
    return groups


def split_by_query(groups, heldout_frac, seed=13):
    """Deterministic query-level split (hash of qhash, not RNG state)."""
    train, heldout = {}, {}
    for qh, items in groups.items():
        bucket = int(hashlib.sha1(f"{seed}:{qh}".encode()).hexdigest(), 16) % 1000
        (heldout if bucket < heldout_frac * 1000 else train)[qh] = items
    return train, heldout


def make_pairs(groups, max_pairs, min_gap=2, seed=7):
    """Within each query-group, every (better, worse) pair whose grade gap >= min_gap.
    Per spec, train/eval on the EXTREMES: adjacent-grade pairs (gap 1, e.g. 2-vs-3) are
    noisy even for the teacher and dilute agreement toward chance, so default min_gap=2
    keeps only clear pairs (0-2, 0-3, 1-3). Returns [(q, mem_pos, mem_neg)]."""
    pairs = []
    for items in groups.values():
        for a, b in itertools.combinations(items, 2):
            if abs(a[2] - b[2]) < min_gap:
                continue
            hi, lo = (a, b) if a[2] > b[2] else (b, a)
            pairs.append((hi[0], hi[1], lo[1]))  # same query for both sides
    rng = random.Random(seed); rng.shuffle(pairs)
    return pairs[:max_pairs]


def agreement(model, tok, groups, device, max_eval=2000, min_gap=2):
    """Fraction of clear (gap>=min_gap) within-query pairs the model orders correctly."""
    import torch
    pairs = make_pairs(groups, max_eval, min_gap)
    if not pairs:
        return None, 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), 32):
            chunk = pairs[i:i+32]
            qs = [p[0] for p in chunk]
            hi = tok(qs, [p[1] for p in chunk], padding=True, truncation=True,
                     max_length=320, return_tensors="pt").to(device)
            lo = tok(qs, [p[2] for p in chunk], padding=True, truncation=True,
                     max_length=320, return_tensors="pt").to(device)
            sh = model(**hi).logits.squeeze(-1)
            sl = model(**lo).logits.squeeze(-1)
            correct += int((sh > sl).sum())
    return correct / len(pairs), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--base", default="BAAI/bge-reranker-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--gate", type=float, default=0.90, help="auto-promote-safe flag only, NOT the deploy test")
    ap.add_argument("--incumbent", default=None, help="live reranker to beat (default: resolve_reranker())")
    ap.add_argument("--deploy-margin", type=float, default=0.02, help="student must beat incumbent by this (noise guard)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--eval-cap", type=int, default=2000, help="max held-out pairs scored")
    ap.add_argument("--min-gap", type=int, default=2, help="min grade gap for a pair (2=extremes only)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify wiring")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = torch.device(args.device)

    groups = load_groups(args.labels)
    if not groups:
        sys.exit(f"no labels in {args.labels}")
    train_g, held_g = split_by_query(groups, args.heldout_frac)
    tr_pairs = make_pairs(train_g, args.max_pairs, args.min_gap)
    if args.smoke:
        tr_pairs = tr_pairs[:32]; args.epochs = 1; args.eval_cap = 64
    print(f"queries: {len(groups)} (train {len(train_g)} / held {len(held_g)})  "
          f"train pairs: {len(tr_pairs)}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(dev)

    # baseline (pretrained, pre-finetune) held-out agreement
    base_agr, n_eval = agreement(model, tok, held_g, dev, args.eval_cap, args.min_gap)
    print(f"baseline held-out pairwise agreement: {base_agr*100:.1f}%  ({n_eval} pairs)",
          file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = torch.nn.MarginRankingLoss(margin=args.margin)
    target = torch.ones(1, device=dev)
    for ep in range(args.epochs):
        model.train(); random.Random(ep).shuffle(tr_pairs); tot = 0.0
        for i in range(0, len(tr_pairs), args.batch):
            chunk = tr_pairs[i:i+args.batch]
            qs = [p[0] for p in chunk]
            hi = tok(qs, [p[1] for p in chunk], padding=True, truncation=True,
                     max_length=320, return_tensors="pt").to(dev)
            lo = tok(qs, [p[2] for p in chunk], padding=True, truncation=True,
                     max_length=320, return_tensors="pt").to(dev)
            sh = model(**hi).logits.squeeze(-1)
            sl = model(**lo).logits.squeeze(-1)
            loss = lossf(sh, sl, target.expand_as(sh))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        print(f"epoch {ep+1}/{args.epochs}  avg loss {tot/max(1,len(tr_pairs)//args.batch):.4f}",
              file=sys.stderr)

    ft_agr, _ = agreement(model, tok, held_g, dev, args.eval_cap, args.min_gap)

    # THE deploy test: beat the INCUMBENT (currently-deployed live reranker) on the SAME
    # held-out pairs. That is the real "is it better" question — not an arbitrary 90%.
    # If the incumbent IS the training base, its pre-finetune score already == baseline.
    from cairn.config import resolve_reranker
    inc_name = args.incumbent or resolve_reranker()[0]
    if inc_name == args.base:
        inc_agr = base_agr
    else:
        from transformers import AutoTokenizer as _AT
        inc_tok = _AT.from_pretrained(inc_name)
        inc_model = AutoModelForSequenceClassification.from_pretrained(inc_name, num_labels=1).to(dev)
        inc_agr, _ = agreement(inc_model, inc_tok, held_g, dev, args.eval_cap, args.min_gap)

    beats = ft_agr > (inc_agr or 0) + args.deploy_margin
    promote_safe = ft_agr >= args.gate
    print("\n=== reranker student result (held-out clear pairs) ===")
    print(f"incumbent [{inc_name.split('/')[-1]}]: {inc_agr*100:.1f}%")
    print(f"student   [finetuned {args.base.split('/')[-1]}]: {ft_agr*100:.1f}%")
    print(f"DEPLOY (student > incumbent + {args.deploy_margin*100:.0f}pt): {'YES' if beats else 'NO'}"
          f"   [auto-promote-safe (>= {args.gate*100:.0f}%): {'yes' if promote_safe else 'no'}]")
    if beats and not args.smoke:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out); tok.save_pretrained(args.out)
        print(f"saved -> {args.out}")
    else:
        print("NOT saved — student does not beat the incumbent on the test set.")


if __name__ == "__main__":
    main()
