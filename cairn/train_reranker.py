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


def load_groups(path, enrich=False):
    """-> {qhash: [(query, mem, grade), ...]} grouped by query. enrich=True re-renders
    each passage from its memory_id with keywords+facts (reusing the stored grade —
    validate_relabel showed enrichment shifts grades < the judge noise floor)."""
    db = None
    if enrich:
        import pysqlite3 as sqlite3
        import cairn.query as q
        from cairn.label_relevance import _memtext
        db = sqlite3.connect(q.DB_PATH)
    groups = {}
    with open(path) as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            if not (d.get("query") and d.get("grade") is not None):
                continue
            mem = d.get("mem")
            if enrich and d.get("memory_id") is not None:
                em = _memtext(db, d["memory_id"], enrich=True)
                if em:
                    mem = em
            if not mem:
                continue
            groups.setdefault(qhash(d["query"]), []).append(
                (d["query"], mem, int(d["grade"])))
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


# Minimum engaged_score for a weak positive. Measured 2026-07-25 over 1,675
# engaged=1 rows, the LEXICAL overlap ratio runs min 0.018 / median 0.110 /
# p90 0.262 / max 0.657 — so the originally-specified 0.5 sat above the 99th
# percentile and admitted just 13 rows, collapsing the pool to a single usable
# training group. 0.2 (~p79) keeps 345.
# CAVEAT: engaged_score is bimodal by source. Semantic-second-chance rows store
# cos(response, memory), which is >= ENGAGEMENT_SEM_THRESHOLD (0.55) by
# construction, so any threshold above ~0.55 selects semantic rows exclusively
# and silently discards every lexical positive. Recalibrate here, not upward.
ENGAGEMENT_MIN_POS_DEFAULT = 0.2


def _engagement_grade(engaged, engaged_score, agent_grade, min_pos=ENGAGEMENT_MIN_POS_DEFAULT):
    """Fold a behavioural engagement observation into a 0/3 pseudo-grade.

    Returns None for anything undecidable — dropping an ambiguous row beats
    guessing, because these weak labels are merged into the training pairs.
    The agent's own grade wins every conflict: it is a considered judgement,
    while engagement is a mechanical proxy that both over- and under-fires.
    """
    if engaged == 1 and (engaged_score or 0) >= min_pos:
        if agent_grade is not None and agent_grade <= 1:
            return None  # behavioural yes vs agent no — trust the agent, drop the row
        return 3
    if engaged == 0 and agent_grade is None:
        return 0  # weak lexical negative, with no agent opinion to outrank it
    return None


def load_engagement_groups(min_pos=ENGAGEMENT_MIN_POS_DEFAULT, eph_path=None, durable_path=None):
    """-> {"eng:<session>:<turn>": [(query, mem, grade), ...]} from engagement.

    Same shape as load_groups() so both label sources merge into one pair pool.
    The "eng:" key prefix stops these groups colliding with rg-label groups.

    GROUPED BY TURN, NOT BY CONTEXT HASH. A within-query pair is only meaningful
    if both members were candidates for the SAME retrieval; grouping by
    hash(context_text) silently violates that for any layer whose context is a
    constant placeholder. `project-bootstrap` ("project standing context") and
    `correction-bootstrap` ("behavioural corrections") are standing context with
    no prompt to embed against, so every session shares one string: measured
    here, that pooled 70 and 56 unrelated sessions into two groups which alone
    produced 68,700 and 3,060 pairs — 99.8% of the total, all of them teaching a
    query-free popularity prior, and outnumbering the genuine agent-rg pairs
    ~800:1 once merged.

    Keying on (session_id, turn_index) makes a group exactly one retrieval. A
    placeholder-context turn then forms its own small group and contributes only
    if that single turn saw both classes, which is the honest reading.
    """
    import pysqlite3 as sqlite3
    from cairn.relevance import _eph_path, _durable_path

    try:
        from cairn.label_relevance import _memtext
    except Exception:
        _memtext = None

    econn = sqlite3.connect(_eph_path(eph_path))
    dconn = sqlite3.connect(_durable_path(durable_path))
    groups = {}
    try:
        # `engaged IS NOT NULL` is NOT sufficient. Historical untagged rows
        # recorded engagement only when it was detected, so they contribute
        # 1,699 positives and zero negatives — filtering on non-nullity alone
        # admits them and inflates the positive class with rows drawn from a
        # regime that could never produce a negative (spec 1.7). Restrict to
        # strata that contain BOTH classes; neutralised rows come back with
        # engaged=None, which _engagement_grade already drops.
        from cairn.query import load_qualifying_strata, _neutralise_unusable_engagement
        qualifying = load_qualifying_strata(econn)
        try:
            rows = econn.execute(
                "SELECT context_text, memory_id, engaged, engaged_score, grade, "
                "engaged_method, session_id, turn_index FROM memory_deliveries "
                "WHERE engaged IS NOT NULL "
                "AND context_text IS NOT NULL AND context_text != ''"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = [r[:5] + (None,) + r[5:] for r in econn.execute(
                "SELECT context_text, memory_id, engaged, engaged_score, grade, "
                "session_id, turn_index FROM memory_deliveries WHERE engaged IS NOT NULL "
                "AND context_text IS NOT NULL AND context_text != ''"
            ).fetchall()]
        for ctx, mid, engaged, escore, grade, method, sess, turn in rows:
            engaged, escore = _neutralise_unusable_engagement(
                engaged, escore, method, qualifying)
            g = _engagement_grade(engaged, escore, grade, min_pos)
            if g is None:
                continue
            mem = None
            if _memtext is not None:
                try:
                    # enrich=False — must match the inference-time candidate format.
                    mem = _memtext(dconn, mid, enrich=False)
                except Exception:
                    mem = None
            if not mem:
                r = dconn.execute(
                    "SELECT content, topic, keywords FROM memories WHERE id = ?", (mid,)
                ).fetchone()
                mem = " ".join(p for p in r if p) if r else None
            if not mem:
                continue
            groups.setdefault(f"eng:{sess}:{turn}", []).append((ctx, mem, g))
    finally:
        econn.close()
        dconn.close()
    # A group yields pairs only if it holds both a positive and a negative.
    return {k: v for k, v in groups.items()
            if any(g == 3 for _, _, g in v) and any(g == 0 for _, _, g in v)}


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
    ap.add_argument("--floor-sample", type=int, default=800, help="labels sampled to calibrate the shipped floor")
    ap.add_argument("--min-gap", type=int, default=2, help="min grade gap for a pair (2=extremes only)")
    ap.add_argument("--enrich", action="store_true", help="render passages with keywords+facts (must match inference)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify wiring")
    ap.add_argument("--engagement", action="store_true",
                    help="merge behavioural engagement weak labels into TRAIN pairs only")
    ap.add_argument("--engagement-max-pairs", type=int, default=2000)
    ap.add_argument("--engagement-min-pos", type=float, default=ENGAGEMENT_MIN_POS_DEFAULT,
                    help="min engaged_score for a weak positive")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = torch.device(args.device)

    groups = load_groups(args.labels, args.enrich)
    if not groups:
        sys.exit(f"no labels in {args.labels}")
    train_g, held_g = split_by_query(groups, args.heldout_frac)
    tr_pairs = make_pairs(train_g, args.max_pairs, args.min_gap)
    if args.engagement:
        # TRAIN-ONLY by construction: merged after split_by_query, so held-out
        # stays pure agent-rg and the deploy gate is never judged on weak labels.
        eng_groups = load_engagement_groups(args.engagement_min_pos)
        eng_pairs = make_pairs(eng_groups, args.engagement_max_pairs, min_gap=2, seed=11)
        tr_pairs += eng_pairs
        print(f"engagement: {len(eng_groups)} groups -> {len(eng_pairs)} weak pairs "
              f"(train-only)", file=sys.stderr)
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

    # CHANCE-LINE CONTROL. Pairwise agreement has a 50% null: guessing scores 0.5.
    # "Beat the incumbent" is only meaningful when the incumbent beats a coin, and
    # the first live run had an incumbent at 39.6% — BELOW chance. A ranker that
    # loses to random is not weak, it is anti-correlated with the labels, which
    # means either the scoring is inverted or the labels are. Promoting against
    # such a baseline reads as a decisive win while proving nothing, so it is
    # blocked rather than merely reported.
    CHANCE = 0.5
    inc_below_chance = inc_agr is not None and inc_agr < CHANCE
    student_below_chance = ft_agr is not None and ft_agr < CHANCE

    beats = ft_agr > (inc_agr or 0) + args.deploy_margin
    if inc_below_chance or student_below_chance:
        beats = False
    promote_safe = ft_agr >= args.gate
    print("\n=== reranker student result (held-out clear pairs) ===")
    print(f"chance (pairwise null): {CHANCE*100:.0f}%")
    print(f"incumbent [{inc_name.split('/')[-1]}]: {inc_agr*100:.1f}%"
          f"{'   <-- BELOW CHANCE' if inc_below_chance else ''}")
    print(f"student   [finetuned {args.base.split('/')[-1]}]: {ft_agr*100:.1f}%"
          f"{'   <-- BELOW CHANCE' if student_below_chance else ''}")
    if inc_below_chance:
        print("BLOCKED: the incumbent scores below chance, so beating it is not "
              "evidence of quality. Inspect for inverted scoring or inverted "
              "labels before trusting any comparison against it.", file=sys.stderr)
    if student_below_chance:
        print("BLOCKED: the student itself scores below chance.", file=sys.stderr)
    print(f"DEPLOY (student > incumbent + {args.deploy_margin*100:.0f}pt, both above chance): "
          f"{'YES' if beats else 'NO'}"
          f"   [auto-promote-safe (>= {args.gate*100:.0f}%): {'yes' if promote_safe else 'no'}]")
    if beats and not args.smoke:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out); tok.save_pretrained(args.out)
        print(f"saved -> {args.out}")
        # Calibrate + SHIP the suppression floor with the model (floor.txt in the model
        # dir, read by resolve_reranker) so it never goes stale on retrain — the score
        # scale shifts each retrain, so a hand-set config constant would silently mis-gate.
        try:
            from cairn.calibrate_bge_floor import load_labels, bge_scores, recommend_floor
            cp, cg = load_labels(args.labels, enrich=args.enrich)
            k = list(range(len(cp))); random.Random(0).shuffle(k); k = k[:args.floor_sample]
            cp, cg = [cp[i] for i in k], [cg[i] for i in k]
            (fl, kr, dn, klb), _ = recommend_floor(bge_scores(cp, args.device, model=args.out), cg, 0.95)
            with open(os.path.join(args.out, "floor.txt"), "w") as fh:
                fh.write(f"{fl:.4f}\n")
            print(f"calibrated floor -> {fl:.3f} (keep_rel {kr*100:.0f}% / drop_noise {dn*100:.0f}% "
                  f"/ keep_LB {klb*100:.0f}%) -> {args.out}/floor.txt")
        except Exception as e:
            print(f"floor calibration skipped ({e}); resolve_reranker falls back to config default")
    else:
        print("NOT saved — student does not beat the incumbent on the test set.")


if __name__ == "__main__":
    main()
