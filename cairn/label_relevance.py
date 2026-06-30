#!/usr/bin/env python3
"""Generate relevance silver-labels for the local relevance-gate training set.

Judges real (current-prompt, delivered-memory) pairs 0-3 with an LLM, via a
CLEAN claude -p invocation that BYPASSES all cairn injection so the judge sees
ONLY the pair — never injected context that would poison the labels.

Clean combo (baked in): the judge subprocess runs with
  ANTHROPIC_BASE_URL unset   -> bypass the cairn proxy (no <!--cairn-context-->)
  CAIRN_ENABLED=0            -> disable the prompt-hook injection
  CAIRN_MODE=read-only       -> don't write the 1000s of judge calls into cairn
It still routes through `claude -p` / OAuth, so it draws on the subscription quota.

A CONTAMINATION TRIPWIRE aborts if any judge response contains a cairn artifact
marker — contamination can never silently re-enter the training set.

Resumable: appends JSONL incrementally and skips memory_ids already labelled in
the output file, so spare-quota windows can be chained across days/machines.

Usage:
  python3 cairn/label_relevance.py --n 200 --model claude-opus-4-8 --out labels.jsonl
  python3 cairn/label_relevance.py --n 50  --model claude-haiku-4-5-20251001 --out gold_clean.jsonl --revalidate
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
import pysqlite3 as sqlite3
from cairn.config import EPHEMERAL_DB_PATH
import cairn.query as q

CLEAN_ENV = {**os.environ, "CAIRN_ENABLED": "0", "CAIRN_MODE": "read-only"}
CLEAN_ENV.pop("ANTHROPIC_BASE_URL", None)
MARKER_RE = re.compile(r"cairn[_-]context|<memory>|<cairn_context|\[cm\]:", re.IGNORECASE)

JUDGE_HEADER = (
    "You are grading whether a retrieved MEMORY is relevant to a developer's QUERY.\n"
    "Grade each item 0-3: 0=irrelevant/noise, 1=weak/tangential, 2=relevant, 3=directly on-point.\n"
    "Output ONLY lines of the form `N: G` (item number, colon, grade). No other text.\n")


def current_prompt(ctx: str, cap: int = 1500) -> str:
    """Extract the current prompt from a build_context_window string, robust to
    both old (current prompt last) and new (current prompt first) orderings."""
    i = ctx.find("[user] ")
    if i < 0:
        return ctx.strip()[:cap]
    seg = ctx[i + len("[user] "):]
    for mk in ("\n[prev user]", "\n[prev assistant]"):
        j = seg.find(mk)
        if j >= 0:
            seg = seg[:j]
    return seg.strip()[:cap]


def sample_pairs(n: int, stratify: bool, seen_ids: set) -> list[dict]:
    e = sqlite3.connect(EPHEMERAL_DB_PATH); e.execute("PRAGMA busy_timeout=5000")
    d = sqlite3.connect(q.DB_PATH); d.execute("PRAGMA busy_timeout=5000")
    # deterministic pseudo-random order; optionally stratify by engaged for class balance
    def fetch(where, lim):
        return e.execute(
            "SELECT memory_id, context_text FROM memory_deliveries "
            "WHERE context_text IS NOT NULL AND length(context_text)>30 " + where +
            " GROUP BY memory_id ORDER BY (memory_id*2654435761) % 7919 LIMIT ?", (lim,)).fetchall()
    rows = (fetch("AND engaged=1", n) + fetch("AND (engaged=0 OR engaged IS NULL)", n)) if stratify else fetch("", n*3)
    out = []
    for mid, ctx in rows:
        if mid in seen_ids or any(o["memory_id"] == mid for o in out):
            continue
        r = d.execute("SELECT type,topic,content FROM memories WHERE id=?", (mid,)).fetchone()
        if not r:
            continue
        out.append({"memory_id": mid, "query": current_prompt(ctx),
                    "mem": f"{r[0]} {r[1]}: {r[2]}"[:600]})
        if len(out) >= n:
            break
    return out


def judge_batch(batch: list[dict], model: str, offset: int) -> dict:
    lines = [JUDGE_HEADER]
    for i, it in enumerate(batch):
        lines.append(f"ITEM {offset+i+1}\nQUERY: {it['query']}\nMEMORY: {it['mem']}\n")
    prompt = "\n".join(lines)
    res = subprocess.run(["claude", "-p", prompt, "--model", model],
                         capture_output=True, text=True, timeout=300, env=CLEAN_ENV)
    out = res.stdout
    if MARKER_RE.search(out):
        sys.exit(f"CONTAMINATION TRIPWIRE: cairn artifact in judge response — ABORTING.\n{out[:400]}")
    return {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"(\d+)\s*:\s*([0-3])", out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--no-stratify", action="store_true")
    args = ap.parse_args()

    seen = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try: seen.add(json.loads(line)["memory_id"])
            except Exception: pass
    print(f"resuming: {len(seen)} already labelled in {args.out}", file=sys.stderr)

    items = sample_pairs(args.n, not args.no_stratify, seen)
    print(f"sampled {len(items)} new pairs to label with {args.model}", file=sys.stderr)
    f = open(args.out, "a")
    done = 0
    for off in range(0, len(items), args.batch):
        batch = items[off:off+args.batch]
        grades = judge_batch(batch, args.model, off)
        for i, it in enumerate(batch):
            g = grades.get(off+i+1)
            if g is None:
                continue
            it["grade"] = g
            f.write(json.dumps(it) + "\n"); f.flush()
            done += 1
        print(f"  batch {off//args.batch+1}: +{len([1 for i in range(len(batch)) if grades.get(off+i+1) is not None])} (total {done})", file=sys.stderr)
    f.close()
    print(f"done: {done} labelled -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
