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
import argparse, hashlib, json, os, re, subprocess, sys
import pysqlite3 as sqlite3
from cairn.config import EPHEMERAL_DB_PATH
import cairn.query as q

CLEAN_ENV = {**os.environ, "CAIRN_ENABLED": "0", "CAIRN_MODE": "read-only"}
CLEAN_ENV.pop("ANTHROPIC_BASE_URL", None)
# FATAL: genuine input-injection markers — prove cairn context leaked INTO the judge.
INJECTION_RE = re.compile(r"cairn[_-]context|<memory>|<cairn_context", re.IGNORECASE)
# BENIGN: the judge's own [cm] output block (it reads CLAUDE.md and self-annotates) — stripped, not fatal.
CM_BLOCK_RE = re.compile(r"(?m)^\s*\[cm\]:.*$")
USAGE_LIMIT_RE = re.compile(r"usage limit|rate limit|quota|exceeded your|Claude usage|Please try again", re.IGNORECASE)
# Replaces the default system prompt so CLAUDE.md-driven agent behaviour (preamble,
# [cm] blocks, tool routing) can't corrupt the grader. Combined with cwd=/tmp (no
# project CLAUDE.md) this keeps the judge a pure grader.
JUDGE_SYSTEM = ("You are a precise relevance grader. Follow the user's grading "
                "instructions exactly. Output ONLY the requested `N: G` lines — no "
                "preamble, no explanation, no restatement of the items, no memory "
                "blocks, no other text.")
NEUTRAL_CWD = "/tmp"

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


def qhash(qy: str) -> str:
    """Stable short hash of a query — the pair identity is (memory_id, qhash) so a
    memory delivered against many different queries yields many distinct pairs."""
    return hashlib.sha1(qy.encode("utf-8", "ignore")).hexdigest()[:12]


def _memtext(d, mid, enrich=False):
    r = d.execute("SELECT type,topic,content,keywords,facts FROM memories WHERE id=?", (mid,)).fetchone()
    if not r:
        return None
    from cairn.passage import render_passage
    return render_passage(r[0], r[1], r[2], r[3], r[4], enrich=enrich)


def sample_pairs(n: int, stratify: bool, seen: set, per_delivery: bool = False) -> list[dict]:
    e = sqlite3.connect(EPHEMERAL_DB_PATH); e.execute("PRAGMA busy_timeout=5000")
    d = sqlite3.connect(q.DB_PATH); d.execute("PRAGMA busy_timeout=5000")

    if per_delivery:
        # Distinct (memory_id, delivery-context) rows: the SAME memory judged against
        # DIFFERENT queries — the contrasts a relevance gate must learn. `seen` is a
        # set of (memory_id, qhash). Deterministic pseudo-random order for resumability.
        rows = e.execute(
            "SELECT DISTINCT memory_id, context_text FROM memory_deliveries "
            "WHERE context_text IS NOT NULL AND length(context_text)>30 "
            "ORDER BY (memory_id*2654435761 + id*40503) % 100003 LIMIT ?", (n * 6,)).fetchall()
        out, local = [], set()
        for mid, ctx in rows:
            qy = current_prompt(ctx); key = (mid, qhash(qy))
            if key in seen or key in local:
                continue
            mem = _memtext(d, mid)
            if not mem:
                continue
            local.add(key)
            out.append({"memory_id": mid, "query": qy, "mem": mem})
            if len(out) >= n:
                break
        return out

    # default: one pair per memory; optionally stratify by engaged for class balance.
    def fetch(where, lim):
        return e.execute(
            "SELECT memory_id, context_text FROM memory_deliveries "
            "WHERE context_text IS NOT NULL AND length(context_text)>30 " + where +
            " GROUP BY memory_id ORDER BY (memory_id*2654435761) % 7919 LIMIT ?", (lim,)).fetchall()
    rows = (fetch("AND engaged=1", n) + fetch("AND (engaged=0 OR engaged IS NULL)", n)) if stratify else fetch("", n*3)
    out = []
    for mid, ctx in rows:
        if mid in seen or any(o["memory_id"] == mid for o in out):
            continue
        mem = _memtext(d, mid)
        if not mem:
            continue
        out.append({"memory_id": mid, "query": current_prompt(ctx), "mem": mem})
        if len(out) >= n:
            break
    return out


def judge_batch(batch: list[dict], model: str, offset: int):
    """Returns (grades, ok, err). ok=False signals a real call failure (nonzero
    exit / usage-limit / empty stdout) so the caller can stop on quota exhaustion
    rather than spin on 300s timeouts. Parse-empty on a healthy call is ok=True."""
    lines = [JUDGE_HEADER]
    for i, it in enumerate(batch):
        lines.append(f"ITEM {offset+i+1}\nQUERY: {it['query']}\nMEMORY: {it['mem']}\n")
    prompt = "\n".join(lines)
    try:
        res = subprocess.run(
            ["claude", "-p", prompt, "--model", model, "--system-prompt", JUDGE_SYSTEM],
            capture_output=True, text=True, timeout=300, env=CLEAN_ENV, cwd=NEUTRAL_CWD)
    except subprocess.TimeoutExpired:
        return {}, False, "timeout"
    except OSError as e:
        # e.g. FileNotFoundError if `claude` momentarily vanishes during a CLI
        # auto-update. Treat as a failed batch so the max-empty guard tolerates a
        # transient disappearance (streak resets when it returns) instead of crashing.
        return {}, False, f"exec failed: {e}"
    raw = res.stdout or ""
    # Contamination check robust to this corpus being FULL of cairn-self-referential
    # memories: a marker in the OUTPUT is fatal ONLY if it was NOT in the batch we sent
    # (real injection into the judge), never when the judge merely echoes/restates
    # memory content that legitimately contains the marker as data. Env-level bypass
    # (CAIRN_ENABLED=0, no proxy) is the actual guarantee; this is the backstop.
    batch_blob = "\n".join(f"{it['query']}\n{it['mem']}" for it in batch)
    sent = {m.group(0).lower() for m in INJECTION_RE.finditer(batch_blob)}
    leaked = {m.group(0).lower() for m in INJECTION_RE.finditer(raw)} - sent
    if leaked:
        sys.exit(f"CONTAMINATION TRIPWIRE: injection marker(s) in judge response not "
                 f"present in our input {sorted(leaked)} — ABORTING.\n{raw[:400]}")
    # Strip the judge's own [cm] self-annotation before parsing (benign output artifact).
    out = CM_BLOCK_RE.sub("", raw)
    if res.returncode != 0 or not out.strip() or USAGE_LIMIT_RE.search(out) \
            or USAGE_LIMIT_RE.search(res.stderr or ""):
        return {}, False, ((res.stderr or out).strip()[:200] or "empty/nonzero")
    # line-anchored so an inline restatement ("211: query ...") can't mis-parse as a grade
    grades = {int(m.group(1)): int(m.group(2))
              for m in re.finditer(r"(?m)^\s*(\d+)\s*:\s*([0-3])\b", out)}
    return grades, True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--max-empty", type=int, default=3,
                    help="abort after this many consecutive failed batches (quota-exhaustion guard)")
    ap.add_argument("--per-delivery", action="store_true",
                    help="sample distinct (memory,query) delivery pairs, not one-per-memory")
    args = ap.parse_args()

    seen = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                d = json.loads(line)
                # per-delivery keys on (memory_id, qhash); default keys on memory_id.
                seen.add((d["memory_id"], qhash(d.get("query", ""))) if args.per_delivery
                         else d["memory_id"])
            except Exception: pass
    mode = "per-delivery (memory,query)" if args.per_delivery else "one-per-memory"
    print(f"resuming: {len(seen)} already labelled in {args.out} [{mode}]", file=sys.stderr)

    items = sample_pairs(args.n, not args.no_stratify, seen, per_delivery=args.per_delivery)
    print(f"sampled {len(items)} new pairs to label with {args.model}", file=sys.stderr)
    f = open(args.out, "a")
    done = 0
    empty_streak = 0
    for off in range(0, len(items), args.batch):
        batch = items[off:off+args.batch]
        grades, ok, err = judge_batch(batch, args.model, off)
        if not ok:
            empty_streak += 1
            print(f"  batch {off//args.batch+1}: FAILED ({empty_streak}/{args.max_empty}) — {err}", file=sys.stderr)
            if empty_streak >= args.max_empty:
                print(f"aborting: {empty_streak} consecutive failed batches (likely quota exhausted). "
                      f"{done} labelled so far — re-run same command to resume.", file=sys.stderr)
                break
            continue
        empty_streak = 0
        got = 0
        for i, it in enumerate(batch):
            g = grades.get(off+i+1)
            if g is None:
                continue
            it["grade"] = g
            f.write(json.dumps(it) + "\n"); f.flush()
            done += 1; got += 1
        print(f"  batch {off//args.batch+1}: +{got} (total {done})", file=sys.stderr)
    f.close()
    print(f"done: {done} labelled -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
