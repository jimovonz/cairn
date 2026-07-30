#!/usr/bin/env python3
"""Export this node's reranker training labels as a portable archive.

WHY THIS EXISTS: the trained student ships as a per-machine artifact
(`training_data/` is gitignored), so a node that has not trained falls back to
the pretrained base. Moving the MODEL is one half of the portability problem;
this is the other half — moving the LABELS, so one node can train on the pooled
observations of several.

WHAT TRAVELS, AND WHY ONLY THIS:
  * The label payload is text, not database. `memory_deliveries` here is 2.0 GB,
    almost entirely `context_vec` BLOBs and unlabelled rows; the labels inside it
    are a few hundred KB. Shipping the DB would move ~4000x the useful bytes and
    would also force a schema-version handshake at the far end.
  * Passages are rendered ONCE, at export, via the same `render_passage` the
    trainer and inference use. The far end therefore needs none of this node's
    memories to read the file.

MEMORY IDS ARE DELIBERATELY NOT EMITTED AS `memory_id`. `train_reranker.
load_groups(enrich=True)` re-renders a passage from `memory_id` against the
LOCAL durable DB. Memory ids are per-node autoincrements, so id 4212 here is an
unrelated memory there — `--enrich` on an imported file would silently swap in
wrong passages, producing labels attached to text no agent ever judged. The id
is preserved as `src_memory_id` (which no loader reads) for provenance only, so
`--enrich` degrades safely to the shipped `mem` text.

THE TWO LABEL SOURCES STAY IN SEPARATE FILES. Agent `rg` grades are the eval-
grade signal; engagement pseudo-grades are weak labels. `train_reranker` merges
engagement only AFTER `split_by_query`, so the held-out set stays pure agent-rg
and the beat-the-incumbent deploy gate is never judged on weak labels. Flatten
them into one file and that discipline is silently destroyed — the held-out
would contain weak labels and the gate would be measuring the wrong thing.

Usage:
    .venv/bin/python cairn/export_training_data.py                # -> exports/
    .venv/bin/python cairn/export_training_data.py --out DIR --node-id home-pc
    .venv/bin/python cairn/export_training_data.py --stats        # count only
"""

import argparse
import gzip
import json
import os
import platform
import socket
import subprocess
import sys
import tarfile

try:
    import pysqlite3 as sqlite3
except ImportError:  # pragma: no cover - guarded by tests/test_sqlite_guard.py
    if os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") != "1":
        raise RuntimeError(
            "cairn requires pysqlite3; set CAIRN_ALLOW_STDLIB_SQLITE=1 to override"
        )
    import sqlite3

SCHEMA_VERSION = 1


def _ro(path):
    """Read-only connection. The exporter must never be able to write a label
    DB: it runs against live WAL files while the daemon is up."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _node_id(explicit=None):
    return explicit or os.environ.get("CAIRN_NODE_ID") or socket.gethostname()


def _git_commit():
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def collect_rg(durable_path=None, eph_path=None):
    """Agent `rg` grades -> [{query, mem, grade, ...}]. The eval-grade signal.

    Rows whose memory has since been deleted are dropped: the grade was a
    judgement about a specific passage, and without that text it cannot be
    reattached to anything.
    """
    from cairn.label_relevance import _memtext
    from cairn.relevance import _eph_path, _durable_path

    e = _ro(_eph_path(eph_path))
    d = _ro(_durable_path(durable_path))
    out, dropped = [], 0
    try:
        rows = e.execute(
            "SELECT context_text, memory_id, grade, hard_negative, layer, scope, "
            "reranker_model, ce_score, delivered_at FROM memory_deliveries "
            "WHERE grade IS NOT NULL AND context_text IS NOT NULL AND context_text != ''"
        ).fetchall()
        for ctx, mid, grade, hard_neg, layer, scope, rr, ce, at in rows:
            # enrich=False — must match the inference-time candidate format.
            mem = _memtext(d, mid, enrich=False)
            if not mem:
                dropped += 1
                continue
            out.append({
                "query": ctx,
                "mem": mem,
                "grade": int(grade),
                # Provenance only. NOT `memory_id` — see module docstring.
                "src_memory_id": int(mid),
                "hard_negative": int(hard_neg or 0),
                "layer": layer,
                "scope": scope,
                "reranker_model": rr,
                "ce_score": ce,
                "delivered_at": at,
            })
    finally:
        e.close()
        d.close()
    return out, dropped


def collect_engagement(min_pos=None):
    """Behavioural engagement pseudo-grades -> [{query, mem, grade, group, weak}].

    Delegates to the trainer's own loader so the qualifying-strata filter is
    applied identically here and at training time. That filter matters: untagged
    historical rows recorded engagement only when detected, contributing
    positives and zero negatives, and admitting them inflates the positive class
    with a regime that could never produce a negative.

    `group` is preserved so the far end can keep these out of the held-out split.
    """
    from cairn.train_reranker import (ENGAGEMENT_MIN_POS_DEFAULT,
                                      load_engagement_groups)
    groups = load_engagement_groups(
        min_pos=ENGAGEMENT_MIN_POS_DEFAULT if min_pos is None else min_pos)
    return [
        {"query": q, "mem": m, "grade": int(g), "group": key, "weak": True}
        for key, items in groups.items() for q, m, g in items
    ]


def collect_fit(durable_path=None):
    """Relative fit pairs -> [{winner, loser, ...}] with passages resolved.

    No trainer path consumes these yet (the student is trained on pointwise
    grades via MarginRankingLoss over induced order; fit pairs are already
    pairwise and would feed that loss directly). Exported now so the format is
    settled before volume accumulates.
    """
    from cairn.label_relevance import _memtext
    from cairn.relevance import _durable_path

    d = _ro(_durable_path(durable_path))
    out, dropped = [], 0
    try:
        try:
            rows = d.execute(
                "SELECT session_id, turn_index, winner_id, loser_id, created_at "
                "FROM delivery_fit_pairs").fetchall()
        except sqlite3.OperationalError:
            return [], 0
        for sess, turn, win, lose, at in rows:
            wm, lm = _memtext(d, win, enrich=False), _memtext(d, lose, enrich=False)
            if not wm or not lm:
                dropped += 1
                continue
            out.append({
                "winner": wm, "loser": lm,
                "src_winner_id": int(win), "src_loser_id": int(lose),
                "session_id": sess, "turn_index": turn, "created_at": at,
            })
    finally:
        d.close()
    return out, dropped


def _write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return os.path.getsize(path)


def build(out_dir, node_id=None, stats_only=False):
    from cairn import config

    node = _node_id(node_id)
    rg, rg_dropped = collect_rg()
    eng = collect_engagement()
    fit, fit_dropped = collect_fit()

    counts = {
        "rg_labels": len(rg),
        "rg_dropped_missing_memory": rg_dropped,
        "rg_distinct_queries": len({r["query"] for r in rg}),
        "engagement_labels": len(eng),
        "engagement_groups": len({r["group"] for r in eng}),
        "engagement_positive": sum(1 for r in eng if r["grade"] == 3),
        "fit_pairs": len(fit),
        "fit_dropped_missing_memory": fit_dropped,
    }
    if stats_only:
        return counts, None

    os.makedirs(out_dir, exist_ok=True)
    stem = f"cairn-labels-{node}"
    paths = {}
    paths["relevance_silver.jsonl.gz"] = _write_jsonl_gz(
        os.path.join(out_dir, "relevance_silver.jsonl.gz"), rg)
    paths["engagement_weak.jsonl.gz"] = _write_jsonl_gz(
        os.path.join(out_dir, "engagement_weak.jsonl.gz"), eng)
    paths["fit_pairs.jsonl.gz"] = _write_jsonl_gz(
        os.path.join(out_dir, "fit_pairs.jsonl.gz"), fit)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "node_id": node,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "cairn_version": getattr(__import__("cairn"), "__version__", None),
        "generation_prompt_version": getattr(config, "GENERATION_PROMPT_VERSION", None),
        "reranker_at_export": list(config.resolve_reranker()),
        "counts": counts,
        "files": {k: {"bytes": v} for k, v in paths.items()},
        "notes": [
            "Passages pre-rendered with render_passage(enrich=False) to match "
            "inference-time candidate format.",
            "src_memory_id is provenance only; memory ids are per-node "
            "autoincrements and MUST NOT be fed to load_groups(enrich=True).",
            "engagement_weak.jsonl.gz holds WEAK labels — merge only AFTER "
            "split_by_query so held-out stays pure agent-rg.",
            "context_text contains prompt snippets; treat as private and do not "
            "commit (training_data/ and exports/ are gitignored).",
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    readme = f"""# Cairn label export — node `{node}`

Counts: {json.dumps(counts, indent=2)}

## Train on the receiving node

    mkdir -p cairn/training_data
    gunzip -c relevance_silver.jsonl.gz >> cairn/training_data/relevance_silver.jsonl
    .venv/bin/python cairn/train_reranker.py \\
        --labels cairn/training_data/relevance_silver.jsonl \\
        --base BAAI/bge-reranker-base --device cuda

Appending pools this node's labels with the receiving node's own. The deploy
gate is beat-the-incumbent, so a pooled run that does not improve simply will
not be saved.

## Do NOT pass --enrich to an imported file

`--enrich` re-renders passages from `memory_id` against the LOCAL durable DB.
This export omits `memory_id` for that reason (it is `src_memory_id`), so
`--enrich` degrades safely — but do not rename the field back.

## engagement_weak.jsonl.gz

Weak labels, already filtered to qualifying strata. `--engagement` reads the
LOCAL DBs, so consuming this file needs a loader flag that merges it after
`split_by_query`. Until that exists, these rows are archival.

## fit_pairs.jsonl.gz

Relative fit labels (winner/loser passages). No trainer path yet.
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(readme)

    archive = os.path.join(out_dir, f"{stem}.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("relevance_silver.jsonl.gz", "engagement_weak.jsonl.gz",
                     "fit_pairs.jsonl.gz", "manifest.json", "README.md"):
            tar.add(os.path.join(out_dir, name), arcname=f"{stem}/{name}")
    return counts, archive


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports"))
    ap.add_argument("--node-id", default=None,
                    help="identifier for this node (default: $CAIRN_NODE_ID or hostname)")
    ap.add_argument("--stats", action="store_true", help="count labels, write nothing")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    counts, archive = build(a.out, a.node_id, stats_only=a.stats)
    if a.json:
        print(json.dumps({"counts": counts, "archive": archive}, indent=2))
        return 0
    for k, v in counts.items():
        print(f"{k:32s} {v}")
    if archive:
        print(f"\narchive: {archive} ({os.path.getsize(archive)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
