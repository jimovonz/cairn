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

SCHEMA_VERSION = 2


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

    Ranking provenance travels with the label because `ce_score` is NOT
    comparable across nodes or across time: it is a raw logit from whichever
    model scored that delivery, and this corpus spans an ms-marco -> bge
    transition plus a locally trained student. `reranker_model` is what makes a
    score interpretable, and a NULL there means the gate was DOWN for that
    delivery — not that the layer is ungated. A receiver that cannot tell those
    apart will manufacture ungated-vs-reranked conclusions.

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
            "reranker_model, ce_score, score_components, served_rank, gate_status, "
            "engaged, engaged_score, engaged_method, session_id, turn_index, "
            "delivered_at FROM memory_deliveries "
            "WHERE grade IS NOT NULL AND context_text IS NOT NULL AND context_text != ''"
        ).fetchall()
        for (ctx, mid, grade, hard_neg, layer, scope, rr, ce, comps, rank, gate,
             engaged, escore, emethod, sess, turn, at) in rows:
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
                # NULL reranker_model = gate was down for this delivery.
                "reranker_model": rr,
                "ce_score": ce,
                "score_components": comps,
                "served_rank": rank,
                "gate_status": gate,
                # Behavioural signal alongside the agent grade, so the receiver
                # can cross-check the two rather than treating them as one.
                "engaged": engaged,
                "engaged_score": escore,
                "engaged_method": emethod,
                "src_session_id": sess,
                "turn_index": turn,
                "delivered_at": at,
            })
    finally:
        e.close()
        d.close()
    return out, dropped


def collect_engagement(min_pos=None, durable_path=None, eph_path=None):
    """Behavioural engagement pseudo-grades -> [{query, mem, grade, group, ...}].

    Reimplements `train_reranker.load_engagement_groups` rather than calling it,
    for one reason: that loader returns only (query, mem, grade), and the
    receiving node needs the RAW `engaged` / `engaged_score` / `engaged_method`
    to re-derive the filter itself instead of trusting this exporter's word for
    it. Every semantic decision below reuses the trainer's own helpers, so the
    two cannot drift: `load_qualifying_strata`, `_neutralise_unusable_engagement`,
    `_engagement_grade`, the `eng:` group prefix, and the final both-classes
    filter are all the trainer's.

    The strata filter is the load-bearing part. Untagged historical rows recorded
    engagement only when it was DETECTED, so they contribute positives and zero
    negatives; admitting them inflates the positive class with a regime that
    could never have produced a negative. Rows it neutralises come back as
    engaged=None and are dropped by `_engagement_grade`.

    Returns (rows, accounting) so the manifest can report what each stage
    removed — a reader can then audit the yield instead of inferring it.
    """
    from cairn.query import load_qualifying_strata, _neutralise_unusable_engagement
    from cairn.train_reranker import (ENGAGEMENT_MIN_POS_DEFAULT, _engagement_grade,
                                      qhash)
    from cairn.label_relevance import _memtext
    from cairn.relevance import _eph_path, _durable_path

    min_pos = ENGAGEMENT_MIN_POS_DEFAULT if min_pos is None else min_pos
    e = _ro(_eph_path(eph_path))
    d = _ro(_durable_path(durable_path))
    acct = {"candidate_rows": 0, "dropped_by_strata_or_grade": 0,
            "dropped_missing_memory": 0, "dropped_single_class_group": 0}
    groups = {}
    try:
        qualifying = load_qualifying_strata(e)
        rows = e.execute(
            "SELECT context_text, memory_id, engaged, engaged_score, grade, "
            "engaged_method, layer, scope, reranker_model, session_id, turn_index "
            "FROM memory_deliveries WHERE engaged IS NOT NULL "
            "AND context_text IS NOT NULL AND context_text != ''"
        ).fetchall()
        acct["candidate_rows"] = len(rows)
        for (ctx, mid, engaged, escore, agrade, emethod, layer, scope, rr,
             sess, turn) in rows:
            neu_engaged, neu_score = _neutralise_unusable_engagement(
                engaged, escore, emethod, qualifying)
            g = _engagement_grade(neu_engaged, neu_score, agrade, min_pos)
            if g is None:
                acct["dropped_by_strata_or_grade"] += 1
                continue
            mem = _memtext(d, mid, enrich=False)
            if not mem:
                acct["dropped_missing_memory"] += 1
                continue
            groups.setdefault("eng:" + qhash(ctx), []).append({
                "query": ctx,
                "mem": mem,
                "grade": int(g),
                "weak": True,
                "src_memory_id": int(mid),
                # Raw, pre-neutralisation values: these are what let the receiver
                # reconstruct the strata filter rather than trust it.
                "engaged_raw": engaged,
                "engaged_score_raw": escore,
                "engaged_method": emethod,
                "agent_grade": agrade,
                "layer": layer,
                "scope": scope,
                "reranker_model": rr,
                "src_session_id": sess,
                "turn_index": turn,
            })
    finally:
        e.close()
        d.close()

    # A group yields pairs only if it holds both a positive and a negative.
    out = []
    for key, items in groups.items():
        if any(i["grade"] == 3 for i in items) and any(i["grade"] == 0 for i in items):
            for i in items:
                i["group"] = key
            out.extend(items)
        else:
            acct["dropped_single_class_group"] += len(items)
    return out, acct


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
    eng, eng_acct = collect_engagement()
    fit, fit_dropped = collect_fit()

    # Labels/query is the number that decides this export's worth, so it is
    # reported rather than left to be derived. A pairwise trainer learns only
    # from WITHIN-query pairs, so a query carrying one label contributes
    # nothing — 600 labels at 1.8/query are worth far less than 300 at 6/query.
    rg_per_query = {}
    for r in rg:
        rg_per_query[r["query"]] = rg_per_query.get(r["query"], 0) + 1
    singles = sum(1 for n in rg_per_query.values() if n == 1)

    counts = {
        "rg_labels": len(rg),
        "rg_dropped_missing_memory": rg_dropped,
        "rg_distinct_queries": len(rg_per_query),
        "rg_labels_per_query": round(len(rg) / len(rg_per_query), 2) if rg_per_query else 0,
        "rg_single_label_queries": singles,
        "rg_pairable_queries": len(rg_per_query) - singles,
        "rg_gate_down_rows": sum(1 for r in rg if r["reranker_model"] is None),
        "engagement_labels": len(eng),
        "engagement_groups": len({r["group"] for r in eng}),
        "engagement_positive": sum(1 for r in eng if r["grade"] == 3),
        "engagement_accounting": eng_acct,
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
        "merge_on": "content hash of query+mem — NEVER src_memory_id",
        "absent_columns": {
            "prefilter_n": "not a column on memory_deliveries; suppression rate "
                           "is not reconstructable from this export",
            "postfilter_n": "as above",
        },
        "notes": [
            "Passages pre-rendered with render_passage(enrich=False) to match "
            "inference-time candidate format.",
            "src_memory_id is provenance only; memory ids are per-node "
            "autoincrements and MUST NOT be fed to load_groups(enrich=True).",
            "engagement_weak.jsonl.gz holds WEAK labels — merge only AFTER "
            "split_by_query so held-out stays pure agent-rg.",
            "ce_score is a raw logit from the model named in reranker_model and "
            "is NOT comparable across nodes or across the ms-marco->bge->student "
            "transitions. The grades are agent-assigned and unaffected.",
            "reranker_model NULL means the gate was DOWN for that delivery, not "
            "that the layer is ungated — exclude those rows before comparing "
            "gated against ungated.",
            "The reranker A/B arm identifier IS reranker_model (an arm is a "
            "model), so matched arms across nodes require identical "
            "CAIRN_RERANKER_AB_ARMS and CAIRN_RERANKER_AB=1 on both.",
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

## Merging

Merge on a content hash of `query + mem`. **Never on `src_memory_id`** — ids are
per-node autoincrements.

`ce_score` is a raw logit from the model named in `reranker_model` and is not
comparable across nodes or across the ms-marco -> bge -> student transitions.
The grades themselves are agent-assigned and unaffected. A NULL
`reranker_model` means the gate was down for that delivery, not that the layer
is ungated; exclude those before comparing gated against ungated.

## engagement_weak.jsonl.gz

Weak labels, already filtered to qualifying strata — but the raw
`engaged_raw` / `engaged_score_raw` / `engaged_method` ride along so the
receiver can reconstruct that filter rather than trust it, and
`manifest.counts.engagement_accounting` reports what each stage dropped.

Merge only AFTER `split_by_query`, so the held-out split stays pure agent-rg
and the beat-the-incumbent deploy gate is never judged on weak labels.
`--engagement` reads the LOCAL DBs, so consuming this file needs a loader flag.

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
