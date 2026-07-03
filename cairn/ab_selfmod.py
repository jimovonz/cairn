"""Automated write-side A/B assessment + changeover.

Cairn runs a live per-prompt A/B test on the memory-generation rules
(cairn/config.py: AB_TEST_ENABLED, AB_ARM_VERSIONS, AB_B_INSTRUCTION).
Outcomes are measurable via `query.py --delivery-stats`, but until now
deciding a winner and updating config.py was entirely manual (done once
by hand for genB-v1 -> genA-v4, per the history comment above
GENERATION_PROMPT_VERSION in config.py).

This module ports the min-N + rate-threshold decision pattern already
used by cairn/calibration_selfmod.py (auto_promote_corroborated /
auto_archive_low_follow) to the write-side A/B experiment:

- `get_or_create_running_experiment` — self-bootstraps an `ab_experiments`
  row from the live config the first time it's assessed (no separate
  "start experiment" step needed since AB_B_INSTRUCTION is already
  hand-edited directly today).
- `assess_experiment` — computes each arm's delivery count + engaged% and
  applies the decision rule: promote / reject / inconclusive / (still)
  running.
- `_promote` / `_reject` — mechanically edit cairn/config.py. Promotion
  bumps GENERATION_PROMPT_VERSION (mechanical) and disables testing
  (AB_TEST_ENABLED = False) rather than inventing a new AB_B_INSTRUCTION
  hypothesis — that's an editorial step left to a human/future session.
  Rejection only disables testing, leaving GENERATION_PROMPT_VERSION
  untouched.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err


DB_PATH = os.environ.get(
    "CAIRN_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db"),
)
EPH_DB_PATH = os.environ.get(
    "CAIRN_EPHEMERAL_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db"),
)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")

# Decision-rule thresholds — bounded, conservative, same shape as
# calibration_selfmod's AUTO_PROMOTE_MIN_DELIVERIES/AUTO_PROMOTE_MIN_FOLLOW_RATE.
AB_MIN_DELIVERIES_PER_ARM = 30
AB_PROMOTE_GAP_PCT = 15.0
AB_REJECT_GAP_PCT = -15.0


def _open(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _emit_metric(event: str, detail: dict,
                  eph_path: Optional[str] = None) -> None:
    """Best-effort metric write — swallows errors so selfmod operations
    never fail because the metrics table is unreachable."""
    path = eph_path or EPH_DB_PATH
    try:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "INSERT INTO metrics (event, session_id, detail) VALUES (?, ?, ?)",
            (event, None, json.dumps(detail)),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _arm_stats(db_path: str, eph_path: str, version: str) -> tuple[int, Optional[float]]:
    """Delivery count + engaged% for memories stamped with `version` in
    memories.source_ref. Cross-database join done in Python (memories
    lives in the durable DB, memory_deliveries in the ephemeral DB) —
    same approach query.py:delivery_stats() already uses."""
    from cairn.query import _aggregate_outcomes

    d = _open(db_path)
    try:
        ids = [r[0] for r in d.execute(
            "SELECT id FROM memories WHERE source_ref = ?", (version,)
        ).fetchall()]
    finally:
        d.close()
    if not ids:
        return 0, None

    e = _open(eph_path)
    try:
        records = []
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = ",".join("?" * len(chunk))
            rows = e.execute(
                f"SELECT engaged, engaged_score, grade FROM memory_deliveries "
                f"WHERE memory_id IN ({q})", chunk
            ).fetchall()
            records.extend({"key": version, "engaged": r[0],
                            "engaged_score": r[1], "grade": r[2]} for r in rows)
    finally:
        e.close()

    if not records:
        return 0, None
    agg = _aggregate_outcomes(records)
    stat = agg.get(version)
    if not stat:
        return 0, None
    return stat["n"], stat["engaged_pct"]


# ---------------------------------------------------------------------------
# Compliance gate — mechanical check that a candidate arm actually writes
# less self-referential/meta content, independent of engaged_pct (which
# only measures downstream usefulness and cannot confirm the targeted
# writing behavior changed at all).
# ---------------------------------------------------------------------------

# Named pattern sets, one per compliance check. A check is opt-in per
# candidate (config.AB_COMPLIANCE_CHECKS / an AB_B_QUEUE entry's "compliance"
# key), so a candidate whose hypothesis is unrelated to writing style is never
# gated. Extend this dict to add a new check; register it against a candidate
# version in config to activate it.
_COMPLIANCE_PATTERNS = {
    "meta": (
        "session handoff", "in progress:", "memories captured", "were captured",
        "this session", "this conversation", "session-arc",
    ),
}

_COMPLIANCE_SIG_P = 0.05


def _compliance_check_for(candidate_version: str) -> Optional[str]:
    """Which compliance check (if any) the running candidate registered —
    resolved from its AB_B_QUEUE entry's optional "compliance" key first, then
    the standalone config.AB_COMPLIANCE_CHECKS map (which also covers a live
    candidate not in the queue, e.g. the current genB-v2). None => the gate is
    skipped entirely and the engagement verdict stands alone."""
    from cairn import config

    for cand in getattr(config, "AB_B_QUEUE", []):
        if cand.get("version") == candidate_version and cand.get("compliance"):
            return cand["compliance"]
    return getattr(config, "AB_COMPLIANCE_CHECKS", {}).get(candidate_version)


def _is_meta_flagged(topic: str, content: str, check: str = "meta") -> bool:
    """Case-insensitive substring match against the named check's pattern set
    (default "meta": self-referential/bookkeeping content — session-arc
    summaries, "this session/conversation" framing). Mechanical, not
    LLM-judged, so it is cheap and reproducible."""
    blob = f"{topic} {content}".lower()
    return any(p in blob for p in _COMPLIANCE_PATTERNS.get(check, ()))


def _meta_stats(db_path: str, version: str, check: str = "meta",
                 since: Optional[str] = None) -> tuple[int, Optional[float], int]:
    """Count, flagged%, and flagged-count for memories stamped with `version`
    in memories.source_ref, matched against the `check` pattern set. `since`
    (a SQL/ISO timestamp) bounds the pool to memories written at/after it, so
    arm A's pre-experiment history under the same base_version does not
    contaminate the controlled in-window comparison."""
    conn = _open(db_path)
    try:
        if since is not None:
            rows = conn.execute(
                "SELECT topic, content FROM memories WHERE source_ref = ? "
                "AND created_at >= ?", (version, since),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT topic, content FROM memories WHERE source_ref = ?", (version,),
            ).fetchall()
    finally:
        conn.close()
    n = len(rows)
    if n == 0:
        return 0, None, 0
    flagged = sum(1 for topic, content in rows
                  if _is_meta_flagged(topic or "", content or "", check))
    return n, 100.0 * flagged / n, flagged


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test p-value for 2x2 table [[a,b],[c,d]],
    computed by summing hypergeometric probabilities over all tables with
    the same margins that are no more likely than the observed table. Pure
    Python (math.comb) — no scipy dependency, fine at the small counts
    (dozens of deliveries per arm) this module operates on."""
    from math import comb

    row1, row2 = a + b, c + d
    col1, total = a + c, a + b + c + d

    def hyp_prob(x: int) -> float:
        y, z, w = row1 - x, col1 - x, row2 - (col1 - x)
        if x < 0 or y < 0 or z < 0 or w < 0:
            return 0.0
        return (comb(row1, x) * comb(row2, z)) / comb(total, col1)

    observed = hyp_prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(
        px for px in (hyp_prob(x) for x in range(lo, hi + 1))
        if px <= observed * (1 + 1e-9)
    ))


_NO_CHECK = {
    "meta_n_a": None, "meta_n_b": None, "meta_pct_a": None, "meta_pct_b": None,
    "meta_p_value": None, "compliance_blocked": False,
}


def _compliance_gate(db_path: str, base_version: str,
                      candidate_version: str) -> dict:
    """Compliance stats for both arms plus a two-sided Fisher exact p-value
    comparing their flagged rates. `blocked=True` means the candidate arm is
    significantly *worse* on compliance (higher flagged rate, p < 0.05) even if
    it won on engaged_pct — a hypothesis-mismatch signal engaged_pct alone
    cannot surface.

    Opt-in: returns an all-None, unblocked result when the candidate registered
    no compliance check (config.AB_COMPLIANCE_CHECKS / AB_B_QUEUE "compliance"
    key), so an unrelated candidate is never false-vetoed. Both arms are bounded
    to the candidate's earliest memory (its go-live), so arm A's pre-experiment
    history under the same base_version does not contaminate the comparison."""
    check = _compliance_check_for(candidate_version)
    if check is None:
        return dict(_NO_CHECK)

    # Go-live proxy: the first time the candidate arm wrote anything. Both arms
    # accrue in parallel after go-live, so bounding both at this floor clips
    # arm A's long pre-experiment tail without touching the in-window rows.
    conn = _open(db_path)
    try:
        since = conn.execute(
            "SELECT MIN(created_at) FROM memories WHERE source_ref = ?",
            (candidate_version,),
        ).fetchone()[0]
    finally:
        conn.close()

    n_a, pct_a, flagged_a = _meta_stats(db_path, base_version, check, since)
    n_b, pct_b, flagged_b = _meta_stats(db_path, candidate_version, check, since)
    p_value = None
    if n_a and n_b:
        p_value = _fisher_exact_two_sided(flagged_a, n_a - flagged_a,
                                          flagged_b, n_b - flagged_b)
    blocked = bool(p_value is not None and p_value < _COMPLIANCE_SIG_P
                   and (pct_b or 0.0) > (pct_a or 0.0))
    return {
        "meta_n_a": n_a, "meta_n_b": n_b,
        "meta_pct_a": pct_a, "meta_pct_b": pct_b,
        "meta_p_value": p_value, "compliance_blocked": blocked,
    }


# ---------------------------------------------------------------------------
# Experiment lifecycle
# ---------------------------------------------------------------------------

def get_or_create_running_experiment(db_path: Optional[str] = None) -> Optional[dict]:
    """Read the live A/B config and return the current running
    ab_experiments row (creating it on first contact for this
    AB_B_INSTRUCTION text). Returns None if AB_TEST_ENABLED is False."""
    from cairn import config

    if not getattr(config, "AB_TEST_ENABLED", False):
        return None

    instruction_b = getattr(config, "AB_B_INSTRUCTION", None)
    arm_versions = getattr(config, "AB_ARM_VERSIONS", {})
    base_version = arm_versions.get("A")
    candidate_version = arm_versions.get("B")
    if not instruction_b or not base_version or not candidate_version:
        return None

    path = db_path or DB_PATH
    conn = _open(path)
    try:
        row = conn.execute(
            "SELECT id, instruction_b, base_version, candidate_version, status, "
            "n_a, n_b, engaged_pct_a, engaged_pct_b, decision_reason "
            "FROM ab_experiments WHERE status = 'running' AND instruction_b = ? "
            "ORDER BY id DESC LIMIT 1",
            (instruction_b,),
        ).fetchone()
        if row:
            return {
                "id": row[0], "instruction_b": row[1], "base_version": row[2],
                "candidate_version": row[3], "status": row[4],
                "n_a": row[5], "n_b": row[6],
                "engaged_pct_a": row[7], "engaged_pct_b": row[8],
                "decision_reason": row[9],
            }
        cur = conn.execute(
            "INSERT INTO ab_experiments (instruction_b, base_version, candidate_version) "
            "VALUES (?, ?, ?)",
            (instruction_b, base_version, candidate_version),
        )
        conn.commit()
        return {
            "id": cur.lastrowid, "instruction_b": instruction_b,
            "base_version": base_version, "candidate_version": candidate_version,
            "status": "running", "n_a": 0, "n_b": 0,
            "engaged_pct_a": None, "engaged_pct_b": None, "decision_reason": None,
        }
    finally:
        conn.close()


def assess_experiment(row: dict, db_path: Optional[str] = None,
                       eph_path: Optional[str] = None,
                       dry_run: bool = False) -> dict:
    """Compute current arm stats for `row` and apply the promote/reject/
    inconclusive/running decision rule. Always persists the latest stats
    snapshot; only flips status (and, unless dry_run, edits config.py)
    once both arms cross AB_MIN_DELIVERIES_PER_ARM."""
    db_path = db_path or DB_PATH
    eph_path = eph_path or EPH_DB_PATH

    n_a, pct_a = _arm_stats(db_path, eph_path, row["base_version"])
    n_b, pct_b = _arm_stats(db_path, eph_path, row["candidate_version"])

    status = "running"
    reason = None
    if n_a >= AB_MIN_DELIVERIES_PER_ARM and n_b >= AB_MIN_DELIVERIES_PER_ARM:
        gap = (pct_b or 0.0) - (pct_a or 0.0)
        if gap >= AB_PROMOTE_GAP_PCT:
            status = "promoted"
            reason = (f"n_a={n_a} pct_a={pct_a} n_b={n_b} pct_b={pct_b} "
                      f"gap={gap:.1f} >= {AB_PROMOTE_GAP_PCT}")
        elif gap <= AB_REJECT_GAP_PCT:
            status = "rejected"
            reason = (f"n_a={n_a} pct_a={pct_a} n_b={n_b} pct_b={pct_b} "
                      f"gap={gap:.1f} <= {AB_REJECT_GAP_PCT}")
        else:
            status = "inconclusive"
            reason = f"n_a={n_a} pct_a={pct_a} n_b={n_b} pct_b={pct_b} gap={gap:.1f} within band"

    # Compliance gate: engaged_pct alone cannot confirm the candidate arm's
    # instruction actually changed writing behavior — only that the memories
    # it produced were later more/less engaged with, or a promotable win
    # here could just as easily be an unrelated confound. Veto a promotion
    # if the candidate arm is significantly *worse* on the mechanical
    # meta-content check (see _compliance_gate docstring).
    compliance = _compliance_gate(db_path, row["base_version"], row["candidate_version"])
    if status == "promoted" and compliance["compliance_blocked"]:
        # Engagement favoured B, but B is significantly worse on the writing
        # behaviour its own hypothesis targets — treat the win as spurious and
        # REJECT it, which advances the queue to the next candidate rather than
        # looping "inconclusive" on the same gaming candidate every cron pass.
        status = "rejected"
        reason = (f"{reason} | PROMOTION BLOCKED (compliance), rejected + advancing: "
                  f"candidate flagged rate {compliance['meta_pct_b']:.1f}% > base "
                  f"{compliance['meta_pct_a']:.1f}% "
                  f"(p={compliance['meta_p_value']:.4f} < {_COMPLIANCE_SIG_P})")

    conn = _open(db_path)
    try:
        ended_at = "CURRENT_TIMESTAMP" if status in ("promoted", "rejected") else "ended_at"
        conn.execute(
            f"UPDATE ab_experiments SET n_a=?, n_b=?, engaged_pct_a=?, engaged_pct_b=?, "
            f"status=?, decision_reason=?, ended_at={ended_at}, "
            f"meta_n_a=?, meta_n_b=?, meta_pct_a=?, meta_pct_b=?, meta_p_value=?, "
            f"compliance_blocked=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (n_a, n_b, pct_a, pct_b, status, reason,
             compliance["meta_n_a"], compliance["meta_n_b"],
             compliance["meta_pct_a"], compliance["meta_pct_b"],
             compliance["meta_p_value"], int(compliance["compliance_blocked"]),
             row["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    result = {"id": row["id"], "status": status, "n_a": n_a, "n_b": n_b,
              "engaged_pct_a": pct_a, "engaged_pct_b": pct_b, "decision_reason": reason,
              **compliance}

    if dry_run:
        return result

    if status == "promoted":
        candidate = _promote(row, result, db_path=db_path)
        _emit_metric("ab_experiment_promoted", result, eph_path=eph_path)
        if candidate:
            _emit_metric("ab_experiment_advanced",
                         {"from_version": row["candidate_version"],
                          "to_version": candidate["version"],
                          "label": candidate["label"]},
                         eph_path=eph_path)
    elif status == "rejected":
        candidate = _reject(row, result, db_path=db_path)
        _emit_metric("ab_experiment_rejected", result, eph_path=eph_path)
        if compliance["compliance_blocked"]:
            _emit_metric("ab_experiment_promotion_blocked_by_compliance", result, eph_path=eph_path)
        if candidate:
            _emit_metric("ab_experiment_advanced",
                         {"from_version": row["candidate_version"],
                          "to_version": candidate["version"],
                          "label": candidate["label"]},
                         eph_path=eph_path)
    elif status == "inconclusive":
        _emit_metric("ab_experiment_inconclusive", result, eph_path=eph_path)

    return result


# ---------------------------------------------------------------------------
# config.py mechanical edits
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r'^GENERATION_PROMPT_VERSION = "([^"]+)"$', re.MULTILINE)
_AB_ENABLED_RE = re.compile(r'^AB_TEST_ENABLED = True$', re.MULTILINE)
_AB_INSTRUCTION_RE = re.compile(r'AB_B_INSTRUCTION = \(.*?\n\)\n', re.DOTALL)
_AB_ARM_VERSIONS_RE = re.compile(
    r'AB_ARM_VERSIONS = \{"A": GENERATION_PROMPT_VERSION, "B": "[^"]+"\}'
)


def _next_untried_candidate(db_path: Optional[str] = None) -> Optional[dict]:
    """First AB_B_QUEUE entry (cairn/config.py) whose version has never
    appeared as a candidate_version in ab_experiments — or None if the
    queue is empty or every entry has already been tried."""
    from cairn import config

    queue = getattr(config, "AB_B_QUEUE", [])
    if not queue:
        return None
    conn = _open(db_path or DB_PATH)
    try:
        tried = {r[0] for r in conn.execute(
            "SELECT DISTINCT candidate_version FROM ab_experiments"
        ).fetchall()}
    finally:
        conn.close()
    for cand in queue:
        if cand["version"] not in tried:
            return cand
    return None


def _format_instruction_literal(text: str) -> str:
    """Render an AB_B_INSTRUCTION assignment block for `text` using
    repr() for quoting — any candidate string round-trips safely with no
    manual escaping or line-wrapping needed."""
    return f"AB_B_INSTRUCTION = (\n    {text!r}\n)\n"


def _bump_version(version: str) -> str:
    m = re.match(r'^(.*-v)(\d+)$', version)
    if not m:
        raise ValueError(f"cannot bump unrecognised version format: {version!r}")
    prefix, num = m.groups()
    return f"{prefix}{int(num) + 1}"


def _write_config_text(text: str, config_path: str) -> None:
    """Atomic write — temp file + os.replace, same durability pattern as
    the interactive cch-write.py helper."""
    tmp_path = config_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(text)
    os.replace(tmp_path, config_path)


def _advance_or_disable(text: str, db_path: Optional[str], date: str) -> tuple[str, Optional[dict]]:
    """Shared tail for _promote/_reject: splice in the next untried
    AB_B_QUEUE candidate (leaving AB_TEST_ENABLED on) or disable testing
    if the queue is empty/exhausted. Returns (new_text, candidate)."""
    candidate = _next_untried_candidate(db_path)
    if candidate:
        cand_version = candidate["version"]
        cand_instruction = candidate["instruction"]
        text = _AB_ARM_VERSIONS_RE.sub(
            'AB_ARM_VERSIONS = {"A": GENERATION_PROMPT_VERSION, "B": "%s"}' % cand_version,
            text, count=1,
        )
        text = _AB_INSTRUCTION_RE.sub(
            _format_instruction_literal(cand_instruction), text, count=1,
        )
        text += (
            f"# auto-advanced {date}: queued candidate {candidate['version']} "
            f"({candidate['label']}) now testing.\n"
        )
    else:
        text = _AB_ENABLED_RE.sub("AB_TEST_ENABLED = False", text, count=1)
        text += f"# queue exhausted {date}: no untried AB_B_QUEUE candidate remains.\n"
    return text, candidate


def _promote(row: dict, result: dict, config_path: Optional[str] = None,
             db_path: Optional[str] = None) -> Optional[dict]:
    """Bump GENERATION_PROMPT_VERSION to the winning candidate's version
    number, prepend a history comment, and either advance to the next
    AB_B_QUEUE hypothesis or disable testing if none remain. Returns the
    candidate advanced to, or None if testing was disabled instead."""
    config_path = config_path or CONFIG_PATH
    with open(config_path) as f:
        text = f.read()

    m = _VERSION_RE.search(text)
    if not m:
        raise RuntimeError("GENERATION_PROMPT_VERSION assignment not found in config.py")
    old_version = m.group(1)
    new_version = _bump_version(old_version)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    comment = (
        f"#   {old_version} -> {new_version}: promoted {row['candidate_version']} "
        f"({row['instruction_b'][:80]!r}...) after live A/B {date}: "
        f"engaged_pct_b={result['engaged_pct_b']} vs engaged_pct_a={result['engaged_pct_a']} "
        f"(n_a={result['n_a']}, n_b={result['n_b']}).\n"
    )
    text = _VERSION_RE.sub(
        comment + f'GENERATION_PROMPT_VERSION = "{new_version}"', text, count=1
    )
    text, candidate = _advance_or_disable(text, db_path, date)
    _write_config_text(text, config_path)
    return candidate


def _reject(row: dict, result: dict, config_path: Optional[str] = None,
            db_path: Optional[str] = None) -> Optional[dict]:
    """Record the rejection as a history comment, leaving
    GENERATION_PROMPT_VERSION untouched, and either advance to the next
    AB_B_QUEUE hypothesis or disable testing if none remain. Returns the
    candidate advanced to, or None if testing was disabled instead."""
    config_path = config_path or CONFIG_PATH
    with open(config_path) as f:
        text = f.read()

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    comment = (
        f"#   {row['candidate_version']} REJECTED {date}: "
        f"({row['instruction_b'][:80]!r}...) engaged_pct_b={result['engaged_pct_b']} "
        f"vs engaged_pct_a={result['engaged_pct_a']} "
        f"(n_a={result['n_a']}, n_b={result['n_b']}).\n"
    )
    m = _VERSION_RE.search(text)
    if m:
        text = _VERSION_RE.sub(comment + m.group(0), text, count=1)
    text, candidate = _advance_or_disable(text, db_path, date)
    _write_config_text(text, config_path)
    return candidate


# ---------------------------------------------------------------------------
# Orchestrator + CLI entry
# ---------------------------------------------------------------------------

def run_all(db_path: Optional[str] = None, eph_path: Optional[str] = None,
            dry_run: bool = False) -> dict:
    """One cron pass: assess the current live experiment (if any) and
    apply its verdict. No-ops cleanly when AB_TEST_ENABLED is False."""
    row = get_or_create_running_experiment(db_path=db_path)
    if row is None:
        return {"status": "no-op", "reason": "AB_TEST_ENABLED is False or config incomplete"}
    return assess_experiment(row, db_path=db_path, eph_path=eph_path, dry_run=dry_run)


def show_queue(db_path: Optional[str] = None) -> list[dict]:
    """List AB_B_QUEUE hypotheses (cairn/config.py) with each entry's
    tried status — the latest ab_experiments status for its version, or
    "untried" if it has never run."""
    from cairn import config

    queue = getattr(config, "AB_B_QUEUE", [])
    conn = _open(db_path or DB_PATH)
    try:
        rows = conn.execute(
            "SELECT candidate_version, status FROM ab_experiments ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    status_by_version = {v: s for v, s in rows}
    return [
        {
            "version": cand["version"],
            "label": cand["label"],
            "status": status_by_version.get(cand["version"], "untried"),
        }
        for cand in queue
    ]


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="cairn-ab-selfmod",
        description="Automated write-side A/B assessment + changeover")
    sub = p.add_subparsers(dest="command", required=True)
    assess_p = sub.add_parser("assess", help="assess the current live A/B experiment")
    assess_p.add_argument("--dry-run", action="store_true",
                          help="compute the verdict without editing config.py")
    sub.add_parser("show-queue", help="list AB_B_QUEUE hypotheses and their tried/untried status")
    args = p.parse_args(argv)

    if args.command == "assess":
        report = run_all(dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "show-queue":
        print(json.dumps(show_queue(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
