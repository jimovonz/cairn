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

    conn = _open(db_path)
    try:
        ended_at = "CURRENT_TIMESTAMP" if status in ("promoted", "rejected") else "ended_at"
        conn.execute(
            f"UPDATE ab_experiments SET n_a=?, n_b=?, engaged_pct_a=?, engaged_pct_b=?, "
            f"status=?, decision_reason=?, ended_at={ended_at}, "
            f"updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (n_a, n_b, pct_a, pct_b, status, reason, row["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    result = {"id": row["id"], "status": status, "n_a": n_a, "n_b": n_b,
              "engaged_pct_a": pct_a, "engaged_pct_b": pct_b, "decision_reason": reason}

    if dry_run:
        return result

    if status == "promoted":
        _promote(row, result)
        _emit_metric("ab_experiment_promoted", result, eph_path=eph_path)
    elif status == "rejected":
        _reject(row, result)
        _emit_metric("ab_experiment_rejected", result, eph_path=eph_path)
    elif status == "inconclusive":
        _emit_metric("ab_experiment_inconclusive", result, eph_path=eph_path)

    return result


# ---------------------------------------------------------------------------
# config.py mechanical edits
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r'^GENERATION_PROMPT_VERSION = "([^"]+)"$', re.MULTILINE)
_AB_ENABLED_RE = re.compile(r'^AB_TEST_ENABLED = True$', re.MULTILINE)


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


def _promote(row: dict, result: dict, config_path: Optional[str] = None) -> None:
    """Bump GENERATION_PROMPT_VERSION to the winning candidate's version
    number, prepend a history comment, and disable testing (no next
    hypothesis has been authored yet)."""
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
    text = _AB_ENABLED_RE.sub("AB_TEST_ENABLED = False", text, count=1)
    _write_config_text(text, config_path)


def _reject(row: dict, result: dict, config_path: Optional[str] = None) -> None:
    """Disable testing for a rejected hypothesis, leaving
    GENERATION_PROMPT_VERSION untouched, and record the rejection as a
    history comment so failed experiments are visible in config.py too."""
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
    text = _AB_ENABLED_RE.sub("AB_TEST_ENABLED = False", text, count=1)
    _write_config_text(text, config_path)


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


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="cairn-ab-selfmod",
        description="Automated write-side A/B assessment + changeover")
    sub = p.add_subparsers(dest="command", required=True)
    assess_p = sub.add_parser("assess", help="assess the current live A/B experiment")
    assess_p.add_argument("--dry-run", action="store_true",
                          help="compute the verdict without editing config.py")
    args = p.parse_args(argv)

    if args.command == "assess":
        report = run_all(dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
