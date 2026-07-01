"""Memory-side utility-prior suppression (2026-07-02 relevance review).

The memories analogue of calibration_selfmod.auto_archive_low_follow: a memory
that has been PUSH-injected many times and never used — never behaviourally
engaged (relevance.apply_engagement) and never positively graded (agent rg) —
is dead weight in the context window. Flag it push_suppressed=1 so the push
layers (build_context_xml -> _push_suppression_filter) skip it. The memory is
NOT archived: it stays fully pull-searchable via query.py, keeps its embedding,
and can corroborate/contradict as before. Suppression is reversible.

Thresholds are conservative and require the delivery to have actually been
*scoreable* (engaged IS NOT NULL on >= MIN_SCORED rows) — a memory whose
deliveries were all undecidable (no distinctive terms vs the prompt) is never
suppressed on silence alone.

Corrections get a stricter deliveries bar: behavioural corrections can be
followed without term overlap (the agent changes behaviour rather than citing
the text), so the engagement signal under-detects them.

Reactivation: any suppressed memory that has since received engaged=1 or a
grade >= 2 on some delivery is un-suppressed (belt and braces — normally a
suppressed memory is no longer delivered, but pull-path citations and late
grades can still land).

CLI:
    python -m cairn.memory_selfmod            # run suppression + reactivation
    python -m cairn.memory_selfmod --dry-run  # report without writing
    python -m cairn.memory_selfmod --list     # list currently suppressed
    python -m cairn.memory_selfmod --unsuppress ID   # manual reversal
"""

from __future__ import annotations

import os
import sys

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

# Conservative thresholds. MIN_SCORED guards against suppressing on pure silence
# (deliveries that were never engagement-scoreable carry no signal).
SUPPRESS_MIN_DELIVERIES = 10
SUPPRESS_MIN_DELIVERIES_CORRECTION = 25
SUPPRESS_MIN_SCORED = 3
# Types never auto-suppressed: biographical/cross-cutting entries are low-volume
# and engage via behaviour, not term overlap (same reasoning as SCOPE_BIAS_EXEMPT_TYPES).
SUPPRESS_EXEMPT_TYPES = ("person", "preference")


def _open(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _delivery_rollup(eph: sqlite3.Connection) -> dict[int, tuple[int, int, int, int]]:
    """memory_id -> (delivered, scored, engaged, max_grade)."""
    out: dict[int, tuple[int, int, int, int]] = {}
    for mid, n, scored, eng, gmax in eph.execute(
        "SELECT memory_id, COUNT(*), "
        "SUM(CASE WHEN engaged IS NOT NULL THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN engaged = 1 THEN 1 ELSE 0 END), "
        "MAX(COALESCE(grade, 0)) "
        "FROM memory_deliveries GROUP BY memory_id"
    ):
        out[int(mid)] = (n, scored or 0, eng or 0, gmax or 0)
    return out


def suppress_dead_weight(dry_run: bool = False) -> list[int]:
    """Flag never-used, heavily-delivered memories push_suppressed=1."""
    eph = _open(EPH_DB_PATH)
    try:
        rollup = _delivery_rollup(eph)
    finally:
        eph.close()
    if not rollup:
        return []

    dur = _open(DB_PATH)
    try:
        candidates = []
        ids = sorted(rollup)
        for chunk_start in range(0, len(ids), 500):
            chunk = ids[chunk_start:chunk_start + 500]
            qmarks = ",".join("?" * len(chunk))
            for mid, mtype in dur.execute(
                f"SELECT id, type FROM memories WHERE id IN ({qmarks}) "
                f"AND COALESCE(push_suppressed, 0) = 0 AND deleted_at IS NULL",
                chunk,
            ):
                n, scored, eng, gmax = rollup[int(mid)]
                if mtype in SUPPRESS_EXEMPT_TYPES:
                    continue
                min_n = (SUPPRESS_MIN_DELIVERIES_CORRECTION
                         if mtype == "correction" else SUPPRESS_MIN_DELIVERIES)
                if n >= min_n and scored >= SUPPRESS_MIN_SCORED and eng == 0 and gmax <= 0:
                    candidates.append(int(mid))
        if not dry_run and candidates:
            for mid in candidates:
                dur.execute(
                    "UPDATE memories SET push_suppressed = 1 WHERE id = ?", (mid,))
            dur.commit()
        return candidates
    finally:
        dur.close()


def reactivate_engaged(dry_run: bool = False) -> list[int]:
    """Un-suppress memories that have since shown use (engaged or grade >= 2)."""
    eph = _open(EPH_DB_PATH)
    try:
        used = {int(r[0]) for r in eph.execute(
            "SELECT DISTINCT memory_id FROM memory_deliveries "
            "WHERE engaged = 1 OR COALESCE(grade, 0) >= 2")}
    finally:
        eph.close()
    if not used:
        return []
    dur = _open(DB_PATH)
    try:
        ids = sorted(used)
        hits: list[int] = []
        for chunk_start in range(0, len(ids), 500):
            chunk = ids[chunk_start:chunk_start + 500]
            qmarks = ",".join("?" * len(chunk))
            hits.extend(int(r[0]) for r in dur.execute(
                f"SELECT id FROM memories WHERE id IN ({qmarks}) "
                f"AND push_suppressed = 1", chunk))
        if not dry_run and hits:
            for mid in hits:
                dur.execute(
                    "UPDATE memories SET push_suppressed = 0 WHERE id = ?", (mid,))
            dur.commit()
        return hits
    finally:
        dur.close()


def list_suppressed() -> list[tuple[int, str, str]]:
    dur = _open(DB_PATH)
    try:
        return [(int(r[0]), r[1], (r[2] or "")[:90]) for r in dur.execute(
            "SELECT id, type, content FROM memories WHERE push_suppressed = 1 "
            "ORDER BY id")]
    finally:
        dur.close()


def unsuppress(memory_id: int) -> bool:
    dur = _open(DB_PATH)
    try:
        cur = dur.execute(
            "UPDATE memories SET push_suppressed = 0 WHERE id = ?", (memory_id,))
        dur.commit()
        return cur.rowcount > 0
    finally:
        dur.close()


def main(argv: list[str]) -> int:
    if "--unsuppress" in argv:
        mid = int(argv[argv.index("--unsuppress") + 1])
        ok = unsuppress(mid)
        print(f"unsuppress {mid}: {'ok' if ok else 'not found / not suppressed'}")
        return 0 if ok else 1
    if "--list" in argv:
        rows = list_suppressed()
        for mid, mtype, content in rows:
            print(f"#{mid} [{mtype}] {content}")
        print(f"{len(rows)} suppressed")
        return 0
    dry = "--dry-run" in argv
    suppressed = suppress_dead_weight(dry_run=dry)
    reactivated = reactivate_engaged(dry_run=dry)
    prefix = "[dry-run] " if dry else ""
    print(f"{prefix}suppressed {len(suppressed)}: {suppressed}")
    print(f"{prefix}reactivated {len(reactivated)}: {reactivated}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
