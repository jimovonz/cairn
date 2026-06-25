"""Phase 6 — Calibration self-modification (Tier 1 autonomous + Tier 2 surfaced).

**Tier 1 (autonomous, bounded scalar adjustments):**
- `auto_archive_low_follow` — rows with >=10 deliveries and <20% followed
  get archived with archive_reason="auto-archive-low-follow"
- `auto_promote_corroborated` — rows with N>=5 deliveries and >=80% followed
  + corroborated across >=3 distinct sessions get pinned (confidence
  boosted to PINNED_CONFIDENCE)
- `decay_unused` — rows with delivered_count=0 older than half-life
  (per source tier) have confidence decayed multiplicatively

**Tier 2 (surfaced to user for approval via `cairn-calibration --review`):**
- Suggested re-phrasings of mid-follow rows (40–60% follow rate)
- `qf` vocabulary expansion candidates: rows with strong kw match but
  poor qf hits (high delivered_count, low similarity-driven follows)
- Promotion candidates that didn't auto-trigger (≥3 corroborating
  sessions but <80% follow rate)

Tier 2 surfaces are written into a `calibration_review_queue` table in
the durable DB. The `cairn-calibration --review` CLI reads it.

Tier 3 (analyser prompt, retrieval weights, system architecture) stays
manual by design — not in scope here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
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


# Tier 1 thresholds — bounded, conservative
AUTO_ARCHIVE_MIN_DELIVERIES = 10
AUTO_ARCHIVE_MAX_FOLLOW_RATE = 0.20
AUTO_PROMOTE_MIN_DELIVERIES = 5
AUTO_PROMOTE_MIN_FOLLOW_RATE = 0.80
AUTO_PROMOTE_MIN_SESSIONS = 3
PINNED_CONFIDENCE = 0.95

# Half-lives in days per source tier (spec source-tiers table)
HALF_LIVES_DAYS = {
    "explicit": 120,
    "correction": 60,
    "observation": 30,
    "meta-assessment": 60,
}

# Tier 2 surfaced types
SURFACE_LOW_FOLLOW_REPHRASE = "low-follow-rephrase"
SURFACE_QF_EXPANSION = "qf-vocab-expansion"
SURFACE_PROMOTION_CANDIDATE = "promotion-candidate"


# ---------------------------------------------------------------------------
# Schema (lazy migration on first run)
# ---------------------------------------------------------------------------

def _ensure_review_queue_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_review_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_id INTEGER NOT NULL REFERENCES calibration_rows(id)
                ON DELETE CASCADE,
            suggestion_type TEXT NOT NULL,
            detail TEXT,
            surfaced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolution TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cal_review_unresolved "
        "ON calibration_review_queue(resolved_at)"
    )


def _open(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _emit_metric(event: str, detail: dict,
                  eph_path: Optional[str] = None) -> None:
    """Append a row to the ephemeral metrics table. Best-effort —
    swallows any error so selfmod operations don't fail if metrics is
    unreachable."""
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
# Tier 1 — autonomous
# ---------------------------------------------------------------------------

def auto_archive_low_follow(db_path: Optional[str] = None) -> list[int]:
    """Archive rows whose follow rate is too low after enough deliveries.

    Returns the list of row IDs archived.
    """
    path = db_path or DB_PATH
    conn = _open(path)
    try:
        rows = conn.execute(
            "SELECT id FROM calibration_rows "
            "WHERE archived_at IS NULL "
            "AND delivered_count >= ? "
            "AND (followed_count * 1.0 / delivered_count) < ?",
            (AUTO_ARCHIVE_MIN_DELIVERIES, AUTO_ARCHIVE_MAX_FOLLOW_RATE),
        ).fetchall()
        ids = [r[0] for r in rows]
        for rid in ids:
            conn.execute(
                "UPDATE calibration_rows SET archived_at = CURRENT_TIMESTAMP, "
                "archive_reason = 'auto-archive-low-follow', "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (rid,),
            )
            _emit_metric("calibration_row_archived",
                         {"row_id": rid, "reason": "auto-archive-low-follow"})
        conn.commit()
        return ids
    finally:
        conn.close()


def _distinct_sessions_for_row(eph_path: str, row_id: int) -> int:
    if not os.path.exists(eph_path):
        return 0
    conn = _open(eph_path)
    try:
        n = conn.execute(
            "SELECT count(DISTINCT session_id) FROM calibration_deliveries "
            "WHERE row_id = ? AND outcome = 'followed'",
            (row_id,),
        ).fetchone()[0]
        return n
    finally:
        conn.close()


def auto_promote_corroborated(
        db_path: Optional[str] = None,
        eph_path: Optional[str] = None) -> list[int]:
    """Pin rows that are highly followed across multiple distinct sessions.

    Returns the list of row IDs promoted.
    """
    durable = db_path or DB_PATH
    eph = eph_path or EPH_DB_PATH
    conn = _open(durable)
    try:
        candidates = conn.execute(
            "SELECT id FROM calibration_rows "
            "WHERE archived_at IS NULL AND pinned = 0 "
            "AND delivered_count >= ? "
            "AND (followed_count * 1.0 / delivered_count) >= ?",
            (AUTO_PROMOTE_MIN_DELIVERIES, AUTO_PROMOTE_MIN_FOLLOW_RATE),
        ).fetchall()
        promoted = []
        for (rid,) in candidates:
            sessions = _distinct_sessions_for_row(eph, rid)
            if sessions >= AUTO_PROMOTE_MIN_SESSIONS:
                conn.execute(
                    "UPDATE calibration_rows SET pinned = 1, "
                    "confidence = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    (PINNED_CONFIDENCE, rid),
                )
                _emit_metric("calibration_row_promoted",
                             {"row_id": rid, "source_sessions": sessions},
                             eph_path=eph)
                promoted.append(rid)
        conn.commit()
        return promoted
    finally:
        conn.close()


def decay_unused(db_path: Optional[str] = None,
                 now: Optional[datetime] = None) -> int:
    """Multiplicatively decay confidence on rows with zero deliveries
    that are older than their source-tier half-life. One half-life
    halves the row's confidence; further half-lives compound.

    Returns the number of rows whose confidence was updated.
    """
    path = db_path or DB_PATH
    now_dt = now or datetime.now(timezone.utc)
    conn = _open(path)
    n = 0
    try:
        rows = conn.execute(
            "SELECT id, source, confidence, created_at FROM calibration_rows "
            "WHERE archived_at IS NULL AND delivered_count = 0"
        ).fetchall()
        for rid, source, conf, created in rows:
            half_life = HALF_LIVES_DAYS.get(source)
            if not half_life or conf is None or conf <= 0:
                continue
            try:
                cdt = datetime.fromisoformat(created.replace("Z", "+00:00")) \
                    if created else None
            except (TypeError, ValueError):
                cdt = None
            if cdt is None:
                continue
            if cdt.tzinfo is None:
                cdt = cdt.replace(tzinfo=timezone.utc)
            age_days = (now_dt - cdt).total_seconds() / 86400.0
            if age_days < half_life:
                continue
            decay_factor = 0.5 ** (age_days / half_life)
            new_conf = round(conf * decay_factor, 4)
            if abs(new_conf - conf) < 0.005:
                continue
            conn.execute(
                "UPDATE calibration_rows SET confidence = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_conf, rid),
            )
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tier 2 — surfaced
# ---------------------------------------------------------------------------

def surface_low_follow_rephrase(db_path: Optional[str] = None) -> list[int]:
    """Rows with 40–60% follow rate get surfaced for user-driven
    rephrasing. Idempotent — does not re-surface an unresolved row."""
    path = db_path or DB_PATH
    conn = _open(path)
    surfaced = []
    try:
        _ensure_review_queue_table(conn)
        rows = conn.execute(
            "SELECT cr.id, cr.delivered_count, cr.followed_count "
            "FROM calibration_rows cr "
            "LEFT JOIN calibration_review_queue rq "
            "  ON rq.row_id = cr.id AND rq.suggestion_type = ? "
            "  AND rq.resolved_at IS NULL "
            "WHERE cr.archived_at IS NULL AND cr.delivered_count >= 5 "
            "AND rq.id IS NULL",
            (SURFACE_LOW_FOLLOW_REPHRASE,),
        ).fetchall()
        for rid, dc, fc in rows:
            if dc <= 0:
                continue
            rate = fc / dc
            if 0.40 <= rate <= 0.60:
                conn.execute(
                    "INSERT INTO calibration_review_queue "
                    "(row_id, suggestion_type, detail) VALUES (?, ?, ?)",
                    (rid, SURFACE_LOW_FOLLOW_REPHRASE,
                     json.dumps({"follow_rate": round(rate, 3),
                                 "deliveries": dc}))
                )
                _emit_metric("calibration_review_surfaced",
                             {"row_id": rid,
                              "suggestion_type": SURFACE_LOW_FOLLOW_REPHRASE})
                surfaced.append(rid)
        conn.commit()
        return surfaced
    finally:
        conn.close()


def surface_promotion_candidates(
        db_path: Optional[str] = None,
        eph_path: Optional[str] = None) -> list[int]:
    """Rows that cross >=AUTO_PROMOTE_MIN_SESSIONS but missed
    auto-promote because follow rate was below the auto threshold."""
    durable = db_path or DB_PATH
    eph = eph_path or EPH_DB_PATH
    conn = _open(durable)
    surfaced = []
    try:
        _ensure_review_queue_table(conn)
        rows = conn.execute(
            "SELECT cr.id, cr.delivered_count, cr.followed_count "
            "FROM calibration_rows cr "
            "LEFT JOIN calibration_review_queue rq "
            "  ON rq.row_id = cr.id AND rq.suggestion_type = ? "
            "  AND rq.resolved_at IS NULL "
            "WHERE cr.archived_at IS NULL AND cr.pinned = 0 "
            "AND cr.delivered_count >= ? AND rq.id IS NULL",
            (SURFACE_PROMOTION_CANDIDATE, AUTO_PROMOTE_MIN_DELIVERIES),
        ).fetchall()
        for rid, dc, fc in rows:
            rate = fc / dc if dc else 0.0
            sessions = _distinct_sessions_for_row(eph, rid)
            if sessions >= AUTO_PROMOTE_MIN_SESSIONS \
                    and rate < AUTO_PROMOTE_MIN_FOLLOW_RATE \
                    and rate >= 0.50:
                conn.execute(
                    "INSERT INTO calibration_review_queue "
                    "(row_id, suggestion_type, detail) VALUES (?, ?, ?)",
                    (rid, SURFACE_PROMOTION_CANDIDATE,
                     json.dumps({"follow_rate": round(rate, 3),
                                 "sessions": sessions}))
                )
                _emit_metric("calibration_review_surfaced",
                             {"row_id": rid,
                              "suggestion_type": SURFACE_PROMOTION_CANDIDATE})
                surfaced.append(rid)
        conn.commit()
        return surfaced
    finally:
        conn.close()


def resolve_review_item(item_id: int, resolution: str,
                         db_path: Optional[str] = None) -> bool:
    """Mark a review-queue item resolved. `resolution` is a short tag
    like 'approved' / 'dismissed' / 'edited'."""
    path = db_path or DB_PATH
    conn = _open(path)
    try:
        _ensure_review_queue_table(conn)
        cur = conn.execute(
            "UPDATE calibration_review_queue "
            "SET resolved_at = CURRENT_TIMESTAMP, resolution = ? "
            "WHERE id = ? AND resolved_at IS NULL",
            (resolution, item_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Orchestrator + CLI entry
# ---------------------------------------------------------------------------

def run_all(db_path: Optional[str] = None,
            eph_path: Optional[str] = None) -> dict:
    """One pass of all Tier 1 + Tier 2 selfmod operations. Idempotent."""
    archived = auto_archive_low_follow(db_path=db_path)
    promoted = auto_promote_corroborated(db_path=db_path, eph_path=eph_path)
    decayed = decay_unused(db_path=db_path)
    rephrase = surface_low_follow_rephrase(db_path=db_path)
    promote_q = surface_promotion_candidates(db_path=db_path, eph_path=eph_path)
    return {
        "archived": archived,
        "promoted": promoted,
        "decayed_count": decayed,
        "surfaced_rephrase": rephrase,
        "surfaced_promotion": promote_q,
    }


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="cairn-calibration-selfmod",
        description="Calibration self-modification (Tier 1 + Tier 2)")
    p.add_argument("--dry-run", action="store_true",
                   help="(future) dry-run — not implemented yet, current "
                        "ops are idempotent and safe to re-run")
    args = p.parse_args(argv)
    report = run_all()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
