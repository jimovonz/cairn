#!/usr/bin/env python3
"""`cairn-calibration` CLI — Phase 4 implementations.

Agent-invoked from natural-language intent (see docs/spec-calibration-system.md §4
and CLAUDE.md "Calibration system" section). Each subcommand maps to a
specific user instruction the agent should detect and translate.

State conventions:
- Permanent mute: `calibration_rows.archived_at` set, archive_reason="muted"
- Session-only mute: ephemeral `hook_state` key=`calibration_mute_<row_id>`
- Disable calibration injection: hook_state key=`calibration_disabled`
  (session-scoped) or session_id=`__global__` (global)
- Interaction mode: hook_state key=`calibration_mode`, value="novice"|"expert"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3


DB_PATH = os.environ.get(
    "CAIRN_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db"),
)
EPH_DB_PATH = os.environ.get(
    "CAIRN_EPHEMERAL_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db"),
)

GLOBAL_STATE_SESSION = "__global__"

# Source-tier initial confidences (spec §"Source tiers")
SOURCE_INITIAL_CONFIDENCE = {
    "explicit": 0.90,
    "correction": 0.60,
    "observation": 0.15,
    "meta-assessment": 0.20,
}
PINNED_CONFIDENCE = 0.95

VALID_SOURCES = tuple(SOURCE_INITIAL_CONFIDENCE.keys())
VALID_LEVELS = ("novice", "expert")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _row_exists(row_id: int, db_path: Optional[str] = None) -> bool:
    path = db_path or DB_PATH
    if not os.path.exists(path):
        return False
    conn = _open(path)
    try:
        return conn.execute(
            "SELECT 1 FROM calibration_rows WHERE id = ?", (row_id,)
        ).fetchone() is not None
    finally:
        conn.close()


def _set_state(session_id: str, key: str, value: str,
               eph_path: Optional[str] = None) -> None:
    conn = _open(eph_path or EPH_DB_PATH)
    try:
        conn.execute(
            "INSERT INTO hook_state (session_id, key, value, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(session_id, key) DO UPDATE SET "
            "value = excluded.value, updated_at = CURRENT_TIMESTAMP",
            (session_id, key, value),
        )
        conn.commit()
    finally:
        conn.close()


def _clear_state(session_id: str, key: str,
                 eph_path: Optional[str] = None) -> int:
    conn = _open(eph_path or EPH_DB_PATH)
    try:
        cur = conn.execute(
            "DELETE FROM hook_state WHERE session_id = ? AND key = ?",
            (session_id, key),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def _current_session() -> Optional[str]:
    """Best-effort current session ID. Used when CLI commands take effect
    in the session that invoked them."""
    return os.environ.get("CLAUDE_SESSION_ID") or os.environ.get("SESSION_ID")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_show_profile(args) -> int:
    """List calibration rows, optionally filtered by subject keyword."""
    conn = _open(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id, source, confidence, pinned, layer, content, kw, "
            "delivered_count, followed_count, ignored_count, corrected_count, "
            "archived_at FROM calibration_rows WHERE archived_at IS NULL "
            "AND superseded_by IS NULL ORDER BY pinned DESC, source, id"
        ).fetchall()
    finally:
        conn.close()
    subject = (args.subject or "").lower().strip()
    if subject:
        rows = [
            r for r in rows
            if subject in (r[5] or "").lower()
            or subject in (r[6] or "").lower()
        ]
    if not rows:
        print(f"No calibration rows{' matching ' + repr(subject) if subject else ''}")
        return 0
    for r in rows:
        rid, source, conf, pinned, layer, content, kw, dc, fc, ic, cc, _ = r
        flag = "★" if pinned else " "
        rate = f"{(fc/dc*100):.0f}%" if dc else "—"
        print(f"{flag} #{rid} [{source} c={conf:.2f} {layer}] "
              f"deliv={dc} follow={rate}")
        print(f"     {content}")
        if kw:
            print(f"     kw: {kw}")
    return 0


def cmd_review(args) -> int:
    """Surface Tier 2 review-queue items: low-follow rows flagged for
    archive (>=10 deliveries, <20% followed), and rows already auto-
    archived by Phase 6 Tier 1 for reference."""
    conn = _open(DB_PATH)
    try:
        low_follow = conn.execute(
            "SELECT id, content, delivered_count, followed_count, "
            "ignored_count, corrected_count "
            "FROM calibration_rows "
            "WHERE archived_at IS NULL AND delivered_count >= 10 "
            "AND (followed_count * 1.0 / delivered_count) < 0.20 "
            "ORDER BY delivered_count DESC"
        ).fetchall()
        recent_archives = conn.execute(
            "SELECT id, content, archive_reason, archived_at "
            "FROM calibration_rows WHERE archived_at IS NOT NULL "
            "ORDER BY archived_at DESC LIMIT 10"
        ).fetchall()
    finally:
        conn.close()
    if not low_follow and not recent_archives:
        print("Review queue empty — nothing to surface")
        return 0
    if low_follow:
        print(f"=== Low-follow candidates ({len(low_follow)}) — archive? ===")
        for rid, content, dc, fc, ic, cc in low_follow:
            print(f"  #{rid} deliv={dc} follow={fc} ignore={ic} correct={cc}")
            print(f"     {content}")
    if recent_archives:
        print(f"\n=== Recent archives ({len(recent_archives)}) — reference ===")
        for rid, content, reason, at in recent_archives:
            print(f"  #{rid} [{reason}] {at}")
            print(f"     {content}")
    return 0


def cmd_history(args) -> int:
    if not _row_exists(args.row_id):
        print(f"error: no calibration_row with id={args.row_id}",
              file=sys.stderr)
        return 1
    conn = _open(DB_PATH)
    try:
        row = conn.execute(
            "SELECT id, content, source, confidence, pinned, layer, "
            "delivered_count, followed_count, ignored_count, corrected_count, "
            "archived_at, archive_reason, superseded_by, created_at, "
            "updated_at FROM calibration_rows WHERE id = ?",
            (args.row_id,),
        ).fetchone()
    finally:
        conn.close()
    print(f"#{row[0]} [{row[2]} c={row[3]:.2f} pinned={row[4]} layer={row[5]}]")
    print(f"  created_at: {row[13]}    updated_at: {row[14]}")
    print(f"  content: {row[1]}")
    print(f"  effectiveness: deliv={row[6]} follow={row[7]} "
          f"ignore={row[8]} correct={row[9]}")
    if row[10]:
        print(f"  ARCHIVED at {row[10]}: {row[11]}")
    if row[12]:
        print(f"  superseded by #{row[12]}")
    return 0


def cmd_add(args) -> int:
    source = args.source
    if source not in VALID_SOURCES:
        print(f"error: --source must be one of {VALID_SOURCES}",
              file=sys.stderr)
        return 1
    content = (args.content or "").strip()
    if not content:
        print("error: --content must be non-empty", file=sys.stderr)
        return 1
    conf = PINNED_CONFIDENCE if args.pin else SOURCE_INITIAL_CONFIDENCE[source]
    layer = "general" if not args.scope else "subject"

    # Embed for downstream retrieval
    emb_blob = None
    try:
        from cairn import embeddings as emb
        vec = emb.embed(content + (" " + args.scope if args.scope else ""))
        if vec is not None:
            emb_blob = emb.to_blob(vec)
    except Exception:
        pass

    conn = _open(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO calibration_rows (content, kw, qf, source, "
            "confidence, pinned, layer, embedding) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?)",
            (content, args.scope or "", json.dumps([]), source, conf,
             1 if args.pin else 0, layer, emb_blob),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    print(f"Added calibration_row #{rid} [{source} c={conf:.2f}]")
    print(f"  {content}")
    return 0


def cmd_mute(args) -> int:
    if not _row_exists(args.row_id):
        print(f"error: no calibration_row with id={args.row_id}",
              file=sys.stderr)
        return 1
    if args.session_only:
        sid = _current_session() or GLOBAL_STATE_SESSION
        if sid == GLOBAL_STATE_SESSION:
            print("warning: no current session — using global scope",
                  file=sys.stderr)
        _set_state(sid, f"calibration_mute_{args.row_id}", "1")
        print(f"Muted row #{args.row_id} for session {sid}")
    else:
        conn = _open(DB_PATH)
        try:
            conn.execute(
                "UPDATE calibration_rows SET archived_at = CURRENT_TIMESTAMP, "
                "archive_reason = 'muted', updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?", (args.row_id,))
            conn.commit()
        finally:
            conn.close()
        print(f"Muted row #{args.row_id} permanently (archived)")
    return 0


def cmd_unmute(args) -> int:
    if not _row_exists(args.row_id):
        print(f"error: no calibration_row with id={args.row_id}",
              file=sys.stderr)
        return 1
    sid = _current_session() or GLOBAL_STATE_SESSION
    cleared = _clear_state(sid, f"calibration_mute_{args.row_id}")
    conn = _open(DB_PATH)
    try:
        cur = conn.execute(
            "UPDATE calibration_rows SET archived_at = NULL, "
            "archive_reason = NULL, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND archive_reason = 'muted'", (args.row_id,))
        unarch = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    print(f"Unmuted row #{args.row_id} (session-cleared={cleared} "
          f"permanent-cleared={unarch})")
    return 0


def cmd_disable(args) -> int:
    if args.session_only:
        sid = _current_session() or GLOBAL_STATE_SESSION
    else:
        sid = GLOBAL_STATE_SESSION
    _set_state(sid, "calibration_disabled", "1")
    print(f"Calibration injection disabled for {sid}")
    return 0


def cmd_enable(args) -> int:
    sid = _current_session() or GLOBAL_STATE_SESSION
    n_sess = _clear_state(sid, "calibration_disabled")
    n_glob = _clear_state(GLOBAL_STATE_SESSION, "calibration_disabled")
    print(f"Calibration injection enabled (session={n_sess} global={n_glob})")
    return 0


def cmd_mode(args) -> int:
    if args.level not in VALID_LEVELS:
        print(f"error: --level must be one of {VALID_LEVELS}",
              file=sys.stderr)
        return 1
    sid = _current_session() or GLOBAL_STATE_SESSION
    if not args.session_only:
        sid = GLOBAL_STATE_SESSION
    _set_state(sid, "calibration_mode", args.level)
    print(f"Calibration mode set to {args.level} for {sid}")
    return 0


def cmd_delete(args) -> int:
    if not _row_exists(args.row_id):
        print(f"error: no calibration_row with id={args.row_id}",
              file=sys.stderr)
        return 1
    conn = _open(DB_PATH)
    try:
        conn.execute("DELETE FROM calibration_rows WHERE id = ?",
                     (args.row_id,))
        conn.commit()
    finally:
        conn.close()
    # Cascade clean calibration_deliveries
    if os.path.exists(EPH_DB_PATH):
        eph = _open(EPH_DB_PATH)
        try:
            eph.execute("DELETE FROM calibration_deliveries WHERE row_id = ?",
                        (args.row_id,))
            eph.commit()
        finally:
            eph.close()
    print(f"Deleted calibration_row #{args.row_id}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cairn-calibration",
        description="Cairn calibration system CLI (Phase 4).",
    )
    p.add_argument("--show-profile", dest="show_profile_subject",
                   nargs="?", const="", default=None,
                   help="Show calibration profile (optionally for a subject)")
    p.add_argument("--review", action="store_true",
                   help="Surface Tier 2 review queue items")
    p.add_argument("--history", type=int, default=None, metavar="ROW_ID",
                   help="Show supersession/archive history for a row")

    sub = p.add_subparsers(dest="cmd")

    sp_add = sub.add_parser("add", help="Add a calibration row")
    sp_add.add_argument("--source", required=True,
                        help=f"one of {VALID_SOURCES}")
    sp_add.add_argument("--content", required=True)
    sp_add.add_argument("--scope", default=None,
                        help="Optional subject keyword(s)")
    sp_add.add_argument("--pin", action="store_true",
                        help="Pin the row at high confidence")

    sp_mute = sub.add_parser("mute", help="Mute a calibration row")
    sp_mute.add_argument("row_id", type=int)
    sp_mute.add_argument("--session-only", action="store_true")

    sp_unmute = sub.add_parser("unmute", help="Unmute a calibration row")
    sp_unmute.add_argument("row_id", type=int)

    sp_disable = sub.add_parser("disable",
                                 help="Disable calibration injection")
    sp_disable.add_argument("--session-only", action="store_true")

    sub.add_parser("enable", help="Re-enable calibration injection")

    sp_mode = sub.add_parser("mode", help="Set interaction level mode")
    sp_mode.add_argument("--level", required=True,
                         help=f"one of {VALID_LEVELS}")
    sp_mode.add_argument("--session-only", action="store_true")

    sp_delete = sub.add_parser("delete", help="Delete a calibration row")
    sp_delete.add_argument("row_id", type=int)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_profile_subject is not None:
        args.subject = args.show_profile_subject or None
        return cmd_show_profile(args)
    if args.review:
        return cmd_review(args)
    if args.history is not None:
        args.row_id = args.history
        return cmd_history(args)

    dispatch = {
        "add": cmd_add, "mute": cmd_mute, "unmute": cmd_unmute,
        "disable": cmd_disable, "enable": cmd_enable,
        "mode": cmd_mode, "delete": cmd_delete,
    }
    if args.cmd in dispatch:
        return dispatch[args.cmd](args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
