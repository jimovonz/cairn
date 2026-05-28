"""One-off backfill: embed each qf string of every active calibration row
into the calibration_qf_embeddings sidecar.

Idempotent — INSERT OR REPLACE on (row_id, qf_index) PK. Safe to rerun.
Local embedder (no $ cost, no LLM call). For 462 rows × ~4 qf strings
each, expect ~1-2k local embed calls.

Usage:
    python3 cairn/calibration_qf_backfill.py [--db PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3


DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db")


def backfill(db_path: Optional[str] = None, dry_run: bool = False) -> dict:
    path = db_path or DEFAULT_DB
    from cairn.embeddings import embed, to_blob

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT id, qf FROM calibration_rows "
            "WHERE archived_at IS NULL AND superseded_by IS NULL"
        ).fetchall()
        already = {
            rid for (rid,) in conn.execute(
                "SELECT DISTINCT row_id FROM calibration_qf_embeddings"
            ).fetchall()
        }

        rows_processed = 0
        rows_skipped_no_qf = 0
        rows_skipped_already = 0
        qf_strings_embedded = 0

        for row_id, qf_json in rows:
            qf_list: list[str] = []
            try:
                parsed = json.loads(qf_json) if qf_json else []
                qf_list = [s for s in parsed if isinstance(s, str) and s.strip()]
            except (json.JSONDecodeError, TypeError):
                pass

            if not qf_list:
                rows_skipped_no_qf += 1
                continue
            if row_id in already:
                rows_skipped_already += 1
                continue

            for qf_index, qf_text in enumerate(qf_list):
                vec = embed(qf_text)
                if vec is None:
                    continue
                blob = to_blob(vec)
                if not dry_run:
                    conn.execute(
                        "INSERT OR REPLACE INTO calibration_qf_embeddings "
                        "(row_id, qf_index, qf_text, embedding) "
                        "VALUES (?, ?, ?, ?)",
                        (row_id, qf_index, qf_text, blob),
                    )
                qf_strings_embedded += 1
            rows_processed += 1

        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    return {
        "rows_processed": rows_processed,
        "rows_skipped_no_qf": rows_skipped_no_qf,
        "rows_skipped_already_backfilled": rows_skipped_already,
        "qf_strings_embedded": qf_strings_embedded,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="Path to cairn.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute embeddings but do not write")
    args = parser.parse_args()

    result = backfill(db_path=args.db, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
