"""Backfill memories.topic_embedding (schema v8) for rows lacking it.

Idempotent — UPDATE only rows WHERE topic_embedding IS NULL. Local
embedder (no $ cost, no LLM call). Safe to rerun. Run automatically by
install.sh on each install/upgrade.

Usage:
    python3 cairn/memory_topic_embedding_backfill.py [--db PATH] [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
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


DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db")


def backfill(db_path: Optional[str] = None, dry_run: bool = False,
             limit: Optional[int] = None) -> dict:
    path = db_path or DEFAULT_DB
    from cairn.embeddings import embed, to_blob

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        sql = (
            "SELECT id, topic FROM memories "
            "WHERE topic_embedding IS NULL AND deleted_at IS NULL AND topic != ''"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
        total = len(rows)
        embedded = 0
        skipped = 0
        for mid, topic in rows:
            if not topic or not topic.strip():
                skipped += 1
                continue
            try:
                vec = embed(topic)
            except Exception:
                vec = None
            if vec is None:
                skipped += 1
                continue
            if not dry_run:
                conn.execute(
                    "UPDATE memories SET topic_embedding = ? WHERE id = ?",
                    (to_blob(vec), mid),
                )
            embedded += 1
        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    return {
        "rows_needing_backfill": total,
        "embedded": embedded,
        "skipped": skipped,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="Path to cairn.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute counts but do not write")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of rows processed (default: all)")
    args = parser.parse_args()

    result = backfill(db_path=args.db, dry_run=args.dry_run, limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
