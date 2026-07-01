"""One-off backfill: embed question-form keywords of every active memory into
the memory_qf_embeddings sidecar (schema v14 — calibration v7 port).

Historic arm-B memories (genB-v1) and any older entries whose keywords happen
to contain question phrasings get sidecar rows, so per-qf retrieval works on
the existing corpus, not just genA-v4 writes going forward.

Idempotent — store_qf_embeddings DELETEs then INSERTs per memory. Local
embedder via the daemon fast path when available (no LLM cost).

Usage:
    python3 cairn/memory_qf_backfill.py [--db PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
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


def backfill(db_path: Optional[str] = None, dry_run: bool = False) -> dict:
    path = db_path or DEFAULT_DB
    from cairn import embeddings
    from cairn.relevance import extract_question_forms, store_qf_embeddings

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT id, keywords FROM memories "
            "WHERE keywords IS NOT NULL AND keywords != '' "
            "AND deleted_at IS NULL "
            "AND (archived_reason IS NULL OR archived_reason = '')"
        ).fetchall()
        already = {rid for (rid,) in conn.execute(
            "SELECT DISTINCT memory_id FROM memory_qf_embeddings").fetchall()}

        processed = skipped_no_qf = skipped_already = embedded = 0
        for mem_id, keywords_csv in rows:
            if mem_id in already:
                skipped_already += 1
                continue
            kws = [k.strip() for k in (keywords_csv or "").split(",") if k.strip()]
            qfs = extract_question_forms(kws)
            if not qfs:
                skipped_no_qf += 1
                continue
            if dry_run:
                processed += 1
                embedded += len(qfs[:8])
                continue
            n = store_qf_embeddings(conn, mem_id, kws, embeddings)
            if n:
                processed += 1
                embedded += n
        if not dry_run:
            conn.commit()
        return {
            "rows_processed": processed,
            "rows_skipped_no_qf": skipped_no_qf,
            "rows_skipped_already": skipped_already,
            "qf_strings_embedded": embedded,
            "dry_run": dry_run,
        }
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stats = backfill(args.db, args.dry_run)
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
