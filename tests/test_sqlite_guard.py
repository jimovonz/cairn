"""Enforce a SINGLE sqlite library across all cairn DB writers.

Mixed stdlib-sqlite3 (3.45) vs pysqlite3 (3.51) writers on the same WAL-mode
cairn DB are a documented cause of corruption (commit be91366 standardised 32
files on the pysqlite3 guard for exactly this reason; ingest.py /
backfill_ingestion.py / confluence_ingest.py each slipped through at various
points). This test prevents recurrence: every module under cairn/ and hooks/ that
imports sqlite3 must go through the pysqlite3 guard (`import pysqlite3 as sqlite3`,
falling back to stdlib only under explicit CAIRN_ALLOW_STDLIB_SQLITE opt-in).
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# graph.py operates ONLY on .code-review-graph/graph.db — a separate file, not the
# cairn memory or ephemeral DBs — so it is exempt. Keep this list minimal and
# justified; do NOT add cairn-DB writers here.
ALLOWLIST = {"cairn/graph.py"}

_IMPORT_SQLITE = re.compile(r"^\s*import sqlite3", re.MULTILINE)


def _python_modules():
    for base in ("cairn", "hooks"):
        for p in (ROOT / base).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def test_all_sqlite_writers_use_pysqlite3_guard():
    offenders = []
    for p in _python_modules():
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOWLIST:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if _IMPORT_SQLITE.search(text) and "import pysqlite3" not in text:
            offenders.append(rel)
    assert not offenders, (
        "Module(s) import stdlib sqlite3 without the pysqlite3 guard — a mixed-lib "
        "writer on a WAL cairn DB risks corruption (be91366). Copy the guard from "
        "the top of cairn/ingest.py, or (only if it touches a non-cairn DB) add it "
        "to ALLOWLIST with justification:\n  " + "\n  ".join(sorted(offenders))
    )
