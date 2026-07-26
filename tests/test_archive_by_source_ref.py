"""Bulk retraction verb — archive every memory written by a given source_ref.

Spec: docs/spec-remediation-2026-07.md item 1.8. Provenance stamping is inert
without a retraction verb: source_ref makes a write-path experiment
*attributable*, this makes it *retractable*.
"""
import pysqlite3 as sqlite3
import pytest

import cairn.init_db as init_db
import cairn.query as query


@pytest.fixture
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "archive_test.db")
    old = init_db.DB_PATH
    init_db.DB_PATH = db_path
    try:
        init_db.init()
    finally:
        init_db.DB_PATH = old
    monkeypatch.setattr(query, "DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    rows = [
        ("fact", "keep-a", "unstamped row", None),
        ("fact", "drop-a", "arm B row one", "genB-v2"),
        ("fact", "drop-b", "arm B row two", "genB-v2"),
        ("fact", "keep-b", "arm A row", "genA-v4"),
        ("decision", "keep-c", "writeback row", "review-writeback"),
        ("fact", "drop-c", "ingest row", "ingest-extractors-v3"),
    ]
    for t, topic, content, sref in rows:
        conn.execute(
            "INSERT INTO memories (type, topic, content, source_ref) VALUES (?, ?, ?, ?)",
            (t, topic, content, sref),
        )
    conn.commit()
    conn.close()
    return db_path


def _archived(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT topic FROM memories WHERE archived_reason IS NOT NULL "
        "AND archived_reason != '' ORDER BY topic"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def test_exact_match_archives_only_that_source_ref(db):
    n = query.archive_by_source_ref("genB-v2", "arm B regressed")
    assert n == 2
    assert _archived(db) == ["drop-a", "drop-b"]


def test_archived_rows_get_zero_confidence_and_reason(db):
    query.archive_by_source_ref("genB-v2", "arm B regressed")
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT confidence, archived_reason FROM memories WHERE topic = 'drop-a'"
    ).fetchall()
    conn.close()
    assert rows[0][0] == 0
    assert rows[0][1] == "arm B regressed"


def test_dry_run_reports_count_but_changes_nothing(db):
    n = query.archive_by_source_ref("genB-v2", "arm B regressed", dry_run=True)
    assert n == 2
    assert _archived(db) == []


def test_idempotent_second_run_archives_nothing(db):
    assert query.archive_by_source_ref("genB-v2", "arm B regressed") == 2
    assert query.archive_by_source_ref("genB-v2", "arm B regressed") == 0
    assert _archived(db) == ["drop-a", "drop-b"]


def test_no_match_is_a_no_op(db):
    assert query.archive_by_source_ref("does-not-exist", "why") == 0
    assert _archived(db) == []


def test_like_mode_matches_a_family_of_stamps(db):
    n = query.archive_by_source_ref("ingest-%", "bad extractor version", like=True)
    assert n == 1
    assert _archived(db) == ["drop-c"]


def test_bare_wildcard_is_refused(db):
    """A '%' pattern would archive the entire corpus — the one irreversible
    mistake this verb could make. Refuse it rather than rely on --dry-run."""
    n = query.archive_by_source_ref("%", "oops", like=True)
    assert n == 0
    assert _archived(db) == []


def test_null_source_ref_rows_are_never_matched(db):
    query.archive_by_source_ref("genB-v2", "arm B regressed")
    query.archive_by_source_ref("ingest-%", "bad extractor", like=True)
    assert "keep-a" not in _archived(db)


def test_writes_an_audit_row_per_archived_memory(db):
    query.archive_by_source_ref("genB-v2", "arm B regressed")
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT direction, reason FROM memory_annotation_log ORDER BY memory_id"
    ).fetchall()
    conn.close()
    assert len(rows) == 2
    assert all(r[0] == "archive" for r in rows)
    assert all(r[1] == "arm B regressed" for r in rows)


def test_dry_run_writes_no_audit_rows(db):
    query.archive_by_source_ref("genB-v2", "arm B regressed", dry_run=True)
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM memory_annotation_log").fetchone()[0]
    conn.close()
    assert n == 0
