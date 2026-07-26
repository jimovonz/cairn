"""Fail-soft instrumentation must not convert a migration gap into data loss.

log_memory_deliveries swallows errors so instrumentation never breaks delivery.
Combined with a new column in the INSERT that is the worst of both worlds: every
insert returns 0, nothing is recorded, and nothing complains. That happened live
on 2026-07-26 when gate_status shipped — five stops, zero deliveries logged, and
it was only caught by asking whether the data was actually being collected.
"""
import pysqlite3 as sqlite3
import pytest

import cairn.init_db as init_db
import cairn.relevance as relevance


@pytest.fixture
def stale_eph(tmp_path):
    """An ephemeral DB predating the gate_status column."""
    path = str(tmp_path / "stale.db")
    init_db.init_ephemeral(path)
    conn = sqlite3.connect(path)
    cols = [r[1] for r in conn.execute("pragma table_info(memory_deliveries)")]
    assert "gate_status" in cols, "fixture assumes the current schema"
    conn.execute("ALTER TABLE memory_deliveries RENAME TO md_old")
    conn.execute("""CREATE TABLE memory_deliveries (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, turn_index INTEGER,
        memory_id INTEGER, context_text TEXT, context_vec BLOB, ce_score REAL,
        served_rank INTEGER, reranker_model TEXT, score_components TEXT,
        layer TEXT, scope TEXT, engaged INTEGER, engaged_score REAL,
        engaged_method TEXT, delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()
    return path


def test_missing_column_is_migrated_and_the_write_succeeds(stale_eph):
    n = relevance.log_memory_deliveries(
        [{"id": 1, "gate_status": "reranked"}], session_id="s1", eph_path=stale_eph)
    assert n == 1, "a migration gap must not silently discard deliveries"

    conn = sqlite3.connect(stale_eph)
    cols = [r[1] for r in conn.execute("pragma table_info(memory_deliveries)")]
    row = conn.execute(
        "SELECT memory_id, gate_status FROM memory_deliveries").fetchone()
    conn.close()
    assert "gate_status" in cols
    assert row == (1, "reranked")


def test_healthy_db_still_writes_normally(tmp_path):
    path = str(tmp_path / "ok.db")
    init_db.init_ephemeral(path)
    assert relevance.log_memory_deliveries(
        [{"id": 7}], session_id="s2", eph_path=path) == 1


def test_retry_does_not_recurse_forever(stale_eph, monkeypatch):
    """If migration fails to fix the schema, the retry must give up rather than
    loop — the healing path must never become its own outage."""
    monkeypatch.setattr("cairn.init_db.init_ephemeral", lambda *a, **k: None)
    assert relevance.log_memory_deliveries(
        [{"id": 2}], session_id="s3", eph_path=stale_eph) == 0
