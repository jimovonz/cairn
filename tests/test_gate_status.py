"""Gate provenance on delivery rows (spec 2S.3).

A NULL reranker_model conflates "the daemon was down" with "this layer is not
gated by design". Only the first indicates degraded retrieval, and a reranker
comparison that re-admits it manufactures ungated-vs-reranked conclusions.
"""
import pysqlite3 as sqlite3
import pytest

import cairn.init_db as init_db
import cairn.relevance as relevance
from cairn.embeddings import _classify_gate


def test_attempted_but_no_scores_is_unavailable():
    """The only value meaning degraded retrieval."""
    assert _classify_gate(True, True, True) == "gate-unavailable"


def test_cross_encoder_disabled_is_not_unavailable():
    assert _classify_gate(False, True, True) == "disabled"


def test_caller_opted_out_is_by_design():
    assert _classify_gate(True, False, True) == "ungated-by-design"


def test_too_few_candidates_is_by_design():
    assert _classify_gate(True, True, False) == "below-min-candidates"


def test_disabled_wins_over_opt_out():
    """Ordering matters: a disabled encoder explains the absence regardless of
    what the caller asked for."""
    assert _classify_gate(False, False, False) == "disabled"


@pytest.fixture
def eph(tmp_path):
    path = str(tmp_path / "eph.db")
    init_db.init_ephemeral(path)
    return path


def test_gate_status_is_persisted_per_delivery(eph):
    n = relevance.log_memory_deliveries(
        [{"id": 1, "gate_status": "reranked", "reranker_model": "m"},
         {"id": 2, "gate_status": "gate-unavailable"}],
        session_id="s1", eph_path=eph,
    )
    assert n == 2
    conn = sqlite3.connect(eph)
    rows = dict(conn.execute(
        "SELECT memory_id, gate_status FROM memory_deliveries").fetchall())
    conn.close()
    assert rows == {1: "reranked", 2: "gate-unavailable"}


def test_missing_gate_status_stays_null_not_guessed(eph):
    """Rows predating the column must be excluded from comparisons, not assumed
    to be one case or the other."""
    relevance.log_memory_deliveries([{"id": 3}], session_id="s2", eph_path=eph)
    conn = sqlite3.connect(eph)
    row = conn.execute(
        "SELECT gate_status FROM memory_deliveries WHERE memory_id = 3").fetchone()
    conn.close()
    assert row[0] is None
