"""Fit labels must be validated against what was actually delivered.

Measured on the rg path, which reports its misses: 875 volunteered grades named
a memory never delivered in that session, against 610 that landed. rg survives
that because a non-matching UPDATE affects zero rows. apply_fit_labels does an
INSERT into the DURABLE db, so it has no such backstop — unvalidated, it would
accumulate fabricated training data and the growth would look like progress.
"""

import os
import sys
import tempfile
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import init_db, relevance


def _dbs():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "eph.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    return durable, eph


def _deliver(eph, session_id, ids):
    c = sqlite3.connect(eph)
    for mid in ids:
        c.execute("INSERT INTO memory_deliveries (session_id, turn_index, memory_id, "
                  "context_text) VALUES (?,?,?,?)", (session_id, 0, mid, "ctx"))
    c.commit(); c.close()


def _pairs(durable):
    c = sqlite3.connect(durable)
    try:
        return c.execute("SELECT winner_id, loser_id FROM delivery_fit_pairs").fetchall()
    finally:
        c.close()


def test_pairs_naming_delivered_ids_are_kept():
    durable, eph = _dbs()
    _deliver(eph, "s1", [10, 20])
    n = relevance.apply_fit_labels([(10, 20)], session_id="s1",
                                   durable_path=durable, eph_path=eph)
    assert n == 1
    assert _pairs(durable) == [(10, 20)]


def test_pair_naming_an_undelivered_id_is_rejected():
    """The failure this prevents: an id the agent was never shown becomes a
    permanent training label in the durable DB."""
    durable, eph = _dbs()
    _deliver(eph, "s1", [10])
    n = relevance.apply_fit_labels([(10, 999)], session_id="s1",
                                   durable_path=durable, eph_path=eph)
    assert n == 0
    assert _pairs(durable) == []


def test_valid_pairs_survive_alongside_rejected_ones():
    durable, eph = _dbs()
    _deliver(eph, "s1", [10, 20])
    n = relevance.apply_fit_labels([(10, 20), (10, 999)], session_id="s1",
                                   durable_path=durable, eph_path=eph)
    assert n == 1
    assert _pairs(durable) == [(10, 20)]


def test_ids_delivered_only_in_another_session_are_rejected():
    """Cross-session id reuse is coincidence, not evidence — the agent did not
    see that memory in THIS turn's context."""
    durable, eph = _dbs()
    _deliver(eph, "s1", [10])
    _deliver(eph, "other", [20])
    assert relevance.apply_fit_labels([(10, 20)], session_id="s1",
                                      durable_path=durable, eph_path=eph) == 0


def test_unreadable_delivery_log_keeps_labels_rather_than_dropping_them():
    """None from _delivered_ids means 'could not ask', not 'nothing delivered'.
    Discarding irreplaceable labels on a transient DB error is the worse error."""
    durable, _ = _dbs()
    n = relevance.apply_fit_labels([(10, 20)], session_id="s1", durable_path=durable,
                                   eph_path="/nonexistent/dir/eph.db")
    assert n == 1
    assert _pairs(durable) == [(10, 20)]


def test_rejection_is_recorded_with_nowhere_vs_elsewhere_split():
    """A count alone cannot distinguish an invented id from a plumbing fault,
    and that ambiguity previously pointed at the wrong fix."""
    import json
    durable, eph = _dbs()
    _deliver(eph, "s1", [10])
    _deliver(eph, "other", [20])
    relevance.apply_fit_labels([(10, 20), (10, 999)], session_id="s1",
                               durable_path=durable, eph_path=eph)
    c = sqlite3.connect(eph)
    rows = c.execute("SELECT detail FROM metrics WHERE event='fit_pair_dropped'").fetchall()
    c.close()
    assert rows, "rejection must be visible as a metric"
    detail = json.loads(rows[0][0])
    assert detail["elsewhere"] == 1, "20 was delivered, just in another session"
    assert detail["nowhere"] == 1, "999 was never delivered anywhere"
