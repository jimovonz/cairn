"""Phase 6 self-modification tests."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import calibration_selfmod as sm, init_db


def _fresh_dbs():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "eph.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    return durable, eph, td


def _seed(conn, **kw):
    defaults = {"content": "row", "source": "explicit", "confidence": 0.7,
                "pinned": 0, "delivered_count": 0, "followed_count": 0,
                "ignored_count": 0, "corrected_count": 0,
                "created_at": None, "archived_at": None}
    defaults.update(kw)
    cols = [k for k in defaults if defaults[k] is not None]
    vals = [defaults[k] for k in cols]
    placeholders = ",".join("?" * len(cols))
    conn.execute(
        f"INSERT INTO calibration_rows ({','.join(cols)}) VALUES ({placeholders})",
        vals)
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


# ---------------------------------------------------------------------------
# auto_archive_low_follow
# ---------------------------------------------------------------------------

def test_auto_archive_archives_below_threshold():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    bad = _seed(conn, delivered_count=15, followed_count=2)  # 13.3%
    ok = _seed(conn, delivered_count=15, followed_count=10)  # 66.7%
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        archived = sm.auto_archive_low_follow()
    assert bad in archived
    assert ok not in archived
    conn = sqlite3.connect(durable)
    row = conn.execute(
        "SELECT archived_at, archive_reason FROM calibration_rows WHERE id = ?",
        (bad,)).fetchone()
    conn.close()
    assert row[0] is not None
    assert row[1] == "auto-archive-low-follow"


def test_auto_archive_respects_minimum_deliveries():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    _seed(conn, delivered_count=5, followed_count=0)
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        archived = sm.auto_archive_low_follow()
    assert archived == []  # below min deliveries threshold


def test_auto_archive_idempotent():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    _seed(conn, delivered_count=15, followed_count=0)
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        first = sm.auto_archive_low_follow()
        second = sm.auto_archive_low_follow()
    assert len(first) == 1
    assert second == []  # already archived


# ---------------------------------------------------------------------------
# auto_promote_corroborated
# ---------------------------------------------------------------------------

def test_auto_promote_requires_min_sessions():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed(conn, delivered_count=10, followed_count=9)  # 90%
    conn.commit()
    conn.close()
    # 2 distinct followed sessions — below min of 3
    econn = sqlite3.connect(eph)
    for sid in ("s1", "s2"):
        econn.execute(
            "INSERT INTO calibration_deliveries "
            "(session_id, turn_index, row_id, outcome) VALUES (?, ?, ?, ?)",
            (sid, 0, rid, "followed"))
    econn.commit()
    econn.close()
    with patch.object(sm, "DB_PATH", durable), \
         patch.object(sm, "EPH_DB_PATH", eph):
        promoted = sm.auto_promote_corroborated()
    assert promoted == []


def test_auto_promote_pins_corroborated_row():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed(conn, delivered_count=10, followed_count=9)
    conn.commit()
    conn.close()
    econn = sqlite3.connect(eph)
    for sid in ("s1", "s2", "s3", "s4"):
        econn.execute(
            "INSERT INTO calibration_deliveries "
            "(session_id, turn_index, row_id, outcome) VALUES (?, ?, ?, ?)",
            (sid, 0, rid, "followed"))
    econn.commit()
    econn.close()
    with patch.object(sm, "DB_PATH", durable), \
         patch.object(sm, "EPH_DB_PATH", eph):
        promoted = sm.auto_promote_corroborated()
    assert rid in promoted
    conn = sqlite3.connect(durable)
    pinned, conf = conn.execute(
        "SELECT pinned, confidence FROM calibration_rows WHERE id = ?",
        (rid,)).fetchone()
    conn.close()
    assert pinned == 1
    assert conf == sm.PINNED_CONFIDENCE


# ---------------------------------------------------------------------------
# decay_unused
# ---------------------------------------------------------------------------

def test_decay_unused_halves_at_half_life():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    # observation half_life=30; created 30 days ago → should halve
    long_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    rid = _seed(conn, source="observation", confidence=0.20,
                created_at=long_ago)
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        n = sm.decay_unused()
    assert n == 1
    conn = sqlite3.connect(durable)
    new_conf = conn.execute(
        "SELECT confidence FROM calibration_rows WHERE id = ?", (rid,)
    ).fetchone()[0]
    conn.close()
    # ~0.10 (halved from 0.20)
    assert 0.08 < new_conf < 0.12


def test_decay_unused_skips_fresh_rows():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    rid = _seed(conn, source="observation", confidence=0.20,
                created_at=yesterday)
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        n = sm.decay_unused()
    assert n == 0


def test_decay_unused_skips_delivered_rows():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    rid = _seed(conn, source="explicit", confidence=0.9,
                created_at=long_ago, delivered_count=5, followed_count=3)
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        n = sm.decay_unused()
    assert n == 0  # delivered_count > 0, untouched


# ---------------------------------------------------------------------------
# Tier 2 surfaces
# ---------------------------------------------------------------------------

def test_surface_low_follow_rephrase_picks_mid_band():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    mid = _seed(conn, delivered_count=10, followed_count=5)  # 50% — in band
    high = _seed(conn, delivered_count=10, followed_count=9)  # 90% — skip
    low = _seed(conn, delivered_count=10, followed_count=1)   # 10% — skip
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        surfaced = sm.surface_low_follow_rephrase()
    assert surfaced == [mid]


def test_surface_low_follow_idempotent():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed(conn, delivered_count=10, followed_count=5)
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        first = sm.surface_low_follow_rephrase()
        second = sm.surface_low_follow_rephrase()
    assert first == [rid]
    assert second == []  # already in queue


def test_surface_promotion_candidates_picks_corroborated_but_low_followrate():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed(conn, delivered_count=10, followed_count=6)  # 60%
    conn.commit()
    conn.close()
    econn = sqlite3.connect(eph)
    for sid in ("s1", "s2", "s3"):
        econn.execute(
            "INSERT INTO calibration_deliveries "
            "(session_id, turn_index, row_id, outcome) VALUES (?, ?, ?, ?)",
            (sid, 0, rid, "followed"))
    econn.commit()
    econn.close()
    with patch.object(sm, "DB_PATH", durable), \
         patch.object(sm, "EPH_DB_PATH", eph):
        surfaced = sm.surface_promotion_candidates()
    assert rid in surfaced


def test_resolve_review_item_marks_resolved():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed(conn, delivered_count=10, followed_count=5)
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable):
        sm.surface_low_follow_rephrase()
        conn = sqlite3.connect(durable)
        item = conn.execute(
            "SELECT id FROM calibration_review_queue"
        ).fetchone()
        conn.close()
        ok = sm.resolve_review_item(item[0], "approved")
    assert ok
    conn = sqlite3.connect(durable)
    res = conn.execute(
        "SELECT resolved_at, resolution FROM calibration_review_queue "
        "WHERE id = ?", (item[0],)).fetchone()
    conn.close()
    assert res[0] is not None
    assert res[1] == "approved"


# ---------------------------------------------------------------------------
# run_all orchestrator
# ---------------------------------------------------------------------------

def test_run_all_orchestrates_all_passes():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    archive_me = _seed(conn, delivered_count=15, followed_count=2)
    surface_me = _seed(conn, delivered_count=10, followed_count=5)
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable), \
         patch.object(sm, "EPH_DB_PATH", eph):
        report = sm.run_all()
    assert archive_me in report["archived"]
    assert surface_me in report["surfaced_rephrase"]
    assert "decayed_count" in report
