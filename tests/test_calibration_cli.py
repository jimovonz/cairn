"""Phase 4 cairn-calibration CLI behaviour tests.

Replaces the Phase 1 stub-tests with assertions on actual side effects
(rows added/muted/archived, hook_state writes, etc.).
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import calibration, init_db


def _fresh_dbs():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "eph.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    return durable, eph, td


def _seed_row(conn, **kw):
    defaults = {
        "content": "row", "kw": "k", "qf": "[]", "source": "explicit",
        "confidence": 0.7, "pinned": 0, "layer": "subject",
    }
    defaults.update(kw)
    conn.execute(
        "INSERT INTO calibration_rows (content, kw, qf, source, confidence, "
        "pinned, layer) VALUES (:content, :kw, :qf, :source, :confidence, "
        ":pinned, :layer)", defaults,
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _run(args):
    return calibration.main(args)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

def test_add_inserts_row_with_source_initial_confidence():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration, "DB_PATH", durable):
        rc = _run(["add", "--source", "explicit",
                   "--content", "user prefers terse"])
    assert rc == 0
    conn = sqlite3.connect(durable)
    row = conn.execute(
        "SELECT content, source, confidence, pinned, layer "
        "FROM calibration_rows"
    ).fetchone()
    conn.close()
    assert row[0] == "user prefers terse"
    assert row[1] == "explicit"
    assert row[2] == calibration.SOURCE_INITIAL_CONFIDENCE["explicit"]
    assert row[3] == 0
    assert row[4] == "general"


def test_add_pinned_uses_pinned_confidence_and_sets_flag():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration, "DB_PATH", durable):
        _run(["add", "--source", "explicit", "--content", "X", "--pin"])
    conn = sqlite3.connect(durable)
    row = conn.execute(
        "SELECT confidence, pinned FROM calibration_rows"
    ).fetchone()
    conn.close()
    assert row[0] == calibration.PINNED_CONFIDENCE
    assert row[1] == 1


def test_add_with_scope_sets_layer_subject():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration, "DB_PATH", durable):
        _run(["add", "--source", "observation", "--content", "Y",
              "--scope", "python"])
    conn = sqlite3.connect(durable)
    row = conn.execute(
        "SELECT layer, kw FROM calibration_rows"
    ).fetchone()
    conn.close()
    assert row[0] == "subject"
    assert row[1] == "python"


def test_add_rejects_invalid_source():
    rc = _run(["add", "--source", "bogus", "--content", "x"])
    assert rc == 1


# ---------------------------------------------------------------------------
# mute / unmute
# ---------------------------------------------------------------------------

def test_mute_permanent_archives_row():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn)
    conn.commit()
    conn.close()
    with patch.object(calibration, "DB_PATH", durable):
        rc = _run(["mute", str(rid)])
    assert rc == 0
    conn = sqlite3.connect(durable)
    archived, reason = conn.execute(
        "SELECT archived_at, archive_reason FROM calibration_rows WHERE id = ?",
        (rid,)).fetchone()
    conn.close()
    assert archived is not None
    assert reason == "muted"


def test_mute_session_only_writes_hook_state():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn)
    conn.commit()
    conn.close()
    with patch.object(calibration, "DB_PATH", durable), \
         patch.object(calibration, "EPH_DB_PATH", eph), \
         patch.dict(os.environ, {"CLAUDE_SESSION_ID": "sess-x"}):
        rc = _run(["mute", str(rid), "--session-only"])
    assert rc == 0
    econn = sqlite3.connect(eph)
    val = econn.execute(
        "SELECT value FROM hook_state WHERE session_id = 'sess-x' "
        "AND key = ?", (f"calibration_mute_{rid}",)
    ).fetchone()
    econn.close()
    assert val == ("1",)


def test_unmute_clears_both_archive_and_session_state():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn)
    conn.execute(
        "UPDATE calibration_rows SET archived_at = '2026-05-19', "
        "archive_reason = 'muted' WHERE id = ?", (rid,))
    conn.commit()
    conn.close()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", f"calibration_mute_{rid}", "1"))
    econn.commit()
    econn.close()
    with patch.object(calibration, "DB_PATH", durable), \
         patch.object(calibration, "EPH_DB_PATH", eph), \
         patch.dict(os.environ, {"CLAUDE_SESSION_ID": "sess-x"}):
        rc = _run(["unmute", str(rid)])
    assert rc == 0
    conn = sqlite3.connect(durable)
    row = conn.execute(
        "SELECT archived_at, archive_reason FROM calibration_rows WHERE id = ?",
        (rid,)).fetchone()
    conn.close()
    assert row == (None, None)


# ---------------------------------------------------------------------------
# disable / enable
# ---------------------------------------------------------------------------

def test_disable_session_only_writes_session_scoped_state():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration, "DB_PATH", durable), \
         patch.object(calibration, "EPH_DB_PATH", eph), \
         patch.dict(os.environ, {"CLAUDE_SESSION_ID": "sess-x"}):
        rc = _run(["disable", "--session-only"])
    assert rc == 0
    econn = sqlite3.connect(eph)
    row = econn.execute(
        "SELECT value FROM hook_state WHERE session_id = 'sess-x' "
        "AND key = 'calibration_disabled'"
    ).fetchone()
    econn.close()
    assert row == ("1",)


def test_disable_global_writes_global_state():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration, "DB_PATH", durable), \
         patch.object(calibration, "EPH_DB_PATH", eph):
        rc = _run(["disable"])
    assert rc == 0
    econn = sqlite3.connect(eph)
    row = econn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? "
        "AND key = 'calibration_disabled'",
        (calibration.GLOBAL_STATE_SESSION,)
    ).fetchone()
    econn.close()
    assert row == ("1",)


def test_enable_clears_both_global_and_session_state():
    durable, eph, td = _fresh_dbs()
    econn = sqlite3.connect(eph)
    for sid in ("sess-x", calibration.GLOBAL_STATE_SESSION):
        econn.execute(
            "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
            (sid, "calibration_disabled", "1"))
    econn.commit()
    econn.close()
    with patch.object(calibration, "DB_PATH", durable), \
         patch.object(calibration, "EPH_DB_PATH", eph), \
         patch.dict(os.environ, {"CLAUDE_SESSION_ID": "sess-x"}):
        _run(["enable"])
    econn = sqlite3.connect(eph)
    n = econn.execute(
        "SELECT count(*) FROM hook_state WHERE key = 'calibration_disabled'"
    ).fetchone()[0]
    econn.close()
    assert n == 0


# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------

def test_mode_writes_level_to_global_state():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration, "DB_PATH", durable), \
         patch.object(calibration, "EPH_DB_PATH", eph):
        rc = _run(["mode", "--level", "expert"])
    assert rc == 0
    econn = sqlite3.connect(eph)
    row = econn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? "
        "AND key = 'calibration_mode'",
        (calibration.GLOBAL_STATE_SESSION,)
    ).fetchone()
    econn.close()
    assert row == ("expert",)


def test_mode_session_only_writes_session_state():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration, "DB_PATH", durable), \
         patch.object(calibration, "EPH_DB_PATH", eph), \
         patch.dict(os.environ, {"CLAUDE_SESSION_ID": "s1"}):
        _run(["mode", "--level", "novice", "--session-only"])
    econn = sqlite3.connect(eph)
    row = econn.execute(
        "SELECT value FROM hook_state WHERE session_id = 's1' "
        "AND key = 'calibration_mode'"
    ).fetchone()
    econn.close()
    assert row == ("novice",)


def test_mode_rejects_invalid_level():
    rc = _run(["mode", "--level", "wizard"])
    assert rc == 1


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_removes_row_and_cascades_to_deliveries():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn)
    conn.commit()
    conn.close()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO calibration_deliveries (session_id, turn_index, row_id) "
        "VALUES (?, ?, ?)", ("sess-x", 0, rid))
    econn.commit()
    econn.close()
    with patch.object(calibration, "DB_PATH", durable), \
         patch.object(calibration, "EPH_DB_PATH", eph):
        rc = _run(["delete", str(rid)])
    assert rc == 0
    conn = sqlite3.connect(durable)
    n = conn.execute(
        "SELECT count(*) FROM calibration_rows WHERE id = ?", (rid,)
    ).fetchone()[0]
    conn.close()
    econn = sqlite3.connect(eph)
    nd = econn.execute(
        "SELECT count(*) FROM calibration_deliveries WHERE row_id = ?", (rid,)
    ).fetchone()[0]
    econn.close()
    assert n == 0
    assert nd == 0


# ---------------------------------------------------------------------------
# show-profile / review / history
# ---------------------------------------------------------------------------

def test_show_profile_renders_rows(capsys):
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    _seed_row(conn, content="prefer terse", source="explicit")
    _seed_row(conn, content="python style: pep8", source="observation",
              kw="python")
    conn.commit()
    conn.close()
    with patch.object(calibration, "DB_PATH", durable):
        rc = _run(["--show-profile"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prefer terse" in out
    assert "python style" in out


def test_show_profile_subject_filter(capsys):
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    _seed_row(conn, content="prefer terse", source="explicit", kw="brevity")
    _seed_row(conn, content="python style: pep8", source="observation",
              kw="python")
    conn.commit()
    conn.close()
    with patch.object(calibration, "DB_PATH", durable):
        _run(["--show-profile", "python"])
    out = capsys.readouterr().out
    assert "python" in out
    assert "prefer terse" not in out


def test_review_surfaces_low_follow_rows(capsys):
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    conn.execute(
        "INSERT INTO calibration_rows (content, source, confidence, "
        "delivered_count, followed_count) VALUES "
        "('row-A', 'explicit', 0.7, 15, 2)")  # follow=13.3% (below 20%)
    conn.execute(
        "INSERT INTO calibration_rows (content, source, confidence, "
        "delivered_count, followed_count) VALUES "
        "('row-B', 'explicit', 0.7, 15, 12)")  # follow=80% (above)
    conn.commit()
    conn.close()
    with patch.object(calibration, "DB_PATH", durable):
        _run(["--review"])
    out = capsys.readouterr().out
    assert "row-A" in out
    assert "row-B" not in out


def test_history_unknown_row_errors():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration, "DB_PATH", durable):
        rc = _run(["--history", "999"])
    assert rc == 1


def test_history_existing_row_prints_details(capsys):
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn, content="historical")
    conn.commit()
    conn.close()
    with patch.object(calibration, "DB_PATH", durable):
        rc = _run(["--history", str(rid)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "historical" in out
    assert "deliv=" in out


def test_no_args_prints_help():
    rc = _run([])
    assert rc == 1


def test_build_parser_all_subcommands():
    p = calibration.build_parser()
    p.parse_args(["--show-profile"])
    p.parse_args(["--review"])
    p.parse_args(["--history", "1"])
    p.parse_args(["add", "--source", "explicit", "--content", "x"])
    p.parse_args(["mute", "1"])
    p.parse_args(["unmute", "1"])
    p.parse_args(["disable"])
    p.parse_args(["enable"])
    p.parse_args(["mode", "--level", "expert"])
    p.parse_args(["delete", "1"])
