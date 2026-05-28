"""Phase 7 CLAUDE.md import tests."""

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

from cairn import calibration_import_claude_md as imp, init_db


def _fresh_dbs():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "eph.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    return durable, eph, td


def _write_md(td, body, name="CLAUDE.md"):
    path = os.path.join(td, name)
    with open(path, "w") as f:
        f.write(body)
    return path


SAMPLE_CLAUDE_MD = """\
# Project rules

## How I work

I prefer terse responses without preamble.
- Always run tests before commit.
- Never use sed/awk for code edits; use the Edit tool.

Stop guessing the hash — do the mechanical work every time.

## Examples

Example: this is an example line and should not be picked up.
For instance: also not a preference.
This is just narrative text, not a preference statement.

## Other

Random note that doesn't start with a preference marker.
"""


def test_extract_preferences_picks_first_person_markers():
    prefs = imp.extract_preferences(SAMPLE_CLAUDE_MD)
    joined = "\n".join(prefs)
    assert "prefer terse responses" in joined
    assert "Always run tests" in joined
    assert "Never use sed/awk" in joined
    assert "Stop guessing the hash" in joined


def test_extract_preferences_skips_examples_and_narrative():
    prefs = imp.extract_preferences(SAMPLE_CLAUDE_MD)
    joined = "\n".join(prefs)
    assert "this is an example line" not in joined
    assert "Random note" not in joined
    assert "narrative text" not in joined


def test_extract_preferences_skips_short_lines():
    out = imp.extract_preferences("I prefer X.\nI like Y.\n")
    # "I prefer X." is 12 chars after stripping — picks up
    # "I like Y." is 9 — skipped
    assert any("I prefer X" in p for p in out)
    assert all("I like Y" not in p for p in out)


def test_import_file_inserts_rows_with_high_confidence():
    durable, eph, td = _fresh_dbs()
    path = _write_md(td, SAMPLE_CLAUDE_MD)
    with patch.object(imp, "DB_PATH", durable), \
         patch.object(imp, "EPH_DB_PATH", eph):
        report = imp.import_file(path)
    assert report["imported"] >= 3
    conn = sqlite3.connect(durable)
    rows = conn.execute(
        "SELECT source, confidence, pinned, layer FROM calibration_rows"
    ).fetchall()
    conn.close()
    for src, conf, pinned, layer in rows:
        assert src == "explicit"
        assert conf == 0.90
        assert pinned == 1
        assert layer == "general"


def test_import_file_idempotent_via_sha():
    durable, eph, td = _fresh_dbs()
    path = _write_md(td, SAMPLE_CLAUDE_MD)
    with patch.object(imp, "DB_PATH", durable), \
         patch.object(imp, "EPH_DB_PATH", eph):
        first = imp.import_file(path)
        second = imp.import_file(path)
    assert first["imported"] >= 3
    assert second.get("skipped") == "already-imported"
    assert second["imported"] == 0


def test_import_file_dedups_against_existing_rows():
    durable, eph, td = _fresh_dbs()
    path = _write_md(td, SAMPLE_CLAUDE_MD)
    # Pre-insert an identical preference
    conn = sqlite3.connect(durable)
    conn.execute(
        "INSERT INTO calibration_rows (content, source, confidence) "
        "VALUES ('I prefer terse responses without preamble.', "
        "'explicit', 0.5)")
    conn.commit()
    conn.close()
    with patch.object(imp, "DB_PATH", durable), \
         patch.object(imp, "EPH_DB_PATH", eph):
        report = imp.import_file(path)
    # The exact-text dup should be skipped — the other prefs still land
    conn = sqlite3.connect(durable)
    n = conn.execute(
        "SELECT count(*) FROM calibration_rows WHERE content LIKE "
        "'%terse responses without preamble%'").fetchone()[0]
    conn.close()
    assert n == 1


def test_import_file_force_reimports():
    durable, eph, td = _fresh_dbs()
    path = _write_md(td, SAMPLE_CLAUDE_MD)
    with patch.object(imp, "DB_PATH", durable), \
         patch.object(imp, "EPH_DB_PATH", eph):
        imp.import_file(path)
        second = imp.import_file(path, force=True)
    assert "skipped" not in second
    # But content-level dedup still applies — most prefs not re-inserted
    assert second["imported"] == 0


def test_import_file_handles_missing_path():
    durable, eph, td = _fresh_dbs()
    with patch.object(imp, "DB_PATH", durable), \
         patch.object(imp, "EPH_DB_PATH", eph):
        report = imp.import_file("/tmp/does-not-exist-CLAUDE.md")
    assert report.get("error") == "file not found"
    assert report["imported"] == 0


def test_main_cli_smoke(capsys):
    durable, eph, td = _fresh_dbs()
    path = _write_md(td, "I prefer terse responses without preamble.")
    with patch.object(imp, "DB_PATH", durable), \
         patch.object(imp, "EPH_DB_PATH", eph):
        rc = imp.main([path])
    out = capsys.readouterr().out
    assert rc == 0
    # Strip any init noise above the JSON
    first = out.find("{")
    last = out.rfind("}")
    rep = json.loads(out[first:last + 1])
    assert rep["imported"] >= 1
