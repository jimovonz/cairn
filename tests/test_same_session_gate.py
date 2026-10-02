#!/usr/bin/env python3
"""Same-session injection gate (hooks/hook_helpers.py).

A memory written in the current session is redundant while its producing turn is
still in the model's context, but becomes the only surviving copy after
compaction. The gate drops same-session rows unless a compaction watermark shows
they predate the cut. These tests pin that distinction and the watermark parsing.

Regression for the observed self-echo: memories written by session 01a0c833 were
re-injected on later turns of the same session (cairn ids 2336462287228/233/235).
"""

import json
import os
import tempfile

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3

import pytest
from unittest.mock import patch

import hooks.hook_helpers as hh

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"ssg_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        archived_reason TEXT, keywords TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE hook_state (session_id TEXT NOT NULL, key TEXT NOT NULL,
        value TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))"""
    )
    conn.commit()
    return db_path, conn


def _row(mem_id, session_id, updated_at):
    return {"id": mem_id, "session_id": session_id, "updated_at": updated_at}


# --- the gate itself -------------------------------------------------------

def test_no_compaction_drops_all_same_session():
    p = [_row(1, "s1", "2026-09-22 08:20:41"), _row(2, "other", "2026-09-22 08:20:41")]
    with patch.object(hh, "load_hook_state", return_value=None), \
         patch.object(hh, "record_metric"):
        kp, kg = hh._drop_live_same_session(p, [], "s1")
    assert [r["id"] for r in kp] == [2], "same-session row must be dropped with no watermark"


def test_pre_compaction_same_session_recovered():
    p = [_row(1, "s1", "2026-09-22 08:20:41")]
    with patch.object(hh, "load_hook_state", return_value="2026-09-22 09:00:00"), \
         patch.object(hh, "record_metric"):
        kp, _ = hh._drop_live_same_session(p, [], "s1")
    assert [r["id"] for r in kp] == [1], "pre-compaction row is recovery and must survive"


def test_post_compaction_same_session_dropped():
    p = [_row(1, "s1", "2026-09-22 09:30:00")]
    with patch.object(hh, "load_hook_state", return_value="2026-09-22 09:00:00"), \
         patch.object(hh, "record_metric"):
        kp, _ = hh._drop_live_same_session(p, [], "s1")
    assert kp == [], "row written after the cut is still live and must be dropped"


def test_cross_session_untouched_without_watermark():
    g = [_row(9, "other", "2026-09-22 08:20:41")]
    with patch.object(hh, "load_hook_state", return_value=None), \
         patch.object(hh, "record_metric"):
        _, kg = hh._drop_live_same_session([], g, "s1")
    assert [r["id"] for r in kg] == [9]


def test_unparseable_timestamp_is_fail_open_for_cross_session():
    # A cross-session row with a junk timestamp must still be kept.
    g = [_row(9, "other", "not-a-date")]
    with patch.object(hh, "load_hook_state", return_value="2026-09-22 09:00:00"), \
         patch.object(hh, "record_metric"):
        _, kg = hh._drop_live_same_session([], g, "s1")
    assert [r["id"] for r in kg] == [9]


def test_norm_ts_formats_equivalent():
    assert hh._norm_ts("2026-09-22T09:00:00.000Z") == "2026-09-22 09:00:00"
    assert hh._norm_ts("2026-09-22 09:00:00") == "2026-09-22 09:00:00"
    assert hh._norm_ts("garbage") is None


# --- compaction parsing ----------------------------------------------------

def test_last_compaction_ts_takes_the_latest():
    path = os.path.join(TEST_DIR, "transcript.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "session", "timestamp": "2026-09-22T08:00:00Z"}) + "\n")
        f.write(json.dumps({"type": "message", "timestamp": "2026-09-22T08:10:00Z"}) + "\n")
        f.write(json.dumps({"type": "compaction", "timestamp": "2026-09-22T08:30:00.000Z"}) + "\n")
        f.write(json.dumps({"type": "compaction", "timestamp": "2026-09-22T09:00:00.000Z"}) + "\n")
    assert hh.last_compaction_ts(path) == "2026-09-22 09:00:00"


def test_last_compaction_ts_none_when_absent_or_missing():
    assert hh.last_compaction_ts(None) is None
    assert hh.last_compaction_ts("/nonexistent/xyz.jsonl") is None
    path = os.path.join(TEST_DIR, "nocomp.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "message"}) + "\n")
    assert hh.last_compaction_ts(path) is None


def test_last_compaction_ts_recognises_claude_summary_marker():
    path = os.path.join(TEST_DIR, "claude_comp.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "user", "isCompactSummary": True,
                            "timestamp": "2026-09-22T10:00:00.000Z"}) + "\n")
    assert hh.last_compaction_ts(path) == "2026-09-22 10:00:00"


def test_record_compaction_watermark_caches_timestamp():
    db_path, conn = fresh_db()
    conn.close()
    path = os.path.join(TEST_DIR, "wm.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "compaction", "timestamp": "2026-09-22T09:00:00Z"}) + "\n")
    with patch.object(hh, "DB_PATH", db_path), \
         patch("cairn.config.EPHEMERAL_DB_PATH", db_path):
        hh.record_compaction_watermark("s1", path)
        assert hh.load_hook_state("s1", "compaction_ts") == "2026-09-22 09:00:00"
        # Second call with unchanged size is a no-op and must not error.
        hh.record_compaction_watermark("s1", path)


def test_record_compaction_watermark_incremental_picks_up_new_records():
    db_path, conn = fresh_db()
    conn.close()
    path = os.path.join(TEST_DIR, "inc.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "message"}) + "\n")
    with patch.object(hh, "DB_PATH", db_path), \
         patch("cairn.config.EPHEMERAL_DB_PATH", db_path):
        hh.record_compaction_watermark("s2", path)
        assert hh.load_hook_state("s2", "compaction_ts") is None
        assert int(hh.load_hook_state("s2", "compaction_scan_offset")) == os.path.getsize(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "compaction", "timestamp": "2026-09-22T09:00:00Z"}) + "\n")
        hh.record_compaction_watermark("s2", path)
        assert hh.load_hook_state("s2", "compaction_ts") == "2026-09-22 09:00:00"
        assert int(hh.load_hook_state("s2", "compaction_scan_offset")) == os.path.getsize(path)


def test_record_compaction_watermark_leaves_partial_line():
    db_path, conn = fresh_db()
    conn.close()
    path = os.path.join(TEST_DIR, "partial.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "message"}) + "\n" + '{"type":"comp')
    with patch.object(hh, "DB_PATH", db_path), \
         patch("cairn.config.EPHEMERAL_DB_PATH", db_path):
        hh.record_compaction_watermark("s3", path)
        # Offset must stop before the partial line so it is re-read once complete.
        assert int(hh.load_hook_state("s3", "compaction_scan_offset")) < os.path.getsize(path)


# --- render-boundary integration ------------------------------------------

def test_build_context_xml_drops_same_session_entry():
    db_path, conn = fresh_db()
    conn.close()
    p = [_row(1, "s1", "2026-09-22 08:20:41"), _row(2, "other", "2026-09-22 08:20:41")]
    with patch.object(hh, "DB_PATH", db_path), \
         patch("cairn.config.EPHEMERAL_DB_PATH", db_path):
        xml = hh.build_context_xml("q", "Proj", "per-prompt", p, [], session_id="s1")
    assert 'id="2"' in xml
    assert 'id="1"' not in xml, "same-session entry leaked through the render boundary"
