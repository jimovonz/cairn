"""Tests for the historical semantic-engagement backfill."""
import json
from datetime import datetime

import numpy as np
import pysqlite3 as sqlite3
import pytest

import cairn.backfill_semantic_engagement as bf


# --------------------------------------------------------------------------
# Timestamp + transcript alignment
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("2026-07-25 08:11:40", datetime(2026, 7, 25, 8, 11, 40)),      # SQLite UTC
    ("2026-07-25T08:11:40Z", datetime(2026, 7, 25, 8, 11, 40)),      # transcript ISO
    ("2026-07-25T08:11:40.123Z", datetime(2026, 7, 25, 8, 11, 40)),  # with millis
    ("2026-07-25T08:11:40+00:00", datetime(2026, 7, 25, 8, 11, 40)), # with offset
])
def test_parse_ts_normalises_both_stamp_formats(raw, expected):
    """Delivery rows and transcripts use different stamp formats; both are UTC."""
    assert bf._parse_ts(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "not-a-date"])
def test_parse_ts_returns_none_on_garbage(raw):
    assert bf._parse_ts(raw) is None


def _write_transcript(tmp_path, entries):
    p = tmp_path / "t.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return str(p)


def _assistant(ts, text):
    return {"timestamp": ts, "message": {"role": "assistant",
            "content": [{"type": "text", "text": text}]}}


def test_assistant_turns_extracts_text_in_time_order(tmp_path):
    t = _write_transcript(tmp_path, [
        _assistant("2026-07-25T10:00:02Z", "second"),
        {"timestamp": "2026-07-25T10:00:00Z", "message": {"role": "user", "content": "q"}},
        _assistant("2026-07-25T10:00:01Z", "first"),
    ])
    turns = bf._assistant_turns(t)
    assert [x[1] for x in turns] == ["first", "second"], "must sort by timestamp"


def test_assistant_turns_skips_toolonly_and_empty_messages(tmp_path):
    t = _write_transcript(tmp_path, [
        {"timestamp": "2026-07-25T10:00:00Z", "message": {"role": "assistant",
         "content": [{"type": "tool_use", "name": "Bash", "input": {}}]}},
        _assistant("2026-07-25T10:00:01Z", "   "),
        _assistant("2026-07-25T10:00:02Z", "real reply"),
    ])
    assert [x[1] for x in bf._assistant_turns(t)] == ["real reply"]


def test_assistant_turns_missing_file_is_empty():
    assert bf._assistant_turns("/nonexistent/path.jsonl") == []


def test_response_for_picks_the_turn_the_delivery_landed_in(tmp_path):
    turns = bf._assistant_turns(_write_transcript(tmp_path, [
        _assistant("2026-07-25T09:59:００Z".replace("０", "0"), "before delivery"),
        _assistant("2026-07-25T10:00:05Z", "the response"),
        _assistant("2026-07-25T10:05:00Z", "a later turn"),
    ]))
    assert bf._response_for(turns, "2026-07-25 10:00:00") == "the response"


def test_response_for_returns_none_when_nothing_follows(tmp_path):
    turns = bf._assistant_turns(_write_transcript(tmp_path, [
        _assistant("2026-07-25T09:00:00Z", "only an earlier turn"),
    ]))
    assert bf._response_for(turns, "2026-07-25 10:00:00") is None


# --------------------------------------------------------------------------
# End-to-end backfill
# --------------------------------------------------------------------------

MEM_VEC = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
RESP_VEC = np.asarray([0.9, 0.4359, 0.0], dtype=np.float32)  # cos(mem) ~= 0.90
CTX_VEC = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)      # cos(mem) == 0.0


@pytest.fixture
def backfill_env(tmp_path, monkeypatch):
    from cairn.embeddings import to_blob
    import cairn.embeddings as emb

    transcript = _write_transcript(tmp_path, [
        _assistant("2026-07-25T10:00:05Z", "paraphrased application of the memory"),
    ])
    eph, dur = tmp_path / "e.db", tmp_path / "d.db"

    e = sqlite3.connect(str(eph))
    e.execute("""CREATE TABLE memory_deliveries (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        memory_id INTEGER NOT NULL, context_text TEXT, grade INTEGER,
        engaged INTEGER, engaged_score REAL, engaged_method TEXT,
        delivered_at TIMESTAMP)""")
    e.execute("INSERT INTO memory_deliveries (session_id, memory_id, context_text, "
              "engaged, engaged_score, delivered_at) VALUES "
              "('sess', 1, 'the prompt context', 0, 0.02, '2026-07-25 10:00:00')")
    e.commit()

    d = sqlite3.connect(str(dur))
    d.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, embedding BLOB)")
    d.execute("INSERT INTO memories VALUES (1, ?)", (to_blob(MEM_VEC),))
    d.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, transcript_path TEXT)")
    d.execute("INSERT INTO sessions VALUES ('sess', ?)", (transcript,))
    d.commit()
    d.close()

    def fake_embed(text, allow_slow=True):
        return RESP_VEC if text.startswith("paraphrased") else CTX_VEC
    monkeypatch.setattr(emb, "embed", fake_embed)

    yield e, str(eph), str(dur)
    e.close()


def test_backfill_rescues_and_tags_the_row(backfill_env):
    conn, eph, dur = backfill_env
    stats = bf.backfill(eph_path=eph, durable_path=dur)
    assert stats == {"examined": 1, "rescued": 1, "no_response": 0,
                     "no_vector": 0, "no_transcript": 0}
    engaged, score, method = conn.execute(
        "SELECT engaged, engaged_score, engaged_method FROM memory_deliveries").fetchone()
    assert engaged == 1
    assert score == pytest.approx(0.90, abs=0.02), "stores cos(response, memory)"
    assert method == "semantic-backfill", "era must stay distinguishable from live rows"


def test_backfill_dry_run_writes_nothing(backfill_env):
    conn, eph, dur = backfill_env
    stats = bf.backfill(dry_run=True, eph_path=eph, durable_path=dur)
    assert stats["rescued"] == 1
    engaged, method = conn.execute(
        "SELECT engaged, engaged_method FROM memory_deliveries").fetchone()
    assert engaged == 0 and method is None, "dry run must not mutate"


def test_backfill_is_idempotent(backfill_env):
    conn, eph, dur = backfill_env
    bf.backfill(eph_path=eph, durable_path=dur)
    second = bf.backfill(eph_path=eph, durable_path=dur)
    assert second["examined"] == 0, "tagged rows must not be re-examined"


def test_backfill_tags_examined_but_unrescued_rows_as_lexical(backfill_env, monkeypatch):
    """A row the semantic rule rejects is still marked, so recall is measurable."""
    import cairn.embeddings as emb
    conn, eph, dur = backfill_env
    monkeypatch.setattr(emb, "embed", lambda text, allow_slow=True: CTX_VEC)  # no margin
    stats = bf.backfill(eph_path=eph, durable_path=dur)
    assert stats["rescued"] == 0
    engaged, method = conn.execute(
        "SELECT engaged, engaged_method FROM memory_deliveries").fetchone()
    assert engaged == 0 and method == "lexical"


def test_backfill_reports_missing_transcript_without_crashing(backfill_env):
    conn, eph, dur = backfill_env
    d = sqlite3.connect(dur)
    d.execute("UPDATE sessions SET transcript_path = '/nonexistent/x.jsonl'")
    d.commit()
    d.close()
    stats = bf.backfill(eph_path=eph, durable_path=dur)
    assert stats["no_transcript"] == 1 and stats["rescued"] == 0
