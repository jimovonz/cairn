"""Tests for the 2026-07-02 relevance-review changes: question-form sidecar
(schema v14), push suppression (memory_selfmod + injection filter), gated
correction bootstrap, and bootstrap query hygiene."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3

import numpy as np
import pytest

from cairn import init_db
from cairn.relevance import extract_question_forms, store_qf_embeddings
from cairn import memory_selfmod


def _fresh_dbs():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "cairn-ephemeral.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    return durable, eph


class _StubEmbedder:
    """Deterministic 4-dim embedder for sidecar tests."""

    def embed_batch(self, texts, allow_slow=True):
        out = []
        for t in texts:
            v = np.zeros(4, dtype=np.float32)
            v[hash(t) % 4] = 1.0
            out.append(v)
        return out

    def to_blob(self, v):
        return np.asarray(v, dtype=np.float32).tobytes()


# --- extract_question_forms ---------------------------------------------------

def test_extract_question_forms_selects_only_questions():
    kws = ["reranker", "how do I raise the CE floor", "what sets Z",
           "bge calibration", "why does retrieval miss?", "what floor"]
    got = extract_question_forms(kws)
    assert "how do I raise the CE floor" in got
    assert "what sets Z" in got
    assert "why does retrieval miss?" in got   # '?' qualifies regardless of length
    assert "reranker" not in got
    assert "bge calibration" not in got
    assert "what floor" not in got             # 2 words, no '?': not a question form


def test_extract_question_forms_empty_and_none():
    assert extract_question_forms(None) == []
    assert extract_question_forms([]) == []
    assert extract_question_forms(["", "  "]) == []


# --- store_qf_embeddings / schema v14 ------------------------------------------

def test_schema_v14_sidecar_table_exists():
    durable, _ = _fresh_dbs()
    conn = sqlite3.connect(durable)
    assert conn.execute(
        "SELECT 1 FROM schema_version WHERE version = 14").fetchone()
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info(memory_qf_embeddings)").fetchall()}
    assert cols == {"memory_id", "qf_index", "qf_text", "embedding"}
    conn.close()


def test_store_qf_embeddings_writes_and_rewrites():
    durable, _ = _fresh_dbs()
    conn = sqlite3.connect(durable)
    conn.execute("INSERT INTO memories (id, type, topic, content) "
                 "VALUES (1, 'fact', 't', 'c')")
    emb = _StubEmbedder()
    n = store_qf_embeddings(conn, 1, ["how do I test this", "plainkw"], emb)
    assert n == 1
    rows = conn.execute("SELECT qf_text FROM memory_qf_embeddings "
                        "WHERE memory_id = 1").fetchall()
    assert [r[0] for r in rows] == ["how do I test this"]
    # Rewrite replaces, never accumulates stale rows
    n = store_qf_embeddings(conn, 1, ["why does X fail now"], emb)
    assert n == 1
    rows = conn.execute("SELECT qf_text FROM memory_qf_embeddings "
                        "WHERE memory_id = 1").fetchall()
    assert [r[0] for r in rows] == ["why does X fail now"]
    conn.close()


def test_store_qf_embeddings_fail_soft():
    durable, _ = _fresh_dbs()
    conn = sqlite3.connect(durable)
    assert store_qf_embeddings(conn, 1, ["how do I x y"], None) == 0
    assert store_qf_embeddings(conn, 1, [], _StubEmbedder()) == 0
    conn.close()


# --- memory_selfmod suppression -------------------------------------------------

def _seed_deliveries(eph, memory_id, n, scored=0, engaged=0, grade=None):
    conn = sqlite3.connect(eph)
    for i in range(n):
        eng = None
        if i < engaged:
            eng = 1
        elif i < scored:
            eng = 0
        conn.execute(
            "INSERT INTO memory_deliveries (session_id, memory_id, engaged, grade) "
            "VALUES ('s', ?, ?, ?)",
            (memory_id, eng, grade if i == 0 else None))
    conn.commit()
    conn.close()


def _seed_memory(durable, memory_id, mtype="fact", suppressed=0):
    conn = sqlite3.connect(durable)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, push_suppressed) "
        "VALUES (?, ?, 't', 'c', ?)", (memory_id, mtype, suppressed))
    conn.commit()
    conn.close()


def test_suppress_dead_weight_flags_only_scored_silence():
    durable, eph = _fresh_dbs()
    _seed_memory(durable, 1)                       # scored silence -> suppress
    _seed_deliveries(eph, 1, 12, scored=5)
    _seed_memory(durable, 2)                       # unscored silence -> keep
    _seed_deliveries(eph, 2, 12, scored=0)
    _seed_memory(durable, 3)                       # engaged -> keep
    _seed_deliveries(eph, 3, 12, scored=5, engaged=1)
    _seed_memory(durable, 4, mtype="correction")   # correction: higher bar -> keep at 12
    _seed_deliveries(eph, 4, 12, scored=5)
    _seed_memory(durable, 5, mtype="preference")   # exempt type -> keep
    _seed_deliveries(eph, 5, 30, scored=10)
    with patch.object(memory_selfmod, "DB_PATH", durable), \
         patch.object(memory_selfmod, "EPH_DB_PATH", eph):
        suppressed = memory_selfmod.suppress_dead_weight()
    assert suppressed == [1]
    conn = sqlite3.connect(durable)
    assert conn.execute("SELECT push_suppressed FROM memories WHERE id=1").fetchone()[0] == 1
    assert conn.execute("SELECT push_suppressed FROM memories WHERE id=2").fetchone()[0] == 0
    conn.close()


def test_reactivate_engaged_unsuppresses():
    durable, eph = _fresh_dbs()
    _seed_memory(durable, 1, suppressed=1)
    _seed_deliveries(eph, 1, 3, scored=2, engaged=1)
    with patch.object(memory_selfmod, "DB_PATH", durable), \
         patch.object(memory_selfmod, "EPH_DB_PATH", eph):
        assert memory_selfmod.reactivate_engaged() == [1]
    conn = sqlite3.connect(durable)
    assert conn.execute("SELECT push_suppressed FROM memories WHERE id=1").fetchone()[0] == 0
    conn.close()


# --- push suppression injection filter ------------------------------------------

def test_push_suppression_filter_drops_flagged():
    durable, _ = _fresh_dbs()
    _seed_memory(durable, 1, suppressed=1)
    _seed_memory(durable, 2, suppressed=0)
    import hooks.hook_helpers as hh
    pr = [{"id": 1, "content": "a"}, {"id": 2, "content": "b"}]
    with patch.object(hh, "DB_PATH", durable):
        kept_pr, kept_gr = hh._push_suppression_filter(pr, [], "sess")
    assert [r["id"] for r in kept_pr] == [2]
    assert kept_gr == []


def test_push_suppression_filter_fails_open_without_column():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "old.db")
    conn = sqlite3.connect(durable)
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO memories (id, content) VALUES (1, 'a')")
    conn.commit()
    conn.close()
    import hooks.hook_helpers as hh
    pr = [{"id": 1, "content": "a"}]
    with patch.object(hh, "DB_PATH", durable):
        kept_pr, _ = hh._push_suppression_filter(pr, [], "sess")
    assert [r["id"] for r in kept_pr] == [1]


# --- gated correction bootstrap ---------------------------------------------------

def test_correction_bootstrap_noise_safe_without_embedder(monkeypatch):
    monkeypatch.setenv("CAIRN_SKIP_EMBEDDER", "1")
    from hooks.prompt_hook import correction_bootstrap
    assert correction_bootstrap("sess", "some first prompt") is None


def test_correction_bootstrap_requires_prompt():
    from hooks.prompt_hook import correction_bootstrap
    assert correction_bootstrap("sess", "") is None
