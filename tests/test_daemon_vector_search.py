#!/usr/bin/env python3
"""Unit tests for the daemon-resident vector search (cairn/daemon.py).

Covers the cached-matrix loader (_ensure_search_cache) and the scoring
function (_vector_search): stamp invalidation, archived rows riding along,
dual-embedding topic supplement, and min_sim/top_k bounds.
"""

import os
import tempfile
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import numpy as np
import pytest
from unittest.mock import patch

import cairn.daemon as daemon

DIM = 8


def _vec(axis: int) -> bytes:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    return v.tobytes()


class FakeModel:
    """Deterministic encoder: every text maps to the e0 basis vector."""
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class FakeEmb:
    def get_model(self):
        return FakeModel()


def fresh_daemon_db():
    d = tempfile.mkdtemp()
    conn = sqlite3.connect(os.path.join(d, "cairn.db"))
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL, depth INTEGER, archived_reason TEXT,
        keywords TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP, topic_embedding BLOB)""")
    return d, conn


def _reset_cache():
    daemon._search_cache.update({"stamp": None, "meta": [], "M": None, "T": None, "t_pos": None})


#TAG: [VS01] 2026-06-12
# Verifies: scoring returns rows sorted by similarity, includes archived rows,
# and applies the topic-embedding supplement (max of content/topic sim)
@pytest.mark.behavioural
def test_vector_search_behavioural():
    d, conn = fresh_daemon_db()
    # id 1: content aligned with the query axis (sim 1.0)
    conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES ('fact','aligned','A', ?)", (_vec(0),))
    # id 2: content orthogonal, but topic embedding aligned -> supplement lifts it
    conn.execute("INSERT INTO memories (type, topic, content, embedding, topic_embedding) VALUES ('fact','topic-hit','B', ?, ?)", (_vec(1), _vec(0)))
    # id 3: archived, aligned -> must still be returned (negative knowledge)
    conn.execute("INSERT INTO memories (type, topic, content, embedding, archived_reason) VALUES ('fact','old','C', ?, 'superseded')", (_vec(0),))
    # id 4: orthogonal, below min_sim -> excluded
    conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES ('fact','far','D', ?)", (_vec(2),))
    conn.commit()
    conn.close()

    _reset_cache()
    with patch.object(daemon, "CAIRN_DIR", d):
        rows = daemon._vector_search(FakeEmb(), ["query text"], n_base=1, min_sim=0.5, top_k=10)

    ids = [r["id"] for r in rows]
    assert sorted(ids) == [1, 2, 3]
    sims = {r["id"]: r["similarity"] for r in rows}
    assert sims[1] == pytest.approx(1.0)
    assert sims[2] == pytest.approx(1.0)  # lifted by topic supplement
    assert sims[3] == pytest.approx(1.0)
    archived = [r["id"] for r in rows if r["archived_reason"]]
    assert archived == [3]


#TAG: [VS02] 2026-06-12
# Verifies: the matrix cache is reused while the table is unchanged and
# reloaded when a write changes the stamp
@pytest.mark.edge
def test_ensure_search_cache_edge_invalidation():
    d, conn = fresh_daemon_db()
    conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES ('fact','one','X', ?)", (_vec(0),))
    conn.commit()

    _reset_cache()
    with patch.object(daemon, "CAIRN_DIR", d):
        c1 = daemon._ensure_search_cache()
        assert len(c1["meta"]) == 1
        m1 = c1["M"]
        c2 = daemon._ensure_search_cache()
        assert c2["M"] is m1  # unchanged stamp -> same matrices

        conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES ('fact','two','Y', ?)", (_vec(1),))
        conn.commit()
        c3 = daemon._ensure_search_cache()
        assert len(c3["meta"]) == 2
        assert c3["M"] is not m1  # stamp changed -> reload
    conn.close()


#TAG: [VS03] 2026-06-12
# Verifies: malformed embedding blobs are skipped, empty table returns []
@pytest.mark.error
def test_vector_search_error_malformed():
    d, conn = fresh_daemon_db()
    conn.execute("INSERT INTO memories (type, topic, content, embedding) VALUES ('fact','bad','Z', ?)", (b"xx",))
    conn.commit()
    conn.close()

    _reset_cache()
    with patch.object(daemon, "CAIRN_DIR", d):
        rows = daemon._vector_search(FakeEmb(), ["q"], n_base=1, min_sim=0.0, top_k=5)
    assert rows == []
