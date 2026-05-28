"""Tests for memory dual-embedding (schema v8) — topic_embedding column +
write path in storage.py + max-cos retrieval in embeddings.py +
idempotent backfill."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from cairn import init_db


def _fresh_db():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    return durable, td


def test_schema_v8_topic_embedding_column_exists():
    durable, _ = _fresh_db()
    conn = sqlite3.connect(durable)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "topic_embedding" in cols
    conn.execute("SELECT 1 FROM schema_version WHERE version = 8").fetchone()


def test_backfill_is_idempotent_and_fills_null_rows():
    durable, _ = _fresh_db()
    conn = sqlite3.connect(durable)
    for i, topic in enumerate(["alpha topic", "beta topic", "gamma topic"]):
        conn.execute(
            "INSERT INTO memories (id, type, topic, content, embedding) VALUES (?, ?, ?, ?, X'00')",
            (i + 1, "fact", topic, f"content {i}"),
        )
    conn.commit()
    conn.close()

    from cairn import memory_topic_embedding_backfill as bf
    r1 = bf.backfill(db_path=durable)
    assert r1["rows_needing_backfill"] == 3
    assert r1["embedded"] == 3
    r2 = bf.backfill(db_path=durable)
    assert r2["rows_needing_backfill"] == 0
    assert r2["embedded"] == 0


def test_backfill_skips_empty_topic():
    durable, _ = _fresh_db()
    conn = sqlite3.connect(durable)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, embedding) VALUES (1, 'fact', '', 'x', X'00')"
    )
    conn.commit()
    conn.close()
    from cairn import memory_topic_embedding_backfill as bf
    r = bf.backfill(db_path=durable)
    # Empty topic excluded by SQL filter — counts as 0 needing backfill
    assert r["rows_needing_backfill"] == 0


def test_dual_embedding_retrieval_picks_topic_when_content_misses():
    """A row whose topic strongly matches the prompt should be retrieved
    even when its content embedding doesn't."""
    durable, _ = _fresh_db()
    from cairn.embeddings import embed, to_blob, _brute_force_candidates
    import numpy as np

    # Content embedding intentionally orthogonal to topic; topic matches the query.
    target_topic = "calibration profile injection validation"
    query = "calibration injection validation"

    topic_vec = embed(target_topic)
    query_vec = embed(query)
    bogus_content_vec = np.zeros_like(topic_vec)
    bogus_content_vec[0] = 1.0  # arbitrary non-matching vector

    conn = sqlite3.connect(durable)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, embedding, topic_embedding) "
        "VALUES (1, 'fact', ?, 'unrelated text', ?, ?)",
        (target_topic, to_blob(bogus_content_vec), to_blob(topic_vec)),
    )
    conn.commit()

    candidates = _brute_force_candidates(conn, query_vec, k=5)
    conn.close()

    assert len(candidates) == 1
    # similarity should be the topic_sim, much higher than content_sim
    assert candidates[0]["similarity"] > 0.5


def test_dual_embedding_falls_back_when_topic_embedding_null():
    """Legacy rows without topic_embedding must still return their content cosine."""
    durable, _ = _fresh_db()
    from cairn.embeddings import embed, to_blob, _brute_force_candidates

    content_vec = embed("the user prefers terse responses")
    query_vec = embed("the user prefers terse responses")

    conn = sqlite3.connect(durable)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, embedding) "
        "VALUES (1, 'fact', 'terse-preference', 'the user prefers terse responses', ?)",
        (to_blob(content_vec),),
    )
    conn.commit()

    candidates = _brute_force_candidates(conn, query_vec, k=5)
    conn.close()

    assert len(candidates) == 1
    # Self-match should be near 1.0 regardless of topic_embedding presence
    assert candidates[0]["similarity"] > 0.8
