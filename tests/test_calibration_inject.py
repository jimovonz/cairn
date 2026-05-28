"""Phase 3 UserPromptSubmit injector tests."""

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

from cairn import calibration_inject, init_db


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
        "content": "row content",
        "kw": "tests,commit",
        "qf": '["should I commit", "ready to push"]',
        "source": "explicit",
        "confidence": 0.7,
        "pinned": 0,
        "layer": "subject",
        "session_scope": None,
        "embedding": None,
        "archived_at": None,
        "superseded_by": None,
    }
    defaults.update(kw)
    conn.execute(
        "INSERT INTO calibration_rows (content, kw, qf, source, confidence, "
        "pinned, layer, session_scope, embedding, archived_at, superseded_by) "
        "VALUES (:content, :kw, :qf, :source, :confidence, :pinned, :layer, "
        ":session_scope, :embedding, :archived_at, :superseded_by)",
        defaults,
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def test_tokenise_handles_hyphens_and_paths():
    toks = calibration_inject._tokenise(
        "How do I run cairn-calibration-analyser to debug session arc?"
    )
    # Compound identifiers preserved verbatim by prompt_keywords.
    assert "cairn-calibration-analyser" in toks
    # Stopwords / function words filtered.
    assert "how" not in toks
    assert "the" not in toks


def test_kw_overlap_jaccard():
    toks = {"commit", "tests"}
    # row carries "tests,commit,review" — 2 in common / 3 union = 0.667
    score = calibration_inject._kw_overlap(toks, "tests,commit,review")
    assert 0.6 < score < 0.7


def test_kw_overlap_empty_returns_zero():
    assert calibration_inject._kw_overlap(set(), "anything") == 0.0
    assert calibration_inject._kw_overlap({"x"}, None) == 0.0
    assert calibration_inject._kw_overlap({"x"}, "") == 0.0


def test_retrieve_filters_archived():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    archived = _seed_row(conn, content="archived row",
                          archived_at="2026-05-01")
    active = _seed_row(conn, content="active row",
                       kw="prefer,terse,brief",
                       qf='["how should I respond", "keep it short"]')
    conn.commit()
    conn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        rows = calibration_inject.retrieve_calibration(
            "I prefer terse responses", "sess-x",
            similarity_floor=0.0)
    ids = [r["id"] for r in rows]
    assert active in ids
    assert archived not in ids


def test_retrieve_filters_already_delivered():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn, content="prefer terse", kw="terse,brief",
                    qf='["short response", "keep brief"]')
    conn.commit()
    conn.close()
    # Mark already-delivered in this session
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO calibration_deliveries (session_id, turn_index, row_id) "
        "VALUES ('sess-x', 0, ?)", (rid,))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        rows = calibration_inject.retrieve_calibration(
            "I prefer terse responses", "sess-x", similarity_floor=0.0)
    assert all(r["id"] != rid for r in rows)


def test_retrieve_includes_session_scoped_to_same_session():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    same = _seed_row(conn, content="same scope row",
                     kw="terse",
                     session_scope="sess-x")
    other = _seed_row(conn, content="other scope row",
                      kw="terse",
                      session_scope="sess-y")
    conn.commit()
    conn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        rows = calibration_inject.retrieve_calibration(
            "terse responses", "sess-x", similarity_floor=0.0)
    ids = [r["id"] for r in rows]
    assert same in ids
    assert other not in ids


def test_retrieve_applies_similarity_floor():
    """The similarity floor drops rows with computable-but-low cosine
    similarity. Rows that lack an embedding entirely (degraded mode)
    are NOT filtered by the floor — they fall back to composite ranking
    only. To test the floor we seed a synthetic blob unlikely to match
    the prompt.
    """
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    # Seed a random-ish embedding blob — unlikely to match any prompt
    try:
        from cairn import embeddings as emb
        vec = emb.embed("totally unrelated alpha beta")
        blob = emb.to_blob(vec) if vec is not None else None
    except Exception:
        blob = None
    _seed_row(conn, content="totally unrelated", kw="x,y,z",
              qf='["alpha", "beta"]', embedding=blob)
    conn.commit()
    conn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        rows = calibration_inject.retrieve_calibration(
            "completely different topic foo bar", "sess-x",
            similarity_floor=0.99)  # near-impossible match
    # With embedding present + impossible floor, the row is filtered
    # away. With no embedding (daemon unavailable in CI), the row
    # passes through degraded — either is acceptable.
    if blob is not None:
        assert rows == []


def test_render_block_groups_by_layer():
    rows = [
        {"id": 1, "content": "always X", "source": "explicit",
         "confidence": 0.9, "pinned": 0, "layer": "general"},
        {"id": 2, "content": "for python Y", "source": "observation",
         "confidence": 0.6, "pinned": 0, "layer": "subject"},
        {"id": 3, "content": "OVERRIDE Z", "source": "explicit",
         "confidence": 0.95, "pinned": 1, "layer": "general"},
    ]
    block = calibration_inject.render_calibration_block(rows)
    assert "<calibration_profile>" in block
    assert "<override" in block
    assert "<general" in block
    assert "<subject>" in block
    assert "OVERRIDE Z" in block
    assert "always X" in block
    assert "for python Y" in block


def test_render_block_empty_returns_empty_string():
    assert calibration_inject.render_calibration_block([]) == ""


def test_log_deliveries_writes_rows_and_increments_turn_index():
    durable, eph, td = _fresh_dbs()
    rows = [{"id": 1, "similarity": 0.71},
            {"id": 2, "similarity": 0.65}]
    with patch.object(calibration_inject, "EPH_DB_PATH", eph):
        n1 = calibration_inject.log_deliveries(rows, "sess-x")
        n2 = calibration_inject.log_deliveries(rows, "sess-x")
    assert n1 == 2
    assert n2 == 2
    conn = sqlite3.connect(eph)
    deliveries = conn.execute(
        "SELECT row_id, turn_index, similarity FROM calibration_deliveries "
        "ORDER BY id"
    ).fetchall()
    conn.close()
    # 4 rows total — two per call, turn_index 0 then 1
    assert len(deliveries) == 4
    turn_indexes = {d[1] for d in deliveries}
    assert turn_indexes == {0, 1}


def test_log_deliveries_no_op_on_empty():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration_inject, "EPH_DB_PATH", eph):
        n = calibration_inject.log_deliveries([], "sess-x")
    assert n == 0


def test_inject_for_prompt_end_to_end_with_no_match():
    durable, eph, td = _fresh_dbs()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt(
            "tell me about python decorators", "sess-x")
    assert out == ""


def test_inject_for_prompt_fails_open_on_missing_db():
    # Point at a non-existent DB — should return empty string, not raise
    with patch.object(calibration_inject, "DB_PATH", "/tmp/no-such-db.db"), \
         patch.object(calibration_inject, "EPH_DB_PATH", "/tmp/no-such-eph.db"):
        out = calibration_inject.inject_for_prompt("hi", "sess-x")
    assert out == ""


def test_pinned_rows_promoted_over_higher_score_non_pinned():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    # Pinned with mediocre kw match
    pinned_id = _seed_row(conn, content="PINNED rule",
                          kw="anything", confidence=0.5,
                          pinned=1, layer="general")
    # Non-pinned with strong kw match
    strong_id = _seed_row(conn, content="strong-match rule",
                          kw="commit,push,tests",
                          confidence=0.9, pinned=0)
    conn.commit()
    conn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        rows = calibration_inject.retrieve_calibration(
            "should I commit and push these tests", "sess-x",
            similarity_floor=0.0)
    assert rows[0]["id"] == pinned_id


def test_db_query_excludes_superseded():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    superseded_id = _seed_row(conn, content="old rule",
                              kw="terse", superseded_by=999)
    fresh_id = _seed_row(conn, content="new rule", kw="terse")
    conn.commit()
    conn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        rows = calibration_inject.retrieve_calibration(
            "terse responses", "sess-x", similarity_floor=0.0)
    ids = [r["id"] for r in rows]
    assert fresh_id in ids
    assert superseded_id not in ids


def test_inject_short_circuits_when_disabled():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    _seed_row(conn, content="row")
    conn.commit()
    conn.close()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", "calibration_disabled", "1"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt("anything", "sess-x")
    assert out == ""


def test_inject_short_circuits_when_globally_disabled():
    durable, eph, td = _fresh_dbs()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        (calibration_inject.GLOBAL_STATE_SESSION, "calibration_disabled", "1"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt("anything", "sess-x")
    assert out == ""


def test_session_muted_row_excluded_from_inject():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn, content="row should be muted", kw="terse,brief")
    conn.commit()
    conn.close()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", f"calibration_mute_{rid}", "1"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        rows = calibration_inject.retrieve_calibration(
            "terse responses please", "sess-x", similarity_floor=0.0)
        muted = calibration_inject._session_muted_ids("sess-x")
    assert rid in muted
    # retrieve doesn't filter — inject_for_prompt does. Verify via end-to-end:
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt(
            "terse responses please", "sess-x")
    assert "row should be muted" not in out


def test_mode_override_row_prepended_to_block():
    durable, eph, td = _fresh_dbs()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", "calibration_mode", "expert"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt("hello", "sess-x")
    assert "EXPERT" in out
    assert "override" in out


def test_mode_override_does_not_log_synthetic_delivery():
    durable, eph, td = _fresh_dbs()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", "calibration_mode", "novice"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        calibration_inject.inject_for_prompt("hello", "sess-x")
    econn = sqlite3.connect(eph)
    n = econn.execute(
        "SELECT count(*) FROM calibration_deliveries WHERE row_id < 0"
    ).fetchone()[0]
    econn.close()
    assert n == 0


def test_inject_short_circuits_when_disabled():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    _seed_row(conn, content="row")
    conn.commit()
    conn.close()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", "calibration_disabled", "1"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt("anything", "sess-x")
    assert out == ""


def test_inject_short_circuits_when_globally_disabled():
    durable, eph, td = _fresh_dbs()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        (calibration_inject.GLOBAL_STATE_SESSION, "calibration_disabled", "1"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt("anything", "sess-x")
    assert out == ""


def test_session_muted_row_excluded_from_inject():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn, content="row should be muted", kw="terse,brief")
    conn.commit()
    conn.close()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", f"calibration_mute_{rid}", "1"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        rows = calibration_inject.retrieve_calibration(
            "terse responses please", "sess-x", similarity_floor=0.0)
        muted = calibration_inject._session_muted_ids("sess-x")
    assert rid in muted
    # retrieve doesn't filter — inject_for_prompt does. Verify via end-to-end:
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt(
            "terse responses please", "sess-x")
    assert "row should be muted" not in out


def test_mode_override_row_prepended_to_block():
    durable, eph, td = _fresh_dbs()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", "calibration_mode", "expert"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.inject_for_prompt("hello", "sess-x")
    assert "EXPERT" in out
    assert "override" in out


def test_mode_override_does_not_log_synthetic_delivery():
    durable, eph, td = _fresh_dbs()
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("sess-x", "calibration_mode", "novice"))
    econn.commit()
    econn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        calibration_inject.inject_for_prompt("hello", "sess-x")
    econn = sqlite3.connect(eph)
    n = econn.execute(
        "SELECT count(*) FROM calibration_deliveries WHERE row_id < 0"
    ).fetchone()[0]
    econn.close()
    assert n == 0

# ---------------------------------------------------------------------------
# Per-qf sidecar (v7 schema) — symmetric intent retrieval tests
# ---------------------------------------------------------------------------

def _seed_qf(conn, row_id: int, qf_strings: list) -> None:
    from cairn.embeddings import embed, to_blob
    for i, s in enumerate(qf_strings):
        v = embed(s)
        if v is None:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO calibration_qf_embeddings "
            "(row_id, qf_index, qf_text, embedding) VALUES (?, ?, ?, ?)",
            (row_id, i, s, to_blob(v)))


def test_per_qf_max_cos_picks_best_qf_not_average():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn, content="user prefers brevity",
                    qf='["irrelevant kernel BPF probe stuff", "totally unrelated automotive CAN bus", "just give me the answer briefly"]')
    conn.commit()
    _seed_qf(conn, rid, [
        "irrelevant kernel BPF probe stuff",
        "totally unrelated automotive CAN bus",
        "just give me the answer briefly",
    ])
    conn.commit()
    conn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.retrieve_calibration(
            "be brief just answer", "sess-x", similarity_floor=0.30)
    assert len(out) == 1
    assert out[0]["id"] == rid
    assert out[0]["similarity"] >= 0.30


def test_row_without_qf_embeddings_falls_back_to_single_vector():
    durable, eph, td = _fresh_dbs()
    from cairn.embeddings import embed, to_blob
    vec = embed("the user prefers terse responses with no hedging")
    blob = to_blob(vec)
    conn = sqlite3.connect(durable)
    rid = _seed_row(conn, content="terse responses preferred",
                    qf='["be terse"]', embedding=blob)
    conn.commit()
    conn.close()
    with patch.object(calibration_inject, "DB_PATH", durable), \
         patch.object(calibration_inject, "EPH_DB_PATH", eph):
        out = calibration_inject.retrieve_calibration(
            "the user prefers terse responses with no hedging",
            "sess-x", similarity_floor=0.30)
    assert len(out) == 1
    assert out[0]["id"] == rid
    assert out[0]["similarity"] is not None


def test_per_qf_cascade_delete_when_row_hard_deleted():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    conn.execute("PRAGMA foreign_keys = ON")
    rid = _seed_row(conn, content="x", qf='["a","b"]')
    conn.commit()
    _seed_qf(conn, rid, ["a", "b"])
    conn.commit()
    n_before = conn.execute(
        "SELECT COUNT(*) FROM calibration_qf_embeddings WHERE row_id = ?",
        (rid,)).fetchone()[0]
    assert n_before == 2
    conn.execute("DELETE FROM calibration_rows WHERE id = ?", (rid,))
    conn.commit()
    n_after = conn.execute(
        "SELECT COUNT(*) FROM calibration_qf_embeddings WHERE row_id = ?",
        (rid,)).fetchone()[0]
    assert n_after == 0
    conn.close()


def test_backfill_is_idempotent():
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    _seed_row(conn, content="x", qf='["alpha", "beta"]')
    _seed_row(conn, content="y", qf='["gamma"]')
    _seed_row(conn, content="z", qf='[]')
    conn.commit()
    conn.close()
    from cairn import calibration_qf_backfill
    r1 = calibration_qf_backfill.backfill(db_path=durable)
    assert r1["rows_processed"] == 2
    assert r1["rows_skipped_no_qf"] == 1
    assert r1["qf_strings_embedded"] == 3
    r2 = calibration_qf_backfill.backfill(db_path=durable)
    assert r2["rows_processed"] == 0
    assert r2["rows_skipped_already_backfilled"] == 2
    assert r2["qf_strings_embedded"] == 0
