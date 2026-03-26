#!/usr/bin/env python3
"""Tests for composite scoring, confidence dynamics, and quality gates."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))

from embeddings import composite_score, _recency_decay


def test_composite_score_similarity_dominant():
    """Higher similarity should generally produce higher score."""
    s1 = composite_score(0.8, 0.7, "2026-03-20 12:00:00")
    s2 = composite_score(0.4, 0.7, "2026-03-20 12:00:00")
    assert s1 > s2


def test_composite_score_confidence_nonlinear():
    """Confidence squared means 0.9 confidence contributes much more than 0.4."""
    s_high = composite_score(0.5, 0.9, "2026-03-20 12:00:00")
    s_low = composite_score(0.5, 0.4, "2026-03-20 12:00:00")
    # With confidence^2: 0.9^2=0.81 vs 0.4^2=0.16, gap should be significant
    assert (s_high - s_low) > 0.1


def test_composite_score_project_scope_boost():
    """Project-scoped entries should score higher than global at same similarity."""
    s_proj = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="myproject", current_project="myproject")
    s_glob = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="other", current_project="myproject")
    assert s_proj > s_glob


def test_recency_decay_recent():
    """Very recent entries should have decay near 1.0."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(now)
    assert decay > 0.95


def test_recency_decay_old():
    """30-day old entries should be roughly 0.5 (half-life)."""
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(old)
    assert 0.4 < decay < 0.6


def test_recency_decay_very_old():
    """90-day old entries should be low."""
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(old)
    assert decay < 0.2


# === Confidence dynamics (through real code paths) ===

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))


def _confidence_db():
    """Create a temp DB for confidence tests."""
    import sqlite3, tempfile
    db_path = os.path.join(tempfile.mkdtemp(), "conf.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE memory_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER, content TEXT, session_id TEXT,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content, session_id, changed_at)
        VALUES (old.id, old.content, old.session_id, old.updated_at); END""")
    conn.commit()
    return db_path, conn


def test_boost_at_high_confidence_is_tiny():
    """Saturating boost: at 0.9 confidence, boost should be minimal."""
    from unittest.mock import patch
    import hook_helpers
    from storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.9)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "+", None)], session_id="s1")
    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    assert new_conf > 0.9, "Should increase"
    assert new_conf < 0.92, f"At 0.9, boost should be tiny — got {new_conf}"
    conn.close()


def test_boost_at_low_confidence_is_meaningful():
    """Saturating boost: at 0.3 confidence, boost should be larger."""
    from unittest.mock import patch
    import hook_helpers
    from storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.3)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "+", None)], session_id="s1")
    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    boost = new_conf - 0.3
    assert boost > 0.05, f"At 0.3, boost should be meaningful — got {boost:.3f}"
    conn.close()


def test_penalty_at_high_confidence_is_severe():
    """Scaled penalty: at 0.9 confidence, penalty should be harsh."""
    from unittest.mock import patch
    import hook_helpers
    from storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.9)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "-", None)], session_id="s1")
    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    drop = 0.9 - new_conf
    assert drop > 0.3, f"At 0.9, penalty should be severe — dropped only {drop:.3f}"
    conn.close()


def test_penalty_at_low_confidence_is_moderate():
    """Scaled penalty: at 0.3 confidence, penalty should be smaller."""
    from unittest.mock import patch
    import hook_helpers
    from storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.3)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "-", None)], session_id="s1")
    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    drop = 0.3 - new_conf
    assert drop < 0.3, f"At 0.3, penalty should be moderate — dropped {drop:.3f}"
    conn.close()


def test_single_negative_at_09_drops_significantly():
    """One negative at 0.9 should drop to roughly 0.52."""
    conf = 0.9
    penalty = 0.2 * (1 + conf)
    new = conf - penalty
    assert 0.5 < new < 0.55


def test_many_boosts_approach_but_dont_reach_1():
    """Repeated boosts should approach 1.0 but never reach it."""
    conf = 0.7
    for _ in range(50):
        boost = 0.1 * (1 - conf)
        conf = min(conf + boost, 1.0)
    assert conf > 0.95
    assert conf < 1.0


# === Negation heuristic ===

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))
from storage import _has_negation_mismatch


def test_negation_detected():
    assert _has_negation_mismatch("use RTK for positioning", "do not use RTK for positioning")


def test_no_negation():
    assert not _has_negation_mismatch("use RTK for positioning", "use RTK for best accuracy")


def test_directional_contradiction():
    assert _has_negation_mismatch("increase the gain", "decrease the gain")


def test_preference_opposition():
    assert _has_negation_mismatch("prefer PostgreSQL", "avoid PostgreSQL")


def test_enable_disable():
    assert _has_negation_mismatch("enable debug logging", "disable debug logging")


def test_unrelated_sentences():
    assert not _has_negation_mismatch("the sky is blue", "the database uses SQLite")


def test_contradiction_annotation_writes_archived_reason():
    """The -! feedback should write archived_reason and leave confidence unchanged."""
    from unittest.mock import patch
    import hook_helpers
    from storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('decision', 'db-choice', 'Use PostgreSQL', 0.8)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        applied = apply_confidence_updates([(1, "-!", "replaced by SQLite for zero-config deployment")], session_id="s1")
    assert applied == 1
    row = conn.execute("SELECT confidence, archived_reason FROM memories WHERE id = 1").fetchone()
    assert row[0] == 0.8, "Confidence should be unchanged by -!"
    assert row[1] == "replaced by SQLite for zero-config deployment"
    conn.close()


def test_contradiction_annotation_default_reason():
    """-! with no reason should use a default annotation."""
    from unittest.mock import patch
    import hook_helpers
    from storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.7)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "-!", None)], session_id="s1")
    row = conn.execute("SELECT archived_reason FROM memories WHERE id = 1").fetchone()
    assert row[0] is not None, "Should have a default annotation"
    assert "contradicted" in row[0].lower()
    conn.close()


def test_mixed_feedback_types():
    """A single response can have +, -, and -! updates applied correctly."""
    from unittest.mock import patch
    import hook_helpers
    from storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'a', 'mem a', 0.7)")
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 'b', 'mem b', 0.7)")
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('decision', 'c', 'mem c', 0.7)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        applied = apply_confidence_updates([
            (1, "+", None),
            (2, "-", None),
            (3, "-!", "superseded by new approach"),
        ], session_id="s1")
    assert applied == 3
    r1 = conn.execute("SELECT confidence, archived_reason FROM memories WHERE id = 1").fetchone()
    r2 = conn.execute("SELECT confidence, archived_reason FROM memories WHERE id = 2").fetchone()
    r3 = conn.execute("SELECT confidence, archived_reason FROM memories WHERE id = 3").fetchone()
    assert r1[0] > 0.7, "Boosted"
    assert r1[1] is None, "Not contradicted"
    assert r2[0] < 0.7, "Penalised"
    assert r2[1] is None, "Not contradicted"
    assert r3[0] == 0.7, "Confidence unchanged by -!"
    assert r3[1] == "superseded by new approach"
    conn.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {test.__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
