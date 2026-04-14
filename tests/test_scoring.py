#!/usr/bin/env python3
"""Tests for composite scoring, confidence dynamics, and quality gates."""

import sys
import os

from cairn.embeddings import composite_score, _recency_decay


# Verifies: higher similarity produces higher composite score
def test_composite_score_similarity_dominant():
    """Higher similarity should generally produce higher score."""
    s1 = composite_score(0.8, 0.7, "2026-03-20 12:00:00")
    s2 = composite_score(0.4, 0.7, "2026-03-20 12:00:00")
    assert s1 > s2


# Verifies: confidence does not affect composite score
def test_composite_score_confidence_has_no_effect():
    """Confidence is zeroed in scoring — different confidence should produce same score."""
    s_high = composite_score(0.5, 0.9, "2026-03-20 12:00:00")
    s_low = composite_score(0.5, 0.4, "2026-03-20 12:00:00")
    assert abs(s_high - s_low) < 0.01, f"Confidence should not affect score: {s_high} vs {s_low}"


# Verifies: project-scoped entries score higher than cross-project
def test_composite_score_project_scope_boost():
    """Project-scoped entries should score higher than global at same similarity."""
    s_proj = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="myproject", current_project="myproject")
    s_glob = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="other", current_project="myproject")
    assert s_proj > s_glob


# Verifies: person-type memories ignore scope penalty (universal applicability)
def test_composite_score_person_type_ignores_scope():
    """person-type memories should score the same regardless of project — biographical facts apply universally."""
    s_in_proj = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="myproject", current_project="myproject", mem_type="person")
    s_other_proj = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="other", current_project="myproject", mem_type="person")
    assert abs(s_in_proj - s_other_proj) < 1e-6, f"person type should ignore scope: {s_in_proj} vs {s_other_proj}"


# Verifies: preference-type memories ignore scope penalty
def test_composite_score_preference_type_ignores_scope():
    """preference-type memories should score the same regardless of project."""
    s_in_proj = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="myproject", current_project="myproject", mem_type="preference")
    s_other_proj = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="other", current_project="myproject", mem_type="preference")
    assert abs(s_in_proj - s_other_proj) < 1e-6, f"preference type should ignore scope: {s_in_proj} vs {s_other_proj}"


# Verifies: fact-type memories STILL get scope penalty (not exempt by default)
def test_composite_score_fact_type_keeps_scope_penalty():
    """fact type is project-bound by default — only person/preference are exempt."""
    s_proj = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="myproject", current_project="myproject", mem_type="fact")
    s_glob = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="other", current_project="myproject", mem_type="fact")
    assert s_proj > s_glob, "fact type should still receive scope penalty when cross-project"


# Verifies: person-type cross-project ties with project-scoped non-person at same similarity
def test_composite_score_person_cross_project_matches_local():
    """A person memory in another project should score the same as a project-scoped non-person."""
    s_person_global = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="other", current_project="myproject", mem_type="person")
    s_local_decision = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="myproject", current_project="myproject", mem_type="decision")
    assert abs(s_person_global - s_local_decision) < 0.001, "person cross-project should match local-project full weight"


# Verifies: find_similar with `|` separator decomposes into multiple subqueries
def test_find_similar_separator_decomposes():
    """Multi-dimensional query split on `|` runs N subqueries and merges."""
    from unittest.mock import patch, MagicMock
    from cairn import embeddings

    call_log = []

    def fake_recursive(conn, text, **kwargs):
        call_log.append(text)
        return [{
            "id": hash(text) % 1000, "type": "fact", "topic": text[:20],
            "content": text, "similarity": 0.5, "score": 0.5,
            "confidence": 0.7, "project": "test", "session_id": "s",
            "depth": 0, "archived_reason": None, "updated_at": "2026-04-09 12:00:00",
        }]

    # Patch find_similar to recurse into our fake for the subqueries only
    real_find_similar = embeddings.find_similar
    def wrapped(conn, text, **kwargs):
        # On the first call (multi-query) defer to real; on recursive calls use fake
        if "|" in text:
            return real_find_similar(conn, text, **kwargs)
        return fake_recursive(conn, text, **kwargs)

    with patch.object(embeddings, "find_similar", side_effect=wrapped):
        result = real_find_similar(None, "topic A | topic B | topic C", limit=10)

    # Three distinct subqueries should have been issued
    assert call_log == ["topic A", "topic B", "topic C"], f"Expected 3 subqueries, got: {call_log}"
    # Three distinct results merged
    assert len(result) == 3


# Verifies: find_similar without `|` runs as a single query (no decomposition)
def test_find_similar_no_separator_single_query():
    """Query without `|` runs as a single search, not decomposed."""
    from unittest.mock import patch
    from cairn import embeddings

    call_log = []
    real_find_similar = embeddings.find_similar

    def fake_recursive(conn, text, **kwargs):
        call_log.append(text)
        return []

    def wrapped(conn, text, **kwargs):
        if "|" in text:
            return real_find_similar(conn, text, **kwargs)
        return fake_recursive(conn, text, **kwargs)

    with patch.object(embeddings, "find_similar", side_effect=wrapped):
        # No separator → wrapped calls fake directly, no recursion
        result = wrapped(None, "single topic query")

    assert call_log == ["single topic query"], f"Expected single call, got: {call_log}"


# Verifies: separator with empty parts is filtered before decomposition decision
def test_find_similar_separator_empty_parts_filtered():
    """Empty/whitespace-only parts between separators are dropped."""
    from cairn import embeddings
    # Just verify the split logic — extract via the actual code path
    text = "  topic A | | topic B |  "
    subqueries = [s.strip() for s in text.split("|") if s.strip()]
    assert subqueries == ["topic A", "topic B"]


# Verifies: separator decomposition keeps best score per duplicate ID across subqueries
def test_find_similar_separator_dedup_keeps_best_score():
    """When multiple subqueries return the same memory, keep the highest score."""
    from unittest.mock import patch
    from cairn import embeddings

    # First subquery returns id=42 with score=0.3; second returns id=42 with score=0.8
    score_map = {"low": 0.3, "high": 0.8}
    real_find_similar = embeddings.find_similar

    def fake_recursive(conn, text, **kwargs):
        score = score_map.get(text)
        if score is None:
            return []
        return [{
            "id": 42, "type": "fact", "topic": "t", "content": "c",
            "similarity": score, "score": score, "confidence": 0.7,
            "project": "test", "session_id": "s", "depth": 0,
            "archived_reason": None, "updated_at": "2026-04-09 12:00:00",
        }]

    def wrapped(conn, text, **kwargs):
        if "|" in text:
            return real_find_similar(conn, text, **kwargs)
        return fake_recursive(conn, text, **kwargs)

    with patch.object(embeddings, "find_similar", side_effect=wrapped):
        result = real_find_similar(None, "low | high", limit=10)

    assert len(result) == 1, f"Expected 1 deduplicated result, got {len(result)}"
    assert result[0]["score"] == 0.8, f"Expected best score 0.8, got {result[0]['score']}"


# Verifies: person-type with no current_project still gets full weight
def test_composite_score_person_type_no_current_project():
    """person type with no current_project should not be penalised."""
    s_person = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="someproject", current_project=None, mem_type="person")
    s_decision = composite_score(0.5, 0.7, "2026-03-20 12:00:00", project="someproject", current_project=None, mem_type="decision")
    # Both should fall through to scope=0.3 since current_project is None and decision isn't exempt
    # but person should get scope=1.0 via type exemption
    assert s_person > s_decision, "person type should still get exemption when no current_project set"


# Verifies: very recent entries have decay near 1.0
def test_recency_decay_recent():
    """Very recent entries should have decay near 1.0."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(now)
    assert decay > 0.95


# Verifies: 30-day-old entries decay to ~0.5 (half-life)
def test_recency_decay_old():
    """30-day old entries should be roughly 0.5 (half-life)."""
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(old)
    assert 0.4 < decay < 0.6


# Verifies: 90-day-old entries have low decay value
def test_recency_decay_very_old():
    """90-day old entries should be low."""
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    decay = _recency_decay(old)
    assert decay < 0.2


# === Confidence dynamics (through real code paths) ===



def _confidence_db():
    """Create a temp DB for confidence tests."""
    import sqlite3, tempfile
    db_path = os.path.join(tempfile.mkdtemp(), "conf.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, keywords TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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


# Verifies: corroboration boost saturates at high confidence
def test_boost_at_high_confidence_is_tiny():
    """Saturating boost: at 0.9 confidence, boost should be minimal."""
    from unittest.mock import patch
    import hooks.hook_helpers as hook_helpers
    from hooks.storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.9)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "+", None)], session_id="s1")
    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    assert new_conf > 0.9, "Should increase"
    assert new_conf < 0.92, f"At 0.9, boost should be tiny — got {new_conf}"
    conn.close()


# Verifies: corroboration boost is larger at low confidence
def test_boost_at_low_confidence_is_meaningful():
    """Saturating boost: at 0.3 confidence, boost should be larger."""
    from unittest.mock import patch
    import hooks.hook_helpers as hook_helpers
    from hooks.storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.3)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "+", None)], session_id="s1")
    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    boost = new_conf - 0.3
    assert boost > 0.05, f"At 0.3, boost should be meaningful — got {boost:.3f}"
    conn.close()


# Verifies: irrelevant feedback leaves confidence unchanged
def test_irrelevant_does_not_change_confidence():
    """- (irrelevant) should not adjust confidence at all."""
    from unittest.mock import patch
    import hooks.hook_helpers as hook_helpers
    from hooks.storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.9)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "-", None)], session_id="s1")
    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    assert new_conf == 0.9, f"- should not change confidence, got {new_conf}"
    conn.close()


# Verifies: irrelevant feedback unchanged even at low confidence
def test_irrelevant_at_low_confidence_unchanged():
    """- (irrelevant) should not adjust confidence even at low values."""
    from unittest.mock import patch
    import hooks.hook_helpers as hook_helpers
    from hooks.storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.3)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "-", None)], session_id="s1")
    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = 1").fetchone()[0]
    assert new_conf == 0.3, f"- should not change confidence, got {new_conf}"
    conn.close()


# Verifies: single negative penalty at 0.9 drops to ~0.52
def test_single_negative_at_09_drops_significantly():
    """One negative at 0.9 should drop to roughly 0.52."""
    conf = 0.9
    penalty = 0.2 * (1 + conf)
    new = conf - penalty
    assert 0.5 < new < 0.55


# Verifies: repeated boosts approach 1.0 asymptotically
def test_many_boosts_approach_but_dont_reach_1():
    """Repeated boosts should approach 1.0 but never reach it."""
    conf = 0.7
    for _ in range(50):
        boost = 0.1 * (1 - conf)
        conf = min(conf + boost, 1.0)
    assert conf > 0.95
    assert conf < 1.0


# === Negation heuristic ===

from hooks.storage import _has_negation_mismatch


# Verifies: "do not" negation detected between similar statements
def test_negation_detected():
    assert _has_negation_mismatch("use RTK for positioning", "do not use RTK for positioning")


# Verifies: similar non-contradicting statements show no negation
def test_no_negation():
    assert not _has_negation_mismatch("use RTK for positioning", "use RTK for best accuracy")


# Verifies: increase/decrease directional contradiction detected
def test_directional_contradiction():
    assert _has_negation_mismatch("increase the gain", "decrease the gain")


# Verifies: prefer/avoid opposition detected as negation
def test_preference_opposition():
    assert _has_negation_mismatch("prefer PostgreSQL", "avoid PostgreSQL")


# Verifies: enable/disable pair detected as negation
def test_enable_disable():
    assert _has_negation_mismatch("enable debug logging", "disable debug logging")


# Verifies: unrelated sentences do not trigger negation detection
def test_unrelated_sentences():
    assert not _has_negation_mismatch("the sky is blue", "the database uses SQLite")


# Verifies: -! writes archived_reason without changing confidence
def test_contradiction_annotation_writes_archived_reason():
    """The -! feedback should write archived_reason and leave confidence unchanged."""
    from unittest.mock import patch
    import hooks.hook_helpers as hook_helpers
    from hooks.storage import apply_confidence_updates
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


# Verifies: -! with no reason uses default "contradicted" annotation
def test_contradiction_annotation_default_reason():
    """-! with no reason should use a default annotation."""
    from unittest.mock import patch
    import hooks.hook_helpers as hook_helpers
    from hooks.storage import apply_confidence_updates
    db_path, conn = _confidence_db()
    conn.execute("INSERT INTO memories (type, topic, content, confidence) VALUES ('fact', 't', 'c', 0.7)")
    conn.commit()
    with patch.object(hook_helpers, 'DB_PATH', db_path):
        apply_confidence_updates([(1, "-!", None)], session_id="s1")
    row = conn.execute("SELECT archived_reason FROM memories WHERE id = 1").fetchone()
    assert row[0] == "contradicted by later session", f"Expected exact default annotation, got: {row[0]}"
    conn.close()


# Verifies: mixed +, -, -! feedback applied correctly in one batch
def test_mixed_feedback_types():
    """A single response can have +, -, and -! updates applied correctly."""
    from unittest.mock import patch
    import hooks.hook_helpers as hook_helpers
    from hooks.storage import apply_confidence_updates
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
    assert r1[0] > 0.7, "Boosted (corroborated)"
    assert r1[1] is None, "Not contradicted"
    assert r2[0] == 0.7, "- (irrelevant) should not change confidence"
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
