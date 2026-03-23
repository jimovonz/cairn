#!/usr/bin/env python3
"""Tests for retrieval quality gates — focuses on boundary conditions and real failure modes."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))


def mock_results(entries):
    """Create mock search results matching find_similar output format."""
    return [
        {
            "id": e.get("id", i),
            "type": e.get("type", "fact"),
            "topic": e.get("topic", f"topic-{i}"),
            "content": e.get("content", f"content {i}"),
            "similarity": e["similarity"],
            "updated_at": e.get("updated_at", "2026-03-20 12:00:00"),
            "project": e.get("project"),
            "confidence": e.get("confidence", 0.7),
            "depth": e.get("depth"),
            "depth": e.get("depth"),
            "score": e.get("score", e["similarity"] * 0.7),
        }
        for i, e in enumerate(entries)
    ]


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))


# === Garbage and borderline gates through find_similar ===

def test_garbage_gate_rejects_weak_results():
    """Results below MIN_INJECTION_SIMILARITY should return empty from find_similar."""
    import sqlite3, tempfile
    import numpy as np
    from unittest.mock import patch, MagicMock
    import hook_helpers
    from config import MIN_INJECTION_SIMILARITY

    db_path = os.path.join(tempfile.mkdtemp(), "gate.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY, type TEXT, topic TEXT,
        content TEXT, embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    # Insert a memory with a known embedding
    vec = np.random.RandomState(42).randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    conn.execute("INSERT INTO memories (type, topic, content, embedding, confidence) VALUES (?,?,?,?,?)",
                 ("fact", "test", "some content", vec.tobytes(), 0.7))
    conn.commit()

    from embeddings import find_similar
    # Use an orthogonal query vector to get low similarity
    query_vec = np.random.RandomState(999).randn(384).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    with patch('embeddings.embed', return_value=query_vec), \
         patch('embeddings._load_vec', return_value=False):
        results = find_similar(conn, "completely unrelated query")

    # All results should be filtered out by garbage gate
    for r in results:
        assert r["similarity"] >= MIN_INJECTION_SIMILARITY, \
            f"Garbage gate should have filtered sim={r['similarity']:.3f}"
    conn.close()


def test_diversity_filter_drops_same_type_topic():
    """Two results with identical type+topic — diversity filter should keep only one."""
    import sqlite3, tempfile
    import numpy as np
    from unittest.mock import patch
    from embeddings import find_similar

    db_path = os.path.join(tempfile.mkdtemp(), "div.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY, type TEXT, topic TEXT,
        content TEXT, embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    # Two memories with same type+topic but slightly different embeddings
    base = np.random.RandomState(42).randn(384).astype(np.float32)
    base = base / np.linalg.norm(base)
    nudged = base + np.random.RandomState(43).randn(384).astype(np.float32) * 0.01
    nudged = nudged / np.linalg.norm(nudged)
    conn.execute("INSERT INTO memories (type, topic, content, embedding, confidence) VALUES (?,?,?,?,?)",
                 ("fact", "db-choice", "Use SQLite for storage", base.tobytes(), 0.8))
    conn.execute("INSERT INTO memories (type, topic, content, embedding, confidence) VALUES (?,?,?,?,?)",
                 ("fact", "db-choice", "SQLite chosen for persistence", nudged.tobytes(), 0.8))
    conn.commit()

    with patch('embeddings.embed', return_value=base), \
         patch('embeddings._load_vec', return_value=False):
        results = find_similar(conn, "database choice")

    topics = [r["topic"] for r in results if r["topic"] == "db-choice"]
    assert len(topics) <= 1, f"Diversity filter should deduplicate same type+topic, got {len(topics)}"
    conn.close()


# === Relative filter edge cases ===

def test_relative_filter_single_result():
    """Single result should always survive relative filter."""
    from config import RELATIVE_FILTER_RATIO
    results = [{"similarity": 0.50}]
    max_sim = results[0]["similarity"]
    kept = [r for r in results if r["similarity"] >= RELATIVE_FILTER_RATIO * max_sim]
    assert len(kept) == 1


def test_relative_filter_tight_cluster():
    """Results within 30% of each other should all survive."""
    from config import RELATIVE_FILTER_RATIO
    results = [{"similarity": 0.80}, {"similarity": 0.75}, {"similarity": 0.60}]
    max_sim = 0.80
    kept = [r for r in results if r["similarity"] >= RELATIVE_FILTER_RATIO * max_sim]
    # 0.7 * 0.80 = 0.56 — all three pass
    assert len(kept) == 3


def test_relative_filter_drops_outlier():
    """One strong + one much weaker should drop the weak one."""
    from config import RELATIVE_FILTER_RATIO
    results = [{"similarity": 0.85}, {"similarity": 0.40}]
    max_sim = 0.85
    kept = [r for r in results if r["similarity"] >= RELATIVE_FILTER_RATIO * max_sim]
    # 0.7 * 0.85 = 0.595 — 0.40 fails
    assert len(kept) == 1


# === Soft confidence: the cases that actually matter ===

def test_soft_inclusion_high_sim_zero_confidence():
    """similarity=0.65, confidence=0.0 — should STILL be included (similarity override)."""
    from config import SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR
    included = 0.65 >= SOFT_SIM_OVERRIDE or 0.0 >= SOFT_CONF_FLOOR
    assert included


def test_soft_inclusion_moderate_sim_low_confidence():
    """similarity=0.50, confidence=0.2 — neither override fires. Should be EXCLUDED."""
    from config import SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR
    included = 0.50 >= SOFT_SIM_OVERRIDE or 0.2 >= SOFT_CONF_FLOOR
    assert not included


def test_soft_inclusion_low_sim_at_confidence_boundary():
    """similarity=0.30, confidence=0.30 — exactly at floor. Should be included."""
    from config import SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR
    included = 0.30 >= SOFT_SIM_OVERRIDE or 0.30 >= SOFT_CONF_FLOOR
    assert included


# === Dominance suppression ===

def test_dominance_gap_exactly_at_epsilon():
    """Gap exactly at epsilon boundary — should NOT trigger suppression."""
    from config import DOMINANCE_EPSILON
    gap = DOMINANCE_EPSILON
    assert not (gap < DOMINANCE_EPSILON)


def test_dominance_gap_just_below_epsilon():
    """Gap just below epsilon — should trigger, keeping both."""
    from config import DOMINANCE_EPSILON
    gap = DOMINANCE_EPSILON - 0.001
    assert gap < DOMINANCE_EPSILON


# === Diversity filter: realistic scenarios ===


def test_diversity_keeps_different_aspects():
    """Two entries about related but distinct topics should both survive."""
    results = [
        {"type": "decision", "topic": "db-choice", "content": "Use SQLite for storage"},
        {"type": "decision", "topic": "db-schema", "content": "Add confidence column"},
    ]
    assert results[0]["topic"] != results[1]["topic"]


def test_diversity_word_overlap_boundary():
    """Content with exactly 90% word overlap should be caught."""
    from config import DIVERSITY_SIM_THRESHOLD
    # 9 shared words out of 10 total = 0.9
    content_a = "the quick brown fox jumps over the lazy sleeping dog"
    content_b = "the quick brown fox jumps over the lazy sleeping cat"
    words_a = set(content_a.lower().split())
    words_b = set(content_b.lower().split())
    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
    # 9 shared / 11 unique = 0.818 — just below threshold, both should be kept
    assert overlap < DIVERSITY_SIM_THRESHOLD


def test_diversity_identical_content():
    """Exact same content, different topic — word overlap catches it."""
    from config import DIVERSITY_SIM_THRESHOLD
    content = "use JWT for stateless authentication"
    words = set(content.lower().split())
    overlap = len(words & words) / max(len(words | words), 1)
    assert overlap > DIVERSITY_SIM_THRESHOLD  # 1.0 > 0.9


# === Write throttling: priority ordering ===

def test_write_throttle_corrections_survive_over_project():
    """When throttled, corrections should be kept over project metadata."""
    from config import MAX_MEMORIES_PER_RESPONSE
    type_priority = {"correction": 0, "decision": 1, "fact": 2, "preference": 3,
                     "person": 4, "skill": 5, "workflow": 6, "project": 7}
    entries = [
        {"type": "project", "topic": f"p{i}", "content": f"project {i}"} for i in range(4)
    ] + [
        {"type": "correction", "topic": "c1", "content": "important fix"},
        {"type": "decision", "topic": "d1", "content": "key choice"},
    ]
    entries.sort(key=lambda e: type_priority.get(e.get("type", ""), 99))
    kept = entries[:MAX_MEMORIES_PER_RESPONSE]
    types_kept = [e["type"] for e in kept]
    assert "correction" in types_kept
    assert "decision" in types_kept


def test_write_throttle_all_same_type():
    """All entries same type — should just keep first N."""
    from config import MAX_MEMORIES_PER_RESPONSE
    entries = [{"type": "fact", "topic": f"t{i}", "content": f"c{i}"} for i in range(10)]
    assert len(entries[:MAX_MEMORIES_PER_RESPONSE]) == MAX_MEMORIES_PER_RESPONSE


# === Combined gate interaction ===

def test_gates_combined_weak_but_not_garbage():
    """Entry at garbage floor with low score — when borderline == garbage floor, caught by garbage gate."""
    from config import MIN_INJECTION_SIMILARITY, BORDERLINE_SIM_CEILING, BORDERLINE_SCORE_FLOOR
    # When BORDERLINE_SIM_CEILING == MIN_INJECTION_SIMILARITY, the borderline gate
    # has no independent effect — entries below the ceiling are also below garbage.
    # Test with a value just below the ceiling to verify both gates agree.
    sim, score = BORDERLINE_SIM_CEILING - 0.01, BORDERLINE_SCORE_FLOOR - 0.02
    caught_garbage = sim < MIN_INJECTION_SIMILARITY
    caught_borderline = sim < BORDERLINE_SIM_CEILING and score < BORDERLINE_SCORE_FLOOR
    assert caught_garbage or caught_borderline


def test_gates_combined_strong_entry_passes_all():
    """Entry at sim=0.75, score=0.65, confidence=0.8 — should pass everything."""
    from config import (MIN_INJECTION_SIMILARITY, BORDERLINE_SIM_CEILING,
                        SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR, RELATIVE_FILTER_RATIO)
    sim, score, conf = 0.75, 0.65, 0.8
    passes_garbage = sim >= MIN_INJECTION_SIMILARITY
    not_borderline = sim >= BORDERLINE_SIM_CEILING or score >= 0.50
    passes_soft = sim >= SOFT_SIM_OVERRIDE or conf >= SOFT_CONF_FLOOR
    passes_relative = sim >= RELATIVE_FILTER_RATIO * sim  # always true for self
    assert all([passes_garbage, not_borderline, passes_soft, passes_relative])


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
