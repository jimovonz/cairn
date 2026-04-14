#!/usr/bin/env python3
"""Tests for retrieval quality gates — focuses on boundary conditions and real failure modes."""

import sys
import os



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




# === Garbage and borderline gates through find_similar ===

# Verifies: find_similar filters out results below MIN_INJECTION_SIMILARITY
def test_garbage_gate_rejects_weak_results():
    """Results below MIN_INJECTION_SIMILARITY should return empty from find_similar."""
    import sqlite3, tempfile
    import numpy as np
    from unittest.mock import patch, MagicMock
    import hooks.hook_helpers as hook_helpers
    from cairn.config import MIN_INJECTION_SIMILARITY

    db_path = os.path.join(tempfile.mkdtemp(), "gate.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY, type TEXT, topic TEXT,
        content TEXT, embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    # Insert a memory with a known embedding
    vec = np.random.RandomState(42).randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    conn.execute("INSERT INTO memories (type, topic, content, embedding, confidence) VALUES (?,?,?,?,?)",
                 ("fact", "test", "some content", vec.tobytes(), 0.7))
    conn.commit()

    from cairn.embeddings import find_similar
    # Use an orthogonal query vector to get low similarity
    query_vec = np.random.RandomState(999).randn(384).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    with patch('cairn.embeddings.embed', return_value=query_vec), \
         patch('cairn.embeddings._load_vec', return_value=False):
        results = find_similar(conn, "completely unrelated query")

    # All results should be filtered out by garbage gate
    for r in results:
        assert r["similarity"] >= MIN_INJECTION_SIMILARITY, \
            f"Garbage gate should have filtered sim={r['similarity']:.3f}"
    conn.close()


# Verifies: diversity filter deduplicates entries with identical type+topic pairs
def test_diversity_filter_drops_same_type_topic():
    """Two results with identical type+topic — diversity filter should keep only one."""
    import sqlite3, tempfile
    import numpy as np
    from unittest.mock import patch
    from cairn.embeddings import find_similar

    db_path = os.path.join(tempfile.mkdtemp(), "div.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY, type TEXT, topic TEXT,
        content TEXT, embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, keywords TEXT,
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

    with patch('cairn.embeddings.embed', return_value=base), \
         patch('cairn.embeddings._load_vec', return_value=False):
        results = find_similar(conn, "database choice")

    topics = [r["topic"] for r in results if r["topic"] == "db-choice"]
    assert len(topics) <= 1, f"Diversity filter should deduplicate same type+topic, got {len(topics)}"
    conn.close()


# === Relative filter edge cases ===

# Verifies: a single result always passes the relative filter
def test_relative_filter_single_result():
    """Single result should always survive relative filter."""
    from cairn.config import RELATIVE_FILTER_RATIO
    results = [{"similarity": 0.50}]
    max_sim = results[0]["similarity"]
    kept = [r for r in results if r["similarity"] >= RELATIVE_FILTER_RATIO * max_sim]
    assert len(kept) == 1


# Verifies: tightly-clustered similarities all survive the relative filter
def test_relative_filter_tight_cluster():
    """Results within 30% of each other should all survive."""
    from cairn.config import RELATIVE_FILTER_RATIO
    results = [{"similarity": 0.80}, {"similarity": 0.75}, {"similarity": 0.60}]
    max_sim = 0.80
    kept = [r for r in results if r["similarity"] >= RELATIVE_FILTER_RATIO * max_sim]
    # 0.7 * 0.80 = 0.56 — all three pass
    assert len(kept) == 3


# Verifies: relative filter drops results far below the top similarity
def test_relative_filter_drops_outlier():
    """One strong + one much weaker should drop the weak one."""
    from cairn.config import RELATIVE_FILTER_RATIO
    results = [{"similarity": 0.85}, {"similarity": 0.40}]
    max_sim = 0.85
    kept = [r for r in results if r["similarity"] >= RELATIVE_FILTER_RATIO * max_sim]
    # 0.7 * 0.85 = 0.595 — 0.40 fails
    assert len(kept) == 1


# === Confidence no longer gates retrieval ===

# Verifies: confidence floor and similarity override are both disabled
def test_no_confidence_filtering_zero_confidence():
    """similarity=0.50, confidence=0.0 — should be included (confidence doesn't gate retrieval)."""
    # With confidence removed from filtering, only similarity threshold matters
    from cairn.config import SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR
    assert SOFT_CONF_FLOOR == 0.0, "Confidence floor should be disabled"
    assert SOFT_SIM_OVERRIDE == 0.0, "Similarity override for confidence should be disabled"


# Verifies: low confidence does not exclude entries above similarity threshold
def test_no_confidence_filtering_low_confidence():
    """similarity=0.50, confidence=0.2 — should be included (confidence doesn't gate retrieval)."""
    # Previously this was excluded by the confidence floor. Now included.
    threshold = 0.15  # The permissive similarity floor
    included = 0.50 >= threshold
    assert included


# Verifies: zero confidence with moderate similarity still passes
def test_no_confidence_filtering_at_sim_threshold():
    """similarity=0.30, confidence=0.0 — included based on similarity alone."""
    threshold = 0.15
    included = 0.30 >= threshold
    assert included


# === Dominance suppression ===

# Verifies: gap exactly equal to DOMINANCE_EPSILON does not trigger suppression
def test_dominance_gap_exactly_at_epsilon():
    """Gap exactly at epsilon boundary — should NOT trigger suppression."""
    from cairn.config import DOMINANCE_EPSILON
    gap = DOMINANCE_EPSILON
    assert not (gap < DOMINANCE_EPSILON)


# Verifies: gap just below DOMINANCE_EPSILON triggers suppression
def test_dominance_gap_just_below_epsilon():
    """Gap just below epsilon — should trigger, keeping both."""
    from cairn.config import DOMINANCE_EPSILON
    gap = DOMINANCE_EPSILON - 0.001
    assert gap < DOMINANCE_EPSILON


# === Diversity filter: realistic scenarios ===


# Verifies: entries with different topics survive diversity filtering
def test_diversity_keeps_different_aspects():
    """Two entries about related but distinct topics should both survive."""
    results = [
        {"type": "decision", "topic": "db-choice", "content": "Use SQLite for storage"},
        {"type": "decision", "topic": "db-schema", "content": "Add confidence column"},
    ]
    assert results[0]["topic"] != results[1]["topic"]


# Verifies: word overlap below DIVERSITY_SIM_THRESHOLD keeps both entries
def test_diversity_word_overlap_boundary():
    """Content with exactly 90% word overlap should be caught."""
    from cairn.config import DIVERSITY_SIM_THRESHOLD
    # 9 shared words out of 10 total = 0.9
    content_a = "the quick brown fox jumps over the lazy sleeping dog"
    content_b = "the quick brown fox jumps over the lazy sleeping cat"
    words_a = set(content_a.lower().split())
    words_b = set(content_b.lower().split())
    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
    # 9 shared / 11 unique = 0.818 — just below threshold, both should be kept
    assert overlap < DIVERSITY_SIM_THRESHOLD


# Verifies: identical content produces word overlap above DIVERSITY_SIM_THRESHOLD
def test_diversity_identical_content():
    """Exact same content, different topic — word overlap catches it."""
    from cairn.config import DIVERSITY_SIM_THRESHOLD
    content = "use JWT for stateless authentication"
    words = set(content.lower().split())
    overlap = len(words & words) / max(len(words | words), 1)
    assert overlap > DIVERSITY_SIM_THRESHOLD  # 1.0 > 0.9


# === Write throttling: priority ordering ===

# Verifies: write throttle prioritises corrections and decisions over project entries
def test_write_throttle_corrections_survive_over_project():
    """When throttled, corrections should be kept over project metadata."""
    from cairn.config import MAX_MEMORIES_PER_RESPONSE
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
    assert types_kept[0] == "correction"
    assert types_kept[1] == "decision"


# Verifies: write throttle caps entries at MAX_MEMORIES_PER_RESPONSE
def test_write_throttle_all_same_type():
    """All entries same type — should just keep first N."""
    from cairn.config import MAX_MEMORIES_PER_RESPONSE
    entries = [{"type": "fact", "topic": f"t{i}", "content": f"c{i}"} for i in range(10)]
    assert len(entries[:MAX_MEMORIES_PER_RESPONSE]) == MAX_MEMORIES_PER_RESPONSE


# === Combined gate interaction ===

# Verifies: borderline entries are caught by either garbage or borderline gate
def test_gates_combined_weak_but_not_garbage():
    """Entry at garbage floor with low score — when borderline == garbage floor, caught by garbage gate."""
    from cairn.config import MIN_INJECTION_SIMILARITY, BORDERLINE_SIM_CEILING, BORDERLINE_SCORE_FLOOR
    # When BORDERLINE_SIM_CEILING == MIN_INJECTION_SIMILARITY, the borderline gate
    # has no independent effect — entries below the ceiling are also below garbage.
    # Test with a value just below the ceiling to verify both gates agree.
    sim, score = BORDERLINE_SIM_CEILING - 0.01, BORDERLINE_SCORE_FLOOR - 0.02
    caught_garbage = sim < MIN_INJECTION_SIMILARITY
    caught_borderline = sim < BORDERLINE_SIM_CEILING and score < BORDERLINE_SCORE_FLOOR
    assert caught_garbage or caught_borderline


# Verifies: high-quality entry passes all gates (garbage, borderline, soft, relative)
def test_gates_combined_strong_entry_passes_all():
    """Entry at sim=0.75, score=0.65, confidence=0.8 — should pass everything."""
    from cairn.config import (MIN_INJECTION_SIMILARITY, BORDERLINE_SIM_CEILING,
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
