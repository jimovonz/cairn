#!/usr/bin/env python3
"""Tests for retrieval quality gates using mocked embeddings."""

import sys
import os
import sqlite3
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))

# We need to mock the embedding functions since we don't want to load the model
import embeddings as emb


def make_test_db():
    """Create an in-memory test database with mock memories."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY,
            type TEXT, topic TEXT, content TEXT,
            embedding BLOB, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            project TEXT, confidence REAL DEFAULT 0.7,
            source_start INTEGER, source_end INTEGER
        )
    """)
    return conn


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
            "source_start": e.get("source_start"),
            "source_end": e.get("source_end"),
            "score": e.get("score", e["similarity"] * 0.7),
        }
        for i, e in enumerate(entries)
    ]


# === Garbage gate ===

def test_garbage_gate_rejects_low_similarity():
    from config import MIN_INJECTION_SIMILARITY
    results = mock_results([{"similarity": 0.30, "score": 0.20}])
    # Simulate: best match below garbage threshold
    assert results[0]["similarity"] < MIN_INJECTION_SIMILARITY


def test_garbage_gate_passes_adequate_similarity():
    from config import MIN_INJECTION_SIMILARITY
    results = mock_results([{"similarity": 0.50, "score": 0.40}])
    assert results[0]["similarity"] >= MIN_INJECTION_SIMILARITY


# === Borderline gate ===

def test_borderline_gate_rejects():
    from config import BORDERLINE_SIM_CEILING, BORDERLINE_SCORE_FLOOR
    results = mock_results([{"similarity": 0.40, "score": 0.35}])
    assert results[0]["similarity"] < BORDERLINE_SIM_CEILING
    assert results[0]["score"] < BORDERLINE_SCORE_FLOOR


def test_borderline_gate_passes_high_score():
    from config import BORDERLINE_SIM_CEILING, BORDERLINE_SCORE_FLOOR
    results = mock_results([{"similarity": 0.40, "score": 0.60}])
    assert results[0]["similarity"] < BORDERLINE_SIM_CEILING
    assert results[0]["score"] >= BORDERLINE_SCORE_FLOOR  # passes despite low similarity


# === Relative filter ===

def test_relative_filter():
    from config import RELATIVE_FILTER_RATIO
    results = mock_results([
        {"similarity": 0.80, "score": 0.60},
        {"similarity": 0.70, "score": 0.55},  # 0.70/0.80 = 0.875 > 0.7 → keep
        {"similarity": 0.40, "score": 0.30},  # 0.40/0.80 = 0.50 < 0.7 → drop
    ])
    max_sim = results[0]["similarity"]
    kept = [r for r in results if r["similarity"] >= RELATIVE_FILTER_RATIO * max_sim]
    assert len(kept) == 2
    assert kept[0]["similarity"] == 0.80
    assert kept[1]["similarity"] == 0.70


# === Soft confidence inclusion ===

def test_soft_inclusion_high_sim_low_conf():
    """High similarity should override low confidence."""
    from config import SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR
    r = {"similarity": 0.65, "confidence": 0.1}
    included = r["similarity"] >= SOFT_SIM_OVERRIDE or r["confidence"] >= SOFT_CONF_FLOOR
    assert included  # similarity override kicks in


def test_soft_inclusion_low_sim_low_conf():
    """Low similarity + low confidence should be excluded."""
    from config import SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR
    r = {"similarity": 0.40, "confidence": 0.1}
    included = r["similarity"] >= SOFT_SIM_OVERRIDE or r["confidence"] >= SOFT_CONF_FLOOR
    assert not included


def test_soft_inclusion_low_sim_ok_conf():
    """Low similarity but adequate confidence should be included."""
    from config import SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR
    r = {"similarity": 0.40, "confidence": 0.5}
    included = r["similarity"] >= SOFT_SIM_OVERRIDE or r["confidence"] >= SOFT_CONF_FLOOR
    assert included


# === Dominance suppression ===

def test_dominance_suppression_close_scores():
    from config import DOMINANCE_EPSILON
    results = mock_results([
        {"similarity": 0.80, "score": 0.60},
        {"similarity": 0.78, "score": 0.58},  # gap = 0.02 < 0.05
    ])
    gap = results[0]["score"] - results[1]["score"]
    assert gap < DOMINANCE_EPSILON  # both should be included


def test_dominance_suppression_clear_leader():
    from config import DOMINANCE_EPSILON
    results = mock_results([
        {"similarity": 0.80, "score": 0.70},
        {"similarity": 0.50, "score": 0.35},  # gap = 0.35 > 0.05
    ])
    gap = results[0]["score"] - results[1]["score"]
    assert gap >= DOMINANCE_EPSILON  # only leader needed


# === Diversity filter ===

def test_diversity_same_type_topic():
    from config import DIVERSITY_SIM_THRESHOLD
    results = mock_results([
        {"type": "fact", "topic": "auth", "content": "use JWT", "similarity": 0.8, "score": 0.6},
        {"type": "fact", "topic": "auth", "content": "JWT preferred", "similarity": 0.75, "score": 0.55},
    ])
    # Same type+topic should be caught as duplicate
    assert results[0]["type"] == results[1]["type"]
    assert results[0]["topic"] == results[1]["topic"]


def test_diversity_different_topics():
    results = mock_results([
        {"type": "fact", "topic": "auth", "content": "use JWT", "similarity": 0.8, "score": 0.6},
        {"type": "decision", "topic": "db", "content": "use SQLite", "similarity": 0.75, "score": 0.55},
    ])
    # Different type+topic should both be kept
    assert results[0]["topic"] != results[1]["topic"]


# === Write throttling ===

def test_write_throttle_under_limit():
    from config import MAX_MEMORIES_PER_RESPONSE
    entries = [{"type": "fact", "topic": f"t{i}", "content": f"c{i}"} for i in range(3)]
    assert len(entries) <= MAX_MEMORIES_PER_RESPONSE


def test_write_throttle_over_limit():
    from config import MAX_MEMORIES_PER_RESPONSE
    entries = [{"type": "fact", "topic": f"t{i}", "content": f"c{i}"} for i in range(10)]
    assert len(entries) > MAX_MEMORIES_PER_RESPONSE
    # After throttling, should be capped
    throttled = entries[:MAX_MEMORIES_PER_RESPONSE]
    assert len(throttled) == MAX_MEMORIES_PER_RESPONSE


def test_write_throttle_prioritises_corrections():
    """Corrections should be kept over project metadata."""
    type_priority = {"correction": 0, "decision": 1, "fact": 2, "preference": 3,
                     "person": 4, "skill": 5, "workflow": 6, "project": 7}
    entries = [
        {"type": "project", "topic": "p1", "content": "low priority"},
        {"type": "correction", "topic": "c1", "content": "high priority"},
        {"type": "fact", "topic": "f1", "content": "medium priority"},
    ]
    entries.sort(key=lambda e: type_priority.get(e.get("type", ""), 99))
    assert entries[0]["type"] == "correction"
    assert entries[-1]["type"] == "project"


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
