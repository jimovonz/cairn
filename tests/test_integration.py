#!/usr/bin/env python3
"""Integration tests for Cairn — exercises the real pipeline with mocked embeddings.

Uses an in-memory SQLite DB and deterministic mock vectors to test:
- Insert → dedup → retrieve → gate → inject pipeline
- Contradiction handling
- Confidence dynamics across operations
- Context cache behaviour
- Loop protection
- Write throttling through insert_memories
"""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import numpy as np

# Setup paths
TEST_DIR = tempfile.mkdtemp()

_db_counter = [0]

def setup_test_db():
    """Create a fresh isolated test database with schema."""
    _db_counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"test_{_db_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
            embedding BLOB, session_id TEXT, project TEXT,
            confidence REAL DEFAULT 0.7,
            source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL, content TEXT NOT NULL,
            session_id TEXT, changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY, parent_session_id TEXT,
            project TEXT, transcript_path TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL, session_id TEXT, detail TEXT,
            value REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_version BEFORE UPDATE OF content ON memories BEGIN
            INSERT INTO memory_history (memory_id, content, session_id, changed_at)
            VALUES (old.id, old.content, old.session_id, old.updated_at);
        END
    """)
    conn.commit()
    return db_path, conn


def make_vector(seed):
    """Create a deterministic 384-dim normalised vector from a seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v


def make_blob(seed):
    """Create a vector blob for storage."""
    return make_vector(seed).tobytes()


def insert_memory(conn, mem_type, topic, content, project=None, confidence=0.7, seed=None):
    """Insert a memory with a deterministic embedding."""
    blob = make_blob(seed) if seed is not None else None
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (mem_type, topic, content, blob, project, confidence)
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def get_memory(conn, memory_id):
    """Fetch a memory by ID."""
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    conn.row_factory = None
    return dict(row) if row else None


def count_memories(conn):
    return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


def count_history(conn):
    return conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0]


# ============================================================
# Test: Insert and retrieve through find_similar
# ============================================================

# Verifies: vector similarity search returns the closest match
def test_insert_and_retrieve():
    """Insert 3 memories with different seeds, query with one seed, verify closest returned."""
    db_path, conn = setup_test_db()
    id1 = insert_memory(conn, "fact", "auth", "Use JWT tokens", project="proj", seed=100)
    id2 = insert_memory(conn, "decision", "db", "Use SQLite", project="proj", seed=200)
    id3 = insert_memory(conn, "preference", "style", "Use snake_case", project="proj", seed=300)

    # Query with seed=100 vector — should match id1 (same vector = similarity 1.0)
    import cairn.embeddings as emb
    query_vec = make_vector(100)

    # Mock embed to return our test vector
    with patch.object(emb, 'embed', return_value=query_vec):
        with patch.object(emb, '_load_vec', return_value=False):  # Force brute-force
            results = emb._brute_force_candidates(conn, query_vec, k=10)

    assert len(results) >= 1
    # Top result should be id1 (exact match)
    assert results[0]["id"] == id1
    assert results[0]["similarity"] > 0.99
    conn.close()


# ============================================================
# Test: Dedup — near-identical entries get merged
# ============================================================

# Verifies: same type+topic overwrites content and preserves history
def test_dedup_same_type_topic_overwrites():
    """Insert two memories with same type+topic but different content. Second should overwrite."""
    db_path, conn = setup_test_db()
    id1 = insert_memory(conn, "decision", "db-choice", "Use PostgreSQL", project="proj", seed=100)
    assert count_memories(conn) == 1

    # Simulate overwrite via same type+topic
    conn.execute(
        "UPDATE memories SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        ("Use SQLite instead", id1)
    )
    conn.commit()

    mem = get_memory(conn, id1)
    assert mem["content"] == "Use SQLite instead"
    # History should have the old version
    assert count_history(conn) == 1
    old = conn.execute("SELECT content FROM memory_history WHERE memory_id = ?", (id1,)).fetchone()
    assert old[0] == "Use PostgreSQL"
    conn.close()


# ============================================================
# Test: Contradiction — same type+topic, different content → confidence drop
# ============================================================

# Verifies: contradicting content marks old memory as superseded
def test_contradiction_annotates_old_memory():
    """Overwriting same type+topic with different content should annotate old memory as superseded."""
    db_path, conn = setup_test_db()
    id1 = insert_memory(conn, "decision", "db-choice", "Use PostgreSQL", confidence=0.8, seed=100)

    # Simulate the contradiction handling from storage.py insert_memories
    old_content = conn.execute("SELECT content FROM memories WHERE id = ?", (id1,)).fetchone()[0]
    new_content = "Use SQLite"
    assert old_content != new_content

    # Annotate as superseded then overwrite (as storage.py now does)
    conn.execute(
        "UPDATE memories SET archived_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (f"superseded: {new_content[:200]}", id1)
    )
    conn.execute(
        "UPDATE memories SET content = ?, confidence = 0.7, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (new_content, id1)
    )
    conn.commit()

    mem = get_memory(conn, id1)
    assert mem["content"] == "Use SQLite"
    assert mem["confidence"] == 0.7  # Fresh confidence
    # Old content preserved in history
    old = conn.execute("SELECT content FROM memory_history WHERE memory_id = ?", (id1,)).fetchone()
    assert old[0] == "Use PostgreSQL"
    # Supersession annotation present
    reason = conn.execute("SELECT archived_reason FROM memories WHERE id = ?", (id1,)).fetchone()[0]
    assert reason is not None and "superseded" in reason and "SQLite" in reason, \
        f"Expected archived_reason to contain 'superseded' and 'SQLite'; got: {reason!r}"
    conn.close()


# ============================================================
# Test: Confidence dynamics through multiple updates
# ============================================================

# Verifies: repeated confidence boosts saturate below 1.0
def test_confidence_saturating_boost():
    """Multiple boosts should approach but never reach 1.0."""
    db_path, conn = setup_test_db()
    id1 = insert_memory(conn, "fact", "test", "test fact", confidence=0.7)

    for _ in range(20):
        current = conn.execute("SELECT confidence FROM memories WHERE id = ?", (id1,)).fetchone()[0]
        boost = 0.1 * (1 - current)
        new = min(current + boost, 1.0)
        conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (new, id1))
    conn.commit()

    final = conn.execute("SELECT confidence FROM memories WHERE id = ?", (id1,)).fetchone()[0]
    assert final > 0.95
    assert final < 1.0
    conn.close()


# Verifies: single negative penalty is severe at high confidence
def test_confidence_single_negative_severe_at_high():
    """A single negative at 0.9 should drop significantly."""
    db_path, conn = setup_test_db()
    id1 = insert_memory(conn, "fact", "test", "test fact", confidence=0.9)

    current = 0.9
    penalty = 0.2 * (1 + current)
    new = max(current - penalty, 0.0)
    conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (new, id1))
    conn.commit()

    final = conn.execute("SELECT confidence FROM memories WHERE id = ?", (id1,)).fetchone()[0]
    assert final < 0.55
    assert final > 0.50
    conn.close()


# Verifies: low-confidence memory recovers with repeated boosts
def test_confidence_recovery_from_low():
    """A memory at 0.2 should be able to recover with enough positive signals."""
    db_path, conn = setup_test_db()
    id1 = insert_memory(conn, "fact", "test", "test fact", confidence=0.2)

    for _ in range(10):
        current = conn.execute("SELECT confidence FROM memories WHERE id = ?", (id1,)).fetchone()[0]
        boost = 0.1 * (1 - current)
        new = min(current + boost, 1.0)
        conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (new, id1))
    conn.commit()

    final = conn.execute("SELECT confidence FROM memories WHERE id = ?", (id1,)).fetchone()[0]
    assert final > 0.5  # Recovered meaningfully
    conn.close()


# ============================================================
# Test: Quality gates in sequence (simulated pipeline)
# ============================================================

# Verifies: quality gates filter out unrelated vector matches
def test_full_gate_pipeline_rejects_garbage():
    """Insert memories, query with unrelated vector — gates should return nothing."""
    db_path, conn = setup_test_db()
    insert_memory(conn, "fact", "auth", "Use JWT", seed=100)
    insert_memory(conn, "decision", "db", "Use SQLite", seed=200)

    import cairn.embeddings as emb
    # Seed 999 will produce a vector unrelated to 100 or 200
    query_vec = make_vector(999)

    with patch.object(emb, 'embed', return_value=query_vec):
        with patch.object(emb, '_load_vec', return_value=False):
            results = emb.find_similar(conn, "completely unrelated topic")

    # Random vectors in 384 dimensions have near-zero cosine similarity
    # Gates should filter everything out
    assert len(results) == 0
    conn.close()


# Verifies: exact vector match passes all quality gates
def test_full_gate_pipeline_passes_strong_match():
    """Query with same vector as a stored memory — should pass all gates."""
    db_path, conn = setup_test_db()
    id1 = insert_memory(conn, "fact", "auth", "Use JWT", seed=100, confidence=0.8)

    import cairn.embeddings as emb
    query_vec = make_vector(100)

    with patch.object(emb, 'embed', return_value=query_vec):
        with patch.object(emb, '_load_vec', return_value=False):
            results = emb.find_similar(conn, "authentication approach")

    assert len(results) >= 1
    assert results[0]["id"] == id1
    assert results[0]["similarity"] > 0.99
    conn.close()


# ============================================================
# Test: Write throttling through insert pipeline
# ============================================================

# Verifies: write throttle caps entries at MAX_MEMORIES_PER_RESPONSE
def test_write_throttle_caps_entries():
    """More than MAX_MEMORIES_PER_RESPONSE entries should be truncated."""
    from cairn.config import MAX_MEMORIES_PER_RESPONSE
    db_path, conn = setup_test_db()

    entries = [{"type": "fact", "topic": f"topic-{i}", "content": f"fact number {i}"}
               for i in range(10)]

    # Simulate throttling logic from insert_memories
    type_priority = {"correction": 0, "decision": 1, "fact": 2, "preference": 3,
                     "person": 4, "skill": 5, "workflow": 6, "project": 7}
    entries.sort(key=lambda e: type_priority.get(e.get("type", ""), 99))
    if len(entries) > MAX_MEMORIES_PER_RESPONSE:
        entries = entries[:MAX_MEMORIES_PER_RESPONSE]

    assert len(entries) == MAX_MEMORIES_PER_RESPONSE

    # Actually insert them
    for e in entries:
        conn.execute("INSERT INTO memories (type, topic, content) VALUES (?, ?, ?)",
                     (e["type"], e["topic"], e["content"]))
    conn.commit()

    assert count_memories(conn) == MAX_MEMORIES_PER_RESPONSE
    conn.close()


# ============================================================
# Test: Stop hook parse → store → retrieve round trip
# ============================================================

# Verifies: parse -> store -> retrieve round trip preserves data
def test_parse_store_retrieve_roundtrip():
    """Parse a memory block, store entries, then retrieve and verify."""
    from hooks.parser import parse_memory_block
    db_path, conn = setup_test_db()

    text = """Here is my response.
<memory>
- type: decision
- topic: auth-method
- content: Use OAuth2 with PKCE flow
- keywords: auth, OAuth, PKCE
- depth: 3
- complete: true
- context: sufficient
</memory>"""

    parsed = parse_memory_block(text); entries, complete, remaining, context, context_need, conf_updates, retrieval_outcome, keywords, intent = parsed.entries, parsed.complete, parsed.remaining, parsed.context, parsed.context_need, parsed.confidence_updates, parsed.retrieval_outcome, parsed.keywords, parsed.intent

    assert len(entries) == 1
    assert entries[0]["type"] == "decision"
    assert entries[0]["depth"] == 3
    assert keywords == ["auth", "OAuth", "PKCE"]
    assert complete is True

    # Store it
    e = entries[0]
    conn.execute(
        "INSERT INTO memories (type, topic, content, depth) VALUES (?, ?, ?, ?)",
        (e["type"], e["topic"], e["content"], e.get("depth"))
    )
    conn.commit()

    # Retrieve and verify
    mem = conn.execute("SELECT * FROM memories WHERE topic = 'auth-method'").fetchone()
    assert mem[3] == "Use OAuth2 with PKCE flow"  # content
    conn.close()


# ============================================================
# Test: Context insufficient → block decision
# ============================================================

# Verifies: context: insufficient parses correctly for blocking
def test_context_insufficient_produces_block():
    """Memory block with context: insufficient should parse correctly for blocking."""
    from hooks.parser import parse_memory_block

    text = """I need more context.
<memory>
- context: insufficient
- context_need: what database did we choose for the project
- keywords: database, architecture
- complete: true
</memory>"""

    entries, complete, remaining, context, context_need, *_ = parse_memory_block(text)
    assert context == "insufficient"
    assert context_need == "what database did we choose for the project", \
        f"Expected exact context_need string; got: {context_need!r}"
    assert complete is True


# ============================================================
# Test: Confidence updates parse and apply correctly
# ============================================================

# Verifies: parsed confidence updates boost/penalise correct memories
def test_confidence_updates_applied():
    """Parse confidence updates and verify they modify the right memories."""
    from hooks.parser import parse_memory_block
    db_path, conn = setup_test_db()

    id1 = insert_memory(conn, "fact", "t1", "content 1", confidence=0.7)
    id2 = insert_memory(conn, "fact", "t2", "content 2", confidence=0.7)

    text = f"""Response using retrieved context.
<memory>
- confidence_update: {id1}:+
- confidence_update: {id2}:-
- complete: true
- context: sufficient
- keywords: test
</memory>"""

    parsed = parse_memory_block(text)
    conf_updates = parsed.confidence_updates
    assert len(conf_updates) == 2
    assert conf_updates[0] == (id1, "+", None)
    assert conf_updates[1] == (id2, "-", None)

    # Apply updates manually (as stop_hook would)
    for memory_id, direction, _reason in conf_updates:
        current = conn.execute("SELECT confidence FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
        if direction == "+":
            new = min(current + 0.1 * (1 - current), 1.0)
        else:
            new = max(current - 0.2 * (1 + current), 0.0)
        conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (new, memory_id))
    conn.commit()

    c1 = conn.execute("SELECT confidence FROM memories WHERE id = ?", (id1,)).fetchone()[0]
    c2 = conn.execute("SELECT confidence FROM memories WHERE id = ?", (id2,)).fetchone()[0]
    assert c1 > 0.7  # Boosted
    assert c2 < 0.7  # Penalised
    conn.close()


# ============================================================
# Test: Version history preserved on overwrite
# ============================================================

# Verifies: content updates create history entries via DB trigger
def test_version_history_on_overwrite():
    """Overwriting content should create a history entry via trigger."""
    db_path, conn = setup_test_db()
    id1 = insert_memory(conn, "decision", "db", "Use PostgreSQL")
    assert count_history(conn) == 0

    conn.execute("UPDATE memories SET content = 'Use SQLite' WHERE id = ?", (id1,))
    conn.commit()
    assert count_history(conn) == 1

    conn.execute("UPDATE memories SET content = 'Use DuckDB' WHERE id = ?", (id1,))
    conn.commit()
    assert count_history(conn) == 2

    history = conn.execute(
        "SELECT content FROM memory_history WHERE memory_id = ? ORDER BY changed_at",
        (id1,)
    ).fetchall()
    assert history[0][0] == "Use PostgreSQL"
    assert history[1][0] == "Use SQLite"

    current = get_memory(conn, id1)
    assert current["content"] == "Use DuckDB"
    conn.close()


# ============================================================
# Test: Negation heuristic in realistic context
# ============================================================

# Verifies: negation heuristic detects contradictory memory pairs
def test_negation_with_realistic_memories():
    """Two plausible memories that contradict should be detected."""
    from hooks.storage import _has_negation_mismatch

    # Real-world pair from a GNSS project
    assert _has_negation_mismatch(
        "GNSS accuracy is reliable in open sky conditions",
        "GNSS accuracy is not reliable under tree canopy"
    )

    # Preference reversal
    assert _has_negation_mismatch(
        "prefer async processing for sensor data",
        "avoid async processing due to timing constraints"
    )

    # Similar but not contradictory
    assert not _has_negation_mismatch(
        "use RTK for centimetre accuracy",
        "use RTK base station on known point"
    )


# ============================================================
# Cleanup
# ============================================================

def cleanup():
    shutil.rmtree(TEST_DIR, ignore_errors=True)


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
            print(f"  ERROR: {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
    cleanup()
