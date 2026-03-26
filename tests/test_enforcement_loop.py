#!/usr/bin/env python3
"""Tests for the full enforcement loop — the core two-pass mechanism where:
1. First call to main() blocks (missing block / incomplete / needs context)
2. System re-prompts the LLM
3. Second call to main() completes

Also covers the branches in stop_hook.py that have zero coverage:
- Continuation cap hit
- Context retrieval → weak entry suppression
- Low-info pre-filter
- Malformed block with open but no close tag
- Layer 2 keyword search through main()
- Retrieval outcome recording
- Write throttle through main()
"""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


HOOK_STATE_SQL = """CREATE TABLE IF NOT EXISTS hook_state (
    session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, key))"""


def fresh_env():
    _counter[0] += 1
    n = _counter[0]
    db_path = os.path.join(TEST_DIR, f"loop_{n}.db")
    conn = sqlite3.connect(db_path)
    for sql in [
        """CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
            embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
            source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE memory_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER, content TEXT, session_id TEXT,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
            parent_session_id TEXT, project TEXT, transcript_path TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT, session_id TEXT, detail TEXT, value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
        HOOK_STATE_SQL,
        """CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
            INSERT INTO memory_history (memory_id, content, session_id, changed_at)
            VALUES (old.id, old.content, old.session_id, old.updated_at); END""",
        """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            topic, content, content=memories, content_rowid=id)""",
        """CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content); END""",
    ]:
        conn.execute(sql)
    conn.commit()
    return db_path, conn


def run_hook(db_path, payload):
    """Run stop_hook.main() with full patching."""
    import hook_helpers
    import stop_hook
    captured = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    orig = {
        'DB_PATH': hook_helpers.DB_PATH,
        'LOG_PATH': hook_helpers.LOG_PATH,
    }

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, f'loop_{_counter[0]}.log')

        with patch('sys.stdin', StringIO(json.dumps(payload))), \
             patch('sys.stdout', captured), \
             patch('sys.exit', mock_exit), \
             patch.object(hook_helpers, 'get_embedder', return_value=None):
            try:
                stop_hook.main()
            except SystemExit:
                pass
    finally:
        for k, v in orig.items():
            setattr(hook_helpers, k, v)

    output = captured.getvalue()
    result = json.loads(output) if output.strip() else None
    return result, exit_code[0]


# ============================================================
# TWO-PASS ENFORCEMENT: missing block → re-prompt → complete
# ============================================================

def test_two_pass_missing_block_then_complete():
    """Pass 1: no memory block → blocks. Pass 2 (continuation): has block → allows stop."""
    db_path, conn = fresh_env()

    # Mark session as instructed (has prior hook activity) so enforcement kicks in
    conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', 's1')")
    conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', 's1')")
    conn.commit()

    # Pass 1: no block
    r1, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s1", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "Answer without memory block."
    })
    assert r1 is not None
    assert r1["decision"] == "block"

    # Pass 2: continuation with block
    r2, _ = run_hook(db_path, {
        "stop_hook_active": True,
        "session_id": "s1", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "Fixed.\n<memory>\n- type: fact\n- topic: fixed\n- content: response now includes a proper memory block\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    })
    assert r2 is None  # Allowed to stop

    # Memory should be stored
    row = conn.execute("SELECT content FROM memories WHERE topic = 'fixed'").fetchone()
    assert row is not None
    conn.close()


def test_two_pass_incomplete_then_complete():
    """Pass 1: complete: false → blocks. Pass 2: complete: true → allows stop."""
    db_path, conn = fresh_env()

    # Pass 1: incomplete
    r1, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s1", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "Working.\n<memory>\n- type: fact\n- topic: partial\n- content: half done\n- complete: false\n- remaining: finish the rest\n</memory>"
    })
    assert r1 is not None
    assert r1["decision"] == "block"

    # Pass 2: complete
    r2, _ = run_hook(db_path, {
        "stop_hook_active": True,
        "session_id": "s1", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "Done.\n<memory>\n- type: fact\n- topic: finished\n- content: all done\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    })
    assert r2 is None  # Allowed to stop
    conn.close()


# ============================================================
# CONTINUATION CAP — 3 consecutive blocks → forced stop
# ============================================================

def test_continuation_cap_forces_stop():
    """After MAX_CONTINUATIONS blocks, the hook forces a stop regardless."""
    from config import MAX_CONTINUATIONS
    db_path, conn = fresh_env()

    # Burn through the cap with incomplete responses
    for i in range(MAX_CONTINUATIONS):
        run_hook(db_path, {
            "stop_hook_active": i > 0,
            "session_id": "s-cap", "transcript_path": "", "cwd": "/tmp",
            "last_assistant_message": f"Still working.\n<memory>\n- complete: false\n- remaining: attempt {i}\n</memory>"
        })

    # Next attempt should be forced to stop
    r, _ = run_hook(db_path, {
        "stop_hook_active": True,
        "session_id": "s-cap", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "Yet another incomplete.\n<memory>\n- complete: false\n- remaining: still going\n</memory>"
    })
    assert r is None, "Should force stop after cap reached"
    conn.close()


# ============================================================
# MALFORMED BLOCK — open tag, no close tag
# ============================================================

def test_malformed_open_no_close():
    """<memory> tag present but no </memory> — should still parse best-effort."""
    db_path, conn = fresh_env()
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-malformed", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "response\n<memory>\n- type: fact\n- topic: unclosed\n- content: no closing tag but should still be parsed correctly\n- complete: true\n- context: sufficient\n- keywords: test"
    })

    # Parser should recover via unclosed tag fallback
    row = conn.execute("SELECT content FROM memories WHERE topic = 'unclosed'").fetchone()
    assert row is not None, "Should parse memory from unclosed block"
    conn.close()


# ============================================================
# LOW-INFO PRE-FILTER — short/generic context_need skipped
# ============================================================

def test_low_info_context_need_skipped():
    """context_need='help' should be pre-filtered — no retrieval attempted."""
    db_path, conn = fresh_env()
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-lowinfo", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "need help\n<memory>\n- context: insufficient\n- keywords: test\n- context_need: help\n- complete: true\n</memory>"
    })

    # Should allow stop (pre-filtered, no retrieval)
    assert r is None
    # Check metric was recorded
    event = conn.execute("SELECT event FROM metrics WHERE event = 'context_prefiltered'").fetchone()
    assert event is not None, "Pre-filter metric should be recorded"
    conn.close()


def test_substantive_context_need_not_filtered():
    """Real context_need should not be pre-filtered (even if retrieval returns nothing)."""
    db_path, conn = fresh_env()
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-real", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "need context\n<memory>\n- context: insufficient\n- keywords: test\n- context_need: what database architecture decisions were made\n- complete: true\n</memory>"
    })

    # Should allow stop (no data found, but retrieval was attempted)
    # Check that context_requested metric exists (not prefiltered)
    event = conn.execute("SELECT event FROM metrics WHERE event = 'context_requested'").fetchone()
    assert event is not None, "Substantive query should reach retrieval, not be pre-filtered"
    conn.close()


# ============================================================
# RETRIEVAL OUTCOME — recorded in metrics
# ============================================================

def test_retrieval_outcome_recorded():
    """retrieval_outcome: useful should be recorded as a metric."""
    db_path, conn = fresh_env()
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-outcome", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "Got it.\n<memory>\n- retrieval_outcome: useful\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    })

    event = conn.execute("SELECT event FROM metrics WHERE event = 'retrieval_useful'").fetchone()
    assert event is not None, "retrieval_outcome: useful should be recorded"
    conn.close()


def test_retrieval_outcome_harmful_recorded():
    db_path, conn = fresh_env()
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-harmful", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "Bad context.\n<memory>\n- retrieval_outcome: harmful\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    })

    event = conn.execute("SELECT event FROM metrics WHERE event = 'retrieval_harmful'").fetchone()
    assert event is not None
    conn.close()


# ============================================================
# KEYWORDS — Layer 2 staging triggered through main()
# ============================================================

def test_keywords_parsed_and_logged():
    """Keywords in memory block should be parsed (Layer 2 runs but finds nothing)."""
    db_path, conn = fresh_env()
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-kw", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "Done.\n<memory>\n- type: fact\n- topic: kw-test\n- content: keyword test for cross-project search validation\n- keywords: authentication, JWT, tokens\n- complete: true\n- context: sufficient\n</memory>"
    })

    # Memory should be stored
    row = conn.execute("SELECT content FROM memories WHERE topic = 'kw-test'").fetchone()
    assert row is not None
    conn.close()


# ============================================================
# WRITE THROTTLE — too many entries through main()
# ============================================================

def test_write_throttle_through_main():
    """8 entries in one block — should be capped at MAX_MEMORIES_PER_RESPONSE."""
    from config import MAX_MEMORIES_PER_RESPONSE
    db_path, conn = fresh_env()

    entries = ""
    for i in range(8):
        entries += f"\n- type: fact\n- topic: throttle-{i}\n- content: entry {i}"

    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-throttle", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": f"Lots.\n<memory>{entries}\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    })

    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count <= MAX_MEMORIES_PER_RESPONSE, f"Expected max {MAX_MEMORIES_PER_RESPONSE}, got {count}"
    conn.close()


# ============================================================
# CONFIDENCE UPDATES — through main()
# ============================================================

def test_confidence_update_through_main():
    """Confidence updates in memory block should modify DB entries."""
    db_path, conn = fresh_env()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s-conf', 'P')")
    conn.execute("INSERT INTO memories (type, topic, content, confidence, session_id) VALUES (?,?,?,?,?)",
                 ("fact", "target", "update target", 0.7, "s-conf"))
    conn.commit()
    mem_id = conn.execute("SELECT id FROM memories").fetchone()[0]

    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-conf", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": f"Context was helpful.\n<memory>\n- confidence_update: {mem_id}:+\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    })

    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = ?", (mem_id,)).fetchone()[0]
    assert new_conf > 0.7, f"Expected confidence boost, got {new_conf}"
    conn.close()


def test_confidence_penalty_through_main():
    db_path, conn = fresh_env()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s-pen', 'P')")
    conn.execute("INSERT INTO memories (type, topic, content, confidence, session_id) VALUES (?,?,?,?,?)",
                 ("fact", "target", "penalty target", 0.7, "s-pen"))
    conn.commit()
    mem_id = conn.execute("SELECT id FROM memories").fetchone()[0]

    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-pen", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": f"Wrong info.\n<memory>\n- confidence_update: {mem_id}:-\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    })

    new_conf = conn.execute("SELECT confidence FROM memories WHERE id = ?", (mem_id,)).fetchone()[0]
    assert new_conf < 0.7, f"Expected confidence penalty, got {new_conf}"
    conn.close()


# ============================================================
# SOURCE MESSAGES — stored through main()
# ============================================================

def test_depth_stored_through_main():
    db_path, conn = fresh_env()
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-src", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "answer\n<memory>\n- type: fact\n- topic: sourced\n- content: has depth for context recovery via mechanical anchor\n- depth: 4\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    })

    row = conn.execute("SELECT depth FROM memories WHERE topic = 'sourced'").fetchone()
    assert row is not None
    assert row[0] == 4
    conn.close()


# ============================================================
# EMPTY last_assistant_message — should allow stop
# ============================================================

def test_empty_message_allows_stop():
    db_path, conn = fresh_env()
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-empty", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": ""
    })
    assert r is None, "Empty message should allow stop"
    conn.close()


# ============================================================
# CONTEXT CACHE — second identical request skipped
# ============================================================

def test_context_cache_prevents_second_retrieval():
    """Pre-populate cache for a session, then verify second request is a cache hit."""
    import stop_hook
    db_path, conn = fresh_env()

    # Pre-populate the cache as if a previous retrieval succeeded
    cache_data = [{"text": "what architecture decisions exist"}]
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'context_cache', ?)",
        ("s-cache", json.dumps(cache_data))
    )
    conn.commit()

    # Now make a request with the same context_need — should hit cache
    r, _ = run_hook(db_path, {
        "stop_hook_active": False,
        "session_id": "s-cache", "transcript_path": "", "cwd": "/tmp",
        "last_assistant_message": "need info\n<memory>\n- context: insufficient\n- keywords: test\n- context_need: what architecture decisions exist\n- complete: true\n</memory>"
    })

    # Should allow stop (cache hit, no re-retrieval)
    assert r is None, f"Expected cache hit to allow stop, got: {r}"
    hits = conn.execute("SELECT COUNT(*) FROM metrics WHERE event = 'context_cache_hit'").fetchone()[0]
    assert hits >= 1, "Should record context_cache_hit metric"
    conn.close()


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
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
    cleanup()
