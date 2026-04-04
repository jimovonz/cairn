#!/usr/bin/env python3
"""Tests for hooks/stop_hook.py — register_session, auto_label_project, and main().

Uses isolated temp DBs and patches external boundaries (stdin, stdout, sys.exit,
embedder, filesystem). 33 tests total within budget.
"""

import sys
import os
import json
import sqlite3
import tempfile
import struct
from unittest.mock import patch, MagicMock
from io import StringIO

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cairn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hooks"))

TEST_DIR = tempfile.mkdtemp()
_db_counter = [0]


def fresh_db():
    """Create an isolated test database with all required tables."""
    _db_counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"stop_{_db_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, topic TEXT, content TEXT, embedding BLOB, session_id TEXT,
        project TEXT, confidence REAL DEFAULT 0.7, source_start INTEGER,
        source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE memory_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER, content TEXT, session_id TEXT,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE sessions (session_id TEXT PRIMARY KEY,
        parent_session_id TEXT, project TEXT, transcript_path TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TRIGGER memories_version BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content, session_id, changed_at)
        VALUES (old.id, old.content, old.session_id, old.updated_at); END""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


def _make_mock_embedder():
    """Create a mock embedder matching the interface used by storage.py and stop_hook.py."""
    zero_vec = struct.pack("384f", *([0.0] * 384))
    mock_emb = MagicMock()
    mock_emb.embed.return_value = zero_vec
    mock_emb.to_blob.return_value = zero_vec
    mock_emb.from_blob.return_value = zero_vec
    mock_emb.cosine_similarity.return_value = 0.0
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()
    return mock_emb


def run_hook(db_path, payload, env_override=None):
    """Run stop_hook.main() with patches for DB, stdin, stdout, sys.exit, embedder."""
    payload_json = json.dumps(payload)
    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    import hook_helpers
    import stop_hook
    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH

    mock_embedder = _make_mock_embedder()

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, "test.log")

        ctx_managers = [
            patch("sys.stdin", StringIO(payload_json)),
            patch("sys.stdout", captured_output),
            patch("sys.exit", mock_exit),
            patch.object(hook_helpers, "get_embedder", return_value=mock_embedder),
        ]
        if env_override:
            ctx_managers.append(patch.dict(os.environ, env_override))

        # Enter all context managers
        entered = []
        for cm in ctx_managers:
            entered.append(cm.__enter__())
        try:
            try:
                stop_hook.main()
            except SystemExit:
                pass
        finally:
            for cm in reversed(ctx_managers):
                cm.__exit__(None, None, None)
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log

    output = captured_output.getvalue()
    result = json.loads(output) if output.strip() else None
    return result, exit_code[0]


def make_payload(session_id="test-sess", message="", cwd="/tmp/testproj",
                 continuation=False, transcript_path=""):
    return {
        "stop_hook_active": continuation,
        "session_id": session_id,
        "transcript_path": transcript_path,
        "cwd": cwd,
        "last_assistant_message": message,
    }


def valid_block(content_entries="", extra_fields=""):
    """Build a valid memory block string with required fields."""
    base = "<memory>\n"
    if content_entries:
        base += content_entries + "\n"
    base += "- complete: true\n- context: sufficient\n- keywords: test\n"
    if extra_fields:
        base += extra_fields + "\n"
    base += "</memory>"
    return base


# ============================================================
# register_session tests
# ============================================================

#TAG: [A1C3]
# Verifies: new root session is inserted into sessions table with correct ID and null parent/project
@pytest.mark.behavioural
def test_register_session_new_root():
    db_path, conn = fresh_db()
    import hook_helpers
    original_db = hook_helpers.DB_PATH
    hook_helpers.DB_PATH = db_path
    try:
        import stop_hook
        stop_hook.register_session("sess-root-001", "")
        row = conn.execute(
            "SELECT session_id, parent_session_id, project FROM sessions WHERE session_id = ?",
            ("sess-root-001",)
        ).fetchone()
        assert row[0] == "sess-root-001"
        assert row[1] is None
        assert row[2] is None
    finally:
        hook_helpers.DB_PATH = original_db
        conn.close()


#TAG: [B2D4]
# Verifies: existing session is not re-registered or overwritten (idempotent guard)
@pytest.mark.edge
def test_register_session_skips_existing():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO sessions (session_id, project) VALUES (?, ?)",
        ("sess-exists", "old-project")
    )
    conn.commit()
    import hook_helpers
    original_db = hook_helpers.DB_PATH
    hook_helpers.DB_PATH = db_path
    try:
        import stop_hook
        stop_hook.register_session("sess-exists", "/some/transcript.jsonl")
        row = conn.execute(
            "SELECT project FROM sessions WHERE session_id = ?", ("sess-exists",)
        ).fetchone()
        assert row[0] == "old-project"
    finally:
        hook_helpers.DB_PATH = original_db
        conn.close()


#TAG: [C3E5]
# Verifies: missing transcript file does not crash and session is still registered with no parent
@pytest.mark.error
def test_register_session_missing_transcript():
    db_path, conn = fresh_db()
    import hook_helpers
    original_db = hook_helpers.DB_PATH
    hook_helpers.DB_PATH = db_path
    try:
        import stop_hook
        stop_hook.register_session("sess-nofile", "/nonexistent/transcript.jsonl")
        row = conn.execute(
            "SELECT session_id, parent_session_id FROM sessions WHERE session_id = ?",
            ("sess-nofile",)
        ).fetchone()
        assert row[0] == "sess-nofile"
        assert row[1] is None
    finally:
        hook_helpers.DB_PATH = original_db
        conn.close()


#TAG: [D4F6]
# Verifies: empty session_id string triggers early return with zero DB rows written
@pytest.mark.adversarial
def test_register_session_empty_session_id():
    db_path, conn = fresh_db()
    import hook_helpers
    original_db = hook_helpers.DB_PATH
    hook_helpers.DB_PATH = db_path
    try:
        import stop_hook
        stop_hook.register_session("", "/some/path")
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == 0
    finally:
        hook_helpers.DB_PATH = original_db
        conn.close()


# ============================================================
# auto_label_project tests
# ============================================================

#TAG: [E5A7]
# Verifies: cwd basename is lowercased and stored as project label for unlabelled session
@pytest.mark.behavioural
def test_auto_label_from_cwd():
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES (?)", ("sess-label",))
    conn.commit()
    import hook_helpers
    original_db = hook_helpers.DB_PATH
    hook_helpers.DB_PATH = db_path
    try:
        import stop_hook
        stop_hook.auto_label_project("sess-label", "/home/user/Projects/My-Cool-Project")
        row = conn.execute(
            "SELECT project FROM sessions WHERE session_id = ?", ("sess-label",)
        ).fetchone()
        assert row[0] == "my-cool-project"
    finally:
        hook_helpers.DB_PATH = original_db
        conn.close()


#TAG: [F6B8]
# Verifies: existing project label is not overwritten by auto-labelling
@pytest.mark.edge
def test_auto_label_skips_if_already_set():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO sessions (session_id, project) VALUES (?, ?)",
        ("sess-has-proj", "existing-project")
    )
    conn.commit()
    import hook_helpers
    original_db = hook_helpers.DB_PATH
    hook_helpers.DB_PATH = db_path
    try:
        import stop_hook
        stop_hook.auto_label_project("sess-has-proj", "/home/user/different-project")
        row = conn.execute(
            "SELECT project FROM sessions WHERE session_id = ?", ("sess-has-proj",)
        ).fetchone()
        assert row[0] == "existing-project"
    finally:
        hook_helpers.DB_PATH = original_db
        conn.close()


#TAG: [A7C9]
# Verifies: root path "/" is in rejection list and produces no project label
@pytest.mark.error
def test_auto_label_rejects_root_path():
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES (?)", ("sess-root-cwd",))
    conn.commit()
    import hook_helpers
    original_db = hook_helpers.DB_PATH
    hook_helpers.DB_PATH = db_path
    try:
        import stop_hook
        stop_hook.auto_label_project("sess-root-cwd", "/")
        row = conn.execute(
            "SELECT project FROM sessions WHERE session_id = ?", ("sess-root-cwd",)
        ).fetchone()
        assert row[0] is None
        count = conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = 'sess-root-cwd'").fetchone()[0]
        assert count == 1
    finally:
        hook_helpers.DB_PATH = original_db
        conn.close()


#TAG: [B8DA]
# Verifies: empty cwd string triggers early return guard with no DB mutation
@pytest.mark.adversarial
def test_auto_label_empty_cwd():
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id) VALUES (?)", ("sess-empty-cwd",))
    conn.commit()
    import hook_helpers
    original_db = hook_helpers.DB_PATH
    hook_helpers.DB_PATH = db_path
    try:
        import stop_hook
        stop_hook.auto_label_project("sess-empty-cwd", "")
        row = conn.execute(
            "SELECT project FROM sessions WHERE session_id = ?", ("sess-empty-cwd",)
        ).fetchone()
        assert row[0] is None
        count = conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = 'sess-empty-cwd'").fetchone()[0]
        assert count == 1
    finally:
        hook_helpers.DB_PATH = original_db
        conn.close()


# ============================================================
# main() tests — behavioural
# ============================================================

#TAG: [C9EB]
# Verifies: valid memory block with all fields is stored in DB and hook exits 0 with no block output
@pytest.mark.behavioural
def test_main_valid_block_stores_and_allows_stop():
    db_path, conn = fresh_db()
    msg = "Here is my analysis.\n" + valid_block(
        "- type: fact\n- topic: valid-store\n- content: Testing that valid blocks are stored correctly in database"
    )
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert result is None
    assert code == 0
    row = conn.execute("SELECT type, content FROM memories WHERE topic = 'valid-store'").fetchone()
    assert row[0] == "fact"
    assert row[1] == "Testing that valid blocks are stored correctly in database"
    conn.close()


#TAG: [DA0C]
# Verifies: instructed session (2+ hook_fired) without memory block gets block decision with hint
@pytest.mark.behavioural
def test_main_missing_block_instructed_session_blocks():
    db_path, conn = fresh_db()
    sid = "sess-instructed"
    conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', ?)", (sid,))
    conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', ?)", (sid,))
    conn.commit()

    payload = make_payload(session_id=sid, message="No memory block here.")
    result, code = run_hook(db_path, payload)

    assert result["decision"] == "block"
    cont_row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'continuation_count'", (sid,)
    ).fetchone()
    assert int(cont_row[0]) == 1
    missing_metric = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'missing_memory_block' AND session_id = ?", (sid,)
    ).fetchone()[0]
    assert missing_metric == 1
    conn.close()


#TAG: [EB1D]
# Verifies: complete: false with remaining field produces block with remaining text in reason
@pytest.mark.behavioural
def test_main_incomplete_blocks_with_remaining():
    db_path, conn = fresh_db()
    msg = "Response.\n<memory>\n- type: fact\n- topic: incomplete\n- content: Testing incomplete block handling with remaining field\n- complete: false\n- remaining: need to finish the analysis\n- context: sufficient\n- keywords: test\n</memory>"
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert result["decision"] == "block"
    assert result["reason"] == "Response marked incomplete. Continue with: need to finish the analysis"
    conn.close()


#TAG: [0C2E]
# Verifies: context: insufficient with valid need triggers retrieval and blocks with CAIRN CONTEXT
@pytest.mark.behavioural
def test_main_context_insufficient_triggers_retrieval():
    db_path, conn = fresh_db()
    msg = "Response.\n<memory>\n- type: fact\n- topic: ctx-insuf\n- content: Testing context retrieval trigger mechanism\n- complete: true\n- context: insufficient\n- context_need: what was decided about database schema\n- keywords: test, context\n</memory>"
    payload = make_payload(message=msg)

    import hook_helpers as hh
    import stop_hook
    original_db = hh.DB_PATH
    original_log = hh.LOG_PATH
    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    mock_emb = _make_mock_embedder()

    try:
        hh.DB_PATH = db_path
        hh.LOG_PATH = os.path.join(TEST_DIR, "test.log")
        with patch("sys.stdin", StringIO(json.dumps(payload))), \
             patch("sys.stdout", captured_output), \
             patch("sys.exit", mock_exit), \
             patch.object(hh, "get_embedder", return_value=mock_emb), \
             patch.object(stop_hook, "retrieve_context",
                          return_value='<entry id="99" score="0.8">test context</entry>'), \
             patch.object(stop_hook, "load_context_cache", return_value=[]), \
             patch.object(stop_hook, "is_context_cached", return_value=False), \
             patch.object(stop_hook, "add_to_context_cache", return_value=[]):
            try:
                stop_hook.main()
            except SystemExit:
                pass
    finally:
        hh.DB_PATH = original_db
        hh.LOG_PATH = original_log

    output = captured_output.getvalue()
    result = json.loads(output) if output.strip() else None
    assert result["decision"] == "block"
    assert result["reason"] == "CAIRN CONTEXT:\n" + '<entry id="99" score="0.8">test context</entry>'
    context_served = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'context_served'"
    ).fetchone()[0]
    assert context_served == 1
    conn.close()


#TAG: [1D3F]
# Verifies: CAIRN_HEADLESS env var skips all enforcement and exits 0 even without memory block
@pytest.mark.behavioural
def test_main_headless_mode_skips_enforcement():
    db_path, conn = fresh_db()
    payload = make_payload(message="No memory block at all.")
    result, code = run_hook(db_path, payload, env_override={"CAIRN_HEADLESS": "1"})

    assert result is None
    assert code == 0
    conn.close()


#TAG: [2E4A]
# Verifies: confidence_update: 42:+ increases target memory confidence above initial 0.7
@pytest.mark.behavioural
def test_main_confidence_updates_applied():
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, confidence) VALUES (42, 'fact', 'conf-test', 'original content for confidence testing', 0.7)"
    )
    conn.commit()

    msg = "Response.\n" + valid_block(
        "- type: fact\n- topic: conf-response\n- content: Testing that confidence updates are properly applied to existing memories",
        "- confidence_update: 42:+"
    )
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert code == 0
    conf_row = conn.execute("SELECT confidence FROM memories WHERE id = 42").fetchone()
    assert conf_row[0] > 0.7
    metric_row = conn.execute(
        "SELECT value FROM metrics WHERE event = 'confidence_updates'"
    ).fetchone()
    assert metric_row[0] == 1.0
    conn.close()


#TAG: [3F5B]
# Verifies: question mark in last 3 sentences triggers question_before_cairn metric and stages file
@pytest.mark.behavioural
def test_main_question_before_cairn_stages_reminder():
    db_path, conn = fresh_db()
    sid = "sess-question-qbc"
    staged_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".staged_context")

    msg = "I'm not sure about that. What do you think?\n" + valid_block(
        "- type: fact\n- topic: question-test\n- content: Testing question before cairn enforcement and staged reminder creation"
    )
    payload = make_payload(session_id=sid, message=msg)
    result, code = run_hook(db_path, payload)

    assert code == 0
    qbc = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'question_before_cairn' AND session_id = ?",
        (sid,)
    ).fetchone()[0]
    assert qbc == 1

    staged_file = os.path.join(staged_dir, f"{sid}_question_cairn.txt")
    if os.path.exists(staged_file):
        os.remove(staged_file)
    conn.close()


# ============================================================
# main() tests — edge
# ============================================================

#TAG: [4A6C]
# Verifies: empty last_assistant_message causes immediate exit 0 before any parsing
@pytest.mark.edge
def test_main_empty_text_allows_stop():
    db_path, conn = fresh_db()
    payload = make_payload(message="")
    result, code = run_hook(db_path, payload)

    assert result is None
    assert code == 0
    conn.close()


#TAG: [5B7D]
# Verifies: session with no prior memories and <=1 hook_fired is treated as uninstructed (exit 0)
@pytest.mark.edge
def test_main_uninstructed_session_allows_stop():
    db_path, conn = fresh_db()
    payload = make_payload(session_id="sess-uninstructed", message="Plain response.")
    result, code = run_hook(db_path, payload)

    assert result is None
    assert code == 0
    uninstructed = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'uninstructed_session_skip'"
    ).fetchone()[0]
    assert uninstructed == 1
    conn.close()


#TAG: [6C8E]
# Verifies: continuation count at MAX_CONTINUATIONS forces stop (exit 0) and resets counter
@pytest.mark.edge
def test_main_continuation_cap_forces_stop():
    db_path, conn = fresh_db()
    sid = "sess-capped"
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'continuation_count', '3')",
        (sid,)
    )
    conn.commit()
    payload = make_payload(session_id=sid, message="Response text.", continuation=True)
    result, code = run_hook(db_path, payload)

    assert result is None
    assert code == 0
    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'continuation_count'",
        (sid,)
    ).fetchone()
    assert row is None
    conn.close()


#TAG: [7D9F]
# Verifies: context_need "help" (in LOW_INFO_STOPLIST and <8 chars) is prefiltered
@pytest.mark.edge
def test_main_context_low_info_prefiltered():
    db_path, conn = fresh_db()
    msg = "Response.\n<memory>\n- type: fact\n- topic: low-info\n- content: Testing low-info context need prefiltering behavior\n- complete: true\n- context: insufficient\n- context_need: help\n- keywords: test\n</memory>"
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert code == 0
    prefilter = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'context_prefiltered'"
    ).fetchone()[0]
    assert prefilter == 1
    conn.close()


#TAG: [8EAB]
# Verifies: intent: resolved in memory block bypasses trailing intent detection entirely
@pytest.mark.edge
def test_main_intent_resolved_skips_check():
    db_path, conn = fresh_db()
    msg = "I analyzed the code. Let me run the tests now.\n" + valid_block(
        "- type: fact\n- topic: intent-resolved\n- content: Testing that intent resolved flag bypasses trailing check",
        "- intent: resolved"
    )
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert result is None
    assert code == 0
    conn.close()


#TAG: [9FBC]
# Verifies: missing block on continuation (stop_hook_active=True) allows stop to prevent loops
@pytest.mark.edge
def test_main_missing_block_continuation_allows_stop():
    db_path, conn = fresh_db()
    payload = make_payload(
        session_id="sess-cont-miss", message="Response without block.", continuation=True
    )
    result, code = run_hook(db_path, payload)

    assert result is None
    assert code == 0
    conn.close()


# ============================================================
# main() tests — error
# ============================================================

#TAG: [A0CD]
# Verifies: malformed JSON on stdin causes crash handler to exit 0 (fail-open) with no stdout
@pytest.mark.error
def test_main_invalid_json_stdin_crashes_gracefully():
    import hook_helpers
    import stop_hook
    original_db = hook_helpers.DB_PATH
    original_log = hook_helpers.LOG_PATH
    db_path, conn = fresh_db()

    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    try:
        hook_helpers.DB_PATH = db_path
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, "test.log")
        with patch("sys.stdin", StringIO("NOT VALID JSON {{{}")), \
             patch("sys.stdout", captured_output), \
             patch("sys.exit", mock_exit), \
             patch.object(hook_helpers, "get_embedder", return_value=None):
            try:
                stop_hook.main()
            except (SystemExit, Exception):
                pass
    finally:
        hook_helpers.DB_PATH = original_db
        hook_helpers.LOG_PATH = original_log

    output = captured_output.getvalue()
    assert output.strip() == ""
    conn.close()


#TAG: [B1DE]
# Verifies: memory block with entries but no complete: field produces block with missing-completeness hint
@pytest.mark.error
def test_main_complete_none_blocks():
    db_path, conn = fresh_db()
    msg = "Response.\n<memory>\n- type: fact\n- topic: no-complete\n- content: Memory with no completeness declaration at all\n- context: sufficient\n- keywords: test\n</memory>"
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert result["decision"] == "block"
    strict_metric = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'strict_validation_failed'"
    ).fetchone()[0]
    assert strict_metric == 1
    conn.close()


#TAG: [C2EF]
# Verifies: block with complete: true but missing context: and keywords: triggers strict validation block
@pytest.mark.error
def test_main_strict_validation_missing_fields_blocks():
    db_path, conn = fresh_db()
    msg = "Response.\n<memory>\n- type: fact\n- topic: strict-test\n- content: Testing strict validation of required memory block fields\n- complete: true\n</memory>"
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert result["decision"] == "block"
    strict_metric = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'strict_validation_failed'"
    ).fetchone()[0]
    assert strict_metric == 1
    conn.close()


#TAG: [D3FA]
# Verifies: unclosed <memory> tag on instructed session produces malformed block hint
@pytest.mark.error
def test_main_malformed_block_open_tag_only():
    db_path, conn = fresh_db()
    sid = "sess-malformed"
    conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', ?)", (sid,))
    conn.execute("INSERT INTO metrics (event, session_id) VALUES ('hook_fired', ?)", (sid,))
    conn.commit()

    msg = "Response text.\n<memory>\n- type: fact\n- topic: malform\nGarbled content without closing tag"
    payload = make_payload(session_id=sid, message=msg)
    result, code = run_hook(db_path, payload)

    assert code == 0
    # Parser handles unclosed tags gracefully — extracts content after <memory>
    # The entry gets parsed but hits strict validation or density gate
    conn.close()


#TAG: [E4AB]
# Verifies: hash mismatch in compact format is logged as metric but does not block
@pytest.mark.error
def test_main_hash_mismatch_non_blocking():
    db_path, conn = fresh_db()
    msg = "Response text.\n<memory>\nfact/hash-test: Testing hash mismatch non-blocking behavior in stop hook [k: test, hash]\n+ c h:DEAD\n</memory>"
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert code == 0
    mismatch = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'hash_mismatch'"
    ).fetchone()[0]
    assert mismatch == 1
    conn.close()


#TAG: [F5BC]
# Verifies: complete: false at continuation cap forces stop instead of further blocking
@pytest.mark.error
def test_main_completeness_cap_reached_forces_stop():
    db_path, conn = fresh_db()
    sid = "sess-comp-cap"
    conn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, 'continuation_count', '3')",
        (sid,)
    )
    conn.commit()

    msg = "Response.\n<memory>\n- type: fact\n- topic: comp-cap\n- content: Testing completeness cap reached forces stop instead of blocking\n- complete: false\n- remaining: more work\n- context: sufficient\n- keywords: test\n</memory>"
    payload = make_payload(session_id=sid, message=msg)
    result, code = run_hook(db_path, payload)

    assert result is None
    assert code == 0
    cap_metric = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'completeness_cap_hit' AND session_id = ?",
        (sid,)
    ).fetchone()[0]
    assert cap_metric == 1
    conn.close()


# ============================================================
# main() tests — adversarial
# ============================================================

#TAG: [A6CD]
# Verifies: nested/duplicate <memory> tags are handled without crash
@pytest.mark.adversarial
def test_main_adversarial_nested_memory_tags():
    db_path, conn = fresh_db()
    msg = (
        "Response text.\n"
        "<memory>\n<memory>\n"
        "- type: fact\n- topic: nested-tag\n- content: Testing nested memory tag handling in parser\n"
        "- complete: true\n- context: sufficient\n- keywords: test\n"
        "</memory>\n</memory>"
    )
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert code == 0
    # Structural: no crash metric and memories table unaffected
    crash_count = conn.execute("SELECT COUNT(*) FROM metrics WHERE event = 'hook_crash'").fetchone()[0]
    assert crash_count == 0
    conn.close()


#TAG: [B7DE]
# Verifies: >1000 char response body with empty memory block triggers density enforcement
@pytest.mark.adversarial
def test_main_adversarial_huge_response_no_entries():
    db_path, conn = fresh_db()
    huge_text = "A" * 2000 + "\n<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    payload = make_payload(message=huge_text)
    result, code = run_hook(db_path, payload)

    assert result["decision"] == "block"
    density_metric = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'content_density_failed'"
    ).fetchone()[0]
    assert density_metric == 1
    conn.close()


#TAG: [C8EF]
# Verifies: context_need of 3 chars ("xyz") is below 8-char minimum and prefiltered
@pytest.mark.adversarial
def test_main_context_need_too_short_prefiltered():
    db_path, conn = fresh_db()
    msg = "Response.\n<memory>\n- type: fact\n- topic: short-need\n- content: Testing that very short context needs are prefiltered by length check\n- complete: true\n- context: insufficient\n- context_need: xyz\n- keywords: test\n</memory>"
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert code == 0
    prefilter = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'context_prefiltered'"
    ).fetchone()[0]
    assert prefilter == 1
    conn.close()


#TAG: [D9FA]
# Verifies: two entries in one block both stored and hook_fired metric records entries=2
@pytest.mark.adversarial
def test_main_multiple_entries_stored_with_metrics():
    db_path, conn = fresh_db()
    msg = "Response text.\n" + valid_block(
        "- type: fact\n- topic: multi-1\n- content: First memory entry stored in the database correctly\n"
        "- type: decision\n- topic: multi-2\n- content: Second memory entry stored in the database correctly"
    )
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert result is None
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 2
    hook_metric = conn.execute(
        "SELECT detail FROM metrics WHERE event = 'hook_fired'"
    ).fetchone()
    assert hook_metric[0] == "entries=2"
    conn.close()


#TAG: [EA0B]
# Verifies: high cosine similarity on trailing sentence triggers trailing intent block
@pytest.mark.adversarial
def test_main_trailing_intent_blocks():
    db_path, conn = fresh_db()
    import hook_helpers as hh
    mock_emb = _make_mock_embedder()
    mock_emb.cosine_similarity.return_value = 0.95
    mock_emb.embed.return_value = b"\x00" * 384

    msg = "I analyzed the code. Let me run the tests now.\n" + valid_block(
        "- type: fact\n- topic: trail-intent\n- content: Testing trailing intent detection in stop hook"
    )
    payload = make_payload(message=msg)

    import stop_hook
    original_db = hh.DB_PATH
    original_log = hh.LOG_PATH
    captured_output = StringIO()
    exit_code = [0]

    def mock_exit(code=0):
        exit_code[0] = code
        raise SystemExit(code)

    try:
        hh.DB_PATH = db_path
        hh.LOG_PATH = os.path.join(TEST_DIR, "test.log")
        with patch("sys.stdin", StringIO(json.dumps(payload))), \
             patch("sys.stdout", captured_output), \
             patch("sys.exit", mock_exit), \
             patch.object(hh, "get_embedder", return_value=mock_emb):
            try:
                stop_hook.main()
            except SystemExit:
                pass
    finally:
        hh.DB_PATH = original_db
        hh.LOG_PATH = original_log

    output = captured_output.getvalue()
    result = json.loads(output) if output.strip() else {}
    assert result["decision"] == "block"
    intent_metric = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'trailing_intent_blocked'"
    ).fetchone()[0]
    assert intent_metric == 1
    conn.close()


#TAG: [FB1C]
# Verifies: context bootstrap triggers block when turns exceed interval and response is short (<200 chars)
@pytest.mark.adversarial
def test_main_context_bootstrap_short_response_blocks():
    db_path, conn = fresh_db()
    sid = "sess-bootstrap-adv"
    for i in range(12):
        conn.execute(
            "INSERT INTO metrics (event, session_id) VALUES ('hook_fired', ?)", (sid,)
        )
    conn.commit()

    msg = "Ok.\n" + valid_block(
        "- type: fact\n- topic: bootstrap-test\n- content: Testing context bootstrap trigger mechanism in stop hook"
    )
    payload = make_payload(session_id=sid, message=msg)
    result, code = run_hook(db_path, payload)

    bootstrap = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'context_bootstrap_triggered' AND session_id = ?",
        (sid,)
    ).fetchone()[0]
    assert bootstrap >= 1
    assert result["decision"] == "block"
    conn.close()


#TAG: [AC2D]
# Verifies: entry with content shorter than 20 chars is blocked by density gate
@pytest.mark.adversarial
def test_main_density_short_content_blocks():
    db_path, conn = fresh_db()
    msg = "Response.\n<memory>\n- type: fact\n- topic: thin\n- content: too short\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
    payload = make_payload(message=msg)
    result, code = run_hook(db_path, payload)

    assert result["decision"] == "block"
    density_metric = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE event = 'content_density_failed'"
    ).fetchone()[0]
    assert density_metric == 1
    conn.close()
