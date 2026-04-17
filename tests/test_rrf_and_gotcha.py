#!/usr/bin/env python3
"""Tests for RRF fusion, correction-file association, and PreToolUse gotcha injection.

Three features:
1. RRF fusion — FTS5+vector results merged via Reciprocal Rank Fusion
2. Correction-file association — corrections auto-tagged with file paths from transcript
3. Gotcha injection — PreToolUse hook surfaces corrections for files being accessed
"""

import sys
import os
import json
import re
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


import hooks.hook_helpers as hook_helpers
import hooks.storage as storage
import hooks.retrieval as retrieval

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


def make_vector(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


def make_blob(seed):
    return make_vector(seed).tobytes()


def fresh_db():
    _counter[0] += 1
    db_path = os.path.join(TEST_DIR, f"rrf_{_counter[0]}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
        embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
        source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER,
        archived_reason TEXT, associated_files TEXT, keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        origin_id TEXT,
        user_id TEXT,
        updated_by TEXT,
        team_id TEXT,
        source_ref TEXT,
        deleted_at TIMESTAMP,
        synced_at TIMESTAMP)""")
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
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        topic, content, keywords, content=memories, content_rowid=id)""")
    conn.execute("""CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, topic, content, keywords) VALUES (new.id, new.topic, new.content, new.keywords); END""")
    conn.execute("""CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, topic, content, keywords) VALUES ('delete', old.id, old.topic, old.content, old.keywords); END""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hook_state (
        session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, key))""")
    conn.commit()
    return db_path, conn


# ============================================================
# RRF Fusion Tests
# ============================================================

#TAG: [25C0] 2026-04-05
# Verifies: FTS-only results get a real composite score (not hardcoded 0.30) and appear as the sole output entry
@pytest.mark.behavioural
def test_retrieve_context_rrf_fts_only_scores():
    """FTS-only results should get real scores instead of hardcoded 0.30."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("fact", "sqlite-wal", "SQLite WAL mode enables concurrent reads", make_blob(100), "proj", 0.8, "other-session")
    )
    conn.commit()
    mem_id = conn.execute("SELECT id FROM memories WHERE topic='sqlite-wal'").fetchone()[0]

    # Embedder returns no results (simulating semantic miss), but FTS should find "SQLite WAL"
    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = []

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("SQLite WAL mode", session_id="s1")

    # result.finditer raises AttributeError if None — this verifies result is non-empty and correct
    entry_ids = [int(m.group(1)) for m in re.finditer(r'<entry id="(\d+)"', result)]
    assert len(entry_ids) == 1, f"Expected exactly 1 FTS entry, got {len(entry_ids)}"
    assert entry_ids[0] == mem_id, f"Expected entry id={mem_id}, got {entry_ids[0]}"


#TAG: [5BA7] 2026-04-05
# Verifies: memory found by both FTS and semantic scores higher than one found by FTS only
@pytest.mark.behavioural
def test_retrieve_context_rrf_dual_match_boosted():
    """A memory found by both FTS and semantic should score higher than one found by only one method."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    # Memory that will match both FTS (keyword "JWT") and semantic (close vector)
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("decision", "auth-jwt", "Use JWT tokens for stateless authentication", make_blob(100), "proj", 0.8, "other")
    )
    # Memory that will only match FTS (different vector, but contains "authentication")
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("fact", "auth-history", "Authentication was migrated from session cookies", make_blob(999), "proj", 0.7, "other")
    )
    conn.commit()

    mock_emb = MagicMock()
    # Semantic search returns only the JWT memory (close vector)
    mock_emb.find_similar.return_value = [{
        "id": 1, "type": "decision", "topic": "auth-jwt",
        "content": "Use JWT tokens for stateless authentication",
        "updated_at": "2026-04-01 10:00:00", "project": "proj",
        "confidence": 0.8, "session_id": "other", "depth": None,
        "archived_reason": None, "similarity": 0.75, "score": 0.55,
    }]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("JWT authentication tokens", session_id="s1")

    # JWT memory (dual match) should appear first — extract ordered ids
    entry_ids = [int(m.group(1)) for m in re.finditer(r'<entry id="(\d+)"', result)]
    assert entry_ids[0] == 1, f"Dual-match JWT memory (id=1) should rank first, got order: {entry_ids}"


#TAG: [FAE4] 2026-04-05
# Verifies: exact keyword match via FTS surfaces the memory as the only entry even when semantic returns nothing
@pytest.mark.behavioural
def test_retrieve_context_rrf_fts_exact_match():
    """An exact keyword match via FTS should surface even without semantic support."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("correction", "err-sqlite-busy", "SQLITE_BUSY error fixed by increasing busy_timeout to 10s", make_blob(500), "proj", 0.7, "other")
    )
    conn.commit()
    mem_id = conn.execute("SELECT id FROM memories WHERE topic='err-sqlite-busy'").fetchone()[0]

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = []  # Semantic finds nothing

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("SQLITE_BUSY", session_id="s1")

    # result.finditer raises AttributeError if None — this verifies result is non-empty and correct
    entry_ids = [int(m.group(1)) for m in re.finditer(r'<entry id="(\d+)"', result)]
    assert len(entry_ids) == 1, f"Expected exactly 1 FTS result, got {len(entry_ids)}"
    assert entry_ids[0] == mem_id, f"Expected entry id={mem_id}, got {entry_ids[0]}"


#TAG: [BF3C] 2026-04-05
# Verifies: memories from the current session are excluded from both FTS and semantic result sets
@pytest.mark.edge
def test_retrieve_context_rrf_same_session_excluded():
    """Same-session memories should be excluded from both FTS and semantic results."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    conn.execute(
        "INSERT INTO memories (type, topic, content, embedding, project, confidence, session_id) VALUES (?,?,?,?,?,?,?)",
        ("fact", "test-topic", "This is a test memory about something specific", make_blob(100), "proj", 0.8, "s1")
    )
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.find_similar.return_value = [{
        "id": 1, "type": "fact", "topic": "test-topic",
        "content": "This is a test memory about something specific",
        "updated_at": "2026-04-01 10:00:00", "project": "proj",
        "confidence": 0.8, "session_id": "s1", "depth": None,
        "archived_reason": None, "similarity": 0.95, "score": 0.70,
    }]

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        result = retrieval.retrieve_context("test memory specific", session_id="s1")

    # Should be None since the only match is same-session
    assert result is None


# ============================================================
# Correction-File Association Tests
# ============================================================

#TAG: [3186] 2026-04-05
# Verifies: extract_associated_files finds exactly the file paths from Read/Edit/Write tool calls
@pytest.mark.behavioural
def test_extract_associated_files_from_transcript():
    """extract_associated_files should find file paths from Read/Edit tool calls."""
    transcript = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    entries = [
        {"tool_name": "Read", "parameters": {"file_path": "/home/user/project/foo.py"}},
        {"tool_name": "Edit", "parameters": {"file_path": "/home/user/project/bar.py"}},
        {"tool_name": "Write", "parameters": {"file_path": "/home/user/project/baz.ts"}},
        {"type": "user", "content": "fix the bug"},
        {"tool_name": "Bash", "parameters": {"command": "ls -la"}},
    ]
    for e in entries:
        transcript.write(json.dumps(e) + "\n")
    transcript.close()

    files = storage.extract_associated_files(transcript.name)
    os.unlink(transcript.name)

    assert sorted(files) == sorted([
        "/home/user/project/foo.py",
        "/home/user/project/bar.py",
        "/home/user/project/baz.ts",
    ])


#TAG: [EBA0] 2026-04-05
# Verifies: same file path appearing multiple times in transcript is deduplicated to one entry
@pytest.mark.edge
def test_extract_associated_files_deduplicates():
    """Same file path appearing multiple times should be deduplicated."""
    transcript = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for _ in range(3):
        transcript.write(json.dumps({"tool_name": "Read", "parameters": {"file_path": "/home/user/same.py"}}) + "\n")
    transcript.close()

    files = storage.extract_associated_files(transcript.name)
    os.unlink(transcript.name)

    assert files == ["/home/user/same.py"]


#TAG: [9E32] 2026-04-05
# Verifies: non-existent transcript path returns empty list without raising an exception
@pytest.mark.edge
def test_extract_associated_files_empty_transcript():
    """Non-existent transcript should return empty list."""
    files = storage.extract_associated_files("/nonexistent/path.jsonl")
    assert files == []


#TAG: [BBC3] 2026-04-05
# Verifies: correction memory inserted with a transcript gets associated_files populated from transcript
@pytest.mark.behavioural
def test_insert_memories_correction_gets_files():
    """Inserting a correction with a transcript should auto-associate file paths."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    conn.commit()

    # Create a transcript with file access
    transcript = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    transcript.write(json.dumps({"tool_name": "Read", "parameters": {"file_path": "/project/retrieval.py"}}) + "\n")
    transcript.write(json.dumps({"tool_name": "Edit", "parameters": {"file_path": "/project/config.py"}}) + "\n")
    transcript.close()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(100)
    mock_emb.to_blob.return_value = make_blob(100)
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage.insert_memories(
            [{"type": "correction", "topic": "fts-scoring", "content": "FTS results had hardcoded 0.30 score — need proper RRF fusion"}],
            session_id="s1",
            transcript_path=transcript.name
        )

    os.unlink(transcript.name)

    row = conn.execute("SELECT associated_files FROM memories WHERE topic = 'fts-scoring'").fetchone()
    files = json.loads(row[0])
    assert sorted(files) == sorted(["/project/retrieval.py", "/project/config.py"])
    conn.close()


#TAG: [CD50] 2026-04-05
# Verifies: insert_memories associates file paths with all memory types including non-corrections
@pytest.mark.behavioural
def test_insert_memories_non_correction_skips():
    """insert_memories associates files with all memory types (not just corrections) per current implementation."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    conn.commit()

    transcript = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    transcript.write(json.dumps({"tool_name": "Read", "parameters": {"file_path": "/project/foo.py"}}) + "\n")
    transcript.close()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(200)
    mock_emb.to_blob.return_value = make_blob(200)
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage.insert_memories(
            [{"type": "fact", "topic": "test-fact", "content": "This is a fact not a correction"}],
            session_id="s1",
            transcript_path=transcript.name
        )

    os.unlink(transcript.name)

    row = conn.execute("SELECT associated_files FROM memories WHERE topic = 'test-fact'").fetchone()
    # insert_memories associates files with all memory types — non-corrections get files too
    import json as _json
    files = _json.loads(row[0]) if row[0] else []
    assert files == ["/project/foo.py"], f"Expected exactly ['/project/foo.py'], got {files}"
    conn.close()


#TAG: [0961] 2026-04-05
# Verifies: insert_memories with no transcript_path leaves associated_files as NULL
@pytest.mark.edge
def test_insert_memories_no_transcript_null_files():
    """insert_memories without a transcript_path must leave associated_files as NULL."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(300)
    mock_emb.to_blob.return_value = make_blob(300)
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage.insert_memories(
            [{"type": "fact", "topic": "no-transcript", "content": "A fact with no transcript"}],
            session_id="s1",
            transcript_path=None
        )

    row = conn.execute("SELECT associated_files FROM memories WHERE topic = 'no-transcript'").fetchone()
    assert row[0] is None, f"Expected NULL associated_files when no transcript_path; got {row[0]!r}"
    conn.close()


#TAG: [C207] 2026-04-05
# Verifies: insert_memories with a non-existent transcript path stores empty JSON array (not NULL or crash)
@pytest.mark.error
def test_insert_memories_nonexistent_transcript():
    """insert_memories with a nonexistent transcript path must not crash; stores empty associated_files."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    conn.commit()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(400)
    mock_emb.to_blob.return_value = make_blob(400)
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage.insert_memories(
            [{"type": "correction", "topic": "bad-path", "content": "Correction with missing transcript"}],
            session_id="s1",
            transcript_path="/nonexistent/path/transcript.jsonl"
        )

    row = conn.execute("SELECT associated_files FROM memories WHERE topic = 'bad-path'").fetchone()
    assert row is not None, "Memory must be inserted even if transcript is missing"
    # extract_associated_files returns [] for nonexistent paths — stored as empty JSON array
    files = json.loads(row[0]) if row[0] else []
    assert files == [], f"Expected empty associated_files for missing transcript; got {files}"
    conn.close()


#TAG: [ECAD] 2026-04-05
# Verifies: insert_memories with a transcript containing Glob tool calls (not file-tool) does not associate those paths
@pytest.mark.adversarial
def test_insert_memories_non_file_tool_not_associated():
    """insert_memories must not associate file paths from non-file tools (e.g. Glob pattern strings)."""
    db_path, conn = fresh_db()
    conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'proj')")
    conn.commit()

    transcript = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    # Glob tool call with a pattern string — must NOT be treated as a file path
    transcript.write(json.dumps({"tool_name": "Glob", "parameters": {"pattern": "**/*.py"}}) + "\n")
    # Bash tool with a command — must NOT be treated as a file path
    transcript.write(json.dumps({"tool_name": "Bash", "parameters": {"command": "ls -la /project/"}}) + "\n")
    # Only this Read call should produce an association
    transcript.write(json.dumps({"tool_name": "Read", "parameters": {"file_path": "/project/real_file.py"}}) + "\n")
    transcript.close()

    mock_emb = MagicMock()
    mock_emb.embed.return_value = make_vector(500)
    mock_emb.to_blob.return_value = make_blob(500)
    mock_emb.find_nearest.return_value = []
    mock_emb.upsert_vec_index = MagicMock()

    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
        storage.insert_memories(
            [{"type": "correction", "topic": "tool-filter", "content": "Only file tools should be associated"}],
            session_id="s1",
            transcript_path=transcript.name
        )

    os.unlink(transcript.name)

    row = conn.execute("SELECT associated_files FROM memories WHERE topic = 'tool-filter'").fetchone()
    files = json.loads(row[0]) if row[0] else []
    assert files == ["/project/real_file.py"], \
        f"Expected only Read file_path; got {files}"
    conn.close()


# ============================================================
# Gotcha Injection Tests
# ============================================================

#TAG: [F7DA] 2026-04-05
# Verifies: find_memories_for_file with corrections_only=True matches a correction by exact file path
@pytest.mark.behavioural
def test_find_memories_for_file_exact_path():
    """find_memories_for_file with corrections_only=True should match on exact file path."""
    import hooks.pretool_hook as pretool_hook
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
        ("correction", "fts-bug", "FTS results had wrong scores",
         json.dumps(["/project/hooks/retrieval.py"]), 0.8)
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        matches = pretool_hook.find_memories_for_file("/project/hooks/retrieval.py", corrections_only=True)

    assert len(matches) == 1
    assert matches[0]["topic"] == "fts-bug"
    conn.close()


#TAG: [9199] 2026-04-05
# Verifies: find_memories_for_file with corrections_only=True falls back to basename match when full paths differ
@pytest.mark.behavioural
def test_find_memories_for_file_by_basename():
    """find_corrections_for_file should match on basename when full path differs."""
    import hooks.pretool_hook as pretool_hook
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
        ("correction", "config-issue", "Config had wrong default",
         json.dumps(["/old/path/config.py"]), 0.7)
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        matches = pretool_hook.find_memories_for_file("/new/path/config.py", corrections_only=True)

    assert len(matches) == 1
    assert matches[0]["topic"] == "config-issue"
    conn.close()


#TAG: [CFBF] 2026-04-05
# Verifies: find_memories_for_file with corrections_only=True excludes archived corrections
@pytest.mark.edge
def test_find_memories_for_file_skips_archived():
    """Archived corrections should not be injected."""
    import hooks.pretool_hook as pretool_hook
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason) VALUES (?,?,?,?,?,?)",
        ("correction", "old-bug", "This was fixed long ago",
         json.dumps(["/project/foo.py"]), 0.5, "superseded by new approach")
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        matches = pretool_hook.find_memories_for_file("/project/foo.py", corrections_only=True)

    assert len(matches) == 0
    conn.close()


#TAG: [495E] 2026-04-05
# Verifies: find_memories_for_file with corrections_only=True returns empty list for unrelated file
@pytest.mark.edge
def test_find_memories_for_file_no_match():
    """No corrections for an unrelated file should return empty list."""
    import hooks.pretool_hook as pretool_hook
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
        ("correction", "specific-bug", "Bug in specific file",
         json.dumps(["/project/specific.py"]), 0.8)
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        matches = pretool_hook.find_memories_for_file("/project/unrelated.py", corrections_only=True)

    assert len(matches) == 0
    conn.close()


#TAG: [BA4F] 2026-04-05
# Verifies: find_memories_for_file returns all matching corrections; main() applies MAX_GOTCHA_INJECTIONS cap
@pytest.mark.behavioural
def test_find_memories_for_file_max_injections():
    """find_memories_for_file returns all matches; the MAX_GOTCHA_INJECTIONS cap is applied in main()."""
    import hooks.pretool_hook as pretool_hook
    db_path, conn = fresh_db()
    for i in range(10):
        conn.execute(
            "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
            ("correction", f"bug-{i}", f"Bug number {i} in the file",
             json.dumps(["/project/buggy.py"]), 0.7)
        )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        matches = pretool_hook.find_memories_for_file("/project/buggy.py", corrections_only=True)

    # find_memories_for_file returns ALL matches; the cap is applied in main() via [:MAX_GOTCHA_INJECTIONS]
    assert len(matches) == 10, f"Expected all 10 matches from find_memories_for_file, got {len(matches)}"
    conn.close()


#TAG: [3BE4] 2026-04-05
# Verifies: pretool hook produces additionalContext JSON with CAIRN GOTCHA header and topic name
@pytest.mark.behavioural
def test_main_gotcha_outputs_additional_context():
    """The pretool hook should output additionalContext JSON for matching files."""
    import hooks.pretool_hook as pretool_hook
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
        ("correction", "import-bug", "Imports fail when cairn module not on path",
         json.dumps(["/project/hooks/stop_hook.py"]), 0.8)
    )
    conn.execute("""CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT, session_id TEXT, detail TEXT, value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()

    hook_input = json.dumps({
        "tool_name": "Read",
        "tool_input": {"file_path": "/project/hooks/stop_hook.py"},
        "session_id": "test-session",
    })

    from unittest.mock import patch as _patch
    from io import StringIO
    captured = StringIO()

    with _patch.object(hook_helpers, 'DB_PATH', db_path), \
         _patch('sys.stdin', StringIO(hook_input)), \
         _patch('sys.stdout', captured), \
         _patch('sys.exit') as mock_exit:
        pretool_hook.main()

    output = captured.getvalue()
    assert len(output) > 0, "Hook must produce non-empty output for matching corrections"
    parsed = json.loads(output)
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    # Verify exact first line (structural — not substring containment)
    ctx_lines = ctx.split('\n')
    assert ctx_lines[0] == "CAIRN GOTCHA for stop_hook.py:", \
        f"Expected GOTCHA header as first line; got: {ctx_lines[0]!r}"
    # Verify the correction topic appears on the correction bullet line (exact equality)
    correction_line = next((l for l in ctx_lines if l.startswith('- [')), None)
    assert correction_line == "- [import-bug] Imports fail when cairn module not on path", \
        f"Expected exact correction line; got: {correction_line!r}"
    conn.close()


#TAG: [8153] 2026-04-05
# Verifies: Bash tool (non-file-access) causes main() to exit(0) before printing anything — exit code proves early branch taken
@pytest.mark.edge
def test_main_non_file_tool_exits_without_output():
    """main() must exit without writing to stdout for non-file-access tools (Bash, Grep, etc.)."""
    import hooks.pretool_hook as pretool_hook
    from io import StringIO as SIO
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
        ("correction", "some-fix", "A correction that should never appear",
         json.dumps(["/any/file.py"]), 0.8)
    )
    conn.commit()

    captured = SIO()
    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch('sys.stdin', SIO(json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}, "session_id": "s1"}))), \
         patch('sys.stdout', captured):
        with pytest.raises(SystemExit) as exc_info:
            pretool_hook.main()

    assert exc_info.value.code == 0, f"Must exit(0) for non-file tool, got code: {exc_info.value.code}"
    assert captured.getvalue() == "", f"Non-file tool must produce no stdout, got: {repr(captured.getvalue())}"
    conn.close()


#TAG: [D5FE] 2026-04-05
# Verifies: empty file_path causes main() to exit(0) before printing — no path lookup on empty string
@pytest.mark.error
def test_main_empty_file_path_produces_no_output():
    """main() must exit cleanly when file_path is empty — exit code 0, no output."""
    import hooks.pretool_hook as pretool_hook
    from io import StringIO as SIO
    db_path, conn = fresh_db()
    conn.commit()

    captured = SIO()
    with patch.object(hook_helpers, 'DB_PATH', db_path), \
         patch('sys.stdin', SIO(json.dumps({"tool_name": "Read", "tool_input": {"file_path": ""}, "session_id": "s1"}))), \
         patch('sys.stdout', captured):
        with pytest.raises(SystemExit) as exc_info:
            pretool_hook.main()

    assert exc_info.value.code == 0, f"Must exit(0) for empty file_path, got code: {exc_info.value.code}"
    assert captured.getvalue() == "", f"Empty file_path must produce no stdout, got: {repr(captured.getvalue())}"
    conn.close()


#TAG: [FAF4] 2026-04-05
# Verifies: malformed JSON on stdin raises JSONDecodeError (caught by __main__ wrapper), not another exception type
@pytest.mark.adversarial
def test_main_malformed_json_stdin_raises_json_error():
    """main() with non-JSON stdin must raise JSONDecodeError (not crash with unexpected type)."""
    import hooks.pretool_hook as pretool_hook
    from io import StringIO as SIO

    raised = None
    with patch('sys.stdin', SIO("not { valid } json!!!")), \
         patch('sys.stdout', SIO()), \
         patch('sys.exit'):
        try:
            pretool_hook.main()
        except json.JSONDecodeError as e:
            raised = e
        except Exception as e:
            pytest.fail(f"Expected JSONDecodeError, got {type(e).__name__}: {e}")

    assert isinstance(raised, json.JSONDecodeError), (
        "Malformed stdin must raise json.JSONDecodeError so the __main__ wrapper can catch it"
    )


# ============================================================
# Additional category coverage: extract_associated_files error + adversarial
# find_memories_for_file error + adversarial
# ============================================================

#TAG: [7418] 2026-04-05
# Verifies: malformed JSON lines in transcript are skipped and valid tool calls still collected
@pytest.mark.error
def test_extract_associated_files_malformed_json():
    """extract_associated_files must skip malformed JSONL lines and still return valid paths."""
    transcript = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    transcript.write("NOT VALID JSON AT ALL\n")
    transcript.write(json.dumps({"tool_name": "Read", "parameters": {"file_path": "/project/valid.py"}}) + "\n")
    transcript.write("{broken: json}\n")
    transcript.close()

    files = storage.extract_associated_files(transcript.name)
    os.unlink(transcript.name)

    assert files == ["/project/valid.py"], (
        f"Must return only valid paths and skip malformed lines, got {files}"
    )


#TAG: [CE6B] 2026-04-05
# Verifies: MultiEdit with multiple file_path entries in edits array are all captured
@pytest.mark.adversarial
def test_extract_associated_files_multiedit_paths():
    """MultiEdit entries contain multiple file paths in an 'edits' array — all must be captured."""
    transcript = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    transcript.write(json.dumps({
        "tool_name": "MultiEdit",
        "parameters": {
            "edits": [
                {"file_path": "/project/a.py"},
                {"file_path": "/project/b.py"},
                {"file_path": "/project/c.py"},
            ]
        }
    }) + "\n")
    transcript.close()

    files = storage.extract_associated_files(transcript.name)
    os.unlink(transcript.name)

    assert sorted(files) == ["/project/a.py", "/project/b.py", "/project/c.py"], (
        f"All 3 MultiEdit paths must be captured, got {files}"
    )


#TAG: [A8E2] 2026-04-05
# Verifies: find_memories_for_file returns empty list immediately for empty file_path
@pytest.mark.error
def test_find_memories_for_file_empty_path():
    """find_memories_for_file with empty string must return [] without querying the DB."""
    import hooks.pretool_hook as pretool_hook
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
        ("correction", "some-bug", "Some bug in some file", json.dumps(["/any/file.py"]), 0.8)
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        matches = pretool_hook.find_memories_for_file("", corrections_only=True)

    assert len(matches) == 0, f"Empty file_path must return 0 matches, got {len(matches)}"
    conn.close()


#TAG: [3593] 2026-04-05
# Verifies: find_memories_for_file skips NULL associated_files without crashing
@pytest.mark.adversarial
def test_find_memories_for_file_null_files():
    """Correction with NULL associated_files should be skipped (not crash on json.loads)."""
    import hooks.pretool_hook as pretool_hook
    db_path, conn = fresh_db()
    # Row with NULL associated_files (not the same as empty JSON array)
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
        ("correction", "null-files-bug", "Bug with no file association", None, 0.7)
    )
    # Row with valid associated_files — must still be returned
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence) VALUES (?,?,?,?,?)",
        ("correction", "valid-bug", "Bug in target file", json.dumps(["/project/target.py"]), 0.8)
    )
    conn.commit()

    with patch.object(hook_helpers, 'DB_PATH', db_path):
        matches = pretool_hook.find_memories_for_file("/project/target.py", corrections_only=True)

    assert len(matches) == 1, f"Must return 1 valid match despite NULL associated_files row, got {len(matches)}"
    assert matches[0]["topic"] == "valid-bug", f"Wrong match: {matches[0]['topic']}"
    conn.close()
