"""Tests for show_context in query.py — mechanical verification suite."""

import json
import os
import re
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import sys
import tempfile
from unittest.mock import patch
from io import StringIO

import pytest


# We import show_context directly — the module-level DB_PATH will be patched per test
import cairn.query as query


def _create_test_db(db_path):
    """Create a minimal cairn DB schema for testing."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            type TEXT,
            topic TEXT,
            content TEXT,
            session_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            depth INTEGER,
            confidence REAL DEFAULT 0.7,
            embedding BLOB,
            source_start INTEGER,
            source_end INTEGER,
            project TEXT,
            archived_reason TEXT,
            origin_id TEXT,
            user_id TEXT,
            updated_by TEXT,
            team_id TEXT,
            source_ref TEXT,
            deleted_at TIMESTAMP,
            synced_at TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            transcript_path TEXT,
            project TEXT
        )
    """)
    conn.commit()
    return conn


def _write_transcript(path, messages):
    """Write a JSONL transcript file from a list of message dicts.

    Each dict should have: role, text, timestamp (ISO string).
    """
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            entry = {
                "timestamp": msg.get("timestamp", "2026-03-28T10:00:00Z"),
                "message": {
                    "role": msg["role"],
                    "content": msg["text"],
                },
            }
            f.write(json.dumps(entry) + "\n")


def _parse_context_lines(output):
    """Extract the context message lines from show_context output.

    Returns list of dicts with keys: time, role, text, is_anchor.
    """
    results = []
    # Context lines look like: [HH:MM:SS] role: text content <<<
    pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2}|\?)\]\s+(user|assistant):\s+(.+?)(\s+<<<)?$")
    for line in output.strip().split("\n"):
        line = line.strip()
        m = pattern.match(line)
        if m:
            results.append({
                "time": m.group(1),
                "role": m.group(2),
                "text": m.group(3).rstrip(),
                "is_anchor": m.group(4) is not None,
            })
    return results


@pytest.fixture
def cairn_env(tmp_path):
    """Set up a temp DB and transcript directory, patching DB_PATH."""
    db_path = str(tmp_path / "test_cairn.db")
    conn = _create_test_db(db_path)
    conn.close()
    with patch.object(query, "DB_PATH", db_path):
        yield tmp_path, db_path


#TAG: [996B] 2026-04-05
# Verifies: show_context extracts correct conversation window around memory timestamp with proper anchor
def test_show_context_behavioural_normal(cairn_env, capsys):
    tmp_path, db_path = cairn_env
    transcript_path = str(tmp_path / "transcript.jsonl")

    # Write a 6-message transcript spanning 10:00-10:05
    messages = [
        {"role": "user", "text": "Hello, starting work", "timestamp": "2026-03-28T10:00:00Z"},
        {"role": "assistant", "text": "Ready to help", "timestamp": "2026-03-28T10:01:00Z"},
        {"role": "user", "text": "Fix the auth bug", "timestamp": "2026-03-28T10:02:00Z"},
        {"role": "assistant", "text": "Found the issue in middleware", "timestamp": "2026-03-28T10:03:00Z"},
        {"role": "user", "text": "Great, deploy it", "timestamp": "2026-03-28T10:04:00Z"},
        {"role": "assistant", "text": "Deployed successfully", "timestamp": "2026-03-28T10:05:00Z"},
    ]
    _write_transcript(transcript_path, messages)

    # Insert memory created at 10:03:30 (between msg 3 and msg 4)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, session_id, created_at, depth) "
        "VALUES (1, 'fact', 'auth-bug', 'Found auth middleware issue', 'sess-001', '2026-03-28 10:03:30', NULL)"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, transcript_path) VALUES ('sess-001', ?)",
        (transcript_path,),
    )
    conn.commit()
    conn.close()

    query.show_context(1)
    output = capsys.readouterr().out
    ctx = _parse_context_lines(output)

    # Header line must contain memory metadata with correct format
    header_line = output.split("\n")[0]
    assert header_line == "=== Memory [1] fact/auth-bug ==="

    # Anchor is message index 3 (10:03, last msg at or before 10:03:30).
    # Default margin=5, so start_idx=max(0,3-5)=0, end_idx=min(6,3+3)=6, showing all 6 messages.
    assert len(ctx) == 6
    # Exactly one anchor marker, and it must be on the "Found the issue" message
    anchors = [m for m in ctx if m["is_anchor"]]
    assert len(anchors) == 1
    assert anchors[0]["text"] == "Found the issue in middleware"
    assert anchors[0]["time"] == "10:03:00"
    # Messages appear in chronological order
    times = [m["time"] for m in ctx]
    assert times == sorted(times)


#TAG: [B149] 2026-04-05
# Verifies: when all transcript messages precede memory timestamp, anchor is last message
def test_show_context_edge_memory_after_all_messages(cairn_env, capsys):
    tmp_path, db_path = cairn_env
    transcript_path = str(tmp_path / "transcript.jsonl")

    messages = [
        {"role": "user", "text": "First message", "timestamp": "2026-03-28T09:00:00Z"},
        {"role": "assistant", "text": "Second message", "timestamp": "2026-03-28T09:01:00Z"},
        {"role": "user", "text": "Third message", "timestamp": "2026-03-28T09:02:00Z"},
    ]
    _write_transcript(transcript_path, messages)

    # Memory created well after all messages — anchor should default to last
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, session_id, created_at, depth) "
        "VALUES (2, 'decision', 'late-mem', 'Decided after transcript ended', 'sess-002', '2026-03-28 12:00:00', NULL)"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, transcript_path) VALUES ('sess-002', ?)",
        (transcript_path,),
    )
    conn.commit()
    conn.close()

    query.show_context(2)
    output = capsys.readouterr().out
    ctx = _parse_context_lines(output)

    # All 3 messages shown (anchor=2, start=max(0,2-5)=0, end=min(3,2+3)=3)
    assert len(ctx) == 3
    # Anchor must be the LAST message since no message exceeds mem_time
    anchors = [m for m in ctx if m["is_anchor"]]
    assert len(anchors) == 1
    assert anchors[0]["text"] == "Third message"
    assert anchors[0] == ctx[-1]


#TAG: [3C14] 2026-04-05
# Verifies: show_context prints error and returns for nonexistent memory ID
def test_show_context_error_no_memory(cairn_env, capsys):
    tmp_path, db_path = cairn_env

    query.show_context(9999)
    output = capsys.readouterr().out

    # Output should be exactly the error line (plus newline), nothing else
    output_lines = [l for l in output.strip().split("\n") if l.strip()]
    assert len(output_lines) == 1
    assert output_lines[0] == "No memory with id 9999"


#TAG: [D255] 2026-04-05
# Verifies: malformed JSONL lines are skipped and valid messages still produce correct context
def test_show_context_adversarial_malformed_jsonl(cairn_env, capsys):
    tmp_path, db_path = cairn_env
    transcript_path = str(tmp_path / "transcript.jsonl")

    # Write transcript: 5 lines total, 2 garbage + 3 valid
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("NOT VALID JSON\n")
        f.write(json.dumps({
            "timestamp": "2026-03-28T10:00:00Z",
            "message": {"role": "user", "content": "Valid first"},
        }) + "\n")
        f.write("{broken json\n")
        f.write(json.dumps({
            "timestamp": "2026-03-28T10:01:00Z",
            "message": {"role": "assistant", "content": "Valid second"},
        }) + "\n")
        f.write("\n")  # empty line
        f.write(json.dumps({
            "timestamp": "2026-03-28T10:02:00Z",
            "message": {"role": "user", "content": "Valid third"},
        }) + "\n")

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, session_id, created_at, depth) "
        "VALUES (3, 'fact', 'test', 'Test memory', 'sess-003', '2026-03-28 10:01:30', NULL)"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, transcript_path) VALUES ('sess-003', ?)",
        (transcript_path,),
    )
    conn.commit()
    conn.close()

    query.show_context(3)
    output = capsys.readouterr().out
    ctx = _parse_context_lines(output)

    # Exactly 3 valid messages parsed (garbage lines silently skipped)
    assert len(ctx) == 3
    # Verify the parsed texts match exactly the 3 valid entries in order
    assert [m["text"] for m in ctx] == ["Valid first", "Valid second", "Valid third"]
    # Anchor should be "Valid second" (10:01, last msg at or before 10:01:30)
    anchors = [m for m in ctx if m["is_anchor"]]
    assert len(anchors) == 1
    assert anchors[0]["text"] == "Valid second"


#TAG: [D0D6] 2026-04-05
# Verifies: memory with no session_id prints "No session ID" and returns without error
def test_show_context_error_no_session_id(cairn_env, capsys):
    tmp_path, db_path = cairn_env

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, session_id, created_at) "
        "VALUES (4, 'fact', 'orphan', 'No session attached', NULL, '2026-03-28 10:00:00')"
    )
    conn.commit()
    conn.close()

    query.show_context(4)
    output = capsys.readouterr().out
    output_lines = [l.strip() for l in output.strip().split("\n") if l.strip()]

    # Should print memory header then session-missing message, then stop (no transcript output)
    assert len(output_lines) == 3  # header, content, "No session ID" message
    assert output_lines[0] == "=== Memory [4] fact/orphan ==="
    assert output_lines[2] == "No session ID \u2014 cannot locate transcript."


#TAG: [1CB7] 2026-04-05
# Verifies: when transcript file path is recorded but file is missing on disk, prints error path
def test_show_context_error_missing_transcript_file(cairn_env, capsys):
    tmp_path, db_path = cairn_env
    nonexistent = str(tmp_path / "ghost_transcript.jsonl")

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, session_id, created_at) "
        "VALUES (5, 'fact', 'gone', 'Transcript deleted', 'sess-005', '2026-03-28 10:00:00')"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, transcript_path) VALUES ('sess-005', ?)",
        (nonexistent,),
    )
    conn.commit()
    conn.close()

    query.show_context(5)
    output = capsys.readouterr().out
    output_lines = [l.strip() for l in output.strip().split("\n") if l.strip()]

    # The "Transcript not found" line must include the actual missing path for debuggability
    not_found_lines = [l for l in output_lines if "Transcript not found" in l]
    assert len(not_found_lines) == 1
    assert not_found_lines[0] == f"Transcript not found: {nonexistent}"


#TAG: [37B3] 2026-04-05
# Verifies: depth field on memory controls lookback window size instead of default margin
def test_show_context_edge_depth_controls_lookback(cairn_env, capsys):
    tmp_path, db_path = cairn_env
    transcript_path = str(tmp_path / "transcript.jsonl")

    # 10 messages, memory at message 8 with depth=2 (should only look back 2 turns)
    messages = []
    for i in range(10):
        messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "text": f"Message number {i}",
            "timestamp": f"2026-03-28T10:{i:02d}:00Z",
        })
    _write_transcript(transcript_path, messages)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, session_id, created_at, depth) "
        "VALUES (6, 'fact', 'deep', 'Memory with depth', 'sess-006', '2026-03-28 10:08:30', 2)"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, transcript_path) VALUES ('sess-006', ?)",
        (transcript_path,),
    )
    conn.commit()
    conn.close()

    query.show_context(6)
    output = capsys.readouterr().out
    ctx = _parse_context_lines(output)

    # depth=2, anchor at msg index 8 — start=max(0,8-2)=6, end=min(10,8+3)=10
    # So messages 6,7,8,9 are shown (4 messages total)
    assert len(ctx) == 4
    # First message in window should be "Message number 6" (not earlier)
    assert ctx[0]["text"] == "Message number 6"
    # Anchor should be "Message number 8"
    anchors = [m for m in ctx if m["is_anchor"]]
    assert len(anchors) == 1
    assert anchors[0]["text"] == "Message number 8"
    # Verify depth is reported in the output metadata line
    depth_lines = [l.strip() for l in output.split("\n") if "depth:" in l and "turns" in l]
    assert len(depth_lines) == 1
    assert depth_lines[0] == "Created: 2026-03-28 10:08:30, depth: 2 turns"


#TAG: [5B14] 2026-04-05
# Verifies: transcript with only non-text content entries results in "No text-bearing messages"
def test_show_context_adversarial_no_text_content(cairn_env, capsys):
    tmp_path, db_path = cairn_env
    transcript_path = str(tmp_path / "transcript.jsonl")

    # Write entries: tool_use content (list with no text), system role (filtered), empty string content
    with open(transcript_path, "w", encoding="utf-8") as f:
        entry1 = {
            "timestamp": "2026-03-28T10:00:00Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "read", "input": {}}],
            },
        }
        entry2 = {
            "timestamp": "2026-03-28T10:01:00Z",
            "message": {"role": "system", "content": "System message ignored"},
        }
        entry3 = {
            "timestamp": "2026-03-28T10:02:00Z",
            "message": {"role": "user", "content": ""},
        }
        f.write(json.dumps(entry1) + "\n")
        f.write(json.dumps(entry2) + "\n")
        f.write(json.dumps(entry3) + "\n")

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO memories (id, type, topic, content, session_id, created_at) "
        "VALUES (7, 'fact', 'empty', 'No real messages', 'sess-007', '2026-03-28 10:01:00')"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, transcript_path) VALUES ('sess-007', ?)",
        (transcript_path,),
    )
    conn.commit()
    conn.close()

    query.show_context(7)
    output = capsys.readouterr().out
    ctx = _parse_context_lines(output)

    # Zero context messages extracted from the non-text transcript
    assert len(ctx) == 0
    # The output must contain the specific "No text-bearing messages" error
    output_lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    no_text_lines = [l for l in output_lines if l == "No text-bearing messages found in transcript."]
    assert len(no_text_lines) == 1


# === backfill_embeddings ===

from unittest.mock import MagicMock


@pytest.fixture
def backfill_env(tmp_path):
    """Create a temp DB with the memories table for backfill tests."""
    db_path = str(tmp_path / "backfill_test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY,
            type TEXT,
            topic TEXT,
            content TEXT,
            project TEXT,
            embedding BLOB,
            keywords TEXT,
            origin_id TEXT,
            user_id TEXT,
            updated_by TEXT,
            team_id TEXT,
            source_ref TEXT,
            deleted_at TIMESTAMP,
            synced_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    with patch.object(query, "DB_PATH", db_path):
        yield db_path


def _seed_memories(db_path, memories):
    """Insert test memories into the backfill test DB."""
    conn = sqlite3.connect(db_path)
    for m in memories:
        conn.execute(
            "INSERT INTO memories (id, type, topic, content, project, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            (m["id"], m["type"], m["topic"], m["content"], m.get("project"), m.get("embedding"))
        )
    conn.commit()
    conn.close()


#TAG: [35A5] 2026-04-05
# Verifies: backfill generates embeddings for all NULL-embedding memories, stores blobs, and commits
def test_backfill_embeddings_behavioural(backfill_env, capsys):
    db_path = backfill_env
    _seed_memories(db_path, [
        {"id": 1, "type": "fact", "topic": "t1", "content": "content one", "project": "proj"},
        {"id": 2, "type": "decision", "topic": "t2", "content": "content two", "project": None},
        {"id": 3, "type": "skill", "topic": "t3", "content": "content three", "project": "proj"},
    ])

    blob_map = {1: b"\x01", 2: b"\x02", 3: b"\x03"}
    call_order = []

    mock_emb = MagicMock()
    mock_emb.embed.side_effect = lambda text: (call_order.append(text), [0.1])[1]
    mock_emb.to_blob.side_effect = lambda v: blob_map[len(call_order)]

    import sys as _sys
    import cairn as _cairn_pkg
    with patch.dict(_sys.modules, {"cairn.embeddings": mock_emb}), \
         patch.object(_cairn_pkg, "embeddings", mock_emb):
        query.backfill_embeddings()

    # All 3 should have been processed
    assert mock_emb.embed.call_count == 3
    assert mock_emb.to_blob.call_count == 3
    assert mock_emb.upsert_vec_index.call_count == 3

    # Verify DB has no NULL embeddings left and blobs match
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, embedding FROM memories ORDER BY id").fetchall()
    conn.close()
    assert rows[0][1] == b"\x01"
    assert rows[1][1] == b"\x02"
    assert rows[2][1] == b"\x03"

    output = capsys.readouterr().out
    done_lines = [l for l in output.strip().split("\n") if "generated" in l]
    assert len(done_lines) == 1
    assert done_lines[0] == "Done. 3 embeddings generated."


#TAG: [0953] 2026-04-05
# Verifies: when all memories already have embeddings, prints message and makes no DB writes
def test_backfill_embeddings_edge_all_have_embeddings(backfill_env, capsys):
    db_path = backfill_env
    _seed_memories(db_path, [
        {"id": 1, "type": "fact", "topic": "t1", "content": "c1", "embedding": b"\x99"},
        {"id": 2, "type": "fact", "topic": "t2", "content": "c2", "embedding": b"\xAA"},
    ])

    mock_emb = MagicMock()
    import sys as _sys
    import cairn as _cairn_pkg
    with patch.dict(_sys.modules, {"cairn.embeddings": mock_emb}), \
         patch.object(_cairn_pkg, "embeddings", mock_emb):
        query.backfill_embeddings()

    # embed should never be called — nothing to backfill
    assert mock_emb.embed.call_count == 0
    assert mock_emb.to_blob.call_count == 0

    output = capsys.readouterr().out
    assert output.strip() == "All memories have embeddings."


#TAG: [B5FC] 2026-04-05
# Verifies: when embeddings module is not importable, prints error and returns without DB access
def test_backfill_embeddings_error_import_failure(backfill_env, capsys):
    db_path = backfill_env
    _seed_memories(db_path, [{"id": 1, "type": "fact", "topic": "t", "content": "c"}])

    # Make 'from cairn import embeddings' raise ImportError by setting module to None
    import sys as _sys
    import cairn as _cairn_pkg
    _saved = getattr(_cairn_pkg, "embeddings", None)
    with patch.dict(_sys.modules, {"cairn.embeddings": None}):
        delattr(_cairn_pkg, "embeddings") if hasattr(_cairn_pkg, "embeddings") else None
        try:
            query.backfill_embeddings()
        finally:
            if _saved is not None:
                _cairn_pkg.embeddings = _saved

    output = capsys.readouterr().out
    assert output.strip() == "sentence-transformers not available"

    # DB should be untouched — embedding still NULL
    conn = sqlite3.connect(db_path)
    null_count = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL").fetchone()[0]
    conn.close()
    assert null_count == 1


#TAG: [BB4E] 2026-04-05
# Verifies: if embed() raises mid-backfill, earlier rows are NOT committed (transaction atomicity)
def test_backfill_embeddings_adversarial_partial_failure(backfill_env):
    db_path = backfill_env
    _seed_memories(db_path, [
        {"id": 1, "type": "fact", "topic": "t1", "content": "c1"},
        {"id": 2, "type": "fact", "topic": "t2", "content": "c2"},
        {"id": 3, "type": "fact", "topic": "t3", "content": "c3"},
    ])

    call_count = [0]

    def embed_then_crash(text):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise RuntimeError("model crashed on second embed")
        return [0.5]

    mock_emb = MagicMock()
    mock_emb.embed.side_effect = embed_then_crash
    mock_emb.to_blob.return_value = b"\xBB"

    import sys as _sys
    import cairn as _cairn_pkg
    with patch.dict(_sys.modules, {"cairn.embeddings": mock_emb}), \
         patch.object(_cairn_pkg, "embeddings", mock_emb):
        with pytest.raises(RuntimeError, match="model crashed"):
            query.backfill_embeddings()

    # Since commit happens AFTER the loop, a crash mid-loop means no commit.
    conn = sqlite3.connect(db_path)
    null_count = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL").fetchone()[0]
    conn.close()
    # All 3 should still be NULL because conn.commit() was never reached
    assert null_count == 3


#TAG: [1705] 2026-04-05
# Verifies: project prefix is prepended to search_text when project is non-NULL
@pytest.mark.behavioural
def test_backfill_embeddings_project_prefix_included(backfill_env, capsys):
    db_path = backfill_env
    _seed_memories(db_path, [
        {"id": 1, "type": "fact", "topic": "auth", "content": "use JWT", "project": "myproj"},
    ])

    mock_emb = MagicMock()
    mock_emb.embed.return_value = [0.1]
    mock_emb.to_blob.return_value = b"\xDD"

    import sys as _sys
    import cairn as _cairn_pkg
    with patch.dict(_sys.modules, {"cairn.embeddings": mock_emb}), \
         patch.object(_cairn_pkg, "embeddings", mock_emb):
        query.backfill_embeddings()

    # Check the search_text passed to embed has project prefix and correct structure
    embed_arg = mock_emb.embed.call_args[0][0]
    assert embed_arg == "myproj fact auth use JWT"


#TAG: [D5DC] 2026-04-05
# Verifies: when project is NULL, search_text has no leading space or prefix
@pytest.mark.behavioural
def test_backfill_embeddings_no_project_prefix(backfill_env, capsys):
    db_path = backfill_env
    _seed_memories(db_path, [
        {"id": 1, "type": "decision", "topic": "db", "content": "use sqlite", "project": None},
    ])

    mock_emb = MagicMock()
    mock_emb.embed.return_value = [0.1]
    mock_emb.to_blob.return_value = b"\xDD"

    import sys as _sys
    import cairn as _cairn_pkg
    with patch.dict(_sys.modules, {"cairn.embeddings": mock_emb}), \
         patch.object(_cairn_pkg, "embeddings", mock_emb):
        query.backfill_embeddings()

    embed_arg = mock_emb.embed.call_args[0][0]
    assert embed_arg == "decision db use sqlite"
