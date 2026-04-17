#!/usr/bin/env python3
"""Tests for trailing intent detection — blocks responses that end with
unfulfilled action intent (e.g. 'let me test that') and allows FINISHED escape."""

import sys
import os
import json
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO


import hooks.hook_helpers as hook_helpers
import hooks.enforcement as enforcement
import hooks.storage as storage

TEST_DIR = tempfile.mkdtemp()
_counter = [0]


HOOK_STATE_SQL = """CREATE TABLE IF NOT EXISTS hook_state (
    session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, key))"""


def fresh_env():
    _counter[0] += 1
    n = _counter[0]
    db_path = os.path.join(TEST_DIR, f"intent_{n}.db")
    conn = sqlite3.connect(db_path)
    for sql in [
        """CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
            embedding BLOB, session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
            source_start INTEGER, source_end INTEGER, anchor_line INTEGER, depth INTEGER, archived_reason TEXT, keywords TEXT,
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
            topic, content, keywords, content=memories, content_rowid=id)""",
        """CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, topic, content, keywords) VALUES (new.id, new.topic, new.content, new.keywords); END""",
    ]:
        conn.execute(sql)
    conn.commit()
    return db_path, conn


def run_hook(db_path, payload):
    """Run stop_hook.main() with full patching."""
    import hooks.hook_helpers as hook_helpers
    import hooks.stop_hook as stop_hook
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
        hook_helpers.LOG_PATH = os.path.join(TEST_DIR, f'intent_{_counter[0]}.log')

        with patch('sys.stdin', StringIO(json.dumps(payload))), \
             patch('sys.stdout', captured), \
             patch('sys.exit', mock_exit):
            try:
                stop_hook.main()
            except SystemExit:
                pass
            finally:
                hook_helpers.flush_metrics()
    finally:
        for k, v in orig.items():
            setattr(hook_helpers, k, v)

    output = captured.getvalue()
    result = json.loads(output) if output.strip() else None
    return exit_code[0], result


# --- Unit tests for extraction and detection ---

class TestExtractTailSentences:
    # Verifies: memory block is stripped before sentence extraction
    def test_strips_memory_block(self):
        text = "Here is my answer.\n<memory>\n- type: fact\n- topic: test\n- content: test\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
        result = enforcement._extract_tail_sentences(text)
        assert result == ["Here is my answer"]

    # Verifies: trailing code fences are stripped from extraction
    def test_strips_trailing_code_fence(self):
        text = "Some code here.\n```\ncode block\n```"
        result = enforcement._extract_tail_sentences(text)
        assert len(result) > 0
        assert all("```" not in s for s in result)

    # Verifies: empty/whitespace input returns empty list
    def test_returns_empty_for_empty(self):
        assert enforcement._extract_tail_sentences("") == []
        assert enforcement._extract_tail_sentences("   ") == []

    # Verifies: short fragments are skipped (returns empty list)
    def test_skips_short_fragments(self):
        text = "Done. Ok. Yes."
        assert enforcement._extract_tail_sentences(text) == []

    # Verifies: last meaningful sentence is correctly extracted
    def test_extracts_last_meaningful_sentence(self):
        text = "First I did this. Then I checked the logs. Let me investigate the error further."
        result = enforcement._extract_tail_sentences(text, n=1)
        assert result == ["Let me investigate the error further"], \
            f"Expected exact last sentence (period stripped); got: {result!r}"

    # Verifies: returns up to n tail sentences
    def test_returns_multiple_tail_sentences(self):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        result = enforcement._extract_tail_sentences(text, n=3)
        assert len(result) == 3
        assert result[0] == "Second sentence here"
        assert result[-1] == "Fourth sentence here"

    # Verifies: intent in non-final sentence is captured
    def test_captures_intent_in_penultimate(self):
        text = "Let me revise the plan to strip out the old code. The only mechanism that matters is re-ingestion."
        result = enforcement._extract_tail_sentences(text, n=3)
        assert any("revise the plan" in s for s in result)


class TestCheckTrailingIntent:
    """Test the embedding-based intent detection with a real embedder."""

    def _get_embedder(self):
        try:
            import cairn.embeddings as embeddings
            vec = embeddings.embed("test", allow_slow=True)
            if vec is None:
                return None
            return embeddings
        except Exception:
            return None

    # Verifies: obvious action intent ("let me run") is detected
    def test_detects_obvious_intent(self):
        import hooks.stop_hook as stop_hook
        emb = self._get_embedder()
        if emb is None:
            return  # skip if no embedder available
        # Clear cached embeddings so they're recomputed
        enforcement._intent_embeddings = None
        with patch.object(hook_helpers, 'get_embedder', return_value=emb):
            result = enforcement.check_trailing_intent(
                "The config looks fine. Let me run the tests to verify.\n"
                "<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
            )
            assert result is not None, "Should detect 'let me run the tests'"

    # Verifies: clean completion statement is not flagged as intent
    def test_allows_clean_ending(self):
        import hooks.stop_hook as stop_hook
        emb = self._get_embedder()
        if emb is None:
            return
        enforcement._intent_embeddings = None
        with patch.object(hook_helpers, 'get_embedder', return_value=emb):
            result = enforcement.check_trailing_intent(
                "All 171 tests pass and the refactor is complete.\n"
                "<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
            )
            assert result is None, "Should not flag a clean completion statement"

    # Verifies: question to user does not crash intent detection
    def test_allows_question_to_user(self):
        import hooks.stop_hook as stop_hook
        emb = self._get_embedder()
        if emb is None:
            return
        enforcement._intent_embeddings = None
        with patch.object(hook_helpers, 'get_embedder', return_value=emb):
            result = enforcement.check_trailing_intent(
                "The implementation is done. Do you want me to also update the docs?\n"
                "<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
            )
            # Questions to the user are borderline — the key is we don't
            # false-positive on conversational endings
            # This is acceptable either way, but shouldn't crash

    # Verifies: intent in non-final sentence is still detected
    def test_detects_intent_in_penultimate_sentence(self):
        emb = self._get_embedder()
        if emb is None:
            return
        enforcement._intent_embeddings = None
        with patch.object(hook_helpers, 'get_embedder', return_value=emb):
            result = enforcement.check_trailing_intent(
                "Let me revise the plan to strip out the old code. "
                "The only mechanism that matters is re-ingestion.\n"
                "[cm]: # '{\"ok\":true,\"ctx\":\"s\",\"kw\":[\"test\"]}'"
            )
            assert result is not None, "Should detect intent in second-to-last sentence"

    # Verifies: missing embedder gracefully returns None
    def test_no_embedder_returns_none(self):
        import hooks.stop_hook as stop_hook
        enforcement._intent_embeddings = None
        with patch.object(hook_helpers, 'get_embedder', return_value=None):
            result = enforcement.check_trailing_intent(
                "Let me test that now.\n<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
            )
            assert result is None, "Should gracefully return None without embedder"


# --- Integration tests through main() ---

class TestTrailingIntentIntegration:
    # Verifies: response with trailing action intent is blocked
    def test_blocks_trailing_intent(self):
        """Response ending with action intent should be blocked."""
        db_path, conn = fresh_env()
        conn.close()

        # Mock embedder that returns distinguishable vectors
        mock_emb = MagicMock()
        intent_vec = np.random.randn(384).astype(np.float32)
        intent_vec /= np.linalg.norm(intent_vec)
        mock_emb.embed.return_value = intent_vec  # same vector = high similarity
        mock_emb.cosine_similarity.return_value = 0.9

        import hooks.stop_hook as stop_hook
        enforcement._intent_embeddings = None

        payload = {
            "session_id": "test-intent-1",
            "last_assistant_message": (
                "The config looks correct.\nLet me run the tests to check.\n"
                "<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
            ),
        }

        with patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
            code, result = run_hook(db_path, payload)

        assert result is not None, "Should have blocked"
        assert result["decision"] == "block"
        assert "intent: resolved" in result["reason"]

    # Verifies: intent:resolved in memory block bypasses intent check
    def test_intent_resolved_allows_stop(self):
        """Memory block with intent: resolved should skip trailing intent check."""
        db_path, conn = fresh_env()
        conn.close()

        # Even with a mock that would flag intent, resolved should bypass
        mock_emb = MagicMock()
        mock_emb.embed.return_value = np.random.randn(384).astype(np.float32)
        mock_emb.cosine_similarity.return_value = 0.9  # would normally trigger

        import hooks.stop_hook as stop_hook
        enforcement._intent_embeddings = None

        payload = {
            "session_id": "test-intent-2",
            "last_assistant_message": (
                "I considered running the tests but they aren't needed here.\n"
                "<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n- intent: resolved\n</memory>"
            ),
        }

        with patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
            code, result = run_hook(db_path, payload)

        assert result is None, "intent: resolved should allow stop"

    # Verifies: normal response without intent passes through
    def test_clean_response_passes(self):
        """Normal response without trailing intent should pass."""
        db_path, conn = fresh_env()
        conn.close()

        mock_emb = MagicMock()
        mock_emb.embed.return_value = np.random.randn(384).astype(np.float32)
        mock_emb.cosine_similarity.return_value = 0.2  # low similarity

        import hooks.stop_hook as stop_hook
        enforcement._intent_embeddings = None

        payload = {
            "session_id": "test-intent-3",
            "last_assistant_message": (
                "All done, the refactor is complete.\n"
                "<memory>\n- complete: true\n- context: sufficient\n- keywords: test\n</memory>"
            ),
        }

        with patch.object(hook_helpers, 'get_embedder', return_value=mock_emb):
            code, result = run_hook(db_path, payload)

        assert result is None, "Clean response should pass through"


# --- Content quality gate tests ---

class TestContentQualityGate:
    # Verifies: empty, short, and whitespace strings are classified as empty memories
    def test_rejects_empty_content(self):
        import hooks.stop_hook as stop_hook
        assert storage._is_empty_memory("") is True
        assert storage._is_empty_memory("short") is True
        assert storage._is_empty_memory("   ") is True

    # Verifies: known no-info patterns ("no context available" etc.) are classified as empty
    def test_rejects_no_context_patterns(self):
        import hooks.stop_hook as stop_hook
        assert storage._is_empty_memory("User asked what was on their lawn - no context available") is True
        assert storage._is_empty_memory("No relevant context for this question") is True
        assert storage._is_empty_memory("Unable to determine what the user meant") is True
        assert storage._is_empty_memory("no information available about this topic") is True

    # Verifies: substantive content strings are correctly classified as non-empty
    def test_allows_real_content(self):
        import hooks.stop_hook as stop_hook
        assert storage._is_empty_memory("User observed a pūkeko on their lawn") is False
        assert storage._is_empty_memory("Changed install.sh to overwrite global CLAUDE.md on re-install") is False
        assert storage._is_empty_memory("WAL mode enabled for concurrent session safety") is False

    # Verifies: empty memories are filtered out during insert_memories and not stored in DB
    def test_rejects_in_insert(self):
        """Empty memories should not be inserted into the database."""
        db_path, conn = fresh_env()
        conn.execute("INSERT INTO sessions (session_id, project) VALUES ('s1', 'P')")
        conn.commit()

        import hooks.stop_hook as stop_hook
        with patch.object(hook_helpers, 'DB_PATH', db_path), \
             patch.object(hook_helpers, 'LOG_PATH', os.path.join(TEST_DIR, 'quality.log')), \
             patch.object(hook_helpers, 'get_embedder', return_value=None):
            count = storage.insert_memories([
                {"type": "fact", "topic": "good", "content": "User prefers dark mode in all editors"},
                {"type": "fact", "topic": "bad", "content": "no context available for this question"},
                {"type": "fact", "topic": "empty", "content": ""},
            ], session_id="s1")

        assert count == 1, "Only the good memory should be inserted"
        row = conn.execute("SELECT topic FROM memories").fetchone()
        assert row[0] == "good"
