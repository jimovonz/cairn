"""Tests for incremental ingestion — fingerprinting, diffing, selective archival."""

import json
import os
import sqlite3
import tempfile

import pytest

# Patch DB_PATH before importing ingest
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()

import cairn.ingest as ingest

ingest.DB_PATH = _tmp_db.name


@pytest.fixture(autouse=True)
def fresh_db():
    """Reset the test database before each test."""
    if os.path.exists(ingest.DB_PATH):
        os.unlink(ingest.DB_PATH)
    conn = sqlite3.connect(ingest.DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT, topic TEXT, content TEXT, embedding BLOB,
            session_id TEXT, project TEXT, confidence REAL DEFAULT 0.7,
            origin_id TEXT, source_ref TEXT, keywords TEXT, depth INTEGER,
            associated_files TEXT, archived_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            parent_session_id TEXT, project TEXT,
            transcript_path TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    yield
    if os.path.exists(ingest.DB_PATH):
        os.unlink(ingest.DB_PATH)


# --- Fingerprinting ---

class TestFingerprinting:
    def test_same_data_same_fingerprint(self):
        data = [{"file": "README.md", "content": "hello"}]
        fp1 = ingest._fingerprint_section("docs", data)
        fp2 = ingest._fingerprint_section("docs", data)
        assert fp1 == fp2

    def test_different_data_different_fingerprint(self):
        fp1 = ingest._fingerprint_section("docs", [{"file": "a.md", "content": "x"}])
        fp2 = ingest._fingerprint_section("docs", [{"file": "a.md", "content": "y"}])
        assert fp1 != fp2

    def test_different_section_name_different_fingerprint(self):
        data = [{"file": "a.md", "content": "x"}]
        fp1 = ingest._fingerprint_section("docs", data)
        fp2 = ingest._fingerprint_section("config", data)
        assert fp1 != fp2

    def test_version_change_changes_fingerprint(self):
        data = [{"file": "a.md", "content": "x"}]
        fp1 = ingest._fingerprint_section("docs", data)
        old_ver = ingest.EXTRACTOR_VERSIONS["docs"]
        ingest.EXTRACTOR_VERSIONS["docs"] = 99
        fp2 = ingest._fingerprint_section("docs", data)
        ingest.EXTRACTOR_VERSIONS["docs"] = old_ver
        assert fp1 != fp2

    def test_compute_fingerprints_all_sections(self):
        extractions = {
            "docs": [{"file": "README.md", "content": "hello"}],
            "tree": ["src/", "lib/"],
            "config": [],
        }
        fps = ingest.compute_fingerprints(extractions)
        assert set(fps.keys()) == {"docs", "tree", "config"}
        assert all(len(v) == 64 for v in fps.values())  # SHA256 hex


# --- Diff ---

class TestDiffSections:
    def test_no_changes(self):
        fps = {"docs": "abc", "tree": "def"}
        assert ingest.diff_sections(fps, fps) == set()

    def test_new_section(self):
        current = {"docs": "abc", "tree": "def", "config": "ghi"}
        cached = {"docs": "abc", "tree": "def"}
        assert ingest.diff_sections(current, cached) == {"config"}

    def test_changed_section(self):
        current = {"docs": "abc", "tree": "CHANGED"}
        cached = {"docs": "abc", "tree": "def"}
        assert ingest.diff_sections(current, cached) == {"tree"}

    def test_empty_cache_all_changed(self):
        current = {"docs": "abc", "tree": "def"}
        assert ingest.diff_sections(current, {}) == {"docs", "tree"}

    def test_removed_section_not_in_changed(self):
        current = {"docs": "abc"}
        cached = {"docs": "abc", "tree": "def"}
        assert ingest.diff_sections(current, cached) == set()


# --- Fingerprint persistence ---

class TestFingerprintPersistence:
    def test_store_and_retrieve(self):
        fps = {"docs": "abc123", "tree": "def456"}
        ingest.store_fingerprints("myproject", fps, "ingest-test-001")
        loaded = ingest.get_cached_fingerprints("myproject")
        assert loaded == fps

    def test_overwrite_on_re_store(self):
        ingest.store_fingerprints("myproject", {"docs": "old"}, "s1")
        ingest.store_fingerprints("myproject", {"docs": "new", "tree": "x"}, "s2")
        loaded = ingest.get_cached_fingerprints("myproject")
        assert loaded == {"docs": "new", "tree": "x"}

    def test_project_isolation(self):
        ingest.store_fingerprints("proj_a", {"docs": "aaa"}, "s1")
        ingest.store_fingerprints("proj_b", {"docs": "bbb"}, "s2")
        assert ingest.get_cached_fingerprints("proj_a") == {"docs": "aaa"}
        assert ingest.get_cached_fingerprints("proj_b") == {"docs": "bbb"}

    def test_empty_cache_returns_empty_dict(self):
        assert ingest.get_cached_fingerprints("nonexistent") == {}


# --- Selective archival in insert_memories ---

class TestSelectiveArchival:
    def _insert_old_memories(self, conn, project, session_id, memories):
        """Helper: insert pre-existing ingestion memories with source_ref sections."""
        for mem in memories:
            src_ref = json.dumps({"repo": "test", "sections": mem["sections"]})
            conn.execute(
                "INSERT INTO memories (type, topic, content, session_id, project, source_ref) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (mem.get("type", "fact"), mem["topic"], mem["content"],
                 session_id, project, src_ref),
            )
        conn.commit()

    def test_full_mode_archives_everything(self):
        conn = sqlite3.connect(ingest.DB_PATH)
        self._insert_old_memories(conn, "proj", "ingest-old", [
            {"topic": "a", "content": "memory about docs" * 3, "sections": ["docs"]},
            {"topic": "b", "content": "memory about config" * 3, "sections": ["config"]},
        ])
        conn.close()

        entries = [{"type": "fact", "topic": "c", "content": "new memory content here which is long enough",
                     "source_sections": ["docs"]}]
        inserted = ingest.insert_memories(entries, "proj", {"path": "/tmp/test"},
                                          session_id="ingest-new", changed_sections=None)
        assert len(inserted) == 1

        conn = sqlite3.connect(ingest.DB_PATH)
        archived = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived_reason IS NOT NULL"
        ).fetchone()[0]
        assert archived == 2  # both old memories archived

    def test_incremental_archives_only_changed_sections(self):
        conn = sqlite3.connect(ingest.DB_PATH)
        self._insert_old_memories(conn, "proj", "ingest-old", [
            {"topic": "a", "content": "memory about docs section" * 3, "sections": ["docs"]},
            {"topic": "b", "content": "memory about config section" * 3, "sections": ["config"]},
            {"topic": "c", "content": "memory about tree structure" * 3, "sections": ["tree"]},
        ])
        conn.close()

        entries = [{"type": "fact", "topic": "d", "content": "new docs memory that is long enough to pass",
                     "source_sections": ["docs"]}]
        inserted = ingest.insert_memories(entries, "proj", {"path": "/tmp/test"},
                                          session_id="ingest-new",
                                          changed_sections={"docs"})
        assert len(inserted) == 1

        conn = sqlite3.connect(ingest.DB_PATH)
        archived = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived_reason IS NOT NULL"
        ).fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived_reason IS NULL OR archived_reason = ''"
        ).fetchone()[0]
        conn.close()
        assert archived == 1  # only docs memory archived
        assert active == 3    # config + tree kept + new docs

    def test_memories_without_sections_get_archived_in_incremental(self):
        """Old memories without section tags get archived (conservative)."""
        conn = sqlite3.connect(ingest.DB_PATH)
        conn.execute(
            "INSERT INTO memories (type, topic, content, session_id, project, source_ref) "
            "VALUES ('fact', 'x', 'old memory without sections tag long enough', 'ingest-old', 'proj', '{}')",
        )
        conn.commit()
        conn.close()

        entries = [{"type": "fact", "topic": "y", "content": "new memory content that is sufficiently long",
                     "source_sections": ["docs"]}]
        inserted = ingest.insert_memories(entries, "proj", {"path": "/tmp/test"},
                                          session_id="ingest-new",
                                          changed_sections={"docs"})

        conn = sqlite3.connect(ingest.DB_PATH)
        archived = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived_reason IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        assert archived == 1  # no sections = gets archived (safe default)

    def test_source_sections_stored_in_source_ref(self):
        entries = [{"type": "fact", "topic": "t", "content": "test content that is long enough to pass gate",
                     "source_sections": ["docs", "config"],
                     "source_files": ["README.md"]}]
        inserted = ingest.insert_memories(entries, "proj", {"path": "/tmp/r"},
                                          session_id="ingest-test")

        conn = sqlite3.connect(ingest.DB_PATH)
        row = conn.execute("SELECT source_ref FROM memories WHERE id = ?",
                           (inserted[0],)).fetchone()
        conn.close()
        ref = json.loads(row[0])
        assert ref["sections"] == ["docs", "config"]


# --- _prepare_extracts_text filtering ---

class TestExtractsFilter:
    def test_no_filter_includes_all(self):
        result = {
            "extractions": {
                "docs": [{"file": "README.md", "content": "hello world"}],
                "tree": ["src/", "lib/"],
            },
            "git": {},
        }
        text = ingest._prepare_extracts_text(result)
        assert "README.md" in text
        assert "Directory tree" in text

    def test_filter_includes_only_matching(self):
        result = {
            "extractions": {
                "docs": [{"file": "README.md", "content": "hello world"}],
                "tree": ["src/", "lib/"],
                "config": [{"file": "config.yml", "content": "key: val"}],
            },
            "git": {},
        }
        text = ingest._prepare_extracts_text(result, sections_filter={"docs"})
        assert "README.md" in text
        assert "Directory tree" not in text
        assert "config.yml" not in text

    def test_filter_empty_set_produces_empty(self):
        result = {
            "extractions": {
                "docs": [{"file": "README.md", "content": "hello"}],
            },
            "git": {},
        }
        text = ingest._prepare_extracts_text(result, sections_filter=set())
        assert text == ""


# --- Dry-run shows incremental info ---

class TestDryRunIncremental:
    def test_dry_run_shows_changed_sections(self, capsys):
        entries = [{"type": "fact", "topic": "t", "content": "content",
                     "source_sections": ["docs"]}]
        result = ingest.insert_memories(entries, "proj", {"path": "/tmp"},
                                        session_id="s1", dry_run=True,
                                        changed_sections={"docs", "tree"})
        assert result == []
        captured = capsys.readouterr()
        assert "Changed sections:" in captured.out
        assert "docs" in captured.out
