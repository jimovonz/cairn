#!/usr/bin/env python3
"""Initialize the Cairn SQLite database."""

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import os
import uuid

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")

def init():
    conn = sqlite3.connect(DB_PATH)
    # Enable WAL mode for concurrent access (persistent across connections)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migration: add columns to existing DB
    for col, coltype in [("embedding", "BLOB"), ("session_id", "TEXT"), ("project", "TEXT"), ("confidence", "REAL DEFAULT 0.7"), ("source_start", "INTEGER"), ("source_end", "INTEGER"), ("archived_reason", "TEXT"), ("anchor_line", "INTEGER"), ("depth", "INTEGER"), ("associated_files", "TEXT"), ("keywords", "TEXT"), ("origin_id", "TEXT"), ("user_id", "TEXT"), ("updated_by", "TEXT"), ("team_id", "TEXT"), ("source_ref", "TEXT"), ("deleted_at", "TIMESTAMP"), ("synced_at", "TIMESTAMP")]:
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)")
    # Version history — preserves old content when memories are updated
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            session_id TEXT,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_history_memory_id ON memory_history(memory_id)
    """)
    # Session tracking — links sessions into chains across compaction
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            project TEXT,
            transcript_path TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_session_id) REFERENCES sessions(session_id)
        )
    """)
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN project TEXT")
    except sqlite3.OperationalError:
        pass
    # Performance metrics
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            session_id TEXT,
            detail TEXT,
            value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_event ON metrics(event)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id)")
    # Hook state — replaces file-based state for atomicity under concurrent access
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hook_state (
            session_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, key)
        )
    """)
    # Correction triggers — behavioural pattern detection
    conn.execute("""
        CREATE TABLE IF NOT EXISTS correction_triggers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            trigger TEXT NOT NULL,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_triggers_memory ON correction_triggers(memory_id)")
    # Vector search index via sqlite-vec
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                memory_id INTEGER PRIMARY KEY,
                embedding float[384]
            )
        """)
    except (ImportError, Exception) as e:
        pass  # sqlite-vec not available, brute-force fallback will be used
    # Trigger: snapshot old content before update
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_version BEFORE UPDATE OF content ON memories BEGIN
            INSERT INTO memory_history (memory_id, content, session_id, changed_at)
            VALUES (old.id, old.content, old.session_id, old.updated_at);
        END
    """)
    # Trigger: null embedding when content changes without a new embedding being set.
    # insert_memories always sets a fresh embedding blob alongside content, so
    # NEW.embedding != OLD.embedding and this trigger skips. Other update paths
    # (query.py, consolidate.py) don't touch embedding, so it fires.
    conn.execute("DROP TRIGGER IF EXISTS null_embedding_on_content_edit")
    conn.execute("""
        CREATE TRIGGER null_embedding_on_content_edit
        AFTER UPDATE OF content ON memories
        FOR EACH ROW
        WHEN NEW.content != OLD.content AND NEW.embedding IS OLD.embedding
        BEGIN
            UPDATE memories SET embedding = NULL WHERE id = NEW.id;
        END
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)
    """)
    # FTS5 index — migrate from (topic, content) to (topic, content, keywords)
    # Check if existing FTS table needs migration by inspecting its columns
    _fts_needs_rebuild = False
    try:
        # If keywords column doesn't exist in FTS, we need to rebuild
        conn.execute("SELECT keywords FROM memories_fts LIMIT 0")
    except sqlite3.OperationalError:
        # Column missing or table doesn't exist — rebuild
        _fts_needs_rebuild = True

    if _fts_needs_rebuild:
        # Drop old FTS table and triggers, recreate with keywords
        conn.execute("DROP TRIGGER IF EXISTS memories_ai")
        conn.execute("DROP TRIGGER IF EXISTS memories_ad")
        conn.execute("DROP TRIGGER IF EXISTS memories_au")
        conn.execute("DROP TABLE IF EXISTS memories_fts")

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            topic, content, keywords, content=memories, content_rowid=id
        )
    """)
    # Triggers to keep FTS in sync
    conn.execute("DROP TRIGGER IF EXISTS memories_ai")
    conn.execute("""
        CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, topic, content, keywords)
            VALUES (new.id, new.topic, new.content, new.keywords);
        END
    """)
    conn.execute("DROP TRIGGER IF EXISTS memories_ad")
    conn.execute("""
        CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, topic, content, keywords)
            VALUES ('delete', old.id, old.topic, old.content, old.keywords);
        END
    """)
    conn.execute("DROP TRIGGER IF EXISTS memories_au")
    conn.execute("""
        CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, topic, content, keywords)
            VALUES ('delete', old.id, old.topic, old.content, old.keywords);
            INSERT INTO memories_fts(rowid, topic, content, keywords)
            VALUES (new.id, new.topic, new.content, new.keywords);
        END
    """)
    # Pair assessment cache — records which memory pairs have been assessed
    # for consolidation/contradiction so incremental runs skip already-checked pairs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pair_assessments (
            memory_id_a INTEGER NOT NULL,
            memory_id_b INTEGER NOT NULL,
            mode TEXT NOT NULL,
            verdict TEXT NOT NULL,
            reason TEXT,
            assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (memory_id_a, memory_id_b, mode)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pair_mode ON pair_assessments(mode)")
    # Rebuild FTS index if we migrated
    if _fts_needs_rebuild:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        print("FTS index rebuilt with keywords column")
    # Schema version metadata — DB-level compatibility tracking for sync
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
    """)
    # Record current schema version if not already present
    if not conn.execute("SELECT 1 FROM schema_version WHERE version = 2").fetchone():
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (2, 'multi-user and sync columns: origin_id, user_id, updated_by, team_id, source_ref, deleted_at, synced_at')"
        )
    # Annotation audit trail — records every confidence update event
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_annotation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            direction TEXT NOT NULL,
            reason TEXT,
            session_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ann_memory ON memory_annotation_log(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ann_session ON memory_annotation_log(session_id)")
    if not conn.execute("SELECT 1 FROM schema_version WHERE version = 3").fetchone():
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (3, 'null_embedding_on_content_edit trigger — invalidates stale embeddings when content changes without re-embedding')"
        )
    # Indexes for new columns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_origin_id ON memories(origin_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_deleted_at ON memories(deleted_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_synced_at ON memories(synced_at)")
    # Backfill origin_id for existing rows that don't have one
    rows_without_origin = conn.execute(
        "SELECT id FROM memories WHERE origin_id IS NULL"
    ).fetchall()
    if rows_without_origin:
        for (mem_id,) in rows_without_origin:
            conn.execute(
                "UPDATE memories SET origin_id = ? WHERE id = ?",
                (str(uuid.uuid4()), mem_id)
            )
        print(f"Backfilled origin_id for {len(rows_without_origin)} existing memories")
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def init_ephemeral(path=None):
    """Initialize the ephemeral DB (metrics, hook_state, pair_assessments)."""
    if path is None:
        from cairn.config import EPHEMERAL_DB_PATH
        path = EPHEMERAL_DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            session_id TEXT,
            detail TEXT,
            value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_event ON metrics(event)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hook_state (
            session_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, key)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pair_assessments (
            memory_id_a INTEGER NOT NULL,
            memory_id_b INTEGER NOT NULL,
            mode TEXT NOT NULL,
            verdict TEXT NOT NULL,
            reason TEXT,
            assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (memory_id_a, memory_id_b, mode)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pair_mode ON pair_assessments(mode)")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init()
