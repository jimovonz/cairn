#!/usr/bin/env python3
"""Initialize the Cairn SQLite database."""

import sqlite3
import os

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
    for col, coltype in [("embedding", "BLOB"), ("session_id", "TEXT"), ("project", "TEXT"), ("confidence", "REAL DEFAULT 0.7"), ("source_start", "INTEGER"), ("source_end", "INTEGER")]:
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
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            topic, content, content=memories, content_rowid=id
        )
    """)
    # Triggers to keep FTS in sync
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, topic, content)
            VALUES (new.id, new.topic, new.content);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, topic, content)
            VALUES ('delete', old.id, old.topic, old.content);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, topic, content)
            VALUES ('delete', old.id, old.topic, old.content);
            INSERT INTO memories_fts(rowid, topic, content)
            VALUES (new.id, new.topic, new.content);
        END
    """)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    init()
