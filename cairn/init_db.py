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
    # metrics, hook_state, pair_assessments live in cairn-ephemeral.db (see
    # init_ephemeral). Keeping high-frequency writers off the durable file
    # contains corruption blast radius — they corrupt first under concurrent
    # WAL writers and would cascade to memories if hosted here.
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
    # pair_assessments lives in cairn-ephemeral.db (see init_ephemeral)
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
    # === Schema v4: multi-node sync ===
    # New columns on memories — sync provenance and ordering
    for col, coltype in [
        ("created_by_node", "TEXT"),
        ("updated_by_node", "TEXT"),
        ("lamport", "INTEGER DEFAULT 0"),
        ("visibility", "TEXT DEFAULT 'team'"),
        ("embedding_model_version", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_lamport ON memories(lamport)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_visibility ON memories(visibility)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_by_node ON memories(created_by_node)")
    # node_state — single-row-per-key store for node identity, lamport clock, embedding model version
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # confidence_log — convergent counter for multi-node confidence accumulation.
    # confidence column on memories is recomputed deterministically from these entries.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS confidence_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            log_uuid      TEXT NOT NULL UNIQUE,
            memory_origin TEXT NOT NULL,
            direction     TEXT NOT NULL,
            reason        TEXT,
            node_id       TEXT NOT NULL,
            user_id       TEXT,
            session_id    TEXT,
            lamport       INTEGER NOT NULL,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conf_log_memory ON confidence_log(memory_origin)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conf_log_lamport ON confidence_log(lamport)")
    # sync_peers — outbound peer registry; bearer_token stored locally only (never synced)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_peers (
            peer_node_id     TEXT PRIMARY KEY,
            url              TEXT NOT NULL,
            bearer_token     TEXT NOT NULL,
            label            TEXT,
            schema_version   INTEGER,
            include_excerpts INTEGER DEFAULT 0,
            last_attempted_at TEXT,
            last_succeeded_at TEXT,
            last_error       TEXT
        )
    """)
    # sync_state — per-(peer, source-node) high-water lamport; the vector clock that makes
    # pull-based gossip converge in O(N) bandwidth.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_state (
            peer_node_id    TEXT NOT NULL,
            source_node_id  TEXT NOT NULL,
            last_lamport    INTEGER NOT NULL,
            updated_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (peer_node_id, source_node_id)
        )
    """)
    # memory_history needs a global UUID so re-anchoring on receipt is unambiguous
    try:
        conn.execute("ALTER TABLE memory_history ADD COLUMN history_uuid TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE memory_history ADD COLUMN memory_origin TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE memory_history ADD COLUMN changed_by_node TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE memory_history ADD COLUMN lamport INTEGER")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_uuid ON memory_history(history_uuid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_memory_origin ON memory_history(memory_origin)")
    # memory_annotation_log mirrors confidence_log during transition; add UUIDs for sync
    try:
        conn.execute("ALTER TABLE memory_annotation_log ADD COLUMN annotation_uuid TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE memory_annotation_log ADD COLUMN node_id TEXT")
    except sqlite3.OperationalError:
        pass
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
    # Source excerpt snapshots — preserves transcript context for --context recovery after JSONL purge
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_source_excerpt (
            memory_id INTEGER PRIMARY KEY,
            session_id TEXT,
            transcript_path TEXT,
            excerpt TEXT NOT NULL,
            context_before TEXT,
            context_after TEXT,
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source_excerpt_session ON memory_source_excerpt(session_id)")
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
    # Dependency graph — stores import/inheritance/include edges from repo ingestion
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            target TEXT NOT NULL,
            kind TEXT NOT NULL,
            names TEXT,
            line INTEGER,
            project TEXT,
            session_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_file)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_project ON memory_relations(project)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_kind ON memory_relations(kind)")
    if not conn.execute("SELECT 1 FROM schema_version WHERE version = 3").fetchone():
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (3, 'null_embedding_on_content_edit trigger — invalidates stale embeddings when content changes without re-embedding')"
        )
    if not conn.execute("SELECT 1 FROM schema_version WHERE version = 4").fetchone():
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (4, 'memory_relations table — dependency graph edges from tree-sitter AST parsing')"
        )
    # calibration_rows — store for *how to interact with this user* (style,
    # level, preferences). Populated by the session analyser; consumed by
    # the UserPromptSubmit hook to inject <calibration_profile> blocks.
    # Sibling system to memories: complements *what* knowledge with *how*
    # behavioural priming. See docs/spec-calibration-system.md.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            kw TEXT,
            qf TEXT,
            source TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            pinned INTEGER NOT NULL DEFAULT 0,
            layer TEXT NOT NULL DEFAULT 'subject',
            session_scope TEXT,
            superseded_by INTEGER REFERENCES calibration_rows(id),
            archived_at TIMESTAMP,
            archive_reason TEXT,
            delivered_count INTEGER NOT NULL DEFAULT 0,
            followed_count INTEGER NOT NULL DEFAULT 0,
            ignored_count INTEGER NOT NULL DEFAULT 0,
            corrected_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            embedding BLOB
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_rows_source ON calibration_rows(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_rows_layer ON calibration_rows(layer)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_rows_archived ON calibration_rows(archived_at)")
    if not conn.execute("SELECT 1 FROM schema_version WHERE version = 5").fetchone():
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (5, 'calibration_rows table — calibration system Phase 1 foundation')"
        )
    # v6 — origin_session_id on calibration_rows so the row's analyser
    # session can be cross-linked from the dashboard popup.
    try:
        conn.execute("ALTER TABLE calibration_rows ADD COLUMN origin_session_id TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_rows_origin_session ON calibration_rows(origin_session_id)")
    if not conn.execute("SELECT 1 FROM schema_version WHERE version = 6").fetchone():
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (6, 'calibration_rows.origin_session_id for source-session cross-link')"
        )
    # v7 — per-qf sidecar embeddings. Each qf string is embedded individually
    # so retrieval can score row as max_i cos(prompt_embedding, qf_i_embedding).
    # Fixes the conflated first-person/third-person embedding problem where
    # row.embedding (built from content+kw+qf joined) clustered prompt
    # similarities at 0.20-0.36, below any sensible floor.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS calibration_qf_embeddings (
            row_id INTEGER NOT NULL,
            qf_index INTEGER NOT NULL,
            qf_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (row_id, qf_index),
            FOREIGN KEY (row_id) REFERENCES calibration_rows(id) ON DELETE CASCADE
        )
    ''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_qf_row ON calibration_qf_embeddings(row_id)")
    if not conn.execute("SELECT 1 FROM schema_version WHERE version = 7").fetchone():
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (7, 'calibration_qf_embeddings sidecar table — per-qf symmetric intent retrieval')"
        )
    # v8 — dual embedding for memories. topic_embedding column stores the
    # topic field's embedding separately from the row''s main embedding (which
    # is content+kw+project mashup). Retrieval scores row as
    # max(cos(prompt, embedding), cos(prompt, topic_embedding)) — empirically
    # lifts top-K candidate count by ~189% on prompt-shaped short queries
    # against the existing memory corpus (audit 2026-05-28).
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "topic_embedding" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN topic_embedding BLOB")
    if not conn.execute("SELECT 1 FROM schema_version WHERE version = 8").fetchone():
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (8, 'memories.topic_embedding column — dual embedding for symmetric topic retrieval')"
        )
            "INSERT INTO schema_version (version, description) VALUES (4, 'multi-node sync: created_by_node, updated_by_node, lamport, visibility, embedding_model_version + node_state, confidence_log, sync_peers, sync_state tables; memory_history.history_uuid')"
        )
    # v4 backfill — runs once per DB. Idempotent: guarded by NULL checks.
    # Lazy-import to avoid a hard cycle (cairn.sync.identity imports nothing from init_db).
    try:
        from cairn.sync.identity import ensure_node_id, get_embedding_model_version
        node_id = ensure_node_id()
        model_version = get_embedding_model_version()
    except Exception:
        # If sync identity module isn't installed yet (e.g. partial install), defer backfill.
        # Next init() call will complete it.
        node_id = None
        model_version = None
    if node_id:
        rows_no_node = conn.execute(
            "SELECT id FROM memories WHERE created_by_node IS NULL"
        ).fetchall()
        if rows_no_node:
            for (mem_id,) in rows_no_node:
                conn.execute(
                    "UPDATE memories SET created_by_node = ?, updated_by_node = ?, "
                    "lamport = CASE WHEN lamport IS NULL OR lamport = 0 THEN id ELSE lamport END, "
                    "visibility = COALESCE(visibility, 'team'), "
                    "embedding_model_version = COALESCE(embedding_model_version, ?) WHERE id = ?",
                    (node_id, node_id, model_version, mem_id)
                )
            print(f"v4 backfill: tagged {len(rows_no_node)} memories with node_id={node_id[:8]}…")
        # Backfill memory_history.history_uuid for rows missing it (synced unit of append)
        hist_no_uuid = conn.execute(
            "SELECT id, memory_id FROM memory_history WHERE history_uuid IS NULL"
        ).fetchall()
        if hist_no_uuid:
            for (hist_id, mem_id) in hist_no_uuid:
                # Look up origin_id of the parent memory
                origin = conn.execute("SELECT origin_id FROM memories WHERE id = ?", (mem_id,)).fetchone()
                conn.execute(
                    "UPDATE memory_history SET history_uuid = ?, memory_origin = ?, changed_by_node = ?, lamport = COALESCE(lamport, ?) WHERE id = ?",
                    (str(uuid.uuid4()), origin[0] if origin else None, node_id, hist_id, hist_id)
                )
            print(f"v4 backfill: tagged {len(hist_no_uuid)} memory_history rows")
        # Initialize node lamport clock to current max so subsequent local edits
        # are causally after observed history
        max_lamport = conn.execute("SELECT COALESCE(MAX(lamport), 0) FROM memories").fetchone()[0]
        existing = conn.execute("SELECT value FROM node_state WHERE key = 'lamport'").fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO node_state (key, value) VALUES ('lamport', ?)",
                (str(max_lamport),)
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
    # Pending memory writes — Stop hook enqueues parsed memory entries here and
    # exits fast. A drain worker (hooks/drain_queue.py) holds an exclusive flock
    # and processes rows in seq order via storage.insert_memories. Keeps the
    # synchronous hook critical path off the slow embed+dedup pipeline.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_writes (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            session_id TEXT,
            transcript_path TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_writes_seq ON pending_writes(seq)")
    # Migrate existing DBs that pre-date attempts/last_error columns
    for col, coltype in [("attempts", "INTEGER NOT NULL DEFAULT 0"), ("last_error", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE pending_writes ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass
    # calibration_deliveries — turn-indexed log of which calibration_rows
    # were injected into which UserPromptSubmit turn. High-frequency
    # write path (every user prompt), so lives in ephemeral alongside
    # metrics/hook_state. The analyser joins this against subsequent
    # turns to score outcome (followed / ignored / corrected).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            row_id INTEGER NOT NULL,
            delivered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            similarity REAL,
            outcome TEXT,
            outcome_evidence TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_deliv_session ON calibration_deliveries(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_deliv_row ON calibration_deliveries(row_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_deliv_outcome ON calibration_deliveries(outcome)")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init()
    init_ephemeral()
