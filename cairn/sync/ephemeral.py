"""Attach the per-node ephemeral DB to a sync connection.

The operational sync tables — `discovered_peers` (LAN beacon cache) and
`sync_state` (per-(peer, source-node) pull high-water vector clock) — live in
the ephemeral DB, not the durable memory DB (schema v12). They are high-churn
and fully recoverable, so keeping them off the durable file stops sync from
contending with the memory-capture path for the durable write lock.

`attach_ephemeral(conn)` attaches that ephemeral DB to an existing durable
connection under the alias ``eph`` and ensures the two tables exist there, so
every sync query can reference ``eph.discovered_peers`` / ``eph.sync_state``
while keeping the single-connection calling convention the sync module already
uses. It is idempotent (re-attaching is a no-op) and self-healing (creates the
tables if the ephemeral file pre-dates this change or was never init'd — e.g.
in tests).

The ephemeral path is derived so that each node gets its OWN ephemeral file:
the configured `EPHEMERAL_DB_PATH` when the connection's main DB is the
configured durable DB (honours `CAIRN_EPHEMERAL_DB_PATH`), otherwise a sibling
``<stem>-ephemeral<ext>`` of whatever main DB the connection is attached to.
That sibling rule gives per-node isolation in tests (each Node has its own
``cairn.db`` -> its own ``cairn-ephemeral.db``).
"""

import os


def _main_db_path(conn):
    """Return the file path of the connection's `main` database ('' for :memory:)."""
    for _seq, name, file in conn.execute("PRAGMA database_list").fetchall():
        if name == "main":
            return file or ""
    return ""


def ephemeral_path_for(conn):
    """Resolve the ephemeral DB path for a connection, or None if undeterminable."""
    main = _main_db_path(conn)
    if not main:
        return None  # in-memory DB — no sibling to derive
    try:
        from cairn import config
        durable = getattr(config, "DB_PATH", "") or ""
        if durable and os.path.abspath(main) == os.path.abspath(durable):
            eph = getattr(config, "EPHEMERAL_DB_PATH", None)
            if eph:
                return eph
    except Exception:
        pass
    root, ext = os.path.splitext(main)
    return f"{root}-ephemeral{ext or '.db'}"


def _ensure_tables(conn):
    """Create the operational sync tables in the attached `eph` DB if absent."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eph.discovered_peers (
            node_id          TEXT PRIMARY KEY,
            user_id          TEXT,
            url              TEXT,
            public_key       TEXT,
            schema_version   INTEGER,
            first_seen       TEXT DEFAULT CURRENT_TIMESTAMP,
            last_seen        TEXT DEFAULT CURRENT_TIMESTAMP,
            cert_fingerprint TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eph.sync_state (
            peer_node_id    TEXT NOT NULL,
            source_node_id  TEXT NOT NULL,
            last_lamport    INTEGER NOT NULL,
            updated_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (peer_node_id, source_node_id)
        )
    """)


def attach_ephemeral(conn):
    """Attach the per-node ephemeral DB as `eph` and ensure operational tables.

    Idempotent: returns immediately if `eph` is already attached. No-op (returns
    False) when the ephemeral path can't be resolved (e.g. an in-memory conn);
    callers using `eph.*` tables require a file-backed connection.
    """
    names = {name for _seq, name, _file in conn.execute("PRAGMA database_list").fetchall()}
    if "eph" in names:
        return True
    path = ephemeral_path_for(conn)
    if not path:
        return False
    conn.execute("ATTACH DATABASE ? AS eph", (path,))
    conn.execute("PRAGMA eph.busy_timeout=5000")
    _ensure_tables(conn)
    return True
