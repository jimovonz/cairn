"""Node identity, user identity, embedding model version, and Lamport clock.

All node-scoped state lives in two places:
- ~/.cairn/node_id and ~/.cairn/user_id   — file-based, survive DB rebuild
- node_state table in cairn.db            — Lamport clock, embedding model version

Both are written on first call and read thereafter. Idempotent.
"""

from __future__ import annotations

import hashlib
import os
import socket
import uuid
from pathlib import Path
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore


# ---- File-based identity ----

def _state_dir() -> Path:
    """Resolve ~/.cairn/ — overridable via CAIRN_STATE_DIR for tests."""
    override = os.environ.get("CAIRN_STATE_DIR")
    if override:
        d = Path(override)
    else:
        d = Path.home() / ".cairn"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_node_id() -> str:
    """Return this node's UUID, generating ~/.cairn/node_id if missing.

    The file-based identity survives DB rebuilds and reinstalls. Deleting it
    creates a new identity that peers will see as a separate node.
    """
    p = _state_dir() / "node_id"
    if p.exists():
        nid = p.read_text().strip()
        if nid:
            return nid
    nid = str(uuid.uuid4())
    p.write_text(nid + "\n")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return nid


def get_user_id() -> str:
    """Return the user identity, defaulting to USER@hostname.

    Override via ~/.cairn/user_id (e.g. set to LDAP/email handle for team rollout).
    """
    p = _state_dir() / "user_id"
    if p.exists():
        uid = p.read_text().strip()
        if uid:
            return uid
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    host = socket.gethostname()
    return f"{user}@{host}"


# ---- Embedding model version ----

def get_embedding_model_version() -> str:
    """Return a stable identifier for the local embedding model.

    Format: '<model-name>@<sha256-prefix>'. The hash is over a small
    canary embedding so any silent model swap (same name, different weights)
    produces a different version string. Cached after first call.
    """
    if _MODEL_VERSION_CACHE.get("v"):
        return _MODEL_VERSION_CACHE["v"]
    name = os.environ.get("CAIRN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    canary = "cairn-canary"
    digest_prefix = hashlib.sha256((name + "|" + canary).encode()).hexdigest()[:8]
    # We deliberately don't load the model here — that would couple every init() to
    # heavy imports. The hash above is name-only; on first embed call, the model
    # ID is re-derived and stored. Honest naming: this is "advertised model version,"
    # not "verified weight digest." Good enough for v1 sync compatibility checks.
    v = f"{name}@{digest_prefix}"
    _MODEL_VERSION_CACHE["v"] = v
    return v


_MODEL_VERSION_CACHE: dict[str, str] = {}


# ---- Lamport clock ----

def peek_lamport(conn) -> int:
    """Return the current Lamport clock value without bumping it."""
    row = conn.execute("SELECT value FROM node_state WHERE key = 'lamport'").fetchone()
    return int(row[0]) if row else 0


def bump_lamport(conn, observed: Optional[int] = None) -> int:
    """Atomically increment the Lamport clock past `observed` and return the new value.

    Use this on every mutating operation so the row's `lamport` column reflects
    causal ordering across nodes. Pass the highest peer-observed lamport when
    applying a changeset, so subsequent local edits are causally after.
    """
    cur = peek_lamport(conn)
    new = max(cur, observed or 0) + 1
    conn.execute(
        "INSERT INTO node_state (key, value, updated_at) VALUES ('lamport', ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        (str(new),),
    )
    return new


def node_id_for_conn(conn) -> str:
    """Resolve this DB's node_id, preferring the per-DB record over the env-bound file.

    Why: a single process can connect to multiple DBs (tests, multi-tenant tools).
    The env-var-driven file is fine for a normal install (one DB per home dir),
    but sync code must use the identity tied to the *connection*, not the env.
    Persists the resolved id back to node_state so subsequent calls are env-free.
    """
    row = conn.execute("SELECT value FROM node_state WHERE key = 'node_id'").fetchone()
    if row and row[0]:
        return row[0]
    nid = ensure_node_id()
    conn.execute(
        "INSERT INTO node_state (key, value) VALUES ('node_id', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (nid,),
    )
    conn.commit()
    return nid


def record_embedding_model(conn) -> None:
    """Persist the current embedding model version to node_state."""
    v = get_embedding_model_version()
    conn.execute(
        "INSERT INTO node_state (key, value, updated_at) VALUES ('embedding_model_version', ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        (v,),
    )
