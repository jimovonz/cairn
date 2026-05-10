"""Sync HTTP client — pulls changesets from listed peers and applies them.

Reads peer registry from sync_peers table (managed via `cairn sync add-peer`).
Designed to be called periodically (cron, systemd timer, or daemon).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore

from cairn.sync import SCHEMA_VERSION
from cairn.sync.changeset import apply_changeset
from cairn.sync.identity import ensure_node_id, node_id_for_conn

log = logging.getLogger("cairn.sync.client")


class PullResult:
    def __init__(self, peer_node_id: str) -> None:
        self.peer_node_id = peer_node_id
        self.ok = False
        self.error: Optional[str] = None
        self.apply_stats: dict[str, Any] = {}
        self.row_counts: dict[str, int] = {}


def _vector_clock_for_peer(conn, peer_node_id: str) -> dict[str, int]:
    """Build the since_lamport_by_node map for a given peer.

    The vector clock has two parts: (a) the high-water lamport per source-node
    we've already pulled via this peer, and (b) the local node's own lamport
    (so the peer doesn't echo our own writes back).
    """
    rows = conn.execute(
        "SELECT source_node_id, last_lamport FROM sync_state WHERE peer_node_id = ?",
        (peer_node_id,),
    ).fetchall()
    vec = {src: lam for src, lam in rows}
    # Always exclude our own writes
    local_node = node_id_for_conn(conn)
    local_max = conn.execute(
        "SELECT COALESCE(MAX(lamport), 0) FROM memories WHERE created_by_node = ?",
        (local_node,),
    ).fetchone()[0]
    vec[local_node] = max(vec.get(local_node, 0), int(local_max or 0))
    return vec


def _update_sync_state(conn, peer_node_id: str, payload: dict) -> None:
    """Update high-water lamport per source-node observed in this payload."""
    max_by_source: dict[str, int] = {}
    for rec in payload.get("memories", []):
        src = rec.get("created_by_node")
        if src:
            max_by_source[src] = max(max_by_source.get(src, 0), int(rec.get("lamport") or 0))
    for entry in payload.get("confidence_log", []):
        src = entry.get("node_id")
        if src:
            max_by_source[src] = max(max_by_source.get(src, 0), int(entry.get("lamport") or 0))
    for h in payload.get("memory_history", []):
        src = h.get("changed_by_node")
        if src:
            max_by_source[src] = max(max_by_source.get(src, 0), int(h.get("lamport") or 0))
    for src, lam in max_by_source.items():
        conn.execute(
            "INSERT INTO sync_state (peer_node_id, source_node_id, last_lamport) VALUES (?, ?, ?) "
            "ON CONFLICT(peer_node_id, source_node_id) DO UPDATE SET "
            "last_lamport = MAX(last_lamport, excluded.last_lamport), updated_at = CURRENT_TIMESTAMP",
            (peer_node_id, src, lam),
        )


def pull_from_peer(
    conn,
    peer_node_id: str,
    *,
    embedder=None,
    timeout: float = 30.0,
    max_rows: int = 5000,
) -> PullResult:
    """Pull from one peer. Updates sync_state high-water marks on success."""
    result = PullResult(peer_node_id)
    peer_row = conn.execute(
        "SELECT url, bearer_token, include_excerpts FROM sync_peers WHERE peer_node_id = ?",
        (peer_node_id,),
    ).fetchone()
    if not peer_row:
        result.error = f"unknown peer {peer_node_id}"
        return result
    url, token, include_excerpts = peer_row

    since = _vector_clock_for_peer(conn, peer_node_id)
    body = json.dumps({
        "since_lamport_by_node": since,
        "include_excerpts": bool(include_excerpts),
        "max_rows": max_rows,
    }).encode("utf-8")

    req = urllib.request.Request(
        url.rstrip("/") + "/sync",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Cairn-Node": node_id_for_conn(conn),
            "X-Cairn-Schema-Version": str(SCHEMA_VERSION),
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = ""
        result.error = f"HTTP {e.code}: {err_body[:200]}"
        _record_attempt(conn, peer_node_id, ok=False, error=result.error)
        return result
    except (urllib.error.URLError, TimeoutError) as e:
        result.error = f"network: {e}"
        _record_attempt(conn, peer_node_id, ok=False, error=result.error)
        return result

    # Apply
    try:
        apply_result = apply_changeset(conn, payload, embedder=embedder)
        result.apply_stats = apply_result.to_dict()
        result.row_counts = {
            k: len(payload.get(k, []))
            for k in ("memories", "memory_history", "confidence_log",
                      "correction_triggers", "pair_assessments", "tombstones")
        }
        _update_sync_state(conn, peer_node_id, payload)
        _record_attempt(conn, peer_node_id, ok=True, error=None)
        conn.commit()
        result.ok = True
    except Exception as e:
        result.error = f"apply failed: {e}"
        _record_attempt(conn, peer_node_id, ok=False, error=result.error)
    return result


def _record_attempt(conn, peer_node_id: str, *, ok: bool, error: Optional[str]) -> None:
    if ok:
        conn.execute(
            "UPDATE sync_peers SET last_attempted_at = CURRENT_TIMESTAMP, "
            "last_succeeded_at = CURRENT_TIMESTAMP, last_error = NULL WHERE peer_node_id = ?",
            (peer_node_id,),
        )
    else:
        conn.execute(
            "UPDATE sync_peers SET last_attempted_at = CURRENT_TIMESTAMP, last_error = ? "
            "WHERE peer_node_id = ?",
            (error, peer_node_id),
        )


def pull_all(conn, *, embedder=None) -> list[PullResult]:
    """Pull from every registered peer."""
    peers = conn.execute("SELECT peer_node_id FROM sync_peers").fetchall()
    return [pull_from_peer(conn, p[0], embedder=embedder) for p in peers]


# ---- Peer registry CLI helpers ----

def add_peer(conn, *, peer_node_id: str, url: str, bearer_token: str,
             label: Optional[str] = None) -> None:
    conn.execute(
        "INSERT INTO sync_peers (peer_node_id, url, bearer_token, label) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(peer_node_id) DO UPDATE SET url = excluded.url, "
        "bearer_token = excluded.bearer_token, label = excluded.label",
        (peer_node_id, url, bearer_token, label),
    )
    conn.commit()


def remove_peer(conn, peer_node_id: str) -> None:
    conn.execute("DELETE FROM sync_peers WHERE peer_node_id = ?", (peer_node_id,))
    conn.execute("DELETE FROM sync_state WHERE peer_node_id = ?", (peer_node_id,))
    conn.commit()


def list_peers(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT peer_node_id, url, label, last_attempted_at, last_succeeded_at, last_error "
        "FROM sync_peers"
    ).fetchall()
    return [
        {"peer_node_id": p, "url": u, "label": lbl,
         "last_attempted_at": la, "last_succeeded_at": ls, "last_error": le}
        for p, u, lbl, la, ls, le in rows
    ]
