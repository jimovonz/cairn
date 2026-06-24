"""Pairing handshake management — the host side of v2 trust establishment.

Inbound pairing requests land in `pairing_requests` (via the /pair endpoint).
The host approves or denies them from the dashboard or CLI; approving pins the
peer's public key into `sync_peers` (status='approved'), which is what the /sync
signature check authenticates against. Revoking flips status to 'revoked'.
See docs/multi-node-sync.md v2.
"""

from __future__ import annotations

from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    import sqlite3  # type: ignore  # noqa: F401

_REQ_COLS = [
    "id", "peer_node_id", "peer_public_key", "user_id", "url", "source_ip",
    "direction", "status", "requested_at", "decided_at",
]


def list_pairing_requests(conn, *, pending_only: bool = False) -> list[dict]:
    q = f"SELECT {', '.join(_REQ_COLS)} FROM pairing_requests"
    if pending_only:
        q += " WHERE status = 'pending'"
    q += " ORDER BY (status = 'pending') DESC, requested_at DESC"
    return [dict(zip(_REQ_COLS, r)) for r in conn.execute(q).fetchall()]


def approve_pairing(conn, request_id: int, *, label: Optional[str] = None) -> dict:
    """Approve a pending request: pin its public key into sync_peers."""
    row = conn.execute(
        "SELECT peer_node_id, peer_public_key, user_id, url, cert_fingerprint "
        "FROM pairing_requests WHERE id = ?",
        (request_id,),
    ).fetchone()
    if not row:
        return {"ok": False, "error": "no such pairing request"}
    node_id, pub, user_id, url, cert_fp = row
    conn.execute(
        "INSERT INTO sync_peers "
        "(peer_node_id, url, label, peer_public_key, peer_cert_fingerprint, status, approved_at) "
        "VALUES (?, ?, ?, ?, ?, 'approved', CURRENT_TIMESTAMP) "
        "ON CONFLICT(peer_node_id) DO UPDATE SET url = excluded.url, "
        "peer_public_key = excluded.peer_public_key, "
        "peer_cert_fingerprint = COALESCE(excluded.peer_cert_fingerprint, sync_peers.peer_cert_fingerprint), "
        "label = COALESCE(excluded.label, sync_peers.label), "
        "status = 'approved', approved_at = CURRENT_TIMESTAMP",
        (node_id, url or "", label or user_id, pub, cert_fp),
    )
    conn.execute(
        "UPDATE pairing_requests SET status = 'approved', decided_at = CURRENT_TIMESTAMP WHERE id = ?",
        (request_id,),
    )
    conn.commit()
    return {"ok": True, "peer_node_id": node_id, "url": url}


def deny_pairing(conn, request_id: int) -> dict:
    cur = conn.execute(
        "UPDATE pairing_requests SET status = 'denied', decided_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND status = 'pending'",
        (request_id,),
    )
    conn.commit()
    return {"ok": cur.rowcount > 0}


def revoke_peer(conn, peer_node_id: str) -> dict:
    """Revoke an approved peer. Future /sync requests from it are rejected;
    already-pulled memories are retained with attribution."""
    cur = conn.execute(
        "UPDATE sync_peers SET status = 'revoked' WHERE peer_node_id = ?",
        (peer_node_id,),
    )
    conn.commit()
    return {"ok": cur.rowcount > 0}


def list_peers(conn) -> list[dict]:
    cols = ["peer_node_id", "url", "label", "status", "approved_at",
            "last_attempted_at", "last_succeeded_at", "last_error"]
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM sync_peers ORDER BY approved_at DESC"
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]
