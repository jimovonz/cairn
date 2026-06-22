"""Sync HTTP client — pulls changesets from listed peers and applies them.

Reads peer registry from sync_peers table (managed via `cairn sync add-peer`).
Designed to be called periodically (cron, systemd timer, or daemon).
"""

from __future__ import annotations

import json
import logging
import secrets
import time
import urllib.error
import urllib.request
from typing import Any, Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore

from cairn.sync import SCHEMA_VERSION
from cairn.sync.changeset import apply_changeset
from cairn.sync.identity import (
    ensure_node_id,
    node_id_for_conn,
    get_node_fingerprint,
    get_public_key_b64,
    get_user_id,
    sign_request,
)

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

    ts = str(time.time())
    nonce = secrets.token_hex(16)
    headers = {
        "X-Cairn-Node": get_node_fingerprint(),
        "X-Cairn-Timestamp": ts,
        "X-Cairn-Nonce": nonce,
        "X-Cairn-Signature": sign_request("POST", "/sync", body, ts, nonce),
        "X-Cairn-Schema-Version": str(SCHEMA_VERSION),
        "Content-Type": "application/json",
    }
    if token:  # v1 bearer fallback for peers paired before v2
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url.rstrip("/") + "/sync",
        data=body,
        method="POST",
        headers=headers,
    )

    # Build an opener that explicitly disables system proxies — cairn peer URLs
    # are direct connections to known endpoints. Falling through to ALL_PROXY /
    # HTTP_PROXY / HTTPS_PROXY env vars (e.g. corporate SOCKS5) would silently
    # route sync traffic somewhere that doesn't speak the protocol.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
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
    """Pull from every registered, approved peer."""
    peers = conn.execute(
        "SELECT peer_node_id FROM sync_peers WHERE status IS NULL OR status = 'approved'"
    ).fetchall()
    return [pull_from_peer(conn, p[0], embedder=embedder) for p in peers]


def send_pairing_request(url: str, *, user_id: Optional[str] = None,
                         my_url: Optional[str] = None, timeout: float = 15.0) -> dict:
    """Send a signed, self-certifying pairing request to a peer's /pair endpoint.

    The peer queues it for manual approval (dashboard); we get no access until
    they approve us AND we approve them (mutual). Returns the peer's response
    dict: {ok, status, body|error}."""
    pub = get_public_key_b64()
    payload = {
        "node_id": get_node_fingerprint(),
        "public_key": pub,
        "user_id": user_id or get_user_id(),
        "url": my_url or "",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ts = str(time.time())
    nonce = secrets.token_hex(16)
    sig = sign_request("POST", "/pair", raw, ts, nonce)
    req = urllib.request.Request(
        url.rstrip("/") + "/pair",
        data=raw,
        method="POST",
        headers={
            "X-Cairn-Node": get_node_fingerprint(),
            "X-Cairn-Timestamp": ts,
            "X-Cairn-Nonce": nonce,
            "X-Cairn-Signature": sig,
            "X-Cairn-Schema-Version": str(SCHEMA_VERSION),
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status,
                    "body": json.loads(resp.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        try:
            return {"ok": False, "status": e.code,
                    "body": json.loads(e.read().decode("utf-8"))}
        except Exception:
            return {"ok": False, "status": e.code, "body": None}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"ok": False, "status": None, "error": str(e)}


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
