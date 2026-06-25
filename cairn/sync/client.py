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
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err

from cairn.sync import SCHEMA_VERSION
from cairn.sync.changeset import apply_changeset
from cairn.sync.ephemeral import attach_ephemeral
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


class CertPinError(Exception):
    """Raised when a peer's TLS cert fingerprint != the pinned value."""


def _https_post(url: str, path: str, body_bytes: bytes, headers: dict,
                *, pinned_fp: Optional[str] = None, timeout: float = 30.0):
    """POST over HTTPS to a self-signed peer, pinning its cert fingerprint.

    Returns (status, resp_bytes, peer_cert_fp). No CA chain is used; trust comes
    from comparing the presented cert's SHA-256 to the fingerprint pinned at
    pairing (TOFU on first contact). Bypasses system proxies (direct LAN dial)."""
    import http.client
    import ssl
    from urllib.parse import urlparse
    from cairn.sync.identity import cert_fingerprint_from_der
    u = urlparse(url if "://" in url else "https://" + url)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    conn = http.client.HTTPSConnection(u.hostname, u.port or 8787,
                                       timeout=timeout, context=ctx)
    try:
        conn.connect()
        der = conn.sock.getpeercert(binary_form=True)
        peer_fp = cert_fingerprint_from_der(der) if der else None
        if pinned_fp and peer_fp != pinned_fp:
            raise CertPinError(
                f"cert fingerprint mismatch: pinned {pinned_fp[:12]}… got {(peer_fp or 'none')[:12]}…")
        conn.request("POST", path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read(), peer_fp
    finally:
        conn.close()


def _vector_clock_for_peer(conn, peer_node_id: str) -> dict[str, int]:
    """Build the since_lamport_by_node map for a given peer.

    The vector clock has two parts: (a) the high-water lamport per source-node
    we've already pulled via this peer, and (b) the local node's own lamport
    (so the peer doesn't echo our own writes back).
    """
    rows = conn.execute(
        "SELECT source_node_id, last_lamport FROM eph.sync_state WHERE peer_node_id = ?",
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
            "INSERT INTO eph.sync_state (peer_node_id, source_node_id, last_lamport) VALUES (?, ?, ?) "
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
    attach_ephemeral(conn)
    result = PullResult(peer_node_id)
    peer_row = conn.execute(
        "SELECT url, include_excerpts, peer_cert_fingerprint "
        "FROM sync_peers WHERE peer_node_id = ?",
        (peer_node_id,),
    ).fetchone()
    if not peer_row:
        result.error = f"unknown peer {peer_node_id}"
        return result
    url, include_excerpts, pinned_fp = peer_row

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
    try:
        status, data, peer_fp = _https_post(url, "/sync", body, headers,
                                            pinned_fp=pinned_fp, timeout=timeout)
    except CertPinError as e:
        result.error = f"cert pin: {e}"
        _record_attempt(conn, peer_node_id, ok=False, error=result.error)
        return result
    except (OSError, TimeoutError) as e:
        result.error = f"network: {e}"
        _record_attempt(conn, peer_node_id, ok=False, error=result.error)
        return result
    if status != 200:
        result.error = f"HTTP {status}: {data[:200].decode('utf-8', 'replace')}"
        _record_attempt(conn, peer_node_id, ok=False, error=result.error)
        return result
    try:
        payload = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        result.error = f"bad response: {e}"
        _record_attempt(conn, peer_node_id, ok=False, error=result.error)
        return result
    # TOFU: pin the peer's cert fingerprint on first successful contact.
    if not pinned_fp and peer_fp:
        conn.execute("UPDATE sync_peers SET peer_cert_fingerprint = ? WHERE peer_node_id = ?",
                     (peer_fp, peer_node_id))
        conn.commit()

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
                         my_url: Optional[str] = None, timeout: float = 15.0,
                         conn=None) -> dict:
    """Send a signed, self-certifying pairing request to a peer's /pair (HTTPS).

    TOFU-captures the peer's TLS cert fingerprint. If `conn` is given, records an
    outbound pairing request locally (direction='outbound') so the dashboard can
    show "waiting for approval". Mutual approval is required before any sync."""
    from cairn.sync.identity import get_tls_cert_fingerprint
    payload = {
        "node_id": get_node_fingerprint(),
        "public_key": get_public_key_b64(),
        "user_id": user_id or get_user_id(),
        "url": my_url or "",
        "cert_fingerprint": get_tls_cert_fingerprint(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ts = str(time.time())
    nonce = secrets.token_hex(16)
    headers = {
        "X-Cairn-Node": get_node_fingerprint(),
        "X-Cairn-Timestamp": ts,
        "X-Cairn-Nonce": nonce,
        "X-Cairn-Signature": sign_request("POST", "/pair", raw, ts, nonce),
        "X-Cairn-Schema-Version": str(SCHEMA_VERSION),
        "Content-Type": "application/json",
    }
    try:
        status, data, peer_fp = _https_post(url, "/pair", raw, headers,
                                            pinned_fp=None, timeout=timeout)
    except (OSError, TimeoutError, CertPinError) as e:
        return {"ok": False, "status": None, "error": str(e)}
    try:
        body = json.loads(data.decode("utf-8")) if data else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None
    ok = 200 <= status < 300
    if conn is not None and ok and body and body.get("node_id"):
        conn.execute(
            "INSERT INTO pairing_requests (peer_node_id, peer_public_key, user_id, url, "
            "direction, status, cert_fingerprint) VALUES (?, '', ?, ?, 'outbound', ?, ?) "
            "ON CONFLICT(peer_node_id, direction) DO UPDATE SET url=excluded.url, "
            "status=excluded.status, cert_fingerprint=excluded.cert_fingerprint, "
            "requested_at=CURRENT_TIMESTAMP, decided_at=NULL",
            (body["node_id"], user_id or get_user_id(), url,
             "already_paired" if body.get("status") == "already_paired" else "pending",
             peer_fp),
        )
        conn.commit()
    return {"ok": ok, "status": status, "body": body, "peer_cert_fingerprint": peer_fp}


def check_pair_status(url: str, *, timeout: float = 10.0) -> Optional[str]:
    """Ask a peer whether they've approved us. Returns 'approved'/'pending'/
    'denied'/'none', or None if unreachable."""
    raw = b"{}"
    ts = str(time.time())
    nonce = secrets.token_hex(16)
    headers = {
        "X-Cairn-Node": get_node_fingerprint(),
        "X-Cairn-Timestamp": ts,
        "X-Cairn-Nonce": nonce,
        "X-Cairn-Signature": sign_request("POST", "/pair/status", raw, ts, nonce),
        "X-Cairn-Schema-Version": str(SCHEMA_VERSION),
        "Content-Type": "application/json",
    }
    try:
        status, data, _ = _https_post(url, "/pair/status", raw, headers,
                                      pinned_fp=None, timeout=timeout)
    except (OSError, TimeoutError, CertPinError):
        return None
    if status == 200:
        try:
            return json.loads(data.decode("utf-8")).get("status")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    if status == 404:
        return "none"
    return None


def refresh_outbound(conn) -> list[dict]:
    """Poll each outbound-pending peer; when approved, promote it into sync_peers
    so we can start pulling. Returns [{peer_node_id, status}]."""
    rows = conn.execute(
        "SELECT peer_node_id, url, cert_fingerprint FROM pairing_requests "
        "WHERE direction = 'outbound' AND status IN ('pending', 'already_paired')"
    ).fetchall()
    out = []
    for node_id, url, cert_fp in rows:
        st = check_pair_status(url) if url else None
        if st == "approved":
            conn.execute(
                "INSERT INTO sync_peers (peer_node_id, url, "
                "peer_cert_fingerprint, status, approved_at) "
                "VALUES (?, ?, ?, 'approved', CURRENT_TIMESTAMP) "
                "ON CONFLICT(peer_node_id) DO UPDATE SET url = excluded.url, "
                "peer_cert_fingerprint = COALESCE(excluded.peer_cert_fingerprint, sync_peers.peer_cert_fingerprint), "
                "status = 'approved'",
                (node_id, url or "", cert_fp),
            )
            conn.execute(
                "UPDATE pairing_requests SET status = 'approved', decided_at = CURRENT_TIMESTAMP "
                "WHERE peer_node_id = ? AND direction = 'outbound'",
                (node_id,),
            )
            conn.commit()
        out.append({"peer_node_id": node_id, "status": st or "unreachable"})
    return out


def fetch_session(conn, origin_id: str, *, timeout: float = 15.0, cache: bool = True) -> dict:
    """Retrieve the raw session excerpt behind a (synced) memory from the peer
    that authored it. Excerpts aren't bulk-synced, so this is on-demand and
    requires being connected to (approved by) the authoring peer, which must have
    opted into session sharing. Caches the result locally for offline re-view."""
    attach_ephemeral(conn)
    row = conn.execute(
        "SELECT id, created_by_node FROM memories WHERE origin_id = ?", (origin_id,)
    ).fetchone()
    if not row:
        return {"ok": False, "error": "unknown memory"}
    mem_id, src_node = row
    local = conn.execute(
        "SELECT excerpt FROM memory_source_excerpt WHERE memory_id = ?", (mem_id,)
    ).fetchone()
    if local:
        return {"ok": True, "cached": True, "excerpt": local[0]}
    # created_by_node is the author's provenance UUID; the reachable peer is keyed
    # by its auth fingerprint. sync_state links them (peer fingerprint -> source
    # UUID, recorded on pull). Map UUID -> fingerprint, then fall back to trying
    # the UUID directly (defensive).
    candidates = [r[0] for r in conn.execute(
        "SELECT DISTINCT peer_node_id FROM eph.sync_state WHERE source_node_id = ?", (src_node,)
    ).fetchall()]
    candidates.append(src_node)
    peer = None
    for fp in candidates:
        peer = conn.execute(
            "SELECT url, peer_cert_fingerprint FROM sync_peers WHERE peer_node_id = ? "
            "AND (status IS NULL OR status = 'approved')", (fp,)
        ).fetchone()
        if peer:
            break
    if not peer:
        return {"ok": False, "error": f"not connected to the peer that authored {(src_node or '?')[:12]}…"}
    url, pinned = peer
    raw = json.dumps({"origin_id": origin_id}, sort_keys=True, separators=(",", ":")).encode()
    ts = str(time.time())
    nonce = secrets.token_hex(16)
    headers = {
        "X-Cairn-Node": get_node_fingerprint(),
        "X-Cairn-Timestamp": ts,
        "X-Cairn-Nonce": nonce,
        "X-Cairn-Signature": sign_request("POST", "/session", raw, ts, nonce),
        "X-Cairn-Schema-Version": str(SCHEMA_VERSION),
        "Content-Type": "application/json",
    }
    try:
        status, data, _ = _https_post(url, "/session", raw, headers, pinned_fp=pinned, timeout=timeout)
    except (OSError, TimeoutError, CertPinError) as e:
        return {"ok": False, "error": str(e)}
    try:
        payload = json.loads(data.decode("utf-8")) if data else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}
    if status != 200:
        return {"ok": False, "status": status, "error": payload.get("error", f"HTTP {status}")}
    if cache and payload.get("excerpt"):
        conn.execute(
            "INSERT OR REPLACE INTO memory_source_excerpt "
            "(memory_id, session_id, transcript_path, excerpt, context_before, context_after) "
            "VALUES (?, ?, '', ?, ?, ?)",
            (mem_id, payload.get("session_id"), payload["excerpt"],
             payload.get("context_before"), payload.get("context_after")),
        )
        conn.commit()
    return {"ok": True, "cached": False, **payload}


# ---- Peer registry CLI helpers ----

def add_peer(conn, *, peer_node_id: str, url: str,
             label: Optional[str] = None) -> None:
    conn.execute(
        "INSERT INTO sync_peers (peer_node_id, url, label) VALUES (?, ?, ?) "
        "ON CONFLICT(peer_node_id) DO UPDATE SET url = excluded.url, "
        "label = excluded.label",
        (peer_node_id, url, label),
    )
    conn.commit()


def remove_peer(conn, peer_node_id: str) -> None:
    attach_ephemeral(conn)
    conn.execute("DELETE FROM sync_peers WHERE peer_node_id = ?", (peer_node_id,))
    conn.execute("DELETE FROM eph.sync_state WHERE peer_node_id = ?", (peer_node_id,))
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
