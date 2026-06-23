"""Sync HTTP server — exposes /sync (changeset pull) and /pair (pairing request).

stdlib-only HTTP. v2 auth: each /sync request is signed with the peer's Ed25519
key and verified against the public key pinned in sync_peers at pairing-approval
time (status='approved'). A v1 bearer-token fallback remains for peers paired
before v2. /pair accepts a self-certifying, signed pairing request and queues it
in pairing_requests for the host to approve from the dashboard — no access is
granted until approval. See docs/multi-node-sync.md v2.

Usage:
    python -m cairn.sync.server --bind 0.0.0.0:8787
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore

from cairn.sync import SCHEMA_VERSION, MIN_COMPATIBLE_SCHEMA_VERSION
from cairn.sync.changeset import extract_changeset
from cairn.sync.identity import (
    ensure_node_id,
    get_node_fingerprint,
    node_id_for_conn,
    fingerprint,
    verify_request,
)

log = logging.getLogger("cairn.sync.server")

# Replay protection: timestamp tolerance + a short-lived seen-nonce set.
_TS_TOLERANCE_SEC = 300
_seen_nonces: dict[str, float] = {}
_seen_lock = threading.Lock()


def _resolve_db_path(override: Optional[str] = None) -> str:
    if override:
        return override
    if os.environ.get("CAIRN_DB_PATH"):
        return os.environ["CAIRN_DB_PATH"]
    from cairn import init_db
    return init_db.DB_PATH


def _fresh_timestamp(ts: str) -> bool:
    try:
        return abs(time.time() - float(ts)) <= _TS_TOLERANCE_SEC
    except (TypeError, ValueError):
        return False


def _nonce_unseen(nonce: str) -> bool:
    """True if nonce is fresh; records it. Prunes entries older than tolerance."""
    if not nonce:
        return False
    now = time.time()
    with _seen_lock:
        for n, t in list(_seen_nonces.items()):
            if now - t > _TS_TOLERANCE_SEC:
                del _seen_nonces[n]
        if nonce in _seen_nonces:
            return False
        _seen_nonces[nonce] = now
        return True


def _authorized(method: str, path: str, headers, body: bytes, conn) -> tuple[bool, Optional[str]]:
    """Authorize a /sync request. v2: Ed25519 signature against the pinned,
    approved peer_public_key. v1 fallback: bearer token. Returns (ok, peer_node)."""
    peer_node = headers.get("X-Cairn-Node", "").strip()
    if not peer_node:
        return False, None
    row = conn.execute(
        "SELECT peer_public_key, bearer_token, status FROM sync_peers WHERE peer_node_id = ?",
        (peer_node,),
    ).fetchone()
    if not row:
        return False, None
    pub, token, status = row
    if status is not None and status != "approved":
        return False, None  # revoked / denied

    # v2 signature path (preferred).
    sig = headers.get("X-Cairn-Signature", "").strip()
    if sig and pub:
        ts = headers.get("X-Cairn-Timestamp", "")
        nonce = headers.get("X-Cairn-Nonce", "")
        if not _fresh_timestamp(ts) or not _nonce_unseen(nonce):
            return False, None
        if verify_request(pub, method, path, body, ts, nonce, sig):
            return True, peer_node
        return False, None

    # v1 bearer fallback (peers paired before v2).
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer ") and token:
        if hmac.compare_digest(token, auth[len("Bearer "):].strip()):
            return True, peer_node
    return False, None


def _handle_pair(body: dict, headers, source_ip: str, conn) -> tuple[int, dict]:
    """Process a pairing request: verify it self-certifies and is signed by the
    presented key, then queue it (status='pending'). Grants no access."""
    pub = (body.get("public_key") or "").strip()
    claimed = (body.get("node_id") or "").strip()
    user_id = body.get("user_id")
    url = body.get("url")
    if not pub or not claimed:
        return 400, {"error": "public_key and node_id required"}
    # Self-certification: node_id must be the fingerprint of the presented key.
    if fingerprint(pub) != claimed:
        return 400, {"error": "node_id is not the fingerprint of public_key"}
    # Proof of possession: the request is signed by the presented key.
    sig = headers.get("X-Cairn-Signature", "")
    ts = headers.get("X-Cairn-Timestamp", "")
    nonce = headers.get("X-Cairn-Nonce", "")
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if not verify_request(pub, "POST", "/pair", raw, ts, nonce, sig):
        return 401, {"error": "signature does not verify against presented key"}

    # Already an approved peer? Idempotent no-op.
    existing = conn.execute(
        "SELECT status FROM sync_peers WHERE peer_node_id = ?", (claimed,)
    ).fetchone()
    if existing and existing[0] == "approved":
        return 200, {"status": "already_paired"}

    conn.execute(
        "INSERT INTO pairing_requests "
        "(peer_node_id, peer_public_key, user_id, url, source_ip, direction, status) "
        "VALUES (?, ?, ?, ?, ?, 'inbound', 'pending') "
        "ON CONFLICT(peer_node_id, direction) DO UPDATE SET "
        "peer_public_key=excluded.peer_public_key, user_id=excluded.user_id, "
        "url=excluded.url, source_ip=excluded.source_ip, status='pending', "
        "requested_at=CURRENT_TIMESTAMP, decided_at=NULL",
        (claimed, pub, user_id, url, source_ip),
    )
    conn.commit()
    return 202, {"status": "pending", "node_id": get_node_fingerprint()}


def _handle_pair_status(headers, body: bytes, conn) -> tuple[int, dict]:
    """A peer asks whether we have approved them. Verify their signature against
    the pubkey we hold (approved peer or pending request), then report status."""
    peer = headers.get("X-Cairn-Node", "").strip()
    if not peer:
        return 400, {"error": "missing node"}
    pub, status = None, None
    row = conn.execute(
        "SELECT peer_public_key, status FROM sync_peers WHERE peer_node_id = ?", (peer,)
    ).fetchone()
    if row and row[0]:
        pub, status = row[0], (row[1] or "approved")
    if not pub:
        rr = conn.execute(
            "SELECT peer_public_key, status FROM pairing_requests "
            "WHERE peer_node_id = ? AND direction = 'inbound'", (peer,)
        ).fetchone()
        if rr:
            pub, status = rr[0], rr[1]
    if not pub:
        return 404, {"status": "none"}
    ts = headers.get("X-Cairn-Timestamp", "")
    nonce = headers.get("X-Cairn-Nonce", "")
    if not verify_request(pub, "POST", "/pair/status", body, ts, nonce,
                          headers.get("X-Cairn-Signature", "")):
        return 401, {"error": "bad signature"}
    return 200, {"status": status or "pending"}


def _handle_session(headers, body: bytes, conn) -> tuple[int, dict]:
    """Serve the raw session excerpt behind one of OUR memories to an approved
    peer, on demand. Gated by CAIRN_SYNC_SHARE_SESSIONS (transcripts are sensitive
    and never bulk-synced). Own-data-only + never for private memories."""
    ok, _peer = _authorized("POST", "/session", headers, body, conn)
    if not ok:
        return 401, {"error": "unauthorized"}
    from cairn import config
    if not config.CAIRN_SYNC_SHARE_SESSIONS:
        return 403, {"error": "session sharing disabled on this node"}
    try:
        req = json.loads(body.decode("utf-8")) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        req = {}
    origin = (req.get("origin_id") or "").strip()
    if not origin:
        return 400, {"error": "origin_id required"}
    self_node = node_id_for_conn(conn)
    row = conn.execute(
        "SELECT mse.excerpt, mse.context_before, mse.context_after, mse.session_id, mse.captured_at "
        "FROM memories m JOIN memory_source_excerpt mse ON mse.memory_id = m.id "
        "WHERE m.origin_id = ? AND m.created_by_node = ? "
        "AND (m.visibility != 'private' OR m.visibility IS NULL)",
        (origin, self_node),
    ).fetchone()
    if not row:
        return 404, {"error": "no shareable session for that memory"}
    return 200, {
        "origin_id": origin, "excerpt": row[0], "context_before": row[1],
        "context_after": row[2], "session_id": row[3], "captured_at": row[4],
    }


class SyncHandler(BaseHTTPRequestHandler):
    db_path: str = ""  # set by build_server

    def log_message(self, fmt, *args):  # quiet stderr unless debug
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _read_body(self) -> Optional[bytes]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > 10_000_000:  # 10MB cap
            self._send_json(413, {"error": "payload too large"})
            return None
        return self.rfile.read(length) if length else b""

    def _schema_ok(self) -> bool:
        client_schema = self.headers.get("X-Cairn-Schema-Version", "")
        try:
            client_schema_int = int(client_schema)
        except ValueError:
            self._send_json(400, {"error": "missing X-Cairn-Schema-Version header"})
            return False
        # Accept any peer at or above our compatibility floor — apply tolerates
        # additive drift (unknown fields ignored, missing default NULL). Only a
        # peer older than the floor (a breaking-change boundary) is refused.
        if client_schema_int < MIN_COMPATIBLE_SCHEMA_VERSION:
            self._send_json(409, {
                "error": "schema_version_mismatch",
                "server": SCHEMA_VERSION,
                "min_compatible": MIN_COMPATIBLE_SCHEMA_VERSION,
                "client": client_schema_int,
            })
            return False
        return True

    def do_POST(self) -> None:  # noqa: N802 — stdlib API
        if self.path not in ("/sync", "/pair", "/pair/status", "/session"):
            self._send_json(404, {"error": "not found"})
            return
        if not self._schema_ok():
            return
        body_bytes = self._read_body()
        if body_bytes is None:
            return
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_json(400, {"error": f"invalid JSON: {e}"})
            return

        source_ip = self.client_address[0] if self.client_address else ""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            if self.path == "/pair":
                status, obj = _handle_pair(body, self.headers, source_ip, conn)
                self._send_json(status, obj)
                return
            if self.path == "/pair/status":
                status, obj = _handle_pair_status(self.headers, body_bytes, conn)
                self._send_json(status, obj)
                return
            if self.path == "/session":
                status, obj = _handle_session(self.headers, body_bytes, conn)
                self._send_json(status, obj)
                return

            # /sync
            ok, peer_node = _authorized("POST", "/sync", self.headers, body_bytes, conn)
            if not ok:
                self._send_json(401, {"error": "unauthorized"})
                return
            since = body.get("since_lamport_by_node") or {}
            if not isinstance(since, dict):
                self._send_json(400, {"error": "since_lamport_by_node must be object"})
                return
            payload = extract_changeset(
                conn,
                since_lamport_by_node={k: int(v) for k, v in since.items()},
                include_excerpts=bool(body.get("include_excerpts")),
                max_rows=int(body.get("max_rows") or 5000),
            )
            payload["node_id"] = node_id_for_conn(conn)
            self._send_json(200, payload)
        finally:
            conn.close()

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Cairn-Schema-Version", str(SCHEMA_VERSION))
        self.end_headers()
        self.wfile.write(body)


def _tls_context() -> "ssl.SSLContext":
    """Server-side TLS using this node's self-signed cert. Peers authenticate it
    by pinning the cert fingerprint at pairing, so no CA chain is involved."""
    import ssl
    from cairn.sync.identity import ensure_tls_cert
    cert_p, key_p = ensure_tls_cert()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert_p), keyfile=str(key_p))
    return ctx


def build_server(host: str = "127.0.0.1", port: int = 8787,
                 db_path: Optional[str] = None, *, tls: bool = True) -> ThreadingHTTPServer:
    handler_class = type("BoundSyncHandler", (SyncHandler,),
                          {"db_path": _resolve_db_path(db_path)})
    httpd = ThreadingHTTPServer((host, port), handler_class)
    if tls:
        httpd.socket = _tls_context().wrap_socket(httpd.socket, server_side=True)
    return httpd


def serve_in_thread(host: str = "127.0.0.1", port: int = 8787,
                    db_path: Optional[str] = None, *, tls: bool = True
                    ) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the server in a daemon thread. For tests."""
    httpd = build_server(host, port, db_path, tls=tls)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="127.0.0.1:8787")
    p.add_argument("--db", default=None)
    args = p.parse_args()
    host, port = args.bind.rsplit(":", 1)
    httpd = build_server(host, int(port), args.db)
    log.info("cairn sync server on %s db=%s node=%s",
             args.bind, _resolve_db_path(args.db), get_node_fingerprint()[:12])
    print(f"cairn sync server on {args.bind} (node {get_node_fingerprint()[:12]})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _main()
