"""Sync HTTP server — exposes /sync POST endpoint to listed peers.

stdlib-only HTTP. Auth via per-peer bearer token recorded in sync_peers.
For prod: terminate TLS in front (caddy/stunnel/nginx) — the server itself is plaintext.

Usage:
    python -m cairn.sync.server --bind 0.0.0.0:8787

Or programmatically:
    httpd = build_server(host='127.0.0.1', port=8787, db_path=...)
    httpd.serve_forever()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore

from cairn.sync import SCHEMA_VERSION
from cairn.sync.changeset import extract_changeset
from cairn.sync.identity import ensure_node_id, node_id_for_conn, peek_lamport

log = logging.getLogger("cairn.sync.server")


def _resolve_db_path(override: Optional[str] = None) -> str:
    if override:
        return override
    if os.environ.get("CAIRN_DB_PATH"):
        return os.environ["CAIRN_DB_PATH"]
    from cairn import init_db
    return init_db.DB_PATH


def _authorized(headers, conn) -> tuple[bool, Optional[str]]:
    """Validate Bearer token against sync_peers.bearer_token. Returns (ok, peer_node_id)."""
    auth = headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False, None
    token = auth[len("Bearer "):].strip()
    peer_node = headers.get("X-Cairn-Node", "").strip()
    if not peer_node:
        return False, None
    row = conn.execute(
        "SELECT bearer_token FROM sync_peers WHERE peer_node_id = ?", (peer_node,)
    ).fetchone()
    if not row:
        return False, None
    # Constant-time compare
    import hmac
    if not hmac.compare_digest(row[0] or "", token):
        return False, None
    return True, peer_node


class SyncHandler(BaseHTTPRequestHandler):
    db_path: str = ""  # set by build_server

    def log_message(self, fmt, *args):  # quiet stderr unless debug
        log.debug("%s - %s", self.address_string(), fmt % args)

    def do_POST(self) -> None:  # noqa: N802 — stdlib API
        if self.path != "/sync":
            self._send_json(404, {"error": "not found"})
            return

        # Schema version handshake
        client_schema = self.headers.get("X-Cairn-Schema-Version", "")
        try:
            client_schema_int = int(client_schema)
        except ValueError:
            self._send_json(400, {"error": "missing X-Cairn-Schema-Version header"})
            return
        if client_schema_int != SCHEMA_VERSION:
            self._send_json(409, {
                "error": "schema_version_mismatch",
                "server": SCHEMA_VERSION, "client": client_schema_int,
            })
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > 10_000_000:  # 10MB request cap
            self._send_json(413, {"error": "payload too large"})
            return
        try:
            body_raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            body = json.loads(body_raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_json(400, {"error": f"invalid JSON: {e}"})
            return

        conn = sqlite3.connect(self.db_path)
        try:
            ok, peer_node = _authorized(self.headers, conn)
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


def build_server(host: str = "127.0.0.1", port: int = 8787,
                 db_path: Optional[str] = None) -> ThreadingHTTPServer:
    handler_class = type("BoundSyncHandler", (SyncHandler,),
                          {"db_path": _resolve_db_path(db_path)})
    return ThreadingHTTPServer((host, port), handler_class)


def serve_in_thread(host: str = "127.0.0.1", port: int = 8787,
                    db_path: Optional[str] = None) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the server in a daemon thread. For tests."""
    httpd = build_server(host, port, db_path)
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
             args.bind, _resolve_db_path(args.db), ensure_node_id()[:8])
    print(f"cairn sync server on {args.bind}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _main()
