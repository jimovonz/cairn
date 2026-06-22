"""Sync HTTP transport: server + client end-to-end."""

from __future__ import annotations

import socket
import time

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_for_port(port: int, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"server did not bind on {port}")


def test_pull_via_http_replicates(make_node, fake_embedder):
    server_node = make_node("server")
    client_node = make_node("client")
    server_node.insert_memory(type_="fact", topic="t", content="via-http", origin_id="http-1")

    port = _free_port()
    from cairn.sync.server import serve_in_thread
    httpd, thread = serve_in_thread(host="127.0.0.1", port=port, db_path=server_node.db_path)
    try:
        _wait_for_port(port)
        # Register the server peer in the client's local registry (URL lookup).
        from cairn.sync.client import add_peer, pull_from_peer
        from cairn.sync import identity
        client_node.activate()
        client_fp = identity.get_node_fingerprint()
        client_pub = identity.get_public_key_b64()
        add_peer(
            client_node.conn(),
            peer_node_id=server_node.node_id,
            url=f"https://127.0.0.1:{port}",
            bearer_token="",
        )
        # Server-side: pin the client's public key as an approved peer — the
        # post-pairing state. v2 authenticates /sync by Ed25519 signature.
        server_conn = server_node.conn()
        server_conn.execute(
            "INSERT INTO sync_peers (peer_node_id, url, bearer_token, peer_public_key, status, approved_at) "
            "VALUES (?, '', '', ?, 'approved', CURRENT_TIMESTAMP)",
            (client_fp, client_pub),
        )
        server_conn.commit()
        client_node.activate()
        result = pull_from_peer(client_node.conn(), server_node.node_id, embedder=fake_embedder)
        assert result.ok, f"pull failed: {result.error}"
        assert result.row_counts["memories"] == 1
        # Verify replicated
        got = client_node.conn().execute(
            "SELECT content FROM memories WHERE origin_id = 'http-1'"
        ).fetchone()
        assert got[0] == "via-http"
    finally:
        httpd.shutdown()


def test_unauthorized_token_rejected(make_node):
    server_node = make_node("server")
    client_node = make_node("client")
    port = _free_port()
    from cairn.sync.server import serve_in_thread
    httpd, _ = serve_in_thread(host="127.0.0.1", port=port, db_path=server_node.db_path)
    try:
        _wait_for_port(port)
        from cairn.sync.client import add_peer, pull_from_peer
        add_peer(
            client_node.conn(),
            peer_node_id=server_node.node_id,
            url=f"https://127.0.0.1:{port}",
            bearer_token="WRONG-TOKEN",
        )
        # Server registers the client peer with a DIFFERENT token
        add_peer(
            server_node.conn(),
            peer_node_id=client_node.node_id,
            url="http://unused",
            bearer_token="REAL-TOKEN",
        )
        result = pull_from_peer(client_node.conn(), server_node.node_id)
        assert not result.ok
        assert "401" in (result.error or ""), f"expected 401, got: {result.error}"
    finally:
        httpd.shutdown()


def test_schema_version_mismatch_returns_409(make_node, monkeypatch):
    server_node = make_node("server")
    client_node = make_node("client")
    port = _free_port()
    from cairn.sync.server import serve_in_thread
    httpd, _ = serve_in_thread(host="127.0.0.1", port=port, db_path=server_node.db_path)
    try:
        _wait_for_port(port)
        from cairn.sync.client import add_peer, pull_from_peer
        from cairn.sync import client as client_mod
        add_peer(
            client_node.conn(),
            peer_node_id=server_node.node_id,
            url=f"https://127.0.0.1:{port}",
            bearer_token="t",
        )
        add_peer(
            server_node.conn(),
            peer_node_id=client_node.node_id,
            url="http://unused",
            bearer_token="t",
        )
        # Client claims a schema_version BELOW the server's compatibility floor
        monkeypatch.setattr(client_mod, "SCHEMA_VERSION", 1)
        result = pull_from_peer(client_node.conn(), server_node.node_id)
        assert not result.ok
        assert "409" in (result.error or "") or "schema_version" in (result.error or "")
    finally:
        httpd.shutdown()
