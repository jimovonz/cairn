"""v2 sync: Ed25519 identity, pairing handshake, signature auth, discovery."""

from __future__ import annotations

import socket
import time

import pytest


# ───────────────────────── identity / crypto units ──────────────────────────

def test_fingerprint_is_stable_and_self_certifying(node):
    node.activate()
    from cairn.sync import identity
    pub = identity.get_public_key_b64()
    fp1 = identity.fingerprint(pub)
    fp2 = identity.get_node_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 52 and fp1.isalnum()
    # A different key → different fingerprint
    other = identity.fingerprint(
        identity._b64.b64encode(b"\x01" * 32).decode())
    assert other != fp1


def test_sign_verify_roundtrip_and_tamper(node):
    node.activate()
    from cairn.sync import identity
    pub = identity.get_public_key_b64()
    body = b'{"hello":"world"}'
    sig = identity.sign_request("POST", "/sync", body, "123", "nonce-1")
    assert identity.verify_request(pub, "POST", "/sync", body, "123", "nonce-1", sig)
    # Tamper with body / path / ts → fail
    assert not identity.verify_request(pub, "POST", "/sync", b"X", "123", "nonce-1", sig)
    assert not identity.verify_request(pub, "POST", "/pair", body, "123", "nonce-1", sig)
    assert not identity.verify_request(pub, "POST", "/sync", body, "999", "nonce-1", sig)


# ───────────────────────────── pairing host side ────────────────────────────

def _make_request_row(node, *, fp="PEERFINGERPRINT", pub="cHVia2V5", user="bob"):
    conn = node.conn()
    conn.execute(
        "INSERT INTO pairing_requests (peer_node_id, peer_public_key, user_id, url, "
        "source_ip, direction, status) VALUES (?, ?, ?, 'http://x:8787', '10.0.0.2', 'inbound', 'pending')",
        (fp, pub, user),
    )
    rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return rid


def test_approve_pins_pubkey_into_sync_peers(node):
    from cairn.sync import pairing
    rid = _make_request_row(node)
    res = pairing.approve_pairing(node.conn(), rid)
    assert res["ok"]
    row = node.conn().execute(
        "SELECT peer_public_key, status FROM sync_peers WHERE peer_node_id = 'PEERFINGERPRINT'"
    ).fetchone()
    assert row == ("cHVia2V5", "approved")
    req = node.conn().execute(
        "SELECT status FROM pairing_requests WHERE id = ?", (rid,)).fetchone()
    assert req[0] == "approved"


def test_deny_then_revoke(node):
    from cairn.sync import pairing
    rid = _make_request_row(node)
    assert pairing.deny_pairing(node.conn(), rid)["ok"]
    assert node.conn().execute(
        "SELECT status FROM pairing_requests WHERE id = ?", (rid,)).fetchone()[0] == "denied"
    # Approve a fresh one then revoke
    rid2 = _make_request_row(node, fp="PEER2")
    pairing.approve_pairing(node.conn(), rid2)
    assert pairing.revoke_peer(node.conn(), "PEER2")["ok"]
    assert node.conn().execute(
        "SELECT status FROM sync_peers WHERE peer_node_id = 'PEER2'").fetchone()[0] == "revoked"


# ───────────────────────────────── discovery ────────────────────────────────

def test_beacon_build_parse_record(node):
    from cairn.sync import discovery
    beacon = discovery.build_beacon("NODEX", "alice", "http://1.2.3.4:8787", "PUBX", 10)
    parsed = discovery.parse_beacon(beacon)
    assert parsed["node_id"] == "NODEX" and parsed["schema_version"] == 10
    assert discovery.parse_beacon(b"not json") is None
    assert discovery.parse_beacon(b'{"no":"magic"}') is None
    # record, and ignore our own
    conn = node.conn()
    assert discovery.record_beacon(conn, parsed, self_node_id="NODEX") is False
    assert discovery.record_beacon(conn, parsed, self_node_id="SOMEONE-ELSE") is True
    got = discovery.list_discovered(conn)
    assert any(d["node_id"] == "NODEX" for d in got)


def test_discovery_udp_loopback():
    """advertise_once → listen over localhost UDP."""
    from cairn.sync import discovery
    port = _free_udp_port()
    import threading
    results = []
    def _run():
        results.extend(discovery.listen(duration=2.0, port=port))
    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.4)
    beacon = discovery.build_beacon("LOOPNODE", "u", "http://127.0.0.1:8787", "PK", 10)
    discovery.advertise_once(beacon, port=port, addr="127.0.0.1")
    t.join(timeout=3.0)
    assert any(b["node_id"] == "LOOPNODE" for b in results)


# ─────────────────────── end-to-end pair + signed pull ──────────────────────

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
    raise TimeoutError(f"no bind on {port}")


def test_pair_approve_then_signed_pull(make_node, fake_embedder):
    server_node = make_node("server")
    client_node = make_node("client")
    server_node.insert_memory(type_="fact", topic="t", content="signed-payload", origin_id="sig-1")

    port = _free_port()
    from cairn.sync.server import serve_in_thread
    from cairn.sync import client as client_mod, pairing
    httpd, _ = serve_in_thread(host="127.0.0.1", port=port, db_path=server_node.db_path)
    try:
        _wait_for_port(port)
        url = f"https://127.0.0.1:{port}"

        # 1. Client requests pairing (signed, self-certifying).
        client_node.activate()
        resp = client_mod.send_pairing_request(url, my_url="http://client:8787")
        assert resp["ok"] and resp["body"]["status"] == "pending", resp

        # 2. Host sees a pending request and approves it.
        pending = pairing.list_pairing_requests(server_node.conn(), pending_only=True)
        assert len(pending) == 1
        assert pairing.approve_pairing(server_node.conn(), pending[0]["id"])["ok"]

        # 3. Client registers the peer and pulls — authenticated by signature.
        client_conn = client_node.conn()
        client_mod.add_peer(client_conn, peer_node_id="server-peer", url=url, bearer_token="")
        client_node.activate()
        result = client_mod.pull_from_peer(client_node.conn(), "server-peer",
                                           embedder=fake_embedder)
        assert result.ok, f"pull failed: {result.error}"
        assert result.row_counts["memories"] == 1
        got = client_node.conn().execute(
            "SELECT content FROM memories WHERE origin_id = 'sig-1'").fetchone()
        assert got[0] == "signed-payload"
    finally:
        httpd.shutdown()


def test_unapproved_peer_rejected(make_node):
    """A signed request from a node that was never approved → 401."""
    server_node = make_node("server")
    client_node = make_node("client")
    port = _free_port()
    from cairn.sync.server import serve_in_thread
    from cairn.sync import client as client_mod
    httpd, _ = serve_in_thread(host="127.0.0.1", port=port, db_path=server_node.db_path)
    try:
        _wait_for_port(port)
        url = f"https://127.0.0.1:{port}"
        client_node.activate()
        client_mod.add_peer(client_node.conn(), peer_node_id="server-peer", url=url, bearer_token="")
        # No pairing/approval done → server has no pinned key for us.
        result = client_mod.pull_from_peer(client_node.conn(), "server-peer")
        assert not result.ok
        assert "401" in (result.error or ""), result.error
    finally:
        httpd.shutdown()


def test_revoked_peer_rejected(make_node, fake_embedder):
    server_node = make_node("server")
    client_node = make_node("client")
    port = _free_port()
    from cairn.sync.server import serve_in_thread
    from cairn.sync import client as client_mod, pairing
    httpd, _ = serve_in_thread(host="127.0.0.1", port=port, db_path=server_node.db_path)
    try:
        _wait_for_port(port)
        url = f"https://127.0.0.1:{port}"
        client_node.activate()
        client_mod.send_pairing_request(url)
        pending = pairing.list_pairing_requests(server_node.conn(), pending_only=True)
        approved = pairing.approve_pairing(server_node.conn(), pending[0]["id"])
        pairing.revoke_peer(server_node.conn(), approved["peer_node_id"])
        client_mod.add_peer(client_node.conn(), peer_node_id="server-peer", url=url, bearer_token="")
        client_node.activate()
        result = client_mod.pull_from_peer(client_node.conn(), "server-peer", embedder=fake_embedder)
        assert not result.ok and "401" in (result.error or "")
    finally:
        httpd.shutdown()


# ───────────────── approval polling (/pair/status + refresh_outbound) ────────

def test_pair_status_reflects_approval(make_node):
    server_node = make_node("server")
    client_node = make_node("client")
    port = _free_port()
    from cairn.sync.server import serve_in_thread
    from cairn.sync import client as client_mod, pairing
    httpd, _ = serve_in_thread(host="127.0.0.1", port=port, db_path=server_node.db_path)
    try:
        _wait_for_port(port)
        url = f"https://127.0.0.1:{port}"
        client_node.activate()
        client_mod.send_pairing_request(url, conn=client_node.conn())
        # Before approval: pending
        client_node.activate()
        assert client_mod.check_pair_status(url) == "pending"
        # Approve on the server side
        pend = pairing.list_pairing_requests(server_node.conn(), pending_only=True)
        pairing.approve_pairing(server_node.conn(), pend[0]["id"])
        # After approval: approved
        client_node.activate()
        assert client_mod.check_pair_status(url) == "approved"
    finally:
        httpd.shutdown()


def test_refresh_outbound_promotes_and_pulls(make_node, fake_embedder):
    server_node = make_node("server")
    client_node = make_node("client")
    server_node.insert_memory(type_="fact", topic="t", content="promoted", origin_id="promo-1")
    port = _free_port()
    from cairn.sync.server import serve_in_thread
    from cairn.sync import client as client_mod, pairing
    httpd, _ = serve_in_thread(host="127.0.0.1", port=port, db_path=server_node.db_path)
    try:
        _wait_for_port(port)
        url = f"https://127.0.0.1:{port}"
        # 1. Client requests (records outbound) and server approves.
        client_node.activate()
        client_mod.send_pairing_request(url, conn=client_node.conn())
        pend = pairing.list_pairing_requests(server_node.conn(), pending_only=True)
        pairing.approve_pairing(server_node.conn(), pend[0]["id"])
        # 2. Client polls + promotes the approved outbound request to a peer.
        client_node.activate()
        refreshed = client_mod.refresh_outbound(client_node.conn())
        assert any(r["status"] == "approved" for r in refreshed), refreshed
        peer = client_node.conn().execute(
            "SELECT peer_node_id FROM sync_peers WHERE status='approved' AND url=?", (url,)
        ).fetchone()
        assert peer is not None, "promotion did not create an approved peer"
        # 3. Now the client can pull.
        client_node.activate()
        res = client_mod.pull_from_peer(client_node.conn(), peer[0], embedder=fake_embedder)
        assert res.ok, res.error
        got = client_node.conn().execute(
            "SELECT content FROM memories WHERE origin_id='promo-1'").fetchone()
        assert got[0] == "promoted"
    finally:
        httpd.shutdown()
