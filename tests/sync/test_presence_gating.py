"""pull_all must never assume a paired peer is present on the network.

A paired peer is trusted, not necessarily reachable. pull_all gates on the
discovery beacon (eph.discovered_peers): fresh -> pull; never-seen -> first-contact
attempt; seen but stale -> SKIP (offline, no socket). These tests exercise the
gating decision with an unreachable URL (127.0.0.1:1 refuses instantly) so a
"skip" is observably distinct from an "attempt" (network error).

Operational tables live in the attached `eph` DB (schema v12+), so beacons are
seeded via eph.discovered_peers on an attach_ephemeral'd connection.
"""

from __future__ import annotations

from cairn.sync.client import add_peer, pull_all
from cairn.sync.ephemeral import attach_ephemeral


def _add_discovered(conn, node_id, *, age_sec):
    attach_ephemeral(conn)
    conn.execute(
        "INSERT INTO eph.discovered_peers (node_id, url, last_seen) "
        "VALUES (?, ?, datetime('now', ?)) "
        "ON CONFLICT(node_id) DO UPDATE SET last_seen = excluded.last_seen",
        (node_id, "https://127.0.0.1:1/", f"-{age_sec} seconds"),
    )
    conn.commit()


def test_offline_paired_peer_is_skipped_without_a_socket(node):
    """Paired + previously seen + now stale beacon -> skipped, no network call."""
    conn = node.conn()
    add_peer(conn, peer_node_id="PEERX", url="https://127.0.0.1:1/")
    _add_discovered(conn, "PEERX", age_sec=99999)  # seen long ago -> offline
    results = pull_all(node.conn())
    assert len(results) == 1
    assert results[0].peer_node_id == "PEERX"
    assert results[0].ok is False
    assert "offline: skipped" in (results[0].error or "")


def test_fresh_peer_is_attempted(node):
    """Fresh beacon within the online window -> pull is attempted (peer present)."""
    conn = node.conn()
    add_peer(conn, peer_node_id="PEERY", url="https://127.0.0.1:1/")
    _add_discovered(conn, "PEERY", age_sec=1)  # fresh
    results = pull_all(node.conn())
    assert len(results) == 1
    # Attempted (no server at :1) => a network error, NOT a presence skip.
    assert "offline: skipped" not in (results[0].error or "")
    assert results[0].ok is False


def test_never_beaconed_peer_is_first_contact_attempt(node):
    """Approved but no beacon ever (paired by URL) -> attempt once (first contact)."""
    conn = node.conn()
    add_peer(conn, peer_node_id="PEERZ", url="https://127.0.0.1:1/")
    # No discovered_peers row at all.
    results = pull_all(node.conn())
    assert len(results) == 1
    assert "offline: skipped" not in (results[0].error or "")


def test_no_approved_peers_returns_empty(node):
    assert pull_all(node.conn()) == []


def test_record_beacon_writes_to_ephemeral(node):
    """Beacon upserts land in eph.discovered_peers (off the durable write lock)."""
    from cairn.sync import discovery
    conn = node.conn()
    beacon = {"node_id": "P1", "user_id": "u", "url": "https://x/",
              "public_key": "k", "schema_version": 12, "cert_fingerprint": "cf"}
    assert discovery.record_beacon(conn, beacon, self_node_id="ME") is True
    attach_ephemeral(conn)
    n = conn.execute("SELECT COUNT(*) FROM eph.discovered_peers WHERE node_id='P1'").fetchone()[0]
    assert n == 1
