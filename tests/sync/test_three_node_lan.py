"""Three-node simulation — proxy for the 20-engineer LAN rollout.

Spins up three independent cairn nodes (each with its own DB, node_id, and
HTTP server on a random port), wires them as full-mesh peers, runs concurrent
writes, and verifies that gossip converges every node to the same view.
"""

from __future__ import annotations

import socket
import threading
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
    raise TimeoutError(f"port {port} not bound")


def test_three_node_full_mesh_convergence(make_node, fake_embedder):
    nodes = [make_node(f"n{i}") for i in range(3)]
    ports = [_free_port() for _ in range(3)]

    from cairn.sync.server import serve_in_thread
    from cairn.sync.client import add_peer, pull_from_peer

    httpds = []
    try:
        for n, p in zip(nodes, ports):
            httpd, _ = serve_in_thread(host="127.0.0.1", port=p, db_path=n.db_path)
            httpds.append(httpd)
            _wait_for_port(p)

        # Full mesh: every node knows every other node, with a shared bearer token
        token = "shared-mesh-token"
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if i == j:
                    continue
                add_peer(src.conn(), peer_node_id=dst.node_id,
                         url=f"http://127.0.0.1:{ports[j]}", bearer_token=token)
                add_peer(dst.conn(), peer_node_id=src.node_id,
                         url=f"http://127.0.0.1:{ports[i]}", bearer_token=token)

        # Each node writes a unique memory
        for i, n in enumerate(nodes):
            n.insert_memory(type_="fact", topic=f"from-{i}", content=f"node-{i}-data",
                            origin_id=f"oid-{i}")

        # Three rounds of full-mesh pulls — should be plenty for 3 nodes
        # (gossip converges in O(diameter) rounds; mesh diameter is 1)
        for _ in range(3):
            for i, src in enumerate(nodes):
                for j, dst in enumerate(nodes):
                    if i == j:
                        continue
                    res = pull_from_peer(src.conn(), dst.node_id, embedder=fake_embedder)
                    assert res.ok, f"pull {src.node_id[:8]}<-{dst.node_id[:8]} failed: {res.error}"

        # Verify convergence — every node has all 3 origin_ids
        for n in nodes:
            origins = {r[0] for r in n.conn().execute(
                "SELECT origin_id FROM memories WHERE origin_id LIKE 'oid-%'"
            ).fetchall()}
            assert origins == {"oid-0", "oid-1", "oid-2"}, f"node {n.node_id[:8]} missing rows: {origins}"

        # Confidence convergence: every node votes + on oid-0
        for n in nodes:
            n.confidence_vote("oid-0", "+", reason=f"voted by {n.node_id[:8]}")

        for _ in range(3):
            for i, src in enumerate(nodes):
                for j, dst in enumerate(nodes):
                    if i == j:
                        continue
                    pull_from_peer(src.conn(), dst.node_id, embedder=fake_embedder)

        confidences = []
        for n in nodes:
            c = n.conn().execute(
                "SELECT confidence FROM memories WHERE origin_id = 'oid-0'"
            ).fetchone()[0]
            confidences.append(c)
        assert len(set(confidences)) == 1, f"non-convergent confidence across mesh: {confidences}"
        # Three +1 votes, saturating: 0.7 → 0.73 → 0.757 → 0.7813
        assert confidences[0] > 0.78, f"three boosts should accumulate, got {confidences[0]}"

    finally:
        for httpd in httpds:
            httpd.shutdown()


def test_offline_edit_then_reconcile(make_node, fake_embedder):
    """Simulate a node going offline, both sides editing, then reconciling.

    Equivalent of: laptop on a train edits memories; desktop also edits same
    memories; later they sync — deterministic LWW convergence.
    """
    laptop = make_node("laptop")
    desktop = make_node("desktop")

    laptop.insert_memory(type_="fact", topic="t", content="initial", origin_id="shared")

    from cairn.sync.changeset import extract_changeset, apply_changeset
    apply_changeset(desktop.conn(), extract_changeset(laptop.conn(), {}), embedder=fake_embedder)

    # Both nodes vote + offline
    laptop.confidence_vote("shared", "+", reason="laptop says yes")
    desktop.confidence_vote("shared", "+", reason="desktop also yes")

    # Laptop also edits content offline
    from cairn.sync.identity import bump_lamport
    lc = laptop.conn()
    lam = bump_lamport(lc)
    lc.execute(
        "UPDATE memories SET content = ?, lamport = ?, updated_by_node = ? WHERE origin_id = ?",
        ("laptop-edit", lam, laptop.node_id, "shared")
    )
    lc.commit()

    # Reconcile
    apply_changeset(desktop.conn(), extract_changeset(laptop.conn(), {}), embedder=fake_embedder)
    apply_changeset(laptop.conn(), extract_changeset(desktop.conn(), {}), embedder=fake_embedder)

    # Both nodes converge: laptop's content edit (higher lamport) wins,
    # both + votes accumulate
    for label, n in [("laptop", laptop), ("desktop", desktop)]:
        row = n.conn().execute(
            "SELECT content, confidence FROM memories WHERE origin_id = 'shared'"
        ).fetchone()
        assert row[0] == "laptop-edit", f"{label} content didn't converge: {row[0]}"
        assert row[1] > 0.75, f"{label} confidence missing a boost: {row[1]}"
