"""Changeset extract + apply: bidirectional sync, LWW ordering, confidence convergence."""

from __future__ import annotations


def test_extract_excludes_private_visibility(node):
    public_oid = "pub-1"
    private_oid = "priv-1"
    node.insert_memory(type_="fact", topic="public", content="visible", origin_id=public_oid)
    node.insert_memory(type_="fact", topic="secret", content="hidden", origin_id=private_oid)
    conn = node.conn()
    conn.execute("UPDATE memories SET visibility = 'private' WHERE origin_id = ?", (private_oid,))
    conn.commit()

    from cairn.sync.changeset import extract_changeset
    payload = extract_changeset(conn, since_lamport_by_node={})
    origins = {r["origin_id"] for r in payload["memories"]}
    assert public_oid in origins
    assert private_oid not in origins, "private memory leaked into changeset"


def test_unidirectional_pull_replicates_rows(make_node, fake_embedder):
    a = make_node("A")
    b = make_node("B")
    a.insert_memory(type_="fact", topic="t1", content="from-A", origin_id="a-1")
    a.insert_memory(type_="decision", topic="t2", content="from-A-2", origin_id="a-2")

    from cairn.sync.changeset import extract_changeset, apply_changeset
    payload = extract_changeset(a.conn(), since_lamport_by_node={})
    res = apply_changeset(b.conn(), payload, embedder=fake_embedder)
    assert res.memories_inserted == 2
    assert res.errors == []

    b_origins = {r[0] for r in b.conn().execute("SELECT origin_id FROM memories").fetchall()}
    assert {"a-1", "a-2"}.issubset(b_origins)


def test_apply_is_idempotent(make_node, fake_embedder):
    a = make_node("A")
    b = make_node("B")
    a.insert_memory(type_="fact", topic="t", content="c", origin_id="a-x")
    from cairn.sync.changeset import extract_changeset, apply_changeset
    payload = extract_changeset(a.conn(), since_lamport_by_node={})
    apply_changeset(b.conn(), payload, embedder=fake_embedder)
    res2 = apply_changeset(b.conn(), payload, embedder=fake_embedder)
    # Second apply is LWW-no-op (incoming.lamport == local.lamport)
    assert res2.memories_inserted == 0
    assert res2.memories_updated == 0
    assert res2.memories_skipped_lww == 1


def test_lww_higher_lamport_wins(make_node, fake_embedder):
    a = make_node("A")
    b = make_node("B")
    a.insert_memory(type_="fact", topic="t", content="A-original", origin_id="contested")

    # B receives A's row
    from cairn.sync.changeset import extract_changeset, apply_changeset
    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)

    # Both nodes edit the same row independently (offline)
    from cairn.sync.identity import bump_lamport
    a_conn = a.conn()
    lam_a = bump_lamport(a_conn)
    a_conn.execute(
        "UPDATE memories SET content = ?, lamport = ?, updated_by_node = ? WHERE origin_id = ?",
        ("A-edit", lam_a, a.node_id, "contested"),
    )
    a_conn.commit()

    b_conn = b.conn()
    lam_b = bump_lamport(b_conn)
    # Force B's lamport to be higher to make the test deterministic
    lam_b = max(lam_b, lam_a + 5)
    b_conn.execute(
        "UPDATE memories SET content = ?, lamport = ?, updated_by_node = ? WHERE origin_id = ?",
        ("B-edit", lam_b, b.node_id, "contested"),
    )
    b_conn.commit()

    # A pulls B's changes — B's edit wins (higher lamport)
    apply_changeset(a.conn(), extract_changeset(b.conn(), {}), embedder=fake_embedder)
    final_a = a.conn().execute(
        "SELECT content, lamport FROM memories WHERE origin_id = 'contested'"
    ).fetchone()
    assert final_a[0] == "B-edit", f"LWW failed: A still has {final_a[0]!r}"

    # B pulls A's changes — A's older edit must NOT clobber B
    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)
    final_b = b.conn().execute(
        "SELECT content FROM memories WHERE origin_id = 'contested'"
    ).fetchone()
    assert final_b[0] == "B-edit"


def test_confidence_log_converges_across_nodes(make_node, fake_embedder):
    """Two nodes both vote + on the same memory while offline — converges deterministically."""
    a = make_node("A")
    b = make_node("B")
    a.insert_memory(type_="fact", topic="t", content="c", origin_id="vote-target")

    from cairn.sync.changeset import extract_changeset, apply_changeset
    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)

    # Both nodes vote + independently
    a.confidence_vote("vote-target", "+", reason="A says yes")
    b.confidence_vote("vote-target", "+", reason="B says yes")

    # Cross-sync
    apply_changeset(a.conn(), extract_changeset(b.conn(), {}), embedder=fake_embedder)
    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)

    # Both nodes should have IDENTICAL confidence (deterministic recompute)
    conf_a = a.conn().execute("SELECT confidence FROM memories WHERE origin_id = 'vote-target'").fetchone()[0]
    conf_b = b.conn().execute("SELECT confidence FROM memories WHERE origin_id = 'vote-target'").fetchone()[0]
    assert conf_a == conf_b, f"non-convergent confidence: A={conf_a} B={conf_b}"
    # And BOTH boosts must have applied (not just one)
    # Default 0.7, two saturating boosts: c1 = 0.7 + 0.1*0.3 = 0.73, c2 = 0.73 + 0.1*0.27 = 0.757
    assert conf_a > 0.75, f"only one boost applied: {conf_a}"


def test_contradiction_annotation_propagates(make_node, fake_embedder):
    a = make_node("A")
    b = make_node("B")
    a.insert_memory(type_="fact", topic="t", content="c", origin_id="bad-memory")
    from cairn.sync.changeset import extract_changeset, apply_changeset
    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)

    a.confidence_vote("bad-memory", "-!", reason="superseded by GCE approach")
    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)

    arch = b.conn().execute(
        "SELECT archived_reason FROM memories WHERE origin_id = 'bad-memory'"
    ).fetchone()[0]
    assert arch == "superseded by GCE approach"


def test_tombstone_propagates(make_node, fake_embedder):
    a = make_node("A")
    b = make_node("B")
    a.insert_memory(type_="fact", topic="t", content="c", origin_id="to-delete")
    from cairn.sync.changeset import extract_changeset, apply_changeset
    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)

    from cairn.sync.identity import bump_lamport
    a_conn = a.conn()
    lam = bump_lamport(a_conn)
    a_conn.execute(
        "UPDATE memories SET deleted_at = '2026-05-11', updated_by_node = ?, lamport = ? "
        "WHERE origin_id = ?",
        (a.node_id, lam, "to-delete"),
    )
    a_conn.commit()

    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)
    deleted = b.conn().execute(
        "SELECT deleted_at FROM memories WHERE origin_id = 'to-delete'"
    ).fetchone()[0]
    assert deleted is not None, "tombstone did not propagate"


def test_three_node_gossip_converges(make_node, fake_embedder):
    """A→B→C: rows originated at A reach C via B without A talking to C."""
    a = make_node("A")
    b = make_node("B")
    c = make_node("C")
    a.insert_memory(type_="fact", topic="t", content="from-A", origin_id="a-row")

    from cairn.sync.changeset import extract_changeset, apply_changeset
    # A → B
    apply_changeset(b.conn(), extract_changeset(a.conn(), {}), embedder=fake_embedder)
    # B → C (gossip: C never talks to A)
    apply_changeset(c.conn(), extract_changeset(b.conn(), {}), embedder=fake_embedder)

    c_row = c.conn().execute(
        "SELECT content, created_by_node FROM memories WHERE origin_id = 'a-row'"
    ).fetchone()
    assert c_row[0] == "from-A"
    assert c_row[1] == a.node_id, "creator node must travel through gossip"


def test_schema_version_mismatch_rejected(make_node, fake_embedder):
    a = make_node("A")
    a.insert_memory(type_="fact", topic="t", content="c", origin_id="x")
    from cairn.sync.changeset import extract_changeset, apply_changeset
    payload = extract_changeset(a.conn(), {})
    payload["schema_version"] = 99
    b = make_node("B")
    import pytest
    with pytest.raises(ValueError, match="schema_version mismatch"):
        apply_changeset(b.conn(), payload, embedder=fake_embedder)
