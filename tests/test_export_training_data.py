"""Portable label export (cairn/export_training_data.py).

The contract under test is CROSS-NODE safety. An export is consumed on a
machine whose durable DB is unrelated to the exporting one, so the failure mode
these guard against is silent: labels attached to passages no agent ever judged,
or weak labels leaking into a held-out set the deploy gate is measured on.
Neither raises; both quietly corrupt a training run.
"""

import gzip
import json
import os
import sys
import tempfile
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import export_training_data as ex, init_db


def _fresh_dbs():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "eph.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    return durable, eph, td


def _seed(durable, eph, n=3, grade_base=0):
    d = sqlite3.connect(durable)
    e = sqlite3.connect(eph)
    ids = []
    for i in range(n):
        cur = d.execute(
            "INSERT INTO memories (type, topic, content, keywords) VALUES (?,?,?,?)",
            ("fact", f"topic{i}", f"content number {i}", f"kw{i}"))
        ids.append(cur.lastrowid)
    for i, mid in enumerate(ids):
        e.execute(
            "INSERT INTO memory_deliveries (session_id, memory_id, context_text, grade) "
            "VALUES (?,?,?,?)", ("sess", mid, "the user asked about X", grade_base + i))
    d.commit(); e.commit(); d.close(); e.close()
    return ids


def _collect(durable, eph):
    with patch("cairn.relevance._durable_path", lambda p=None: durable), \
         patch("cairn.relevance._eph_path", lambda p=None: eph):
        return ex.collect_rg()


def test_export_never_emits_memory_id():
    """The core cross-node guard. load_groups(enrich=True) re-renders passages
    from `memory_id` against the LOCAL durable DB; ids are per-node
    autoincrements, so emitting one lets an import silently swap in an unrelated
    memory's text. Provenance rides as `src_memory_id`, which no loader reads."""
    durable, eph, _ = _fresh_dbs()
    _seed(durable, eph)
    rows, _ = _collect(durable, eph)
    assert rows
    for r in rows:
        assert "memory_id" not in r, "memory_id would be resolved against the wrong DB"
        assert isinstance(r["src_memory_id"], int)


def test_enrich_on_an_imported_file_is_a_no_op():
    """Consequence of the above, asserted end-to-end through the real loader:
    a receiving node that passes --enrich must get the shipped passages back,
    not whatever its own DB holds at those ids."""
    durable, eph, td = _fresh_dbs()
    _seed(durable, eph)
    rows, _ = _collect(durable, eph)
    path = os.path.join(td, "labels.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from cairn.train_reranker import load_groups
    plain = load_groups(path)
    enriched = load_groups(path, enrich=True)
    assert [t for v in plain.values() for t in v] == \
           [t for v in enriched.values() for t in v]


def test_rendered_passages_are_self_contained():
    """The receiving node has none of this node's memories, so the passage text
    must travel with the label rather than being resolvable later."""
    durable, eph, _ = _fresh_dbs()
    _seed(durable, eph)
    rows, _ = _collect(durable, eph)
    for r in rows:
        assert r["mem"] and r["query"]
        assert "content number" in r["mem"]


def test_deleted_memory_drops_the_label_rather_than_shipping_an_empty_passage():
    """A grade is a judgement about specific text. Without it the label cannot be
    reattached to anything, and an empty `mem` would train on the empty string."""
    durable, eph, _ = _fresh_dbs()
    ids = _seed(durable, eph)
    d = sqlite3.connect(durable)
    d.execute("DELETE FROM memories WHERE id=?", (ids[0],))
    d.commit(); d.close()
    rows, dropped = _collect(durable, eph)
    assert dropped == 1
    assert len(rows) == len(ids) - 1
    assert all(r["mem"] for r in rows)


def _seed_engagement(durable, eph, method="lexical"):
    """One two-class group: same context, one engaged positive and one negative.

    `method` is the stratum key — a stratum qualifies only if it recorded BOTH
    classes, so seeding both under one method is what makes these rows survive
    the filter.
    """
    d = sqlite3.connect(durable)
    e = sqlite3.connect(eph)
    ids = []
    for i in range(2):
        cur = d.execute(
            "INSERT INTO memories (type, topic, content) VALUES (?,?,?)",
            ("fact", f"e{i}", f"engagement content {i}"))
        ids.append(cur.lastrowid)
    ctx = "shared context for the engagement group"
    e.execute("INSERT INTO memory_deliveries (session_id, memory_id, context_text, "
              "engaged, engaged_score, engaged_method) VALUES (?,?,?,?,?,?)",
              ("s", ids[0], ctx, 1, 0.9, method))
    e.execute("INSERT INTO memory_deliveries (session_id, memory_id, context_text, "
              "engaged, engaged_score, engaged_method) VALUES (?,?,?,?,?,?)",
              ("s", ids[1], ctx, 0, 0.0, method))
    d.commit(); e.commit(); d.close(); e.close()
    return ids


def _collect_eng(durable, eph):
    with patch("cairn.relevance._durable_path", lambda p=None: durable), \
         patch("cairn.relevance._eph_path", lambda p=None: eph):
        return ex.collect_engagement()


def test_weak_engagement_labels_are_flagged_and_grouped():
    """They must stay distinguishable from agent-rg after transport: the trainer
    merges them only AFTER split_by_query, so held-out stays pure agent-rg and
    the beat-the-incumbent gate is never judged on weak labels."""
    durable, eph, _ = _fresh_dbs()
    _seed_engagement(durable, eph)
    rows, acct = _collect_eng(durable, eph)
    assert rows and all(r["weak"] is True for r in rows)
    assert all(r["group"].startswith("eng:") for r in rows)
    assert {r["grade"] for r in rows} == {0, 3}, "group must carry both classes"


def test_raw_engagement_fields_let_the_receiver_redo_the_filter():
    """The strata filter is the exporter's most consequential judgement, and the
    receiving node has 94.6% single-class rows of its own. Shipping only the
    derived grade would make that judgement unauditable at the far end."""
    durable, eph, _ = _fresh_dbs()
    _seed_engagement(durable, eph)
    rows, acct = _collect_eng(durable, eph)
    for r in rows:
        assert r["engaged_raw"] in (0, 1)
        assert r["engaged_score_raw"] is not None
        assert r["engaged_method"] == "lexical"
    assert acct["candidate_rows"] == 2
    assert sum(acct[k] for k in acct if k.startswith("dropped")) == 0


def test_placeholder_context_layers_do_not_pool_across_sessions():
    """The defect this guards produced 99.8% of a real export's pairs.

    project-bootstrap and correction-bootstrap are standing context with no
    prompt to embed against, so every session shares one context string
    ("project standing context"). Keyed by hash(context) they collapsed 70
    unrelated sessions into a single 608-row pseudo-query worth 68,700 pairs,
    all of them teaching a query-free popularity prior. Keyed by turn, each
    session stays its own group and contributes only what it actually saw.
    """
    durable, eph, _ = _fresh_dbs()
    d = sqlite3.connect(durable)
    e = sqlite3.connect(eph)
    cur = d.execute("INSERT INTO memories (type, topic, content) VALUES ('fact','t','c1')")
    m1 = cur.lastrowid
    cur = d.execute("INSERT INTO memories (type, topic, content) VALUES ('fact','t','c2')")
    m2 = cur.lastrowid
    # Six sessions, all sharing the placeholder context, each seeing both classes.
    for s in range(6):
        for mid, eng, sc in ((m1, 1, 0.9), (m2, 0, 0.0)):
            e.execute(
                "INSERT INTO memory_deliveries (session_id, turn_index, memory_id, "
                "context_text, engaged, engaged_score, engaged_method, layer) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f"sess{s}", 0, mid, "project standing context", eng, sc,
                 "lexical", "project-bootstrap"))
    d.commit(); e.commit(); d.close(); e.close()

    rows, _ = _collect_eng(durable, eph)
    groups = {r["group"] for r in rows}
    assert len(groups) == 6, "each session's turn must stay its own group"

    from cairn.train_reranker import make_pairs
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append((r["query"], r["mem"], r["grade"]))
    pairs = len(make_pairs(by_group, 10**6, min_gap=2))
    # 6 groups x (1 positive x 1 negative) = 6. Pooled by context it would be
    # 6 positives x 6 negatives = 36 — six times the real evidence.
    assert pairs == 6, f"expected 6 within-turn pairs, got {pairs}"


def test_single_class_stratum_is_dropped_and_accounted_for():
    """A positives-only stratum cannot yield a rate; admitting it inflates the
    positive class with a regime that could never produce a negative. Dropping
    it silently would leave the receiver unable to see why the yield is low."""
    durable, eph, _ = _fresh_dbs()
    d = sqlite3.connect(durable)
    e = sqlite3.connect(eph)
    cur = d.execute("INSERT INTO memories (type, topic, content) VALUES ('fact','t','c')")
    mid = cur.lastrowid
    for i in range(3):  # positives only, one stratum
        e.execute("INSERT INTO memory_deliveries (session_id, memory_id, context_text, "
                  "engaged, engaged_score, engaged_method) VALUES (?,?,?,?,?,?)",
                  ("s", mid, f"ctx{i}", 1, 0.9, "untagged-regime"))
    d.commit(); e.commit(); d.close(); e.close()
    rows, acct = _collect_eng(durable, eph)
    assert rows == []
    assert acct["candidate_rows"] == 3
    assert acct["dropped_by_strata_or_grade"] == 3


def test_archive_round_trips_through_the_trainer_loader():
    durable, eph, td = _fresh_dbs()
    _seed(durable, eph, n=4)
    out = os.path.join(td, "exports")
    with patch("cairn.relevance._durable_path", lambda p=None: durable), \
         patch("cairn.relevance._eph_path", lambda p=None: eph):
        counts, archive = ex.build(out, node_id="testnode")

    assert counts["rg_labels"] == 4
    assert os.path.exists(archive)

    manifest = json.load(open(os.path.join(out, "manifest.json")))
    assert manifest["node_id"] == "testnode"
    assert manifest["schema_version"] == ex.SCHEMA_VERSION

    plain = os.path.join(td, "rt.jsonl")
    with gzip.open(os.path.join(out, "relevance_silver.jsonl.gz"), "rt") as f, \
            open(plain, "w") as o:
        o.write(f.read())
    from cairn.train_reranker import load_groups
    assert sum(len(v) for v in load_groups(plain).values()) == 4
