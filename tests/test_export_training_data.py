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


def test_weak_engagement_labels_are_flagged_and_grouped():
    """They must stay distinguishable from agent-rg after transport: the trainer
    merges them only AFTER split_by_query, so held-out stays pure agent-rg and
    the beat-the-incumbent gate is never judged on weak labels."""
    with patch("cairn.train_reranker.load_engagement_groups",
               lambda min_pos=None: {"eng:abc": [("q", "m", 3), ("q", "m2", 0)]}):
        rows = ex.collect_engagement()
    assert rows and all(r["weak"] is True for r in rows)
    assert all(r["group"].startswith("eng:") for r in rows)


def test_archive_round_trips_through_the_trainer_loader():
    durable, eph, td = _fresh_dbs()
    _seed(durable, eph, n=4)
    out = os.path.join(td, "exports")
    with patch("cairn.relevance._durable_path", lambda p=None: durable), \
         patch("cairn.relevance._eph_path", lambda p=None: eph), \
         patch("cairn.train_reranker.load_engagement_groups", lambda min_pos=None: {}):
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
