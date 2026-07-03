"""Write-side A/B automated assessment + changeover tests (cairn/ab_selfmod.py)."""

import os
import sys
import tempfile
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import ab_selfmod as sm, init_db


def _fresh_dbs():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "eph.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    return durable, eph, td


def _seed_memory(conn, source_ref):
    cur = conn.execute(
        "INSERT INTO memories (type, topic, content, source_ref) VALUES (?, ?, ?, ?)",
        ("fact", "t", "c", source_ref),
    )
    return cur.lastrowid


def _seed_delivery(conn, memory_id, engaged):
    conn.execute(
        "INSERT INTO memory_deliveries (session_id, memory_id, engaged) VALUES (?, ?, ?)",
        ("sess", memory_id, engaged),
    )


def _seed_arm(durable, eph, version, n, engaged_count):
    d = sqlite3.connect(durable)
    e = sqlite3.connect(eph)
    for i in range(n):
        mid = _seed_memory(d, version)
        d.commit()
        _seed_delivery(e, mid, 1 if i < engaged_count else 0)
    e.commit()
    d.close()
    e.close()


def _fresh_config(tmp_dir, *, enabled=True, base_version="genA-v4",
                   candidate_version="genB-v2", instruction_b="test hypothesis"):
    path = os.path.join(tmp_dir, "config.py")
    with open(path, "w") as f:
        f.write(
            "GENERATION_PROMPT_VERSION = \"{base}\"\n"
            "AB_TEST_ENABLED = {enabled}\n"
            "AB_ARM_VERSIONS = {{\"A\": GENERATION_PROMPT_VERSION, \"B\": \"{cand}\"}}\n"
            "AB_B_INSTRUCTION = (\n"
            "    \"{instr}\"\n"
            ")\n".format(base=base_version, enabled=enabled, cand=candidate_version,
                         instr=instruction_b)
        )
    return path


# ---------------------------------------------------------------------------
# _arm_stats
# ---------------------------------------------------------------------------

def test_arm_stats_aggregates_engaged_pct():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=10, engaged_count=3)
    n, pct = sm._arm_stats(durable, eph, "genA-v4")
    assert n == 10
    assert pct == 30.0


def test_arm_stats_no_deliveries_returns_zero():
    durable, eph, td = _fresh_dbs()
    n, pct = sm._arm_stats(durable, eph, "genA-v4")
    assert n == 0
    assert pct is None


# ---------------------------------------------------------------------------
# get_or_create_running_experiment
# ---------------------------------------------------------------------------

def test_get_or_create_returns_none_when_disabled():
    durable, eph, td = _fresh_dbs()
    with patch.object(sm, "DB_PATH", durable):
        with patch("cairn.config.AB_TEST_ENABLED", False):
            row = sm.get_or_create_running_experiment()
    assert row is None


def test_get_or_create_is_idempotent():
    durable, eph, td = _fresh_dbs()
    with patch.object(sm, "DB_PATH", durable):
        with patch("cairn.config.AB_TEST_ENABLED", True), \
             patch("cairn.config.AB_ARM_VERSIONS", {"A": "genA-v4", "B": "genB-v2"}), \
             patch("cairn.config.AB_B_INSTRUCTION", "hypothesis text"):
            row1 = sm.get_or_create_running_experiment()
            row2 = sm.get_or_create_running_experiment()
    assert row1["id"] == row2["id"]
    conn = sqlite3.connect(durable)
    count = conn.execute("SELECT COUNT(*) FROM ab_experiments").fetchone()[0]
    conn.close()
    assert count == 1


# ---------------------------------------------------------------------------
# assess_experiment — decision rule
# ---------------------------------------------------------------------------

def _row(base="genA-v4", cand="genB-v2", instruction="hyp"):
    return {"id": 1, "instruction_b": instruction, "base_version": base,
            "candidate_version": cand, "status": "running",
            "n_a": 0, "n_b": 0, "engaged_pct_a": None, "engaged_pct_b": None,
            "decision_reason": None}


def _insert_row(durable, row):
    conn = sqlite3.connect(durable)
    cur = conn.execute(
        "INSERT INTO ab_experiments (instruction_b, base_version, candidate_version) "
        "VALUES (?, ?, ?)",
        (row["instruction_b"], row["base_version"], row["candidate_version"]),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    row = dict(row)
    row["id"] = rid
    return row


def test_assess_stays_running_below_min_n():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=10, engaged_count=5)
    _seed_arm(durable, eph, "genB-v2", n=10, engaged_count=9)
    row = _insert_row(durable, _row())
    result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=True)
    assert result["status"] == "running"


def test_assess_promotes_on_large_positive_gap():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=8)   # 20%
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=20)  # 50%
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with patch.object(sm, "CONFIG_PATH", config_path):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "promoted"
    with open(config_path) as f:
        text = f.read()
    assert 'GENERATION_PROMPT_VERSION = "genA-v5"' in text
    assert "AB_TEST_ENABLED = False" in text


def test_assess_rejects_on_large_negative_gap():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=20)  # 50%
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=8)   # 20%
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with patch.object(sm, "CONFIG_PATH", config_path):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "rejected"
    with open(config_path) as f:
        text = f.read()
    assert 'GENERATION_PROMPT_VERSION = "genA-v4"' in text
    assert "AB_TEST_ENABLED = False" in text


def test_assess_inconclusive_within_band_no_config_edit():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=15)  # 37.5%
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=17)  # 42.5%, gap=5
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with patch.object(sm, "CONFIG_PATH", config_path):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "inconclusive"
    with open(config_path) as f:
        text = f.read()
    assert 'GENERATION_PROMPT_VERSION = "genA-v4"' in text
    assert "AB_TEST_ENABLED = True" in text
    conn = sqlite3.connect(durable)
    status = conn.execute("SELECT status, ended_at FROM ab_experiments WHERE id = ?",
                          (row["id"],)).fetchone()
    conn.close()
    assert status[0] == "inconclusive"
    assert status[1] is None


def test_dry_run_never_touches_config():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=8)
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=20)
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with open(config_path) as f:
        before = f.read()
    with patch.object(sm, "CONFIG_PATH", config_path):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=True)
    assert result["status"] == "promoted"
    with open(config_path) as f:
        after = f.read()
    assert before == after
