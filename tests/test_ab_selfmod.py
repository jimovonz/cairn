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


def _seed_delivery(conn, memory_id, engaged, grade=None):
    conn.execute(
        "INSERT INTO memory_deliveries (session_id, memory_id, engaged, grade) VALUES (?, ?, ?, ?)",
        ("sess", memory_id, engaged, grade),
    )


def _seed_arm(durable, eph, version, n, engaged_count, grade=None, graded_count=None):
    d = sqlite3.connect(durable)
    e = sqlite3.connect(eph)
    gc = n if graded_count is None else graded_count
    for i in range(n):
        mid = _seed_memory(d, version)
        d.commit()
        g = grade if (grade is not None and i < gc) else None
        _seed_delivery(e, mid, 1 if i < engaged_count else 0, g)
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
    n, pct, _gn, _grade = sm._arm_stats(durable, eph, "genA-v4")
    assert n == 10
    assert pct == 30.0


def test_arm_stats_no_deliveries_returns_zero():
    durable, eph, td = _fresh_dbs()
    n, pct, _gn, _grade = sm._arm_stats(durable, eph, "genA-v4")
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
    with patch.object(sm, "CONFIG_PATH", config_path), \
         patch("cairn.config.AB_B_QUEUE", []):
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
    with patch.object(sm, "CONFIG_PATH", config_path), \
         patch("cairn.config.AB_B_QUEUE", []):
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


def test_assess_grade_promotes_when_engagement_within_band():
    durable, eph, td = _fresh_dbs()
    # Engagement dead-level (within band), but B grades clearly higher.
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=15, grade=1, graded_count=40)
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=15, grade=2, graded_count=40)
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with patch.object(sm, "CONFIG_PATH", config_path):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "promoted"
    assert "grade-promote" in result["decision_reason"]
    assert result["avg_grade_a"] == 1.0 and result["avg_grade_b"] == 2.0


def test_assess_grade_rejects_when_engagement_within_band():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=15, grade=2, graded_count=40)
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=15, grade=1, graded_count=40)
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with patch.object(sm, "CONFIG_PATH", config_path):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "rejected"
    assert "grade-reject" in result["decision_reason"]


def test_assess_grade_ignored_below_min_graded_n():
    durable, eph, td = _fresh_dbs()
    # B grades much higher but too few graded deliveries -> stays inconclusive.
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=15, grade=1, graded_count=20)
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=15, grade=3, graded_count=20)
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with patch.object(sm, "CONFIG_PATH", config_path):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "inconclusive"
    assert "grade-promote" not in (result["decision_reason"] or "")


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


# ---------------------------------------------------------------------------
# AB_B_QUEUE auto-advance
# ---------------------------------------------------------------------------

_QUEUE = [
    {"version": "genB-v3", "label": "first candidate", "instruction": "try hypothesis 1"},
    {"version": "genB-v4", "label": "second candidate", "instruction": "try hypothesis 2"},
]


def test_next_untried_candidate_skips_tried():
    durable, eph, td = _fresh_dbs()
    _insert_row(durable, _row(cand="genB-v3"))
    with patch("cairn.config.AB_B_QUEUE", _QUEUE):
        candidate = sm._next_untried_candidate(durable)
    assert candidate["version"] == "genB-v4"


def test_next_untried_candidate_none_when_queue_empty():
    durable, eph, td = _fresh_dbs()
    with patch("cairn.config.AB_B_QUEUE", []):
        assert sm._next_untried_candidate(durable) is None


def test_promote_advances_to_next_queue_candidate():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=8)   # 20%
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=20)  # 50%
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with patch.object(sm, "CONFIG_PATH", config_path), \
         patch("cairn.config.AB_B_QUEUE", _QUEUE):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "promoted"
    with open(config_path) as f:
        text = f.read()
    assert 'GENERATION_PROMPT_VERSION = "genA-v5"' in text
    assert "AB_TEST_ENABLED = True" in text
    assert "AB_TEST_ENABLED = False" not in text
    assert 'AB_ARM_VERSIONS = {"A": GENERATION_PROMPT_VERSION, "B": "genB-v3"}' in text
    assert "try hypothesis 1" in text


def test_reject_advances_to_next_queue_candidate():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=20)  # 50%
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=8)   # 20%
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    with patch.object(sm, "CONFIG_PATH", config_path), \
         patch("cairn.config.AB_B_QUEUE", _QUEUE):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "rejected"
    with open(config_path) as f:
        text = f.read()
    assert 'GENERATION_PROMPT_VERSION = "genA-v4"' in text  # base untouched on reject
    assert "AB_TEST_ENABLED = True" in text
    assert 'AB_ARM_VERSIONS = {"A": GENERATION_PROMPT_VERSION, "B": "genB-v3"}' in text
    assert "try hypothesis 1" in text


def test_disables_when_queue_exhausted():
    durable, eph, td = _fresh_dbs()
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=8)
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=20)
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    # Both queue versions already tried -> nothing left to advance to.
    _insert_row(durable, _row(cand="genB-v3"))
    _insert_row(durable, _row(cand="genB-v4"))
    with patch.object(sm, "CONFIG_PATH", config_path), \
         patch("cairn.config.AB_B_QUEUE", _QUEUE):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    assert result["status"] == "promoted"
    with open(config_path) as f:
        text = f.read()
    assert "AB_TEST_ENABLED = False" in text
    assert "queue exhausted" in text


def test_show_queue_reports_status_per_candidate():
    durable, eph, td = _fresh_dbs()
    row = _insert_row(durable, _row(cand="genB-v3"))
    conn = sqlite3.connect(durable)
    conn.execute("UPDATE ab_experiments SET status = ? WHERE id = ?", ("rejected", row["id"]))
    conn.commit()
    conn.close()
    with patch.object(sm, "DB_PATH", durable), \
         patch("cairn.config.AB_B_QUEUE", _QUEUE):
        report = sm.show_queue()
    by_version = {r["version"]: r for r in report}
    assert by_version["genB-v3"]["status"] == "rejected"
    assert by_version["genB-v4"]["status"] == "untried"


# ---------------------------------------------------------------------------
# Compliance gate (mechanical meta-content check vs engaged_pct proxy)
# ---------------------------------------------------------------------------

_META_CHECK = {"genB-v2": "meta"}


def _seed_memories_with_content(durable, version, contents, created_at=None):
    conn = sqlite3.connect(durable)
    for content in contents:
        if created_at is not None:
            conn.execute(
                "INSERT INTO memories (type, topic, content, source_ref, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("fact", "t", content, version, created_at),
            )
        else:
            conn.execute(
                "INSERT INTO memories (type, topic, content, source_ref) VALUES (?, ?, ?, ?)",
                ("fact", "t", content, version),
            )
    conn.commit()
    conn.close()


def test_is_meta_flagged_matches_target_patterns():
    assert sm._is_meta_flagged("session handoff", "Branch: main")
    assert sm._is_meta_flagged("t", "In progress: refactor the parser")
    assert sm._is_meta_flagged("t", "captured what was discussed this conversation")
    assert not sm._is_meta_flagged("cairn schema v14", "added memory_qf_embeddings sidecar table")
    # Unknown check name -> no patterns -> never flagged.
    assert not sm._is_meta_flagged("session handoff", "Branch: main", check="nonexistent")


def test_meta_stats_computes_flagged_rate():
    durable, eph, td = _fresh_dbs()
    _seed_memories_with_content(durable, "v1", [
        "session handoff: in progress on X",
        "ordinary fact about the codebase",
        "another ordinary fact",
    ])
    n, pct, flagged = sm._meta_stats(durable, "v1")
    assert n == 3
    assert flagged == 1
    assert abs(pct - (100.0 / 3)) < 0.01


def test_meta_stats_empty_version_returns_none():
    durable, eph, td = _fresh_dbs()
    n, pct, flagged = sm._meta_stats(durable, "nonexistent")
    assert (n, pct, flagged) == (0, None, 0)


def test_meta_stats_since_excludes_pre_window():
    durable, eph, td = _fresh_dbs()
    _seed_memories_with_content(durable, "v1", ["session handoff old"] * 5,
                                created_at="2020-01-01 00:00:00")
    _seed_memories_with_content(durable, "v1", ["ordinary fact new"] * 5,
                                created_at="2026-07-01 00:00:00")
    # Whole pool: 10 rows, 5 flagged.
    assert sm._meta_stats(durable, "v1") == (10, 50.0, 5)
    # In-window only: the 5 pre-2026 meta rows are excluded.
    n, pct, flagged = sm._meta_stats(durable, "v1", "meta", "2026-01-01 00:00:00")
    assert (n, pct, flagged) == (5, 0.0, 0)


def test_fisher_exact_two_sided_bounded_and_symmetric():
    # Identical rates -> p close to 1 (no evidence of a difference).
    p_same = sm._fisher_exact_two_sided(5, 15, 5, 15)
    assert 0.9 <= p_same <= 1.0
    # Starkly different rates at reasonable N -> small p.
    p_diff = sm._fisher_exact_two_sided(0, 30, 20, 10)
    assert 0.0 <= p_diff < 0.001
    # Table transposed (a<->c, b<->d) should give the same p-value.
    assert abs(sm._fisher_exact_two_sided(0, 30, 20, 10)
               - sm._fisher_exact_two_sided(20, 10, 0, 30)) < 1e-9


def test_compliance_gate_skipped_for_unregistered_candidate():
    # A candidate with no registered check is never gated, even if its
    # content is dominated by meta-bookkeeping — its engagement verdict stands.
    durable, eph, td = _fresh_dbs()
    _seed_memories_with_content(durable, "genA-v4", ["ordinary fact"] * 30,
                                created_at="2026-07-01 00:00:00")
    _seed_memories_with_content(durable, "genB-v9", ["session handoff"] * 30,
                                created_at="2026-07-01 00:00:00")
    with patch("cairn.config.AB_COMPLIANCE_CHECKS", _META_CHECK), \
         patch("cairn.config.AB_B_QUEUE", []):
        gate = sm._compliance_gate(durable, "genA-v4", "genB-v9")
    assert gate["compliance_blocked"] is False
    assert gate["meta_n_a"] is None
    assert gate["meta_p_value"] is None


def test_compliance_gate_blocks_significant_regression():
    durable, eph, td = _fresh_dbs()
    _seed_memories_with_content(durable, "genA-v4", ["ordinary fact"] * 30,
                                created_at="2026-07-01 00:00:00")
    _seed_memories_with_content(
        durable, "genB-v2",
        ["session handoff: in progress"] * 20 + ["ordinary fact"] * 10,
        created_at="2026-07-01 00:00:00",
    )
    with patch("cairn.config.AB_COMPLIANCE_CHECKS", _META_CHECK), \
         patch("cairn.config.AB_B_QUEUE", []):
        gate = sm._compliance_gate(durable, "genA-v4", "genB-v2")
    assert gate["meta_pct_a"] == 0.0
    assert abs(gate["meta_pct_b"] - (200.0 / 3)) < 0.01
    assert gate["compliance_blocked"] is True
    assert gate["meta_p_value"] < 0.05


def test_compliance_gate_excludes_pre_experiment_history():
    # Arm A carries a long PRE-experiment tail of meta content under the same
    # base_version; only its in-window rows must count, so the candidate's
    # in-window meta content is NOT falsely compared against stale history.
    durable, eph, td = _fresh_dbs()
    _seed_memories_with_content(durable, "genA-v4", ["session handoff old"] * 40,
                                created_at="2020-01-01 00:00:00")
    _seed_memories_with_content(durable, "genA-v4", ["ordinary fact"] * 30,
                                created_at="2026-07-01 00:00:00")
    _seed_memories_with_content(durable, "genB-v2", ["session handoff"] * 20 + ["ordinary fact"] * 10,
                                created_at="2026-07-01 00:00:00")
    with patch("cairn.config.AB_COMPLIANCE_CHECKS", _META_CHECK), \
         patch("cairn.config.AB_B_QUEUE", []):
        gate = sm._compliance_gate(durable, "genA-v4", "genB-v2")
    # Arm A's counted pool is the 30 in-window rows only (pre-2026 excluded).
    assert gate["meta_n_a"] == 30
    assert gate["meta_pct_a"] == 0.0
    assert gate["compliance_blocked"] is True


def test_compliance_gate_does_not_block_similar_rates():
    durable, eph, td = _fresh_dbs()
    _seed_memories_with_content(durable, "genA-v4",
                                ["session handoff"] * 3 + ["ordinary fact"] * 27,
                                created_at="2026-07-01 00:00:00")
    _seed_memories_with_content(durable, "genB-v2",
                                ["session handoff"] * 4 + ["ordinary fact"] * 26,
                                created_at="2026-07-01 00:00:00")
    with patch("cairn.config.AB_COMPLIANCE_CHECKS", _META_CHECK), \
         patch("cairn.config.AB_B_QUEUE", []):
        gate = sm._compliance_gate(durable, "genA-v4", "genB-v2")
    assert gate["compliance_blocked"] is False


def test_promote_blocked_by_compliance_rejects_and_advances():
    durable, eph, td = _fresh_dbs()
    # Engagement gap alone would promote B (20% -> 50%)...
    _seed_arm(durable, eph, "genA-v4", n=40, engaged_count=8)
    _seed_arm(durable, eph, "genB-v2", n=40, engaged_count=20)
    # ...but B's memory content is dominated by the exact meta-bookkeeping
    # pattern its hypothesis was meant to suppress; stamp both arms to the same
    # in-window timestamp so the go-live bound includes them.
    conn = sqlite3.connect(durable)
    conn.execute(
        "UPDATE memories SET content = 'session handoff: in progress this session', "
        "created_at = '2026-07-01 00:00:00' WHERE source_ref = 'genB-v2'"
    )
    conn.execute(
        "UPDATE memories SET created_at = '2026-07-01 00:00:00' WHERE source_ref = 'genA-v4'"
    )
    conn.commit()
    conn.close()
    row = _insert_row(durable, _row())
    config_path = _fresh_config(td)
    queue = [{"version": "genB-v3", "label": "next up", "instruction": "hypothesis 3"}]
    with patch.object(sm, "CONFIG_PATH", config_path), \
         patch("cairn.config.AB_COMPLIANCE_CHECKS", _META_CHECK), \
         patch("cairn.config.AB_B_QUEUE", queue):
        result = sm.assess_experiment(row, db_path=durable, eph_path=eph, dry_run=False)
    # Spurious engagement win -> rejected, and the queue advanced to genB-v3.
    assert result["status"] == "rejected"
    assert result["compliance_blocked"] is True
    assert "PROMOTION BLOCKED" in result["decision_reason"]
    with open(config_path) as f:
        text = f.read()
    assert 'GENERATION_PROMPT_VERSION = "genA-v4"' in text  # not bumped (B lost)
    assert "AB_TEST_ENABLED = True" in text                 # still testing
    assert 'B": "genB-v3"' in text                          # advanced to next
    conn = sqlite3.connect(durable)
    stored = conn.execute(
        "SELECT status, compliance_blocked FROM ab_experiments WHERE id = ?",
        (row["id"],),
    ).fetchone()
    conn.close()
    assert stored[0] == "rejected"
    assert stored[1] == 1
