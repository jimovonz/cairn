"""Tests for the dashboard graph/health additions (5-part dashboard task):

  #2 metrics DB fix + graph-injection surfacing
  #3 dual-embedding (topic_embedding) coverage tile
  #4 exact-duplicate memory group tile
  #1 /api/graph-fleet observability
  #5 /api/graph-explorer + graph-viz generation

Plus the graph_fleet sweep-state round-trip and the pretool metric-flush guard.
"""

import os
import sqlite3
import tempfile
import time
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import dashboard, init_db, config


# ─── DB fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def dbs(monkeypatch):
    """Distinct durable + ephemeral DBs, wired into dashboard + config."""
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "cairn-ephemeral.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    monkeypatch.setattr(dashboard, "DB_PATH", durable)
    monkeypatch.setattr(config, "EPHEMERAL_DB_PATH", eph, raising=False)
    return durable, eph


def _add_memory(conn, topic, content, *, topic_embedding=b"x", deleted=False):
    conn.execute(
        "INSERT INTO memories (type, topic, content, topic_embedding, deleted_at) "
        "VALUES ('fact', ?, ?, ?, ?)",
        (topic, content, topic_embedding, "2026-01-01" if deleted else None),
    )
    conn.commit()


def _add_metric(conn, event, value=None):
    conn.execute("INSERT INTO metrics (event, session_id, detail, value) VALUES (?, 's', NULL, ?)",
                 (event, value))
    conn.commit()


# ─── #3 dual-embedding coverage ─────────────────────────────────────────────

def test_topic_embedding_coverage(dbs):
    durable, _ = dbs
    conn = sqlite3.connect(durable)
    _add_memory(conn, "a", "one", topic_embedding=b"vec")
    _add_memory(conn, "b", "two", topic_embedding=b"vec")
    _add_memory(conn, "c", "three", topic_embedding=None)   # missing embedding
    _add_memory(conn, "d", "gone", topic_embedding=None, deleted=True)  # excluded
    conn.close()
    cov = dashboard._topic_embedding_coverage()
    assert cov["total"] == 3            # deleted excluded
    assert cov["with_topic_embedding"] == 2
    assert cov["pct"] == round(2 / 3 * 100, 1)
    assert cov["warn"] is True          # < 95%


def test_topic_embedding_coverage_full(dbs):
    durable, _ = dbs
    conn = sqlite3.connect(durable)
    _add_memory(conn, "a", "one")
    conn.close()
    cov = dashboard._topic_embedding_coverage()
    assert cov["pct"] == 100.0 and cov["warn"] is False


# ─── #4 exact-duplicate groups ──────────────────────────────────────────────

def test_exact_duplicate_groups(dbs):
    durable, _ = dbs
    conn = sqlite3.connect(durable)
    _add_memory(conn, "dup", "same body")
    _add_memory(conn, "dup", "same body")          # forms a duplicate group
    _add_memory(conn, "dup", "same body")          # still one group
    _add_memory(conn, "unique", "different")
    _add_memory(conn, dashboard._SMOKE_TEST_TOPIC, "smoke")
    _add_memory(conn, dashboard._SMOKE_TEST_TOPIC, "smoke")  # excluded from count
    conn.close()
    dup = dashboard._exact_duplicate_groups()
    assert dup["count"] == 1            # one (topic,content) group, smoke excluded
    assert dup["warn"] is True


def test_exact_duplicate_none(dbs):
    durable, _ = dbs
    conn = sqlite3.connect(durable)
    _add_memory(conn, "a", "one")
    _add_memory(conn, "b", "two")
    conn.close()
    dup = dashboard._exact_duplicate_groups()
    assert dup["count"] == 0 and dup["warn"] is False


def test_api_health_includes_tiles(dbs):
    data, status = dashboard.api_health({})
    assert status == 200
    assert "topic_embedding" in data and "exact_duplicates" in data


# ─── #2 metrics union + graph-injection surfacing ───────────────────────────

def test_api_metrics_unions_and_surfaces_graph(dbs):
    durable, eph = dbs
    # Live metrics land in ephemeral; graph metrics only exist there.
    ec = sqlite3.connect(eph)
    _add_metric(ec, "graph_orientation_injected")
    _add_metric(ec, "graph_file_context_injected")
    _add_metric(ec, "graph_file_context_injected")
    _add_metric(ec, "file_context_injected")
    ec.close()
    data, status = dashboard.api_metrics({})
    assert status == 200
    gi = {r["event"]: r["count"] for r in data["graph_injection"]}
    assert gi.get("graph_orientation_injected") == 1
    assert gi.get("graph_file_context_injected") == 2
    summ = {r["event"]: r["count"] for r in data["summary"]}
    assert summ.get("file_context_injected") == 1


def test_api_metrics_no_double_count_when_main_has_metrics(dbs):
    """When the (legacy) main DB also has a metrics table, counts union without
    duplication or error."""
    durable, eph = dbs
    # Simulate a pre-migration main DB carrying frozen metrics.
    dc = sqlite3.connect(durable)
    dc.execute("CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, event TEXT, "
               "session_id TEXT, detail TEXT, value REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    dc.execute("INSERT INTO metrics (event, session_id) VALUES ('file_context_injected', 's')")
    dc.commit(); dc.close()
    ec = sqlite3.connect(eph)
    _add_metric(ec, "file_context_injected")   # live one
    ec.close()
    data, _ = dashboard.api_metrics({})
    summ = {r["event"]: r["count"] for r in data["summary"]}
    assert summ.get("file_context_injected") == 2   # 1 frozen + 1 live, no dup


# ─── graph.db fixture for fleet/explorer ────────────────────────────────────

NODES = """CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, name TEXT,
  qualified_name TEXT UNIQUE, file_path TEXT, line_start INTEGER, line_end INTEGER,
  language TEXT, is_test INTEGER DEFAULT 0, updated_at REAL);"""
EDGES = """CREATE TABLE edges (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT,
  source_qualified TEXT, target_qualified TEXT, file_path TEXT, line INTEGER DEFAULT 0);"""
META = "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);"


def _build_graph_repo(root, head_sha="abc123"):
    gdir = os.path.join(root, ".code-review-graph")
    os.makedirs(gdir, exist_ok=True)
    db = os.path.join(gdir, "graph.db")
    conn = sqlite3.connect(db)
    conn.executescript(NODES + EDGES + META)
    now = time.time()
    conn.execute("INSERT INTO nodes (kind,name,qualified_name,file_path,line_start,line_end,language,is_test,updated_at)"
                 " VALUES ('Function','foo','m.foo','m.py',1,9,'python',0,?)", (now,))
    conn.execute("INSERT INTO nodes (kind,name,qualified_name,file_path,line_start,line_end,language,is_test,updated_at)"
                 " VALUES ('Function','bar','m.bar','m.py',10,20,'python',0,?)", (now,))
    conn.execute("INSERT INTO edges (kind,source_qualified,target_qualified,file_path,line)"
                 " VALUES ('CALLS','m.bar','m.foo','m.py',12)")
    conn.execute("INSERT INTO metadata (key,value) VALUES ('git_head_sha',?)", (head_sha,))
    conn.execute("INSERT INTO metadata (key,value) VALUES ('last_updated','2026-05-29T00:00:00')")
    conn.commit(); conn.close()
    return root


# ─── #1 fleet endpoint ──────────────────────────────────────────────────────

def test_api_graph_fleet_coverage(monkeypatch, tmp_path):
    have = _build_graph_repo(str(tmp_path / "repo_a"))
    missing = str(tmp_path / "repo_b")
    os.makedirs(missing, exist_ok=True)
    monkeypatch.setattr(dashboard, "_discovered_repos", lambda: [have, missing])
    # No git in temp dirs → _git_head_sha None → fall through to "fresh".
    monkeypatch.setattr(dashboard, "_git_head_sha", lambda r: None)
    monkeypatch.setattr(dashboard._subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    data, status = dashboard.api_graph_fleet({})
    assert status == 200
    cov = data["coverage"]
    assert cov["total"] == 2 and cov["with_graph"] == 1 and cov["missing"] == 1
    by_alias = {r["alias"]: r for r in data["repos"]}
    assert by_alias["repo_a"]["state"] == "fresh"
    assert by_alias["repo_a"]["nodes"] == 2 and by_alias["repo_a"]["edges"] == 1
    assert by_alias["repo_a"]["languages"] == ["python"]
    assert by_alias["repo_b"]["state"] == "missing"


def test_api_graph_fleet_stale_on_sha_mismatch(monkeypatch, tmp_path):
    have = _build_graph_repo(str(tmp_path / "repo_a"), head_sha="OLD")
    monkeypatch.setattr(dashboard, "_discovered_repos", lambda: [have])
    monkeypatch.setattr(dashboard, "_git_head_sha", lambda r: "NEW")   # HEAD moved
    data, _ = dashboard.api_graph_fleet({})
    assert data["coverage"]["stale"] == 1
    assert data["repos"][0]["state"] == "stale"


# ─── #5 explorer ────────────────────────────────────────────────────────────

def test_api_graph_explorer_orientation(monkeypatch, tmp_path):
    repo = _build_graph_repo(str(tmp_path / "repo_a"))
    monkeypatch.setattr(dashboard, "_discovered_repos", lambda: [repo])
    data, status = dashboard.api_graph_explorer({"repo": repo})
    assert status == 200
    assert "foo" in data["orientation"]   # hub section present


def test_api_graph_explorer_rejects_unmanaged(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "_discovered_repos", lambda: [])
    data, status = dashboard.api_graph_explorer({"repo": "/etc"})
    assert status == 404


def test_api_graph_explorer_no_graph(monkeypatch, tmp_path):
    repo = str(tmp_path / "nograph")
    os.makedirs(repo, exist_ok=True)
    monkeypatch.setattr(dashboard, "_discovered_repos", lambda: [repo])
    data, status = dashboard.api_graph_explorer({"repo": repo})
    assert status == 404


def test_resolve_fleet_repo_by_alias(monkeypatch, tmp_path):
    repo = _build_graph_repo(str(tmp_path / "uniquename"))
    monkeypatch.setattr(dashboard, "_discovered_repos", lambda: [repo])
    assert dashboard._resolve_fleet_repo("uniquename") == os.path.realpath(repo)
    assert dashboard._resolve_fleet_repo("does-not-exist") is None


# ─── graph_fleet sweep-state round-trip ─────────────────────────────────────

def test_graph_fleet_sweep_state_roundtrip(dbs):
    from cairn import graph_fleet
    graph_fleet._record_sweep_state({"discovered": 5, "built": 1, "updated": 4, "failed": ["x"]})
    from hooks.hook_helpers import load_hook_state
    import json
    raw = load_hook_state(graph_fleet._FLEET_STATE_SESSION, graph_fleet._FLEET_LAST_SWEEP_KEY)
    rec = json.loads(raw)
    assert rec["discovered"] == 5 and rec["built"] == 1 and rec["failed"] == 1
    assert "ts" in rec


# ─── #2 metric flush persistence ────────────────────────────────────────────

def test_graph_metric_persists_through_flush(dbs):
    """record_metric + flush_metrics writes a graph_* row to the ephemeral DB."""
    import hooks.hook_helpers as hh
    hh.record_metric("sess", "graph_file_context_injected", "somefile.py")
    hh.flush_metrics()
    _, eph = dbs
    c = sqlite3.connect(eph)
    n = c.execute("SELECT COUNT(*) FROM metrics WHERE event='graph_file_context_injected'").fetchone()[0]
    c.close()
    assert n >= 1
