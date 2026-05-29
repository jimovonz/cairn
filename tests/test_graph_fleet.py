"""Tests for cairn/graph_fleet.py — multi-repo code-graph fleet manager."""

import os
import subprocess
from unittest.mock import patch

from cairn import graph_fleet


def _mk_repo(root, name):
    d = root / name
    (d / ".git").mkdir(parents=True)
    return d


def test_discover_repos_finds_git_dirs(tmp_path):
    a = _mk_repo(tmp_path, "a")
    b = _mk_repo(tmp_path, "b")
    (tmp_path / "not-a-repo").mkdir()
    found = graph_fleet.discover_repos([str(tmp_path)])
    assert str(a.resolve()) in found
    assert str(b.resolve()) in found
    assert len(found) == 2


def test_discover_repos_dedupes_symlinked_root(tmp_path):
    real = tmp_path / "real"; real.mkdir()
    _mk_repo(real, "x")
    link = tmp_path / "link"; link.symlink_to(real)
    # Both the real dir and the symlink as roots → still one repo.
    found = graph_fleet.discover_repos([str(real), str(link)])
    assert len(found) == 1
    assert found[0] == str((real / "x").resolve())


def test_discover_repos_skips_noise_dirs(tmp_path):
    _mk_repo(tmp_path / "node_modules", "pkg")  # under a skip dir
    real = _mk_repo(tmp_path, "ok")
    found = graph_fleet.discover_repos([str(tmp_path)], max_depth=3)
    assert str(real.resolve()) in found
    assert not any("node_modules" in f for f in found)


def test_sweep_builds_missing_no_daemon_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CAIRN_GRAPH_WATCH", raising=False)  # daemon off by default
    _mk_repo(tmp_path, "r1")
    _mk_repo(tmp_path, "r2")
    calls = []
    with patch.object(graph_fleet, "_resolve_crg", return_value="/fake/crg"), \
         patch.object(graph_fleet, "_graph_db_present", return_value=False), \
         patch.object(graph_fleet, "_run", side_effect=lambda c, a, timeout=600: calls.append(a) or (True, "")):
        stats = graph_fleet.sweep([str(tmp_path)], verbose=False)
    assert stats["discovered"] == 2 and stats["built"] == 2
    assert stats["registered"] == 0
    assert [c for c in calls if c[0] == "build"]
    assert not [c for c in calls if c and c[0] == "daemon"]  # no daemon calls when off


def test_sweep_updates_existing_graphs(tmp_path, monkeypatch):
    monkeypatch.delenv("CAIRN_GRAPH_WATCH", raising=False)
    _mk_repo(tmp_path, "r1")
    calls = []
    with patch.object(graph_fleet, "_resolve_crg", return_value="/fake/crg"), \
         patch.object(graph_fleet, "_graph_db_present", return_value=True), \
         patch.object(graph_fleet, "_run", side_effect=lambda c, a, timeout=600: calls.append(a) or (True, "")):
        stats = graph_fleet.sweep([str(tmp_path)], verbose=False)
    assert stats["built"] == 0 and stats["updated"] == 1
    assert [c for c in calls if c[0] == "update"]
    assert not [c for c in calls if c and c[0] == "build"]


def test_sweep_registers_with_daemon_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_GRAPH_WATCH", "1")
    _mk_repo(tmp_path, "r1")
    calls = []
    with patch.object(graph_fleet, "_resolve_crg", return_value="/fake/crg"), \
         patch.object(graph_fleet, "_graph_db_present", return_value=False), \
         patch.object(graph_fleet, "_run", side_effect=lambda c, a, timeout=600: calls.append(a) or (True, "")):
        stats = graph_fleet.sweep([str(tmp_path)], verbose=False)
    assert stats["registered"] == 1
    assert [c for c in calls if c[:2] == ["daemon", "add"]]
    assert any(c[:1] == ["daemon"] and c[1] in ("start", "restart") for c in calls)


def test_sweep_noop_without_crg(tmp_path):
    with patch.object(graph_fleet, "_resolve_crg", return_value=None):
        stats = graph_fleet.sweep([str(tmp_path)], verbose=False)
    assert stats == {"error": "crg-not-found"}
