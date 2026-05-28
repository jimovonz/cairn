"""Tests for cairn/repo_discovery.py — Tier 1 graph build + Tier 2 suggestion."""

from __future__ import annotations

import os
import subprocess
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from cairn import repo_discovery


def _init_git_repo(path: str, file_count: int = 0) -> None:
    subprocess.run(["git", "init", "-q", path], check=True, timeout=10)
    subprocess.run(["git", "-C", path, "config", "user.email", "t@t"], check=True, timeout=5)
    subprocess.run(["git", "-C", path, "config", "user.name", "t"], check=True, timeout=5)
    for i in range(file_count):
        open(os.path.join(path, f"f{i}.py"), "w").write(f"# {i}\n")
    if file_count:
        subprocess.run(["git", "-C", path, "add", "."], check=True, timeout=10)
        subprocess.run(["git", "-C", path, "commit", "-q", "-m", "init"], check=True, timeout=10)


# ---------- Tier 1 ----------

def test_kick_graph_build_skips_non_git():
    with tempfile.TemporaryDirectory() as td:
        assert repo_discovery.kick_graph_build(td) is False


def test_kick_graph_build_skips_when_disabled():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td)
        assert repo_discovery.kick_graph_build(td, env_override={"CAIRN_AUTO_GRAPH": "0"}) is False


def test_kick_graph_build_skips_when_graph_db_already_present():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td)
        os.makedirs(os.path.join(td, ".code-review-graph"))
        open(os.path.join(td, ".code-review-graph", "graph.db"), "w").write("x")
        assert repo_discovery.kick_graph_build(td, env_override={"CAIRN_AUTO_GRAPH": "1"}) is False


def test_kick_graph_build_skips_when_binary_missing():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td)
        with patch.object(repo_discovery.shutil, "which", return_value=None):
            assert repo_discovery.kick_graph_build(td, env_override={"CAIRN_AUTO_GRAPH": "1"}) is False


def test_kick_graph_build_fires_subprocess_when_all_conditions_met():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td)
        with patch.object(repo_discovery.shutil, "which", return_value="/fake/code-review-graph"), \
             patch.object(repo_discovery.subprocess, "Popen") as mock_popen:
            result = repo_discovery.kick_graph_build(td, env_override={"CAIRN_AUTO_GRAPH": "1"})
            assert result is True
            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            assert args[0] == "code-review-graph"
            assert "build" in args


# ---------- Tier 2 ----------

class _FakeConn:
    def __init__(self, has_ingest: bool = False):
        self._has_ingest = has_ingest
    def execute(self, sql, params=()):
        result = MagicMock()
        result.fetchone.return_value = (1,) if self._has_ingest else None
        return result


def test_should_suggest_ingest_returns_none_for_non_git():
    with tempfile.TemporaryDirectory() as td:
        assert repo_discovery.should_suggest_ingest(td, _FakeConn()) is None


def test_should_suggest_ingest_returns_none_when_disabled():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=100)
        assert repo_discovery.should_suggest_ingest(
            td, _FakeConn(), env_override={"CAIRN_INGEST_SUGGEST": "0"}) is None


def test_should_suggest_ingest_returns_none_when_already_ingested():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=100)
        assert repo_discovery.should_suggest_ingest(
            td, _FakeConn(has_ingest=True),
            env_override={"CAIRN_INGEST_SUGGEST": "1"}) is None


def test_should_suggest_ingest_returns_none_when_file_count_below_threshold():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=5)
        assert repo_discovery.should_suggest_ingest(
            td, _FakeConn(), min_files=50,
            env_override={"CAIRN_INGEST_SUGGEST": "1"}) is None


def test_should_suggest_ingest_returns_string_when_all_conditions_met():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=60)
        msg = repo_discovery.should_suggest_ingest(
            td, _FakeConn(), min_files=50,
            env_override={"CAIRN_INGEST_SUGGEST": "1"})
        assert msg is not None
        assert "Cairn ingestion record" in msg
        assert "ingest.py" in msg
        assert os.path.basename(td) in msg or td in msg


def test_should_suggest_ingest_handles_db_exception_gracefully():
    """If the cairn DB query throws, treat as 'no record found' (returns
    suggestion) — fail-open rather than crashing prompt-hook."""
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=60)
        broken_conn = MagicMock()
        broken_conn.execute.side_effect = RuntimeError("boom")
        msg = repo_discovery.should_suggest_ingest(
            td, broken_conn, min_files=50,
            env_override={"CAIRN_INGEST_SUGGEST": "1"})
        assert msg is not None
