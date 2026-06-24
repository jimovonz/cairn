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


def test_kick_graph_build_updates_when_graph_db_already_present():
    """Graph present → incremental `update` (not skip), so it stays fresh
    without relying on git hooks (git-ai bypasses native .git/hooks)."""
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td)
        os.makedirs(os.path.join(td, ".code-review-graph"))
        open(os.path.join(td, ".code-review-graph", "graph.db"), "w").write("x")
        with patch.object(repo_discovery, "_resolve_crg", return_value="/fake/code-review-graph"), \
             patch.object(repo_discovery.subprocess, "Popen") as mock_popen:
            result = repo_discovery.kick_graph_build(td, env_override={"CAIRN_AUTO_GRAPH": "1"})
            assert result is True
            args = mock_popen.call_args[0][0]
            assert args[0] == "/fake/code-review-graph"
            assert "update" in args


def test_kick_graph_build_skips_when_binary_missing():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td)
        with patch.object(repo_discovery, "_resolve_crg", return_value=None):
            assert repo_discovery.kick_graph_build(td, env_override={"CAIRN_AUTO_GRAPH": "1"}) is False


def test_kick_graph_build_fires_subprocess_when_all_conditions_met():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td)
        with patch.object(repo_discovery, "_resolve_crg", return_value="/fake/code-review-graph"), \
             patch.object(repo_discovery.subprocess, "Popen") as mock_popen:
            result = repo_discovery.kick_graph_build(td, env_override={"CAIRN_AUTO_GRAPH": "1"})
            assert result is True
            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            assert args[0] == "/fake/code-review-graph"
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


def test_resolve_crg_prefers_venv_sibling(monkeypatch, tmp_path):
    """_resolve_crg prefers the binary next to the running interpreter (venv)."""
    import sys
    fake_bin = tmp_path / "code-review-graph"
    fake_bin.write_text("#!/bin/sh\n"); fake_bin.chmod(0o755)
    fake_py = tmp_path / "python3"; fake_py.write_text(""); fake_py.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(fake_py))
    assert repo_discovery._resolve_crg() == str(fake_bin)


def test_resolve_crg_falls_back_to_path(monkeypatch, tmp_path):
    """Falls back to PATH lookup when no venv sibling exists."""
    import sys
    empty = tmp_path / "bin"; empty.mkdir()
    fake_py = empty / "python3"; fake_py.write_text(""); fake_py.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(fake_py))
    monkeypatch.delenv("CAIRN_HOME", raising=False)
    monkeypatch.setattr(repo_discovery.shutil, "which", lambda _n: "/usr/bin/code-review-graph")
    assert repo_discovery._resolve_crg() == "/usr/bin/code-review-graph"


# ---------- HEAD-change freshness (branch switch / pull / rebase) ----------

def test_head_signature_returns_sha_for_git_repo():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=1)
        sig = repo_discovery.head_signature(td)
        assert sig is not None and len(sig) == 40


def test_head_signature_none_for_non_git():
    with tempfile.TemporaryDirectory() as td:
        assert repo_discovery.head_signature(td) is None


def test_head_signature_changes_on_new_commit():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=1)
        first = repo_discovery.head_signature(td)
        open(os.path.join(td, "g.py"), "w").write("# new\n")
        subprocess.run(["git", "-C", td, "add", "."], check=True, timeout=10)
        subprocess.run(["git", "-C", td, "commit", "-q", "-m", "second"], check=True, timeout=10)
        assert repo_discovery.head_signature(td) != first


def test_head_signature_changes_on_branch_switch():
    """Switching to a branch at a different commit changes the signature —
    the exact gap this freshness path closes."""
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=1)
        main_sig = repo_discovery.head_signature(td)
        subprocess.run(["git", "-C", td, "checkout", "-q", "-b", "feature"], check=True, timeout=10)
        open(os.path.join(td, "h.py"), "w").write("# feature\n")
        subprocess.run(["git", "-C", td, "add", "."], check=True, timeout=10)
        subprocess.run(["git", "-C", td, "commit", "-q", "-m", "feat"], check=True, timeout=10)
        feature_sig = repo_discovery.head_signature(td)
        assert feature_sig != main_sig
        # switching back returns the original signature
        subprocess.run(["git", "-C", td, "checkout", "-q", "-"], check=True, timeout=10)
        assert repo_discovery.head_signature(td) == main_sig


def test_kick_update_no_change_does_not_kick():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=1)
        head = repo_discovery.head_signature(td)
        with patch.object(repo_discovery, "kick_graph_build") as mock_kick:
            kicked, current = repo_discovery.kick_graph_update_if_head_changed(
                td, head, env_override={"CAIRN_AUTO_GRAPH": "1"})
        assert kicked is False
        assert current == head
        mock_kick.assert_not_called()


def test_kick_update_on_head_change_kicks():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=1)
        head = repo_discovery.head_signature(td)
        with patch.object(repo_discovery, "kick_graph_build", return_value=True) as mock_kick:
            kicked, current = repo_discovery.kick_graph_update_if_head_changed(
                td, "stale-old-sha", env_override={"CAIRN_AUTO_GRAPH": "1"})
        assert kicked is True
        assert current == head
        mock_kick.assert_called_once()


def test_kick_update_first_seen_none_kicks():
    """No prior HEAD recorded (e.g. first observation) → treat as changed."""
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=1)
        with patch.object(repo_discovery, "kick_graph_build", return_value=True) as mock_kick:
            kicked, current = repo_discovery.kick_graph_update_if_head_changed(
                td, None, env_override={"CAIRN_AUTO_GRAPH": "1"})
        assert kicked is True
        assert current == repo_discovery.head_signature(td)
        mock_kick.assert_called_once()


def test_kick_update_disabled_env_no_kick():
    with tempfile.TemporaryDirectory() as td:
        _init_git_repo(td, file_count=1)
        with patch.object(repo_discovery, "kick_graph_build") as mock_kick:
            kicked, current = repo_discovery.kick_graph_update_if_head_changed(
                td, "stale", env_override={"CAIRN_AUTO_GRAPH": "0"})
        assert kicked is False
        assert current == "stale"
        mock_kick.assert_not_called()


def test_kick_update_non_git_returns_false():
    with tempfile.TemporaryDirectory() as td:
        with patch.object(repo_discovery, "kick_graph_build") as mock_kick:
            kicked, current = repo_discovery.kick_graph_update_if_head_changed(
                td, "stale", env_override={"CAIRN_AUTO_GRAPH": "1"})
        assert kicked is False
        assert current == "stale"
        mock_kick.assert_not_called()
