"""Two-tier repo auto-discovery — fires once per session from prompt_hook.

Tier 1 (cheap, automatic): if the cwd is a git repo with no
.code-review-graph/graph.db, kick off `code-review-graph build` in the
background. Free local AST, ~2s. Gated by CAIRN_AUTO_GRAPH env var.

Tier 2 (expensive, surfaced only): if the cwd is a git repo with no
cairn ingest record AND has at least N files, return a one-line
suggestion string for the prompt-hook to inject. Never invokes
ingest.py automatically — running ingest.py costs $0.10-0.50 in Haiku
calls and many encountered repos are throwaway clones. Gated by
CAIRN_INGEST_SUGGEST env var.

Per session de-duplication is handled by the caller via hook_state.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional


def _is_git_repo(path: str) -> bool:
    return bool(path) and os.path.isdir(os.path.join(path, ".git"))


def _graph_db_present(repo_root: str) -> bool:
    return os.path.isfile(os.path.join(repo_root, ".code-review-graph", "graph.db"))


def _git_file_count(repo_root: str) -> int:
    """Number of tracked + cached files in the repo. 0 on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "ls-files"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return 0
        return sum(1 for line in result.stdout.splitlines() if line.strip())
    except (subprocess.SubprocessError, OSError):
        return 0


def _has_ingest_record(conn, repo_root: str) -> bool:
    """True if cairn.memories has any source_ref naming this repo path."""
    try:
        like_pattern = f'%"{repo_root}"%'
        row = conn.execute(
            "SELECT 1 FROM memories WHERE source_ref LIKE ? "
            "AND (deleted_at IS NULL) LIMIT 1",
            (like_pattern,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _resolve_crg() -> Optional[str]:
    """Locate the code-review-graph binary.

    code-review-graph is installed inside cairn's own venv, which is usually not
    on PATH. Prefer the venv sibling of the running interpreter (hooks run under
    .venv/bin/python3), then CAIRN_HOME/.venv/bin, then fall back to PATH.
    """
    import sys
    cand = os.path.join(os.path.dirname(sys.executable), "code-review-graph")
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    home = os.environ.get("CAIRN_HOME")
    if home:
        cand = os.path.join(home, ".venv", "bin", "code-review-graph")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which("code-review-graph")


def kick_graph_build(cwd: str, *, env_override: Optional[dict] = None) -> bool:
    """Tier 1: keep the code-review-graph current, in the background.

    Fired from the cairn prompt hook (not a git hook) so it works regardless of
    the active git wrapper — git-ai and other proxies own the native hook path
    and do not chain repo hooks, so `.git/hooks/post-commit`-based refresh is
    unreliable; this is the portable path.

    No graph yet  -> `build` (full).
    Graph present -> `update` (incremental, ~sub-second, only changed files).

    Returns True if a build/update was kicked, False otherwise. Non-blocking —
    subprocess.Popen with detached stdio so the prompt hook returns immediately.
    """
    env = env_override if env_override is not None else os.environ
    if env.get("CAIRN_AUTO_GRAPH", "1") == "0":
        return False
    if not _is_git_repo(cwd):
        return False
    crg = _resolve_crg()
    if crg is None:
        return False
    subcmd = "update" if _graph_db_present(cwd) else "build"
    try:
        subprocess.Popen(
            [crg, subcmd, "--repo", cwd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def head_signature(cwd: str) -> Optional[str]:
    """Signature of the checked-out commit (HEAD SHA), or None on failure.

    Branch switch / pull / merge / rebase / commit all move HEAD to a
    different commit, so the SHA is a sufficient freshness key for the code
    graph (two branches at the same commit yield an identical tree, so no
    refresh is needed). Uncommitted working-tree edits are deliberately out
    of scope here — the cron sweep and first-prompt build cover those.
    """
    if not _is_git_repo(cwd):
        return None
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def kick_graph_update_if_head_changed(
    cwd: str, last_head: Optional[str], *, env_override: Optional[dict] = None
) -> tuple[bool, Optional[str]]:
    """Tier 1 freshness on HEAD movement (branch switch, pull, rebase, commit).

    Compares the repo's current HEAD against `last_head` (the value the caller
    persisted on the previous prompt). If it changed, kicks a background
    `crg update` (or `build` if no graph exists yet) via `kick_graph_build`.

    Portable across git wrappers — this is driven by the cairn prompt hook, not
    a native git hook (git-ai and other proxies own `.git/hooks` and don't
    chain repo hooks, so a post-checkout hook would be unreliable).

    Returns ``(kicked, current_head)``. The caller should persist
    ``current_head`` whenever it differs from ``last_head``. Non-blocking and
    fails open (returns ``(False, last_head)``) on any error.
    """
    env = env_override if env_override is not None else os.environ
    if env.get("CAIRN_AUTO_GRAPH", "1") == "0":
        return (False, last_head)
    head = head_signature(cwd)
    if head is None or head == last_head:
        return (False, head if head is not None else last_head)
    kicked = kick_graph_build(cwd, env_override=env)
    return (kicked, head)


def should_suggest_ingest(cwd: str, conn,
                          *, min_files: int = 50,
                          env_override: Optional[dict] = None) -> Optional[str]:
    """Tier 2: return a one-line suggestion string if this repo lacks any
    cairn ingest record AND has at least `min_files` tracked files.

    Returns None when no suggestion is warranted. Never invokes ingest.py.
    """
    env = env_override if env_override is not None else os.environ
    if env.get("CAIRN_INGEST_SUGGEST", "1") == "0":
        return None
    if not _is_git_repo(cwd):
        return None
    if _has_ingest_record(conn, cwd):
        return None
    if _git_file_count(cwd) < min_files:
        return None
    repo_name = os.path.basename(cwd.rstrip("/")) or cwd
    return (
        f"This repo ({repo_name}) has no Cairn ingestion record. "
        f"Run `python3 $CAIRN_HOME/cairn/ingest.py {cwd}` to bootstrap "
        f"persistent knowledge for it (~$0.10-0.50 in Haiku calls). "
        f"Optional — Cairn already has live structural navigation via "
        f"cairn-graph; ingestion adds LLM-distilled facts/decisions."
    )
