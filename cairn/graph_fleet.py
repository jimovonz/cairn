#!/usr/bin/env python3
"""Fleet manager for code-review-graph across ALL local repos.

Goal: every git repo under the configured roots has a current code graph,
ready for first contact in any session — independent of whether cairn has been
active in it. Native git hooks can't deliver this (git-ai and other proxies own
the hook path and don't chain repo hooks), so freshness is driven by the
code-review-graph *watch daemon* plus a periodic discovery sweep:

  discover repos -> build any missing graph -> register with the daemon -> daemon keeps them current

Run modes:
  python3 -m cairn.graph_fleet            # sweep: discover, build-missing, register, ensure daemon up
  python3 -m cairn.graph_fleet --status   # show daemon + per-repo graph status
  python3 -m cairn.graph_fleet --roots A:B # override discovery roots (colon-separated)

Roots default to $CAIRN_GRAPH_ROOTS (colon-separated) or, failing that, the
parent directory of the cairn checkout (e.g. /mnt/ssd/Projects). Symlinked roots
are resolved and de-duplicated so a symlink + its target aren't swept twice.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cairn.repo_discovery import _resolve_crg, _graph_db_present

# Sentinel "session" under which the fleet records its last-sweep summary in the
# ephemeral hook_state table (hook_state PK is (session_id, key)).
_FLEET_STATE_SESSION = "graph_fleet"
_FLEET_LAST_SWEEP_KEY = "graph_fleet_last_sweep"


def _record_sweep_state(stats: dict) -> None:
    """Persist the sweep summary so the dashboard can surface freshness without
    scraping logs. Fail-open — observability must never break the sweep."""
    try:
        from hooks.hook_helpers import save_hook_state
        save_hook_state(_FLEET_STATE_SESSION, _FLEET_LAST_SWEEP_KEY, json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "discovered": stats.get("discovered", 0),
            "built": stats.get("built", 0),
            "updated": stats.get("updated", 0),
            "failed": len(stats.get("failed", [])),
        }))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"(could not record sweep state: {exc})", file=sys.stderr)

# Directory names never worth graphing — skip to save build time/noise.
_SKIP_DIRS = {"node_modules", ".venv", "venv", "__pycache__", ".cache", "vendor"}


def _default_roots() -> list[str]:
    env = os.environ.get("CAIRN_GRAPH_ROOTS")
    if env:
        return [p for p in env.split(":") if p]
    # Parent of the cairn checkout (CAIRN_HOME), e.g. /mnt/ssd/Projects/cairn -> /mnt/ssd/Projects
    home = os.environ.get("CAIRN_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [os.path.dirname(home)]


def discover_repos(roots: Optional[list[str]] = None, max_depth: int = 2) -> list[str]:
    """Find git repos under roots. Resolves symlinks, de-dupes by realpath."""
    roots = roots or _default_roots()
    found: set[str] = set()
    for root in roots:
        base = Path(root).resolve()
        if not base.is_dir():
            continue
        # Walk up to max_depth below the root looking for a .git entry.
        for depth in range(0, max_depth + 1):
            pattern = "/".join(["*"] * depth) + "/.git" if depth else ".git"
            for git in base.glob(pattern):
                repo = git.parent.resolve()
                if any(part in _SKIP_DIRS for part in repo.parts):
                    continue
                found.add(str(repo))
    return sorted(found)


def _run(crg: str, args: list[str], timeout: int = 600) -> tuple[bool, str]:
    try:
        r = subprocess.run([crg, *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except (subprocess.SubprocessError, OSError) as e:
        return False, str(e)


def _watch_daemon_enabled() -> bool:
    # Opt-in. The watch daemon adds real-time freshness but is fragile (doesn't
    # always persist when spawned outside a login shell) and churns on volatile
    # files (e.g. cairn's own ephemeral DB). The hourly sweep below is the robust
    # freshness backbone; enable the daemon with CAIRN_GRAPH_WATCH=1 on top.
    return os.environ.get("CAIRN_GRAPH_WATCH", "0").lower() in ("1", "true", "yes")


def sweep(roots: Optional[list[str]] = None, *, build_missing: bool = True,
          refresh_existing: bool = True, verbose: bool = True) -> dict:
    """Make every discovered repo's code graph present and current.

    Build missing graphs, and incrementally `update` existing ones. This is the
    freshness backbone — daemon-independent, so it holds even when the watch
    daemon isn't running. When CAIRN_GRAPH_WATCH=1, also register repos with the
    watch daemon and ensure it's up for real-time updates between sweeps.
    """
    crg = _resolve_crg()
    if crg is None:
        if verbose:
            print("code-review-graph not found (cairn venv / PATH) — nothing to do.", file=sys.stderr)
        return {"error": "crg-not-found"}

    repos = discover_repos(roots)
    stats = {"discovered": len(repos), "built": 0, "updated": 0, "registered": 0,
             "failed": [], "repos": repos}
    use_daemon = _watch_daemon_enabled()
    newly_registered = 0

    for repo in repos:
        present = _graph_db_present(repo)
        if not present and build_missing:
            ok, out = _run(crg, ["build", "--repo", repo])
            if ok:
                stats["built"] += 1
                if verbose:
                    print(f"built  {repo}")
            else:
                stats["failed"].append(repo)
                if verbose:
                    print(f"FAILED build {repo}: {out[:160]}", file=sys.stderr)
                continue
        elif present and refresh_existing:
            # Incremental — only re-parses changed files; ~sub-second when clean.
            ok, _ = _run(crg, ["update", "--repo", repo], timeout=300)
            if ok:
                stats["updated"] += 1
        if use_daemon:
            ok, _ = _run(crg, ["daemon", "add", repo, "--alias", os.path.basename(repo)], timeout=30)
            if ok:
                stats["registered"] += 1
                newly_registered += 1

    if use_daemon:
        # Ensure the daemon is up; restart only if we registered new repos this
        # sweep (a running daemon does not hot-load newly-added repos).
        running, _ = _run(crg, ["daemon", "status"], timeout=15)
        sub = "restart" if (running and newly_registered) else "start"
        _run(crg, ["daemon", sub], timeout=30)

    _record_sweep_state(stats)

    if verbose:
        print(f"\nfleet: {stats['discovered']} repos, {stats['built']} built, "
              f"{stats['updated']} updated, {len(stats['failed'])} failed"
              + (f", {stats['registered']} watched" if use_daemon else ""))
    return stats


def status(roots: Optional[list[str]] = None) -> None:
    crg = _resolve_crg()
    if crg is None:
        print("code-review-graph not found.", file=sys.stderr)
        return
    repos = discover_repos(roots)
    have = sum(1 for r in repos if _graph_db_present(r))
    print(f"Discovered repos: {len(repos)}  |  with graph: {have}  |  missing: {len(repos) - have}")
    ok, out = _run(crg, ["daemon", "status"], timeout=30)
    print(out)


def main() -> None:
    args = sys.argv[1:]
    roots = None
    if "--roots" in args:
        roots = args[args.index("--roots") + 1].split(":")
    if "--status" in args:
        status(roots)
        return
    sweep(roots, build_missing="--no-build" not in args,
          refresh_existing="--no-refresh" not in args)


if __name__ == "__main__":
    _venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "python3")
    if os.path.exists(_venv_python) and sys.prefix == sys.base_prefix:
        os.execv(_venv_python, [_venv_python] + sys.argv)
    main()
