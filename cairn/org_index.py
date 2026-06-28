#!/usr/bin/env python3
"""
org_index.py — the *locatability* layer of the org-wide git index.

Walks every repo / every branch in a GitHub org via `gh api` (zero clones) and
records, in one SQLite/FTS5 catalog, where every file lives and which branches
carry unmerged work that is at risk of being silently lost.

Motivating incident: `tools/board_test.py` (VCS bringup loopback tester) was
stranded on the unmerged branch `JO_Prod_Bringup` and invisible from the branch
we were on. This index answers both questions that were hard that day:
    1. "Where is <file> across the org, on ANY branch?"   -> `find`
    2. "What unmerged work is going stale and at risk?"   -> `stranded`

Storage note: uses the stdlib `sqlite3`. Unlike cairn (concurrent multi-version
readers, hence its pysqlite3 requirement) this is a single-writer nightly job on
a throwaway DB, so stdlib + WAL is fine.

Subcommands:
    build     walk the org and (re)populate the catalog
    find      locate a filename / path fragment across all repos & branches
    stranded  report unmerged branches with unique commits, ranked by staleness
    branches  list indexed branches for one repo with ahead/behind/status
    stats     summary of what is indexed
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "org_index.db")

# --------------------------------------------------------------------------- gh

class GhError(RuntimeError):
    pass

def gh_lines(path, jq=".[]"):
    """Paginated array endpoint -> yields one parsed object per element.
    `gh api --paginate -q '.[]'` runs jq per page, emitting newline-delimited
    JSON objects across all pages, so we parse line by line."""
    cmd = ["gh", "api", "--paginate", path, "-q", jq]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise GhError(f"{' '.join(cmd)}\n{p.stderr.strip()}")
    for line in p.stdout.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)

def gh_obj(path):
    """Single-object endpoint -> parsed JSON (or None on 404/422)."""
    p = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if p.returncode != 0:
        # 404 (no such branch base) / 422 (too many commits to compare) are
        # expected and non-fatal — caller decides.
        return None
    return json.loads(p.stdout)

# ------------------------------------------------------------------------ schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    name           TEXT PRIMARY KEY,
    default_branch TEXT,
    archived       INTEGER DEFAULT 0,
    pushed_at      TEXT,
    indexed_at     REAL
);
CREATE TABLE IF NOT EXISTS branches (
    repo        TEXT NOT NULL,
    branch      TEXT NOT NULL,
    tip_sha     TEXT,
    tip_date    TEXT,        -- ISO8601 of branch-tip commit
    ahead_by    INTEGER,     -- commits on branch not on default
    behind_by   INTEGER,     -- commits on default not on branch
    status      TEXT,        -- identical|ahead|behind|diverged|default
    is_default  INTEGER DEFAULT 0,
    tree_indexed INTEGER DEFAULT 0,
    PRIMARY KEY (repo, branch)
);
CREATE TABLE IF NOT EXISTS files (
    repo     TEXT NOT NULL,
    branch   TEXT NOT NULL,
    path     TEXT NOT NULL,
    blob_sha TEXT,
    size     INTEGER,
    basename TEXT,
    PRIMARY KEY (repo, branch, path)
);
CREATE INDEX IF NOT EXISTS idx_files_basename ON files(basename);
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    path, repo UNINDEXED, branch UNINDEXED, content=''
);
"""

def connect(db_path):
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con

# ------------------------------------------------------------------------- build

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def build(org, db_path, only_repos=None, max_branches=None,
          include_archived=False, verbose=True):
    con = connect(db_path)
    cur = con.cursor()

    def log(*a):
        if verbose:
            print(*a, file=sys.stderr, flush=True)

    repos = list(gh_lines(f"orgs/{org}/repos?per_page=100",
                          jq=".[] | {name, default_branch, archived, pushed_at}"))
    if only_repos:
        want = set(only_repos)
        repos = [r for r in repos if r["name"] in want]
    if not include_archived:
        repos = [r for r in repos if not r.get("archived")]

    log(f"[build] {org}: {len(repos)} repos")
    api_calls = 0
    for ri, repo in enumerate(repos, 1):
        name = repo["name"]
        default = repo.get("default_branch") or "main"
        cur.execute(
            "INSERT OR REPLACE INTO repos VALUES (?,?,?,?,?)",
            (name, default, int(bool(repo.get("archived"))),
             repo.get("pushed_at"), time.time()))
        # fresh per-repo rows
        cur.execute("DELETE FROM branches WHERE repo=?", (name,))
        cur.execute("DELETE FROM files WHERE repo=?", (name,))

        try:
            branches = list(gh_lines(
                f"repos/{org}/{name}/branches?per_page=100",
                jq=".[] | {name: .name, sha: .commit.sha}"))
        except GhError as e:
            log(f"  ! {name}: cannot list branches ({e})")
            continue
        api_calls += 1
        if max_branches:
            # keep default + most-recent others is overkill here; just cap
            branches = branches[:max_branches]
        log(f"[{ri}/{len(repos)}] {name}: {len(branches)} branches")

        for br in branches:
            bname, sha = br["name"], br["sha"]
            is_default = (bname == default)
            ahead = behind = None
            status = "default" if is_default else None
            tip_date = None

            if is_default:
                tip = gh_obj(f"repos/{org}/{name}/commits/{sha}")
                api_calls += 1
                if tip:
                    tip_date = tip["commit"]["committer"]["date"]
            else:
                cmp_ = gh_obj(f"repos/{org}/{name}/compare/{default}...{bname}")
                api_calls += 1
                if cmp_:
                    ahead = cmp_.get("ahead_by")
                    behind = cmp_.get("behind_by")
                    status = cmp_.get("status")  # identical|ahead|behind|diverged
                    commits = cmp_.get("commits") or []
                    if commits:  # last commit in base...head is the head tip
                        tip_date = commits[-1]["commit"]["committer"]["date"]
                if tip_date is None:
                    tip = gh_obj(f"repos/{org}/{name}/commits/{sha}")
                    api_calls += 1
                    if tip:
                        tip_date = tip["commit"]["committer"]["date"]

            # Index the tree only when the branch carries unique state:
            #   - the default branch (baseline), or
            #   - a branch that is ahead/diverged (ahead_by > 0).
            # A branch with ahead_by == 0 is an ancestor of default; its files
            # are a subset of default's tree, so skipping it loses no path.
            index_tree = is_default or (ahead or 0) > 0
            tree_indexed = 0
            if index_tree:
                tree = gh_obj(
                    f"repos/{org}/{name}/git/trees/{sha}?recursive=1")
                api_calls += 1
                if tree:
                    if tree.get("truncated"):
                        log(f"    ~ {name}@{bname}: tree TRUNCATED (huge repo)")
                    rows = []
                    for ent in tree.get("tree", []):
                        if ent.get("type") != "blob":
                            continue
                        path = ent["path"]
                        base = path.rsplit("/", 1)[-1]
                        rows.append((name, bname, path, ent.get("sha"),
                                     ent.get("size"), base))
                    cur.executemany(
                        "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?)", rows)
                    cur.executemany(
                        "INSERT INTO files_fts (path, repo, branch) VALUES (?,?,?)",
                        [(r[2], r[0], r[1]) for r in rows])
                    tree_indexed = 1

            cur.execute(
                "INSERT OR REPLACE INTO branches VALUES (?,?,?,?,?,?,?,?,?)",
                (name, bname, sha, tip_date, ahead, behind, status,
                 int(is_default), tree_indexed))
        con.commit()

    log(f"[build] done. ~{api_calls} gh api calls. db={db_path}")
    con.close()

# ------------------------------------------------------------------------ query

def _age_days(iso):
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - dt).days

def find(db_path, term, limit=50):
    con = connect(db_path)
    # exact-ish basename match first, then FTS path match
    like = f"%{term}%"
    rows = con.execute(
        """SELECT f.repo, f.branch, f.path, b.status, b.is_default, b.tip_date
           FROM files f JOIN branches b ON b.repo=f.repo AND b.branch=f.branch
           WHERE f.basename = ? OR f.path LIKE ?
           ORDER BY b.is_default DESC, f.repo, f.branch
           LIMIT ?""",
        (term, like, limit)).fetchall()
    if not rows:
        print(f"No match for '{term}'. (Is the index built?)")
        return
    for repo, branch, path, status, is_def, tip in rows:
        tag = "default" if is_def else (status or "?")
        age = _age_days(tip)
        agestr = f"{age}d old" if age is not None else "age?"
        flag = "" if is_def or status in ("identical", "behind") else "  ⚠ unmerged"
        print(f"{repo:28} {branch:24} [{tag:9}] {agestr:9} {path}{flag}")

def stranded(db_path, stale_days=90, limit=100):
    """Branches with unique commits (ahead/diverged), ranked by staleness.
    These are the 'work at risk' — unmerged and aging."""
    con = connect(db_path)
    rows = con.execute(
        """SELECT repo, branch, ahead_by, behind_by, status, tip_date
           FROM branches
           WHERE is_default=0 AND status IN ('ahead','diverged') AND ahead_by>0
           ORDER BY tip_date ASC
           LIMIT ?""", (limit,)).fetchall()
    if not rows:
        print("No stranded branches found. (Index built? Org fully merged?)")
        return
    print(f"{'REPO':28} {'BRANCH':28} {'AHEAD':>5} {'BEHIND':>6} {'AGE':>7}  STATUS")
    print("-" * 92)
    for repo, branch, ahead, behind, status, tip in rows:
        age = _age_days(tip)
        agestr = f"{age}d" if age is not None else "?"
        mark = "  ← STALE" if (age is not None and age >= stale_days) else ""
        print(f"{repo:28} {branch:28} {ahead or 0:>5} {behind or 0:>6} "
              f"{agestr:>7}  {status}{mark}")

def branches(db_path, repo):
    con = connect(db_path)
    rows = con.execute(
        """SELECT branch, ahead_by, behind_by, status, is_default, tip_date,
                  tree_indexed
           FROM branches WHERE repo=? ORDER BY is_default DESC, tip_date DESC""",
        (repo,)).fetchall()
    if not rows:
        print(f"No branches indexed for '{repo}'.")
        return
    for br, ahead, behind, status, is_def, tip, ti in rows:
        age = _age_days(tip)
        print(f"{br:30} {'(default)' if is_def else status or '?':10} "
              f"ahead={ahead or 0:<4} behind={behind or 0:<4} "
              f"age={age if age is not None else '?'}d tree={'y' if ti else 'n'}")

def stats(db_path):
    con = connect(db_path)
    r = con.execute("SELECT COUNT(*) FROM repos").fetchone()[0]
    b = con.execute("SELECT COUNT(*) FROM branches").fetchone()[0]
    f = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    st = con.execute(
        "SELECT COUNT(*) FROM branches WHERE status IN ('ahead','diverged') "
        "AND ahead_by>0").fetchone()[0]
    print(f"repos={r}  branches={b}  indexed files={f}  stranded branches={st}")

# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite path (default {DEFAULT_DB})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="walk the org and populate the catalog")
    b.add_argument("--org", required=True)
    b.add_argument("--repos", nargs="*", help="limit to these repo names")
    b.add_argument("--max-branches", type=int, default=None)
    b.add_argument("--include-archived", action="store_true")

    f = sub.add_parser("find", help="locate a file across all repos/branches")
    f.add_argument("term")
    f.add_argument("--limit", type=int, default=50)

    s = sub.add_parser("stranded", help="unmerged work at risk, by staleness")
    s.add_argument("--stale-days", type=int, default=90)

    br = sub.add_parser("branches", help="list indexed branches for a repo")
    br.add_argument("repo")

    sub.add_parser("stats", help="index summary")

    a = ap.parse_args()
    if a.cmd == "build":
        build(a.org, a.db, only_repos=a.repos, max_branches=a.max_branches,
              include_archived=a.include_archived)
    elif a.cmd == "find":
        find(a.db, a.term, a.limit)
    elif a.cmd == "stranded":
        stranded(a.db, a.stale_days)
    elif a.cmd == "branches":
        branches(a.db, a.repo)
    elif a.cmd == "stats":
        stats(a.db)

if __name__ == "__main__":
    # Re-exec under the cairn venv so `import cairn.*` resolves (mirrors
    # graph_fleet.py / query.py). No-op when already inside a venv.
    _venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".venv", "bin", "python3")
    if os.path.exists(_venv_python) and sys.prefix == sys.base_prefix:
        os.execv(_venv_python, [_venv_python] + sys.argv)
    main()
