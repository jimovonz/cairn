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

Storage note: goes through the shared pysqlite3 guard (like every cairn/ module
that touches a SQLite DB — enforced by tests/test_sqlite_guard.py). This catalog
is a single-writer nightly job on its own throwaway org_index.db, so the WAL
mixed-library corruption risk does not strictly apply, but the project keeps one
SQLite library everywhere rather than allowlisting exceptions.

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
import sys

# Re-exec under the cairn venv so pysqlite3 and `import cairn.*` resolve. MUST run
# BEFORE the pysqlite3 guard below — under a bare python3 that guard raises, so a
# re-exec placed in __main__ never runs (mirrors query.py). No-op inside a venv.
if __name__ == "__main__":
    _venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".venv", "bin", "python3")
    if os.path.exists(_venv_python) and sys.prefix == sys.base_prefix:
        os.execv(_venv_python, [_venv_python] + sys.argv)

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

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

# -------------------------------------------------------------------- rate limit

def _rate_status():
    """(remaining, reset_epoch) for the core REST quota. The rate_limit endpoint
    does NOT itself consume quota, so we can poll it freely to stage the walk."""
    p = subprocess.run(
        ["gh", "api", "rate_limit", "--jq",
         '.resources.core | "\\(.remaining) \\(.reset)"'],
        capture_output=True, text=True)
    if p.returncode != 0:
        return None, None
    try:
        rem, reset = p.stdout.split()
        return int(rem), int(reset)
    except ValueError:
        return None, None

def _stage_for_rate(min_rate, log):
    """Block until the core quota recovers when it dips below min_rate, so a
    large org walk never errors out mid-run. Sleeps to the reset + a buffer."""
    rem, reset = _rate_status()
    if rem is None or rem >= min_rate:
        return
    wait = max(0, reset - int(time.time())) + 5
    log(f"  [rate] core remaining {rem} < {min_rate}; sleeping {wait}s to reset")
    time.sleep(wait)

# ------------------------------------------------------------------------ schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    org            TEXT NOT NULL,
    name           TEXT NOT NULL,
    default_branch TEXT,
    archived       INTEGER DEFAULT 0,
    pushed_at      TEXT,
    indexed_at     REAL,
    PRIMARY KEY (org, name)
);
CREATE TABLE IF NOT EXISTS branches (
    org         TEXT NOT NULL,
    repo        TEXT NOT NULL,
    branch      TEXT NOT NULL,
    tip_sha     TEXT,
    tip_date    TEXT,        -- ISO8601 of branch-tip commit
    ahead_by    INTEGER,     -- commits on branch not on default
    behind_by   INTEGER,     -- commits on default not on branch
    status      TEXT,        -- identical|ahead|behind|diverged|default
    is_default  INTEGER DEFAULT 0,
    tree_indexed INTEGER DEFAULT 0,
    PRIMARY KEY (org, repo, branch)
);
CREATE TABLE IF NOT EXISTS files (
    org      TEXT NOT NULL,
    repo     TEXT NOT NULL,
    branch   TEXT NOT NULL,
    path     TEXT NOT NULL,
    blob_sha TEXT,
    size     INTEGER,
    basename TEXT,
    PRIMARY KEY (org, repo, branch, path)
);
CREATE INDEX IF NOT EXISTS idx_files_basename ON files(basename);
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    path, org UNINDEXED, repo UNINDEXED, branch UNINDEXED, content=''
);
"""

def connect(db_path):
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    # Auto-migrate a pre-multi-org db (repos lacks the `org` column): drop the
    # old tables and recreate. The index is fully rederivable from gh, so wiping
    # is cheaper and safer than an in-place ALTER across four tables + fts.
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(repos)")]
        if cols and "org" not in cols:
            for _t in ("files_fts", "files", "branches", "repos"):
                con.execute(f"DROP TABLE IF EXISTS {_t}")
            con.commit()
    except sqlite3.Error:
        pass
    con.executescript(SCHEMA)
    return con

# ------------------------------------------------------------------------- build

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def build(org, db_path, only_repos=None, max_branches=None,
          include_archived=False, pushed_within_months=None, min_rate=200,
          resume=False, gh_host=None, verbose=True):
    con = connect(db_path)
    cur = con.cursor()
    if gh_host:
        os.environ["GH_HOST"] = gh_host  # gh subprocesses inherit this (GHE)

    def log(*a):
        if verbose:
            print(f"[{org}]", *a, file=sys.stderr, flush=True)

    repos = list(gh_lines(f"orgs/{org}/repos?per_page=100",
                          jq=".[] | {name, default_branch, archived, pushed_at}"))
    if only_repos:
        want = set(only_repos)
        repos = [r for r in repos if r["name"] in want]
    if not include_archived:
        repos = [r for r in repos if not r.get("archived")]
    if pushed_within_months:
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=int(pushed_within_months * 30.44))).isoformat()
        before = len(repos)
        repos = [r for r in repos if (r.get("pushed_at") or "") >= cutoff]
        log(f"[build] pushed-within {pushed_within_months}mo: "
            f"{len(repos)}/{before} repos (cutoff {cutoff[:10]})")

    if resume:
        done = {r[0] for r in cur.execute("SELECT repo FROM branches WHERE org=?", (org,))}
        repos = [r for r in repos if r["name"] not in done]
        log(f"[build] resume: skipping {len(done)} already-indexed repos")

    log(f"[build] {org}: {len(repos)} repos to index")
    api_calls = 0
    for ri, repo in enumerate(repos, 1):
        _stage_for_rate(min_rate, log)   # block if quota is low before each repo
        name = repo["name"]
        default = repo.get("default_branch") or "main"
        cur.execute(
            "INSERT OR REPLACE INTO repos VALUES (?,?,?,?,?,?)",
            (org, name, default, int(bool(repo.get("archived"))),
             repo.get("pushed_at"), time.time()))
        # fresh per-repo rows (scoped to this org)
        cur.execute("DELETE FROM branches WHERE org=? AND repo=?", (org, name))
        cur.execute("DELETE FROM files WHERE org=? AND repo=?", (org, name))

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
                        rows.append((org, name, bname, path, ent.get("sha"),
                                     ent.get("size"), base))
                    cur.executemany(
                        "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?)", rows)
                    cur.executemany(
                        "INSERT INTO files_fts (path, org, repo, branch) VALUES (?,?,?,?)",
                        [(r[3], r[0], r[1], r[2]) for r in rows])
                    tree_indexed = 1

            cur.execute(
                "INSERT OR REPLACE INTO branches VALUES (?,?,?,?,?,?,?,?,?,?)",
                (org, name, bname, sha, tip_date, ahead, behind, status,
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
        """SELECT f.org, f.repo, f.branch, f.path, b.status, b.is_default, b.tip_date
           FROM files f JOIN branches b
             ON b.org=f.org AND b.repo=f.repo AND b.branch=f.branch
           WHERE f.basename = ? OR f.path LIKE ?
           ORDER BY b.is_default DESC, f.org, f.repo, f.branch
           LIMIT ?""",
        (term, like, limit)).fetchall()
    if not rows:
        print(f"No match for '{term}'. (Is the index built?)")
        return
    for org, repo, branch, path, status, is_def, tip in rows:
        tag = "default" if is_def else (status or "?")
        age = _age_days(tip)
        agestr = f"{age}d old" if age is not None else "age?"
        flag = "" if is_def or status in ("identical", "behind") else "  ⚠ unmerged"
        print(f"{org+'/'+repo:32} {branch:24} [{tag:9}] {agestr:9} {path}{flag}")

def stranded(db_path, stale_days=90, limit=100, org=None):
    """Branches with unique commits (ahead/diverged), ranked by staleness.
    These are the 'work at risk' — unmerged and aging. Optional org filter."""
    con = connect(db_path)
    q = ("SELECT org, repo, branch, ahead_by, behind_by, status, tip_date "
         "FROM branches WHERE is_default=0 AND status IN ('ahead','diverged') "
         "AND ahead_by>0")
    params = []
    if org:
        q += " AND org=?"
        params.append(org)
    q += " ORDER BY tip_date ASC LIMIT ?"
    params.append(limit)
    rows = con.execute(q, params).fetchall()
    if not rows:
        print("No stranded branches found. (Index built? Org fully merged?)")
        return
    print(f"{'ORG/REPO':38} {'BRANCH':26} {'AHEAD':>5} {'BEHIND':>6} {'AGE':>7}  STATUS")
    print("-" * 100)
    for org_, repo, branch, ahead, behind, status, tip in rows:
        age = _age_days(tip)
        agestr = f"{age}d" if age is not None else "?"
        mark = "  ← STALE" if (age is not None and age >= stale_days) else ""
        print(f"{org_+'/'+repo:38} {branch:26} {ahead or 0:>5} {behind or 0:>6} "
              f"{agestr:>7}  {status}{mark}")

def branches(db_path, repo, org=None):
    con = connect(db_path)
    q = ("SELECT org, branch, ahead_by, behind_by, status, is_default, tip_date, "
         "tree_indexed FROM branches WHERE repo=?")
    params = [repo]
    if org:
        q += " AND org=?"
        params.append(org)
    q += " ORDER BY org, is_default DESC, tip_date DESC"
    rows = con.execute(q, params).fetchall()
    if not rows:
        print(f"No branches indexed for '{repo}'.")
        return
    for org_, br, ahead, behind, status, is_def, tip, ti in rows:
        age = _age_days(tip)
        print(f"{org_+'/'+repo:32} {br:28} {'(default)' if is_def else status or '?':10} "
              f"ahead={ahead or 0:<4} behind={behind or 0:<4} "
              f"age={age if age is not None else '?'}d tree={'y' if ti else 'n'}")

def stats(db_path):
    con = connect(db_path)
    orgs = [o for (o,) in con.execute("SELECT DISTINCT org FROM repos ORDER BY org")]
    if not orgs:
        print("empty index (no orgs built)")
        return
    for o in orgs:
        r = con.execute("SELECT COUNT(*) FROM repos WHERE org=?", (o,)).fetchone()[0]
        b = con.execute("SELECT COUNT(*) FROM branches WHERE org=?", (o,)).fetchone()[0]
        f = con.execute("SELECT COUNT(*) FROM files WHERE org=?", (o,)).fetchone()[0]
        st = con.execute("SELECT COUNT(*) FROM branches WHERE org=? AND "
                         "status IN ('ahead','diverged') AND ahead_by>0", (o,)).fetchone()[0]
        print(f"{o}: repos={r}  branches={b}  files={f}  stranded={st}")

# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite path (default {DEFAULT_DB})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="walk one or more orgs and populate the catalog")
    b.add_argument("--orgs", nargs="*", default=None,
                   help="orgs to index (default: cairn config ORG_INDEX_ORGS)")
    b.add_argument("--repos", nargs="*", help="limit to these repo names")
    b.add_argument("--gh-host", default=None, help="GitHub host (default: github.com; set for GHE)")
    b.add_argument("--max-branches", type=int, default=None)
    b.add_argument("--include-archived", action="store_true")
    b.add_argument("--pushed-within-months", type=float, default=None,
                   help="only index repos pushed within the last N months (default: config)")
    b.add_argument("--min-rate", type=int, default=None,
                   help="pause the walk if core API quota drops below this (default: config)")
    b.add_argument("--resume", action="store_true",
                   help="skip repos already present in the index (continue a run)")

    f = sub.add_parser("find", help="locate a file across all repos/branches")
    f.add_argument("term")
    f.add_argument("--limit", type=int, default=50)

    s = sub.add_parser("stranded", help="unmerged work at risk, by staleness")
    s.add_argument("--stale-days", type=int, default=90)
    s.add_argument("--org", default=None, help="limit to one org")

    br = sub.add_parser("branches", help="list indexed branches for a repo")
    br.add_argument("repo")
    br.add_argument("--org", default=None, help="disambiguate repo across orgs")

    sub.add_parser("stats", help="index summary")

    a = ap.parse_args()
    if a.cmd == "build":
        from cairn import config
        orgs = a.orgs or config.ORG_INDEX_ORGS
        if not orgs:
            sys.exit("no orgs to index: pass --orgs or set ORG_INDEX_ORGS "
                     "(and ORG_INDEX_ENABLED) in cairn config")
        host = a.gh_host or config.ORG_INDEX_GH_HOST
        pwm = (a.pushed_within_months if a.pushed_within_months is not None
               else config.ORG_INDEX_PUSHED_WITHIN_MONTHS)
        mr = a.min_rate if a.min_rate is not None else config.ORG_INDEX_MIN_RATE
        for org in orgs:
            build(org, a.db, only_repos=a.repos, max_branches=a.max_branches,
                  include_archived=a.include_archived,
                  pushed_within_months=pwm, min_rate=mr,
                  resume=a.resume, gh_host=host)
    elif a.cmd == "find":
        find(a.db, a.term, a.limit)
    elif a.cmd == "stranded":
        stranded(a.db, a.stale_days, org=a.org)
    elif a.cmd == "branches":
        branches(a.db, a.repo, org=a.org)
    elif a.cmd == "stats":
        stats(a.db)

if __name__ == "__main__":
    main()
