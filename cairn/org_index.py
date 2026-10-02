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
    deps      harvest dependency declarations (package.xml, .gitmodules,
              requirements, pyproject, Dockerfiles, .repos) into a
              reverse-dependency table
    consumers who declares a dependency on a package or repo
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

# ------------------------------------------------------------------ dependencies
#
# The reverse-dependency layer: who depends on what, from each repo's own
# declarations on its default branch. Per-repo files are the truth; this table
# is the derived cross-repo join, rebuilt nightly and never hand-maintained.
# Blob contents are cached by sha (immutable), so a nightly run fetches only
# files that changed.

import base64
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

DEPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    sha     TEXT PRIMARY KEY,
    content TEXT
);
CREATE TABLE IF NOT EXISTS dependencies (
    org     TEXT NOT NULL,
    repo    TEXT NOT NULL,   -- the declaring repo
    path    TEXT NOT NULL,   -- the file that declares it
    kind    TEXT NOT NULL,   -- provides | ros | python | submodule | git
    target  TEXT NOT NULL,   -- package name, dist name, or owner/repo
    detail  TEXT             -- depend tag, version spec, submodule path, or ref
);
CREATE INDEX IF NOT EXISTS idx_deps_target ON dependencies(kind, target);
CREATE INDEX IF NOT EXISTS idx_deps_repo ON dependencies(org, repo);
"""

_ROS_DEPEND_TAGS = {"depend", "build_depend", "exec_depend", "build_export_depend",
                    "test_depend", "run_depend", "buildtool_depend"}
_GH_REPO = re.compile(r"github\.com[:/]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?"
                      r"(?:@([A-Za-z0-9_./-]+))?(?=[\s\"'#]|$)")
_REQ_LINE = re.compile(r"^\s*[\"']?([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(\[[^\]]*\])?\s*([<>=!~].*?)?[\"',]*\s*$")


def _dep_file_kind(basename):
    if basename == "package.xml":
        return "package.xml"
    if basename == ".gitmodules":
        return "gitmodules"
    if basename == "pyproject.toml":
        return "pyproject"
    if re.match(r"requirements.*\.txt$", basename):
        return "requirements"
    if basename.startswith("Dockerfile") or basename.endswith(".dockerfile"):
        return "dockerfile"
    if basename.endswith(".repos"):
        return "repos"
    return None


def norm_dist(name):
    """PEP 503 normalisation: 'Foo_Bar.baz' -> 'foo-bar-baz'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_package_xml(text):
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    out = []
    name = (root.findtext("name") or "").strip()
    if name:
        out.append(("provides", name, None))
    for el in root:
        if el.tag in _ROS_DEPEND_TAGS and el.text and el.text.strip() != name:
            out.append(("ros", el.text.strip(), el.tag))
    return out


def parse_gitmodules(text):
    out, path = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[submodule"):
            path = None
        elif line.startswith("path"):
            path = line.split("=", 1)[1].strip()
        elif line.startswith("url"):
            m = _GH_REPO.search(line.split("=", 1)[1].strip() + " ")
            if m:
                out.append(("submodule", f"{m.group(1)}/{m.group(2)}".lower(), path))
    return out


def _git_refs(text):
    out = []
    for line in text.splitlines():
        # Comments and Dockerfile LABELs mention repos (homepage, "see ...") without
        # depending on them. `#` only starts a comment at line start or after
        # whitespace; `repo.git#egg=x` fragments are part of the URL.
        line = re.split(r"(?:^|\s)#", line, maxsplit=1)[0]
        if re.match(r"\s*LABEL\b", line, re.I):
            continue
        for m in _GH_REPO.finditer(line):
            ref = m.group(3)
            if not ref:
                b = re.search(r"(?:-b|--branch)[ =]([A-Za-z0-9_./-]+)", line)
                ref = b and b.group(1)
            out.append(("git", f"{m.group(1)}/{m.group(2)}".lower(), ref))
    return out


def parse_requirements(text):
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-") or "://" in line and "@" not in line:
            continue
        if "git+" in line or "github.com" in line:
            continue
        m = _REQ_LINE.match(line)
        if m:
            out.append(("python", norm_dist(m.group(1)), (m.group(3) or "").strip() or None))
    return out + _git_refs(text)


def parse_pyproject(text):
    try:
        import tomllib
        data = tomllib.loads(text)
    except Exception:
        # Unstructured fallback: only lines that are plainly VCS requirements, so
        # [project.urls] homepage links are not read as dependencies.
        return _git_refs("\n".join(l for l in text.splitlines() if "git+" in l))
    deps = list((data.get("project") or {}).get("dependencies") or [])
    for group in ((data.get("project") or {}).get("optional-dependencies") or {}).values():
        deps += group
    poetry = ((data.get("tool") or {}).get("poetry") or {}).get("dependencies") or {}
    deps += [k for k in poetry if k.lower() != "python"]
    deps += (data.get("build-system") or {}).get("requires") or []
    # Dependency fields only; [project.urls] and friends are not dependencies.
    vcs = [d for d in deps if isinstance(d, str)]
    for spec in poetry.values():
        if isinstance(spec, dict) and spec.get("git"):
            vcs.append(f"{spec['git']}" + (f"@{spec['branch']}" if spec.get("branch") else ""))
    for spec in (((data.get("tool") or {}).get("uv") or {}).get("sources") or {}).values():
        if isinstance(spec, dict) and spec.get("git"):
            vcs.append(f"{spec['git']}" + (f"@{spec['branch']}" if spec.get("branch") else ""))
    out = []
    for d in deps:
        if not isinstance(d, str):
            continue
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(\[[^\]]*\])?\s*(.*)", d)
        if m and "github.com" not in d:
            out.append(("python", norm_dist(m.group(1)), m.group(3).strip() or None))
    return out + _git_refs("\n".join(vcs))


def parse_repos_file(text):
    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except Exception:
        return []
    out = []
    for spec in (data.get("repositories") or {}).values():
        m = _GH_REPO.search(str((spec or {}).get("url", "")) + " ")
        if m:
            out.append(("git", f"{m.group(1)}/{m.group(2)}".lower(), (spec or {}).get("version")))
    return out


_PARSERS = {"package.xml": parse_package_xml, "gitmodules": parse_gitmodules,
            "requirements": parse_requirements, "pyproject": parse_pyproject,
            "dockerfile": _git_refs, "repos": parse_repos_file}


def _fetch_blob(org, repo, sha):
    obj = gh_obj(f"repos/{org}/{repo}/git/blobs/{sha}")
    if not obj or obj.get("encoding") != "base64":
        return sha, None
    try:
        return sha, base64.b64decode(obj["content"]).decode("utf-8", "replace")
    except (ValueError, KeyError):
        return sha, None


def build_deps(org, db_path, verbose=True, workers=8):
    """Harvest dependency declarations from every indexed default branch."""
    con = connect(db_path)
    con.executescript(DEPS_SCHEMA)
    rows = con.execute(
        "SELECT f.repo, f.path, f.basename, f.blob_sha FROM files f "
        "JOIN repos r ON r.org=f.org AND r.name=f.repo "
        "WHERE f.org=? AND f.branch=r.default_branch AND r.archived=0", (org,)).fetchall()
    files = [(repo, path, kind, sha) for repo, path, base, sha in rows
             if sha and (kind := _dep_file_kind(base))]
    have = {s for (s,) in con.execute("SELECT sha FROM blobs")}
    todo = {}
    for repo, path, kind, sha in files:
        if sha not in have:
            todo.setdefault(sha, repo)
    if verbose:
        print(f"deps {org}: {len(files)} declaration file(s), {len(todo)} blob(s) to fetch")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sha, content in pool.map(lambda kv: _fetch_blob(org, kv[1], kv[0]), todo.items()):
            if content is not None:
                con.execute("INSERT OR REPLACE INTO blobs(sha, content) VALUES (?,?)", (sha, content))
    con.commit()
    blobs = dict(con.execute("SELECT sha, content FROM blobs"))
    con.execute("DELETE FROM dependencies WHERE org=?", (org,))
    n = 0
    for repo, path, kind, sha in files:
        text = blobs.get(sha)
        if text is None:
            continue
        for dkind, target, detail in _PARSERS[kind](text):
            con.execute("INSERT INTO dependencies(org, repo, path, kind, target, detail) "
                        "VALUES (?,?,?,?,?,?)", (org, repo, path, dkind, target, detail))
            n += 1
    con.commit()
    if verbose:
        print(f"deps {org}: {n} declaration(s) recorded")
    return n


def consumers(db_path, target, org=None, as_json=False):
    """Who depends on TARGET: a ROS package, a Python dist, or a repo (ORG/REPO
    or bare REPO, matched as a submodule or a git ref)."""
    con = connect(db_path)
    con.executescript(DEPS_SCHEMA)
    # Exact matches only: LIKE would treat the '_' in ROS package names as a
    # wildcard and match 'nav_msgs' against the repo 'nav-msgs'.
    repo = target.lower()
    params = [target, norm_dist(target), repo, "/" + repo, len(repo) + 1]
    q = ("SELECT org, repo, path, kind, target, detail FROM dependencies WHERE "
         "((kind='ros' AND target=?) OR (kind='python' AND target=?) OR "
         "(kind IN ('submodule','git') AND (target=? OR substr(target, -?) = ?)))")
    params = params[:3] + [params[4], params[3]]
    if org:
        q += " AND org=?"
        params.append(org)
    rows = con.execute(q + " ORDER BY org, repo, path", params).fetchall()
    if as_json:
        print(json.dumps([dict(zip(("org", "repo", "path", "kind", "target", "detail"), r))
                          for r in rows], indent=2))
        return rows
    if not rows:
        print(f"No declared consumers of '{target}'.")
    for o, repo, path, kind, tgt, detail in rows:
        print(f"{o+'/'+repo:40} {kind:9} {path}" + (f"  ({detail})" if detail else ""))
    return rows

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

    d = sub.add_parser("deps", help="harvest dependency declarations from default branches")
    d.add_argument("--orgs", nargs="*", default=None,
                   help="orgs to harvest (default: cairn config ORG_INDEX_ORGS)")

    c = sub.add_parser("consumers", help="who declares a dependency on a package or repo")
    c.add_argument("target", help="ROS package, Python dist, or [ORG/]REPO")
    c.add_argument("--org", default=None)
    c.add_argument("--json", action="store_true")

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
    elif a.cmd == "deps":
        from cairn import config
        orgs = a.orgs or config.ORG_INDEX_ORGS
        if not orgs:
            sys.exit("no orgs to harvest: pass --orgs or set ORG_INDEX_ORGS "
                     "(and ORG_INDEX_ENABLED) in cairn config")
        for org in orgs:
            build_deps(org, a.db)
    elif a.cmd == "consumers":
        consumers(a.db, a.target, org=a.org, as_json=a.json)

if __name__ == "__main__":
    main()
