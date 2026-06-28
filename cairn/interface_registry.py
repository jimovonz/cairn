#!/usr/bin/env python3
"""
interface_registry.py — the *interface/symbol* layer of the org-wide index.

`code-review-graph` builds a per-repo graph (`.code-review-graph/graph.db`) with
zero cross-repo edges. This tool rolls those per-repo graphs up into a single
cross-repo *consumer registry* harvested for free from the `IMPORTS_FROM` edges
already in each graph — i.e. the side-index design we'd scoped previously, with
no new extraction pass.

For every IMPORTS_FROM edge we record (consumer_repo, consumer_file, target).
Targets are classified:
    local     — resolvable within the same repo (relative path, or a file/symbol
                node that exists in that repo's graph)  -> intra-repo, ignored
    external  — a bare module name not resolvable locally -> a cross-repo / 3rd
                party dependency edge worth tracking
The registry then answers:
    consumers <module>  who across the org imports <module>
    producers           bare modules ranked by how many repos consume them
                        (the de-facto shared interfaces — change these carefully)
    orphans             modules consumed by nobody's graph but present (dead-ish)

This is intentionally orthogonal to org_index.py (locatability). It indexes the
*default-branch surface* — what graphs exist on disk — not stranded branches.
"""
import argparse
import glob
import json
import os
import re
import sqlite3
import sys

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interface_registry.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS imports (
    consumer_repo TEXT NOT NULL,
    consumer_file TEXT NOT NULL,
    target        TEXT NOT NULL,   -- raw IMPORTS_FROM target_qualified
    target_norm   TEXT NOT NULL,   -- normalised bare module/top-level name
    kind          TEXT NOT NULL,   -- 'local' | 'external'
    line          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_imports_target ON imports(target_norm);
CREATE INDEX IF NOT EXISTS idx_imports_repo   ON imports(consumer_repo);
CREATE TABLE IF NOT EXISTS scanned_repos (
    repo TEXT PRIMARY KEY, graph_path TEXT, n_imports INTEGER
);
"""

def connect(db_path):
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con

# A target is "local" if it looks like a relative path (./ ../) or resolves to a
# node/file that exists inside the same repo's graph. Otherwise it is external
# (a bare module name — candidate cross-repo / third-party interface).
REL_RE = re.compile(r"^\.{1,2}/")

def normalise(target):
    """Reduce an import target to a comparable top-level module/name.
    'rpl_msgs.srv.Foo' -> 'rpl_msgs'; '@rpl/telemetry' -> '@rpl/telemetry';
    'foo/bar.py' -> 'foo'; bare 'numpy' -> 'numpy'."""
    t = target.strip()
    if REL_RE.match(t):
        return t  # keep relative paths verbatim (clearly local)
    t = t.lstrip("./")
    # python dotted -> top package; path-ish -> first segment
    if "/" in t and not t.startswith("@"):
        t = t.split("/", 1)[0]
    if "." in t and not t.startswith("@"):
        t = t.split(".", 1)[0]
    return t

# Heavy dirs to prune in the standalone fallback walk. glob('**') is unusable
# here: it skips dotted path segments, so it never finds .code-review-graph.
PRUNE = {".git", "node_modules", ".venv", "venv", "__pycache__",
         ".tox", "build", "dist", ".mypy_cache", "target"}

def discover_graphs(roots):
    """Map repo-name -> graph.db path. Prefers cairn's fleet discovery
    (symlink-resolved, de-duped, depth-bounded, same roots the watch daemon
    uses) so this layer sits *on top of* graph_fleet rather than reinventing
    it. Falls back to a pruning os.walk when run outside the cairn package."""
    try:
        from cairn.graph_fleet import discover_repos
    except ImportError:
        return _discover_graphs_walk(roots)
    found = {}
    for repo in discover_repos(roots):   # roots=None -> cairn default roots
        gdb = os.path.join(repo, ".code-review-graph", "graph.db")
        if os.path.exists(gdb):
            found.setdefault(os.path.basename(repo.rstrip("/")), gdb)
    return found

def _discover_graphs_walk(roots):
    found = {}
    for root in roots or []:
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == ".code-review-graph":
                if "graph.db" in filenames:
                    repo = os.path.basename(os.path.dirname(dirpath).rstrip("/"))
                    found.setdefault(repo, os.path.join(dirpath, "graph.db"))
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in PRUNE]
    return found

def repo_local_names(con_g):
    """Set of names resolvable within a repo's own graph: file paths, basenames,
    and qualified/simple node names. Used to decide local vs external."""
    names = set()
    for (fp,) in con_g.execute("SELECT DISTINCT file_path FROM nodes"):
        if fp:
            names.add(fp)
            names.add(os.path.basename(fp))
            names.add(os.path.splitext(os.path.basename(fp))[0])
    for (qn,) in con_g.execute("SELECT DISTINCT qualified_name FROM nodes"):
        if qn:
            names.add(qn)
            names.add(qn.rsplit("/", 1)[-1])
            names.add(qn.rsplit(".", 1)[-1])
    return names

def build(roots, db_path, verbose=True):
    con = connect(db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM imports")
    cur.execute("DELETE FROM scanned_repos")
    graphs = discover_graphs(roots)
    if verbose:
        print(f"[build] found {len(graphs)} repo graphs", file=sys.stderr)
    for repo, gpath in sorted(graphs.items()):
        try:
            g = sqlite3.connect(f"file:{gpath}?mode=ro", uri=True)
        except sqlite3.Error as e:
            print(f"  ! {repo}: {e}", file=sys.stderr)
            continue
        local = repo_local_names(g)
        edges = g.execute(
            "SELECT source_qualified, target_qualified, line "
            "FROM edges WHERE kind='IMPORTS_FROM'").fetchall()
        n = 0
        for src, tgt, line in edges:
            norm = normalise(tgt)
            is_local = bool(REL_RE.match(tgt)) or norm in local or tgt in local
            kind = "local" if is_local else "external"
            cur.execute(
                "INSERT INTO imports VALUES (?,?,?,?,?,?)",
                (repo, src, tgt, norm, kind, line))
            n += 1
        cur.execute("INSERT OR REPLACE INTO scanned_repos VALUES (?,?,?)",
                    (repo, gpath, n))
        g.close()
        if verbose:
            print(f"  {repo}: {n} import edges", file=sys.stderr)
    con.commit()
    con.close()

def consumers(db_path, module):
    con = connect(db_path)
    norm = normalise(module)
    rows = con.execute(
        """SELECT consumer_repo, consumer_file, target, line
           FROM imports WHERE target_norm=? OR target=?
           ORDER BY consumer_repo""", (norm, module)).fetchall()
    if not rows:
        print(f"No consumers of '{module}' (norm '{norm}').")
        return
    repos = sorted({r[0] for r in rows})
    print(f"'{module}' consumed by {len(rows)} sites across {len(repos)} repos: "
          f"{', '.join(repos)}")
    for repo, f, tgt, line in rows:
        print(f"  {repo:28} {os.path.basename(f)}:{line or '?'}  ({tgt})")

def producers(db_path, min_repos=2, limit=40):
    """Bare external modules ranked by cross-repo fan-in — the de-facto shared
    interfaces. High repo-count = high blast radius if changed."""
    con = connect(db_path)
    rows = con.execute(
        """SELECT target_norm,
                  COUNT(DISTINCT consumer_repo) AS repos,
                  COUNT(*) AS sites
           FROM imports WHERE kind='external'
           GROUP BY target_norm
           HAVING repos >= ?
           ORDER BY repos DESC, sites DESC
           LIMIT ?""", (min_repos, limit)).fetchall()
    if not rows:
        print(f"No external module consumed by >= {min_repos} repos. "
              f"(Need graphs from multiple repos built.)")
        return
    print(f"{'MODULE':40} {'REPOS':>5} {'SITES':>6}   (cross-repo shared interfaces)")
    print("-" * 70)
    for mod, repos, sites in rows:
        print(f"{mod:40} {repos:>5} {sites:>6}")

def stats(db_path):
    con = connect(db_path)
    sr = con.execute("SELECT COUNT(*) FROM scanned_repos").fetchone()[0]
    tot = con.execute("SELECT COUNT(*) FROM imports").fetchone()[0]
    ext = con.execute("SELECT COUNT(*) FROM imports WHERE kind='external'").fetchone()[0]
    xrepo = con.execute(
        "SELECT COUNT(*) FROM (SELECT target_norm FROM imports WHERE kind='external' "
        "GROUP BY target_norm HAVING COUNT(DISTINCT consumer_repo)>=2)").fetchone()[0]
    print(f"graphs scanned={sr}  import edges={tot}  external={ext}  "
          f"cross-repo shared modules={xrepo}")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="scan .code-review-graph/*.db under roots")
    b.add_argument("--roots", nargs="+", default=None,
                   help="repo-search roots (default: cairn fleet discovery roots)")

    c = sub.add_parser("consumers", help="who imports <module> across the org")
    c.add_argument("module")

    p = sub.add_parser("producers", help="shared interfaces by cross-repo fan-in")
    p.add_argument("--min-repos", type=int, default=2)

    sub.add_parser("stats")

    a = ap.parse_args()
    if a.cmd == "build":
        build(a.roots, a.db)
    elif a.cmd == "consumers":
        consumers(a.db, a.module)
    elif a.cmd == "producers":
        producers(a.db, a.min_repos)
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
