#!/usr/bin/env python3
"""
cairn_verify.py — closes the loop between Cairn memories and ground truth.

Cairn memories make location claims ("file X lives in repo Y") in their `facts`
column (e.g. `repo:/mnt/ssd/Projects/debian-var`, `file:tools/board_test.py`).
Per the memory rules, such a claim is only true *as of when it was written* — a
file can be renamed, deleted, or (the case that bit us) live only on an unmerged
branch. This tool checks every location-claim memory against the locatability
index (org_index.db) and classifies it:

    OK       file is on the repo's DEFAULT branch   -> claim is currently true
    DRIFT    file exists ONLY on non-default/unmerged branches -> at-risk, the
             exact failure mode of memory 15049 (board_test.py on JO_Prod_Bringup)
    MISSING  file not found on any indexed branch    -> stale/likely wrong claim
    UNKNOWN  repo not in the index                   -> can't verify (clone/scope)

Read-only against both DBs (cairn.db opened mode=ro, so no pysqlite3/WAL concern).

    verify            scan all location-claim memories, summarise + list problems
    verify --id N     check a single memory
"""
import argparse
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX = os.path.join(HERE, "org_index.db")
DEFAULT_CAIRN = "/home/jameo/Projects/cairn/cairn/cairn.db"

FILE_RE = re.compile(r"^file:(.+)$")
REPO_RE = re.compile(r"^repo:(.+)$")

def parse_facts(raw):
    """facts may be a JSON list or a newline-joined string of 'key:value' lines."""
    if not raw:
        return []
    raw = raw.strip()
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [str(x) for x in v]
    except (json.JSONDecodeError, ValueError):
        pass
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

def repo_name_from_fact(repo_fact):
    """'repo:/mnt/ssd/Projects/debian-var' -> 'debian-var'."""
    val = repo_fact.split(":", 1)[1].strip()
    return os.path.basename(val.rstrip("/"))

def file_path_from_fact(file_fact):
    """'file:tools/board_test.py (66kb)' -> 'tools/board_test.py'. Strips a
    trailing parenthetical note and any symbol suffix after a space."""
    val = file_fact.split(":", 1)[1].strip()
    val = re.split(r"\s+\(", val, 1)[0]      # drop "(...)" note
    val = val.split(" ", 1)[0]               # drop "file.py foo/bar" symbol lists
    return val.strip()

class Index:
    def __init__(self, path):
        if not os.path.exists(path):
            sys.exit(f"locatability index not found: {path}\n"
                     f"Build it first: org_index.py build --org <org>")
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.repos = {r[0] for r in self.con.execute("SELECT name FROM repos")}

    def locate(self, repo, path):
        """Return (status, branches) for a (repo, path) claim. Org-agnostic:
        matches the repo name in any indexed org (repo names are ~unique), with
        the branches join scoped by org so same-named repos don't cross-join."""
        if repo not in self.repos:
            return "UNKNOWN", []
        base = path.rsplit("/", 1)[-1]
        rows = self.con.execute(
            """SELECT f.branch, b.is_default
               FROM files f JOIN branches b
                 ON b.org=f.org AND b.repo=f.repo AND b.branch=f.branch
               WHERE f.repo=? AND (f.path=? OR f.path LIKE ? OR f.basename=?)""",
            (repo, path, f"%{path}", base)).fetchall()
        if not rows:
            return "MISSING", []
        on_default = [br for br, isd in rows if isd]
        if on_default:
            return "OK", on_default
        return "DRIFT", sorted({br for br, _ in rows})

def iter_location_memories(cairn_path, mem_id=None):
    con = sqlite3.connect(f"file:{cairn_path}?mode=ro", uri=True)
    q = ("SELECT id, topic, facts, project, created_at FROM memories "
         "WHERE deleted_at IS NULL AND facts LIKE '%file:%'")
    args = ()
    if mem_id is not None:
        q += " AND id=?"
        args = (mem_id,)
    for row in con.execute(q, args):
        yield row

def verify(cairn_path, index_path, mem_id=None, show_ok=False):
    idx = Index(index_path)
    counts = {"OK": 0, "DRIFT": 0, "MISSING": 0, "UNKNOWN": 0, "NO_REPO": 0}
    problems = []
    for mid, topic, facts_raw, project, created in iter_location_memories(
            cairn_path, mem_id):
        facts = parse_facts(facts_raw)
        files = [file_path_from_fact(f) for f in facts if FILE_RE.match(f)]
        repos = [repo_name_from_fact(f) for f in facts if REPO_RE.match(f)]
        repo = repos[0] if repos else None
        # If no repo: fact, try to infer from project name (often the repo name).
        if not repo and project:
            cand = os.path.basename(project.rstrip("/"))
            if cand in idx.repos:
                repo = cand
        if not repo:
            counts["NO_REPO"] += 1
            continue
        for path in files:
            if not path or "/" not in path and "." not in path:
                continue
            status, branches = idx.locate(repo, path)
            counts[status] += 1
            if status != "OK" or show_ok:
                problems.append((status, mid, repo, path, branches, topic))

    print("=== Cairn location-claim verification ===")
    print(f"OK={counts['OK']}  DRIFT={counts['DRIFT']}  MISSING={counts['MISSING']}"
          f"  UNKNOWN={counts['UNKNOWN']}  (no-repo skipped={counts['NO_REPO']})")
    print()
    order = {"DRIFT": 0, "MISSING": 1, "UNKNOWN": 2, "OK": 3}
    for status, mid, repo, path, branches, topic in sorted(
            problems, key=lambda x: order.get(x[0], 9)):
        extra = ""
        if status == "DRIFT":
            extra = f"  ONLY on: {', '.join(branches)}  ⚠ unmerged"
        print(f"[{status:7}] #{mid:<6} {repo}/{path}{extra}")
        if status in ("DRIFT", "MISSING"):
            print(f"            ↳ \"{topic}\"")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cairn", default=DEFAULT_CAIRN)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--id", type=int, help="check a single memory id")
    ap.add_argument("--show-ok", action="store_true", help="also list verified claims")
    a = ap.parse_args()
    verify(a.cairn, a.index, mem_id=a.id, show_ok=a.show_ok)

if __name__ == "__main__":
    # Re-exec under the cairn venv so `import cairn.*` resolves (mirrors
    # graph_fleet.py / query.py). No-op when already inside a venv.
    _venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".venv", "bin", "python3")
    if os.path.exists(_venv_python) and sys.prefix == sys.base_prefix:
        os.execv(_venv_python, [_venv_python] + sys.argv)
    main()
