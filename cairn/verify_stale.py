#!/usr/bin/env python3
"""Verify memories that assert current repo state (spec 4.1).

Cairn has no passive decay, and deliberately so: confidence is decoupled from
ranking, so decaying it would relabel without changing what is retrieved. The
real gap decay was proposed to fill is different — a memory can become false
because the world moved and nothing was written about it. Nightly contradiction
detection only fires when a NEWER memory disagrees; it cannot see silent drift.

This closes that gap for the checkable subset: memories asserting the current
existence of a file. If the file is gone, the assertion is at least suspect.

WHAT IS NOT VERIFIED, deliberately:
  * `correction` and `decision` are historical records. "We moved X to Y because
    Z" stays true forever, and the file it names may legitimately no longer
    exist. Archiving those would destroy the learning trail this system exists
    to keep.
  * `preference` and `person` make no repo claim at all.
Only `fact`, `project`, and `skill` assert present-tense state.

MEASURED EVIDENCE — file-absence is WEAK evidence of staleness. The first live
run flagged 52 true memories about CAN register findings solely because the
/tmp file they were harvested into had been cleaned up, and after excluding
ephemeral paths it still flagged memories like "forscan-license-architecture"
whose content is knowledge DERIVED FROM a file rather than a claim that the
file exists. Deleting the artefact does not falsify what was learned from it.

So this pass FLAGS by default and does not archive. `--apply` writes flags to
the annotation log for review; `--archive` additionally archives, stamping a
versioned source_ref so the whole pass is retractable in one command:

    .venv/bin/python cairn/query.py --archive-by-source-ref verify-stale-v1 "bad pass"
"""
import argparse
import os
import re
import sys

try:
    import pysqlite3 as sqlite3
except ImportError:  # pragma: no cover - guarded by tests/test_sqlite_guard.py
    if os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") != "1":
        raise RuntimeError(
            "cairn requires pysqlite3; set CAIRN_ALLOW_STDLIB_SQLITE=1 to override"
        )
    import sqlite3

INPUT_DOMAIN_INVARIANT = (
    "Assumes a missing file means the memory is stale. That is FALSE for a repo "
    "that was moved, renamed, or is simply not checked out on this machine, so "
    "the pass verifies only present-tense types, requires the repo root to exist "
    "before judging anything inside it, and never archives without --apply."
)

VERIFY_VERSION = "verify-stale-v1"

# Types that assert present-tense state. Everything else is a historical record.
VERIFIABLE_TYPES = ("fact", "project", "skill")

_PATH_RE = re.compile(r"(?:^|[\s`'\"(])((?:/|\./|[\w.-]+/)[\w./-]*\.\w{1,6})")

# Locations whose contents are expected to vanish. A memory citing a scratch
# file is not falsified when that file is cleaned up: the FIRST live run of this
# pass flagged 52 true memories about CAN register findings purely because the
# /tmp file they were harvested into no longer existed. Absence of an ephemeral
# path is evidence about the filesystem, not about the claim.
_EPHEMERAL_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/shm/", "/run/",
                       "/private/tmp/", "/proc/", "/sys/")
_EPHEMERAL_PARTS = ("/.cache/", "/scratch/", "/scratchpad/", "/node_modules/",
                    "/.venv/", "/__pycache__/", "/.git/")


def is_ephemeral(path):
    """True for paths whose disappearance carries no information about truth."""
    p = path if path.startswith("/") else "/" + path
    return (p.startswith(_EPHEMERAL_PREFIXES)
            or any(part in p for part in _EPHEMERAL_PARTS))


def extract_paths(text):
    """Candidate file paths mentioned in a memory.

    Requires a directory separator and an extension: bare words and sentence
    fragments produce false staleness, and a false archive is far more costly
    than a missed one.
    """
    if not text:
        return set()
    out = set()
    for m in _PATH_RE.finditer(text):
        p = m.group(1).rstrip(".,);:")
        if len(p) > 3:
            out.add(p)
    return out


def classify(mem_type, paths, repo_root, exists=os.path.exists):
    """-> ("skip"|"ok"|"stale", reason).

    `repo_root` None, or a root that does not exist, yields "skip": the repo is
    not checked out here and nothing inside it can be judged. Treating that as
    staleness would archive a project's memories on any machine that lacks it.
    """
    if mem_type not in VERIFIABLE_TYPES:
        return "skip", f"type {mem_type} is a historical record"
    if not paths:
        return "skip", "no file assertion"
    # Absolute paths are self-resolving and need no repo root. Relative ones do,
    # and are only judged when the root is known AND present on this machine —
    # otherwise "not checked out here" is indistinguishable from "deleted".
    paths = {p for p in paths if not is_ephemeral(p)}
    if not paths:
        return "skip", "only ephemeral paths referenced"
    resolvable = {p: p for p in paths if p.startswith("/")}
    rel = [p for p in paths if not p.startswith("/")]
    if rel and repo_root and exists(repo_root):
        for p in rel:
            resolvable[p] = os.path.join(repo_root, p)
    if not resolvable:
        return "skip", "no resolvable path (repo root unknown or absent here)"

    missing = sorted(p for p, full in resolvable.items() if not exists(full))
    if not missing:
        return "ok", f"all {len(resolvable)} referenced file(s) present"
    if len(missing) == len(resolvable):
        return "stale", f"all referenced files missing: {', '.join(missing[:3])}"
    return "skip", f"partially missing ({len(missing)}/{len(resolvable)}) — ambiguous"


def run(apply_changes=False, limit=None, db_path=None, project=None, archive=False):
    from cairn.relevance import _durable_path

    conn = sqlite3.connect(_durable_path(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    q = ("SELECT m.id, m.type, m.topic, m.content, m.associated_files, m.project, "
         "s.transcript_path FROM memories m LEFT JOIN sessions s "
         "ON s.session_id = m.session_id "
         "WHERE (m.archived_reason IS NULL OR m.archived_reason = '') "
         "AND m.deleted_at IS NULL AND m.type IN "
         f"({','.join('?' * len(VERIFIABLE_TYPES))})")
    args = list(VERIFIABLE_TYPES)
    if project:
        q += " AND m.project = ?"
        args.append(project)
    q += " ORDER BY m.id DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = conn.execute(q, args).fetchall()

    roots = {}
    for pr, path in conn.execute(
            "SELECT project, transcript_path FROM sessions WHERE project IS NOT NULL "
            "AND transcript_path IS NOT NULL"):
        if pr not in roots and path and os.path.isdir(path):
            roots[pr] = path

    stale, checked = [], 0
    for mid, mtype, topic, content, assoc, proj, _tp in rows:
        paths = extract_paths(content) | extract_paths(assoc or "")
        verdict, reason = classify(mtype, paths, roots.get(proj))
        if verdict == "skip":
            continue
        checked += 1
        if verdict == "stale":
            stale.append((mid, mtype, topic, reason))

    print(f"=== Stale-assertion verification ({VERIFY_VERSION}) ===")
    print(f"  candidates: {len(rows)}   verifiable: {checked}   stale: {len(stale)}")
    for mid, mtype, topic, reason in stale[:25]:
        print(f"  [{mid}] {mtype}/{topic}\n      {reason}")
    if len(stale) > 25:
        print(f"  ... and {len(stale) - 25} more")

    if not apply_changes:
        print("\n  DRY RUN — nothing archived. Re-run with --apply to archive.")
        conn.close()
        return {"checked": checked, "stale": len(stale), "archived": 0}

    # Annotate, do not archive. See MEASURED EVIDENCE in the module docstring:
    # a missing file usually means the ARTEFACT was cleaned up, not that the
    # knowledge derived from it became false.
    for mid, _t, _to, reason in stale:
        conn.execute(
            "INSERT INTO memory_annotation_log (memory_id, direction, reason) "
            "VALUES (?, 'flag', ?)", (mid, f"stale-assertion candidate: {reason}"))
    conn.commit()

    archived = 0
    if archive:
        for mid, _t, _to, reason in stale:
            conn.execute(
                "UPDATE memories SET confidence = 0, archived_reason = ?, "
                "source_ref = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (f"stale-assertion: {reason}", VERIFY_VERSION, mid))
            conn.execute(
                "INSERT INTO memory_annotation_log (memory_id, direction, reason) "
                "VALUES (?, 'archive', ?)", (mid, f"stale-assertion: {reason}"))
        archived = len(stale)
        conn.commit()
    conn.close()

    print(f"\n  Flagged {len(stale)} for review (not archived).")
    if archived:
        print(f"  Archived {archived}. Retract the whole pass with:\n"
              f"    query.py --archive-by-source-ref {VERIFY_VERSION} <reason>")
    else:
        print("  --archive would archive them, but read the module docstring "
              "first:\n  file-absence is WEAK evidence and the first live run "
              "flagged true memories.")
    return {"checked": checked, "stale": len(stale), "archived": archived}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="write flags to the annotation log")
    ap.add_argument("--archive", action="store_true",
                    help="ALSO archive flagged memories — file-absence is weak evidence, read the docstring")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--project")
    a = ap.parse_args()
    run(apply_changes=a.apply or a.archive, limit=a.limit, project=a.project,
        archive=a.archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
