#!/usr/bin/env python3
"""Persist DURABLE, code-attached review knowledge into Cairn.

WHAT TO CAPTURE (read this before invoking). Persist the *durable rationale*
that survives the fix — why the code is shaped this way, an intentional coupling
and the ticket behind it, an accepted trade-off, a justified-and-resolved
decision at merge. Do NOT dump raw transient review findings: "this PR has bug
X" is a claim about a diff-at-a-moment — once fixed and merged it is FALSE of the
code, and persisting it pollutes cairn with self-invalidating bug reports that a
6-months-later reader mistakes for current truth. Fire on RESOLVED/JUSTIFIED
items at merge, capturing the why; not on the raw finding list. (Memory is a
PRIOR to verify against current code, never ground truth.)

The orchestrator calls this after a review to store each durable item keyed to
the *target repo* and its *changed file/symbol* — so a future session working on
that code surfaces it via ``cairn-graph --knowledge SYMBOL`` (which joins on
``associated_files LIKE '%path%'`` and an FTS MATCH on the symbol) and via
normal project-scoped retrieval.

Without this, subagent findings live only in the subagent transcript (now also
capturable via the SubagentStop hook, but keyed to the *session*, not the code).
This path attaches the *why* to the code.

Input (stdin or --file), JSON:
    {
      "repo": "/path/to/target/repo",   # optional, default cwd
      "project": "name",                # optional, default = git-toplevel basename
      "commit": "abc123",               # optional, used in the synthetic session id
      "findings": [
        {
          "file": "src/foo.py",         # required (relative to repo or absolute)
          "symbol": "do_thing",         # optional
          "line": 42,                   # optional
          "type": "correction",         # optional, default "correction"
          "topic": "short topic",       # optional, default derived
          "content": "finding text",    # required
          "severity": "high",           # optional
          "pr": 6,                       # optional
          "keywords": ["..."]           # optional
        }
      ]
    }

Findings are inserted through hooks.storage.insert_memories, which dedups at
cosine 0.85 — so re-running a review is idempotent (near-identical findings are
filtered, not duplicated). No archiving of prior findings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

# Make the repo importable when invoked as a console script or directly.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

VALID_TYPES = {"correction", "decision", "fact", "preference",
               "person", "project", "skill", "workflow"}


def _git_toplevel(path: str) -> Optional[str]:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _resolve_repo(repo: Optional[str]) -> str:
    repo = os.path.abspath(repo or os.getcwd())
    return _git_toplevel(repo) or repo


def _build_entry(finding: dict, repo_root: str) -> dict:
    """Map a review finding to a Cairn memory entry with code-attached keys."""
    raw_file = (finding.get("file") or "").strip()
    if not raw_file:
        raise ValueError("finding missing required 'file'")
    content = (finding.get("content") or "").strip()
    if not content:
        raise ValueError(f"finding for {raw_file} missing required 'content'")

    # Resolve absolute + repo-relative forms. knowledge() matches via substring
    # LIKE against the graph's (usually repo-relative) path, so store both: the
    # absolute path (matched because the relative path is its suffix) and the
    # relative path (portable across machines / matched directly).
    abs_path = raw_file if os.path.isabs(raw_file) else os.path.join(repo_root, raw_file)
    abs_path = os.path.normpath(abs_path)
    try:
        rel_path = os.path.relpath(abs_path, repo_root)
    except ValueError:
        rel_path = raw_file
    assoc = list(dict.fromkeys([abs_path, rel_path]))  # de-dup, preserve order

    symbol = (finding.get("symbol") or "").strip()
    base = os.path.basename(rel_path)
    line = finding.get("line")
    pr = finding.get("pr")
    severity = (finding.get("severity") or "").strip()

    # Default to decision: the intended content is durable rationale, not a
    # transient bug finding. Caller may still pass correction/fact explicitly.
    mem_type = (finding.get("type") or "decision").strip()
    if mem_type not in VALID_TYPES:
        mem_type = "correction"

    topic = (finding.get("topic") or "").strip()
    if not topic:
        topic = f"review: {symbol or base}"

    # Keep content self-sufficient AND ensure the symbol token is present in the
    # embedded/FTS text so --knowledge's FTS MATCH on the symbol hits.
    if symbol and symbol not in content:
        content = f"{content} [{base}:{symbol}]"

    # facts feed the FTS index (topic/content/keywords/facts) and survive
    # compaction — carry the precise code coordinates here.
    facts = [f"file:{rel_path}"]
    if symbol:
        facts.append(f"symbol:{symbol}")
    if line:
        facts.append(f"line:{line}")
    if pr:
        facts.append(f"pr:{pr}")
    if severity:
        facts.append(f"severity:{severity}")

    keywords = list(finding.get("keywords") or [])
    for kw in (symbol, base, "code-review", "review-finding"):
        if kw and kw not in keywords:
            keywords.append(kw)

    return {
        "type": mem_type,
        "topic": topic[:120],
        "content": content,
        "associated_files": assoc,
        "facts": facts,
        "keywords": keywords,
    }


def write_back(payload: dict, dry_run: bool = False) -> dict:
    repo_root = _resolve_repo(payload.get("repo"))
    project = (payload.get("project") or os.path.basename(repo_root)).strip()
    commit = (payload.get("commit") or "").strip()
    findings = payload.get("findings") or []
    if not findings:
        return {"inserted": 0, "project": project, "session": None,
                "skipped": 0, "errors": ["no findings"]}

    entries = []
    errors = []
    for i, f in enumerate(findings):
        try:
            entries.append(_build_entry(f, repo_root))
        except Exception as exc:
            errors.append(f"finding[{i}]: {exc}")

    stamp = commit[:12] if commit else time.strftime("%Y%m%d-%H%M%S")
    session_id = f"review-{project}-{stamp}"

    if dry_run:
        return {"inserted": 0, "project": project, "session": session_id,
                "would_insert": len(entries), "skipped": len(errors),
                "errors": errors, "entries": entries}

    from hooks.hook_helpers import get_conn
    from hooks.storage import insert_memories
    from cairn.config import MAX_MEMORIES_PER_RESPONSE

    # Register the synthetic session so insert_memories tags the memories to the
    # TARGET repo's project (it derives project from sessions.project).
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, project, transcript_path, started_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (session_id, project, repo_root),
    )
    conn.commit()
    conn.close()

    # Batch under the per-call throttle so no finding is silently dropped.
    inserted = 0
    cap = max(1, MAX_MEMORIES_PER_RESPONSE)
    for start in range(0, len(entries), cap):
        chunk = entries[start:start + cap]
        inserted += insert_memories(chunk, session_id=session_id, transcript_path=repo_root,
                                    source_ref="review-writeback")

    return {"inserted": inserted, "project": project, "session": session_id,
            "submitted": len(entries), "skipped": len(errors), "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Persist DURABLE code-attached review rationale into Cairn "
                    "(why/decisions/justified trade-offs at merge — NOT transient bug findings).")
    ap.add_argument("--file", help="Path to findings JSON (default: read stdin)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be inserted without writing")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON result")
    args = ap.parse_args()

    raw = open(args.file).read() if args.file else sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON input: {exc}", file=sys.stderr)
        sys.exit(2)

    result = write_back(payload, dry_run=args.dry_run)

    if args.json:
        print(json.dumps(result, indent=2))
    elif args.dry_run:
        print(f"DRY RUN — project={result['project']} session={result['session']}")
        print(f"  would insert {result.get('would_insert', 0)} finding(s), skipped {result['skipped']}")
        for e in result.get("entries", []):
            print(f"  [{e['type']}] {e['topic']}  files={e['associated_files']}")
        for err in result["errors"]:
            print(f"  ! {err}", file=sys.stderr)
    else:
        print(f"Wrote {result['inserted']} review finding(s) to cairn "
              f"(project={result['project']}, session={result['session']}).")
        if result["skipped"]:
            print(f"Skipped {result['skipped']} malformed finding(s):", file=sys.stderr)
            for err in result["errors"]:
                print(f"  ! {err}", file=sys.stderr)

    sys.exit(0 if not result["errors"] or result["inserted"] or args.dry_run else 1)


if __name__ == "__main__":
    main()
