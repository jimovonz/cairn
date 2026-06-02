#!/usr/bin/env python3
"""Monitor Claude Code transcripts for the unrecoverable "thinking block cannot
be modified" 400 error (a CLI regression first seen on v2.1.156, 2026-05-28).

The Anthropic API rejects a session permanently when a `thinking`/`redacted_thinking`
block in the first assistant turn comes back altered from its original form. Every
retry re-sends the broken array and gets the same 400 — the session is bricked.

This is a POST-HOC scanner: the Stop hook never fires on a 400 (no assistant
completion), so we detect at the transcript level instead. For each genuine
occurrence it captures a structural snapshot of the offending assistant turn —
exactly the reproduction detail a bug report needs.

Usage:
    python3 cairn/thinking_block_monitor.py            # report all occurrences
    python3 cairn/thinking_block_monitor.py --new      # only occurrences not yet seen
    python3 cairn/thinking_block_monitor.py --json      # machine-readable

State of already-reported occurrences is kept in cairn/.thinking_block_seen.json
so --new is quiet once you've triaged the backlog (suitable for cron/--loop).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

SIG = "blocks in the latest assistant message cannot be modified"
PROJECTS = os.path.expanduser("~/.claude/projects")
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".thinking_block_seen.json")


def _is_real_error(rec):
    """Return the error text iff this record is a genuine API-error assistant turn,
    not prose/memory that merely quotes the signature string."""
    if rec.get("type") != "assistant":
        return None
    for b in rec.get("message", {}).get("content", []):
        if isinstance(b, dict) and b.get("type") == "text":
            t = b.get("text", "")
            if t.lstrip().startswith("API Error") and "cannot be modified" in t:
                return t.strip()
    return None


def _turn_snapshot(records, error_idx):
    """Structural snapshot of the first assistant turn — the repro signal.

    The 400 always points at messages.1 (first assistant turn). Claude Code emits
    each tool_use as its own JSONL record, so we reconstruct the turn by walking
    the assistant records that precede the error within the same turn block."""
    kinds = Counter()
    thinking = tool_use = 0
    for rec in records:
        if rec.get("type") != "assistant":
            continue
        if _is_real_error(rec):
            break
        for b in rec.get("message", {}).get("content", []):
            if isinstance(b, dict):
                k = b.get("type", "?")
                kinds[k] += 1
                if k in ("thinking", "redacted_thinking"):
                    thinking += 1
                if k == "tool_use":
                    tool_use += 1
    return {"block_composition": dict(kinds), "thinking_blocks": thinking, "tool_use_blocks": tool_use}


def scan():
    out = []
    for f in glob.glob(os.path.join(PROJECTS, "**", "*.jsonl"), recursive=True):
        try:
            records = []
            for line in open(f):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
        errs = [(i, r) for i, r in enumerate(records) if _is_real_error(r)]
        if not errs:
            continue
        first_idx, first_rec = errs[0]
        out.append({
            "session": os.path.basename(f)[:8],
            "file": f,
            "first_error_ts": first_rec.get("timestamp", ""),
            "occurrences": len(errs),
            "error": _is_real_error(first_rec)[:140],
            "turn_snapshot": _turn_snapshot(records, first_idx),
        })
    out.sort(key=lambda h: h["first_error_ts"])
    return out


def _load_seen():
    try:
        return set(json.load(open(STATE)))
    except (OSError, json.JSONDecodeError):
        return set()


def _save_seen(keys):
    try:
        json.dump(sorted(keys), open(STATE, "w"))
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", action="store_true", help="only show occurrences not yet recorded as seen")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    hits = scan()
    seen = _load_seen()
    if args.new:
        hits = [h for h in hits if h["session"] not in seen]

    if args.json:
        print(json.dumps(hits, indent=2))
    else:
        if not hits:
            print("No new thinking-block 400 corruption detected." if args.new
                  else "No thinking-block 400 corruption found in any transcript.")
        else:
            sessions = {h["session"] for h in hits}
            total = sum(h["occurrences"] for h in hits)
            print(f"⚠ thinking-block 400 corruption: {total} error record(s) across {len(sessions)} session(s)\n")
            for h in hits:
                s = h["turn_snapshot"]
                print(f"  session {h['session']}  first: {h['first_error_ts']}  ({h['occurrences']} record(s))")
                print(f"    first assistant turn: {s['thinking_blocks']} thinking, {s['tool_use_blocks']} tool_use "
                      f"| {h['error'].split(':',2)[1].strip() if ':' in h['error'] else ''}")
                print(f"    file: {h['file']}")

    # Record everything scanned as seen so --new stays quiet next run.
    _save_seen(seen | {h["session"] for h in scan()})
    # Non-zero exit when new corruption is present — lets cron/--loop alert.
    return 1 if (args.new and hits) else 0


if __name__ == "__main__":
    sys.exit(main())
