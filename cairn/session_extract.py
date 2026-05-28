#!/usr/bin/env python3
"""Extract a signal-only view of a Claude Code session JSONL.

Strips tool calls/results, injected <cairn_context> and <system-reminder>
blocks, CLAUDE.md repeats, and thinking blocks. Used as the input cleaner
for the calibration analyser (see docs/spec-calibration-system.md).

Reuses transcript parsing patterns from cairn/benchmark_extract.py.

Usage:
    python3 cairn/session_extract.py <jsonl>                  # --signal-only default
    python3 cairn/session_extract.py <jsonl> --with-tools
    python3 cairn/session_extract.py <jsonl> --corrections-only
    python3 cairn/session_extract.py <jsonl> --turn-range 10-30
    python3 cairn/session_extract.py <jsonl> --last-N-minutes 30
    python3 cairn/session_extract.py <jsonl> --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone


# Injected blocks to strip from user-side text. <cairn_context> and
# <system-reminder> are hook-injected, not authored by the user.
CAIRN_CONTEXT_RE = re.compile(r"<cairn_context\b.*?</cairn_context>", re.DOTALL)
SYSTEM_REMINDER_RE = re.compile(r"<system-reminder\b.*?</system-reminder>", re.DOTALL)
LOCAL_COMMAND_RE = re.compile(r"<local-command-(?:stdout|stderr|caveat)\b.*?</local-command-[^>]+>", re.DOTALL)
COMMAND_TAG_RE = re.compile(r"<command-(?:name|message|args)\b.*?</command-[^>]+>", re.DOTALL)
THINKING_RE = re.compile(r"<thinking\b.*?</thinking>", re.DOTALL)
MEMORY_BLOCK_RE = re.compile(r"\[cm\]:\s*#\s*'.*?'", re.DOTALL)

# Heuristic markers for user correction / redirect turns.
CORRECTION_MARKERS = (
    "no,", "no.", "no ", "nope", "actually", "wrong",
    "that's not", "thats not", "don't ", "dont ",
    "stop ", "instead", "i said", "i told you",
    "you misunderstood", "you misread", "incorrect",
    "rather than", "not what i", "not the",
)


def _content_text(content) -> str:
    """Extract plain text from message content (string or content blocks).

    Thinking blocks are always dropped — they are agent internal state.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            # thinking, tool_use, tool_result — skipped in signal mode
        return "\n".join(p for p in parts if p)
    return ""


def _has_tool_blocks(content) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
        for b in content
    )


def _tool_summary(content) -> str:
    """One-line summary of tool blocks for --with-tools mode."""
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "tool_use":
            name = b.get("name", "?")
            inp = b.get("input", {})
            keys = ",".join(sorted(inp.keys())) if isinstance(inp, dict) else ""
            parts.append(f"[tool_use {name}({keys})]")
        elif b.get("type") == "tool_result":
            res = b.get("content", "")
            if isinstance(res, list):
                res = _content_text(res)
            preview = str(res)[:200].replace("\n", " ")
            parts.append(f"[tool_result {preview}]")
    return " ".join(parts)


def _clean_user_text(text: str) -> str:
    """Strip hook-injected blocks from user-side text."""
    text = CAIRN_CONTEXT_RE.sub("", text)
    text = SYSTEM_REMINDER_RE.sub("", text)
    text = LOCAL_COMMAND_RE.sub("", text)
    text = COMMAND_TAG_RE.sub("", text)
    return text.strip()


def _clean_assistant_text(text: str) -> str:
    """Strip thinking and memory blocks from assistant text."""
    text = THINKING_RE.sub("", text)
    text = MEMORY_BLOCK_RE.sub("", text)
    return text.strip()


def _parse_ts(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _looks_like_correction(text: str) -> bool:
    low = text.lower().lstrip()
    return any(low.startswith(m) for m in CORRECTION_MARKERS)


def load_turns(path: str) -> list[dict]:
    """Load JSONL into a turn list. Each turn:
        {idx, role, text, has_tools, tool_summary, timestamp}
    """
    turns = []
    idx = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = entry.get("type", "")
            if etype not in ("user", "human", "assistant"):
                continue
            msg = entry.get("message", {})
            content = msg.get("content", "")
            role = "user" if etype in ("user", "human") else "assistant"
            text = _content_text(content)
            if role == "user":
                text = _clean_user_text(text)
            else:
                text = _clean_assistant_text(text)
            turns.append({
                "idx": idx,
                "role": role,
                "text": text,
                "has_tools": _has_tool_blocks(content),
                "tool_summary": _tool_summary(content),
                "timestamp": entry.get("timestamp", ""),
            })
            idx += 1
    return turns


def filter_turns(turns, *, signal_only=True, corrections_only=False,
                 turn_range=None, last_minutes=None):
    out = []
    if turn_range is not None:
        lo, hi = turn_range
        turns = [t for t in turns if lo <= t["idx"] <= hi]
    if last_minutes is not None and turns:
        latest = None
        for t in reversed(turns):
            ts = _parse_ts(t["timestamp"])
            if ts is not None:
                latest = ts
                break
        if latest is not None:
            cutoff = latest - timedelta(minutes=last_minutes)
            turns = [
                t for t in turns
                if (_parse_ts(t["timestamp"]) or latest) >= cutoff
            ]
    for t in turns:
        # In signal_only mode, skip turns that ONLY carried tool blocks
        # (no actual text) — they're already represented by the
        # surrounding text turns.
        if signal_only and not t["text"]:
            continue
        if corrections_only:
            if t["role"] != "user" or not _looks_like_correction(t["text"]):
                continue
        out.append(t)
    return out


def render(turns, *, with_tools=False) -> str:
    lines = []
    for t in turns:
        header = f"## turn {t['idx']} · {t['role']}"
        if t["timestamp"]:
            header += f" · {t['timestamp']}"
        lines.append(header)
        if t["text"]:
            lines.append(t["text"])
        if with_tools and t["tool_summary"]:
            lines.append(t["tool_summary"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_range(s: str):
    if "-" not in s:
        raise ValueError(f"--turn-range expects A-B, got {s!r}")
    a, b = s.split("-", 1)
    return int(a), int(b)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("jsonl", help="Path to Claude Code session JSONL")
    p.add_argument("--signal-only", action="store_true", default=True,
                   help="(default) drop tool blocks, thinking, injected blocks")
    p.add_argument("--with-tools", action="store_true",
                   help="Include one-line tool summaries for redirect detection")
    p.add_argument("--corrections-only", action="store_true",
                   help="Keep only user turns that look like corrections")
    p.add_argument("--turn-range", type=parse_range, default=None,
                   help="Slice by turn index A-B (inclusive)")
    p.add_argument("--last-N-minutes", dest="last_minutes", type=int, default=None,
                   help="Slice to last N minutes relative to final turn")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON list of turns instead of rendered text")
    args = p.parse_args(argv)
    if args.with_tools:
        # --with-tools implies tools are kept; signal_only filter still drops
        # empty-text turns where nothing else is available.
        pass

    turns = load_turns(args.jsonl)
    turns = filter_turns(
        turns,
        signal_only=args.signal_only,
        corrections_only=args.corrections_only,
        turn_range=args.turn_range,
        last_minutes=args.last_minutes,
    )
    if args.json:
        json.dump(turns, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render(turns, with_tools=args.with_tools))
    return 0


if __name__ == "__main__":
    sys.exit(main())
