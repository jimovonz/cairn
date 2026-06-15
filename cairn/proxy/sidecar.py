"""Shared sidecar protocol between the cairn hooks and the cairn proxy.

The hooks (when ``CAIRN_PROXY_ENABLED``) stage context to files in
``.staged_context/`` instead of printing visible ``additionalContext``; the
proxy reads them and injects under the hood. Conversely the proxy captures the
stripped ``[cm]`` block / memory notes to a capture file the Stop hook reads.

Files (all keyed by Claude Code session id):
  <session>_inject_bootstrap.txt  — standing bootstrap (persistent, byte-stable)
  <session>_inject_prompt.txt     — volatile per-prompt context (append; consumed
                                    and truncated by the proxy each request)
  <session>_cm_capture.jsonl      — one JSON record per assistant turn:
                                    {"emitted_sha","cm","notes"} (proxy appends,
                                    Stop hook reads)
"""

from __future__ import annotations

import json
import os
from typing import Optional

# repo_root/.staged_context — matches hooks' own resolution.
_STAGED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".staged_context",
)


def staged_dir() -> str:
    os.makedirs(_STAGED_DIR, exist_ok=True)
    return _STAGED_DIR


def _path(session_id: str, suffix: str) -> str:
    safe = "".join(c for c in (session_id or "unknown") if c.isalnum() or c in "-_")
    return os.path.join(staged_dir(), f"{safe}{suffix}")


def bootstrap_path(session_id: str) -> str:
    return _path(session_id, "_inject_bootstrap.txt")


def prompt_inject_path(session_id: str) -> str:
    return _path(session_id, "_inject_prompt.txt")


def capture_path(session_id: str) -> str:
    return _path(session_id, "_cm_capture.jsonl")


# -- proxy write / hook read: capture -----------------------------------------
def append_capture(session_id: str, record: dict) -> None:
    if not record.get("cm") and not record.get("notes"):
        return
    with open(capture_path(session_id), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_cm_map(session_id: str) -> dict:
    """Map stripped-text SHA -> verbatim [cm] block, for cache re-injection."""
    out: dict = {}
    path = capture_path(session_id)
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("emitted_sha") and rec.get("cm"):
                    out[rec["emitted_sha"]] = rec["cm"]
    except FileNotFoundError:
        pass
    return out


def load_all_notes(session_id: str) -> list:
    """All captured memory-note strings for the session (proxy-stripped notes)."""
    notes: list = []
    path = capture_path(session_id)
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                notes.extend(rec.get("notes", []) or [])
    except FileNotFoundError:
        pass
    return notes


def lookup_capture_by_sha(session_id: str, emitted_sha: str) -> Optional[dict]:
    """Return the capture record whose stripped text matches emitted_sha."""
    path = capture_path(session_id)
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
    except FileNotFoundError:
        return None
    for line in reversed(lines):  # most recent first
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("emitted_sha") == emitted_sha:
            return rec
    return None


# -- hook write / proxy read: bootstrap + per-prompt context ------------------
def write_bootstrap(session_id: str, text: str) -> None:
    if not text:
        return
    with open(bootstrap_path(session_id), "w", encoding="utf-8") as fh:
        fh.write(text)


def read_bootstrap(session_id: str) -> str:
    try:
        with open(bootstrap_path(session_id), encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def append_prompt_context(session_id: str, text: str) -> None:
    if not text:
        return
    with open(prompt_inject_path(session_id), "a", encoding="utf-8") as fh:
        fh.write(text.rstrip("\n") + "\n")


def consume_prompt_context(session_id: str) -> str:
    """Read and clear the pending per-prompt context (atomic-ish: read+unlink)."""
    path = prompt_inject_path(session_id)
    try:
        with open(path, encoding="utf-8") as fh:
            data = fh.read()
    except FileNotFoundError:
        return ""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return data.strip()
