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

Concurrency: the proxy (writer) and the Stop/prompt hooks (readers/consumers)
touch these files from separate processes. All access is serialized with
``fcntl.flock`` (LOCK_EX for append/consume, LOCK_SH for reads) so a reader
never observes a torn append and ``consume_prompt_context`` can't drop an
append racing between its read and clear.
"""

from __future__ import annotations

import fcntl
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
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(record) + "\n")


def load_cm_map(session_id: str) -> dict:
    """Map stripped-text SHA -> verbatim [cm] block, for cache re-injection."""
    out: dict = {}
    path = capture_path(session_id)
    try:
        with open(path, encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
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
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
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
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
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
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(text)


def read_bootstrap(session_id: str) -> str:
    try:
        with open(bootstrap_path(session_id), encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            return fh.read()
    except FileNotFoundError:
        return ""


def append_prompt_context(session_id: str, text: str) -> None:
    if not text:
        return
    with open(prompt_inject_path(session_id), "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(text.rstrip("\n") + "\n")


def consume_prompt_context(session_id: str) -> str:
    """Read and clear the pending per-prompt context atomically.

    Opens r+ under an exclusive lock, reads, then truncates in place — so an
    ``append_prompt_context`` (also LOCK_EX) cannot interleave between the read
    and the clear and have its write silently discarded.
    """
    path = prompt_inject_path(session_id)
    try:
        with open(path, "r+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            data = fh.read()
            fh.seek(0)
            fh.truncate()
    except FileNotFoundError:
        return ""
    return data.strip()


# -- proxy-internal: per-session cache-prefix integrity state ------------------
def prefix_state_path(session_id: str) -> str:
    return _path(session_id, "_prefix_state.json")


def read_prefix_state(session_id: str) -> tuple:
    """Return (bootstrap_sha, unstable) recorded for the session, else (None, False).

    Used by the proxy to detect when the cached system prefix (the injected
    bootstrap) changes mid-session — a silent prompt-cache re-bill.
    """
    try:
        with open(prefix_state_path(session_id), encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            d = json.load(fh)
        return d.get("sha"), bool(d.get("unstable"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, False


def write_prefix_state(session_id: str, sha: str, unstable: bool) -> None:
    with open(prefix_state_path(session_id), "w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        json.dump({"sha": sha, "unstable": unstable}, fh)


# -- proxy write / reporting read: context-paring savings (Phase 1.5) ----------
# <session>_pare_stats.json — cumulative token-instances removed by paring, so
# the otherwise-invisible benefit is measurable and the digest-bloat regression
# (net savings shrinking as the topic digest grows) is observable. Chars, not
# tokens: the proxy hot path must not run a tokeniser — convert at report time.
def pare_stats_path(session_id: str) -> str:
    return _path(session_id, "_pare_stats.json")


def record_pare_savings(session_id: str, stats: dict) -> None:
    """Fold one request's paring deltas into the per-session cumulative totals.

    ``stats`` carries this request's ``blocks_replaced_chars`` (verbatim [cm]
    lengths that would have been reinjected), ``marker_chars`` and
    ``digest_chars`` (the costs paring adds back). Net saved for the request is
    blocks_replaced − markers − digest. Best-effort: any failure is swallowed so
    a stats write never breaks a request (fail-open, like the rest of the proxy).
    """
    try:
        blocks = int(stats.get("blocks_replaced_chars", 0) or 0)
        markers = int(stats.get("marker_chars", 0) or 0)
        digest = int(stats.get("digest_chars", 0) or 0)
        if blocks == 0 and markers == 0 and digest == 0:
            return
        net = blocks - markers - digest
        path = pare_stats_path(session_id)
        with open(path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            raw = fh.read().strip()
            cur = json.loads(raw) if raw else {}
            cur["requests"] = int(cur.get("requests", 0)) + 1
            cur["blocks_replaced_chars"] = int(cur.get("blocks_replaced_chars", 0)) + blocks
            cur["marker_chars"] = int(cur.get("marker_chars", 0)) + markers
            cur["digest_chars"] = int(cur.get("digest_chars", 0)) + digest
            cur["net_saved_chars"] = int(cur.get("net_saved_chars", 0)) + net
            # Track the largest single-request digest seen — the bloat signal.
            cur["max_digest_chars"] = max(int(cur.get("max_digest_chars", 0)), digest)
            fh.seek(0)
            fh.truncate()
            json.dump(cur, fh)
    except Exception:
        return


def load_pare_stats(session_id: str) -> Optional[dict]:
    try:
        with open(pare_stats_path(session_id), encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            raw = fh.read().strip()
        return json.loads(raw) if raw else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_all_pare_stats() -> list:
    """Every session's pare-stats record, each tagged with its session id."""
    out = []
    try:
        for name in os.listdir(staged_dir()):
            if name.endswith("_pare_stats.json"):
                sid = name[: -len("_pare_stats.json")]
                rec = load_pare_stats(sid)
                if rec:
                    rec = dict(rec, session=sid)
                    out.append(rec)
    except FileNotFoundError:
        pass
    return out
