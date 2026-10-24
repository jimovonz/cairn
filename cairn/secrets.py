"""Secrets: keep values out of Cairn, store pointers instead.

A memory never holds a credential. The value lives in a mode-600 env file
(SECRETS_FILE, default ~/.config/cairn/secrets.env); the memory holds
``$NAME`` plus where to find it, so retrieval still tells the agent what the
secret is for and how to use it (``source <file>`` then ``$NAME``), while the
value stays out of the DB, its FTS index, injections, sync, backups and any
external judge.

- ``extract`` runs on every memory write (hooks/storage.insert_memories) and
  on source excerpts: detected values move to the store and are replaced by
  ``$NAME``.
- ``protect`` scans an existing DB (every text column of every table) and
  applies the same replacement, reusing names for values already stored, so
  history, excerpts and logs are cleaned consistently. Reversible: the store
  holds every value.
- CLI: ``python -m cairn.secrets {list|set NAME|get NAME|scan [DB]|protect [DB...]}``.

Detection is deliberately narrower than outbound redaction: known key
prefixes, key=value pairs, long mixed-case alphanumeric tokens, and long
tokens labelled as a key/token/secret/password just before them. UUIDs, hex
hashes and slugs are not secrets. An unlabelled all-hex key cannot be told
apart from a hash and is missed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

# Prefer pysqlite3 (SQLite 3.51.1) — every other durable-cairn.db writer uses it;
# a stdlib-vs-pysqlite3 version mix on the same WAL file is the documented cause
# of page-1 corruption (be91366); keep this writer on the same library.
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    if os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "set CAIRN_ALLOW_STDLIB_SQLITE=1 to override.") from _pysqlite_err

SECRETS_FILE = os.environ.get("CAIRN_SECRETS_FILE",
                              os.path.expanduser("~/.config/cairn/secrets.env"))

_KNOWN_PREFIX = re.compile(
    r"\b(AIza[0-9A-Za-z_-]{30,}|AKIA[0-9A-Z]{16}|(?:sk|pk|rk)[-_](?:live|test|proj|ant)?[-_]?[A-Za-z0-9_-]{16,}"
    r"|gh[opsu]_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{16,}|xox[bap]-[A-Za-z0-9-]{10,})")
_KV = re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|client[_-]?secret)\b\s*[:=]\s*[\"']?"
                 r"(?P<v>[^\s\"',;)\]}]{6,})")
_LONG = re.compile(r"(?<![\w$-])[A-Za-z0-9_]{24,}(?![\w-])")
_LABEL = re.compile(r"(?i)\b(key|token|secret|password|passwd|credential|api[_-]?key|subscription[_-]?key)\b[^\n]{0,24}$")
_NAME_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "is", "was", "with", "via", "for", "use", "using", "my", "your",
         "header", "value", "base", "live", "verified", "saved", "obtained", "new", "this", "that",
         "here", "superseded", "model", "current", "old", "is", "our"}


def find_secrets(text: str) -> list[tuple[int, int, str]]:
    """(start, end, value) spans of secret values in text, non-overlapping."""
    if not text:
        return []
    spans = []
    for m in _KNOWN_PREFIX.finditer(text):
        spans.append((m.start(), m.end()))
    for m in _KV.finditer(text):
        v = m.group("v")
        if not v.startswith("$"):   # already a pointer
            spans.append((m.start("v"), m.end("v")))
    for m in _LONG.finditer(text):
        tok = m.group(0)
        mixed = bool(re.search(r"[a-z]", tok) and re.search(r"[A-Z]", tok) and re.search(r"\d", tok))
        # Underscore-joined words (ROBOTIS_U2D2_Power_Hub_Board) are identifiers,
        # not keys: three or more segments that each read as a word or model code.
        parts = tok.split("_")
        if len(parts) >= 3 and all(re.fullmatch(r"[A-Za-z]{2,}|[A-Za-z]+\d{0,3}[A-Za-z]*\d{0,3}", p) for p in parts if p):
            mixed = labelled = False
        labelled = bool(_LABEL.search(text[max(0, m.start() - 40):m.start()])) and bool(re.search(r"\d", tok))
        if (mixed and len(tok) >= 28) or labelled:
            spans.append((m.start(), m.end()))
    spans.sort()
    out, last = [], -1
    for s, e in spans:
        if s >= last and not _not_secret(text, s, e):
            out.append((s, e, text[s:e]))
            last = e
    return out


_PLACEHOLDER = re.compile(r"(?i)your|example|placeholder|xxxx|redacted|changeme|<|>")
_PUBLIC_CTX = re.compile(r"(?i)(public|ssh-(rsa|ed25519|ecdsa)|fingerprint|gpg|pgp|filter|commit|sha\d*|hash|checksum|\bid\b)[^\n]{0,30}$")


def _not_secret(text: str, s: int, e: int) -> bool:
    """Shapes that look key-like but are not credentials (found by scanning
    the corpus): placeholders, env-var names, paths and file names, public
    keys and fingerprints, ids."""
    v = text[s:e]
    if _PLACEHOLDER.search(v) or re.fullmatch(r"[A-Z][A-Z0-9_]*", v) or v.startswith("$"):
        return True
    if v.startswith(("/", "~", "./")) or "/" in v or re.match(r"\.[A-Za-z]{2,4}\b", text[e:e + 5]):
        return True
    if s > 0 and text[s - 1] in "/\\.":
        return True
    if re.match(r"(req|file|msg|toolu|call|chatcmpl|gen|run|batch|org|proj|asst|thread)[_-]", v):
        return True   # API object / request ids
    if "\x1b[" in text[max(0, s - 6):s] or re.match(r"\d*m[A-Z]", v) and "\x1b" in text[max(0, s - 4):s]:
        return True   # terminal colour codes before a class name
    if v.startswith("AAAA") or _PUBLIC_CTX.search(text[max(0, s - 40):s]):
        return True
    return False


def has_secret(text: str) -> bool:
    return bool(find_secrets(text))


# --- store ------------------------------------------------------------------

def _load_store(path: str = None) -> dict[str, str]:
    path = path or SECRETS_FILE
    out = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    except FileNotFoundError:
        pass
    return out


def _write_store(store: dict[str, str], path: str = None) -> None:
    path = path or SECRETS_FILE
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("# Cairn secret store. Values referenced from memories as $NAME.\n"
                "# Use: set -a; . ~/.config/cairn/secrets.env; set +a\n")
        for k in sorted(store):
            f.write(f"{k}='{store[k]}'\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _name_for(text: str, start: int, topic: str, value: str, store: dict[str, str]) -> str:
    for k, v in store.items():
        if v == value:
            return k
    before = text[max(0, start - 48):start]
    words = [w for w in _NAME_WORDS.findall(before) if w.lower() not in _STOP][-3:]
    kind = next((w for w in reversed(words) if w.lower() in ("key", "token", "secret", "password", "passwd")), None)
    label = [w for w in words if w != kind][-2:]
    base = "_".join(label or _NAME_WORDS.findall(topic or "")[:3] or ["CAIRN"]).upper()
    name = re.sub(r"[^A-Z0-9_]", "_", f"{base}_{(kind or 'SECRET').upper()}").strip("_")
    if not re.match(r"[A-Z_]", name):
        name = "S_" + name
    cand, n = name, 2
    while cand in store and store[cand] != value:
        cand, n = f"{name}_{n}", n + 1
    return cand


def extract(text: Optional[str], topic: str = "", store: Optional[dict] = None,
            persist: bool = True) -> tuple[Optional[str], list[str]]:
    """Move secret values in text into the store; return (text with $NAME
    pointers, names used). No secret -> text unchanged."""
    if not text:
        return text, []
    spans = find_secrets(text)
    if not spans:
        return text, []
    own = store is None
    store = _load_store() if own else store
    names, out, last = [], [], 0
    for s, e, value in spans:
        name = _name_for(text, s, topic, value, store)
        store[name] = value
        names.append(name)
        out.append(text[last:s] + f"${name}")
        last = e
    out.append(text[last:])
    new = "".join(out)
    if "secrets.env" not in new:
        new += f" [secret values: {SECRETS_FILE.replace(os.path.expanduser('~'), '~')}]"
    if own and persist:
        _write_store(store)
    return new, names


def extract_entry(entry: dict) -> list[str]:
    """In-place on a memory entry dict (content, topic, facts). Returns names."""
    names = []
    store = _load_store()
    topic = str(entry.get("topic") or "")
    for field in ("content", "topic"):
        if isinstance(entry.get(field), str):
            entry[field], n = extract(entry[field], topic, store=store, persist=False)
            names += n
    if isinstance(entry.get("facts"), list):
        new = []
        for fct in entry["facts"]:
            t, n = extract(fct, topic, store=store, persist=False) if isinstance(fct, str) else (fct, [])
            new.append(t)
            names += n
        entry["facts"] = new
    if names:
        _write_store(store)
    return names


# --- corpus scan / protect --------------------------------------------------

def _text_columns(conn) -> list[tuple[str, str]]:
    cols = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        if t.startswith(("sqlite_", "memories_fts", "memories_vec")):
            continue   # FTS/vec shadow tables are maintained by triggers
        for row in conn.execute(f'PRAGMA table_info("{t}")'):
            if (row[2] or "").upper() in ("TEXT", "", "VARCHAR", "JSON"):
                cols.append((t, row[1]))
    return cols


def scan(db_path: str) -> list[dict]:
    """Every (table, column, rowid) holding a detected secret. Values masked."""
    conn = sqlite3.connect(db_path)
    hits = []
    try:
        for t, c in _text_columns(conn):
            try:
                rows = conn.execute(f'SELECT rowid, "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL').fetchall()
            except sqlite3.Error:
                continue
            for rid, val in rows:
                if isinstance(val, str):
                    for s, e, v in find_secrets(val):
                        hits.append({"table": t, "column": c, "rowid": rid,
                                     "masked": v[:4] + "…" + v[-2:], "context": val[max(0, s - 40):s].replace("\n", " ")})
    finally:
        conn.close()
    return hits


def protect(db_paths: list[str]) -> dict:
    """Replace every detected secret value with its $NAME pointer in every text
    column of each DB, storing values first. memories.content updates fire the
    history trigger, so memory_history is scrubbed after the memories pass."""
    store = _load_store()
    report = {}
    for path in db_paths:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA busy_timeout=10000")
        changed = 0
        try:
            cols = _text_columns(conn)
            order = sorted(cols, key=lambda tc: (tc[0] == "memory_history", tc))   # history last
            for t, c in order:
                try:
                    rows = conn.execute(f'SELECT rowid, "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL').fetchall()
                except sqlite3.Error:
                    continue
                for rid, val in rows:
                    if not isinstance(val, str) or not find_secrets(val):
                        continue
                    topic = ""
                    if t == "memories":
                        topic = conn.execute("SELECT topic FROM memories WHERE rowid=?", (rid,)).fetchone()[0] or ""
                    new, _ = extract(val, topic, store=store, persist=False)
                    if new != val:
                        _write_store(store)   # value is safe before the DB loses it
                        conn.execute(f'UPDATE "{t}" SET "{c}" = ? WHERE rowid = ?', (new, rid))
                        changed += 1
            conn.commit()
            # Values already stored but sitting undetected elsewhere (e.g. an
            # unlabelled hex key also quoted beside its label): replace literally.
            for t, c in order:
                for name, value in store.items():
                    if len(value) < 12:
                        continue
                    try:
                        cur = conn.execute(f'UPDATE "{t}" SET "{c}" = replace("{c}", ?, ?) WHERE instr("{c}", ?) > 0',
                                           (value, f"${name}", value))
                        changed += cur.rowcount
                    except sqlite3.Error:
                        pass
            conn.commit()
        finally:
            conn.close()
        report[path] = changed
    return report


def rename(old: str, new: str, db_paths: list[str]) -> dict:
    """Rename a stored secret and every $OLD pointer in the given DBs."""
    store = _load_store()
    if old not in store or new in store:
        raise SystemExit(f"cannot rename {old} -> {new}")
    store[new] = store.pop(old)
    _write_store(store)
    pat = re.compile(r"\$" + re.escape(old) + r"\b")
    report = {}
    for path in db_paths:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA busy_timeout=10000")
        n = 0
        try:
            for t, c in _text_columns(conn):
                try:
                    rows = conn.execute(f'SELECT rowid, "{c}" FROM "{t}" WHERE instr("{c}", ?) > 0',
                                        ("$" + old,)).fetchall()
                except sqlite3.Error:
                    continue
                for rid, val in rows:
                    new_val = pat.sub("$" + new, val)
                    if new_val != val:
                        conn.execute(f'UPDATE "{t}" SET "{c}" = ? WHERE rowid = ?', (new_val, rid))
                        n += 1
            conn.commit()
        finally:
            conn.close()
        report[path] = n
    return report


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "list"
    default_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db")
    if cmd == "list":
        for k in sorted(_load_store()):
            print(k)
    elif cmd == "set" and len(argv) == 2:
        store = _load_store()
        store[argv[1]] = sys.stdin.readline().strip()
        _write_store(store)
        print(f"stored ${argv[1]} in {SECRETS_FILE}")
    elif cmd == "get" and len(argv) == 2:
        print("warning: printing a secret puts it in the transcript; prefer sourcing the file", file=sys.stderr)
        print(_load_store().get(argv[1], ""))
    elif cmd == "scan":
        hits = scan(argv[1] if len(argv) > 1 else default_db)
        print(json.dumps(hits, indent=1))
        print(f"{len(hits)} secret occurrence(s)", file=sys.stderr)
    elif cmd == "rename" and len(argv) >= 3:
        print(json.dumps(rename(argv[1], argv[2], argv[3:] or [default_db]), indent=1))
    elif cmd == "protect":
        print(json.dumps(protect(argv[1:] or [default_db]), indent=1))
    else:
        print("usage: python -m cairn.secrets {list|set NAME|get NAME|scan [DB]|protect [DB...]|rename OLD NEW [DB...]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
