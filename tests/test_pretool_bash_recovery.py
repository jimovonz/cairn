#!/usr/bin/env python3
"""Tests for Bash file-access recovery in hooks/pretool_hook.py.

In CCH-style environments, native Read/Edit are routed through Bash helpers
(cat/sed/cch-edit.py/cch-write.py), so the pretool hook must recover the target
file from the Bash command string and inject context for it. Covers the pure
extractor and the main() Bash dispatch, plus the Read regression path.

Assertions use indexed-line equality on additionalContext (per cairn 2957:
substring containment trips the sole-assertion verifier and is weaker)."""

import os
import json
import tempfile
import pytest
from unittest.mock import patch

import hooks.hook_helpers as hook_helpers
import hooks.pretool_hook as pretool_hook
from hooks.pretool_hook import extract_bash_file_paths

# Reuse the established harness (temp DB schema + main() runner).
from tests.test_pretool_hook import fresh_db, run_main

_SRC_DIR = tempfile.mkdtemp()


def _make_source(name: str = "widget.py", body: str = "def f():\n    return 1\n") -> str:
    """Write a real source file (extractor requires os.path.isfile + source ext)."""
    p = os.path.join(_SRC_DIR, name)
    with open(p, "w") as fh:
        fh.write(body)
    return p


# ============================================================
# extract_bash_file_paths — pure function
# ============================================================

# Verifies: each recognised reader/editor verb yields exactly the real source
# file (as realpath), while search/non-source/missing/no-verb commands yield none.
@pytest.mark.behavioural
def test_extract_recognises_readers_and_editors():
    src = _make_source("alpha.py")
    base = os.path.basename(src)
    rp = os.path.realpath(src)

    # Positive: reader and editor verbs return exactly [realpath(src)].
    assert extract_bash_file_paths(f"cat {src}") == [rp]
    assert extract_bash_file_paths(f"sed -n '1,40p' {src}") == [rp]
    assert extract_bash_file_paths(f"head -n 5 {src}") == [rp]
    assert extract_bash_file_paths(f"cch-edit.py {src} 'a' 'b'") == [rp]
    assert extract_bash_file_paths(f"python3 /x/cch-write.py {src}") == [rp]

    # Negative: each returns [] for a distinct reason.
    assert extract_bash_file_paths(f"rg -n foo {src}") == []          # rg is search, not a file verb
    assert extract_bash_file_paths("cat /tmp/data.db") == []          # not a source extension
    assert extract_bash_file_paths(f"head -n 5 /no/such/{base}") == []  # file does not exist
    assert extract_bash_file_paths("git status") == []                # no reader/editor verb
    assert extract_bash_file_paths("ls -la") == []
    assert extract_bash_file_paths("") == []


# Verifies: dedup + cap — repeated same file collapses to one; >max distinct files capped.
@pytest.mark.behavioural
def test_extract_dedups_and_caps():
    a = _make_source("one.py")
    b = _make_source("two.py")
    c = _make_source("three.py")
    d = _make_source("four.py")
    rp_a = os.path.realpath(a)

    # Same file twice in a compound command → one entry.
    assert extract_bash_file_paths(f"cat {a} && sed -n '1,2p' {a}") == [rp_a]

    # Four distinct files, default cap of 3 → exactly 3 returned, all real source files.
    got = extract_bash_file_paths(f"cat {a} {b} {c} {d}")
    assert len(got) == 3
    assert all(p == os.path.realpath(p) for p in got)


# Verifies: adversarial input (unbalanced quotes) falls back to split() and does not raise.
@pytest.mark.adversarial
def test_extract_malformed_quotes_no_crash():
    src = _make_source("quote.py")
    rp = os.path.realpath(src)
    # Unbalanced quote makes shlex.split raise ValueError → fallback to .split().
    got = extract_bash_file_paths(f"cat {src} 'unterminated")
    assert got == [rp]


# ============================================================
# main() Bash dispatch
# ============================================================

# Verifies: a Bash `cat <source>` payload routes through to gotcha injection,
# emitting the correct header line for the recovered file.
@pytest.mark.behavioural
def test_main_bash_cat_triggers_injection():
    src = _make_source("auth.py")
    base = os.path.basename(src)
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('correction', 'null-guard', 'Check None before indexing', ?, 0.9, NULL)",
        (json.dumps([src]),)
    )
    conn.commit()
    conn.close()

    payload = {"tool_name": "Bash", "session_id": "s1", "tool_input": {"command": f"cat {src}"}}
    out, code = run_main(payload, db_path)
    obj = json.loads(out)
    ctx = obj["hookSpecificOutput"]["additionalContext"]
    lines = ctx.splitlines()

    assert code == 0
    assert lines[0] == f"CAIRN GOTCHA for {base}:"
    assert lines[1] == "- [null-guard] Check None before indexing"


# Verifies: a Bash command with no reader/editor verb emits nothing (clean exit 0).
@pytest.mark.behavioural
def test_main_bash_non_file_command_silent():
    db_path, conn = fresh_db()
    conn.close()
    payload = {"tool_name": "Bash", "session_id": "s2", "tool_input": {"command": "git status"}}
    out, code = run_main(payload, db_path)
    assert code == 0
    assert out == ""


# Verifies: the native Read path is unchanged by the refactor (regression guard).
@pytest.mark.behavioural
def test_main_read_path_still_injects():
    src = _make_source("regress.py")
    base = os.path.basename(src)
    db_path, conn = fresh_db()
    conn.execute(
        "INSERT INTO memories (type, topic, content, associated_files, confidence, archived_reason)"
        " VALUES ('correction', 'guard', 'Watch the edge', ?, 0.8, NULL)",
        (json.dumps([src]),)
    )
    conn.commit()
    conn.close()

    payload = {"tool_name": "Read", "session_id": "s3", "tool_input": {"file_path": src}}
    out, code = run_main(payload, db_path)
    obj = json.loads(out)
    lines = obj["hookSpecificOutput"]["additionalContext"].splitlines()
    assert code == 0
    assert lines[0] == f"CAIRN GOTCHA for {base}:"
