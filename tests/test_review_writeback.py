"""Tests for cairn.review_writeback — the code-attached review-finding capture.

Covers the pure mapping logic (_build_entry) and the dry-run path, which need no
database, so they exercise the keying contract (file + symbol surfaced for both
the associated_files LIKE join and the FTS MATCH in cairn-graph --knowledge)
without DB-write flakiness.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cairn import review_writeback as rw


# Verifies: a finding maps to a memory entry carrying BOTH absolute and relative
# file paths (so the substring LIKE join hits) and the symbol in keywords+facts
# (so the FTS MATCH hits), plus line/pr/severity provenance facts.
@pytest.mark.behavioural
def test_build_entry_keys_file_and_symbol(tmp_path):
    repo = str(tmp_path)
    finding = {"file": "src/foo.py", "symbol": "do_thing", "line": 42,
               "type": "correction", "severity": "high", "pr": 6,
               "content": "predict() hands back a dangling reference after the buffer is freed."}
    e = rw._build_entry(finding, repo)
    assert e["type"] == "correction"
    assert os.path.join(repo, "src/foo.py") in e["associated_files"]
    assert "src/foo.py" in e["associated_files"]
    assert "do_thing" in e["keywords"]
    assert "review-finding" in e["keywords"]
    assert "symbol:do_thing" in e["facts"]
    assert "file:src/foo.py" in e["facts"]
    assert "line:42" in e["facts"]
    assert "pr:6" in e["facts"]
    assert "severity:high" in e["facts"]


# Verifies: when the symbol isn't already in the content, it is appended so the
# FTS MATCH on the symbol name still surfaces the finding.
@pytest.mark.behavioural
def test_build_entry_injects_symbol_into_content(tmp_path):
    e = rw._build_entry({"file": "a/b.py", "symbol": "missing_sym",
                         "content": "Something is wrong here."}, str(tmp_path))
    assert "missing_sym" in e["content"]


# Verifies: file and content are required; malformed findings raise ValueError
# (collected, not crashed, by write_back).
@pytest.mark.edge
def test_build_entry_requires_file_and_content(tmp_path):
    with pytest.raises(ValueError):
        rw._build_entry({"content": "no file here"}, str(tmp_path))
    with pytest.raises(ValueError):
        rw._build_entry({"file": "a.py"}, str(tmp_path))


# Verifies: type defaults to correction, an invalid type falls back to
# correction, and topic is derived when absent.
@pytest.mark.edge
def test_build_entry_type_and_topic_defaults(tmp_path):
    # Default is decision: the intended content is durable rationale, not a bug.
    e = rw._build_entry({"file": "a.py", "content": "x" * 40}, str(tmp_path))
    assert e["type"] == "decision"
    assert e["topic"].startswith("review:")
    # An invalid type falls back to correction (safe default for unknown input).
    e2 = rw._build_entry({"file": "a.py", "content": "x" * 40, "type": "bogus"}, str(tmp_path))
    assert e2["type"] == "correction"
    e3 = rw._build_entry({"file": "a.py", "content": "x" * 40, "type": "decision"}, str(tmp_path))
    assert e3["type"] == "decision"


# Verifies: dry-run reports would_insert / session id without touching the DB,
# and the commit is stamped into the synthetic session id.
@pytest.mark.behavioural
def test_write_back_dry_run(tmp_path):
    payload = {"repo": str(tmp_path), "commit": "abc123def456789",
               "findings": [{"file": "x.py", "symbol": "s", "content": "a finding about x"}]}
    result = rw.write_back(payload, dry_run=True)
    assert result["would_insert"] == 1
    assert result["session"].startswith("review-")
    assert "abc123def456" in result["session"]
    assert result["errors"] == []


# Verifies: malformed findings are collected into errors/skipped rather than
# aborting the whole batch.
@pytest.mark.adversarial
def test_write_back_collects_malformed(tmp_path):
    payload = {"repo": str(tmp_path), "findings": [
        {"file": "ok.py", "content": "a perfectly fine finding"},
        {"symbol": "no_file", "content": "missing the required file"},
    ]}
    result = rw.write_back(payload, dry_run=True)
    assert result["would_insert"] == 1
    assert result["skipped"] == 1
    assert len(result["errors"]) == 1


# Verifies: empty findings list is a clean no-op, not a crash.
@pytest.mark.edge
def test_write_back_no_findings(tmp_path):
    result = rw.write_back({"repo": str(tmp_path), "findings": []}, dry_run=True)
    assert result["inserted"] == 0
