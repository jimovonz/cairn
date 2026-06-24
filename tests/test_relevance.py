"""Read-side relevance grading — Phase 1/2 core (cairn/relevance.py)."""
from __future__ import annotations

import json
import sqlite3

import pytest

from cairn import relevance
from cairn import init_db


# ---- build_context_window ----------------------------------------------------
def _write_transcript(path, turns):
    with open(path, "w") as f:
        for role, text in turns:
            f.write(json.dumps({"type": role, "message": {"content": text}}) + "\n")


def test_context_window_includes_prior_turn_and_current(tmp_path):
    t = tmp_path / "s.jsonl"
    _write_transcript(t, [
        ("user", "how does pull_all gate peers?"),
        ("assistant", "It reads eph.discovered_peers and skips stale beacons."),
    ])
    win = relevance.build_context_window("now add a test for that", str(t))
    assert "[user] now add a test for that" in win
    assert "[prev user] how does pull_all gate peers?" in win
    assert "[prev assistant] It reads eph.discovered_peers" in win


def test_context_window_caps_long_prior_response(tmp_path):
    t = tmp_path / "s.jsonl"
    long_resp = "X" * 5000
    _write_transcript(t, [("user", "q"), ("assistant", long_resp)])
    win = relevance.build_context_window("follow up", str(t), prior_response_cap=100)
    # The capped prior response must not let the long response dominate.
    assert "X" * 100 in win
    assert "X" * 200 not in win
    assert win.endswith("[user] follow up")


def test_context_window_dedupes_current_prompt_if_already_tail(tmp_path):
    t = tmp_path / "s.jsonl"
    _write_transcript(t, [
        ("user", "first question"),
        ("assistant", "an answer"),
        ("user", "the current prompt"),  # current prompt already written to transcript
    ])
    win = relevance.build_context_window("the current prompt", str(t))
    # prior-user referent must be the turn BEFORE the current prompt, not itself
    assert "[prev user] first question" in win
    assert win.count("the current prompt") == 1


def test_context_window_failsoft_no_transcript():
    assert relevance.build_context_window("just the prompt", None) == "[user] just the prompt"


# ---- bucket-4 self-referential meta prefilter --------------------------------
@pytest.mark.parametrize("content", [
    "Cairn contains a profile of James the surveyor",
    "No memory of the user's brother exists yet",
    "This should be captured when the user shares it",
    "cairn has limited info about the deployment",
])
def test_self_referential_meta_positive(content):
    assert relevance.is_self_referential_meta({"content": content}) is True


@pytest.mark.parametrize("content", [
    "Cairn uses ms-marco-MiniLM-L-6-v2 as the rerank cross-encoder",
    "pull_all presence-gates on eph.discovered_peers; stale beacons are skipped",
    "The user prefers feature branches over committing to main",
])
def test_self_referential_meta_negative(content):
    # Legit domain memories that merely mention cairn must NOT be flagged.
    assert relevance.is_self_referential_meta({"content": content}) is False


# ---- delivery log + grade write-back round-trip ------------------------------
@pytest.fixture
def eph(tmp_path):
    p = str(tmp_path / "eph.db")
    init_db.init_ephemeral(p)
    return p


def test_log_and_grade_roundtrip(eph):
    delivered = [
        {"id": 42, "score": 0.8, "ce_score": 1.2},
        {"id": 17, "score": 0.5},
    ]
    n = relevance.log_memory_deliveries(
        delivered, session_id="sess1", context_text="[user] hi",
        context_vec=b"\x00\x01", eph_path=eph)
    assert n == 2

    conn = sqlite3.connect(eph)
    rows = conn.execute(
        "SELECT memory_id, context_text, ce_score, served_rank, grade, hard_negative "
        "FROM memory_deliveries WHERE session_id='sess1' ORDER BY served_rank").fetchall()
    assert rows[0] == (42, "[user] hi", 1.2, 0, None, 0)   # ce_score preferred over score
    assert rows[1][0] == 17 and rows[1][2] == 0.5          # falls back to score

    grades = relevance.parse_relevance_grades(["42:3", "17:0!"])
    assert grades == [(42, 3, False), (17, 0, True)]
    upd = relevance.apply_relevance_grades(grades, session_id="sess1", eph_path=eph)
    assert upd == 2
    g = dict(conn.execute(
        "SELECT memory_id, grade FROM memory_deliveries WHERE session_id='sess1'").fetchall())
    hn = dict(conn.execute(
        "SELECT memory_id, hard_negative FROM memory_deliveries WHERE session_id='sess1'").fetchall())
    assert g[42] == 3 and g[17] == 0
    assert hn[42] == 0 and hn[17] == 1


def test_grade_updates_most_recent_delivery_only(eph):
    # Same memory delivered across two turns; a grade applies to the latest only.
    relevance.log_memory_deliveries([{"id": 7, "score": 0.9}], session_id="s",
                                    context_text="t0", turn_index=0, eph_path=eph)
    relevance.log_memory_deliveries([{"id": 7, "score": 0.9}], session_id="s",
                                    context_text="t1", turn_index=1, eph_path=eph)
    relevance.apply_relevance_grades([(7, 2, False)], session_id="s", eph_path=eph)
    conn = sqlite3.connect(eph)
    graded = conn.execute(
        "SELECT context_text, grade FROM memory_deliveries WHERE memory_id=7 ORDER BY turn_index"
    ).fetchall()
    assert graded == [("t0", None), ("t1", 2)]


def test_log_failsoft_empty():
    assert relevance.log_memory_deliveries([], session_id="s") == 0
    assert relevance.apply_relevance_grades([], session_id="s") == 0
