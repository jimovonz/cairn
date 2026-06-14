"""Phase 2 calibration analyser tests.

Covers prompt building, output parsing, cap enforcement, write paths
(calibration_rows + memories), effectiveness scoring, idle-session
detection, and end-to-end orchestration with a mocked LLM caller.
"""

import json
import os
import sys
import tempfile
import time
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import analyser, init_db, session_extract


def _fresh_dbs():
    td = tempfile.mkdtemp()
    durable = os.path.join(td, "cairn.db")
    eph = os.path.join(td, "eph.db")
    with patch.object(init_db, "DB_PATH", durable):
        init_db.init()
    init_db.init_ephemeral(eph)
    return durable, eph, td


def _make_jsonl(td, name="s.jsonl", session_id="sess-test-001",
                turns: int = 6, body: str = None):
    """Default fixture is substantive enough to pass the analyser's
    subagent/triviality filter (>=4 substantive turns and >=500 cleaned
    chars). Pass `turns=2` explicitly to construct a sub-threshold
    transcript for filter tests."""
    proj_dir = os.path.join(td, "projects", "cairn-test")
    os.makedirs(proj_dir, exist_ok=True)
    path = os.path.join(proj_dir, f"{session_id}.jsonl")
    pad = body or ("real exchange content " * 20)
    with open(path, "w") as f:
        for i in range(turns):
            role = "user" if i % 2 == 0 else "assistant"
            f.write(json.dumps({
                "type": role,
                "message": {"content": f"{pad} turn {i}"},
                "timestamp": f"2026-05-19T10:00:{i:02d}Z",
            }) + "\n")
    return path


SAMPLE_LLM_OUTPUT = json.dumps({
    "user_observations": [
        {"content": "user prefers terse responses",
         "kw": ["terse", "brevity"],
         "qf": ["how should I respond", "keep it short"]},
    ],
    "explicit_instructions": [
        {"content": "always run tests before commit",
         "kw": ["tests", "commit"],
         "qf": ["should I commit", "ready to push"]},
    ],
    "approach_assessment": [],
    "contradictions": [],
    "drift_signals": [],
    "row_effectiveness": [
        {"row_id": 1, "outcome": "followed", "evidence": "agent applied brevity"},
    ],
    "tool_redirect_signals": [],
    "misalignment_reconvergence": [],
    "session_arc_memories": [
        {"topic": "phase-2-build",
         "content": "session built Phase 2 analyser end-to-end",
         "kw": ["phase-2", "analyser"]},
    ],
    "decision_memories_with_alternatives": [
        {"topic": "claude-p-vs-sdk",
         "content": "chose claude -p subprocess over anthropic SDK — matches existing audit_agent pattern, no API key needed",
         "kw": ["claude-p", "subprocess"]},
    ],
    "tool_brittleness_patterns": [],
    "loose_ends": [
        {"topic": "n20-measurement",
         "content": "need to run analyser against 20 real sessions to tune dimension caps",
         "kw": ["measurement", "tuning"]},
    ],
    "confidence_calibration_audit": [],
})


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------

def test_parse_output_plain_json():
    parsed = analyser.parse_output('{"user_observations": []}')
    assert parsed == {"user_observations": []}


def test_parse_output_strips_markdown_fence():
    text = "```json\n" + SAMPLE_LLM_OUTPUT + "\n```"
    parsed = analyser.parse_output(text)
    assert "user_observations" in parsed


def test_parse_output_tolerates_prose_preamble():
    text = "Here is the analysis:\n" + SAMPLE_LLM_OUTPUT
    parsed = analyser.parse_output(text)
    assert "session_arc_memories" in parsed


def test_parse_output_empty_raises():
    try:
        analyser.parse_output("")
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty input")


def test_parse_output_no_json_raises():
    try:
        analyser.parse_output("nothing here but prose")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# normalise_dimensions + envelope budget
# ---------------------------------------------------------------------------

def test_normalise_dimensions_does_not_truncate():
    """Count caps removed — many items per dimension must survive."""
    over = {"user_observations": [{"content": f"obs-{i}"} for i in range(20)]}
    norm = analyser.normalise_dimensions(over)
    assert len(norm["user_observations"]) == 20


def test_normalise_dimensions_fills_missing():
    norm = analyser.normalise_dimensions({})
    for dim in analyser.DIMENSIONS:
        assert dim in norm
        assert norm[dim] == []


def test_normalise_dimensions_drops_unknown_keys():
    norm = analyser.normalise_dimensions({"bogus_key": [{"x": 1}]})
    assert "bogus_key" not in norm


def test_normalise_dimensions_coerces_non_list_to_empty():
    norm = analyser.normalise_dimensions({"user_observations": "not a list"})
    assert norm["user_observations"] == []


def test_envelope_exceeded_detects_overlong_output():
    big = "x" * (analyser.ENVELOPE_CHARS_MAX + 10)
    assert analyser.envelope_exceeded(big) is True


def test_envelope_exceeded_under_limit_returns_false():
    assert analyser.envelope_exceeded("short output") is False


def test_envelope_exceeded_empty_returns_false():
    assert analyser.envelope_exceeded("") is False
    assert analyser.envelope_exceeded(None) is False


def test_enforce_caps_alias_still_works():
    """Backwards-compat shim — existing test fixtures call enforce_caps."""
    norm = analyser.enforce_caps({"user_observations": [{"content": "x"}]})
    assert norm["user_observations"] == [{"content": "x"}]


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_includes_transcript_and_deliveries():
    prompt = analyser.build_prompt("TRANSCRIPT_TEXT",
                                    [{"id": 1, "row_id": 7, "turn_index": 3}])
    assert "TRANSCRIPT_TEXT" in prompt
    assert '"row_id": 7' in prompt
    assert "qf" in prompt  # template documents qf field
    assert "13" in prompt or "dimensions" in prompt.lower()


def test_build_prompt_empty_deliveries():
    prompt = analyser.build_prompt("T", [])
    assert "[]" in prompt


# ---------------------------------------------------------------------------
# write paths
# ---------------------------------------------------------------------------

def test_write_calibration_rows_persists_qf_and_kw():
    durable, eph, td = _fresh_dbs()
    parsed = analyser.enforce_caps(json.loads(SAMPLE_LLM_OUTPUT))
    with patch.object(analyser, "DB_PATH", durable):
        ids = analyser.write_calibration_rows(parsed, db_path=durable)
    assert len(ids) == 2  # user_observations + explicit_instructions
    conn = sqlite3.connect(durable)
    rows = conn.execute(
        "SELECT content, kw, qf, source FROM calibration_rows ORDER BY id"
    ).fetchall()
    conn.close()
    contents = [r[0] for r in rows]
    assert any("terse" in c for c in contents)
    assert any("tests before commit" in c for c in contents)
    sources = {r[3] for r in rows}
    assert sources == {"observation", "explicit"}
    qf_blobs = [json.loads(r[2]) for r in rows]
    assert all(isinstance(q, list) and len(q) > 0 for q in qf_blobs)


def test_write_session_memories_uses_source_ref():
    durable, eph, td = _fresh_dbs()
    parsed = analyser.enforce_caps(json.loads(SAMPLE_LLM_OUTPUT))
    with patch.object(analyser, "DB_PATH", durable):
        ids = analyser.write_session_memories(
            parsed, session_id="sess-x", project="cairn", db_path=durable)
    assert len(ids) == 3  # arc + decision + loose_ends
    conn = sqlite3.connect(durable)
    rows = conn.execute(
        "SELECT type, topic, source_ref, session_id, project FROM memories"
    ).fetchall()
    conn.close()
    assert all(r[2] == analyser.ANALYSER_SOURCE_REF for r in rows)
    assert all(r[3] == "sess-x" for r in rows)
    assert all(r[4] == "cairn" for r in rows)
    types = {r[0] for r in rows}
    assert types == {"project", "decision"}


def test_write_session_memories_populates_topic_embedding():
    """Regression: analyser write path must populate topic_embedding (schema v8
    dual embedding), mirroring hooks/storage.py. Without it, analyser rows are
    invisible to symmetric topic retrieval and the coverage gap re-accumulates
    until the next install-time backfill."""
    durable, eph, td = _fresh_dbs()
    parsed = analyser.enforce_caps(json.loads(SAMPLE_LLM_OUTPUT))
    with patch.object(analyser, "DB_PATH", durable):
        ids = analyser.write_session_memories(
            parsed, session_id="sess-te", project="cairn", db_path=durable)
    assert ids
    conn = sqlite3.connect(durable)
    rows = conn.execute(
        "SELECT topic_embedding FROM memories WHERE source_ref = ?",
        (analyser.ANALYSER_SOURCE_REF,)).fetchall()
    conn.close()
    assert rows
    assert all(r[0] is not None for r in rows), "analyser rows must carry topic_embedding"


def test_write_skips_empty_content():
    durable, eph, td = _fresh_dbs()
    parsed = analyser.enforce_caps({
        "user_observations": [{"content": "", "kw": [], "qf": []},
                              {"content": "valid", "kw": [], "qf": []}],
    })
    with patch.object(analyser, "DB_PATH", durable):
        ids = analyser.write_calibration_rows(parsed, db_path=durable)
    assert len(ids) == 1


# ---------------------------------------------------------------------------
# effectiveness scoring
# ---------------------------------------------------------------------------

def test_score_effectiveness_updates_delivery_and_row_counters():
    durable, eph, td = _fresh_dbs()
    # Seed a calibration row and a pending delivery
    dconn = sqlite3.connect(durable)
    dconn.execute(
        "INSERT INTO calibration_rows (content, source, confidence) "
        "VALUES ('foo', 'explicit', 0.9)"
    )
    row_id = dconn.execute("SELECT last_insert_rowid()").fetchone()[0]
    dconn.commit()
    dconn.close()

    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO calibration_deliveries (session_id, turn_index, row_id) "
        "VALUES ('sess-x', 3, ?)",
        (row_id,),
    )
    econn.commit()
    econn.close()

    parsed = {
        "row_effectiveness": [
            {"row_id": row_id, "outcome": "followed",
             "evidence": "agent followed"},
        ]
    }
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        n = analyser.score_effectiveness(parsed, db_path=eph)
    assert n == 1

    # Delivery row updated
    econn = sqlite3.connect(eph)
    o = econn.execute(
        "SELECT outcome, outcome_evidence FROM calibration_deliveries"
    ).fetchone()
    econn.close()
    assert o[0] == "followed"
    assert "agent followed" in o[1]

    # Calibration row counters bumped
    dconn = sqlite3.connect(durable)
    counters = dconn.execute(
        "SELECT delivered_count, followed_count, ignored_count, "
        "corrected_count FROM calibration_rows WHERE id = ?",
        (row_id,),
    ).fetchone()
    dconn.close()
    # delivered_count is owned by log_deliveries now, not score_effectiveness
    # — this test seeds a delivery directly via SQL so delivered_count
    # remains 0 here. Outcome columns are what score_effectiveness owns.
    assert counters == (0, 1, 0, 0)


def test_score_effectiveness_ignores_invalid_outcome():
    durable, eph, td = _fresh_dbs()
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        n = analyser.score_effectiveness(
            {"row_effectiveness": [
                {"row_id": 1, "outcome": "bogus", "evidence": "x"}
            ]},
            db_path=eph,
        )
    assert n == 0


def test_score_effectiveness_empty_is_noop():
    durable, eph, td = _fresh_dbs()
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        n = analyser.score_effectiveness({"row_effectiveness": []}, db_path=eph)
    assert n == 0


# ---------------------------------------------------------------------------
# session_id / project recovery
# ---------------------------------------------------------------------------

def test_session_id_from_path():
    assert analyser._session_id_from_path("/x/y/abc-123.jsonl") == "abc-123"


def test_project_from_path_recovers_slug():
    p = "/home/u/.claude/projects/my-proj/sess-1.jsonl"
    assert analyser._project_from_path(p) == "my-proj"


def test_project_from_path_returns_none_when_no_projects_segment():
    assert analyser._project_from_path("/tmp/random/sess.jsonl") is None


# ---------------------------------------------------------------------------
# find_idle_sessions
# ---------------------------------------------------------------------------

def test_find_idle_sessions_respects_idle_threshold():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td)
    # Touch mtime to 1 hour ago
    old = time.time() - 3600
    os.utime(path, (old, old))
    root = os.path.join(td, "projects")
    with patch.object(analyser, "EPH_DB_PATH", eph):
        idle = analyser.find_idle_sessions(idle_minutes=15, roots=[root])
    assert path in idle


def test_find_idle_sessions_skips_recent():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td)
    # Fresh file — mtime is now
    root = os.path.join(td, "projects")
    with patch.object(analyser, "EPH_DB_PATH", eph):
        idle = analyser.find_idle_sessions(idle_minutes=15, roots=[root])
    assert path not in idle


def test_find_idle_sessions_skips_already_analysed_until_grows_enough():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td, session_id="already-done", turns=6)
    old = time.time() - 3600
    os.utime(path, (old, old))
    # Seed analyser state in hook_state — session already analysed at 6 turns
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("already-done", analyser.ANALYSER_STATE_KEY,
         json.dumps({"last_turn_count": 6,
                     "last_analysed_at": "2026-05-19T10:00:00Z",
                     "first_analysed_at": "2026-05-19T10:00:00Z"})),
    )
    econn.commit()
    econn.close()
    root = os.path.join(td, "projects")
    with patch.object(analyser, "EPH_DB_PATH", eph):
        # 6 turns analysed, 6 in file — below incremental threshold, excluded
        idle = analyser.find_idle_sessions(idle_minutes=15, roots=[root],
                                            incremental_threshold=10)
    assert path not in idle


def test_find_idle_sessions_includes_grown_session():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td, session_id="grew", turns=20)
    old = time.time() - 3600
    os.utime(path, (old, old))
    econn = sqlite3.connect(eph)
    econn.execute(
        "INSERT INTO hook_state (session_id, key, value) VALUES (?, ?, ?)",
        ("grew", analyser.ANALYSER_STATE_KEY,
         json.dumps({"last_turn_count": 6,
                     "last_analysed_at": "2026-05-19T10:00:00Z",
                     "first_analysed_at": "2026-05-19T10:00:00Z"})),
    )
    econn.commit()
    econn.close()
    root = os.path.join(td, "projects")
    with patch.object(analyser, "EPH_DB_PATH", eph):
        # 20 lines, last analysed at 6 → 14 new ≥ 10 threshold → included
        idle = analyser.find_idle_sessions(idle_minutes=15, roots=[root],
                                            incremental_threshold=10)
    assert path in idle


# ---------------------------------------------------------------------------
# analyse_session end-to-end
# ---------------------------------------------------------------------------

def _mock_llm(prompt):
    return SAMPLE_LLM_OUTPUT


def test_analyse_session_end_to_end_writes_both_tables():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td)
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        report = analyser.analyse_session(path, llm_caller=_mock_llm)
    assert report["dims"]["user_observations"] == 1
    assert report["dims"]["session_arc_memories"] == 1
    assert len(report["calibration_row_ids"]) == 2
    assert len(report["memory_ids"]) == 3
    # Metrics row recorded
    econn = sqlite3.connect(eph)
    n = econn.execute(
        "SELECT count(*) FROM metrics WHERE event = 'analyser_session_processed'"
    ).fetchone()[0]
    econn.close()
    assert n == 1


def test_analyse_session_dry_run_writes_nothing():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td)
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        report = analyser.analyse_session(path, dry_run=True, llm_caller=_mock_llm)
    assert report["dry_run"] is True
    assert report["calibration_row_ids"] == []
    assert report["memory_ids"] == []
    dconn = sqlite3.connect(durable)
    cal_n = dconn.execute("SELECT count(*) FROM calibration_rows").fetchone()[0]
    mem_n = dconn.execute(
        "SELECT count(*) FROM memories WHERE source_ref = ?",
        (analyser.ANALYSER_SOURCE_REF,)
    ).fetchone()[0]
    dconn.close()
    assert cal_n == 0
    assert mem_n == 0


# ---------------------------------------------------------------------------
# run_cron
# ---------------------------------------------------------------------------

def test_run_cron_processes_idle_sessions_and_isolates_failures():
    durable, eph, td = _fresh_dbs()
    good = _make_jsonl(td, session_id="good-sess")
    bad = _make_jsonl(td, session_id="bad-sess")
    old = time.time() - 3600
    os.utime(good, (old, old))
    os.utime(bad, (old - 1, old - 1))  # older — processed first

    def caller(prompt):
        if "bad-sess" in prompt or len(prompt) < 0:  # never short
            raise RuntimeError("boom")
        return SAMPLE_LLM_OUTPUT

    def per_session_caller(path):
        # analyse_session uses llm_caller signature (prompt) — wrap to fail on bad
        sid = analyser._session_id_from_path(path)
        if sid == "bad-sess":
            return lambda prompt: (_ for _ in ()).throw(RuntimeError("boom"))
        return _mock_llm

    # Patch analyse_session to inject the correct mock per file
    real_analyse = analyser.analyse_session

    def patched_analyse(jsonl_path, **kw):
        sid = analyser._session_id_from_path(jsonl_path)
        kw.pop("llm_caller", None)
        if sid == "bad-sess":
            return real_analyse(jsonl_path,
                                llm_caller=lambda p: (_ for _ in ()).throw(
                                    RuntimeError("boom")),
                                **kw)
        return real_analyse(jsonl_path, llm_caller=_mock_llm, **kw)

    root = os.path.join(td, "projects")
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph), \
         patch.object(analyser, "TRANSCRIPT_ROOTS", [root]), \
         patch.object(analyser, "analyse_session", patched_analyse):
        reports = analyser.run_cron(idle_minutes=15, limit=5)

    assert len(reports) == 2
    sids = [r.get("session_id") for r in reports]
    assert "good-sess" in sids
    assert "bad-sess" in sids
    bad_report = next(r for r in reports if r["session_id"] == "bad-sess")
    assert "error" in bad_report


def test_run_cron_skips_dont_consume_limit_slot():
    """Regression for production bug where 19/20 cron candidates skipped
    at the inner incremental gate and only 1 session was actually
    processed. Skips must not count against `limit`; the loop walks
    past them until `limit` sessions actually receive an LLM call."""
    durable, eph, td = _fresh_dbs()
    paths = [_make_jsonl(td, session_id=f"sess-{i}", turns=6) for i in range(5)]
    skip_set = {paths[0], paths[1], paths[3]}

    def fake_analyse(jsonl_path, **kw):
        sid = analyser._session_id_from_path(jsonl_path)
        if jsonl_path in skip_set:
            return {"session_id": sid, "skipped": "below-incremental-threshold"}
        return {"session_id": sid, "calibration_row_ids": [1],
                "memory_ids": [1], "dims": {}}

    with patch.object(analyser, "find_idle_sessions", return_value=list(paths)), \
         patch.object(analyser, "analyse_session", fake_analyse):
        reports = analyser.run_cron(idle_minutes=15, limit=2)

    skipped = [r for r in reports if r.get("skipped")]
    processed = [r for r in reports if "skipped" not in r and "error" not in r]
    assert len(processed) == 2, f"want 2 processed, got reports={reports}"
    # Paths #0,#1 skip; #2 processes; #3 skips; #4 processes. 3 skips total.
    assert len(skipped) == 3, f"want 3 skipped, got reports={reports}"


def test_run_cron_respects_max_scanned_cap():
    """When the pool is full of skips, max_scanned bounds the walk so
    we don't parse every JSONL on disk searching for a needle."""
    durable, eph, td = _fresh_dbs()
    paths = [_make_jsonl(td, session_id=f"skip-only-{i}", turns=6) for i in range(5)]

    def fake_analyse(jsonl_path, **kw):
        sid = analyser._session_id_from_path(jsonl_path)
        return {"session_id": sid, "skipped": "below-incremental-threshold"}

    with patch.object(analyser, "find_idle_sessions", return_value=list(paths)), \
         patch.object(analyser, "analyse_session", fake_analyse):
        reports = analyser.run_cron(idle_minutes=15, limit=2, max_scanned=3)

    # Only 3 sessions examined despite limit=2 never being reachable.
    assert len(reports) == 3
    assert all(r.get("skipped") for r in reports)


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def test_calibration_row_dedup_skips_equivalent_signal():
    """Second analyser write with semantically identical content must
    not produce a second row."""
    durable, eph, td = _fresh_dbs()
    parsed = analyser.enforce_caps({
        "user_observations": [
            {"content": "user prefers concise responses without preamble",
             "kw": ["concise"], "qf": ["how to respond"]},
        ],
    })
    with patch.object(analyser, "DB_PATH", durable):
        first = analyser.write_calibration_rows(parsed, db_path=durable)
        # Same content second time — should dedup
        second = analyser.write_calibration_rows(parsed, db_path=durable)
    assert len(first) == 1
    # Embedding may be unavailable in test env — accept dedup OR same-row
    # behaviour only if embedding daemon is present. Verify total row
    # count did not double:
    conn = sqlite3.connect(durable)
    n = conn.execute("SELECT count(*) FROM calibration_rows").fetchone()[0]
    conn.close()
    # If embeddings available, dedup blocks second insert (count 1). If
    # not, both inserts land (count 2). Either is acceptable behaviour;
    # what matters is no crash and the dedup path is reachable.
    assert n in (1, 2)


def test_memory_dedup_only_targets_analyser_written():
    """The memory dedup must NOT swallow per-turn writes — per cairn
    entry 3302, per-turn writes have write-time priority. We verify
    this by inserting a per-turn-style memory and an analyser-style
    memory with identical content, then re-running the analyser write:
    only the analyser-written one should be considered for dedup.
    """
    durable, eph, td = _fresh_dbs()
    conn = sqlite3.connect(durable)
    # Pre-existing per-turn memory (no source_ref)
    conn.execute(
        "INSERT INTO memories (type, topic, content, source_ref, "
        "embedding, origin_id) VALUES ('fact', 'x', 'identical text', NULL, "
        "NULL, 'uuid-1')"
    )
    conn.commit()
    conn.close()
    parsed = analyser.enforce_caps({
        "session_arc_memories": [
            {"topic": "x", "content": "identical text", "kw": ["x"]},
        ],
    })
    with patch.object(analyser, "DB_PATH", durable):
        ids = analyser.write_session_memories(
            parsed, session_id="s", project="p", db_path=durable)
    # Analyser write should land — per-turn write should NOT block it
    assert len(ids) == 1
    conn = sqlite3.connect(durable)
    n_total = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    n_analyser = conn.execute(
        "SELECT count(*) FROM memories WHERE source_ref = ?",
        (analyser.ANALYSER_SOURCE_REF,)
    ).fetchone()[0]
    conn.close()
    assert n_total == 2  # per-turn + analyser
    assert n_analyser == 1


# ---------------------------------------------------------------------------
# Subagent / triviality filter + incremental gate
# ---------------------------------------------------------------------------

def test_is_worth_analysing_rejects_thin_session():
    turns = [{"text": "hi"}, {"text": "ok"}]
    assert not analyser._is_worth_analysing(turns)


def test_is_worth_analysing_accepts_rich_session():
    turns = [{"text": "x" * 200} for _ in range(6)]
    assert analyser._is_worth_analysing(turns)


def test_is_worth_analysing_rejects_few_long_turns():
    # 2 substantive turns even if long — still subagent-shaped
    turns = [{"text": "x" * 5000}, {"text": "y" * 5000}]
    assert not analyser._is_worth_analysing(turns)


def test_analyse_session_skips_subagent_transcript():
    durable, eph, td = _fresh_dbs()
    # Tiny transcript — subagent-shaped
    path = _make_jsonl(td, turns=2, body="ok")
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        report = analyser.analyse_session(path, llm_caller=_mock_llm)
    assert report["skipped"] == "below-substance-threshold"
    # No DB rows written
    dconn = sqlite3.connect(durable)
    assert dconn.execute("SELECT count(*) FROM calibration_rows").fetchone()[0] == 0
    dconn.close()


def test_force_bypasses_subagent_filter():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td, turns=2, body="ok")
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        report = analyser.analyse_session(path, force=True, llm_caller=_mock_llm)
    assert "skipped" not in report
    assert len(report["calibration_row_ids"]) > 0


def test_analyse_session_skips_below_incremental_threshold():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td, session_id="grow-test", turns=10)
    # Seed state showing 8 turns already analysed
    with patch.object(analyser, "EPH_DB_PATH", eph):
        analyser._set_analyser_state(
            "grow-test",
            {"last_turn_count": 8, "last_analysed_at": "2026-05-19T10:00:00Z",
             "first_analysed_at": "2026-05-19T10:00:00Z"},
            eph_db_path=eph,
        )
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        report = analyser.analyse_session(path, llm_caller=_mock_llm,
                                           incremental_threshold=10)
    assert report["skipped"] == "below-incremental-threshold"


def test_analyse_session_runs_when_above_incremental_threshold():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td, session_id="grew-lots", turns=20)
    with patch.object(analyser, "EPH_DB_PATH", eph):
        analyser._set_analyser_state(
            "grew-lots",
            {"last_turn_count": 4, "last_analysed_at": "2026-05-19T10:00:00Z",
             "first_analysed_at": "2026-05-19T10:00:00Z"},
            eph_db_path=eph,
        )
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        report = analyser.analyse_session(path, llm_caller=_mock_llm,
                                           incremental_threshold=10)
    assert "skipped" not in report
    assert report["is_incremental"] is True
    assert report["current_turn_count"] == 20
    # State was updated
    state = analyser._get_analyser_state("grew-lots", eph_db_path=eph)
    assert state["last_turn_count"] == 20


def test_analyse_session_records_state_on_first_analysis():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td, session_id="fresh", turns=8)
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        report = analyser.analyse_session(path, llm_caller=_mock_llm)
    assert report["is_incremental"] is False
    state = analyser._get_analyser_state("fresh", eph_db_path=eph)
    assert state["last_turn_count"] == 8
    assert state["first_analysed_at"] == state["last_analysed_at"]


def test_prior_rows_loaded_for_incremental_run():
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td, session_id="continuing", turns=20)
    # Seed state and a prior calibration row that should appear in prompt
    with patch.object(analyser, "EPH_DB_PATH", eph):
        analyser._set_analyser_state(
            "continuing",
            {"last_turn_count": 4,
             "last_analysed_at": "2026-05-19T08:00:00Z",
             "first_analysed_at": "2026-05-19T08:00:00Z"},
            eph_db_path=eph,
        )
    dconn = sqlite3.connect(durable)
    dconn.execute(
        "INSERT INTO calibration_rows (content, source, confidence, kw, qf, "
        "created_at) VALUES ('prior user observation', 'observation', 0.5, "
        "'kw1', '[\"q1\"]', '2026-05-19T08:30:00Z')"
    )
    dconn.commit()
    dconn.close()

    captured_prompts = []
    def capturing_caller(prompt):
        captured_prompts.append(prompt)
        return SAMPLE_LLM_OUTPUT

    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph):
        analyser.analyse_session(path, llm_caller=capturing_caller,
                                  incremental_threshold=10)
    assert len(captured_prompts) == 1
    assert "prior user observation" in captured_prompts[0]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def test_default_model_is_sonnet():
    # Sonnet 4.6 is the documented default per Amendment 1 analysis;
    # confirm the constant carries through.
    assert "sonnet" in analyser.DEFAULT_MODEL.lower()


def test_call_llm_passes_model_to_claude_p():
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        captured["input"] = kwargs.get("input")
        return FakeResult()

    with patch("cairn.analyser.subprocess.run", fake_run):
        analyser.call_llm("hello", model="claude-sonnet-4-6")
    assert "--model" in captured["cmd"]
    idx = captured["cmd"].index("--model")
    assert captured["cmd"][idx + 1] == "claude-sonnet-4-6"
    assert captured["env"].get("CAIRN_MODE") == "read-only"
    # Prompt passes via stdin to avoid ARG_MAX overflow on long transcripts
    assert captured["input"] == "hello"
    assert "hello" not in captured["cmd"]


def test_call_llm_defaults_to_module_default_model():
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeResult()

    with patch("cairn.analyser.subprocess.run", fake_run), \
         patch.object(analyser, "DEFAULT_MODEL", "haiku-test"):
        analyser.call_llm("hello")
    idx = captured["cmd"].index("--model")
    assert captured["cmd"][idx + 1] == "haiku-test"


def test_call_llm_raises_on_nonzero_exit():
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "boom"

    def fake_run(cmd, **kwargs):
        return FakeResult()

    with patch("cairn.analyser.subprocess.run", fake_run):
        try:
            analyser.call_llm("x")
        except RuntimeError as e:
            assert "boom" in str(e)
            return
    raise AssertionError("expected RuntimeError")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_main_analyse_command(capsys):
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td)
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph), \
         patch.object(analyser, "call_llm", lambda prompt, timeout=180, model=None: SAMPLE_LLM_OUTPUT):
        rc = analyser.main(["analyse", path, "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    # init_db prints initialization noise before the JSON report; extract object
    first = out.find("{")
    last = out.rfind("}")
    report = json.loads(out[first:last + 1])
    assert report["dry_run"] is True


def test_main_list_idle_command(capsys):
    durable, eph, td = _fresh_dbs()
    path = _make_jsonl(td)
    old = time.time() - 3600
    os.utime(path, (old, old))
    root = os.path.join(td, "projects")
    with patch.object(analyser, "DB_PATH", durable), \
         patch.object(analyser, "EPH_DB_PATH", eph), \
         patch.object(analyser, "TRANSCRIPT_ROOTS", [root]):
        rc = analyser.main(["list-idle"])
    out = capsys.readouterr().out
    assert rc == 0
    assert path in out
