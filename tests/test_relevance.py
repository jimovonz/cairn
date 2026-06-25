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


# ---- step 1a: ranking provenance (reranker_model / score_components / layer / scope) ----
def test_log_records_ranking_provenance(eph):
    delivered = [
        {"id": 42, "score": 0.8, "ce_score": 1.2, "rrf_score": 0.3, "similarity": 0.7,
         "reranker_model": "BAAI/bge-reranker-base", "project": "proj"},
        {"id": 17, "score": 0.5, "project": "other"},  # no CE / no model
    ]
    relevance.log_memory_deliveries(
        delivered, session_id="s", layer="per-prompt", project="proj", eph_path=eph)
    conn = sqlite3.connect(eph)
    rows = conn.execute(
        "SELECT memory_id, reranker_model, score_components, layer, scope "
        "FROM memory_deliveries WHERE session_id='s' ORDER BY served_rank").fetchall()
    # row 0 — full provenance, project scope (project==delivery project)
    assert rows[0][0] == 42
    assert rows[0][1] == "BAAI/bge-reranker-base"
    comp = json.loads(rows[0][2])
    assert comp == {"ce": 1.2, "composite": 0.8, "rrf": 0.3, "sim": 0.7}
    assert rows[0][3] == "per-prompt"
    assert rows[0][4] == "project"
    # row 1 — no reranker model, only composite component, global scope
    assert rows[1][0] == 17
    assert rows[1][1] is None
    assert json.loads(rows[1][2]) == {"composite": 0.5}
    assert rows[1][4] == "global"


def test_score_components_none_when_no_signals(eph):
    relevance.log_memory_deliveries([{"id": 9}], session_id="s", eph_path=eph)
    sc = sqlite3.connect(eph).execute(
        "SELECT score_components, scope FROM memory_deliveries WHERE memory_id=9").fetchone()
    assert sc[0] is None        # no score signals at all
    assert sc[1] == "global"    # no project passed -> global


# ---- step 2: behavioural engagement signal -----------------------------------
def test_score_engagement_used_vs_unused():
    mem = "The reranker uses bge-reranker-base with a sigmoid floor of 0.0005"
    # response surfaces >=2 distinctive memory terms -> engaged
    eng, score = relevance.score_engagement(
        "I set the bge-reranker-base sigmoid floor as discussed", mem)
    assert eng == 1 and score > 0
    # response on an unrelated topic -> not engaged
    eng, score = relevance.score_engagement("the weather is nice today", mem)
    assert eng == 0 and score == 0.0


def test_score_engagement_subtracts_prompt_terms():
    # Terms shared with the prompt don't count (they'd be repeated regardless).
    mem = "reranker floor calibration logit threshold"
    prompt = "what is the reranker floor"           # 'reranker','floor' come from prompt
    # response echoes only the prompt-shared terms -> NOT evidence of memory use
    eng, score = relevance.score_engagement("the reranker floor matters", mem, prompt)
    assert eng == 0
    # response surfaces the memory-distinctive terms -> engaged
    eng, score = relevance.score_engagement(
        "calibration of the logit threshold", mem, prompt)
    assert eng == 1


def test_score_engagement_undecidable_when_redundant_with_prompt():
    mem = "reranker floor"
    prompt = "tune the reranker floor please"   # memory adds nothing over the prompt
    eng, score = relevance.score_engagement("anything at all", mem, prompt)
    assert eng is None and score == -1.0


def _durable_with_memories(tmp_path, rows):
    """Minimal durable DB with a memories table for engagement tests."""
    p = str(tmp_path / "durable.db")
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT, "
              "topic TEXT, keywords TEXT)")
    c.executemany("INSERT INTO memories (id, content, topic, keywords) VALUES (?,?,?,?)", rows)
    c.commit(); c.close()
    return p


def test_apply_engagement_stamps_rows(eph, tmp_path):
    durable = _durable_with_memories(tmp_path, [
        (1, "bge-reranker-base sigmoid floor calibration", "reranker", "bge logit"),
        (2, "the user prefers concise commit messages", "style", "commits"),
    ])
    relevance.log_memory_deliveries(
        [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.5}],
        session_id="s", context_text="[user] how do I tune the floor", eph_path=eph)
    response = "I calibrated the bge-reranker-base sigmoid logit floor for you"
    n = relevance.apply_engagement(response, session_id="s", eph_path=eph,
                                   durable_path=durable)
    assert n == 2
    rows = dict((r[0], (r[1], r[2])) for r in sqlite3.connect(eph).execute(
        "SELECT memory_id, engaged, engaged_score FROM memory_deliveries WHERE session_id='s'"))
    assert rows[1][0] == 1            # memory 1 clearly used
    assert rows[1][1] > 0
    assert rows[2][0] == 0            # memory 2 (commit style) not used


def test_apply_engagement_idempotent_only_unscored(eph, tmp_path):
    durable = _durable_with_memories(tmp_path, [
        (5, "distinctive terraform kubernetes helmchart", "infra", "deploy")])
    relevance.log_memory_deliveries([{"id": 5, "score": 0.8}], session_id="s",
                                    context_text="[user] q", eph_path=eph)
    first = relevance.apply_engagement("using terraform and kubernetes helmchart",
                                       session_id="s", eph_path=eph, durable_path=durable)
    assert first == 1
    # second call: row already scored (engaged_score NOT NULL) -> nothing re-scored
    second = relevance.apply_engagement("totally different response",
                                        session_id="s", eph_path=eph, durable_path=durable)
    assert second == 0
    eng = sqlite3.connect(eph).execute(
        "SELECT engaged FROM memory_deliveries WHERE memory_id=5").fetchone()[0]
    assert eng == 1   # preserved from first scoring, not clobbered


def test_apply_engagement_failsoft_empty():
    assert relevance.apply_engagement("", session_id="s") == 0
    assert relevance.apply_engagement("resp", session_id="") == 0


def test_build_context_xml_stamps_layer_and_scope(tmp_path, monkeypatch):
    p = str(tmp_path / "eph.db")
    init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    from hooks.hook_helpers import build_context_xml
    pr = [{"id": 1, "type": "fact", "topic": "t", "content": "c", "project": "x",
           "updated_at": "now", "confidence": 0.9, "score": 0.9, "similarity": 0.7,
           "ce_score": 2.0, "reranker_model": "m", "archived_reason": None}]
    gr = [{"id": 2, "type": "fact", "topic": "t", "content": "c2", "project": "y",
           "updated_at": "now", "confidence": 0.9, "score": 0.4, "similarity": 0.3,
           "archived_reason": None}]
    build_context_xml("q", "x", "first-prompt", pr, gr, session_id="sess",
                      context_text="[user] q")
    rows = dict((r[0], (r[1], r[2], r[3])) for r in sqlite3.connect(p).execute(
        "SELECT memory_id, layer, scope, reranker_model "
        "FROM memory_deliveries WHERE session_id='sess'").fetchall())
    assert rows[1][0] == "first-prompt" and rows[1][1] == "project" and rows[1][2] == "m"
    assert rows[2][0] == "first-prompt" and rows[2][1] == "global" and rows[2][2] is None


# ---- wiring: build_context_xml prefilter + delivery logging -------------------
def test_build_context_xml_logs_deliveries(tmp_path, monkeypatch):
    import sqlite3
    p = str(tmp_path / "eph.db")
    init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    from hooks.hook_helpers import build_context_xml
    pr = [{"id": 1, "type": "fact", "topic": "t", "content": "c", "project": "x",
           "updated_at": "now", "confidence": 0.9, "score": 0.9, "similarity": 0.7,
           "archived_reason": None}]
    build_context_xml("q", "x", "per-prompt", pr, [], session_id="sess",
                      context_text="[user] q")
    rows = sqlite3.connect(p).execute(
        "SELECT memory_id, context_text FROM memory_deliveries WHERE session_id='sess'").fetchall()
    assert rows == [(1, "[user] q")]


def test_build_context_xml_records_context_vec(tmp_path, monkeypatch):
    # context_vec (the empirical-context join key) is embedded from context_text at
    # delivery time when an embedder is available.
    p = str(tmp_path / "eph.db")
    init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    import hooks.hook_helpers as hh

    class _Emb:
        def embed(self, text, allow_slow=True):
            return [0.1, 0.2, 0.3]
        def to_blob(self, vec):
            return b"\x01\x02\x03\x04"
    monkeypatch.setattr(hh, "get_embedder", lambda: _Emb())
    pr = [{"id": 1, "type": "fact", "topic": "t", "content": "c", "project": "x",
           "updated_at": "now", "confidence": 0.9, "score": 0.9, "similarity": 0.7,
           "archived_reason": None}]
    hh.build_context_xml("q", "x", "per-prompt", pr, [], session_id="sess",
                         context_text="[user] q")
    cv = sqlite3.connect(p).execute(
        "SELECT context_vec FROM memory_deliveries WHERE session_id='sess'").fetchone()[0]
    assert cv == b"\x01\x02\x03\x04"


def _entry(i, typ, content):
    return {"id": i, "type": typ, "topic": "t", "content": content, "project": "x",
            "updated_at": "now", "confidence": 0.9, "score": 0.9, "similarity": 0.7,
            "archived_reason": None}


def test_build_context_xml_prefilter_gated_and_correction_exempt(tmp_path, monkeypatch):
    p = str(tmp_path / "eph.db")
    init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    monkeypatch.setattr("cairn.config.RELEVANCE_PREFILTER_ENABLED", True)
    from hooks.hook_helpers import build_context_xml
    pr = [
        _entry(1, "fact", "No memory of the user's brother exists yet"),  # meta -> dropped
        _entry(2, "correction", "No memory of X exists yet"),             # correction -> exempt
        _entry(3, "fact", "ms-marco rerank model is ms-marco-MiniLM"),    # normal -> kept
    ]
    xml = build_context_xml("q", "x", "per-prompt", pr, [], session_id="s", context_text="t")
    assert 'id="1"' not in xml   # bucket-4 meta dropped
    assert 'id="2"' in xml       # correction never gated
    assert 'id="3"' in xml       # normal kept


def test_build_context_xml_prefilter_off_by_default(tmp_path, monkeypatch):
    p = str(tmp_path / "eph.db")
    init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    # default RELEVANCE_PREFILTER_ENABLED is False -> meta NOT dropped
    from hooks.hook_helpers import build_context_xml
    pr = [_entry(1, "fact", "No memory of the user's brother exists yet")]
    xml = build_context_xml("q", "x", "per-prompt", pr, [], session_id="s", context_text="t")
    assert 'id="1"' in xml


def test_build_context_xml_no_session_skips_logging(tmp_path, monkeypatch):
    import sqlite3
    p = str(tmp_path / "eph.db")
    init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    from hooks.hook_helpers import build_context_xml
    # bootstrap-style call (no session_id) must not log
    build_context_xml("standing", "x", "project-bootstrap", [_entry(9, "fact", "c")], [])
    n = sqlite3.connect(p).execute("SELECT COUNT(*) FROM memory_deliveries").fetchone()[0]
    assert n == 0


# ---- reranker resolution (config.resolve_reranker) ----------------------------
def test_resolve_reranker_ms_marco_default_even_on_cuda(monkeypatch):
    # Step 1b revert: bge dormant by default -> ms-marco + -3.0 floor on EVERY
    # device, even when CUDA is present (it just loads on the GPU).
    import torch
    from cairn import config
    monkeypatch.setattr(config, "RERANKER_BGE_ENABLED", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    m, f = config.resolve_reranker()
    assert m == config.CROSS_ENCODER_MODEL
    assert f == config.CROSS_ENCODER_SCORE_FLOOR


def test_resolve_reranker_bge_when_flag_on_and_cuda(monkeypatch):
    # The dormant bge path the step-1b A/B will validate: flag on + CUDA -> bge.
    import torch
    from cairn import config
    monkeypatch.setattr(config, "RERANKER_BGE_ENABLED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    m, f = config.resolve_reranker()
    assert m == config.CROSS_ENCODER_MODEL_CUDA
    assert f == config.CROSS_ENCODER_SCORE_FLOOR_CUDA
    # flag on but no CUDA -> still ms-marco
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m, f = config.resolve_reranker()
    assert m == config.CROSS_ENCODER_MODEL


def test_resolve_reranker_falls_back_when_torch_missing(monkeypatch):
    # If torch import fails inside resolve, must fall back to the CPU default.
    import builtins
    from cairn import config
    real_import = builtins.__import__
    def _no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _no_torch)
    m, f = config.resolve_reranker()
    assert m == config.CROSS_ENCODER_MODEL and f == config.CROSS_ENCODER_SCORE_FLOOR


# ---- superseded-pair suppression (build_context_xml) --------------------------
def _superseded(i, by_id):
    e = _entry(i, "fact", f"old content {i}")
    e["archived_reason"] = f"superseded: replaced (by #{by_id})"
    return e


def test_superseded_dropped_when_superseder_present(tmp_path, monkeypatch):
    p = str(tmp_path / "eph.db"); init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    from hooks.hook_helpers import build_context_xml
    xml = build_context_xml("q", "x", "per-prompt",
                            [_superseded(10, 11), _entry(11, "fact", "new")], [],
                            session_id="s", context_text="t")
    assert 'id="10"' not in xml   # stale row dropped (its superseder #11 is present)
    assert 'id="11"' in xml


def test_lone_superseded_kept(tmp_path, monkeypatch):
    p = str(tmp_path / "eph.db"); init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    from hooks.hook_helpers import build_context_xml
    # superseder #11 NOT in the set -> keep the superseded row (negative-knowledge trail)
    xml = build_context_xml("q", "x", "per-prompt", [_superseded(10, 11)], [],
                            session_id="s", context_text="t")
    assert 'id="10"' in xml


def test_superseded_dropped_across_scopes(tmp_path, monkeypatch):
    p = str(tmp_path / "eph.db"); init_db.init_ephemeral(p)
    monkeypatch.setattr("cairn.config.EPHEMERAL_DB_PATH", p)
    from hooks.hook_helpers import build_context_xml
    # superseded in project scope, superseder in global scope -> still dropped
    xml = build_context_xml("q", "x", "per-prompt",
                            [_superseded(10, 11)], [_entry(11, "fact", "new")],
                            session_id="s", context_text="t")
    assert 'id="10"' not in xml and 'id="11"' in xml
