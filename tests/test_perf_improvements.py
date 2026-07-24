"""Focused tests for the perf-improvements batch.

Covers the daemon serving-health probes, the Layer 1.5 topic gate, the
source-utility ranking prior, semantic engagement scoring, the engagement
weak-label path into reranker training, and the belt-and-braces
declare-without-searching detector.
"""
import json
import os

import numpy as np
import pysqlite3 as sqlite3
import pytest


# --------------------------------------------------------------------------
# [a] Daemon serving-health probes
# --------------------------------------------------------------------------

def _daemon_responder(ping=None, embed=None, rerank=None):
    """Build a send_request stand-in dispatching on the request action."""
    def _send(req):
        action = req.get("action")
        if action == "ping":
            return ping
        if action == "embed":
            return embed
        return rerank
    return _send


def test_proto_current_accepts_matching_version(monkeypatch):
    import cairn.daemon as d
    monkeypatch.setattr(d, "send_request",
                        _daemon_responder(ping={"status": "ok", "proto": d.PROTOCOL_VERSION}))
    assert d._proto_current() is True


@pytest.mark.parametrize("ping", [
    {"status": "ok"},                 # predates the proto field — stale
    {"status": "ok", "proto": 999},   # wrong version
    None,                             # no response at all
])
def test_proto_current_rejects_stale_daemon(monkeypatch, ping):
    import cairn.daemon as d
    monkeypatch.setattr(d, "send_request", _daemon_responder(ping=ping))
    assert d._proto_current() is False


def test_embed_healthy_accepts_decodable_vector(monkeypatch):
    import cairn.daemon as d
    monkeypatch.setattr(d, "send_request",
                        _daemon_responder(embed={"vector": "00112233"}))
    assert d._embed_healthy() is True


@pytest.mark.parametrize("embed", [
    None,                        # no response
    {"error": "boom"},           # daemon reported failure
    {"vector": "zzzz"},          # not hex
    {"vector": ["not", "str"]},  # not a string
    {"vector": "00"},            # decodes, but under the 4-byte floor
])
def test_embed_healthy_rejects_bad_vector(monkeypatch, embed):
    import cairn.daemon as d
    monkeypatch.setattr(d, "send_request", _daemon_responder(embed=embed))
    assert d._embed_healthy() is False


def test_serving_healthy_requires_every_probe(monkeypatch):
    import cairn.daemon as d
    good = dict(ping={"status": "ok", "proto": d.PROTOCOL_VERSION},
                embed={"vector": "00112233"}, rerank={"scores": [0.5]})
    monkeypatch.setattr(d, "send_request", _daemon_responder(**good))
    assert d._serving_healthy() is True

    for broken in ("ping", "embed", "rerank"):
        kwargs = dict(good)
        kwargs[broken] = None
        monkeypatch.setattr(d, "send_request", _daemon_responder(**kwargs))
        assert d._serving_healthy() is False, f"{broken} failure must fail the gate"


# --------------------------------------------------------------------------
# [b] Layer 1.5 topic gate
# --------------------------------------------------------------------------

@pytest.fixture
def gate_state(monkeypatch):
    """Dict-backed hook_state so the gate's anchor is inspectable."""
    import hooks.hook_helpers as hh
    store = {}
    monkeypatch.setattr(hh, "load_hook_state", lambda sid, key: store.get((sid, key)))
    monkeypatch.setattr(hh, "save_hook_state",
                        lambda sid, key, val: store.__setitem__((sid, key), val))
    return store


def _set_embed(monkeypatch, vec):
    import cairn.embeddings as emb
    monkeypatch.setattr(emb, "embed",
                        lambda text, allow_slow=True: None if vec is None
                        else np.asarray(vec, dtype=np.float32))


def test_topic_gate_skip_first_prompt_anchors_and_searches(monkeypatch, gate_state):
    from hooks.prompt_hook import _topic_gate_skip
    _set_embed(monkeypatch, [1.0, 0.0, 0.0])
    assert _topic_gate_skip("first question", "s1") is False
    assert ("s1", "l1_5_topic_vec") in gate_state, "anchor must be saved when a search runs"


def test_topic_gate_skip_same_topic_skips_without_reanchoring(monkeypatch, gate_state):
    from hooks.prompt_hook import _topic_gate_skip
    _set_embed(monkeypatch, [1.0, 0.0, 0.0])
    assert _topic_gate_skip("first question", "s1") is False
    anchor = gate_state[("s1", "l1_5_topic_vec")]

    assert _topic_gate_skip("same topic restated", "s1") is True
    assert gate_state[("s1", "l1_5_topic_vec")] == anchor, \
        "a skip must NOT re-anchor, or slow drift chain-skips forever"


def test_topic_gate_skip_topic_change_searches_and_reanchors(monkeypatch, gate_state):
    from hooks.prompt_hook import _topic_gate_skip
    _set_embed(monkeypatch, [1.0, 0.0, 0.0])
    _topic_gate_skip("first question", "s1")
    anchor = gate_state[("s1", "l1_5_topic_vec")]

    _set_embed(monkeypatch, [0.0, 1.0, 0.0])  # orthogonal — different topic
    assert _topic_gate_skip("unrelated question", "s1") is False
    assert gate_state[("s1", "l1_5_topic_vec")] != anchor, "a real search must re-anchor"


def test_topic_gate_skip_fails_open_when_embedding_unavailable(monkeypatch, gate_state):
    from hooks.prompt_hook import _topic_gate_skip
    _set_embed(monkeypatch, None)  # daemon down
    assert _topic_gate_skip("anything", "s1") is False, \
        "the gate must never be the reason retrieval stops"


# --------------------------------------------------------------------------
# [c] Source-utility prior
# --------------------------------------------------------------------------

@pytest.fixture
def priors_db(tmp_path):
    path = tmp_path / "durable.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, source_ref TEXT)")
    conn.executemany("INSERT INTO memories (id, source_ref) VALUES (?, ?)",
                     [(1, "analyser-session-arc"), (2, "genA-v4"), (3, None)])
    conn.commit()
    yield conn
    conn.close()


def test_apply_source_priors_demotes_only_the_flagged_source(monkeypatch, priors_db):
    import cairn.config as cfg
    from hooks.retrieval import _apply_source_priors
    monkeypatch.setattr(cfg, "SCORE_SOURCE_PRIORS", {"analyser-session-arc": -0.05})

    fused = [{"id": 1, "score": 0.80}, {"id": 2, "score": 0.80},
             {"id": 3, "score": 0.80}, {"id": 99, "score": 0.80}]
    _apply_source_priors(fused, priors_db)

    by_id = {r["id"]: r["score"] for r in fused}
    assert by_id[1] == pytest.approx(0.75), "analyser row demoted"
    assert by_id[2] == pytest.approx(0.80), "organic row untouched"
    assert by_id[3] == pytest.approx(0.80), "NULL source_ref untouched"
    assert by_id[99] == pytest.approx(0.80), "id absent from the table untouched"


def test_apply_source_priors_never_reorders_a_decisive_lead(monkeypatch, priors_db):
    """The prior demotes by ~a rank position; it must not exclude a strong row."""
    import cairn.config as cfg
    from hooks.retrieval import _apply_source_priors
    monkeypatch.setattr(cfg, "SCORE_SOURCE_PRIORS", {"analyser-session-arc": -0.05})

    fused = [{"id": 1, "score": 0.90}, {"id": 2, "score": 0.70}]
    _apply_source_priors(fused, priors_db)
    fused.sort(key=lambda x: x["score"], reverse=True)
    assert fused[0]["id"] == 1, "a strong analyser row still outranks a weak organic one"


def test_apply_source_priors_empty_inputs_are_noops(priors_db):
    from hooks.retrieval import _apply_source_priors
    _apply_source_priors([], priors_db)  # must not raise


# --------------------------------------------------------------------------
# [d] semantic_engaged boundary matrix
# --------------------------------------------------------------------------

@pytest.mark.parametrize("crm, ccm, expected", [
    (0.60, 0.10, True),   # clearly closer to the response than to the prompt
    (0.50, 0.00, False),  # below the similarity threshold
    (0.60, 0.58, False),  # close, but adds nothing over the prompt — echo
    (0.55, 0.50, True),   # exactly on both boundaries (>= on each)
])
def test_semantic_engaged_boundaries(crm, ccm, expected):
    from cairn.relevance import semantic_engaged
    assert semantic_engaged(crm, ccm, threshold=0.55, margin=0.05) is expected


def test_semantic_engaged_reads_config_defaults():
    from cairn.relevance import semantic_engaged
    assert semantic_engaged(0.90, 0.10) is True
    assert semantic_engaged(0.10, 0.00) is False


# --------------------------------------------------------------------------
# [e] _engagement_grade truth table
# --------------------------------------------------------------------------

@pytest.mark.parametrize("engaged, score, agent, expected", [
    (1, 0.8, None, 3),      # strong behavioural yes, no agent opinion
    (1, 0.8, 3, 3),         # agent agrees
    (1, 0.8, 1, None),      # agent says weak — agent wins, row dropped
    (1, 0.1, None, None),   # engaged but below min_pos — undecidable
    (0, None, None, 0),     # behavioural no, no agent opinion
    (0, None, 2, None),     # agent says relevant — outranks a lexical negative
    (None, -1.0, None, None),  # unscored sentinel
])
def test_engagement_grade_truth_table(engaged, score, agent, expected):
    from cairn.train_reranker import _engagement_grade
    assert _engagement_grade(engaged, score, agent) is expected


def test_engagement_min_pos_default_is_calibrated_to_lexical_scale():
    """0.5 sat above the 99th percentile of the lexical ratio and starved the pool."""
    from cairn.train_reranker import ENGAGEMENT_MIN_POS_DEFAULT
    assert ENGAGEMENT_MIN_POS_DEFAULT < 0.5


# --------------------------------------------------------------------------
# [f]/[g] Engagement DB fixtures — semantic second chance + label loading
# --------------------------------------------------------------------------

MEM_VEC = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
RESP_VEC = np.asarray([0.9, 0.4359, 0.0], dtype=np.float32)   # cos(mem) ~= 0.90
CTX_VEC = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)       # cos(mem) == 0.0

RESPONSE = "delta epsilon zeta eta theta"   # shares no vocabulary with the memory
MEM_TEXT = "alpha beta gamma"
CTX_TEXT = "iota kappa lambda"


@pytest.fixture
def engagement_dbs(tmp_path):
    from cairn.embeddings import to_blob
    eph = tmp_path / "eph.db"
    dur = tmp_path / "dur.db"

    e = sqlite3.connect(str(eph))
    e.execute("""CREATE TABLE memory_deliveries (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        turn_index INTEGER, memory_id INTEGER NOT NULL, context_text TEXT,
        context_vec BLOB, context_intent TEXT, ce_score REAL, served_rank INTEGER,
        grade INTEGER, hard_negative INTEGER DEFAULT 0, reranker_model TEXT,
        score_components TEXT, layer TEXT, scope TEXT, engaged INTEGER,
        engaged_score REAL, delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    e.execute("INSERT INTO memory_deliveries (session_id, memory_id, context_text) "
              "VALUES (?, ?, ?)", ("sess", 1, CTX_TEXT))
    e.commit()

    d = sqlite3.connect(str(dur))
    d.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT, "
              "topic TEXT, keywords TEXT, embedding BLOB)")
    d.execute("INSERT INTO memories VALUES (1, ?, 'topic', 'kw', ?)",
              (MEM_TEXT, to_blob(MEM_VEC)))
    d.commit()
    d.close()
    yield e, str(eph), str(dur)
    e.close()


def _patch_embed_by_prefix(monkeypatch, resp_vec=RESP_VEC):
    import cairn.embeddings as emb

    def fake(text, allow_slow=True):
        if text.startswith("delta"):
            return resp_vec
        if text.startswith("iota"):
            return CTX_VEC
        return None
    monkeypatch.setattr(emb, "embed", fake)


def test_apply_engagement_semantic_rescues_paraphrased_use(monkeypatch, engagement_dbs):
    """No shared vocabulary, but the response is semantically close to the memory."""
    from cairn.relevance import apply_engagement
    conn, eph, dur = engagement_dbs
    _patch_embed_by_prefix(monkeypatch)

    n = apply_engagement(RESPONSE, session_id="sess", eph_path=eph, durable_path=dur)
    assert n == 1
    engaged, score = conn.execute(
        "SELECT engaged, engaged_score FROM memory_deliveries WHERE id = 1").fetchone()
    assert engaged == 1, "semantic pass should rescue a lexically-invisible use"
    assert score == pytest.approx(0.90, abs=0.02), "score stores cos(response, memory)"


def test_apply_engagement_leaves_lexical_verdict_when_embedding_unavailable(
        monkeypatch, engagement_dbs):
    from cairn.relevance import apply_engagement
    import cairn.embeddings as emb
    conn, eph, dur = engagement_dbs
    monkeypatch.setattr(emb, "embed", lambda text, allow_slow=True: None)

    n = apply_engagement(RESPONSE, session_id="sess", eph_path=eph, durable_path=dur)
    assert n == 1
    engaged, _ = conn.execute(
        "SELECT engaged, engaged_score FROM memory_deliveries WHERE id = 1").fetchone()
    assert engaged == 0, "fail-soft: the lexical verdict stands"


def test_apply_engagement_semantic_respects_the_echo_margin(monkeypatch, engagement_dbs):
    """A response no closer to the memory than the prompt was is not engagement."""
    from cairn.relevance import apply_engagement
    conn, eph, dur = engagement_dbs
    # Response vector == context vector: cos(resp,mem) - cos(ctx,mem) == 0 < margin
    _patch_embed_by_prefix(monkeypatch, resp_vec=CTX_VEC)

    apply_engagement(RESPONSE, session_id="sess", eph_path=eph, durable_path=dur)
    engaged, _ = conn.execute(
        "SELECT engaged, engaged_score FROM memory_deliveries WHERE id = 1").fetchone()
    assert engaged == 0, "prompt-echo must not count as use"


def test_apply_engagement_survives_durable_db_without_embedding_column(tmp_path):
    """A durable schema predating `embedding` must still score lexically, not zero."""
    from cairn.relevance import apply_engagement
    eph, dur = tmp_path / "e.db", tmp_path / "d.db"
    e = sqlite3.connect(str(eph))
    e.execute("""CREATE TABLE memory_deliveries (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        memory_id INTEGER NOT NULL, context_text TEXT, grade INTEGER,
        engaged INTEGER, engaged_score REAL)""")
    e.execute("INSERT INTO memory_deliveries (session_id, memory_id, context_text) "
              "VALUES ('sess', 1, 'some context')")
    e.commit()
    d = sqlite3.connect(str(dur))  # NOTE: no embedding column
    d.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT, "
              "topic TEXT, keywords TEXT)")
    d.execute("INSERT INTO memories VALUES (1, 'sigmoid floor calibration', 'rerank', 'kw')")
    d.commit()
    d.close()

    n = apply_engagement("I calibrated the sigmoid floor", session_id="sess",
                         eph_path=str(eph), durable_path=str(dur))
    e.close()
    assert n == 1, "missing embedding column must degrade to lexical, not return 0"


def test_load_engagement_groups_shape_and_pruning(engagement_dbs):
    from cairn.train_reranker import load_engagement_groups
    conn, eph, dur = engagement_dbs
    # One positive and one negative under the SAME context -> a usable group.
    conn.execute("UPDATE memory_deliveries SET engaged = 1, engaged_score = 0.8 WHERE id = 1")
    conn.execute("INSERT INTO memory_deliveries (session_id, memory_id, context_text, "
                 "engaged, engaged_score) VALUES ('sess', 1, ?, 0, NULL)", (CTX_TEXT,))
    # A context with only a negative -> pruned (cannot form a pair).
    conn.execute("INSERT INTO memory_deliveries (session_id, memory_id, context_text, "
                 "engaged, engaged_score) VALUES ('sess', 1, 'lonely context', 0, NULL)")
    conn.commit()

    groups = load_engagement_groups(eph_path=eph, durable_path=dur)
    assert all(k.startswith("eng:") for k in groups), "prefix prevents rg-group collision"
    assert len(groups) == 1, "the negative-only group must be pruned"
    grades = {g for v in groups.values() for _, _, g in v}
    assert grades == {0, 3}


def test_load_engagement_groups_drops_agent_conflict_rows(engagement_dbs):
    """A behavioural yes that the agent graded as noise is undecidable, not a positive."""
    from cairn.train_reranker import load_engagement_groups
    conn, eph, dur = engagement_dbs
    conn.execute("UPDATE memory_deliveries SET engaged = 1, engaged_score = 0.8, "
                 "grade = 0 WHERE id = 1")
    conn.commit()
    assert load_engagement_groups(eph_path=eph, durable_path=dur) == {}


# --------------------------------------------------------------------------
# [h] Belt-and-braces: declaring without searching
# --------------------------------------------------------------------------

def _transcript(tmp_path, *entries):
    p = tmp_path / "t.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return str(p)


def _user(text):
    return {"timestamp": "2026-07-25T00:00:00Z", "message": {"role": "user", "content": text}}


def _tool_result():
    return {"timestamp": "2026-07-25T00:00:01Z", "message": {"role": "user", "content": [
        {"type": "tool_result", "content": "output"}]}}


def _bash(cmd):
    return {"timestamp": "2026-07-25T00:00:02Z", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}


def test_cairn_query_this_turn_detects_a_real_search(tmp_path):
    from hooks.hook_helpers import cairn_query_invoked_this_turn
    t = _transcript(tmp_path, _user("what did we decide?"),
                    _bash('query.py --semantic "prior decision"'))
    assert cairn_query_invoked_this_turn(t) is True


def test_cairn_query_this_turn_rejects_status_only_calls(tmp_path):
    """`query.py --stats` must not satisfy a 'did you search?' gate."""
    from hooks.hook_helpers import cairn_query_invoked_this_turn
    t = _transcript(tmp_path, _user("what did we decide?"), _bash("query.py --stats"))
    assert cairn_query_invoked_this_turn(t) is False


def test_cairn_query_this_turn_ignores_earlier_turns(tmp_path):
    """A search last turn is not compliance for this one."""
    from hooks.hook_helpers import cairn_query_invoked_this_turn
    t = _transcript(tmp_path, _user("first"), _bash('query.py --semantic "old"'),
                    _user("second"), _bash("ls -la"))
    assert cairn_query_invoked_this_turn(t) is False


def test_cairn_query_this_turn_boundary_survives_tool_results(tmp_path):
    """Tool results arrive as role=user; they must not be read as a new prompt."""
    from hooks.hook_helpers import cairn_query_invoked_this_turn
    t = _transcript(tmp_path, _user("what did we decide?"),
                    _bash('query.py --semantic "prior decision"'), _tool_result(),
                    _bash("echo continuing"))
    assert cairn_query_invoked_this_turn(t) is True


def test_cairn_query_this_turn_accepts_graph_knowledge_probe(tmp_path):
    from hooks.hook_helpers import cairn_query_invoked_this_turn
    t = _transcript(tmp_path, _user("who calls this?"),
                    _bash("cairn-graph --knowledge hybrid_search"))
    assert cairn_query_invoked_this_turn(t) is True


def test_cairn_query_this_turn_missing_transcript_is_false(tmp_path):
    from hooks.hook_helpers import cairn_query_invoked_this_turn
    assert cairn_query_invoked_this_turn(str(tmp_path / "nope.jsonl")) is False
