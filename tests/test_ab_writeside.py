"""Write-side generation A/B harness — cairn/ab_writeside.py (mechanics).

All LLM/embedder calls are injected, so these test the harness logic (prompt
shaping, parse, position-swap aggregation, metrics) without a model."""
from __future__ import annotations

import json
import os

import pytest

from cairn import ab_writeside as ab


# ---- 3b: generation prompts --------------------------------------------------
def test_build_generation_prompt_variants_differ():
    a = ab.build_generation_prompt("A", "TRANSCRIPT_BODY")
    b = ab.build_generation_prompt("B", "TRANSCRIPT_BODY")
    assert "TRANSCRIPT_BODY" in a and "TRANSCRIPT_BODY" in b
    assert a != b
    # prompt-B exercises the levers in priority order
    assert "SUPPRESSION" in b and "FINDABILITY" in b and "SELF-SUFFICIENCY" in b
    assert b.index("SUPPRESSION") < b.index("FINDABILITY") < b.index("SELF-SUFFICIENCY")


def test_build_generation_prompt_unknown_variant():
    with pytest.raises(ValueError):
        ab.build_generation_prompt("Z", "x")


def test_generate_memory_set_parses_and_stamps():
    block = ("[cm]: # '" +
             json.dumps({"e": [{"t": "fact", "to": "topic", "c": "a durable fact"}],
                         "ok": True, "ctx": "s", "kw": ["k"]}) + "'")
    calls = []
    def fake_llm(prompt, *, model=None, **kw):
        calls.append(prompt)
        return block
    out = ab.generate_memory_set("cleaned text", "B", call_llm=fake_llm)
    assert len(out) == 1
    assert out[0]["content"] == "a durable fact"
    assert out[0]["generation_prompt_version"] == "genB-v1"
    assert "cleaned text" in calls[0]


def test_generate_memory_set_no_block_returns_empty():
    out = ab.generate_memory_set("x", "A", call_llm=lambda p, **k: "no memory here")
    assert out == []


# ---- 3c: replay --------------------------------------------------------------
def test_replay_session_runs_both_prompts(tmp_path):
    t = tmp_path / "sess.jsonl"
    t.write_text(json.dumps({"type": "user", "message": {"role": "user",
                 "content": "please add a feature"}}) + "\n" +
                 json.dumps({"type": "assistant", "message": {"role": "assistant",
                 "content": "done, added the feature"}}) + "\n")
    seen = {"A": 0, "B": 0}
    def fake_llm(prompt, *, model=None, **kw):
        variant = "B" if "SUPPRESSION" in prompt else "A"
        seen[variant] += 1
        return ("[cm]: # '" + json.dumps(
            {"e": [{"t": "fact", "to": variant, "c": f"fact from {variant}"}],
             "ok": True, "ctx": "s", "kw": ["k"]}) + "'")
    r = ab.replay_session(str(t), call_llm=fake_llm)
    assert seen == {"A": 1, "B": 1}
    assert r["A"][0]["content"] == "fact from A"
    assert r["B"][0]["content"] == "fact from B"
    assert r["session"] == "sess.jsonl"


# ---- 3c: judge ---------------------------------------------------------------
def test_parse_judge_verdict():
    assert ab.parse_judge_verdict('{"overall":"1","findability":"2"}') == \
        {"overall": "1", "findability": "2"}
    assert ab.parse_judge_verdict("garbage no json") == {}
    assert ab.parse_judge_verdict("") == {}


def _verdict(**dims):
    base = {"findability": "tie", "self_sufficiency": "tie", "fitness": "tie", "overall": "tie"}
    base.update(dims)
    return json.dumps(base)


def test_judge_session_consistent_winner_survives_swap():
    # Judge prefers whichever SET contains the marker 'GOLD' -> set A always wins,
    # regardless of position. Both swapped runs must agree -> overall A.
    set_a = [{"type": "fact", "topic": "t", "content": "GOLD load-bearing memory"}]
    set_b = [{"type": "fact", "topic": "t", "content": "weak memory"}]
    def fake_judge(prompt, *, model=None, **kw):
        # Which set number contains GOLD in this prompt?
        gold_set = "1" if prompt.index("GOLD") < prompt.index("weak memory") else "2"
        return _verdict(overall=gold_set, findability=gold_set,
                        self_sufficiency=gold_set, fitness=gold_set)
    v = ab.judge_session(set_a, set_b, call_llm=fake_judge)
    assert v["overall"] == "A"
    assert v["findability"] == "A" and v["fitness"] == "A"


def test_judge_session_position_bias_scores_tie():
    # A judge that ALWAYS says "1" (pure position bias) must resolve to tie, not a
    # spurious winner — this is what the position-swap guard exists to catch.
    set_a = [{"type": "fact", "topic": "t", "content": "alpha"}]
    set_b = [{"type": "fact", "topic": "t", "content": "beta"}]
    def biased(prompt, *, model=None, **kw):
        return _verdict(overall="1", findability="1", self_sufficiency="1", fitness="1")
    v = ab.judge_session(set_a, set_b, call_llm=biased)
    assert v["overall"] == "tie"
    assert v["findability"] == "tie"


# ---- 3d: metrics -------------------------------------------------------------
def _keyword_embed(texts):
    """Deterministic fake: one-hot over a small vocabulary keyed by first vocab
    word found, so identical topics -> cos 1.0, different -> cos 0.0."""
    vocab = ["alpha", "beta", "gamma", "delta"]
    out = []
    for t in texts:
        v = [0.0] * len(vocab)
        for i, w in enumerate(vocab):
            if w in t.lower():
                v[i] = 1.0
        out.append(v or [0.0] * len(vocab))
    return out


def test_dedup_rate():
    mem = [{"content": "alpha thing"}, {"content": "gamma thing"}]
    existing = ["alpha already stored"]
    # alpha is a dup (cos 1.0), gamma is novel -> 1/2
    assert ab.dedup_rate(mem, existing, embed=_keyword_embed, threshold=0.85) == 0.5
    assert ab.dedup_rate([], existing, embed=_keyword_embed) == 0.0
    assert ab.dedup_rate(mem, [], embed=_keyword_embed) == 0.0


def test_findability_backtest():
    mem = {"content": "alpha configuration detail"}
    queries = ["how to alpha", "what is beta"]   # 1 of 2 matches the memory topic
    assert ab.findability_backtest(mem, queries, embed=_keyword_embed, threshold=0.5) == 0.5
    assert ab.findability_backtest(mem, [], embed=_keyword_embed) == 0.0


def test_self_sufficiency_coldread():
    yes = ab.self_sufficiency_coldread(
        {"content": "x"}, call_llm=lambda p, **k: '{"self_sufficient": true}')
    no = ab.self_sufficiency_coldread(
        {"content": "x"}, call_llm=lambda p, **k: '{"self_sufficient": false}')
    bad = ab.self_sufficiency_coldread(
        {"content": "x"}, call_llm=lambda p, **k: "no json")
    assert yes is True and no is False and bad is None
    assert ab.self_sufficiency_coldread({"content": ""}, call_llm=lambda p, **k: "x") is None


def test_cohort_tally():
    results = [{"overall": "A"}, {"overall": "A"}, {"overall": "B"}, {"overall": "tie"}]
    assert ab._cohort_tally(results) == {"A": 2, "B": 1, "tie": 1}


def test_dry_run_lists_without_model(tmp_path, capsys):
    t = tmp_path / "s.jsonl"
    t.write_text("{}\n")
    rc = ab.main(["replay", str(t), "--dry-run"])
    assert rc == 0
    assert str(t) in capsys.readouterr().out
