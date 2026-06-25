#!/usr/bin/env python3
"""Write-side generation A/B harness — docs/spec-memory-relevance-grading.md Part B.

The biggest lever on injected-memory quality is generating fewer, more findable,
more self-sufficient memories in the first place. Online A/B fails (write->surface
->grade is slow + sparse, and usefulness-when-surfaced confounds writer x
retriever), so we A/B OFFLINE: replay the transcript corpus through generation
prompt-A vs prompt-B, producing two memory sets from identical inputs, then judge
them head-to-head.

Pieces (all LLM/embedder calls are injectable so the mechanics test without a
model):
  * build_generation_prompt / generate_memory_set  — prompt-A (baseline) vs
      prompt-B (the levers, in priority order: suppression > findability(qf) >
      self-sufficiency > dedup). 3b.
  * replay_session / replay_corpus                  — clean a transcript via
      session_extract, run both prompts -> two memory sets. 3c.
  * judge_session                                   — Opus 4.8, BLIND (Set 1/Set 2,
      never told which is "ours"), POSITION-SWAPPED (both orders; disagreement =
      position bias = tie), PAIRWISE, rubric = findability/self-sufficiency/fitness.
      A/B unit = the session cohort, never per-memory (per-memory collides under
      dedup). 3c.
  * dedup_rate / findability_backtest / self_sufficiency_coldread — the fast
      write-time metrics. 3d.

Nothing here runs on the hot path; it is an offline analysis tool (CLI below).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Callable, Optional

# Generation-prompt versions stamped onto each produced set (the provenance join
# key from step 3a — config.GENERATION_PROMPT_VERSION is the live production one).
GEN_PROMPT_VERSIONS = {"A": "genA-v1", "B": "genB-v1"}

# The judge is the BEST model, made independent BY CONSTRUCTION (blind + position-
# swapped), never a weaker local model (spec standing constraint).
JUDGE_MODEL = "claude-opus-4-8"


# --- 3b: generation prompts ---------------------------------------------------
_PROMPT_A = """\
You are distilling a development session transcript into durable memory entries
for a persistent memory system.

Read the transcript below and emit the memories worth keeping. Output ONLY a
single memory block in this exact link-definition format on the last line:

[cm]: # '{"e":[{"t":"TYPE","to":"topic","c":"content"}],"ok":true,"ctx":"s","kw":["k1","k2"]}'

TYPE is one of: decision, preference, fact, correction, person, project, skill,
workflow. One entry per durable nugget. Keep each content line to one sentence.

TRANSCRIPT:
{transcript}
"""

_PROMPT_B = """\
You are distilling a development session transcript into durable memory entries
for a persistent memory system. A future session will read these with ZERO other
context. Apply these rules IN PRIORITY ORDER:

1. SUPPRESSION (what NOT to write) — do not emit self-referential meta ("cairn has
   no memory of X", "should be captured when shared"), ephemeral state snapshots,
   or anything already obvious from the code/git. When in doubt, omit. Fewer,
   higher-value entries beat many weak ones.
2. FINDABILITY — write each entry as the ANSWER to the questions a future session
   would ask. Put the distinctive search terms (identifiers, file paths, flags) in
   the content and keywords, not generic tokens.
3. SELF-SUFFICIENCY — each content line must be actionable WITHOUT the transcript:
   include the what, the why, and enough context to act. No dangling references.
4. REDUNDANCY-AWARENESS — collapse near-duplicate nuggets into one entry; do not
   restate the same fact in multiple entries.

Output ONLY a single memory block on the last line:

[cm]: # '{"e":[{"t":"TYPE","to":"topic","c":"content","kw":["k1","k2"]}],"ok":true,"ctx":"s","kw":["k1","k2"]}'

TYPE is one of: decision, preference, fact, correction, person, project, skill,
workflow.

TRANSCRIPT:
{transcript}
"""

_PROMPTS = {"A": _PROMPT_A, "B": _PROMPT_B}


def build_generation_prompt(variant: str, cleaned_transcript: str) -> str:
    """Render the generation prompt for variant 'A' or 'B' with the transcript."""
    tmpl = _PROMPTS.get(variant)
    if tmpl is None:
        raise ValueError(f"unknown generation variant {variant!r}")
    return tmpl.replace("{transcript}", cleaned_transcript or "")


def _default_call_llm(prompt: str, *, model: Optional[str] = None,
                      timeout: int = 600) -> str:
    """Reuse the analyser's read-only `claude -p` invocation (CAIRN_MODE=read-only
    so generation doesn't itself trip the Stop hook)."""
    from cairn.analyser import call_llm
    return call_llm(prompt, timeout=timeout, model=model)


def generate_memory_set(cleaned_transcript: str, variant: str, *,
                        call_llm: Callable[..., str] = _default_call_llm,
                        model: Optional[str] = None) -> list[dict[str, Any]]:
    """Generate one memory set from a cleaned transcript using prompt `variant`.
    Parses the model's [cm] block into entry dicts; stamps each with its
    generation-prompt version. Returns [] if the model emits no parseable block."""
    from hooks.parser import parse_memory_block
    prompt = build_generation_prompt(variant, cleaned_transcript)
    raw = call_llm(prompt, model=model)
    entries = parse_memory_block(raw or "").entries or []
    gv = GEN_PROMPT_VERSIONS.get(variant, variant)
    for e in entries:
        e["generation_prompt_version"] = gv
    return entries


# --- 3c: replay --------------------------------------------------------------
def replay_session(jsonl_path: str, *,
                   call_llm: Callable[..., str] = _default_call_llm,
                   model: Optional[str] = None) -> dict[str, Any]:
    """Clean one session transcript and run BOTH prompts over identical input.
    Returns {session, transcript, A:[...], B:[...]}. The cleaned text is the
    controlled variable — both sets see byte-identical input."""
    from cairn.session_extract import load_turns, render
    turns = load_turns(jsonl_path)
    cleaned = render(turns)
    return {
        "session": os.path.basename(jsonl_path),
        "transcript": jsonl_path,
        "A": generate_memory_set(cleaned, "A", call_llm=call_llm, model=model),
        "B": generate_memory_set(cleaned, "B", call_llm=call_llm, model=model),
    }


def discover_sessions(roots: Optional[list[str]] = None) -> list[str]:
    """Find session JSONLs under the transcript roots (~/.claude/projects/*)."""
    import glob
    if roots is None:
        roots = [os.path.expanduser("~/.claude/projects")]
    out: list[str] = []
    for root in roots:
        out.extend(sorted(glob.glob(os.path.join(root, "*", "*.jsonl"))))
    return out


# --- 3c: judge (blind + position-swapped + pairwise) -------------------------
def _format_set(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "(no memories)"
    lines = []
    for i, m in enumerate(memories, 1):
        t = m.get("type") or m.get("t") or "?"
        to = m.get("topic") or m.get("to") or ""
        c = m.get("content") or m.get("c") or ""
        lines.append(f"{i}. [{t}] {to}: {c}")
    return "\n".join(lines)


def build_judge_prompt(set1: list[dict], set2: list[dict], context: str = "") -> str:
    """A BLIND pairwise judge prompt — the two sets are 'Set 1' and 'Set 2', never
    labelled by which generation prompt produced them, so the judge can't favour
    'ours'."""
    ctx = f"\nSESSION CONTEXT (what the session was about):\n{context}\n" if context else ""
    return f"""\
You are comparing two sets of memory entries distilled from the SAME development
session. A future session will read the winning set with no other context.
{ctx}
Judge which set is better on three axes:
  - findability: would the right entry surface for the questions a future session
    would actually ask?
  - self_sufficiency: is each entry actionable without the original transcript?
  - fitness: does the set capture the durable, high-value knowledge and suppress
    noise / redundancy / ephemeral snapshots?

SET 1:
{_format_set(set1)}

SET 2:
{_format_set(set2)}

Respond with ONLY a JSON object (no prose):
{{"findability":"1|2|tie","self_sufficiency":"1|2|tie","fitness":"1|2|tie","overall":"1|2|tie","reason":"one sentence"}}
"""


def parse_judge_verdict(raw: str) -> dict[str, str]:
    """Extract the judge's JSON verdict. Returns {} if unparseable."""
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return {}
    return {k: str(v) for k, v in d.items()} if isinstance(d, dict) else {}


_DIMS = ("findability", "self_sufficiency", "fitness", "overall")


def _swap_position_label(label: str) -> str:
    """In the swapped run, '1' and '2' refer to the opposite set."""
    return {"1": "2", "2": "1"}.get(label, label)


def judge_session(set_a: list[dict], set_b: list[dict], context: str = "", *,
                  call_llm: Callable[..., str] = _default_call_llm,
                  model: str = JUDGE_MODEL) -> dict[str, Any]:
    """Judge set A vs set B, BLIND and POSITION-SWAPPED. Run 1: A=Set1, B=Set2.
    Run 2: B=Set1, A=Set2 (positions swapped). Map both verdicts back to A/B; per
    dimension, A wins only if BOTH runs agree A is better — disagreement across the
    swap is position bias and scores as a tie. This removes position bias by
    construction rather than trusting the model not to have it."""
    raw1 = call_llm(build_judge_prompt(set_a, set_b, context), model=model)
    raw2 = call_llm(build_judge_prompt(set_b, set_a, context), model=model)
    v1 = parse_judge_verdict(raw1)
    v2 = parse_judge_verdict(raw2)

    def _to_ab(verdict: dict, swapped: bool) -> dict[str, str]:
        out = {}
        for dim in _DIMS:
            lab = verdict.get(dim, "tie")
            if swapped:
                lab = _swap_position_label(lab)
            out[dim] = {"1": "A", "2": "B"}.get(lab, "tie")
        return out

    ab1 = _to_ab(v1, swapped=False)
    ab2 = _to_ab(v2, swapped=True)
    result: dict[str, Any] = {"runs": [v1, v2]}
    for dim in _DIMS:
        result[dim] = ab1[dim] if ab1[dim] == ab2[dim] else "tie"
    return result


# --- 3d: metrics -------------------------------------------------------------
def _cos(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _default_embed(texts: list[str]) -> list[list[float]]:
    """Embed via the local embedder (no LLM cost). Coerces numpy arrays to lists
    so _cos works the same on real and test vectors."""
    from cairn.embeddings import embed_batch
    vecs = embed_batch(texts) or []
    return [list(v) for v in vecs]


def dedup_rate(memory_set: list[dict], existing_texts: list[str], *,
               embed: Callable[[list[str]], list[list[float]]] = _default_embed,
               threshold: float = 0.85) -> float:
    """Fraction of `memory_set` that is a near-duplicate (cosine >= threshold) of
    something already in `existing_texts`. Lower is better (more novel). Returns
    0.0 if either side is empty."""
    texts = [(m.get("content") or m.get("c") or "") for m in memory_set]
    texts = [t for t in texts if t]
    if not texts or not existing_texts:
        return 0.0
    vecs = embed(texts)
    evecs = embed(existing_texts)
    dup = 0
    for v in vecs:
        if any(_cos(v, e) >= threshold for e in evecs):
            dup += 1
    return dup / len(texts)


def findability_backtest(memory: dict, followup_queries: list[str], *,
                         embed: Callable[[list[str]], list[list[float]]] = _default_embed,
                         threshold: float = 0.45) -> float:
    """Fraction of the historical follow-up queries (the questions that ACTUALLY
    came later in this memory's own transcript) for which the memory is similar
    enough (cosine >= threshold) to have been retrieved. Higher is better."""
    text = memory.get("content") or memory.get("c") or ""
    qs = [q for q in (followup_queries or []) if q]
    if not text or not qs:
        return 0.0
    mvec = embed([text])[0]
    qvecs = embed(qs)
    hits = sum(1 for qv in qvecs if _cos(mvec, qv) >= threshold)
    return hits / len(qs)


_COLDREAD_PROMPT = """\
A future AI session is shown ONLY this one memory line, with no other context:

  {memory}

Could it act on this correctly and unambiguously — does it contain the what, the
why, and enough context to be useful on its own? Answer with ONLY a JSON object:
{{"self_sufficient": true|false, "reason": "one sentence"}}
"""


def self_sufficiency_coldread(memory: dict, *,
                              call_llm: Callable[..., str] = _default_call_llm,
                              model: str = JUDGE_MODEL) -> Optional[bool]:
    """Cold-read test: a fresh model judges whether the memory is actionable with
    no transcript. Returns True/False, or None if the verdict is unparseable."""
    text = memory.get("content") or memory.get("c") or ""
    if not text:
        return None
    raw = call_llm(_COLDREAD_PROMPT.replace("{memory}", text), model=model)
    v = parse_judge_verdict(raw)
    if "self_sufficient" not in v:
        return None
    return str(v["self_sufficient"]).lower() in ("true", "1", "yes")


# --- CLI ---------------------------------------------------------------------
def _cohort_tally(results: list[dict]) -> dict[str, int]:
    """Aggregate per-session overall winners into a cohort tally (the A/B unit)."""
    tally = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        tally[r.get("overall", "tie")] = tally.get(r.get("overall", "tie"), 0) + 1
    return tally


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Write-side generation A/B harness (offline).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("replay", help="Replay sessions through prompt-A vs prompt-B.")
    sp.add_argument("paths", nargs="*", help="Session JSONLs (default: discover under ~/.claude/projects)")
    sp.add_argument("--limit", type=int, default=5)
    sp.add_argument("--model", default=None)
    sp.add_argument("--dry-run", action="store_true", help="List sessions, don't call the model.")

    jp = sub.add_parser("ab", help="Replay + judge (Opus, blind, position-swapped).")
    jp.add_argument("paths", nargs="*")
    jp.add_argument("--limit", type=int, default=5)
    jp.add_argument("--model", default=None, help="generation model")
    jp.add_argument("--judge-model", default=JUDGE_MODEL)
    jp.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    paths = args.paths or discover_sessions()
    paths = paths[: args.limit]

    if args.dry_run:
        print(f"{len(paths)} session(s):")
        for p in paths:
            print(" ", p)
        return 0

    if args.cmd == "replay":
        for p in paths:
            r = replay_session(p, model=args.model)
            print(json.dumps({"session": r["session"], "A": len(r["A"]), "B": len(r["B"])}))
        return 0

    if args.cmd == "ab":
        results = []
        for p in paths:
            r = replay_session(p, model=args.model)
            verdict = judge_session(r["A"], r["B"], model=args.judge_model)
            verdict["session"] = r["session"]
            results.append(verdict)
            print(json.dumps({"session": r["session"], "overall": verdict.get("overall"),
                              "findability": verdict.get("findability"),
                              "self_sufficiency": verdict.get("self_sufficiency"),
                              "fitness": verdict.get("fitness")}))
        print("COHORT:", json.dumps(_cohort_tally(results)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
