"""Jev quality gate for [cm] entries — flag at capture, enforce at the next turn.

At capture (turn N) a detached process asks Jev three kinds of question:

- per entry: does any detail (number, name, path, direction of a conclusion)
  differ from the turn? Asked with the turn lines most related to the entry
  placed beside it. Logged alongside: durable vs transient, the entry's type,
  and whether it reads without the conversation.
- per block (empty blocks included): is it generic/lazy or missing a finding,
  and was anything durable left out? Logged alongside: did the user correct
  the assistant, and the turn's task type.
- per delivered memory: does this turn show it is now wrong? (stale candidates)

Locally, an entry that trips secret redaction is flagged too.

Enforced flags (detail mismatch, lazy/incomplete block, secret) become a
one-shot obligation: turn N+1 is told on its prompt, and its [cm] must add a
corrected entry with the same topic or rebut in a "jv" map —
"jv":{"<topic>":"supported: <where>"} or "jv":{"_block":"complete: <why>"}.
An unanswered obligation re-prompts once, then clears. Nothing is deleted.
Everything else is logged only, until spot-check labels show it is right.

Thresholds from the scratchpad evaluation (2026-10-24): mismatch noul >= 0.8
caught 54% of on-topic edits at a 2.5% false-flag rate (AUC 0.87 with
adjacent evidence); block rubric caught boilerplate at AUC 1.0 and dropped
entries at 18/35 with 1/50 false; omission noul separated lazy empty blocks.
Every score and outcome goes to JEV_GATE_LOG; `report` and `spotcheck` read it.

Off with CAIRN_JEV_GATE=0. Fails open everywhere.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Optional

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = os.environ.get("CAIRN_JEV_MODEL", "typesafe/jev-1.13")
THRESHOLD = float(os.environ.get("CAIRN_JEV_GATE_THRESHOLD", "0.8"))
OMISSION_THRESHOLD = 0.5
MAX_ENTRIES = 8
MAX_DELIVERED = 8
STATE_CAP = 40000
CALL_TIMEOUT_S = 20
STATE_KEY = "jev_obligation"
JEV_GATE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_gate_log.jsonl")
CM_TYPES = ("fact", "decision", "correction", "preference", "skill", "workflow", "project", "person")

ENTRY_Q = {
    "verdict": {"type": "choice",
                "instructions": ("Check the ENTRY UNDER REVIEW detail by detail against the turn above: every "
                                 "number, name, file path, and the direction of every conclusion "
                                 "(does/does not, faster/slower, chosen/rejected)."),
                "criteria": {"supported": "every detail matches the turn",
                             "inferred": "a reasonable conclusion from the turn, no detail contradicts it",
                             "unsupported": "a detail differs from or contradicts the turn, or is absent from it",
                             "generic_meta": "no turn-specific content"}},
    "mismatch": {"type": "noul",
                 "instructions": ("Does any specific detail in the ENTRY UNDER REVIEW (a number, name, path, "
                                  "or the direction of a conclusion) differ from what the turn shows?")},
    # logged only
    "durable": {"type": "choice", "instructions": "How long does the ENTRY UNDER REVIEW stay true and useful?",
                "criteria": {"durable": "a lasting fact, decision, correction, preference or technique",
                             "transient": "momentary state: process ids, what is staged now, a passing status"}},
    "kind": {"type": "choice", "instructions": "Which memory type best fits the ENTRY UNDER REVIEW?",
             "criteria": {
                 "fact": "a verified fact about a system, environment or value",
                 "decision": "a choice made, ideally with the alternatives rejected",
                 "correction": "something that was wrong, and how it was put right",
                 "preference": "how the user wants things done",
                 "skill": "an exact command or technique that works",
                 "workflow": "a sequence of steps and when to use it",
                 "project": "the state or blockers of a piece of work",
                 "person": "about a person"}},
    "standalone": {"type": "noul",
                   "instructions": ("Could someone with no access to this conversation understand the ENTRY UNDER "
                                    "REVIEW without guessing what 'this', 'the fix' or similar references mean?")},
}

BLOCK_Q = {
    "quality": {"type": "choice",
                "instructions": ("A memory block is the only record a future session will have of this turn. "
                                 "Good entries are specific to the turn and capture its durable findings. "
                                 "Classify the block's main problem, if any."),
                "criteria": {"good": "specific and covers the turn's durable findings",
                             "generic_or_lazy": "empty, vague, boilerplate or restates instructions despite a substantive turn",
                             "misses_finding": "entries are fine but a durable finding from the turn is not captured"}},
    "omission": {"type": "noul",
                 "instructions": ("Does the turn contain a durable finding (fact, decision, correction, preference "
                                  "or technique worth remembering in a future session) that the memory block does "
                                  "not capture?")},
    # logged only
    "user_correction": {"type": "noul",
                        "instructions": "In this turn, did the user correct, contradict or redirect the assistant?"},
    "task": {"type": "choice", "instructions": "What kind of work was this turn?",
             "criteria": {"debug": "finding or fixing a fault", "build": "writing or changing code or config",
                          "design": "weighing options or planning", "research": "looking things up or comparing",
                          "ops": "running, checking or administering systems", "chat": "conversation or explanation"}},
}

_SECRET = [re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{28,}\b"),
           re.compile(r"\b(sk|gho|ghp|xox[bap])[-_][A-Za-z0-9_-]{10,}\b"),
           re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*\S+")]


def enabled() -> bool:
    return os.environ.get("CAIRN_JEV_GATE", "1") not in ("0", "false", "no", "")


def redact(s: str) -> str:
    """Strip likely secrets before anything leaves the machine."""
    for p in _SECRET:
        s = p.sub("[REDACTED]", s)
    return s


def has_secret(s: str) -> bool:
    return redact(s) != s


def _key() -> Optional[str]:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    try:
        with open(os.path.expanduser("~/.pi/agent/auth.json")) as f:
            return json.load(f)["openrouter"]["key"]
    except Exception:
        return None


def _ask(key: str, state: str, questions: dict) -> tuple[Optional[dict], float, float]:
    """(answers, cost_usd, latency_s); answers is None on any failure."""
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S).read())
        cost = float((r.get("usage") or {}).get("cost") or 0.0)
        return r.get("answers"), cost, round(time.time() - t0, 2)
    except Exception:
        return None, 0.0, round(time.time() - t0, 2)


def unsupported_p(verdict: Optional[dict]) -> float:
    pr = (verdict or {}).get("probabilities") or {}
    return 1.0 - pr.get("supported", 0.0) - pr.get("inferred", 0.0)


def _noul(answers: Optional[dict], q: str) -> Optional[float]:
    v = ((answers or {}).get(q) or {}).get("noul")
    return round(v, 3) if isinstance(v, (int, float)) else None


def _choice(answers: Optional[dict], q: str) -> Optional[str]:
    return ((answers or {}).get(q) or {}).get("choice")


# --- turn state -------------------------------------------------------------

def turn_state(transcript_path: str, per_result: int = 1500) -> str:
    """The last real user turn onward, as text: prompt, assistant text, tool
    calls and (truncated) results. Host-agnostic via the transcript adapter."""
    from hooks.hook_helpers import _is_real_user_turn
    from hooks.transcript_adapter import iter_normalized_entries
    entries = list(iter_normalized_entries(transcript_path))
    start = 0
    for i in range(len(entries) - 1, -1, -1):
        msg = entries[i].get("message") or {}
        if isinstance(msg, dict) and msg.get("role") == "user" and _is_real_user_turn(msg):
            start = i
            break
    while True:
        lines = []
        for entry in entries[start:]:
            msg = entry.get("message") or {}
            content = msg.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    who = "USER PROMPT" if msg.get("role") == "user" else "assistant"
                    lines.append(f"[{who}] {b['text']}")
                elif b.get("type") == "tool_use":
                    lines.append(f"[tool call] {b.get('name')} {json.dumps(b.get('input'))[:400]}")
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    c = c if isinstance(c, str) else "\n".join(
                        x.get("text", "") for x in (c or []) if isinstance(x, dict))
                    lines.append(f"[tool result] {c[:per_result]}"
                                 + (" ...[truncated]" if len(c) > per_result else ""))
        s = "\n".join(lines)
        if len(s) <= STATE_CAP or per_result <= 150:
            return s[-STATE_CAP:]
        per_result //= 2


def turn_number(transcript_path: str) -> int:
    """1-based index of the last real user turn: the position used to test
    whether first-pass quality changes over a session."""
    from hooks.hook_helpers import _is_real_user_turn
    from hooks.transcript_adapter import iter_normalized_entries
    n = 0
    for entry in iter_normalized_entries(transcript_path):
        msg = entry.get("message") or {}
        if isinstance(msg, dict) and msg.get("role") == "user" and _is_real_user_turn(msg):
            n += 1
    return n


def related_lines(turn: str, content: str, n: int = 8) -> list[str]:
    """Turn lines sharing the most words with the entry. Placing these beside
    the entry raised AUC 0.82 -> 0.87 in the lever test."""
    words = set(re.findall(r"[\w.-]{3,}", content.lower()))
    lines = [l for l in turn.splitlines() if l.strip()]
    return [l[:400] for l in sorted(lines, key=lambda l: -len(
        words & set(re.findall(r"[\w.-]{3,}", l.lower()))))[:n]]


def _entry_line(e: dict) -> str:
    return f"{e.get('type')}/{e.get('topic')}: {e.get('content')}"


def delivered_memories(session_id: str) -> list[dict]:
    """Memories injected at this session's latest prompt, for the stale check."""
    try:
        from hooks.hook_helpers import get_conn, get_ephemeral_conn
        with get_ephemeral_conn() as eph:
            row = eph.execute("SELECT MAX(turn_index) FROM memory_deliveries WHERE session_id = ?",
                              (session_id,)).fetchone()
            if not row or row[0] is None:
                return []
            ids = [i for (i,) in eph.execute(
                "SELECT DISTINCT memory_id FROM memory_deliveries WHERE session_id = ? AND turn_index = ?",
                (session_id, row[0]))][:MAX_DELIVERED]
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        with get_conn() as conn:
            return [{"id": i, "type": t, "topic": to, "content": c} for i, t, to, c in conn.execute(
                f"SELECT id, type, topic, content FROM memories WHERE id IN ({q}) AND deleted_at IS NULL", ids)]
    except Exception:
        return []


# --- capture side (turn N) --------------------------------------------------

def assess(session_id: str, transcript_path: str, entries: list[dict]) -> list[dict]:
    """Score the turn's entries, block and delivered memories; save enforced
    flags as this session's obligation; log every score."""
    key = _key()
    turn = redact(turn_state(transcript_path))
    if not key or not turn.strip():
        return []
    entries = [e for e in (entries or []) if e.get("content")][:MAX_ENTRIES]
    block_text = "\n".join(f"{i+1}. {redact(_entry_line(e))}" for i, e in enumerate(entries)) \
        or "(no entries: the block is empty)"
    memories = delivered_memories(session_id)

    jobs = [("entry", e, (f"{turn}\n\nTURN LINES MOST RELATED TO THE ENTRY:\n"
                          + "\n".join(related_lines(turn, str(e.get("content"))))
                          + f"\n\nENTRY UNDER REVIEW: {redact(_entry_line(e))}"), ENTRY_Q) for e in entries]
    jobs.append(("block", None, f"{turn}\n\nMEMORY BLOCK the assistant wrote for this turn:\n{block_text}", BLOCK_Q))
    if memories:
        jobs.append(("stale", None, turn, {f"m{m['id']}": {"type": "noul", "instructions": (
            "Does the turn above show that this stored memory is now wrong or outdated? Memory: "
            + redact(_entry_line(m))[:1200])} for m in memories}))

    with cf.ThreadPoolExecutor(min(8, len(jobs))) as ex:
        results = list(ex.map(lambda j: (j, *_ask(key, j[2], j[3])), jobs))

    scores, block, stale, cost, lat, failed = [], None, [], 0.0, 0.0, 0
    for (kind, e, _, _), ans, c, l in results:
        cost += c
        lat = max(lat, l)
        if ans is None:
            failed += 1
            continue
        if kind == "entry":
            mismatch = _noul(ans, "mismatch") or 0.0
            scores.append({"type": e.get("type"), "topic": e.get("topic"), "content": e.get("content"),
                           "p": mismatch, "verdict_p": round(unsupported_p(ans.get("verdict")), 3),
                           "choice": _choice(ans, "verdict"), "durable": _choice(ans, "durable"),
                           "kind_pred": _choice(ans, "kind"), "standalone": _noul(ans, "standalone"),
                           "secret": has_secret(str(e.get("content"))), "flagged": mismatch >= THRESHOLD})
        elif kind == "block":
            q = _choice(ans, "quality")
            omission = _noul(ans, "omission") or 0.0
            lazy = (omission >= OMISSION_THRESHOLD) if not entries else q in ("generic_or_lazy", "misses_finding")
            block = {"quality": q, "probs": ((ans.get("quality") or {}).get("probabilities")),
                     "omission": omission, "user_correction": _noul(ans, "user_correction"),
                     "task": _choice(ans, "task"), "empty": not entries, "flagged": lazy,
                     "uncaptured_correction": bool((_noul(ans, "user_correction") or 0) >= 0.7
                                                   and not any(x.get("type") == "correction" for x in entries))}
        else:
            stale = sorted(({"id": int(k[1:]), "p": _noul(ans, k)} for k in ans), key=lambda s: -(s["p"] or 0))

    # Secrets are checked locally even if the entry call failed.
    for e in entries:
        if has_secret(str(e.get("content"))) and not any(s["topic"] == e.get("topic") for s in scores):
            scores.append({"type": e.get("type"), "topic": e.get("topic"), "content": e.get("content"),
                           "p": None, "secret": True, "flagged": False})

    flags = [{"kind": "secret", "type": s["type"], "topic": s["topic"], "content": redact(str(s["content"]))}
             for s in scores if s.get("secret")]
    flags += [{"kind": "detail", "type": s["type"], "topic": s["topic"], "content": s["content"], "p": s["p"]}
              for s in scores if s.get("flagged") and not s.get("secret")]
    if block and block["flagged"]:
        flags.append({"kind": "block", "type": "block", "topic": "_block",
                      "content": "empty although the turn had durable content" if block["empty"]
                      else ("generic or boilerplate" if block["quality"] == "generic_or_lazy"
                            else "a durable finding from the turn is not captured")})

    _log({"event": "assessed", "session": session_id, "transcript": transcript_path,
          "turn": turn_number(transcript_path), "threshold": THRESHOLD, "entries": len(entries),
          "failed": failed, "scores": scores, "block": block, "flags": flags,
          "cost": round(cost, 6), "latency_max": lat})
    if stale:
        _log({"event": "stale_candidates", "session": session_id, "memories": stale})
    from hooks.hook_helpers import record_metric, save_hook_state
    record_metric(session_id, "jev_gate_assessed", None, len(entries))
    if flags:
        prior = _load(session_id)
        if prior:   # a newer flag set replaces one that was never answered
            _log({"event": "expired_unshown" if not prior.get("shown") else "expired_unanswered",
                  "session": session_id, "flags": prior.get("flagged")})
        save_hook_state(session_id, STATE_KEY, json.dumps({"flagged": flags, "shown": False,
                                                           "at": time.time()}))
        record_metric(session_id, "jev_gate_flagged", None, len(flags))
    return flags


def spawn_assess(session_id: str, transcript_path: str, entries: list[dict]) -> None:
    """Run assess in a detached process so the stop hook never waits on Jev.
    Empty blocks are assessed too: they are the clearest laziness case."""
    if not (enabled() and session_id and transcript_path):
        return
    try:
        payload = json.dumps({"session": session_id, "transcript": transcript_path,
                              "entries": [{k: e.get(k) for k in ("type", "topic", "content")}
                                          for e in (entries or [])]})
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = subprocess.Popen([sys.executable, "-m", "cairn.jev_gate", "assess"], cwd=root,
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        p.stdin.write(payload.encode())
        p.stdin.close()
    except Exception:
        pass


# --- next turn (N+1) --------------------------------------------------------

def _load(session_id: str) -> Optional[dict]:
    from hooks.hook_helpers import load_hook_state
    try:
        raw = load_hook_state(session_id, STATE_KEY)
        return json.loads(raw) if raw else None
    except Exception:
        return None


_ASK = {
    "detail": "a detail may not match that turn (a number, name, path, or the direction of a conclusion)",
    "secret": "it appears to contain a credential; restate it without the secret value, saying where the secret is kept",
    "block": "the block as a whole",
}


def _items(ob: dict) -> str:
    out = []
    for f in ob["flagged"]:
        kind = f.get("kind", "detail")
        if kind == "block":
            out.append(f"- the memory block: {f['content']}")
        else:
            out.append(f"- {f['type']}/{f['topic']}: {f['content']}  [{_ASK[kind]}]")
    return "\n".join(out)


def pending_reminder(session_id: str) -> str:
    """Prompt-time text for an obligation not yet shown. Marks it shown."""
    if not enabled():
        return ""
    ob = _load(session_id)
    if not ob or ob.get("shown"):
        return ""
    from hooks.hook_helpers import save_hook_state
    ob["shown"] = True
    save_hook_state(session_id, STATE_KEY, json.dumps(ob))
    return ("<cairn_quality_check>\nA quality check on your previous turn's memory block found:\n"
            + _items(ob) + "\nIn this turn's [cm] block answer each one: for an entry, add a corrected "
            "entry with the same topic or rebut with \"jv\":{\"<topic>\":\"supported: <where in the turn>\"}; "
            "for the block, add the missing entries or rebut with \"jv\":{\"_block\":\"complete: <why>\"}. "
            "Re-check against the turn; do not change a detail that was right.\n</cairn_quality_check>")


def _cm_json(text: str) -> dict:
    m = None
    for m in re.finditer(r"^\[(?:cm|cairn-memory)\]:\s*#\s*'(.+)'\s*$", text, re.MULTILINE):
        pass
    try:
        return json.loads(m.group(1)) if m else {}
    except Exception:
        return {}


def check_obligation(session_id: str, text: str, entries: Optional[list[dict]]) -> Optional[str]:
    """At the N+1 stop: log how each flag was answered; return a one-shot
    re-prompt reason if any went unanswered. The obligation clears either way."""
    if not enabled():
        return None
    ob = _load(session_id)
    if not ob or not ob.get("shown"):
        return None   # not yet delivered (assess still running, or no prompt since)
    from hooks.hook_helpers import delete_hook_state, record_metric
    data = _cm_json(text)
    rebut = {str(k).strip().lower(): v for k, v in (data.get("jv") or {}).items()} \
        if isinstance(data.get("jv"), dict) else {}
    topics = {str(e.get("topic") or e.get("to") or "").strip().lower(): e for e in (entries or [])}
    unanswered = []
    for f in ob["flagged"]:
        t = str(f["topic"]).strip().lower()
        if t in rebut:
            outcome, detail = "rebutted", str(rebut[t])
        elif f.get("kind") == "block" and entries:
            outcome, detail = "corrected", "; ".join(_entry_line(e) for e in entries)[:600]
        elif t in topics and f.get("kind") != "block":
            outcome, detail = "corrected", str(topics[t].get("content"))
        else:
            outcome, detail = ("unanswered_reprompted" if not ob.get("reprompted") else "ignored"), ""
            unanswered.append(f)
        _log({"event": outcome, "session": session_id, "flag": f, "detail": detail})
        record_metric(session_id, f"jev_gate_{outcome}", f"{f.get('kind', 'detail')}:{f['topic']}"[:80])
    if unanswered and not ob.get("reprompted"):
        from hooks.hook_helpers import save_hook_state
        ob["reprompted"] = True
        ob["flagged"] = unanswered
        save_hook_state(session_id, STATE_KEY, json.dumps(ob))
        return ("The quality check on your previous turn's memory block is unanswered:\n" + _items(ob)
                + "\nFor an entry, add a corrected entry with the same topic or rebut with "
                "\"jv\":{\"<topic>\":\"supported: <where>\"}; for the block, add the missing entries or "
                "rebut with \"jv\":{\"_block\":\"complete: <why>\"}.")
    delete_hook_state(session_id, STATE_KEY)
    return None


def _log(record: dict) -> None:
    try:
        record["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(JEV_GATE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# --- effectiveness: spot-check labels and report ---------------------------

JEV_GATE_LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_gate_labels.jsonl")


def _records(path: str) -> list[dict]:
    try:
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        return []


def _scored_rows() -> list[dict]:
    """One row per scored entry across all assessments."""
    rows = []
    for r in _records(JEV_GATE_LOG):
        if r.get("event") == "assessed":
            for s in r.get("scores") or []:
                if s.get("p") is None:
                    continue   # secret-only rows carry no Jev score
                rows.append(dict(s, session=r["session"], transcript=r.get("transcript"),
                                 turn=r.get("turn"), ts=r.get("ts")))
    return rows


def _row_key(r: dict) -> str:
    return f"{r['session']}|{r.get('turn')}|{r['topic']}"


def _evidence(row: dict, n: int = 6) -> list[str]:
    """Turn lines sharing the most words with the entry: where to look."""
    try:
        lines = turn_state(row["transcript"], per_result=4000).splitlines()
    except Exception:
        return []
    words = {w for w in re.findall(r"[\w.-]{3,}", row["content"].lower())}
    ranked = sorted(lines, key=lambda l: -len(words & set(re.findall(r"[\w.-]{3,}", l.lower()))))
    return [l[:300] for l in ranked[:n]]


def spotcheck(n: int = 20, seed: Optional[int] = None) -> int:
    """Show n flagged entries plus n//4 unflagged controls beside the turn
    lines that bear on them; on a terminal, record the user's verdict.
    Labels are the only ground truth: a model's opinion would measure
    agreement, not accuracy."""
    import random
    done = {l["key"] for l in _records(JEV_GATE_LABELS)}
    rows = [r for r in _scored_rows() if _row_key(r) not in done and r.get("transcript")]
    rng = random.Random(seed)
    flagged = [r for r in rows if r["flagged"]]
    controls = [r for r in rows if not r["flagged"]]
    picks = rng.sample(flagged, min(n, len(flagged))) + rng.sample(controls, min(max(1, n // 4), len(controls)))
    rng.shuffle(picks)   # blind: the verdict is asked before the flag is shown
    interactive = sys.stdin.isatty()
    for i, r in enumerate(picks, 1):
        print(f"\n[{i}/{len(picks)}] {r['type']}/{r['topic']}: {r['content']}")
        for line in _evidence(r):
            print(f"    | {line}")
        if not interactive:
            print(f"    jev p={r['p']} flagged={r['flagged']}")
            continue
        ans = input("    Does a detail contradict or go beyond the turn? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
        if ans == "q":
            break
        if ans in ("y", "n"):
            with open(JEV_GATE_LABELS, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": _row_key(r), "p": r["p"], "flagged": r["flagged"],
                                    "label": "defect" if ans == "y" else "fine",
                                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
        print(f"    jev p={r['p']} flagged={r['flagged']}")
    return 0


def _deliveries(session: str, topic: str) -> tuple[int, int]:
    """(times delivered, times engaged) for the memory stored from this entry."""
    try:
        from hooks.hook_helpers import get_conn, get_ephemeral_conn
        with get_conn() as conn:
            ids = [i for (i,) in conn.execute(
                "SELECT id FROM memories WHERE session_id = ? AND topic = ?", (session, topic))]
        if not ids:
            return 0, 0
        q = ",".join("?" * len(ids))
        with get_ephemeral_conn() as eph:
            d, e = eph.execute(
                f"SELECT COUNT(*), COALESCE(SUM(engaged), 0) FROM memory_deliveries "
                f"WHERE memory_id IN ({q})", ids).fetchone()
        return int(d), int(e)
    except Exception:
        return 0, 0


def report() -> dict:
    """Effectiveness summary. Precision (from user labels) and coverage (from
    labelled unflagged controls) are reported separately and never merged."""
    import statistics as st
    log = _records(JEV_GATE_LOG)
    rows = _scored_rows()
    assessed = [r for r in log if r.get("event") == "assessed"]
    outcomes = {k: sum(r.get("event") == k for r in log)
                for k in ("rebutted", "corrected", "unanswered_reprompted", "ignored",
                          "expired_unshown", "expired_unanswered")}
    answered = outcomes["rebutted"] + outcomes["corrected"] + outcomes["ignored"]
    out = {
        "turns_assessed": len(assessed),
        "entries_scored": len(rows),
        "failed_calls": sum(r.get("failed", 0) for r in assessed),
        "flag_rate": round(sum(r["flagged"] for r in rows) / len(rows), 3) if rows else None,
        "cost_usd": round(sum(r.get("cost", 0) for r in assessed), 4),
        "latency_max_median_s": st.median([r.get("latency_max", 0) for r in assessed]) if assessed else None,
        "outcomes": outcomes,
        "rebuttal_rate": round(outcomes["rebutted"] / answered, 3) if answered else None,
    }
    # Reflexive rebuttal check: rebuttal rate in the first vs second half of answered flags.
    ans_events = [r["event"] for r in log if r.get("event") in ("rebutted", "corrected", "ignored")]
    if len(ans_events) >= 10:
        h = len(ans_events) // 2
        out["rebuttal_rate_first_vs_second_half"] = [
            round(ans_events[:h].count("rebutted") / h, 3),
            round(ans_events[h:].count("rebutted") / (len(ans_events) - h), 3)]
    # First-pass quality by turn position (the in-session ratchet question).
    buckets = {"1-3": (1, 3), "4-10": (4, 10), "11-25": (11, 25), "26+": (26, 10**9)}
    out["by_turn_position"] = {
        name: {"n": len(g), "mean_p": round(st.mean(r["p"] for r in g), 3),
               "flag_rate": round(sum(r["flagged"] for r in g) / len(g), 3)}
        for name, (lo, hi) in buckets.items()
        for g in [[r for r in rows if r.get("turn") and lo <= r["turn"] <= hi]] if g}
    # Precision and coverage from user labels.
    labels = _records(JEV_GATE_LABELS)
    lf = [l for l in labels if l["flagged"]]
    lc = [l for l in labels if not l["flagged"]]
    out["precision_labelled_flags"] = (
        {"n": len(lf), "precision": round(sum(l["label"] == "defect" for l in lf) / len(lf), 3)} if lf else None)
    out["defects_among_labelled_unflagged"] = (
        {"n": len(lc), "rate": round(sum(l["label"] == "defect" for l in lc) / len(lc), 3)} if lc else None)
    out["threshold_curve"] = {
        str(t): {"flag_rate": round(sum(r["p"] >= t for r in rows) / len(rows), 3) if rows else None,
                 "labelled_precision": (lambda g: round(sum(l["label"] == "defect" for l in g) / len(g), 3)
                                        if g else None)([l for l in labels if l["p"] >= t])}
        for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
    # Downstream use: are flagged / corrected entries retrieved and engaged with?
    def use(pairs):
        ds = [_deliveries(s, t) for s, t in pairs]
        return {"n": len(ds), "delivered": sum(d > 0 for d, _ in ds), "engaged": sum(e > 0 for _, e in ds)}
    flagged_pairs = {(r["session"], r["topic"]) for r in rows if r["flagged"]}
    corrected_pairs = {(r["session"], r["flag"]["topic"]) for r in log if r.get("event") == "corrected"}
    clean_pairs = list({(r["session"], r["topic"]) for r in rows if not r["flagged"]})[:200]
    out["downstream_use"] = {"flagged": use(flagged_pairs), "corrected": use(corrected_pairs),
                             "unflagged_sample": use(clean_pairs)}
    # Enforced flags by kind, and how each kind was answered.
    kinds: dict = {}
    for r in assessed:
        for f in r.get("flags") or []:
            kinds.setdefault(f.get("kind", "detail"), {"raised": 0})["raised"] += 1
    for r in log:
        if r.get("event") in ("rebutted", "corrected", "ignored") and r.get("flag"):
            k = kinds.setdefault(r["flag"].get("kind", "detail"), {"raised": 0})
            k[r["event"]] = k.get(r["event"], 0) + 1
    out["flags_by_kind"] = kinds
    # Block-level and log-only signals (not enforced until spot-checks validate them).
    blocks = [r["block"] for r in assessed if r.get("block")]
    if blocks:
        out["blocks"] = {
            "n": len(blocks), "empty": sum(b["empty"] for b in blocks),
            "flagged": sum(b["flagged"] for b in blocks),
            "quality": {q: sum(b["quality"] == q for b in blocks) for q in ("good", "generic_or_lazy", "misses_finding")},
            "uncaptured_correction": sum(b.get("uncaptured_correction", False) for b in blocks),
            "task": {t: sum(b.get("task") == t for b in blocks) for t in sorted({b.get("task") for b in blocks if b.get("task")})},
        }
    ent = [s for r in assessed for s in r.get("scores") or [] if s.get("p") is not None]
    if ent:
        out["entries_log_only"] = {
            "transient_rate": round(sum(s.get("durable") == "transient" for s in ent) / len(ent), 3),
            "type_mismatch_rate": round(sum(bool(s.get("kind_pred")) and s.get("kind_pred") != s.get("type")
                                            for s in ent) / len(ent), 3),
            "not_standalone_rate": round(sum((s.get("standalone") or 1) < 0.5 for s in ent) / len(ent), 3),
            "secrets": sum(bool(s.get("secret")) for r in assessed for s in r.get("scores") or []),
        }
    cands: dict = {}
    for r in log:
        if r.get("event") == "stale_candidates":
            for m in r.get("memories") or []:
                if (m.get("p") or 0) >= 0.5:
                    c = cands.setdefault(m["id"], {"id": m["id"], "max_p": 0, "times": 0})
                    c["max_p"] = max(c["max_p"], m["p"])
                    c["times"] += 1
    out["stale_candidates_top"] = sorted(cands.values(), key=lambda c: (-c["max_p"], -c["times"]))[:15]
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["assess"]:
        try:
            p = json.loads(sys.stdin.read())
            assess(p["session"], p["transcript"], p["entries"])
        except Exception:
            pass
        return 0
    if argv[:1] == ["report"]:
        print(json.dumps(report(), indent=2))
        return 0
    if argv[:1] == ["spotcheck"]:
        return spotcheck(int(argv[1]) if len(argv) > 1 else 20)
    print("usage: python -m cairn.jev_gate {assess|report|spotcheck [N]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
