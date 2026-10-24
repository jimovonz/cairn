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
# Today's Jev spend, accumulated so a host UI can display it without parsing the log.
JEV_COST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_cost.json")
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

# Turn signals: logged only, asked in the block call (20 extra questions left
# the gate's own verdicts unchanged and added no latency in the capacity test).
TURN_SIGNALS = {
    # semantic twins of the regex Stop-hook checks: disagreement = their precision
    "unkept_promise": "Did the assistant end by promising to do something it did not do in this turn?",
    "deferral": "Did the assistant defer work to a later session, or hand it back to the user, when it could have done it?",
    "declined": "Did the assistant decline or say it could not do something without first trying?",
    "unauthorised_action": "Did the assistant take an action the user had not authorised (commit, push, delete, send, publish)?",
    "unverified_claim": "Did the assistant state something as fact without checking it when it could have checked?",
    # reasoning integrity
    "pushback_shift": "Did the assistant change its position because the user pushed back, without new evidence or argument?",
    # capture coverage
    "future_instruction": "Did the user give an instruction or preference meant to apply in future sessions?",
    "env_fact": "Did the turn reveal a durable fact about the user's systems or environment?",
    # stratifiers
    "difficulty": "Was this a difficult turn (ambiguous, multi-step, or requiring investigation)?",
    "frustration": "Was the user frustrated or annoyed in this turn?",
    "credentials": "Did the turn involve credentials, keys or passwords?",
    "refusal": "Did the assistant refuse or decline part of the request?",
    "completed": "Did the assistant fully complete what the user asked in this turn?",
}
for _k, _q in TURN_SIGNALS.items():
    BLOCK_Q[f"s_{_k}"] = {"type": "noul", "instructions": _q}

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


# One detector for Cairn: capture-time extraction (cairn/secrets.py) and this
# fallback flag agree on what a secret is.
from cairn.secrets import has_secret  # noqa: E402


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

def turn_state(transcript_path: str, per_result: int = 1500, turn_no: Optional[int] = None) -> str:
    """A turn as text: prompt, assistant text, tool calls and (truncated)
    results. The last real user turn by default, or the 1-based turn_no
    (spot-checks read turns that are no longer last). Host-agnostic."""
    from hooks.hook_helpers import _is_real_user_turn
    from hooks.transcript_adapter import iter_normalized_entries
    entries = list(iter_normalized_entries(transcript_path))
    starts = [i for i, e in enumerate(entries)
              if isinstance(e.get("message"), dict) and e["message"].get("role") == "user"
              and _is_real_user_turn(e["message"])]
    if not starts:
        start, end = 0, len(entries)
    else:
        k = len(starts) - 1 if turn_no is None else max(0, min(turn_no, len(starts)) - 1)
        start, end = starts[k], (starts[k + 1] if k + 1 < len(starts) else len(entries))
    while True:
        lines = []
        for entry in entries[start:end]:
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


def delivered_calibration(session_id: str) -> list[dict]:
    """Calibration rows injected at this session's latest prompt."""
    try:
        from hooks.hook_helpers import get_conn, get_ephemeral_conn
        with get_ephemeral_conn() as eph:
            row = eph.execute("SELECT MAX(turn_index) FROM calibration_deliveries WHERE session_id = ?",
                              (session_id,)).fetchone()
            if not row or row[0] is None:
                return []
            ids = [i for (i,) in eph.execute(
                "SELECT DISTINCT row_id FROM calibration_deliveries WHERE session_id = ? AND turn_index = ?",
                (session_id, row[0]))][:MAX_DELIVERED]
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        with get_conn() as conn:
            return [{"id": i, "content": c} for i, c in conn.execute(
                f"SELECT id, content FROM calibration_rows WHERE id IN ({q})", ids)]
    except Exception:
        return []


def final_response(transcript_path: str) -> str:
    """The assistant text after the turn's last tool result: what the regex
    Stop-hook checks judge."""
    from hooks.transcript_adapter import iter_normalized_entries
    texts = []
    for entry in iter_normalized_entries(transcript_path):
        msg = entry.get("message") or {}
        content = msg.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        if msg.get("role") == "user":
            texts = []
        elif msg.get("role") == "assistant":
            texts += [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(t for t in texts if t.strip())


def regex_checks(text: str, transcript_path: str) -> dict:
    """Run the Stop hook's regex/embedding checks on the final response, with
    their metric writes switched off (the Stop hook already recorded them)."""
    import hooks.hook_helpers as hh
    from hooks import enforcement as enf
    real = hh.record_metric
    hh.record_metric = lambda *a, **k: None
    out = {}
    try:
        for name, fn in (("unkept_promise", lambda: enf.check_trailing_intent(text)),
                         ("deferral", lambda: enf.check_deferral(text, complete=True)),
                         ("declined", lambda: enf.check_declined_without_trying(text, transcript_path))):
            try:
                out[name] = bool(fn())
            except Exception:
                out[name] = None
    finally:
        hh.record_metric = real
    return out


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

def _add_cost(usd: float) -> None:
    """Accumulate today's (UTC) Jev spend into JEV_COST_FILE.

    Last-writer-wins: concurrent assess subprocesses can lose an increment. The
    loss is a few cents of display accuracy, not billing, so no lock is taken.
    """
    if not usd:
        return
    try:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        total = 0.0
        try:
            with open(JEV_COST_FILE) as f:
                prev = json.load(f)
            if prev.get("date") == today:
                total = float(prev.get("usd") or 0.0)
        except (OSError, ValueError):
            pass
        tmp = JEV_COST_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"date": today, "usd": round(total + usd, 6)}, f)
        os.replace(tmp, JEV_COST_FILE)
    except OSError:
        pass


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
        p = subprocess.Popen([sys.executable, "-m", "cairn.jev_gate", "assess-delivered"], cwd=root,
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


# --- missed-context check (synchronous, at the stop) -------------------------

CONTEXT_THRESHOLD = float(os.environ.get("CAIRN_CONTEXT_CHECK_THRESHOLD", "0.8"))
CONTEXT_TIMEOUT_S = 4
_MEMORY_FRAME = ("The assistant has a searchable memory of the user's past sessions: earlier decisions, how their systems "
                 "are set up, where credentials live, project history and the user's preferences. ")
CONTEXT_Q = {"q": {"type": "noul", "instructions": (
    "Does the assistant's reply show that it lacked information or context it needed: it guessed or assumed, asked the "
    "user for something it could have found, said it lacks access, a credential or knowledge, or gave a generic answer "
    "where the user's own specifics were needed?")}}


# --- premature-stop check (synchronous, at the stop) -------------------------

STOP_THRESHOLD = float(os.environ.get("CAIRN_STOP_CHECK_THRESHOLD", "0.57"))
_STOP_RULE = ("The user wants the assistant to do any work it can do itself without asking. Stopping to ask is right "
              "only for irreversible or outward-facing actions (delete, push, publish, send) or a genuine choice "
              "between options.")
STOP_QA = {"q": {"type": "noul", "instructions": (
    "Does the reply name work that still needs doing while presenting the task as finished or the turn as complete?")}}
STOP_QB = {"q": {"type": "noul", "instructions": (
    "Under the user's rule above, did the assistant stop or ask when it should have carried on with the work?")},
    # logged only
    "offers": {"type": "noul", "instructions": (
        "Does the reply end by offering, or asking permission, to do work the user has already asked for or plainly wants?")},
    "ending": {"type": "choice", "instructions": "How does the assistant's reply end?",
               "criteria": {"complete": "the request is fully done and nothing remains",
                            "needs_user_decision": "a real choice only the user can make, or an irreversible action needing approval",
                            "remaining_work_listed": "it lists remaining work it could do but stops anyway",
                            "offers_to_continue": "it offers or asks to carry on with work that was already requested",
                            "clarification": "it asks a genuine clarifying question"}}}


def _turn_prompt(turn: str) -> str:
    return next((l[len("[USER PROMPT] "):] for l in turn.splitlines() if l.startswith("[USER PROMPT]")), "")


class Judgement:
    """Result of the one synchronous Jev call at the stop."""
    def __init__(self):
        self.stop_reason: Optional[str] = None
        self.context_reason: Optional[str] = None
        self.flags: list[dict] = []

    @property
    def reprompt(self) -> Optional[str]:
        parts = [r for r in (self.stop_reason, self.context_reason) if r]
        return "\n\n".join(parts) or None


def judge_turn(session_id: str, transcript_path: str, reply: str, entries: Optional[list[dict]],
               intent_resolved: bool = False) -> Judgement:
    """Every turn-level question in ONE Jev call, at the stop.

    One combined state (the user's rule and memory framing, the turn, the
    memory block entry by entry with each entry's most related turn lines,
    and the final reply) answers the premature-stop, missed-context,
    per-entry detail, block-quality and turn-signal questions. On 30 real
    turns this tracked the separate calls closely (r 0.87-0.97; same
    decision at the live thresholds in 57/58 entries, 29-30/30 stop, 28/30
    context) at 63% lower cost. Logs the same events as before (stop_check,
    context_check, assessed) so report and the follow-up labels still apply.
    Flags for the next turn are returned, not saved: the caller saves them
    after checking the previous turn's obligation. Fails open."""
    j = Judgement()
    if not enabled():
        return j
    reply = re.sub(r"(?m)^\[(?:cm|cairn-memory)\]: #.*$", "", reply or "").strip()
    key = _key()
    try:
        turn = redact(turn_state(transcript_path)) if transcript_path else ""
    except Exception:
        turn = ""
    if not key or not (turn.strip() or reply):
        return j
    entries = [e for e in (entries or []) if e.get("content")][:MAX_ENTRIES]
    sections = "\n\n".join(
        f"ENTRY {i+1}: {redact(_entry_line(e))}\nTURN LINES MOST RELATED TO ENTRY {i+1}:\n"
        + "\n".join(related_lines(turn, str(e.get("content")), n=6)) for i, e in enumerate(entries)) \
        or "(no entries: the block is empty)"
    state = (f"{_STOP_RULE}\n{_MEMORY_FRAME}\n\nTHE TURN:\n{turn}\n\n"
             f"MEMORY BLOCK the assistant wrote for this turn, entry by entry:\n{sections}\n\n"
             f"ASSISTANT'S FINAL REPLY:\n{redact(reply[-6000:])}")
    q = {}
    for i in range(len(entries)):
        q[f"e{i}"] = {"type": "noul", "instructions": (
            f"Does any specific detail in ENTRY {i+1} (a number, name, path, or the direction of a conclusion) "
            "differ from what the turn shows?")}
        q[f"d{i}"] = {**ENTRY_Q["durable"], "instructions": f"How long does ENTRY {i+1} stay true and useful?"}
    q.update(BLOCK_Q)
    q.update({
        "stop_a": {"type": "noul", "instructions": (
            "Considering only the ASSISTANT'S FINAL REPLY: does it name work that still needs doing while presenting "
            "the task as finished or the turn as complete?")},
        "stop_b": {"type": "noul", "instructions": (
            "Under the user's rule above, did the assistant's final reply stop or ask when it should have carried on "
            "with the work?")},
        "offers": STOP_QB["offers"], "ending": STOP_QB["ending"],
        "ctx": {"type": "noul", "instructions": (
            "Does the assistant's final reply show that it lacked information or context it needed: it guessed or "
            "assumed, asked the user for something it could have found, said it lacks access, a credential or "
            "knowledge, or gave a generic answer where the user's own specifics were needed?")},
    })
    global CALL_TIMEOUT_S
    saved, CALL_TIMEOUT_S = CALL_TIMEOUT_S, CONTEXT_TIMEOUT_S
    try:
        ans, cost, lat = _ask(key, state, q)
    finally:
        CALL_TIMEOUT_S = saved
    _add_cost(cost)
    turn_no = turn_number(transcript_path) if transcript_path else None
    if ans is None:
        _log({"event": "judge_failed", "session": session_id, "turn": turn_no, "latency": lat})
        return j
    from hooks.hook_helpers import cairn_query_invoked_this_turn, record_metric
    try:
        legacy = regex_checks(reply, transcript_path)
    except Exception:
        legacy = {}

    # premature stop
    pa, pb = _noul(ans, "stop_a"), _noul(ans, "stop_b")
    p_stop = None if pa is None or pb is None else round((pa + pb) / 2, 3)
    stop_fired = p_stop is not None and p_stop >= STOP_THRESHOLD and not intent_resolved \
        and os.environ.get("CAIRN_STOP_CHECK", "1") not in ("0", "false", "no")
    _log({"event": "stop_check", "session": session_id, "transcript": transcript_path, "turn": turn_no,
          "p": p_stop, "p_reply": pa, "p_rule": pb, "offers": _noul(ans, "offers"), "ending": _choice(ans, "ending"),
          "legacy_fired": legacy.get("unkept_promise"), "intent_resolved": intent_resolved,
          "threshold": STOP_THRESHOLD, "fired": stop_fired, "combined": True})
    record_metric(session_id, "stop_check_fired" if stop_fired else "stop_check_clear", None, p_stop or 0)
    if stop_fired:
        j.stop_reason = ("Your reply names work that still remains and stops there. Unless the next step is "
                         "irreversible or outward-facing (delete, push, publish, send) or a genuine choice only the "
                         "user can make, carry on and do it now. If you must stop, say exactly what decision you "
                         "need. If nothing remains, add 'int: resolved' to your [cm] block.")

    # missed context (acts only when the turn made no Cairn query)
    p_ctx = _noul(ans, "ctx")
    cairn_used = bool(transcript_path) and cairn_query_invoked_this_turn(transcript_path)
    ctx_fired = p_ctx is not None and p_ctx >= CONTEXT_THRESHOLD and not cairn_used \
        and os.environ.get("CAIRN_CONTEXT_CHECK", "1") not in ("0", "false", "no")
    _log({"event": "context_check", "session": session_id, "transcript": transcript_path, "turn": turn_no,
          "p": p_ctx, "cairn_used": cairn_used, "threshold": CONTEXT_THRESHOLD, "fired": ctx_fired,
          "combined": True})
    record_metric(session_id, "context_check_fired" if ctx_fired else "context_check_clear", None, p_ctx or 0)
    if ctx_fired:
        query_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query.py")
        j.context_reason = ("Your reply shows missing information or context (a guess, an assumption, a question to "
                            "the user, or a claim of no access or knowledge), and this turn did not search Cairn. "
                            f"Search it now, e.g. python3 {query_py} --semantic \"<the topic> | <what you were "
                            "missing>\" (in pi: the cairn_query tool), then revise your answer. If Cairn has nothing "
                            "relevant, say so briefly and keep your answer.")

    # entries, block, signals -> next-turn obligation
    scores = []
    for i, e in enumerate(entries):
        p = _noul(ans, f"e{i}")
        scores.append({"type": e.get("type"), "topic": e.get("topic"), "content": redact(str(e.get("content"))),
                       "p": p, "durable": _choice(ans, f"d{i}"), "secret": has_secret(str(e.get("content"))),
                       "flagged": p is not None and p >= THRESHOLD})
    quality, omission = _choice(ans, "quality"), _noul(ans, "omission") or 0.0
    block = {"quality": quality, "probs": ((ans.get("quality") or {}).get("probabilities")), "omission": omission,
             "user_correction": _noul(ans, "user_correction"), "task": _choice(ans, "task"), "empty": not entries,
             "flagged": (omission >= OMISSION_THRESHOLD) if not entries else quality in ("generic_or_lazy", "misses_finding"),
             "signals": {k: _noul(ans, f"s_{k}") for k in TURN_SIGNALS}, "regex": legacy}
    block["uncaptured_correction"] = bool((block["user_correction"] or 0) >= 0.7
                                          and not any(x.get("type") == "correction" for x in entries))
    block["uncaptured_instruction"] = bool((block["signals"].get("future_instruction") or 0) >= 0.7
                                           and not any(x.get("type") in ("preference", "workflow") for x in entries))
    flags = [{"kind": "secret", "type": s["type"], "topic": s["topic"], "content": s["content"]}
             for s in scores if s["secret"]]
    flags += [{"kind": "detail", "type": s["type"], "topic": s["topic"], "content": s["content"], "p": s["p"]}
              for s in scores if s["flagged"] and not s["secret"]]
    if block["flagged"]:
        flags.append({"kind": "block", "type": "block", "topic": "_block",
                      "content": "empty although the turn had durable content" if block["empty"]
                      else ("generic or boilerplate" if quality == "generic_or_lazy"
                            else "a durable finding from the turn is not captured")})
    _log({"event": "assessed", "session": session_id, "transcript": transcript_path, "turn": turn_no,
          "threshold": THRESHOLD, "entries": len(entries), "failed": 0, "scores": scores, "block": block,
          "flags": flags, "cost": round(cost, 6), "latency_max": lat, "combined": True, "questions": len(q)})
    record_metric(session_id, "jev_gate_assessed", None, len(entries))
    j.flags = flags
    return j


def save_flags(session_id: str, flags: list[dict]) -> None:
    """Save this turn's enforced flags as the next turn's obligation."""
    if not (enabled() and session_id and flags):
        return
    from hooks.hook_helpers import record_metric, save_hook_state
    prior = _load(session_id)
    if prior:   # a newer flag set replaces one that was never answered
        _log({"event": "expired_unshown" if not prior.get("shown") else "expired_unanswered",
              "session": session_id, "flags": prior.get("flagged")})
    save_hook_state(session_id, STATE_KEY, json.dumps({"flagged": flags, "shown": False, "at": time.time()}))
    record_metric(session_id, "jev_gate_flagged", None, len(flags))


def assess_delivered(session_id: str, transcript_path: str, entries: list[dict]) -> None:
    """Background, log-only: delivered memories (stale / used / relevant),
    delivered calibration rows (applies / followed) and per-entry type and
    standalone signals, in one call. Needs DB lookups, so it stays off the
    stop path."""
    key = _key()
    turn = redact(turn_state(transcript_path)) if transcript_path else ""
    if not key or not turn.strip():
        return
    entries = [e for e in (entries or []) if e.get("content")][:MAX_ENTRIES]
    memories = delivered_memories(session_id)[:6]
    cal_rows = delivered_calibration(session_id)[:5]
    dq = {}
    for m in memories:
        line = redact(_entry_line(m))[:1200]
        dq[f"m{m['id']}"] = {"type": "noul", "instructions": f"Does the turn above show that this stored memory is now wrong or outdated? Memory: {line}"}
        dq[f"u{m['id']}"] = {"type": "noul", "instructions": f"Did the assistant's response in the turn above make use of this memory? Memory: {line}"}
        dq[f"r{m['id']}"] = {"type": "noul", "instructions": f"Is this memory relevant to what the user asked in the turn above? Memory: {line}"}
    for c in cal_rows:
        g = redact(str(c["content"]))[:1000]
        dq[f"a{c['id']}"] = {"type": "noul", "instructions": f"Does this guidance apply to anything the assistant did in the turn above? Guidance: {g}"}
        dq[f"c{c['id']}"] = {"type": "noul", "instructions": f"Did the assistant's behaviour in the turn above follow this guidance? Guidance: {g}"}
    for i, e in enumerate(entries):
        line = redact(_entry_line(e))
        dq[f"k{i}"] = {**ENTRY_Q["kind"], "instructions": f"Which memory type best fits this entry? Entry: {line}"}
        dq[f"s{i}"] = {"type": "noul", "instructions": (
            "Could someone with no access to this conversation understand this entry without guessing what 'this', "
            f"'the fix' or similar references mean? Entry: {line}")}
    if not dq:
        return
    ans, cost, lat = _ask(key, turn, dq)
    _add_cost(cost)
    if ans is None:
        return
    turn_no = turn_number(transcript_path)
    stale = sorted(({"id": m["id"], "p": _noul(ans, f"m{m['id']}")} for m in memories), key=lambda s: -(s["p"] or 0))
    if stale:
        _log({"event": "stale_candidates", "session": session_id, "memories": stale})
    _log({"event": "delivered_judged", "session": session_id, "turn": turn_no, "cost": round(cost, 6),
          "memories": [{"id": m["id"], "stale": _noul(ans, f"m{m['id']}"), "used": _noul(ans, f"u{m['id']}"),
                        "relevant": _noul(ans, f"r{m['id']}")} for m in memories],
          "calibration": [{"row_id": c["id"], "applies": _noul(ans, f"a{c['id']}"),
                           "followed": _noul(ans, f"c{c['id']}")} for c in cal_rows],
          "entries": [{"topic": e.get("topic"), "type": e.get("type"), "kind_pred": _choice(ans, f"k{i}"),
                       "standalone": _noul(ans, f"s{i}")} for i, e in enumerate(entries)]})


# --- the user's next message as a label for the previous turn ---------------

_CORRECTION = re.compile(r"(?i)^\W*(wtf|what the f|ffs|seriously\?|are you (kidding|serious)|why (the hell|tf|on earth)|"
                         r"no[,!.] |nope|that'?s (not|wrong)|you('ve| have)? (ignored|missed|not)|i (already )?(said|told you)|"
                         r"again\?!)")
_STOP_NUDGE = re.compile(r"(?i)^\W*(continue|carry on|keep going|go on|go ahead|proceed|do it|do them|do all|y|yes|yep|"
                         r"build (it|all|them)[^.]{0,20}|implement( the rest| it| them)?|why (did|do) you stop.*|you stopped.*)\W*$")
_CAIRN_NUDGE = re.compile(r"(?i)\b(use|search|check|query|consult|look (in|at))\b[^.?!\n]{0,12}\bcairn\b|"
                          r"did you (not )?(even )?(search|check|query|use|consult)[^?\n]{0,10}cairn|"
                          r"instructions to use cairn|explicit(ly)? instruct")
CORRECTION_KEY = "correction_pending"


def classify_followup(prompt: str) -> Optional[str]:
    head = (prompt or "").strip()
    if not head:
        return None
    if _CAIRN_NUDGE.search(head[:300]):
        return "missed_context"
    if _STOP_NUDGE.match(head[:120]):
        return "premature_stop"
    if _CORRECTION.search(head[:80]):
        return "corrected"
    return "no_complaint"


def label_previous_turn(session_id: str, prompt: str) -> Optional[str]:
    """Log the user's new prompt as a label for this session's previous turn,
    and arm the correction-capture requirement when it is a correction. The
    label joins, by session and time, to the stop/context/assess events logged
    for that turn, so `report` can score every signal against the user."""
    if not enabled() or not session_id:
        return None
    label = classify_followup(prompt)
    if label is None:
        return None
    _log({"event": "followup_label", "session": session_id, "label": label, "prompt_head": redact(prompt[:80])})
    if label in ("corrected", "missed_context"):
        from hooks.hook_helpers import save_hook_state
        save_hook_state(session_id, CORRECTION_KEY, label)
    return label


def check_correction_captured(session_id: str, entries: Optional[list[dict]], text: str) -> Optional[str]:
    """One-shot: when this turn began with a user correction, its [cm] must
    hold a correction entry (or decline via "jv":{"_correction":"none: why"})."""
    if not enabled():
        return None
    from hooks.hook_helpers import delete_hook_state, load_hook_state, record_metric
    pending = load_hook_state(session_id, CORRECTION_KEY)
    if not pending:
        return None
    delete_hook_state(session_id, CORRECTION_KEY)   # one re-prompt at most
    if any((e.get("type") or "") == "correction" for e in (entries or [])):
        record_metric(session_id, "correction_captured")
        return None
    jv = _cm_json(text).get("jv")
    if isinstance(jv, dict) and "_correction" in jv:
        _log({"event": "correction_declined", "session": session_id, "detail": str(jv["_correction"])[:300]})
        return None
    record_metric(session_id, "correction_capture_reprompted")
    return ("The user corrected you at the start of this turn. Add a correction entry to your [cm] block: what you got "
            "wrong, the fix, and how to avoid it. If there was genuinely nothing to learn, say why with "
            "\"jv\":{\"_correction\":\"none: <why>\"}.")


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
        lines = turn_state(row["transcript"], per_result=4000, turn_no=row.get("turn")).splitlines()
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
    all_labels = _records(JEV_GATE_LABELS)
    labels = [l for l in all_labels if "flagged" in l]          # entry labels
    sig_labels = [l for l in all_labels if "signal" in l]       # turn-signal labels
    if sig_labels:
        out["signal_precision"] = {k: {"n": len(g), "right": round(sum(l["label"] == "right" for l in g) / len(g), 3)}
                                   for k in sorted({l["signal"] for l in sig_labels})
                                   for g in [[l for l in sig_labels if l["signal"] == k]]}
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
                if (m.get("p") or 0) >= 0.7:   # unrelated memories score ~0.3-0.5 (live check)
                    c = cands.setdefault(m["id"], {"id": m["id"], "max_p": 0, "times": 0})
                    c["max_p"] = max(c["max_p"], m["p"])
                    c["times"] += 1
    out["stale_candidates_top"] = sorted(cands.values(), key=lambda c: (-c["max_p"], -c["times"]))[:15]

    # Turn signals (log-only): rate >= 0.5, overall and first vs second half.
    sig_blocks = [b for b in blocks if b.get("signals")] if blocks else []
    if sig_blocks:
        def rate(bs, k):
            v = [b["signals"].get(k) for b in bs if b["signals"].get(k) is not None]
            return round(sum(x >= 0.5 for x in v) / len(v), 3) if v else None
        h = len(sig_blocks) // 2
        out["turn_signals"] = {k: {"rate": rate(sig_blocks, k),
                                   "first_vs_second_half": [rate(sig_blocks[:h], k), rate(sig_blocks[h:], k)]
                                   if h >= 5 else None} for k in TURN_SIGNALS}
        out["turn_signals"]["uncaptured_instruction"] = {"count": sum(b.get("uncaptured_instruction", False)
                                                                       for b in sig_blocks)}
        # Regex Stop-hook checks vs their Jev twins: disagreement is their first precision signal.
        agree = {}
        for b in sig_blocks:
            for k, rx in (b.get("regex") or {}).items():
                j = b["signals"].get(k)
                if rx is None or j is None:
                    continue
                cell = ("both" if rx and j >= 0.5 else "regex_only" if rx else "jev_only" if j >= 0.5 else "neither")
                agree.setdefault(k, {"both": 0, "regex_only": 0, "jev_only": 0, "neither": 0})[cell] += 1
        out["regex_vs_jev"] = agree
    # Retrieval: did delivered memories get used, and does Jev agree with the engagement label?
    judged = [r for r in log if r.get("event") == "delivered_judged"]
    mems = [dict(m, session=r["session"]) for r in judged for m in r.get("memories") or []]
    if mems:
        out["retrieval"] = {
            "judged": len(mems),
            "used_rate": round(sum((m.get("used") or 0) >= 0.5 for m in mems) / len(mems), 3),
            "relevant_rate": round(sum((m.get("relevant") or 0) >= 0.5 for m in mems) / len(mems), 3),
            "used_vs_engaged": _agreement(mems, "used", _engaged_label),
        }
    cals_all = [dict(c, session=r["session"]) for r in judged for c in r.get("calibration") or []]
    # "followed" is only meaningful where the guidance applied to the turn.
    cals = [c for c in cals_all if (c.get("applies") if c.get("applies") is not None else 1) >= 0.5]
    if cals_all:
        out["calibration_follow"] = {
            "delivered": len(cals_all),
            "judged": len(cals),
            "followed_rate": round(sum((c.get("followed") or 0) >= 0.5 for c in cals) / len(cals), 3),
            "jev_vs_analyser": _agreement(cals, "followed", _analyser_outcome),
        }
    out["vs_user_labels"] = _vs_user_labels(log)
    return out


def _vs_user_labels(log: list[dict]) -> dict:
    """Score each check against the user's next message. A follow-up label
    joins to the latest event of each kind logged earlier in the same session
    (that is, the previous turn)."""
    def auc(P, N):
        return round(sum((p > n) + .5 * (p == n) for p in P for n in N) / (len(P) * len(N)), 3) if P and N else None
    last: dict = {}
    pairs: dict = {"stop_check": [], "context_check": [], "assessed": []}
    labels: dict = {}
    for r in log:   # the log is append-only, so file order is time order
        ev, s = r.get("event"), r.get("session")
        if ev in pairs:
            last[(s, ev)] = r
        elif ev == "followup_label":
            labels[r["label"]] = labels.get(r["label"], 0) + 1
            for kind in pairs:
                prev = last.pop((s, kind), None)
                if prev is not None:
                    pairs[kind].append((prev, r["label"]))
    out: dict = {"labels": labels}
    for kind, target, field in (("stop_check", "premature_stop", "p"), ("context_check", "missed_context", "p")):
        ps = [(e, l) for e, l in pairs[kind] if e.get(field) is not None]
        if not ps:
            continue
        pos = [e[field] for e, l in ps if l == target]
        neg = [e[field] for e, l in ps if l == "no_complaint"]
        out[kind] = {"n": len(ps), "auc_vs_" + target: auc(pos, neg),
                     "fired_then_" + target: sum(e.get("fired") and l == target for e, l in ps),
                     "fired_then_no_complaint": sum(e.get("fired") and l == "no_complaint" for e, l in ps),
                     "not_fired_then_" + target: sum(not e.get("fired") and l == target for e, l in ps)}
        if kind == "stop_check":
            out[kind]["legacy_fired_then_premature_stop"] = sum(bool(e.get("legacy_fired")) and l == target for e, l in ps)
            out[kind]["legacy_fired_then_no_complaint"] = sum(bool(e.get("legacy_fired")) and l == "no_complaint" for e, l in ps)
    sig = [(e["block"]["signals"], l) for e, l in pairs["assessed"] if (e.get("block") or {}).get("signals")]
    if sig:
        out["turn_signals_auc_vs_any_complaint"] = {
            k: auc([s[k] for s, l in sig if l != "no_complaint" and s.get(k) is not None],
                   [s[k] for s, l in sig if l == "no_complaint" and s.get(k) is not None]) for k in TURN_SIGNALS}
    return out


def _engaged_label(m: dict) -> Optional[bool]:
    try:
        from hooks.hook_helpers import get_ephemeral_conn
        with get_ephemeral_conn() as eph:
            r = eph.execute("SELECT MAX(engaged) FROM memory_deliveries WHERE session_id = ? AND memory_id = ?",
                            (m["session"], m["id"])).fetchone()
        return None if not r or r[0] is None else bool(r[0])
    except Exception:
        return None


def _analyser_outcome(c: dict) -> Optional[bool]:
    try:
        from hooks.hook_helpers import get_ephemeral_conn
        with get_ephemeral_conn() as eph:
            r = eph.execute("SELECT outcome FROM calibration_deliveries WHERE session_id = ? AND row_id = ? "
                            "AND outcome IS NOT NULL ORDER BY id DESC LIMIT 1", (c["session"], c["row_id"])).fetchone()
        return None if not r else r[0] == "followed"
    except Exception:
        return None


def _agreement(items: list[dict], field: str, other) -> dict:
    """2x2 of Jev (field >= 0.5) against another label; unlabelled skipped."""
    t = {"both": 0, "jev_only": 0, "other_only": 0, "neither": 0, "unlabelled": 0}
    for it in items:
        o = other(it)
        if o is None or it.get(field) is None:
            t["unlabelled"] += 1
            continue
        j = it[field] >= 0.5
        t["both" if j and o else "jev_only" if j else "other_only" if o else "neither"] += 1
    return t


def spotcheck_signals(n: int = 10, seed: Optional[int] = None) -> int:
    """Label Jev's confident turn signals (>= 0.8 or <= 0.2) against the turn
    itself, recording the verdicts for per-signal precision in report."""
    import random
    rng = random.Random(seed)
    turns = [r for r in _records(JEV_GATE_LOG)
             if r.get("event") == "assessed" and (r.get("block") or {}).get("signals") and r.get("transcript")]
    interactive = sys.stdin.isatty()
    for r in rng.sample(turns, min(n, len(turns))):
        text = turn_state(r["transcript"], per_result=300, turn_no=r.get("turn"))
        print("\n" + "=" * 70 + f"\n{r['session'][:8]} turn {r.get('turn')}\n" + text[:2500])
        for k, p in r["block"]["signals"].items():
            if p is None or 0.2 < p < 0.8:
                continue
            claim = ("YES" if p >= 0.8 else "NO") + f" ({p}) — {TURN_SIGNALS[k]}"
            if not interactive:
                print("  jev:", claim)
                continue
            ans = input(f"  jev says {claim}\n    correct? [y/n/s/q]: ").strip().lower()
            if ans == "q":
                return 0
            if ans in ("y", "n"):
                with open(JEV_GATE_LABELS, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": f"{r['session']}|{r.get('turn')}|signal:{k}", "signal": k,
                                        "p": p, "label": "right" if ans == "y" else "wrong"}) + "\n")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["assess-delivered"]:
        try:
            p = json.loads(sys.stdin.read())
            assess_delivered(p["session"], p["transcript"], p["entries"])
        except Exception:
            pass
        return 0
    if argv[:1] == ["report"]:
        print(json.dumps(report(), indent=2))
        return 0
    if argv[:1] == ["spotcheck-signals"]:
        return spotcheck_signals(int(argv[1]) if len(argv) > 1 else 10)
    if argv[:1] == ["spotcheck"]:
        return spotcheck(int(argv[1]) if len(argv) > 1 else 20)
    print("usage: python -m cairn.jev_gate {assess-delivered|report|spotcheck [N]|spotcheck-signals [N]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
