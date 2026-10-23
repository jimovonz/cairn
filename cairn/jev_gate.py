"""Jev quality gate for [cm] entries — flag at capture, enforce at the next turn.

At capture (turn N) a detached process asks Jev, per entry, whether every
detail (numbers, names, paths, the direction of each conclusion) matches the
turn. Entries above the threshold become a one-shot obligation. Turn N+1 is
told about them on its prompt, and its [cm] must either correct each flagged
entry (a new entry with the same topic) or rebut it in a "jv" map:

    [cm]: # '{"e":[...],"jv":{"<topic>":"supported: <where in the turn>"}, ...}'

An unanswered obligation re-prompts once, then clears. Nothing is deleted:
a wrong flag costs a rebuttal, a missed defect is the status quo.

Thresholds come from the subtle-defect arm (scratchpad eval, 2026-10-24):
strict prompt at 0.7 caught 46% of on-topic edits (78% of changed numbers)
at a 4% false-flag rate. Every outcome is logged to JEV_GATE_LOG for
spot-checking; retune from the rebuttal rate.

Off with CAIRN_JEV_GATE=0. Fails open everywhere: no key, network error or
timeout means no obligation.
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
THRESHOLD = float(os.environ.get("CAIRN_JEV_GATE_THRESHOLD", "0.7"))
MAX_ENTRIES = 8
STATE_CAP = 40000
CALL_TIMEOUT_S = 20
STATE_KEY = "jev_obligation"
JEV_GATE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_gate_log.jsonl")

QUESTION = {"verdict": {
    "type": "choice",
    "instructions": ("Check the ENTRY UNDER REVIEW detail by detail against the turn above: every "
                     "number, name, file path, and the direction of every conclusion "
                     "(does/does not, faster/slower, chosen/rejected)."),
    "criteria": {"supported": "every detail matches the turn",
                 "inferred": "a reasonable conclusion from the turn, no detail contradicts it",
                 "unsupported": "a detail differs from or contradicts the turn, or is absent from it",
                 "generic_meta": "no turn-specific content"}}}

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


def _key() -> Optional[str]:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    try:
        with open(os.path.expanduser("~/.pi/agent/auth.json")) as f:
            return json.load(f)["openrouter"]["key"]
    except Exception:
        return None


def _ask(key: str, state: str) -> tuple[Optional[dict], float, float]:
    """(verdict, cost_usd, latency_s); verdict is None on any failure."""
    body = json.dumps({"model": MODEL, "state": state, "questions": QUESTION}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S).read())
        cost = float((r.get("usage") or {}).get("cost") or 0.0)
        return (r.get("answers") or {}).get("verdict"), cost, round(time.time() - t0, 2)
    except Exception:
        return None, 0.0, round(time.time() - t0, 2)


def unsupported_p(verdict: Optional[dict]) -> float:
    pr = (verdict or {}).get("probabilities") or {}
    return 1.0 - pr.get("supported", 0.0) - pr.get("inferred", 0.0)


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


def assess(session_id: str, transcript_path: str, entries: list[dict]) -> list[dict]:
    """Score each entry; save the flagged ones as this session's obligation.
    Every score is logged (flagged or not) with the turn locator, so precision,
    the threshold curve and turn-position trends can be computed later."""
    key = _key()
    if not key or not entries:
        return []
    turn = redact(turn_state(transcript_path))
    if not turn.strip():
        return []
    entries = [e for e in entries if e.get("content")][:MAX_ENTRIES]

    def one(e):
        line = f"{e.get('type')}/{e.get('topic')}: {e.get('content')}"
        return e, *_ask(key, f"{turn}\n\nENTRY UNDER REVIEW: {redact(line)}")

    with cf.ThreadPoolExecutor(min(8, len(entries))) as ex:
        scored = list(ex.map(one, entries))
    scores = [{"type": e.get("type"), "topic": e.get("topic"), "content": e.get("content"),
               "p": round(unsupported_p(v), 3), "choice": v.get("choice"),
               "flagged": unsupported_p(v) >= THRESHOLD}
              for e, v, _, _ in scored if v is not None]
    flagged = [{k: s[k] for k in ("type", "topic", "content", "p")} for s in scores if s["flagged"]]
    _log({"event": "assessed", "session": session_id, "transcript": transcript_path,
          "turn": turn_number(transcript_path), "threshold": THRESHOLD, "entries": len(entries),
          "failed": len(entries) - len(scores), "scores": scores,
          "cost": round(sum(c for _, _, c, _ in scored), 6),
          "latency_max": max((l for _, _, _, l in scored), default=0)})
    from hooks.hook_helpers import record_metric, save_hook_state
    record_metric(session_id, "jev_gate_assessed", None, len(entries))
    if flagged:
        prior = _load(session_id)
        if prior:   # a newer flag set replaces one that was never answered
            _log({"event": "expired_unshown" if not prior.get("shown") else "expired_unanswered",
                  "session": session_id, "flags": prior.get("flagged")})
        save_hook_state(session_id, STATE_KEY, json.dumps({"flagged": flagged, "shown": False,
                                                           "at": time.time()}))
        record_metric(session_id, "jev_gate_flagged", None, len(flagged))
    return flagged


def spawn_assess(session_id: str, transcript_path: str, entries: list[dict]) -> None:
    """Run assess in a detached process so the stop hook never waits on Jev."""
    if not (enabled() and session_id and transcript_path and entries):
        return
    try:
        payload = json.dumps({"session": session_id, "transcript": transcript_path,
                              "entries": [{k: e.get(k) for k in ("type", "topic", "content")}
                                          for e in entries]})
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


def _items(ob: dict) -> str:
    return "\n".join(f"- {f['type']}/{f['topic']}: {f['content']}" for f in ob["flagged"])


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
    return ("<cairn_quality_check>\nA quality check found details in these memory entries from your "
            "previous turn that may not match that turn (a number, name, path, or the direction of a "
            "conclusion):\n" + _items(ob) + "\nIn this turn's [cm] block, for each one either add a "
            "corrected entry with the same topic, or rebut it with \"jv\":{\"<topic>\":\"supported: "
            "<where in the turn>\"}. Re-check against the turn; do not change a detail that was "
            "right.\n</cairn_quality_check>")


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
    rebut = data.get("jv") if isinstance(data.get("jv"), dict) else {}
    topics = {str(e.get("topic") or e.get("to") or "").strip().lower(): e for e in (entries or [])}
    unanswered = []
    for f in ob["flagged"]:
        t = str(f["topic"]).strip().lower()
        hit_r = next((v for k, v in rebut.items() if str(k).strip().lower() == t), None)
        if hit_r is not None:
            outcome, detail = "rebutted", str(hit_r)
        elif t in topics:
            outcome, detail = "corrected", str(topics[t].get("content"))
        else:
            outcome, detail = ("unanswered_reprompted" if not ob.get("reprompted") else "ignored"), ""
            unanswered.append(f)
        _log({"event": outcome, "session": session_id, "flag": f, "detail": detail})
        record_metric(session_id, f"jev_gate_{outcome}", f["topic"][:80])
    if unanswered and not ob.get("reprompted"):
        from hooks.hook_helpers import save_hook_state
        ob["reprompted"] = True
        ob["flagged"] = unanswered
        save_hook_state(session_id, STATE_KEY, json.dumps(ob))
        return ("The quality check on your previous turn's memory entries is unanswered. For each "
                "entry below, add a corrected entry with the same topic, or rebut it in your [cm] "
                "with \"jv\":{\"<topic>\":\"supported: <where in the turn>\"}:\n" + _items(ob))
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
