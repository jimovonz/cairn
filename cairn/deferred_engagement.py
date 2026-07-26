#!/usr/bin/env python3
"""Deferred-value engagement window (spec 2S.6).

Engagement scores whether a memory was used in the turn it was delivered. That
is the wrong window for any layer whose value horizon is longer than one turn:
first-prompt fires once at session start to orient the whole session, so a
memory it injects may be load-bearing at turn 12 and still score 0. Ranking
layers on a single-turn window is therefore a category error, and it is why
suppressing first-prompt on that basis was rejected.

This measures the same memories against WIDER windows — was the memory used in
any of the next N assistant turns — so a layer's deferred value becomes visible
instead of being scored as absence.

Read the SHAPE, not the level. A rate that climbs steeply with window size means
the layer pays off late and a single-turn window was mismeasuring it. A flat
rate means the single-turn verdict was already fair.

    .venv/bin/python cairn/deferred_engagement.py [--windows 1,3,5,10] [--limit N]
"""
import argparse
import sys

try:
    import pysqlite3 as sqlite3
except ImportError:  # pragma: no cover - guarded by tests/test_sqlite_guard.py
    import os
    if os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") != "1":
        raise RuntimeError(
            "cairn requires pysqlite3; set CAIRN_ALLOW_STDLIB_SQLITE=1 to override"
        )
    import sqlite3

INPUT_DOMAIN_INVARIANT = (
    "Assumes the session transcript still holds the assistant turns that "
    "followed a delivery. Transcripts are pruned by cleanupPeriodDays, so older "
    "deliveries lose their window and are reported as unmeasurable, never as "
    "non-engagement."
)

DEFAULT_WINDOWS = (1, 3, 5, 10)

# Minimum agreement with LIVE engagement verdicts, at window=1, before any layer
# rate may be reported. At w=1 this tool should reproduce the live detector
# almost exactly — it is the same scorer on the same turn — so agreement is a
# direct test of whether the response reconstruction is right.
#
# Measured 2026-07-26: 43.2% using context_text as the prompt base, 39.5% using
# the recovered user prompt. Both near chance. Diagnosis: assistant MESSAGES are
# not assistant TURNS — a tool-using turn emits several assistant messages, so
# "first message at or after delivery" usually picks a tool preamble rather than
# the final response the live detector scored. Reconstructing turn boundaries is
# the outstanding work.
#
# Until that lands, the gate below refuses to print layer rates. A deferred-value
# curve computed from mis-selected responses would look exactly like a finding.
VALIDATION_MIN_AGREEMENT = 0.75


def _user_turns(transcript_path):
    """-> sorted [(ts, text)] of user messages carrying text.

    The live detector subtracts the actual USER PROMPT from the memory's terms.
    Substituting the delivery's stored context_text is not equivalent: that
    column is a recent-context window and is frequently empty on bootstrap
    layers, which makes every memory term read as distinctive and manufactures
    positives. Validated at 43% agreement with live verdicts before this was
    fixed — barely above chance.
    """
    from hooks.transcript_adapter import iter_normalized_entries
    from cairn.backfill_semantic_engagement import _parse_ts

    out = []
    try:
        for entry in iter_normalized_entries(transcript_path):
            msg = entry.get("message", {})
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text")
            else:
                text = content if isinstance(content, str) else ""
            text = (text or "").strip()
            ts = _parse_ts(entry.get("timestamp", ""))
            if text and ts:
                out.append((ts, text))
    except Exception:
        return []
    out.sort(key=lambda r: r[0])
    return out


def _prompt_before(user_turns, delivered_at):
    """The last user prompt at or before the delivery — what the live detector
    subtracts. Falls back to the empty string, which is conservative only in the
    sense of being explicit: callers should treat a missing prompt as reduced
    confidence, not as a clean measurement."""
    from cairn.backfill_semantic_engagement import _parse_ts

    ts = _parse_ts(delivered_at)
    if ts is None:
        return ""
    best = ""
    for t, text in user_turns:
        if t <= ts:
            best = text
        else:
            break
    return best


def _responses_within(turns, delivered_at, window):
    """Up to `window` assistant responses at or after the delivery."""
    from cairn.backfill_semantic_engagement import _parse_ts

    ts = _parse_ts(delivered_at)
    if ts is None:
        return []
    out = []
    for t, text in turns:
        if t >= ts:
            out.append(text)
            if len(out) >= window:
                break
    return out


def _used(memory_text, context_text, responses):
    """True if the LIVE engagement detector fires on any response in the window.

    Must be the live `score_engagement`, not a hand-rolled token intersection.
    An any-overlap test reports 76-90% where the live detector reports ~15%,
    because it credits terms shared with the prompt — which would appear whether
    or not the memory helped. Using a looser detector for the wider window would
    manufacture a deferred-value effect out of the measurement change alone.

    Returns None when every response in the window is undecidable (the memory is
    redundant with the prompt), so it is excluded rather than scored 0.
    """
    from cairn.relevance import score_engagement

    decided = False
    for r in responses:
        engaged, _score = score_engagement(r, memory_text, context_text or "")
        if engaged is None:
            continue
        decided = True
        if engaged == 1:
            return True
    return False if decided else None


def validate(limit=600, eph_path=None, durable_path=None):
    """Agreement between this tool's window=1 verdict and the LIVE verdict.

    Ground truth is `memory_deliveries.engaged` on rows the live lexical pass
    scored. At window=1 both should be the same scorer on the same turn, so
    anything far below 1.0 indicates the response reconstruction is wrong rather
    than that engagement is ambiguous.

    Returns (agreement, confusion) where confusion maps "live>mine" -> count.
    Agreement is None when there is nothing to validate against.
    """
    from cairn.relevance import _eph_path, _durable_path, score_engagement
    from cairn.backfill_semantic_engagement import _assistant_turns

    econn = sqlite3.connect(_eph_path(eph_path))
    dconn = sqlite3.connect(_durable_path(durable_path))
    paths = dict(dconn.execute("SELECT session_id, transcript_path FROM sessions"))
    rows = econn.execute(
        "SELECT session_id, memory_id, delivered_at, engaged FROM memory_deliveries "
        "WHERE engaged_method = 'lexical' AND engaged IS NOT NULL "
        f"ORDER BY delivered_at DESC LIMIT {int(limit)}"
    ).fetchall()
    if not rows:
        return None, {}

    memtext = {}
    ids = sorted({r[1] for r in rows})
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        for mid, topic, content in dconn.execute(
                f"SELECT id, topic, content FROM memories WHERE id IN ({ph})", chunk):
            memtext[mid] = f"{topic} {content}"

    a_cache, u_cache = {}, {}
    confusion, agree, total = {}, 0, 0
    for sid, mid, delivered_at, live in rows:
        text = memtext.get(mid)
        if not text:
            continue
        if sid not in a_cache:
            p = paths.get(sid)
            a_cache[sid] = _assistant_turns(p) if p else []
            u_cache[sid] = _user_turns(p) if p else []
        if not a_cache[sid]:
            continue
        resp = _responses_within(a_cache[sid], delivered_at, 1)
        if not resp:
            continue
        mine, _ = score_engagement(resp[0], text, _prompt_before(u_cache[sid], delivered_at))
        if mine is None:
            continue
        key = f"{live}>{mine}"
        confusion[key] = confusion.get(key, 0) + 1
        total += 1
        agree += 1 if live == mine else 0
    if not total:
        return None, {}
    return agree / total, confusion


def analyse(windows=DEFAULT_WINDOWS, limit=None, eph_path=None, durable_path=None):
    from cairn.relevance import _eph_path, _durable_path
    from cairn.backfill_semantic_engagement import _assistant_turns

    econn = sqlite3.connect(_eph_path(eph_path))
    dconn = sqlite3.connect(_durable_path(durable_path))
    paths = dict(dconn.execute("SELECT session_id, transcript_path FROM sessions"))

    q = ("SELECT session_id, memory_id, layer, delivered_at, context_text "
         "FROM memory_deliveries "
         "WHERE layer IS NOT NULL ORDER BY delivered_at DESC")
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = econn.execute(q).fetchall()
    if not rows:
        print("No deliveries with a layer recorded.")
        return {}

    memtext = {}
    ids = sorted({r[1] for r in rows})
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        for mid, topic, content in dconn.execute(
                f"SELECT id, topic, content FROM memories WHERE id IN ({ph})", chunk):
            memtext[mid] = f"{topic} {content}"

    turns_cache = {}
    stats = {}
    unmeasurable = 0
    for sid, mid, layer, delivered_at, context_text in rows:
        text = memtext.get(mid)
        if not text:
            continue
        if sid not in turns_cache:
            p = paths.get(sid)
            turns_cache[sid] = _assistant_turns(p) if p else []
        turns = turns_cache[sid]
        if not turns:
            unmeasurable += 1
            continue
        bucket = stats.setdefault(layer, {w: [0, 0] for w in windows})
        for w in windows:
            verdict = _used(text, context_text, _responses_within(turns, delivered_at, w))
            if verdict is None:
                continue
            bucket[w][0] += 1
            bucket[w][1] += 1 if verdict else 0

    agreement, matrix = validate(limit=600, eph_path=eph_path, durable_path=durable_path)
    if agreement is not None and agreement < VALIDATION_MIN_AGREEMENT:
        print("=== Deferred-value engagement: NOT REPORTED ===")
        print(f"  window=1 agreement with live verdicts: {agreement * 100:.1f}% "
              f"(gate: {VALIDATION_MIN_AGREEMENT * 100:.0f}%)")
        print(f"  confusion (live, reconstructed): {matrix}")
        print("  At window=1 this is the SAME scorer on the SAME turn, so low "
              "agreement means\n  the response reconstruction is wrong, not that "
              "engagement is ambiguous.")
        print("  Known cause: assistant messages != assistant turns; a tool-using "
              "turn emits\n  several messages and the first one after delivery is "
              "usually a tool preamble.")
        print("  Layer rates are withheld deliberately — a deferred-value curve "
              "built on\n  mis-selected responses would be indistinguishable from "
              "a real finding.")
        return {"agreement": agreement, "matrix": matrix, "reported": False}

    print("=== Deferred-value engagement by layer ===")
    print(f"  window=1 agreement with live verdicts: "
          f"{agreement * 100:.1f}%" if agreement is not None else
          "  (no live-scored rows available to validate against)")
    print(f"  windows: {list(windows)} assistant turns after delivery")
    if unmeasurable:
        print(f"  unmeasurable (transcript pruned or missing): {unmeasurable} "
              f"— excluded, NOT counted as non-engagement")
    header = "  " + f"{'layer':<24}" + "".join(f"{'w=' + str(w):>9}" for w in windows)
    print("\n" + header)
    for layer in sorted(stats, key=lambda k: -stats[k][windows[0]][0]):
        b = stats[layer]
        cells = ""
        for w in windows:
            n, used = b[w]
            cells += f"{(used / n * 100 if n else 0):>8.1f}%"
        print(f"  {layer[:24]:<24}{cells}   (n={b[windows[0]][0]})")
    print("\n  Read the SHAPE: a rate climbing with window size means the layer "
          "pays off\n  late and the single-turn verdict mismeasured it. Flat means "
          "the single-turn\n  verdict was already fair.")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--limit", type=int, default=4000)
    a = ap.parse_args()
    windows = tuple(int(w) for w in a.windows.split(",") if w.strip())
    analyse(windows=windows, limit=a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
