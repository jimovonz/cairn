#!/usr/bin/env python3
"""Signal liveness — is each instrumented signal still arriving? (spec follow-up)

Cairn's remaining work is gated on data accumulating: enforcement cost needs
hook_fired volume, the reranker verdict needs scored deliveries per arm. Every
one of those gates fails the same way — the signal stops and the correct-looking
behaviour is silence, which is exactly what "still waiting" looks like. That
already happened: delivery logging died for 20 hours behind a fail-soft handler
and was noticed only because someone asked whether the data was being collected.

This makes silence reportable. Four failure modes this session shared one shape —
a signal whose failure is indistinguishable from success — and no amount of
reading dashboards catches that, because the dashboard is empty in both cases.

TWO DESIGN RULES, both learned the hard way:

1. Probe the daemon DIRECTLY. Its liveness is asked of the daemon itself, never
   inferred from embed/rerank metric counts — an indirect proxy reports healthy
   whenever the metric writer is alive, which is not the same question.

2. Distinguish IDLE from STALE. If the reference signal (hook_fired) is itself
   silent, nobody was using the system, so downstream silence is EXPECTED and
   reporting it as a fault would train everyone to ignore this check. A signal is
   only STALE when the system was active and it still did not arrive.

    .venv/bin/python cairn/signal_liveness.py [--hours N] [--json]

Exits non-zero when any signal is STALE, so it can run from cron.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

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
    "Assumes a signal that was arriving and stopped is a fault, while a signal "
    "that never arrived may simply be new. Both are reported distinctly, because "
    "conflating them makes a freshly-shipped instrument look broken and a dead "
    "one look new."
)

# The reference signal for whether anyone was using the system at all.
REFERENCE_SIGNAL = "hook_fired"

# max_silence_h None means "expected to be rare" — reported, never STALE.
SIGNALS = [
    {"name": "hook_fired", "db": "eph", "table": "metrics", "ts": "created_at",
     "where": "event = 'hook_fired'", "max_silence_h": 24,
     "why": "reference signal: the Stop hook completing at all"},
    {"name": "memory_deliveries", "db": "eph", "table": "memory_deliveries",
     "ts": "delivered_at", "max_silence_h": 24,
     "why": "died silently for 20h behind a fail-soft handler"},
    {"name": "memories_written", "db": "dur", "table": "memories",
     "ts": "created_at", "max_silence_h": 48,
     "why": "the durable corpus actually growing"},
    {"name": "gate_status", "db": "eph", "table": "memory_deliveries",
     "ts": "delivered_at", "where": "gate_status IS NOT NULL", "max_silence_h": 24,
     "why": "gates spec 2S.4"},
    {"name": "engagement_labels", "db": "eph", "table": "memory_deliveries",
     "ts": "delivered_at", "where": "engaged_method = 'lexical'", "max_silence_h": 72,
     "why": "the only two-class label stratum; gates every ranker verdict"},
    {"name": "enforcement_block", "db": "eph", "table": "metrics",
     "ts": "created_at", "where": "event = 'enforcement_block'", "max_silence_h": None,
     "why": "gates spec 2F.2; legitimately rare, so never STALE"},
]


def classify(ever_seen, age_hours, max_silence_h, system_active):
    """Status for one signal. Pure, so the policy is testable without a DB.

    NEVER  — no row has ever existed. Possibly just-shipped, not necessarily broken.
    IDLE   — silent, but the system was idle too, so silence is expected.
    STALE  — the system was active and this signal still did not arrive. A fault.
    RARE   — no cadence expectation; reported for visibility only.
    OK     — arrived within its expected window.
    """
    if not ever_seen:
        return "NEVER"
    if max_silence_h is None:
        return "RARE"
    if age_hours is None:
        # Rows exist but their timestamp did not parse. Defaulting to OK would be
        # optimistic-on-unknown, which is the exact instinct this module exists to
        # correct — an unreadable clock is not evidence of health.
        return "UNKNOWN"
    if age_hours <= max_silence_h:
        return "OK"
    return "STALE" if system_active else "IDLE"


def _age_hours(ts_text, now=None):
    """Hours since a stored UTC timestamp. Storage is UTC (see timeutil)."""
    if not ts_text:
        return None
    now = now or datetime.now(timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(str(ts_text)[:26], fmt).replace(tzinfo=timezone.utc)
            return (now - dt).total_seconds() / 3600.0
        except ValueError:
            continue
    return None


def probe_daemon():
    """Ask the daemon directly. Never inferred from metric counts — an indirect
    proxy answers 'is the metric writer alive', not 'is the daemon serving'."""
    try:
        from cairn.daemon import send_request
        resp = send_request({"action": "ping"})
        if resp:
            return True, "responding"
        return False, "no response to ping"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def collect(hours=24, eph_path=None, durable_path=None, now=None):
    from cairn.relevance import _eph_path, _durable_path

    conns = {"eph": sqlite3.connect(f"file:{_eph_path(eph_path)}?mode=ro", uri=True),
             "dur": sqlite3.connect(f"file:{_durable_path(durable_path)}?mode=ro", uri=True)}
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=hours)
    cutoff_s = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    raw = {}
    for sig in SIGNALS:
        conn = conns[sig["db"]]
        where = f" WHERE {sig['where']}" if sig.get("where") else ""
        try:
            last = conn.execute(
                f"SELECT MAX({sig['ts']}) FROM {sig['table']}{where}").fetchone()[0]
            and_ = " AND " if where else " WHERE "
            recent = conn.execute(
                f"SELECT COUNT(*) FROM {sig['table']}{where}{and_}{sig['ts']} >= ?",
                (cutoff_s,)).fetchone()[0]
        except sqlite3.OperationalError as e:
            raw[sig["name"]] = {"error": str(e)}
            continue
        raw[sig["name"]] = {"last": last, "recent": recent,
                            "age_h": _age_hours(last, now)}
    for c in conns.values():
        c.close()

    ref = raw.get(REFERENCE_SIGNAL, {})
    system_active = bool(ref.get("recent"))

    out = []
    for sig in SIGNALS:
        r = raw[sig["name"]]
        if "error" in r:
            out.append({"name": sig["name"], "status": "ERROR", "detail": r["error"],
                        "why": sig["why"], "recent": 0, "age_h": None})
            continue
        status = classify(bool(r["last"]), r["age_h"], sig["max_silence_h"],
                          system_active)
        out.append({"name": sig["name"], "status": status, "recent": r["recent"],
                    "age_h": r["age_h"], "last": r["last"], "why": sig["why"]})
    return out, system_active


def render(rows, system_active, hours, daemon):
    ok, detail = daemon
    print(f"=== Signal liveness (last {hours}h) ===")
    print(f"  daemon (direct probe): {'OK' if ok else 'DOWN'} — {detail}")
    print(f"  system active in window: {'yes' if system_active else 'NO — silence below is expected'}")
    print(f"\n  {'signal':<20} {'status':<7} {'recent':>7} {'age':>9}  why")
    for r in rows:
        age = f"{r['age_h']:.1f}h" if r.get("age_h") is not None else "never"
        print(f"  {r['name']:<20} {r['status']:<7} {r['recent']:>7} {age:>9}  {r['why']}")
    stale = [r for r in rows if r["status"] in ("STALE", "ERROR", "UNKNOWN")]
    if stale:
        print(f"\n  {len(stale)} signal(s) need attention: "
              f"{', '.join(r['name'] for r in stale)}")
        print("  STALE means the system WAS active and this signal still did not "
              "arrive — treat as a fault, not a quiet period.")
    return 1 if stale else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows, active = collect(hours=a.hours)
    daemon = probe_daemon()
    if a.json:
        print(json.dumps({"signals": rows, "system_active": active,
                          "daemon_ok": daemon[0]}, indent=2, default=str))
        return 1 if any(r["status"] in ("STALE", "ERROR", "UNKNOWN") for r in rows) else 0
    return render(rows, active, a.hours, daemon)


if __name__ == "__main__":
    sys.exit(main())
