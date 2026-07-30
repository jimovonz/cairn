"""Measure the behavioural detector against agent fit labels.

The engagement heuristic has always been *assumed* accurate, never measured.
Agent `fit` labels give the first way to check it: both judge the same
deliveries in the same turn, so on a pair the agent called (winner > loser) the
detector should score the winner higher. Disagreement rate is the detector's
error rate on the only ground truth available.

Read this per layer, never in aggregate. Repo/Confluence ingest and the
bootstrap layers carry ORIENTATION material — designed to point you at the right
place, not to be quoted — so they succeed without appearing in the response at
all. Low engagement there is the expected signature, not detector error, and an
aggregate number silently blames the detector for material it cannot see.

Usage:  python3 cairn/detector_eval.py [--layer L] [--min-pairs N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cairn.ingest import sqlite3  # pysqlite3 guard

EPH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db")
DURABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db")


def evaluate(layer=None, eph_path=None, durable_path=None):
    """Join fit labels (durable DB) against delivery scores (ephemeral DB).

    The two now live in different files — fit pairs are irreplaceable training
    data, deliveries are instrumentation — so this ATTACHes the ephemeral DB
    rather than joining within one connection.
    """
    dur = durable_path or DURABLE
    eph = eph_path or EPH
    conn = sqlite3.connect(f"file:{dur}?mode=ro", uri=True)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "delivery_fit_pairs" not in names:
            return None, "no delivery_fit_pairs table yet — no fit labels collected"
        conn.execute("ATTACH DATABASE ? AS eph", (f"file:{eph}?mode=ro",))
        q = """
            SELECT COALESCE(dw.layer,'?'),
                   dw.engaged_score, dl.engaged_score
            FROM delivery_fit_pairs p
            JOIN eph.memory_deliveries dw
              ON dw.session_id = p.session_id AND dw.memory_id = p.winner_id
            JOIN eph.memory_deliveries dl
              ON dl.session_id = p.session_id AND dl.memory_id = p.loser_id
            WHERE dw.engaged_score IS NOT NULL AND dl.engaged_score IS NOT NULL
              AND dw.engaged_score >= 0 AND dl.engaged_score >= 0
        """
        args = []
        if layer:
            q += " AND dw.layer = ?"
            args.append(layer)
        rows = conn.execute(q, args).fetchall()
    except sqlite3.Error as e:
        return None, f"query failed: {type(e).__name__}: {e}"
    finally:
        conn.close()
    if not rows:
        return None, "no comparable pairs yet (need fit labels AND scored deliveries)"
    by = {}
    for lay, ws, ls in rows:
        agree, tie, dis = by.setdefault(lay, [0, 0, 0])
        if ws > ls:
            by[lay][0] += 1
        elif ws == ls:
            by[lay][1] += 1
        else:
            by[lay][2] += 1
    return by, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default=None)
    ap.add_argument("--min-pairs", type=int, default=30,
                    help="below this, report but flag as not yet interpretable")
    a = ap.parse_args()
    by, err = evaluate(a.layer)
    if err:
        print(err)
        return
    print(f"  {'layer':24s} {'pairs':>6} {'agree':>6} {'tie':>5} {'disagree':>9} {'agree%':>8}")
    for lay, (ag, tie, dis) in sorted(by.items(), key=lambda kv: -sum(kv[1])):
        n = ag + tie + dis
        decided = ag + dis
        pct = f"{100.0*ag/decided:.1f}" if decided else "-"
        flag = "" if n >= a.min_pairs else "   (n too low)"
        print(f"  {lay:24s} {n:>6} {ag:>6} {tie:>5} {dis:>9} {pct:>8}{flag}")
    print("\nTies are usually both-unengaged — common and not detector error.")
    print("Judge per layer: orientation material is expected to score low.")


if __name__ == "__main__":
    main()
