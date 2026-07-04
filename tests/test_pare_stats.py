"""Context-paring Phase 1.5 — savings metric accumulation + reporting."""
import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy import request_inject as ri, sidecar


def _cleanup(sid):
    p = sidecar.pare_stats_path(sid)
    if os.path.exists(p):
        os.remove(p)


def test_inject_populates_stats():
    stripped = "Answer."
    cm = "\n\n[cm]: # '{\"ok\":true}'"
    sha = hashlib.sha256(stripped.encode()).hexdigest()
    data = {"messages": [{"role": "assistant", "content": stripped}]}
    stats = {}
    ri.inject_cm_markers(data, {sha: cm}, stats=stats)
    assert stats["blocks_replaced_chars"] == len(cm)
    assert stats["marker_chars"] > 0


def test_digest_stats_only_on_actual_injection():
    data = {"messages": [{"role": "user", "content": "latest"}]}
    stats = {}
    ri.inject_cm_digest(data, "topicA", stats=stats)
    assert stats["digest_chars"] > 0
    # second call is idempotent (sentinel present) — must NOT add cost again
    stats2 = {}
    ri.inject_cm_digest(data, "topicA", stats=stats2)
    assert "digest_chars" not in stats2 or stats2.get("digest_chars", 0) == 0


def test_record_accumulates_across_requests():
    sid = "paretest-accum"
    _cleanup(sid)
    try:
        for _ in range(4):
            sidecar.record_pare_savings(sid, {
                "blocks_replaced_chars": 1000, "marker_chars": 15, "digest_chars": 100})
        rec = sidecar.load_pare_stats(sid)
        assert rec["requests"] == 4
        assert rec["blocks_replaced_chars"] == 4000
        assert rec["net_saved_chars"] == 4 * (1000 - 15 - 100)
        assert rec["max_digest_chars"] == 100
    finally:
        _cleanup(sid)


def test_record_noop_on_all_zero():
    sid = "paretest-zero"
    _cleanup(sid)
    try:
        sidecar.record_pare_savings(sid, {"blocks_replaced_chars": 0, "marker_chars": 0, "digest_chars": 0})
        assert sidecar.load_pare_stats(sid) is None  # nothing written
    finally:
        _cleanup(sid)


def test_max_digest_tracks_peak_not_sum():
    sid = "paretest-peak"
    _cleanup(sid)
    try:
        sidecar.record_pare_savings(sid, {"blocks_replaced_chars": 500, "digest_chars": 50})
        sidecar.record_pare_savings(sid, {"blocks_replaced_chars": 500, "digest_chars": 200})
        sidecar.record_pare_savings(sid, {"blocks_replaced_chars": 500, "digest_chars": 120})
        rec = sidecar.load_pare_stats(sid)
        assert rec["max_digest_chars"] == 200  # peak, not 370
    finally:
        _cleanup(sid)


def test_load_all_aggregates_sessions():
    sids = ["paretest-all-1", "paretest-all-2"]
    for s in sids:
        _cleanup(s)
    try:
        sidecar.record_pare_savings(sids[0], {"blocks_replaced_chars": 100})
        sidecar.record_pare_savings(sids[1], {"blocks_replaced_chars": 200})
        allrows = sidecar.load_all_pare_stats()
        got = {r["session"]: r for r in allrows if r["session"] in sids}
        assert got[sids[0]]["blocks_replaced_chars"] == 100
        assert got[sids[1]]["blocks_replaced_chars"] == 200
    finally:
        for s in sids:
            _cleanup(s)
