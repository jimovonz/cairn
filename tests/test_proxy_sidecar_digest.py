import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy import sidecar


def _cm(*topics):
    entries = ",".join('{"t":"fact","to":"%s","c":"x"}' % t for t in topics)
    return "\n\n[cm]: # '{\"e\":[%s],\"ok\":true}'" % entries


def test_topic_digest_dedup_and_order(tmp_path, monkeypatch):
    sid = "digesttest-order"
    path = sidecar.capture_path(sid)
    if os.path.exists(path):
        os.remove(path)
    try:
        sidecar.append_capture(sid, {"emitted_sha": "a", "cm": _cm("alpha", "beta")})
        sidecar.append_capture(sid, {"emitted_sha": "b", "cm": _cm("beta", "gamma")})
        digest = sidecar.load_topic_digest(sid)
        assert digest == "alpha; beta; gamma"  # deduped, insertion order
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_topic_digest_missing_file():
    assert sidecar.load_topic_digest("no-such-session-xyz") == ""


def test_topic_digest_skips_unparseable(tmp_path):
    sid = "digesttest-bad"
    path = sidecar.capture_path(sid)
    if os.path.exists(path):
        os.remove(path)
    try:
        sidecar.append_capture(sid, {"emitted_sha": "a", "cm": "\n\n[cm]: # '{bad json'"})
        sidecar.append_capture(sid, {"emitted_sha": "b", "cm": _cm("good")})
        assert sidecar.load_topic_digest(sid) == "good"
    finally:
        if os.path.exists(path):
            os.remove(path)
