import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy.cm_filter import CmTextStripper, MARKER

CM = "\n\n[cm]: # '{\"ok\":true,\"ctx\":\"s\",\"kw\":[\"x\"]}'"
BODY = "Here is a normal answer with some length to it.\nSecond line of the reply."
FULL = BODY + CM


def _run(pieces):
    s = CmTextStripper()
    emitted = "".join(s.feed(p) for p in pieces)
    emitted += s.flush()
    return emitted, s


def test_invariant_and_strip_single_piece():
    emitted, s = _run([FULL])
    assert emitted + s.captured == FULL          # byte-exact reconstruct
    assert MARKER not in emitted                   # block hidden from display
    assert s.captured.startswith("\n\n[cm]: # ")   # captured verbatim incl separator
    assert s.stripped


def test_marker_split_across_feeds():
    # split right inside the marker
    cut = BODY.find  # noqa
    pieces = [BODY + "\n\n[c", "m]: # " + "'{\"ok\":true}'"]
    emitted, s = _run(pieces)
    full = "".join(pieces)
    assert emitted + s.captured == full
    assert "[cm]" not in emitted
    assert s.stripped


def test_no_marker_passthrough():
    emitted, s = _run([BODY[:10], BODY[10:]])
    assert emitted == BODY                          # nothing held back at flush
    assert s.captured == ""
    assert not s.stripped


def test_char_by_char_streaming():
    emitted, s = _run(list(FULL))                   # one char at a time
    assert emitted + s.captured == FULL
    assert MARKER not in emitted
    assert emitted == BODY                          # exactly the body survives


def test_random_chunking_invariant():
    rnd = random.Random(1234)
    for _ in range(200):
        text = BODY + (CM if rnd.random() < 0.7 else "")
        i, pieces = 0, []
        while i < len(text):
            n = rnd.randint(1, 7)
            pieces.append(text[i:i+n]); i += n
        emitted, s = _run(pieces)
        assert emitted + s.captured == text, (pieces,)
        if CM in text:
            assert MARKER not in emitted and s.stripped
        else:
            assert emitted == text and not s.stripped
