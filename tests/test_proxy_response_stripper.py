import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy.cm_filter import ResponseTextStripper

CM = "\n\n[cm]: # '{\"ok\":true,\"ctx\":\"s\",\"kw\":[\"x\"]}'"
NOTE_TAG = "<memory_note>fact/topic: an observation here</memory_note>"
NOTE_LINE = "[cairn-note]: # '{\"type\":\"fact\",\"topic\":\"t\",\"content\":\"c\"}'"


def run(pieces):
    s = ResponseTextStripper()
    emitted = "".join(s.feed(p) for p in pieces)
    emitted += s.flush()
    return emitted, s


def test_trailing_cm_stripped_and_captured():
    body = "Here is the answer.\nSecond line."
    full = body + CM
    emitted, s = run([full])
    assert emitted == body
    assert "".join(s.captured) == CM
    assert s.original == full


def test_inline_memnote_removed_text_continues():
    full = "Before note. " + NOTE_TAG + " After note." + CM
    emitted, s = run([full])
    assert emitted == "Before note.  After note."
    assert NOTE_TAG in s.captured
    assert any(c.startswith("\n\n[cm]") for c in s.captured)
    assert s.original == full


def test_cairn_note_line_removed():
    full = "Answer text.\n" + NOTE_LINE + "\nMore text." + CM
    emitted, s = run([full])
    assert NOTE_LINE not in emitted
    assert "Answer text." in emitted and "More text." in emitted
    assert NOTE_LINE in s.captured
    assert s.original == full


def test_no_artifacts_passthrough():
    body = "Just a normal answer, nothing to strip."
    emitted, s = run([body[:5], body[5:]])
    assert emitted == body
    assert s.captured == []
    assert not s.stripped


def test_char_by_char_all_artifacts():
    full = "Lead. " + NOTE_TAG + " mid " + NOTE_LINE + "\n tail" + CM
    emitted, s = run(list(full))
    assert s.original == full
    assert "[cm]" not in emitted
    assert "<memory_note>" not in emitted
    assert "[cairn-note]" not in emitted


def test_random_chunking_invariant():
    rnd = random.Random(99)
    variants = [
        "Plain answer only.",
        "Answer " + NOTE_TAG + " continues." + CM,
        "Lead.\n" + NOTE_LINE + "\nrest." + CM,
        "Multi " + NOTE_TAG + " and " + NOTE_TAG + " notes." + CM,
        "Answer with no cm but " + NOTE_TAG + " inline.",
        "Body text." + CM,
    ]
    for _ in range(400):
        text = rnd.choice(variants)
        i, pieces = 0, []
        while i < len(text):
            n = rnd.randint(1, 9)
            pieces.append(text[i:i+n]); i += n
        emitted, s = run(pieces)
        assert s.original == text, (text, pieces)
        assert "[cm]: # " not in emitted
        assert "<memory_note>" not in emitted
        assert "[cairn-note]: # " not in emitted


def test_multiple_memnotes_captured_separately():
    full = "A " + NOTE_TAG + " B " + NOTE_TAG + " C" + CM
    emitted, s = run([full])
    assert emitted == "A  B  C"
    notes = [c for c in s.captured if c.startswith("<memory_note>")]
    assert len(notes) == 2
