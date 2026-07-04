"""Regression: inline references to link-def markers must NOT be stripped.

Root cause of a real truncation bug (2026-07-05): the response stripper matched
any occurrence of ``[cm]: # `` / ``[cairn-note]: # `` anywhere in assistant text
and suppressed to end-of-message. When the model documented the block format
inline (e.g. `` `[cm]: # '{...}'` `` in prose), the visible reply was truncated
at that point and the real trailing block was swallowed into an unparseable
capture. Fix: link-def openers are only recognised at line start.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy.cm_filter import ResponseTextStripper

REAL_CM = "\n\n[cm]: # '{\"e\":[{\"t\":\"fact\",\"to\":\"real\",\"c\":\"x\"}],\"ok\":true}'"


def strip(msg, chunk=None):
    s = ResponseTextStripper()
    if chunk is None:
        vis = s.feed(msg) + s.flush()
    else:
        vis = "".join(s.feed(msg[i:i + chunk]) for i in range(0, len(msg), chunk)) + s.flush()
    assert vis + "".join(s.captured) == s.original  # byte-exact invariant
    return vis, s


def test_inline_cm_reference_preserved_and_real_block_stripped():
    msg = ("Turns come back as [cm: captured] instead of the full "
           "`[cm]: # '{...}'` block.\n\nMore prose the user must see." + REAL_CM)
    vis, s = strip(msg)
    assert "`[cm]: # '{...}'` block" in vis        # inline ref survives
    assert "More prose the user must see." in vis   # nothing after it truncated
    assert '"to":"real"' not in vis                 # real trailing block removed
    assert s.cm_block.endswith("\"ok\":true}'")     # captured correctly


def test_inline_cairn_note_reference_preserved():
    msg = "Use the `[cairn-note]: # '{...}'` form for mid-response notes." + REAL_CM
    vis, s = strip(msg)
    assert "`[cairn-note]: # '{...}'` form" in vis
    assert "[cairn-note]" in vis


def test_inline_survives_char_by_char_streaming():
    msg = ("Prose with inline `[cm]: # '{}'` reference and a tail." + REAL_CM)
    vis_whole, _ = strip(msg)
    vis_chunked, _ = strip(msg, chunk=1)
    assert vis_whole == vis_chunked
    assert "inline `[cm]: # '{}'` reference and a tail." in vis_chunked


def test_real_block_still_stripped_at_line_start():
    msg = "Answer body.\nSecond line." + REAL_CM
    vis, s = strip(msg)
    assert vis == "Answer body.\nSecond line."
    assert s.cm_block == REAL_CM


def test_block_as_entire_message_stripped():
    only = "[cm]: # '{\"ok\":true}'"
    vis, s = strip(only)
    assert vis == ""
    assert s.cm_block == only
