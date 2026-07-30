"""Bootstrap deliveries must be keyed by the real context, not the layer label.

`build_context_xml(query=...)` doubles as a display label and, when no
context_text is given, as the delivery key AND the text embedded into
context_vec. Both bootstrap layers passed only the label, so every session's
rows shared one constant string ("project standing context"). Consequences
measured before the fix: per-turn grouping pooled 70 unrelated sessions into a
single pseudo-query worth 68,700 training pairs, and the stored context_vec —
described in-tree as the join key for empirical context-targeting — embedded
the literal words of the label for the highest-volume layer.
"""

import os
import sys
import tempfile
from unittest.mock import patch

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cairn import init_db

HOOKS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hooks"))
sys.path.insert(0, HOOKS)


def _eph():
    td = tempfile.mkdtemp()
    eph = os.path.join(td, "eph.db")
    init_db.init_ephemeral(eph)
    return eph


def _logged_context(eph):
    c = sqlite3.connect(eph)
    try:
        return [r[0] for r in c.execute(
            "SELECT context_text FROM memory_deliveries")]
    finally:
        c.close()


def _emit(eph, **kw):
    from hook_helpers import build_context_xml
    results = [{"id": 1, "type": "fact", "topic": "t", "content": "c",
                "project": "p", "confidence": 0.8, "score": 0.8, "similarity": 0}]
    with patch("cairn.relevance._eph_path", lambda p=None: eph):
        return build_context_xml("project standing context", "p", "project-bootstrap",
                                 results, [], session_id="s", **kw)


def test_delivery_is_keyed_by_context_text_when_given():
    eph = _eph()
    _emit(eph, context_text="the user asked about tank slosh modelling")
    assert _logged_context(eph) == ["the user asked about tank slosh modelling"]


def test_delivery_falls_back_to_the_label_when_context_is_absent():
    """The degradation path: an empty first prompt must not crash or lose the
    row, it just yields the old, less useful key."""
    eph = _eph()
    _emit(eph)
    assert _logged_context(eph) == ["project standing context"]


def test_both_bootstrap_call_sites_pass_context_text():
    """The bug was at the CALL SITES, not in build_context_xml — which has
    accepted context_text all along. A unit test of the helper alone would have
    stayed green throughout the defect."""
    src = open(os.path.join(HOOKS, "prompt_hook.py")).read()
    for label in ('"project standing context"', '"behavioural corrections"'):
        i = src.index(label)
        call = src[i:i + 400]
        assert "context_text=" in call, f"{label} call site does not pass context_text"
        assert "build_context_window" in call, f"{label} must use the shared window builder"


def test_project_bootstrap_accepts_the_user_message():
    import prompt_hook
    import inspect
    assert "user_message" in inspect.signature(prompt_hook.project_bootstrap).parameters
