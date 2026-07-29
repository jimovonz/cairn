"""CLAUDE.md must not drift from cairn/config.py.

Every wrong claim I made about retrieval in one session traced to reading the
docs instead of the config: the prefilter was documented "default off" while
ON, the bge floor was documented 0.0005 while 0.10, and SCORE_W_CONFIDENCE /
SCORE_W_RECENCY were undocumented at 0.0 — so reading _recency_decay() and
concluding age affects ranking looked reasonable and was wrong.

Prose describing a constant is a cache with no invalidation. This is the
invalidation. Two rules:

  1. Values the doc DOES quote must match config.
  2. Machine-managed values must NOT be quoted at all — ab_selfmod rewrites
     GENERATION_PROMPT_VERSION on every promotion, so any number written in
     prose is wrong by the next cron run. Point at the symbol instead.
"""
import os
import re

import pytest

from cairn import config as C

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CLAUDE.md")


@pytest.fixture(scope="module")
def doc():
    with open(DOC, encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("pattern,actual,label", [
    (r"similarity ([0-9.]+)", C.SCORE_W_SIMILARITY, "SCORE_W_SIMILARITY"),
    (r"keywords ([0-9.]+)", C.SCORE_W_KEYWORDS, "SCORE_W_KEYWORDS"),
    (r"scope ([0-9.]+)", C.SCORE_W_SCOPE, "SCORE_W_SCOPE"),
    (r"confidence ([0-9.]+) and recency", C.SCORE_W_CONFIDENCE, "SCORE_W_CONFIDENCE"),
    (r"recency ([0-9.]+) — both deliberately disabled", C.SCORE_W_RECENCY, "SCORE_W_RECENCY"),
    (r"floor \*\*([0-9.]+)\*\*", C.CROSS_ENCODER_SCORE_FLOOR_CUDA, "CROSS_ENCODER_SCORE_FLOOR_CUDA"),
    (r"RERANKER_MIN_VRAM_GB` \((\d+) GB\)", C.RERANKER_MIN_VRAM_GB, "RERANKER_MIN_VRAM_GB"),
])
def test_documented_value_matches_config(doc, pattern, actual, label):
    m = re.search(pattern, doc)
    assert m, f"CLAUDE.md no longer states {label} in the expected form ({pattern!r})"
    assert abs(float(m.group(1)) - float(actual)) < 1e-9, (
        f"CLAUDE.md says {label}={m.group(1)} but config.py has {actual}")


def test_prefilter_flag_documented_state_matches(doc):
    on = re.search(r"RELEVANCE_PREFILTER_ENABLED`, \*\*ON\*\*", doc)
    off = re.search(r"RELEVANCE_PREFILTER_ENABLED`, default off", doc)
    assert not (on and off), "CLAUDE.md states both ON and off for the prefilter"
    assert bool(on) == bool(C.RELEVANCE_PREFILTER_ENABLED), (
        f"CLAUDE.md says prefilter {'ON' if on else 'off'} but config has "
        f"{C.RELEVANCE_PREFILTER_ENABLED}")


def test_machine_managed_version_not_frozen_in_prose(doc):
    """ab_selfmod rewrites this on promotion; a quoted number is wrong by design."""
    frozen = re.findall(r"`(gen[AB]-v\d+)`", doc)
    assert not frozen, (
        f"CLAUDE.md hardcodes machine-managed generation version(s) {frozen}. "
        "ab_selfmod rewrites GENERATION_PROMPT_VERSION on every promotion — "
        "reference the config symbol instead of quoting a value.")
