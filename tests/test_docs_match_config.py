"""The prose docs must not drift from cairn/config.py.

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
import glob
import os
import re

import pytest

from cairn import config as C

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The prose lives in CLAUDE.md, in the on-demand skills it delegates to, and in
# README.md. Scan all three: a section moving between them must not take its
# drift check with it, and README.md quoted a stale reranker floor and a frozen
# genA-vN for several releases precisely because it was not scanned.
DOCS = ([os.path.join(_ROOT, "CLAUDE.md"), os.path.join(_ROOT, "README.md")]
        + sorted(glob.glob(os.path.join(_ROOT, ".claude", "skills", "*", "SKILL.md"))))


@pytest.fixture(scope="module")
def doc():
    parts = []
    for path in DOCS:
        with open(path, encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


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
    assert m, f"No doc states {label} in the expected form ({pattern!r})"
    assert abs(float(m.group(1)) - float(actual)) < 1e-9, (
        f"Docs say {label}={m.group(1)} but config.py has {actual}")


def test_prefilter_flag_documented_state_matches(doc):
    on = re.search(r"RELEVANCE_PREFILTER_ENABLED`, \*\*ON\*\*", doc)
    off = re.search(r"RELEVANCE_PREFILTER_ENABLED`, default off", doc)
    assert not (on and off), "Docs state both ON and off for the prefilter"
    assert bool(on) == bool(C.RELEVANCE_PREFILTER_ENABLED), (
        f"Docs say prefilter {'ON' if on else 'off'} but config has "
        f"{C.RELEVANCE_PREFILTER_ENABLED}")


def test_machine_managed_version_not_frozen_in_prose(doc):
    """ab_selfmod rewrites this on promotion; a quoted number is wrong by design."""
    frozen = re.findall(r"`(gen[AB]-v\d+)`", doc)
    assert not frozen, (
        f"Docs hardcode machine-managed generation version(s) {frozen}. "
        "ab_selfmod rewrites GENERATION_PROMPT_VERSION on every promotion — "
        "reference the config symbol instead of quoting a value.")
