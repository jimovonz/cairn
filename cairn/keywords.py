"""Phrase-aware keyword extraction for retrieval and calibration injection.

Combines a regex pre-pass for compound identifiers (FOO_BAR_BAZ,
cairn-calibration-analyser, snake_case paths) — which YAKE would
otherwise fragment on `_`/`-` — with YAKE's unsupervised phrase
extraction. Falls back to a naive stopword-filtered split if YAKE is
unavailable.

Cheap: no model load, ~50KB import. Extractor instances are memoised
per (max_ngram, top) so the regexes only compile once per process.
"""

from __future__ import annotations

import re
from typing import Optional


_COMPOUND_RE = re.compile(r"[A-Za-z0-9]+(?:[_\-/.][A-Za-z0-9]+)+")

# Conservative fallback used when YAKE import fails. Mirrors the
# pre-existing extract_query_terms stopword set so callers see no
# behavioural cliff if the dep goes missing.
_FALLBACK_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "are", "be",
    "has", "had", "have", "do", "did", "does", "will", "can", "could",
    "would", "should", "may", "might", "not", "no", "what", "when",
    "where", "who", "how", "why", "that", "this", "these", "those",
    "my", "your", "his", "her", "our", "their", "me", "you", "him",
    "them", "about", "into", "over", "after", "before", "between",
    "other", "some", "any", "all", "just", "also", "than", "then",
    "very", "too", "here", "there", "been", "being", "were",
})


_extractor_cache: dict[tuple[int, int], object] = {}


def _yake_extractor(max_ngram: int, top: int):
    """Memoised YAKE extractor. Raises ImportError if yake unavailable."""
    key = (max_ngram, top)
    cached = _extractor_cache.get(key)
    if cached is not None:
        return cached
    import yake  # noqa: F401 — ImportError propagates to caller
    extractor = yake.KeywordExtractor(
        lan="en", n=max_ngram, dedupLim=0.9, top=top
    )
    _extractor_cache[key] = extractor
    return extractor


def _naive_fallback(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    meaningful = {w for w in words if len(w) > 2 and w not in _FALLBACK_STOPWORDS}
    return meaningful if meaningful else {w for w in words if len(w) > 2}


def _extract_compounds(text: str) -> list[str]:
    """Compound identifiers worth preserving verbatim (case + structure)."""
    return [m.group(0) for m in _COMPOUND_RE.finditer(text)]


def prompt_keywords(text: str, max_n: int = 8, max_ngram: int = 3) -> set[str]:
    """Extract up to `max_n` high-signal phrases / identifiers from text.

    Compounds (containing `_`, `-`, `/`, or `.`) are preserved verbatim;
    natural-language phrases come from YAKE. Returns lowercase strings
    except for compound identifiers (which keep their original case so
    grep / FTS matching of symbol names stays case-aware).

    Falls back to naive stopword filtering if yake import fails.
    """
    if not text:
        return set()

    compounds = _extract_compounds(text)

    try:
        extractor = _yake_extractor(max_ngram, max(max_n * 2, 4))
        yake_pairs = extractor.extract_keywords(text)
    except ImportError:
        return _naive_fallback(text)
    except Exception:
        # YAKE present but failed (malformed input, internal error): we
        # prefer something over nothing.
        return _naive_fallback(text)

    yake_phrases = [p.lower() for p, _score in yake_pairs]

    out: list[str] = []
    seen: set[str] = set()

    # Compounds first — high-precision identifier matches.
    for c in compounds:
        key = c.lower()
        if key in seen:
            continue
        out.append(c)
        seen.add(key)
        if len(out) >= max_n:
            return set(out)

    for phrase in yake_phrases:
        if len(phrase) <= 2 or phrase in seen:
            continue
        out.append(phrase)
        seen.add(phrase)
        if len(out) >= max_n:
            break

    if not out:
        return _naive_fallback(text)
    return set(out)
