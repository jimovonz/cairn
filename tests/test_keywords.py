"""Tests for cairn.keywords.prompt_keywords — phrase / identifier extraction."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from cairn import keywords as kw_mod


@pytest.fixture(autouse=True)
def _reset_extractor_cache():
    kw_mod._extractor_cache.clear()
    yield
    kw_mod._extractor_cache.clear()


def test_compound_identifier_survives_intact():
    """Snake-case + kebab-case identifiers must not be fragmented."""
    text = "Help me debug NL80211_CMD_TRIGGER_SCAN via cairn-calibration-analyser"
    result = kw_mod.prompt_keywords(text)
    assert "NL80211_CMD_TRIGGER_SCAN" in result
    assert "cairn-calibration-analyser" in result


def test_stopwords_filtered():
    """Generic stopwords / function words must not appear as keywords."""
    text = "what is the best approach for debugging an error"
    result = kw_mod.prompt_keywords(text)
    for sw in ("what", "is", "the", "for", "an"):
        assert sw not in result
    # Substantive content terms should be retained (via YAKE).
    assert any("debug" in r or "error" in r for r in result)


def test_extraction_caps_count():
    """`max_n` is a hard cap on the returned set size."""
    text = (
        "deploy pipeline rollback strategy release management "
        "feature flag gradual rollout monitoring alerting "
        "incident response postmortem retrospective"
    )
    result = kw_mod.prompt_keywords(text, max_n=5)
    assert len(result) <= 5
    result2 = kw_mod.prompt_keywords(text, max_n=3)
    assert len(result2) <= 3


def test_empty_input_returns_empty_set():
    assert kw_mod.prompt_keywords("") == set()
    assert kw_mod.prompt_keywords(None) == set()  # type: ignore[arg-type]


def test_falls_back_when_yake_import_fails(monkeypatch):
    """If yake cannot be imported, fall back to naive stopword filtering."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "yake":
            raise ImportError("yake not available (mocked)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    kw_mod._extractor_cache.clear()

    text = "what is the best approach for debugging an error"
    result = kw_mod.prompt_keywords(text)
    # Naive fallback: stopwords filtered, content words present.
    assert "what" not in result
    assert "the" not in result
    assert "debugging" in result
    assert "approach" in result
    assert "best" in result  # naive path keeps all non-stopwords ≥3 chars
