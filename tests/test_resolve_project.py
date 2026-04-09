#!/usr/bin/env python3
"""Tests for resolve_project — CAIRN_PROJECT env var override of cwd-based labelling."""

import os
from unittest.mock import patch

from hooks.hook_helpers import resolve_project


# Verifies: resolve_project returns basename of cwd when no env var is set
def test_resolve_project_cwd_default():
    """Without CAIRN_PROJECT set, returns lowercased basename of cwd."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAIRN_PROJECT", None)
        assert resolve_project("/home/user/Projects/cairn") == "cairn"


# Verifies: resolve_project lowercases cwd basename
def test_resolve_project_cwd_lowercases():
    """Mixed-case directory names are lowercased."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAIRN_PROJECT", None)
        assert resolve_project("/home/user/Projects/MyProject") == "myproject"


# Verifies: resolve_project strips trailing slash before basename
def test_resolve_project_cwd_strips_trailing_slash():
    """Trailing slash on cwd doesn't produce empty basename."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAIRN_PROJECT", None)
        assert resolve_project("/home/user/Projects/cairn/") == "cairn"


# Verifies: CAIRN_PROJECT env var overrides cwd basename
def test_resolve_project_env_var_overrides_cwd():
    """CAIRN_PROJECT takes precedence over cwd-derived name."""
    with patch.dict(os.environ, {"CAIRN_PROJECT": "alimento"}):
        assert resolve_project("/home/user/Projects/temp") == "alimento"


# Verifies: CAIRN_PROJECT env var is lowercased
def test_resolve_project_env_var_lowercased():
    """Env var value is lowercased to match cwd-based behaviour."""
    with patch.dict(os.environ, {"CAIRN_PROJECT": "Alimento"}):
        assert resolve_project("/home/user/Projects/temp") == "alimento"


# Verifies: empty CAIRN_PROJECT falls back to cwd
def test_resolve_project_empty_env_var_falls_back():
    """Empty string env var doesn't override — falls back to cwd."""
    with patch.dict(os.environ, {"CAIRN_PROJECT": ""}):
        assert resolve_project("/home/user/Projects/cairn") == "cairn"


# Verifies: whitespace-only CAIRN_PROJECT falls back to cwd
def test_resolve_project_whitespace_env_var_falls_back():
    """Whitespace-only env var is treated as unset."""
    with patch.dict(os.environ, {"CAIRN_PROJECT": "   "}):
        assert resolve_project("/home/user/Projects/cairn") == "cairn"


# Verifies: empty cwd with no override returns empty string
def test_resolve_project_empty_cwd_no_override():
    """No env var and empty cwd returns empty string (caller decides what to do)."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAIRN_PROJECT", None)
        assert resolve_project("") == ""


# Verifies: env var overrides even when cwd is empty
def test_resolve_project_env_var_with_empty_cwd():
    """Env var works regardless of cwd."""
    with patch.dict(os.environ, {"CAIRN_PROJECT": "benchmark"}):
        assert resolve_project("") == "benchmark"


# Verifies: env var with leading/trailing whitespace is stripped before lowercase
def test_resolve_project_env_var_stripped():
    """Env var value with surrounding whitespace is stripped."""
    with patch.dict(os.environ, {"CAIRN_PROJECT": "  alimento  "}):
        assert resolve_project("/anywhere") == "alimento"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
