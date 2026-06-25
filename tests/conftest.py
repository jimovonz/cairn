"""Shared pytest fixtures / session setup.

Hermetic proxy state: the cairn dev shell exports CAIRN_PROXY_ENABLED=1 (the API
proxy is installed and enabled on dev machines). That env var leaks into the test
process and flips the hooks to artifact-free proxy/sidecar staging — so the
prompt/pretool/posttool/stop hooks emit a *pointer* ("context is now available")
instead of the inline `CAIRN CONTEXT: <entry>` JSON that the hook/L3 tests assert.
The result is ~18 spurious failures that only reproduce on a machine with the
proxy enabled (green in CI, red locally).

Pop the var at conftest import time — BEFORE any test module imports cairn.config,
so config.PROXY_ENABLED computes False from the start (matching CI) — and also
patch the already-resolved flag defensively per test. Tests that exercise the
proxy set it explicitly themselves (e.g. proxy_e2e_check passes it in a subprocess
env), so they are unaffected.
"""
import os

# Module-level: runs when pytest imports this conftest, before test modules import
# cairn.config. Ensures PROXY_ENABLED is resolved to False at import.
os.environ.pop("CAIRN_PROXY_ENABLED", None)

import pytest


@pytest.fixture(autouse=True)
def _hermetic_proxy_state(monkeypatch):
    """Belt-and-braces: keep the env unset and force config.PROXY_ENABLED False in
    case cairn.config was imported before this conftest (e.g. by a plugin)."""
    monkeypatch.delenv("CAIRN_PROXY_ENABLED", raising=False)
    try:
        import cairn.config as _cfg
        monkeypatch.setattr(_cfg, "PROXY_ENABLED", False, raising=False)
    except Exception:
        pass
