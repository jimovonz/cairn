import sys, os, json, importlib, glob
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy import server, sidecar


@pytest.fixture(autouse=True)
def _clean_pare_locks():
    """Session-start pare locks persist to .staged_context; clear the test
    sessions' locks before and after each test so order-independence holds."""
    def _rm():
        for f in glob.glob(os.path.join(sidecar.staged_dir(), "pytest-*_pare_lock.json")):
            try:
                os.remove(f)
            except OSError:
                pass
    _rm()
    yield
    _rm()


def _tools():
    return [
        {"name": "Bash", "description": "run"},
        {"name": "Edit", "description": "edit"},
        {"name": "Write", "description": "write"},
        {"name": "NotebookEdit", "description": "nb"},
        {"name": "Read", "description": "read"},
        {"name": "mcp__claude_ai_Gmail__search_threads", "description": "gmail"},
        {"name": "mcp__claude_ai_Google_Drive__search_files", "description": "drive"},
    ]


def test_pare_strips_mcp_and_cch_denied():
    data = {"tools": _tools(), "messages": []}
    removed = server.pare_tools(data)
    assert removed == 5  # Edit, Write, NotebookEdit + 2 mcp
    names = [t["name"] for t in data["tools"]]
    # kept: Bash, Read (and NOT the denied/mcp)
    assert names == ["Bash", "Read"]


def test_history_guard_keeps_referenced_tool():
    # An mcp tool that was actually called in history must NOT be stripped
    data = {
        "tools": _tools(),
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "mcp__claude_ai_Gmail__search_threads",
                 "id": "x", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "ok"}]},
        ],
    }
    server.pare_tools(data)
    names = [t["name"] for t in data["tools"]]
    assert "mcp__claude_ai_Gmail__search_threads" in names   # protected by guard
    assert "mcp__claude_ai_Google_Drive__search_files" not in names  # still stripped
    assert "Edit" not in names


def test_deterministic_byte_stable():
    a = {"tools": _tools(), "messages": []}
    b = {"tools": _tools(), "messages": []}
    server.pare_tools(a); server.pare_tools(b)
    assert json.dumps(a["tools"]) == json.dumps(b["tools"])


def test_noop_when_no_tools():
    for empty in ({"messages": []}, {"tools": [], "messages": []}, {"tools": None, "messages": []}):
        assert server.pare_tools(empty) == 0


def test_flag_gate_off_passes_tools_through(monkeypatch):
    # With the flag off, _rewrite_request must not touch tools.
    from cairn import config
    monkeypatch.setattr(config, "PARE_TOOLS_ENABLED", False)
    body = json.dumps({
        "system": [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}],
        "tools": _tools(),
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    out = json.loads(server._rewrite_request(body, "pytest-tp-off"))
    assert len(out["tools"]) == len(_tools())


def test_flag_gate_on_pares(monkeypatch):
    from cairn import config
    monkeypatch.setattr(config, "PARE_TOOLS_ENABLED", True)
    monkeypatch.setattr(config, "PARE_TOOLS_CCH_DENIED", frozenset({"Edit", "Write", "NotebookEdit"}))
    body = json.dumps({
        "system": [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}],
        "tools": _tools(),
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    out = json.loads(server._rewrite_request(body, "pytest-tp-on"))
    names = [t["name"] for t in out["tools"]]
    assert names == ["Bash", "Read"]


def _body():
    return json.dumps({
        "system": [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}],
        "tools": _tools(),
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()


def test_session_lock_freezes_on_decision(monkeypatch):
    # First tool-bearing request with the flag ON freezes ON; a mid-session flip
    # to OFF must NOT un-pare that session (would rebuild the cached prefix).
    from cairn import config
    monkeypatch.setattr(config, "PARE_TOOLS_CCH_DENIED", frozenset({"Edit", "Write", "NotebookEdit"}))
    sid = "pytest-lock-on"
    monkeypatch.setattr(config, "PARE_TOOLS_ENABLED", True)
    out1 = json.loads(server._rewrite_request(_body(), sid))
    assert [t["name"] for t in out1["tools"]] == ["Bash", "Read"]   # pared
    assert sidecar.read_pare_lock(sid) is True                       # recorded
    monkeypatch.setattr(config, "PARE_TOOLS_ENABLED", False)         # mid-session flip
    out2 = json.loads(server._rewrite_request(_body(), sid))
    assert [t["name"] for t in out2["tools"]] == ["Bash", "Read"]   # still pared (frozen)


def test_session_lock_freezes_off_decision(monkeypatch):
    # First request with the flag OFF freezes OFF; a mid-session flip to ON must
    # NOT start paring that session.
    from cairn import config
    monkeypatch.setattr(config, "PARE_TOOLS_CCH_DENIED", frozenset({"Edit", "Write", "NotebookEdit"}))
    sid = "pytest-lock-off"
    monkeypatch.setattr(config, "PARE_TOOLS_ENABLED", False)
    out1 = json.loads(server._rewrite_request(_body(), sid))
    assert len(out1["tools"]) == len(_tools())                       # untouched
    assert sidecar.read_pare_lock(sid) is False                      # recorded
    monkeypatch.setattr(config, "PARE_TOOLS_ENABLED", True)          # mid-session flip
    out2 = json.loads(server._rewrite_request(_body(), sid))
    assert len(out2["tools"]) == len(_tools())                       # still untouched (frozen)


def test_new_session_reads_live_config(monkeypatch):
    # A session locked OFF must not affect a *different* new session, which
    # reads the current live config at its own first request.
    from cairn import config
    monkeypatch.setattr(config, "PARE_TOOLS_CCH_DENIED", frozenset({"Edit", "Write", "NotebookEdit"}))
    monkeypatch.setattr(config, "PARE_TOOLS_ENABLED", False)
    json.loads(server._rewrite_request(_body(), "pytest-newsess-a"))       # locks A OFF
    monkeypatch.setattr(config, "PARE_TOOLS_ENABLED", True)
    out = json.loads(server._rewrite_request(_body(), "pytest-newsess-b"))  # B reads live ON
    assert [t["name"] for t in out["tools"]] == ["Bash", "Read"]
