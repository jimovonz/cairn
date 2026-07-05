import sys, os, json, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy import server


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
    out = json.loads(server._rewrite_request(body, "pytest-tool-pare"))
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
    out = json.loads(server._rewrite_request(body, "pytest-tool-pare"))
    names = [t["name"] for t in out["tools"]]
    assert names == ["Bash", "Read"]
