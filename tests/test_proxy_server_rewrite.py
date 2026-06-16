import sys, os, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy import server, sidecar

CM = "\n\n[cm]: # '{\"ok\":true}'"


def _clean(session):
    for suf in ("_inject_bootstrap.txt", "_inject_prompt.txt", "_cm_capture.jsonl"):
        try:
            os.unlink(sidecar._path(session, suf))
        except FileNotFoundError:
            pass


def test_rewrite_injects_all_three(tmp_session="pytest-sess-1"):
    _clean(tmp_session)
    stripped = "Prior answer."
    sidecar.append_capture(tmp_session, {
        "emitted_sha": hashlib.sha256(stripped.encode()).hexdigest(), "cm": CM, "notes": []})
    sidecar.write_bootstrap(tmp_session, "BOOTSTRAP-MEM")
    sidecar.append_prompt_context(tmp_session, "PER-PROMPT-CTX")

    body = json.dumps({
        "system": [{"type": "text", "text": "Sys", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Bash", "description": "run"}],
        "messages": [
            {"role": "assistant", "content": stripped},
            {"role": "user", "content": "new question"},
        ],
    }).encode()

    out = server._rewrite_request(body, tmp_session)
    data = json.loads(out)
    # reinjected cm onto the assistant turn
    assert data["messages"][0]["content"] == stripped + CM
    # bootstrap in system with relocated cache breakpoint
    assert "BOOTSTRAP-MEM" in data["system"][-1]["text"]
    assert data["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # per-prompt context on the last user turn
    assert "PER-PROMPT-CTX" in json.dumps(data["messages"][-1]["content"])
    # prompt context consumed (one-shot)
    assert sidecar.consume_prompt_context(tmp_session) == ""
    _clean(tmp_session)


def test_rewrite_failopen_on_bad_json():
    assert server._rewrite_request(b"not json", "s") == b"not json"


def test_rewrite_skips_non_message_bodies():
    body = json.dumps({"model": "x", "no_messages": True}).encode()
    assert server._rewrite_request(body, "s") == body


def test_rewrite_noop_without_session():
    body = json.dumps({"messages": []}).encode()
    assert server._rewrite_request(body, "") == body


def test_auxiliary_request_without_tools_does_not_consume(tmp_session="pytest-aux-1"):
    _clean(tmp_session)
    sidecar.write_bootstrap(tmp_session, "BOOT-X")
    sidecar.append_prompt_context(tmp_session, "PROMPT-CTX-X")
    # an auxiliary CC call: no tools, tiny single user message (title/topic gen)
    aux = json.dumps({"model": "claude-haiku", "max_tokens": 32,
                      "messages": [{"role": "user", "content": "summarize this"}]}).encode()
    out = server._rewrite_request(aux, tmp_session)
    data = json.loads(out)
    # nothing injected, and crucially the per-prompt sidecar is NOT consumed
    assert "PROMPT-CTX-X" not in json.dumps(data)
    assert "BOOT-X" not in json.dumps(data)
    assert sidecar.consume_prompt_context(tmp_session) == "PROMPT-CTX-X"  # still there
    _clean(tmp_session)


def test_agentic_request_with_tools_consumes(tmp_session="pytest-aux-2"):
    _clean(tmp_session)
    sidecar.append_prompt_context(tmp_session, "PROMPT-CTX-Y")
    main = json.dumps({"model": "claude-opus", "max_tokens": 4096,
                       "tools": [{"name": "Bash"}],
                       "messages": [{"role": "user", "content": "do the thing"}]}).encode()
    out = server._rewrite_request(main, tmp_session)
    assert "PROMPT-CTX-Y" in json.dumps(json.loads(out))
    assert sidecar.consume_prompt_context(tmp_session) == ""  # consumed
    _clean(tmp_session)
