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
