import sys, os, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy import server, sidecar

CM = "\n\n[cm]: # '{\"ok\":true}'"


def _clean(session):
    for suf in ("_inject_bootstrap.txt", "_inject_prompt.txt", "_cm_capture.jsonl", "_turn_context.json",
                "_directive.txt"):
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


def test_append_prompt_context_dedup_whole_entries_only(tmp_session="pytest-dedup-1"):
    _clean(tmp_session)
    # byte-identical repeats of queued entries are dropped, even non-adjacent
    # (a Stop-block/retry double-fire replays the whole append sequence)
    sidecar.append_prompt_context(tmp_session, "CTX-A\nline2")
    sidecar.append_prompt_context(tmp_session, "CTX-B")
    sidecar.append_prompt_context(tmp_session, "CTX-A\nline2")
    sidecar.append_prompt_context(tmp_session, "CTX-B")
    # a novel payload that appears only mid-line inside a queued block survives
    sidecar.append_prompt_context(tmp_session, "ine2")
    assert sidecar.consume_prompt_context(tmp_session) == "CTX-A\nline2\nCTX-B\nine2"
    _clean(tmp_session)


# --- per-prompt context persists for the whole turn ---------------------------
# Claude Code rebuilds history on every model call; context the proxy added to
# the first call of a turn used to vanish after the first tool use.

TOOLS = [{"name": "Bash", "description": "run"}]


def _req(messages):
    return json.dumps({"system": [{"type": "text", "text": "Sys"}], "tools": TOOLS, "messages": messages}).encode()


def _prompt(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _call(i):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}]}


def _result(i):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}]}


def test_context_reappended_on_every_call_of_the_turn():
    s = "pytest-turn-ctx-1"
    _clean(s)
    sidecar.append_prompt_context(s, "TURN-CTX")
    turn = [_prompt("fix it")]
    first = json.loads(server._rewrite_request(_req(turn), s))
    assert "TURN-CTX" in json.dumps(first["messages"][-1])
    for i in range(3):   # later calls in the same turn: context sits at the end, prompt untouched
        turn = turn + [_call(i), _result(i)]
        later = json.loads(server._rewrite_request(_req(turn), s))
        assert "TURN-CTX" in json.dumps(later["messages"][-1])
        assert "TURN-CTX" not in json.dumps(later["messages"][:-1])   # cached prefix unchanged
    _clean(s)


def test_new_prompt_ends_the_turn_context():
    s = "pytest-turn-ctx-2"
    _clean(s)
    sidecar.append_prompt_context(s, "OLD-CTX")
    turn = [_prompt("first question")]
    server._rewrite_request(_req(turn), s)
    turn = turn + [{"role": "assistant", "content": "answer"}, _prompt("second question")]   # nothing staged
    data = json.loads(server._rewrite_request(_req(turn), s))
    assert "OLD-CTX" not in json.dumps(data)
    _clean(s)


def test_history_rewrite_ends_the_turn_context():
    s = "pytest-turn-ctx-3"
    _clean(s)
    sidecar.append_prompt_context(s, "CTX")
    server._rewrite_request(_req([_call(0), _result(0), _prompt("q")]), s)
    compacted = [_prompt("summary of earlier conversation"), _call(1), _result(1)]   # different prompt at a new index
    data = json.loads(server._rewrite_request(_req(compacted), s))
    assert "CTX" not in json.dumps(data)
    _clean(s)


def test_directive_reappended_but_logged_once():
    s = "pytest-turn-ctx-4"
    _clean(s)
    sidecar.append_prompt_directive(s, "use the code graph")
    turn = [_prompt("where is X?")]
    server._rewrite_request(_req(turn), s)
    log_before = sidecar.directive_log_path(s)
    n = sum(1 for _ in open(log_before)) if os.path.exists(log_before) else 0
    turn = turn + [_call(0), _result(0)]
    data = json.loads(server._rewrite_request(_req(turn), s))
    assert "use the code graph" in json.dumps(data["messages"][-1])
    assert (sum(1 for _ in open(log_before)) if os.path.exists(log_before) else 0) == n
    _clean(s)
    try:
        os.unlink(log_before)
    except FileNotFoundError:
        pass


def test_turn_survives_cache_control_moving_off_the_prompt():
    # Observed live: Claude Code marks the prompt's last block with cache_control
    # on the first call and moves the marker to newer messages afterwards.
    s = "pytest-turn-ctx-5"
    _clean(s)
    sidecar.append_prompt_context(s, "CTX-5")
    first_prompt = {"role": "user", "content": [{"type": "text", "text": "fix it",
                                                 "cache_control": {"type": "ephemeral"}}]}
    server._rewrite_request(_req([first_prompt]), s)
    later = [_prompt("fix it"), _call(0),
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "ok",
                                           "cache_control": {"type": "ephemeral"}}]}]
    data = json.loads(server._rewrite_request(_req(later), s))
    assert "CTX-5" in json.dumps(data["messages"][-1])
    _clean(s)
