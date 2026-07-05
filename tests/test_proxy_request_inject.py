import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy.request_inject import reinject_cm, inject_bootstrap, inject_prompt_context

CM = "\n\n[cm]: # '{\"ok\":true}'"


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def test_reinject_cm_string_content():
    stripped = "The answer."
    data = {"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": stripped},
    ]}
    reinject_cm(data, {sha(stripped): CM})
    assert data["messages"][1]["content"] == stripped + CM


def test_reinject_cm_block_list_content():
    stripped = "Block answer."
    data = {"messages": [
        {"role": "assistant", "content": [{"type": "text", "text": stripped}]},
    ]}
    reinject_cm(data, {sha(stripped): CM})
    assert data["messages"][0]["content"][0]["text"] == stripped + CM


def test_reinject_idempotent():
    stripped = "Hi."
    data = {"messages": [{"role": "assistant", "content": stripped}]}
    m = {sha(stripped): CM}
    reinject_cm(data, m)
    reinject_cm(data, m)
    assert data["messages"][0]["content"].count("[cm]") == 1


def test_reinject_no_match_untouched():
    data = {"messages": [{"role": "assistant", "content": "Other text"}]}
    reinject_cm(data, {sha("different"): CM})
    assert data["messages"][0]["content"] == "Other text"


def test_bootstrap_moves_cache_control():
    data = {"system": [
        {"type": "text", "text": "You are Claude."},
        {"type": "text", "text": "Big prompt prefix.", "cache_control": {"type": "ephemeral"}},
    ]}
    inject_bootstrap(data, "CAIRN bootstrap memories")
    sysblocks = data["system"]
    # the breakpoint moved off the old last block onto the appended bootstrap block
    assert "cache_control" not in sysblocks[1]
    assert sysblocks[-1]["cache_control"] == {"type": "ephemeral"}
    assert "CAIRN bootstrap memories" in sysblocks[-1]["text"]


def test_bootstrap_idempotent_stable_bytes():
    data = {"system": [{"type": "text", "text": "P", "cache_control": {"type": "ephemeral"}}]}
    inject_bootstrap(data, "BOOT")
    import copy
    once = copy.deepcopy(data["system"])
    inject_bootstrap(data, "BOOT")  # second turn re-applies — must be byte-identical
    assert data["system"] == once


def test_bootstrap_string_system():
    data = {"system": "Plain system string"}
    inject_bootstrap(data, "BOOT")
    assert isinstance(data["system"], list)
    assert data["system"][-1]["text"].endswith("BOOT")


def test_prompt_context_appended_to_last_user():
    data = {"messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": [{"type": "text", "text": "latest question"}]},
    ]}
    inject_prompt_context(data, "RETRIEVED MEMORIES")
    last = data["messages"][-1]["content"]
    assert last[-1]["text"].endswith("RETRIEVED MEMORIES")
    # earlier user message untouched
    assert data["messages"][0]["content"] == "first"


def test_prompt_context_idempotent():
    data = {"messages": [{"role": "user", "content": "q"}]}
    inject_prompt_context(data, "CTX")
    inject_prompt_context(data, "CTX")
    assert data["messages"][0]["content"].count("CTX") == 1


def test_bootstrap_no_move_breakpoint_pins_cc_off_bootstrap():
    # Unstable-bootstrap degrade path: cache_control stays on the upstream block,
    # bootstrap is appended uncached so it can't invalidate the cached prefix.
    data = {"system": [
        {"type": "text", "text": "Prefix.", "cache_control": {"type": "ephemeral"}},
    ]}
    inject_bootstrap(data, "VOLATILE BOOT", move_breakpoint=False)
    sysblocks = data["system"]
    assert sysblocks[0]["cache_control"] == {"type": "ephemeral"}
    assert sysblocks[-1]["text"].startswith("<!--cairn-bootstrap-->")
    assert "cache_control" not in sysblocks[-1]


def test_bootstrap_move_breakpoint_true_is_default_behaviour():
    data = {"system": [
        {"type": "text", "text": "Prefix.", "cache_control": {"type": "ephemeral"}},
    ]}
    inject_bootstrap(data, "STABLE BOOT")  # default move_breakpoint=True
    sysblocks = data["system"]
    assert "cache_control" not in sysblocks[0]
    assert sysblocks[-1]["cache_control"] == {"type": "ephemeral"}
    assert sysblocks[-1]["text"].startswith("<!--cairn-bootstrap-->")


# --- Context-paring Phase 1: markers -----------------------------

from cairn.proxy.request_inject import (
    inject_cm_markers, _cm_marker_for,
    CM_MARKER_CAPTURED, CM_MARKER_INVALID,
)

VALID_CM = "\n\n[cm]: # '{\"e\":[{\"t\":\"fact\",\"to\":\"topic\",\"c\":\"x\"}],\"ok\":true}'"


def test_marker_anchor_first_captured_stays_verbatim():
    # The single captured turn is the anchor: it keeps its real block as a live
    # template so a correct [cm] block is always visible.
    stripped = "Only captured turn."
    data = {"messages": [{"role": "assistant", "content": stripped}]}
    inject_cm_markers(data, {sha(stripped): VALID_CM})
    assert data["messages"][0]["content"] == stripped + VALID_CM


def test_marker_second_captured_turn_gets_marker():
    anchor, stripped = "Anchor turn.", "The answer."
    data = {"messages": [
        {"role": "assistant", "content": anchor},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": stripped},
    ]}
    inject_cm_markers(data, {sha(anchor): VALID_CM, sha(stripped): VALID_CM})
    assert data["messages"][0]["content"] == anchor + VALID_CM          # anchor verbatim
    assert data["messages"][2]["content"] == stripped + CM_MARKER_CAPTURED


def test_marker_block_list_second_turn():
    anchor, stripped = "Anchor.", "Block answer."
    data = {"messages": [
        {"role": "assistant", "content": anchor},
        {"role": "assistant", "content": [{"type": "text", "text": stripped}]},
    ]}
    inject_cm_markers(data, {sha(anchor): VALID_CM, sha(stripped): VALID_CM})
    assert data["messages"][1]["content"][0]["text"] == stripped + CM_MARKER_CAPTURED


def test_marker_invalid_for_unparseable_block():
    anchor, stripped = "Anchor.", "Bad block turn."
    bad = "\n\n[cm]: # '{not valid json'"
    data = {"messages": [
        {"role": "assistant", "content": anchor},
        {"role": "assistant", "content": stripped},
    ]}
    inject_cm_markers(data, {sha(anchor): VALID_CM, sha(stripped): bad})
    assert data["messages"][1]["content"] == stripped + CM_MARKER_INVALID


def test_marker_idempotent():
    anchor, stripped = "Anchor.", "Hi."
    data = {"messages": [
        {"role": "assistant", "content": anchor},
        {"role": "assistant", "content": stripped},
    ]}
    m = {sha(anchor): VALID_CM, sha(stripped): VALID_CM}
    inject_cm_markers(data, m)
    inject_cm_markers(data, m)
    assert data["messages"][1]["content"].count("[cm:") == 1


def test_marker_no_match_untouched():
    data = {"messages": [{"role": "assistant", "content": "Other text"}]}
    inject_cm_markers(data, {sha("different"): VALID_CM})
    assert data["messages"][0]["content"] == "Other text"


def test_non_anchor_turns_have_no_verbatim_block():
    anchor, stripped = "Anchor.", "Answer."
    data = {"messages": [
        {"role": "assistant", "content": anchor},
        {"role": "assistant", "content": stripped},
    ]}
    inject_cm_markers(data, {sha(anchor): VALID_CM, sha(stripped): VALID_CM})
    assert "[cm]: #" not in data["messages"][1]["content"]   # non-anchor = marker only
    assert "[cm]: #" in data["messages"][0]["content"]       # anchor keeps the real block


def test_cm_marker_for_helper():
    assert _cm_marker_for(VALID_CM) == CM_MARKER_CAPTURED
    assert _cm_marker_for("\n\n[cm]: # '[1,2,3]'") == CM_MARKER_INVALID  # not a dict
    assert _cm_marker_for("no block here") == CM_MARKER_INVALID
    assert _cm_marker_for("") == CM_MARKER_INVALID


def test_marker_empty_map_noop():
    data = {"messages": [{"role": "assistant", "content": "x"}]}
    inject_cm_markers(data, {})
    assert data["messages"][0]["content"] == "x"


def test_marker_anchor_survives_tool_use_only_assistant_message():
    # Regression: a tool_use-only assistant message (no text blocks) precedes
    # the first captured turn in virtually every agentic session. It must NOT
    # count as "anchor already behind us" — the first captured turn still keeps
    # its verbatim block as the live template.
    anchor, stripped = "Answer one.", "Answer two."
    data = {"messages": [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        {"role": "assistant", "content": anchor},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": stripped},
    ]}
    inject_cm_markers(data, {sha(anchor): VALID_CM, sha(stripped): VALID_CM})
    assert data["messages"][3]["content"] == anchor + VALID_CM           # anchor verbatim
    assert data["messages"][5]["content"] == stripped + CM_MARKER_CAPTURED
    # tool_use-only message untouched
    assert data["messages"][1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]
