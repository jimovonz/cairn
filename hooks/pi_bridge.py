#!/usr/bin/env python3
"""pi host bridge for Cairn.

Thin CLI facades that let the pi coding agent reuse the exact same Cairn engine
(parser, storage, retrieval, enforcement) that the Claude Code hooks use, without
touching those hooks. This preserves Claude Code backward compatibility: the
memory DB and all core logic stay the single source of truth; pi just calls in
through a different front door.

Subcommands (all read the assistant/prompt text from a file to avoid shell quoting):
  retrieve --query Q  --session S --transcript T --cwd C   -> prints <cairn_context> XML (may be empty)
  capture  --text-file F --session S --transcript T --cwd C -> parses [cm] block, stores memories
  enforce  --text-file F --session S --transcript T --cwd C -> prints a re-prompt reason (empty = allow stop)
  spec                                                      -> prints the [cm] memory-block format spec
"""
import argparse
import json
import re
import os
import sys
from pathlib import Path

# Make the cairn package + hooks importable when run as a standalone script.
# resolve() dereferences symlinks deliberately: this file gets installed by
# symlinking into ~/.local/bin, and abspath() would leave sys.path pointing at the
# symlink's directory instead of the cairn repo. Every "from hooks.*" import below
# sits under a bare except, so that failure is silent -- subcommands would exit 0
# with empty output and memory would appear to work while storing nothing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _read_text(path):
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _ensure_project(session, transcript, cwd):
    """Register the session and give it a project label derived from cwd.

    retrieval scopes by the session's project, so an unregistered pi session would
    otherwise fall back to global-only search.
    """
    if not session:
        return
    try:
        from hooks.stop_hook import register_session
        from hooks.hook_helpers import get_conn, resolve_project
    except Exception as e:
        sys.stderr.write(f"cairn ensure-project import error: {e}\n")
        return
    try:
        register_session(session, transcript or "")
    except Exception as e:
        sys.stderr.write(f"cairn register_session error: {e}\n")
    if not cwd:
        return
    try:
        conn = get_conn()
        try:
            row = conn.execute("SELECT project FROM sessions WHERE session_id = ?", (session,)).fetchone()
            if row is not None and (row[0] is None or row[0] == ""):
                project = resolve_project(cwd, transcript or "")
                if project:
                    conn.execute("UPDATE sessions SET project = ? WHERE session_id = ?", (project, session))
                    conn.commit()
        finally:
            conn.close()
    except Exception as e:
        sys.stderr.write(f"cairn ensure-project error: {e}\n")


def _pi_source_ref(model, session):
    """Write provenance for pi-authored memories: pi:<model>:<gen-version>.

    Mirrors stop_hook._arm_source_ref: the live generation version, or the A/B
    arm's version when the experiment is on (the prompt hook records the arm in
    hook_state). Without the version a pi memory is unattributable -- it cannot
    be A/B'd or bulk-retracted, and the source_ref='pi' literal erased both the
    version and the authoring model.
    """
    version = "unknown"
    try:
        from cairn import config as _c
        version = getattr(_c, "GENERATION_PROMPT_VERSION", "unknown")
        if getattr(_c, "AB_TEST_ENABLED", False) and session:
            from hooks.hook_helpers import load_hook_state
            arm = load_hook_state(session, "ab_arm")
            if arm:
                version = getattr(_c, "AB_ARM_VERSIONS", {}).get(arm, version)
    except Exception as e:
        sys.stderr.write(f"cairn source_ref resolve error: {e}\n")
    return f"pi:{model or 'unknown'}:{version}"


def cmd_bootstrap(a):
    """Standing session context: project bootstrap + relevant past corrections.

    Claude Code runs both of these inside prompt_hook's `if is_first_prompt(...)`
    block, so they fire once per session and never again. There is no equivalent
    hook here, so this facade is deliberately ungated: the CALLER must invoke it
    only once per session. Calling it every turn would re-inject standing context
    that is already in the conversation.

    Both halves degrade to silence rather than failing: project_bootstrap returns
    nothing for an unresolvable or generic project name (".", "/", "home", "tmp",
    "temp"), and correction_bootstrap returns nothing when the embedder is down or
    the prompt is empty.
    """
    _ensure_project(a.session, a.transcript, a.cwd)
    query = a.query or _read_text(a.text_file)
    parts = []
    try:
        from hooks.prompt_hook import correction_bootstrap, project_bootstrap
    except Exception as e:
        sys.stderr.write(f"cairn bootstrap import error: {e}\n")
        return
    try:
        pb = project_bootstrap(a.session, a.cwd, a.transcript, query)
        if pb:
            parts.append(pb)
    except Exception as e:
        sys.stderr.write(f"cairn project_bootstrap error: {e}\n")
    # Gated on cosine similarity to the first prompt, so it needs the query text.
    try:
        cb = correction_bootstrap(a.session, query)
        if cb:
            parts.append(cb)
    except Exception as e:
        sys.stderr.write(f"cairn correction_bootstrap error: {e}\n")
    if parts:
        sys.stdout.write("\n\n".join(parts))


def cmd_retrieve(a):
    _ensure_project(a.session, a.transcript, a.cwd)
    query = a.query or _read_text(a.text_file)
    if not query.strip():
        return
    # Mirror Claude Code's layering: layer 1 on the first prompt (threshold 0.30),
    # layer 1.5 on every prompt after it (threshold 0.55, and enriched with the last
    # assistant excerpt so short follow-ups still retrieve well). Previously this
    # always called retrieve_context, which is the on-demand L3 path -- it returned
    # results, but with neither the thresholds nor the ranking the other host uses.
    # retrieve_context stays as the fallback so a broken import degrades to the old
    # behaviour rather than to silence.
    xml = None
    try:
        if a.first:
            from hooks.prompt_hook import layer1_search
            xml = layer1_search(query, a.session)
        else:
            from hooks.prompt_hook import layer1_5_search
            xml = layer1_5_search(query, a.session, a.transcript)
    except Exception as e:
        sys.stderr.write(f"cairn layer search error: {e}\n")
        try:
            from hooks.retrieval import retrieve_context
            xml = retrieve_context(query, a.session)
        except Exception as e2:
            sys.stderr.write(f"cairn retrieve error: {e2}\n")
            return
    if xml:
        sys.stdout.write(xml)


def _staged_dir():
    return Path(__file__).resolve().parent.parent / ".staged_context"


def cmd_staged(a):
    """Drain whatever the previous turn deferred to this prompt.

    Two consume-on-read stores feed one channel. Layer-2 cross-project matches are
    staged into hook_state by the stop hook; the behavioural reminders (deferred
    bootstrap, thin-retrieval escalation, query-quality and relevance-grade nudges)
    are written as {session}_*.txt files. Claude Code drains both in prompt_hook.
    Nothing drained them on the pi side, so anything staged there just accumulated.

    Globbing rather than naming each file deliberately: new reminder kinds get
    picked up without touching this code.
    """
    parts = []
    try:
        from hooks.prompt_hook import load_staged_context
        staged = load_staged_context(a.session)
        if staged:
            parts.append(staged)
    except Exception as e:
        sys.stderr.write(f"cairn staged(state) error: {e}\n")
    if a.session:
        try:
            for path in sorted(_staged_dir().glob(f"{a.session}_*.txt")):
                try:
                    body = path.read_text(encoding="utf-8", errors="replace").strip()
                    path.unlink()
                except OSError:
                    continue
                if body:
                    parts.append(body)
        except Exception as e:
            sys.stderr.write(f"cairn staged(files) error: {e}\n")
    if parts:
        sys.stdout.write("\n\n".join(parts))


def cmd_capture(a):
    text = _read_text(a.text_file)
    if not text.strip():
        return
    _ensure_project(a.session, a.transcript, a.cwd)
    try:
        from hooks.parser import parse_memory_block
        from hooks.storage import apply_confidence_updates, insert_memories
    except Exception as e:
        sys.stderr.write(f"cairn capture import error: {e}\n")
        return
    parsed = parse_memory_block(text)
    stored = 0
    if parsed.entries:
        try:
            stored = insert_memories(
                parsed.entries, session_id=a.session, transcript_path=a.transcript,
                source_ref=_pi_source_ref(getattr(a, "model", ""), a.session),
            )
        except Exception as e:
            sys.stderr.write(f"cairn insert error: {e}\n")
    # Relevance grades and the behavioural engagement label are the measurement
    # apparatus -- the Claude Code Stop hook writes both (stop_hook.py
    # apply_relevance_grades / apply_engagement). The pi bridge parsed the [cm]
    # block but dropped them, so every pi turn widened an unlabelled slice of
    # the store that cannot be backfilled later.
    if parsed.relevance_grades:
        try:
            from cairn.relevance import apply_relevance_grades
            apply_relevance_grades(list(parsed.relevance_grades), session_id=a.session)
        except Exception as e:
            sys.stderr.write(f"cairn relevance grade write-back error: {e}\n")
    if parsed.fit_declared:
        try:
            from cairn.relevance import apply_fit_labels
            apply_fit_labels(list(parsed.fit_pairs), session_id=a.session)
        except Exception as e:
            sys.stderr.write(f"cairn fit label write-back error: {e}\n")
    if a.session:
        try:
            from cairn.relevance import apply_engagement
            from cairn.session_extract import _clean_assistant_text
            # Score the cleaned response so the [cm] tail the agent echoes while
            # WRITING memories does not inflate the primary label (read/write parity).
            apply_engagement(_clean_assistant_text(text), session_id=a.session)
        except Exception as e:
            sys.stderr.write(f"cairn engagement write-back error: {e}\n")
    # Layer 2: stage cross-project keyword matches for the next prompt. The stop
    # hook does this on the other host; without it nothing ever populated the
    # staged channel here, so draining it would always have come back empty.
    try:
        if parsed.keywords and not a.continuation:
            from hooks.retrieval import layer2_cross_project_search
            layer2_cross_project_search(parsed.keywords, session_id=a.session)
    except Exception as e:
        sys.stderr.write(f"cairn layer2 staging error: {e}\n")
    if parsed.confidence_updates:
        try:
            apply_confidence_updates(parsed.confidence_updates, session_id=a.session)
        except Exception as e:
            sys.stderr.write(f"cairn confidence update error: {e}\n")
    print(f"captured {stored}")


def cmd_enforce(a):
    text = _read_text(a.text_file)
    try:
        from hooks.parser import linkdef_error_locus, parse_memory_block
    except Exception:
        return
    is_continuation = (getattr(a, "continuation", 0) or 0) > 0
    parsed = parse_memory_block(text)
    entries = parsed.entries
    has_marker = bool(re.search(r"^\[(?:cm|cairn-memory)\]:", text, re.MULTILINE)) or "<memory>" in text
    fmt = ("Use this format:\n[cm]: # '{\"e\":[{\"t\":\"fact\",\"to\":\"short key\","
           "\"c\":\"one line\"}],\"ok\":true,\"ctx\":\"s\",\"kw\":[\"relevant\",\"words\"]}'")

    # 1. No parseable block at all (malformed marker, or none).
    if entries is None and not (parsed.complete_explicit or parsed.context_explicit or parsed.keywords_explicit):
        if has_marker:
            locus = linkdef_error_locus(text) or ""
            hint = "Your memory block could not be parsed. " + (locus + " " if locus else "") + fmt
            print(hint)
        else:
            print("Response missing required memory block. Add a [cm]: # '{...}' block. "
                  "Minimum: [cm]: # '{\"ok\":true,\"ctx\":\"s\",\"kw\":[\"topic\"]}'")
        return

    # 2. Strict well-formedness validation (skipped on continuation, mirroring Claude Code).
    if not is_continuation:
        missing = []
        if not parsed.is_compact:
            if not parsed.complete_explicit:
                missing.append("complete/ok")
            if not parsed.context_explicit:
                missing.append("context/ctx")
            if not parsed.keywords_explicit:
                missing.append("keywords/kw")
        elif entries and not parsed.keywords_explicit:
            missing.append("[k: keywords] on entry line")
        if parsed.complete is False and not parsed.remaining:
            missing.append("remaining/rem")
        if parsed.context == "insufficient" and not parsed.context_need:
            missing.append("context_need/cn")
        # ctx value sanity (stricter than Claude Code): must be sufficient/insufficient.
        if parsed.context_explicit and parsed.context not in ("sufficient", "insufficient"):
            missing.append("ctx must be 's' or 'i' (not a sentence)")
        incomplete = []
        if entries:
            for i, entry in enumerate(entries):
                miss = [f for f in ("type", "topic", "content") if f not in entry]
                if miss:
                    incomplete.append(f"entry {i + 1} missing {', '.join(miss)}")
        if missing or incomplete:
            parts = []
            if missing:
                parts.append("Memory block missing/invalid: " + ", ".join(missing) + ".")
            if incomplete:
                parts.append("Incomplete entries: " + "; ".join(incomplete) + ".")
            parts.append("All fields required. " + fmt)
            print(" ".join(parts))
            return

    # 3. Marked incomplete.
    if parsed.complete is False:
        print(f"Response marked incomplete. Continue with: {parsed.remaining or '(complete the remaining work)'}")
        return

    # 4. Context insufficient -> inject retrieved memory so the model can re-answer.
    if parsed.context == "insufficient" and parsed.context_need:
        _ensure_project(a.session, a.transcript, a.cwd)
        try:
            from hooks.retrieval import retrieve_context
            xml = retrieve_context(parsed.context_need, a.session)
        except Exception:
            xml = None
        if xml:
            print(f"CAIRN CONTEXT:\n{xml}")
            return
    # Otherwise: allow the stop (print nothing).

def cmd_checkpoint(a):
    """Mid-response capture nudge after a notable tool result.

    Mirrors posttool_hook by importing its detection, its per-session budget and
    its nudge text rather than reimplementing any of them, so the two hosts cannot
    drift apart on what counts as high-signal.

    --text-file carries {"tool": str, "input": {...}, "output": {...}} because tool
    payloads do not survive argv. Prints the nudge, or nothing.
    """
    raw = _read_text(a.text_file)
    if not raw.strip():
        return
    try:
        payload = json.loads(raw)
    except Exception:
        return
    try:
        from cairn.config import CHECKPOINT_MAX_NOTES_PER_SESSION
        from hooks.hook_helpers import load_hook_state, save_hook_state
        from hooks.posttool_hook import NUDGE_TEXT, _is_high_signal_bash, _is_high_signal_edit
    except Exception as e:
        sys.stderr.write(f"cairn checkpoint import error: {e}\n")
        return

    # pi names its tools in lower case where Claude Code capitalises them.
    tool = str(payload.get("tool") or "").lower()
    tool_input = payload.get("input") or {}
    tool_output = payload.get("output") or {}
    try:
        if tool in ("bash", "powershell"):
            high, _reason = _is_high_signal_bash(tool_input, tool_output)
        elif tool in ("edit", "write"):
            high, _reason = _is_high_signal_edit(tool_input, tool_output)
        else:
            return
    except Exception as e:
        sys.stderr.write(f"cairn checkpoint detect error: {e}\n")
        return
    if not high:
        return

    # Per-session note budget. Past the cap the stop hook drops the note anyway,
    # so every further nudge costs prompt and output tokens for nothing.
    try:
        nudge_total = int(load_hook_state(a.session, "checkpoint_nudge_total") or 0)
        if nudge_total >= CHECKPOINT_MAX_NOTES_PER_SESSION:
            return
        save_hook_state(a.session, "checkpoint_nudge_total", str(nudge_total + 1))
    except Exception as e:
        sys.stderr.write(f"cairn checkpoint budget error: {e}\n")
        return
    sys.stdout.write(NUDGE_TEXT)


# pi tool names are lower-case where Claude Code capitalises them; pretool_hook
# dispatches on the Claude Code spelling.
_PI_TOOL_TO_CC = {
    "read": "Read", "edit": "Edit", "write": "Write",
    "multiedit": "MultiEdit", "multi_edit": "MultiEdit", "powershell": "Bash",
}


def _cc_tool_name(tool):
    return _PI_TOOL_TO_CC.get(str(tool or "").lower(), str(tool or ""))


def _extract_additional_context(stdout):
    """Pull the injected text out of pretool_hook's Claude Code JSON envelope.

    The hook prints {"hookSpecificOutput": {"additionalContext": ...}} (or, when
    the proxy is enabled, stages to a sidecar instead). Either way this returns
    the plain text, or "" when the hook served nothing. A non-JSON payload is
    passed through unchanged so a future format change degrades to visible text
    rather than silence.
    """
    out = (stdout or "").strip()
    if not out:
        return ""
    try:
        return json.loads(out).get("hookSpecificOutput", {}).get("additionalContext", "") or ""
    except Exception:
        return out


def cmd_pretool(a):
    """File-keyed injection for a tool call: gotchas/corrections, then context,
    then code-graph structure.

    Reuses hooks/pretool_hook.py verbatim by feeding it the same JSON Claude Code
    sends on stdin, so the two hosts cannot drift on what counts as a file-keyed
    gotcha. The proxy is forced off in the child env so the hook prints its
    envelope here instead of staging it to a sidecar no proxy will drain on pi.
    """
    raw = _read_text(a.text_file)
    if not raw.strip():
        return
    try:
        payload = json.loads(raw)
    except Exception as e:
        sys.stderr.write(f"cairn pretool payload error: {e}\n")
        return
    hook_input = {
        "tool_name": _cc_tool_name(payload.get("tool") or payload.get("tool_name")),
        "tool_input": payload.get("input") or payload.get("tool_input") or {},
        "session_id": a.session,
        "cwd": a.cwd or os.getcwd(),
        "transcript_path": a.transcript,
    }
    hook = Path(__file__).resolve().parent / "pretool_hook.py"
    env = dict(os.environ)
    env["CAIRN_PROXY_ENABLED"] = "0"
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, str(hook)], input=json.dumps(hook_input),
            capture_output=True, text=True, timeout=20, env=env,
        )
    except Exception as e:
        sys.stderr.write(f"cairn pretool run error: {e}\n")
        return
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    text = _extract_additional_context(r.stdout)
    if text.strip():
        sys.stdout.write(text)


def cmd_spec(_a):
    try:
        from hooks.prompt_hook import MEMORY_FORMAT_SPEC
        sys.stdout.write(MEMORY_FORMAT_SPEC)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Cairn pi bridge")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("retrieve", "capture", "enforce", "bootstrap", "staged", "checkpoint", "pretool"):
        sp = sub.add_parser(name)
        sp.add_argument("--session", default="")
        sp.add_argument("--transcript", default="")
        sp.add_argument("--cwd", default="")
        sp.add_argument("--text-file", dest="text_file", default="")
        sp.add_argument("--query", default="")
        sp.add_argument("--model", default="",
                        help="authoring model id, stamped into source_ref for provenance")
        sp.add_argument("--continuation", type=int, default=0)
        sp.add_argument("--first", action="store_true",
                        help="first prompt of the session (selects layer 1 over layer 1.5)")
    sub.add_parser("spec")
    a = p.parse_args()
    {"retrieve": cmd_retrieve, "capture": cmd_capture, "enforce": cmd_enforce, "spec": cmd_spec, "bootstrap": cmd_bootstrap, "staged": cmd_staged,
     "checkpoint": cmd_checkpoint, "pretool": cmd_pretool}[a.cmd](a)


if __name__ == "__main__":
    main()
