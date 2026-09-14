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
import re
import os
import sys

# Make the cairn package + hooks importable when run as a standalone script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    except Exception:
        return
    try:
        register_session(session, transcript or "")
    except Exception:
        pass
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
    except Exception:
        pass


def cmd_retrieve(a):
    _ensure_project(a.session, a.transcript, a.cwd)
    query = a.query or _read_text(a.text_file)
    if not query.strip():
        return
    try:
        from hooks.retrieval import retrieve_context
        xml = retrieve_context(query, a.session)
    except Exception as e:
        sys.stderr.write(f"cairn retrieve error: {e}\n")
        return
    if xml:
        sys.stdout.write(xml)


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
                parsed.entries, session_id=a.session, transcript_path=a.transcript, source_ref="pi"
            )
        except Exception as e:
            sys.stderr.write(f"cairn insert error: {e}\n")
    if parsed.confidence_updates:
        try:
            apply_confidence_updates(parsed.confidence_updates, session_id=a.session)
        except Exception:
            pass
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

def cmd_spec(_a):
    try:
        from hooks.prompt_hook import MEMORY_FORMAT_SPEC
        sys.stdout.write(MEMORY_FORMAT_SPEC)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Cairn pi bridge")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("retrieve", "capture", "enforce"):
        sp = sub.add_parser(name)
        sp.add_argument("--session", default="")
        sp.add_argument("--transcript", default="")
        sp.add_argument("--cwd", default="")
        sp.add_argument("--text-file", dest="text_file", default="")
        sp.add_argument("--query", default="")
        sp.add_argument("--continuation", type=int, default=0)
    sub.add_parser("spec")
    a = p.parse_args()
    {"retrieve": cmd_retrieve, "capture": cmd_capture, "enforce": cmd_enforce, "spec": cmd_spec}[a.cmd](a)


if __name__ == "__main__":
    main()
