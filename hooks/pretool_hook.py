#!/usr/bin/env python3
"""
Claude Code PreToolUse Hook for Cairn — File Context Injection.

Fires before Read/Edit/Write/MultiEdit tool uses. Queries the cairn DB for
memories associated with the target file and injects them as context.

Two injection paths:
  1. Corrections (gotcha) — warnings injected as CAIRN GOTCHA, highest priority
  2. All other types (decisions, facts, etc.) — injected as CAIRN CONTEXT FOR FILE

This creates a closed loop:
  LLM touches file → memories written → file paths captured →
  next access to that file → relevant context injected automatically.
"""

from __future__ import annotations

import json
import os
import re
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err
import sys
from typing import Any, Optional

from hooks.hook_helpers import (
    log, get_conn, record_metric, flush_metrics, load_hook_state, save_hook_state,
    load_injected_ids, save_injected_ids, record_layer_delivery,
    get_session_project, overdelivered_ids, deliver_additional_context,
)

# Max entries to inject per file access (avoid flooding context)
MAX_GOTCHA_INJECTIONS = 3
MAX_CONTEXT_INJECTIONS = 5


# A code search is one of these invoked as a command in its own right. `rg` was
# missing until 2026-07-30: this environment's routing policy mandates rg over
# grep, so the detector was blind to a quarter of all symbol searches and the
# hint simply never fired on them. The lookbehind keeps `--grep` (ccm-get's
# retrieval filter) from reading as a code search.
_SEARCH_CMD = re.compile(r'(?<![-\w])(?:rg|grep|egrep|fgrep)\b')
_IDENT = r'[A-Za-z_][A-Za-z0-9_]{2,}'
# Quoted pattern, optionally with a definition keyword: rg -n 'def target_fn'
_SEARCH_QUOTED = re.compile(
    r'''["'](?:(?:def|class|fn|func|function)\s+)?(''' + _IDENT + r''')["']''')
# Bare pattern, skipping flag-only invocations: rg -n target_fn src/
_SEARCH_BARE = re.compile(
    r'(?<![-\w])(?:rg|grep|egrep|fgrep)\b(?:\s+-\S+)*\s+(' + _IDENT + r')(?:\s|$)')


def _looks_like_code_search(command: str) -> Optional[str]:
    """Return a candidate symbol name if the Bash command looks like a code-symbol search.

    Only the FIRST pipeline segment counts. A later `| grep foo` is a filter over
    some other command's output — the user is narrowing a result set, not looking
    for where a symbol lives — and hinting at the graph there is noise. That
    over-firing was the mirror of the rg blind spot: the old detector missed the
    mandated tool while firing on pipeline filters.
    """
    head = command.split('|')[0]
    if not _SEARCH_CMD.search(head):
        return None
    m = _SEARCH_QUOTED.search(head) or _SEARCH_BARE.search(head)
    return m.group(1) if m else None


def _edit_intent_symbol(tool_name: str, tool_input: dict, edited_files: list):
    """(symbol, def_file_abspath) for a symbol whose DEFINITION lives in a file
    being edited this call, or None. Ties served callers to the symbol actually
    under edit rather than any identifier that merely appears in the command;
    generic hubs are skipped. Native Edit exposes old_string directly; Bash
    cch-edit is parsed from the command string."""
    if tool_name in ("Edit", "MultiEdit", "Write"):
        haystack = tool_input.get("old_string") or tool_input.get("oldString") or ""
    elif tool_name == "Bash":
        cmd = tool_input.get("command") or ""
        if "cch-edit" not in cmd and "cch-write" not in cmd:
            return None
        haystack = cmd
    else:
        return None
    if not haystack or not edited_files:
        return None
    from cairn.graph import location, _GENERIC_HUBS
    edited_norm = {os.path.abspath(f) for f in edited_files}
    seen_idents: set = set()
    for ident in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\b', haystack):
        if ident in seen_idents or ident.lower() in _GENERIC_HUBS:
            continue  # skip generic hubs (path/get/object/...) — noise, high fan-in
        seen_idents.add(ident)
        if len(seen_idents) > 25:  # bounded scan — first identifiers dominate
            break
        try:
            loc = location(ident)  # "file:line-line" or "Symbol not found: X"
        except Exception:
            continue
        if loc.startswith("Symbol not found"):
            continue
        loc_file = loc.split(":", 1)[0]
        loc_abs = os.path.abspath(loc_file)
        if loc_abs in edited_norm:
            return ident, loc_abs
    return None


def symbol_context_block(symbol: str, max_callers: int = 8,
                          include_impact: bool = True,
                          include_callers: bool = True) -> Optional[str]:
    """Served (not reminded) structural context for a resolved symbol, scoped
    to keep token overhead low and avoid duplicating Tier-2 file context:

    - include_impact: prepend the one-line blast radius. Dropped on the edit
      path when the file's Tier-2 map (which already prints per-symbol
      callers:N) was served in the SAME hook call — no point restating it.
    - include_callers: append the capped caller call-site list. Off on the grep
      path (location + impact is enough; the list is one --callers away), on for
      edit-intent where the pre-change blast radius is the whole point.

    Returns None if the symbol is not in the graph."""
    from cairn.graph import impact, callers
    imp = None
    if include_impact:
        imp = impact(symbol)
        if imp.startswith("Symbol not found"):
            return None
    header = f"CODE GRAPH — {symbol}" + (f"  [{imp}]" if imp else "")
    if not include_callers:
        # Grep path: impact + a pointer to expand on demand. No caller query.
        return header + f"\n  expand: cairn-graph --callers {symbol} · --context-pack {symbol}"
    call = callers(symbol)
    if call.startswith("Symbol not found"):
        # Only reachable when include_impact was False (existence unverified).
        return header + "\n  callers:\n    (none)" if include_impact else None
    caller_lines = [] if call.startswith("No callers") else call.splitlines()
    shown = caller_lines[:max_callers]
    more = len(caller_lines) - len(shown)
    body = "\n".join(f"    {ln}" for ln in shown) if shown else "    (none)"
    out = f"{header}\n  callers:\n{body}"
    if more > 0:
        out += f"\n    … +{more} more — cairn-graph --callers {symbol}"
    out += f"\n  full body+tests: cairn-graph --context-pack {symbol}"
    return out


def find_memories_for_file(
    file_path: str,
    corrections_only: bool = False,
    current_session_id: Optional[str] = None,
    project: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Find memories associated with a given file path.

    Matching rules (S/N scoping):
    1. Exact path match in associated_files — always serves.
    2. Basename match — only for memories from the SAME project, and never
       for generic basenames (README.md, __init__.py, ...) which are weak
       retrieval keys that resolve to the same cross-project set everywhere.

    Ordering: match quality (exact > basename), then recency. Confidence is
    deliberately NOT an ordering signal here — it made the same stale
    high-confidence set win in every project forever.

    If corrections_only=True, returns only correction-type memories (gotcha path).
    Otherwise returns all non-correction types (context path).

    Memories written by current_session_id are excluded — they're already in
    the live conversation context, so re-injecting them is pure token noise.
    """
    if not file_path:
        return []
    from cairn.config import GENERIC_BASENAMES

    basename = os.path.basename(file_path)
    generic = basename in GENERIC_BASENAMES

    conn = get_conn()
    type_filter = "type = 'correction'" if corrections_only else "type != 'correction'"

    try:
        # The LIKE prefilter culls the table in C before per-row JSON parsing
        # in Python — any row associated with this file contains the basename
        # as a substring of its associated_files JSON.
        rows = conn.execute(f"""
            SELECT id, type, topic, content, associated_files, confidence,
                   session_id, project, updated_at
            FROM memories
            WHERE {type_filter}
              AND associated_files IS NOT NULL
              AND archived_reason IS NULL
              AND deleted_at IS NULL
              AND associated_files LIKE '%' || ? || '%'
        """, (basename,)).fetchall()
    except sqlite3.Error as e:
        log(f"File context query error: {e}")
        conn.close()
        return []

    conn.close()

    matches: list[dict[str, Any]] = []
    for row in rows:
        (mid, mem_type, topic, content, files_json, confidence,
         mem_session, mem_project, updated_at) = row
        if current_session_id and mem_session == current_session_id:
            continue
        try:
            files = json.loads(files_json)
        except (json.JSONDecodeError, TypeError):
            continue

        quality = 0
        for f in files:
            if f == file_path:
                quality = 2
                break
            if (not generic and project and mem_project == project
                    and os.path.basename(f) == basename):
                quality = max(quality, 1)
        if not quality:
            continue
        matches.append({
            "id": mid,
            "type": mem_type,
            "topic": topic,
            "content": content,
            "confidence": confidence or 0.7,
            "_quality": quality,
            "_updated_at": updated_at or "",
        })

    matches.sort(key=lambda m: (m["_quality"], m["_updated_at"]), reverse=True)
    return matches

# --- Bash file-access recovery -------------------------------------------------
# In environments where Read/Edit/Write are blocked and routed through Bash
# helpers (e.g. claude-context-hooks: cat/sed and cch-edit.py/cch-write.py), the
# native Read/Edit PreToolUse event never fires — so file-context injection would
# silently never trigger. Both hooks fire on the SAME PreToolUse:Bash event, so we
# recover the target file(s) from the command string and inject for them too.

# File-reading verbs whose operand is a path we should inject context for.
_BASH_FILE_VERBS = {"cat", "head", "tail", "sed", "less", "more", "bat", "nl", "view"}
# Helper editors are invoked as scripts; match by suffix to catch python3 .../cch-edit.py.
_BASH_EDITOR_SUFFIXES = ("cch-edit.py", "cch-write.py", "cch-edit", "cch-write")
# Only inject for source/text files — keeps DB/log/binary args from triggering queries.
_SOURCE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".java", ".rb", ".sh", ".sql", ".md", ".toml", ".yaml", ".yml",
}


def extract_bash_file_paths(command: str, max_files: int = 3) -> list[str]:
    """Recover source-file paths a Bash command is about to read or edit.

    Conservative by design: engages only when the command invokes a known
    file-reading/editing verb, and returns only existing files with a source
    extension. Returns realpaths, deduped, capped at max_files.
    """
    if not command:
        return []
    # cch-batch.py with heredoc: extract paths from each embedded command line
    if "cch-batch.py" in command and "\n" in command:
        batch_out: list[str] = []
        for line in command.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or "<<" in line or "cch-batch" in line or line == "EOF":
                continue
            for p in extract_bash_file_paths(line, max_files=max_files - len(batch_out)):
                if p not in batch_out:
                    batch_out.append(p)
                if len(batch_out) >= max_files:
                    return batch_out
        return batch_out
    import shlex
    try:
        toks = shlex.split(command, posix=True)
    except ValueError:
        toks = command.split()
    if not toks:
        return []
    verbs = {os.path.basename(t) for t in toks}
    is_editor = any(t.endswith(_BASH_EDITOR_SUFFIXES) for t in toks)
    if not (verbs & _BASH_FILE_VERBS or is_editor):
        return []
    out: list[str] = []
    for t in toks:
        if t.startswith("-"):
            continue
        if os.path.splitext(t)[1].lower() not in _SOURCE_EXTS:
            continue
        if not os.path.isfile(t):
            continue
        rp = os.path.realpath(t)
        if rp not in out:
            out.append(rp)
        if len(out) >= max_files:
            break
    return out


def sections_for_file(file_path: str, session_id: str, seen: set, graph_cfg,
                      served: set, project: Optional[str] = None,
                      dampened: Optional[set] = None) -> tuple[list[str], list[int]]:
    """Build the gotcha / context / graph injection sections for one file.

    seen is the shared graph_files_seen set (mutated in place when a graph block
    is served). graph_cfg is (enabled, max_symbols, risk_threshold) or None.

    served is the session-wide ledger of memory IDs already injected by ANY
    layer (prompt, stop/L3, per-file), mutated in place. A memory is injected
    at most once per session: each file's top-N is computed first, THEN
    already-served IDs are dropped — so a re-touched file injects nothing
    rather than backfilling with weaker matches. Returns (sections, new_ids).
    """
    basename = os.path.basename(file_path)
    sections: list[str] = []
    new_ids: list[int] = []
    dampened = dampened or set()

    # Path 1: corrections (gotcha warnings) — highest priority.
    # Dampened (over-delivered) entries are retired BEFORE the top-N cut so
    # newer memories can take their slots; served entries are dropped AFTER
    # it so a re-touched file delivers nothing rather than backfilling.
    corrections = find_memories_for_file(file_path, corrections_only=True,
                                         current_session_id=session_id, project=project)
    corrections = [c for c in corrections if c["id"] not in dampened]
    corrections = [c for c in corrections[:MAX_GOTCHA_INJECTIONS] if c["id"] not in served]
    if corrections:
        warnings = [f"- [{c['topic']}] {c['content']}" for c in corrections]
        ids = [c["id"] for c in corrections]
        sections.append(
            f"CAIRN GOTCHA for {basename}:\n" + "\n".join(warnings)
            + f"\nSources: {', '.join(str(i) for i in ids)}"
        )
        served.update(ids)
        new_ids.extend(ids)
        log(f"Gotcha injection: {len(corrections)} corrections for {basename}")
        record_metric(session_id, "gotcha_injected", basename, len(corrections))

    # Path 2: all other memory types (decisions, facts, skills, etc.).
    # Pre-ranked by match quality then recency in find_memories_for_file —
    # confidence ordering is gone (it froze the same stale set everywhere).
    context_memories = find_memories_for_file(file_path, corrections_only=False,
                                              current_session_id=session_id, project=project)
    if context_memories:
        context_memories = [m for m in context_memories if m["id"] not in dampened]
        top = [m for m in context_memories[:MAX_CONTEXT_INJECTIONS] if m["id"] not in served]
        if top:
            lines = [f"- [{m['type']}/{m['topic']}] {m['content']}" for m in top]
            ids = [m["id"] for m in top]
            sections.append(
                f"CAIRN CONTEXT for {basename}:\n" + "\n".join(lines)
                + f"\nSources: {', '.join(str(i) for i in ids)}"
            )
            served.update(ids)
            new_ids.extend(ids)
            log(f"File context injection: {len(top)} memories for {basename}")
            record_metric(session_id, "file_context_injected", basename, len(top))

    # Path 3: code-graph structural context — deterministic, no LLM. Once-per-file
    # via the shared seen cache. Fails open.
    if graph_cfg is not None:
        try:
            enabled, max_symbols, risk_threshold = graph_cfg
            key = os.path.realpath(file_path)
            if enabled and key not in seen:
                from cairn.graph import file_context_block
                block = file_context_block(file_path, max_symbols=max_symbols, risk_threshold=risk_threshold)
                if block:
                    sections.append(
                        f"CAIRN GRAPH for {basename} (code-review-graph — structure, no need to re-read):\n{block}"
                    )
                    seen.add(key)
                    record_metric(session_id, "graph_file_context_injected", basename)
        except Exception as _e:
            log(f"graph file-context failed open: {type(_e).__name__}: {_e}")

    return sections, new_ids


def main() -> None:
    if os.environ.get("CAIRN_ENABLED", "1") == "0":
        sys.exit(0)
    raw = sys.stdin.read()
    hook_input = json.loads(raw)

    tool_name = hook_input.get("tool_name", "")
    session_id = hook_input.get("session_id", "") or hook_input.get("sessionId", "")
    cwd = hook_input.get("cwd", os.getcwd())

    tool_input = hook_input.get("tool_input") or hook_input.get("input") or {}

    # Determine the file(s) this tool call touches.
    file_paths: list[str] = []
    if tool_name in ("Read", "Edit", "Write", "MultiEdit"):
        fp = tool_input.get("file_path") or tool_input.get("filePath") or ""
        if fp:
            file_paths = [fp]
    elif tool_name == "Bash":
        # Read/Edit are routed through Bash helpers in some environments; recover paths.
        file_paths = extract_bash_file_paths(tool_input.get("command") or "")

    # Resolve graph config once (shared across files).
    graph_cfg = None
    try:
        from cairn.config import (
            GRAPH_FILE_CONTEXT_ENABLED,
            GRAPH_FILE_CONTEXT_MAX_SYMBOLS,
            GRAPH_RISK_TAIL_THRESHOLD,
        )
        graph_cfg = (GRAPH_FILE_CONTEXT_ENABLED, GRAPH_FILE_CONTEXT_MAX_SYMBOLS, GRAPH_RISK_TAIL_THRESHOLD)
    except Exception:
        graph_cfg = None

    # Shared once-per-file-per-session graph cache.
    seen_raw = load_hook_state(session_id, "graph_files_seen") or ""
    seen = set(seen_raw.split("\n")) if seen_raw else set()
    seen_before = len(seen)
    seen_at_start = set(seen)  # snapshot: files Tier-2 serves THIS call = seen - seen_at_start

    # Session-wide served-memory ledger — same retrieved_ids key the prompt and
    # stop layers use, so a memory injected by ANY layer is never injected again
    # this session, by any layer.
    served = load_injected_ids(session_id) if session_id else set()
    new_served: list[int] = []

    # Resolve once per invocation: session project (scopes basename matching)
    # and lifetime-overdelivered entries (retired from the per-file path).
    project = None
    dampened: set = set()
    if file_paths:
        try:
            from cairn.config import FILE_INJECTION_DAMPEN_THRESHOLD
            dampened = overdelivered_ids(FILE_INJECTION_DAMPEN_THRESHOLD)
        except Exception:
            dampened = set()
        if session_id:
            try:
                _pconn = get_conn()
                project = get_session_project(_pconn, session_id)
                _pconn.close()
            except Exception:
                project = None

    sections: list[str] = []
    for fp in file_paths:
        fp_sections, fp_new_ids = sections_for_file(fp, session_id, seen, graph_cfg,
                                                    served, project=project,
                                                    dampened=dampened)
        sections.extend(fp_sections)
        new_served.extend(fp_new_ids)

    if len(seen) != seen_before:
        save_hook_state(session_id, "graph_files_seen", "\n".join(seen))
    if new_served and session_id:
        save_injected_ids(session_id, new_served)
        record_layer_delivery(session_id, "per-file", new_served)

    # Symbol context: SERVE the resolved symbol's blast radius + caller list
    # (not a menu of commands the model must then remember to run — that
    # reminder tier is the one that empirically fails). Fires on a symbol grep
    # OR an edit whose target symbol is defined in the file being changed.
    # Deduped per-symbol via a session ledger, so every distinct symbol is
    # served once (the old once-per-session flag served only the first).
    try:
        from cairn.config import (
            GRAPH_SYMBOL_CONTEXT_ENABLED, GRAPH_SYMBOL_CONTEXT_MAX_CALLERS,
        )
    except Exception:
        GRAPH_SYMBOL_CONTEXT_ENABLED, GRAPH_SYMBOL_CONTEXT_MAX_CALLERS = False, 8
    if GRAPH_SYMBOL_CONTEXT_ENABLED and tool_name in ("Bash", "Edit", "MultiEdit", "Write"):
        from cairn.graph import _GENERIC_HUBS
        command = tool_input.get("command") or ""
        gdb = os.path.join(cwd, ".code-review-graph", "graph.db")
        graph_present = os.path.exists(gdb)
        symbol = (_looks_like_code_search(command) if tool_name == "Bash" else None)
        origin = "grep" if symbol else None
        sym_def_file = None
        # Is this call a code edit at all? (Native edit tool, or a Bash cch-edit/
        # cch-write.) Used both to resolve the edit-target symbol and to count
        # uncovered edits as the utilisation denominator.
        is_edit = tool_name in ("Edit", "MultiEdit", "Write") or (
            tool_name == "Bash" and ("cch-edit" in command or "cch-write" in command))
        if not symbol and is_edit:
            hit = _edit_intent_symbol(tool_name, tool_input, file_paths)
            if hit:
                symbol, sym_def_file = hit
                origin = "edit"
        # Generic hubs (path/get/object/...) resolve in the graph but are pure
        # noise — drop them before serving (covers the grep path; the edit path
        # already filtered inside _edit_intent_symbol).
        if symbol and symbol.lower() in _GENERIC_HUBS:
            symbol = None
        served_symbol = False
        if symbol and graph_present:
            sym_seen_raw = load_hook_state(session_id, "graph_symbols_seen") or ""
            sym_seen = set(sym_seen_raw.split("\n")) if sym_seen_raw else set()
            if symbol not in sym_seen:
                # Scope the block to cut overhead and avoid duplicating Tier-2:
                #  - grep -> impact one-liner only (list is one --callers away);
                #  - edit -> caller list, with the impact line dropped when
                #    Tier-2 covered this file THIS call (it already prints the
                #    symbol's callers:N in the file map).
                tier2_served = seen - seen_at_start
                tier2_covered = bool(
                    sym_def_file and os.path.realpath(sym_def_file) in tier2_served)
                block = None
                try:
                    block = symbol_context_block(
                        symbol, max_callers=GRAPH_SYMBOL_CONTEXT_MAX_CALLERS,
                        include_impact=not tier2_covered,
                        include_callers=(origin == "edit"))
                except Exception as _e:
                    log(f"symbol_context_block failed open: {type(_e).__name__}: {_e}")
                if block:
                    sections.append(block)
                    sym_seen.add(symbol)
                    save_hook_state(session_id, "graph_symbols_seen", "\n".join(sym_seen))
                    log(f"graph symbol context served ({origin}): {symbol}")
                    record_metric(session_id, "graph_symbol_context_served", symbol)
                    served_symbol = True
            else:
                served_symbol = True  # already served this session — still "covered"
        # Utilisation denominator: an edit on a graphed repo where no symbol
        # context was served makes "maximally utilised" measurable, not a vibe.
        if is_edit and graph_present and not served_symbol:
            record_metric(session_id, "graph_symbol_edit_uncovered", symbol or "")

    if not sections:
        sys.exit(0)

    deliver_additional_context(session_id, "PreToolUse", "\n\n".join(sections))
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            log(f"PRETOOL HOOK CRASH: {e}")
        except Exception:
            pass
        sys.exit(0)
    finally:
        # main() exits via sys.exit() on every path; flush buffered metrics
        # (e.g. graph_file_context_injected) explicitly here rather than relying
        # solely on the atexit backstop. SystemExit propagates through finally;
        # a second flush is a no-op once the buffer is drained.
        try:
            flush_metrics()
        except Exception:
            pass
