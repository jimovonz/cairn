#!/usr/bin/env python3
"""Query layer over .code-review-graph/graph.db — joins code graph with cairn memories."""

import sqlite3
import sys
import os
import json
import time
from pathlib import Path

USAGE = """Usage: cairn-graph <command> [symbol]

Commands:
  --location SYMBOL      Show file:line for symbol
  --callers SYMBOL       Show what calls this symbol
  --callees SYMBOL       Show what this symbol calls
  --tests SYMBOL         Show tests for this symbol
  --summary              Repo-level overview
  --knowledge SYMBOL     Join graph data with cairn memories
  --context-pack SYMBOL  All-in-one context for a symbol
  --impact SYMBOL        One-line blast-radius summary
  --orientation          Tier-1 repo orientation block (modules/flows/hubs)
  --file-context FILE    Tier-2 structural context for a file's symbols
"""


def _find_graph_db(repo_root=None):
    """Walk up from cwd to find .code-review-graph/graph.db."""
    start = Path(repo_root) if repo_root else Path.cwd()
    d = start.resolve()
    while True:
        candidate = d / ".code-review-graph" / "graph.db"
        if candidate.exists():
            return candidate
        parent = d.parent
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(f"No .code-review-graph/graph.db found above {start}")


def _resolve_symbol(conn, name):
    """Find nodes matching name. Returns list of (qualified_name, kind, file_path, line_start, line_end)."""
    rows = conn.execute(
        "SELECT qualified_name, kind, file_path, line_start, line_end FROM nodes WHERE name = ?",
        (name,)
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT qualified_name, kind, file_path, line_start, line_end FROM nodes WHERE qualified_name LIKE '%' || ?",
            (name,)
        ).fetchall()
    return rows


def _check_freshness(conn, file_path, repo_root):
    """Compare node's updated_at with file mtime. Returns True if fresh."""
    row = conn.execute(
        "SELECT updated_at FROM nodes WHERE file_path = ? LIMIT 1", (file_path,)
    ).fetchone()
    if not row:
        return True
    graph_time = row[0]
    full_path = Path(repo_root) / file_path if repo_root else Path(file_path)
    try:
        file_mtime = full_path.stat().st_mtime
    except OSError:
        return True  # file gone, can't check
    if file_mtime > graph_time + 1:
        print(f"# warning: {file_path} modified after last graph build", file=sys.stderr)
        return False
    return True


def location(symbol, repo_root=None):
    """Resolve symbol, print path:start-end for each match."""
    db_path = _find_graph_db(repo_root)
    rr = str(db_path.parent.parent)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    matches = _resolve_symbol(conn, symbol)
    conn.close()
    if not matches:
        return f"Symbol not found: {symbol}"
    if len(matches) == 1:
        qn, kind, fp, ls, le = matches[0]
        _check_freshness(sqlite3.connect(str(db_path)), fp, rr)
        return f"{fp}:{ls}-{le}"
    lines = []
    for qn, kind, fp, ls, le in matches:
        lines.append(f"{fp}:{ls}-{le}  [{kind}] {qn}")
    return "\n".join(lines)


def _target_forms(qn):
    """Match forms for a call target: exact qualified, bare name, Class.name.

    Cross-file calls are stored with unresolved bare/dotted targets in several
    languages (e.g. Kotlin), so qualified-only matching loses real callers.
    """
    bare = qn.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    return (
        "(target_qualified = ? OR target_qualified = ? OR target_qualified LIKE ?)",
        (qn, bare, f"%.{bare}"),
    )


def callers(symbol, repo_root=None):
    """Get edges where target_qualified matches the symbol (any match form)."""
    db_path = _find_graph_db(repo_root)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    matches = _resolve_symbol(conn, symbol)
    if not matches:
        conn.close()
        return f"Symbol not found: {symbol}"
    qnames = [m[0] for m in matches]
    results = []
    seen = set()
    for qn in qnames:
        clause, params = _target_forms(qn)
        rows = conn.execute(
            f"SELECT file_path, line, source_qualified, kind FROM edges WHERE {clause}",
            params
        ).fetchall()
        for fp, line, src, kind in rows:
            if (fp, line, src) in seen:
                continue
            seen.add((fp, line, src))
            results.append(f"{fp}:{line}  {src}")
    conn.close()
    if not results:
        return f"No callers found for {symbol}"
    return "\n".join(results)


def callees(symbol, repo_root=None):
    """Get edges where source_qualified matches the symbol."""
    db_path = _find_graph_db(repo_root)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    matches = _resolve_symbol(conn, symbol)
    if not matches:
        conn.close()
        return f"Symbol not found: {symbol}"
    qnames = [m[0] for m in matches]
    results = []
    for qn in qnames:
        rows = conn.execute(
            "SELECT file_path, line, target_qualified, kind FROM edges WHERE source_qualified = ?",
            (qn,)
        ).fetchall()
        for fp, line, tgt, kind in rows:
            results.append(f"{fp}:{line}  {tgt}")
    conn.close()
    if not results:
        return f"No callees found for {symbol}"
    return "\n".join(results)


def tests(symbol, repo_root=None):
    """Get TESTED_BY edges for the symbol."""
    db_path = _find_graph_db(repo_root)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    matches = _resolve_symbol(conn, symbol)
    if not matches:
        conn.close()
        return f"Symbol not found: {symbol}"
    qnames = [m[0] for m in matches]
    results = []
    for qn in qnames:
        rows = conn.execute(
            "SELECT target_qualified FROM edges WHERE source_qualified = ? AND kind = 'TESTED_BY'",
            (qn,)
        ).fetchall()
        for (tgt,) in rows:
            # Resolve test location
            loc = conn.execute(
                "SELECT file_path, line_start, line_end FROM nodes WHERE qualified_name = ?",
                (tgt,)
            ).fetchone()
            if loc:
                results.append(f"{loc[0]}:{loc[1]}-{loc[2]}  {tgt}")
            else:
                results.append(f"???  {tgt}")
    conn.close()
    if not results:
        return f"No tests found for {symbol}"
    return "\n".join(results)


def summary(repo_root=None):
    """Repo-level overview."""
    db_path = _find_graph_db(repo_root)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # Entry points: functions with 0 callers, non-test, non-private
    entry_points = conn.execute("""
        SELECT n.qualified_name, n.file_path, n.line_start
        FROM nodes n
        WHERE n.kind IN ('Function', 'method')
          AND n.is_test = 0
          AND n.name NOT LIKE '\\_%' ESCAPE '\\'
          AND n.qualified_name NOT IN (
            SELECT DISTINCT target_qualified FROM edges WHERE kind = 'CALLS'
          )
        ORDER BY n.file_path, n.line_start
        LIMIT 15
    """).fetchall()

    # Top 10 by fan-in
    top_fanin = conn.execute("""
        SELECT target_qualified, COUNT(*) as cnt
        FROM edges WHERE kind = 'CALLS'
        GROUP BY target_qualified
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()

    # Untested public functions
    untested = conn.execute("""
        SELECT n.qualified_name, n.file_path
        FROM nodes n
        WHERE n.kind IN ('Function', 'method')
          AND n.is_test = 0
          AND n.name NOT LIKE '\\_%' ESCAPE '\\'
          AND n.qualified_name NOT IN (
            SELECT source_qualified FROM edges WHERE kind = 'TESTED_BY'
          )
        ORDER BY n.file_path
        LIMIT 15
    """).fetchall()

    lines = [f"Nodes: {node_count}  Edges: {edge_count}", ""]

    if entry_points:
        lines.append(f"Entry points ({len(entry_points)}):")
        for qn, fp, ls in entry_points:
            lines.append(f"  {fp}:{ls}  {qn}")
        lines.append("")

    if top_fanin:
        lines.append("Top fan-in:")
        for qn, cnt in top_fanin:
            lines.append(f"  {cnt:3d}x  {qn}")
        lines.append("")

    if untested:
        lines.append(f"Untested public ({len(untested)}):")
        for qn, fp in untested:
            lines.append(f"  {fp}  {qn}")

    conn.close()
    return "\n".join(lines)


def _cairn_db_path():
    """Resolve cairn DB path."""
    cairn_home = os.environ.get("CAIRN_HOME", os.path.expanduser("~/Projects/cairn"))
    return os.path.join(cairn_home, "cairn", "cairn.db")


def knowledge(symbol, repo_root=None):
    """Join graph data with cairn memories."""
    db_path = _find_graph_db(repo_root)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    matches = _resolve_symbol(conn, symbol)
    conn.close()
    if not matches:
        return f"Symbol not found: {symbol}"

    file_paths = list(set(m[2] for m in matches))
    cairn_path = _cairn_db_path()
    if not os.path.exists(cairn_path):
        return f"Cairn DB not found at {cairn_path}"

    cconn = sqlite3.connect(cairn_path); cconn.execute("PRAGMA busy_timeout=3000")
    results = []
    seen_ids = set()

    # Search by associated_files containing file paths
    for fp in file_paths:
        rows = cconn.execute(
            "SELECT id, type, topic, content, updated_at FROM memories "
            "WHERE associated_files LIKE '%' || ? || '%' AND deleted_at IS NULL "
            "ORDER BY updated_at DESC LIMIT 10",
            (fp,)
        ).fetchall()
        for r in rows:
            if r[0] not in seen_ids:
                seen_ids.add(r[0])
                results.append(r)

    # FTS search on symbol name
    try:
        fts_rows = cconn.execute(
            "SELECT m.id, m.type, m.topic, m.content, m.updated_at "
            "FROM memories_fts f JOIN memories m ON f.rowid = m.id "
            "WHERE memories_fts MATCH ? AND m.deleted_at IS NULL "
            "ORDER BY rank LIMIT 5",
            (symbol,)
        ).fetchall()
        for r in fts_rows:
            if r[0] not in seen_ids:
                seen_ids.add(r[0])
                results.append(r)
    except sqlite3.OperationalError:
        pass  # FTS table may not exist

    cconn.close()
    if not results:
        return f"No cairn memories found for {symbol}"

    lines = []
    for mid, mtype, topic, content, updated in results:
        preview = content[:120] + "..." if len(content) > 120 else content
        date = updated[:10] if updated and len(updated) >= 10 else str(updated)
        lines.append(f"  [{mid}] {mtype}/{topic} ({date}): {preview}")
    return "\n".join(lines)


def context_pack(symbol, repo_root=None):
    """All-in-one context for a symbol."""
    db_path = _find_graph_db(repo_root)
    rr = str(db_path.parent.parent)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    matches = _resolve_symbol(conn, symbol)
    if not matches:
        conn.close()
        return f"Symbol not found: {symbol}"

    # Use first match
    qn, kind, fp, ls, le = matches[0]
    _check_freshness(conn, fp, rr)

    parts = [f"## {qn} [{kind}]", f"## {fp}:{ls}-{le}", ""]

    # Read function body
    full_path = Path(rr) / fp
    if full_path.exists() and ls and le:
        try:
            file_lines = full_path.read_text().splitlines()
            body = file_lines[max(0, ls - 1):le]
            # Trim to keep output compact
            if len(body) > 40:
                body = body[:35] + [f"  ... ({len(body) - 35} more lines)"]
            parts.append("```")
            parts.extend(body)
            parts.append("```")
            parts.append("")
        except (OSError, UnicodeDecodeError):
            parts.append(f"(could not read {fp})")
            parts.append("")

    # Callers (compact)
    _clause, _params = _target_forms(qn)
    caller_rows = conn.execute(
        f"SELECT DISTINCT source_qualified FROM edges WHERE {_clause} AND kind = 'CALLS'",
        _params
    ).fetchall()
    if caller_rows:
        parts.append(f"Callers ({len(caller_rows)}):")
        for (src,) in caller_rows[:10]:
            parts.append(f"  {src}")
        if len(caller_rows) > 10:
            parts.append(f"  ... +{len(caller_rows) - 10} more")
        parts.append("")

    # Tests (compact)
    test_rows = conn.execute(
        "SELECT target_qualified FROM edges WHERE source_qualified = ? AND kind = 'TESTED_BY'",
        (qn,)
    ).fetchall()
    if test_rows:
        parts.append(f"Tests ({len(test_rows)}):")
        for (tgt,) in test_rows:
            parts.append(f"  {tgt}")
        parts.append("")

    conn.close()

    # Cairn memories
    km = knowledge(symbol, repo_root)
    if km and not km.startswith("No cairn") and not km.startswith("Cairn DB"):
        parts.append("Memories:")
        parts.append(km)

    result = "\n".join(parts)
    # Truncate if over 2KB
    if len(result) > 2048:
        result = result[:2000] + "\n... (truncated)"
    return result


def impact(symbol, repo_root=None):
    """One-line blast-radius: callers:N tests:M files:F."""
    db_path = _find_graph_db(repo_root)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    matches = _resolve_symbol(conn, symbol)
    if not matches:
        conn.close()
        return f"Symbol not found: {symbol}"

    qnames = [m[0] for m in matches]
    caller_files = set()
    caller_count = 0
    test_count = 0
    for qn in qnames:
        _clause, _params = _target_forms(qn)
        crows = conn.execute(
            f"SELECT file_path FROM edges WHERE {_clause} AND kind = 'CALLS'",
            _params
        ).fetchall()
        caller_count += len(crows)
        caller_files.update(r[0] for r in crows)
        trows = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_qualified = ? AND kind = 'TESTED_BY'",
            (qn,)
        ).fetchone()
        test_count += trows[0]
    conn.close()
    return f"callers:{caller_count} tests:{test_count} files:{len(caller_files)}"


# Generic/builtin call targets that dominate fan-in counts but carry no
# architectural signal — excluded from hub surfacing.
_GENERIC_HUBS = {
    "execute", "get", "set", "len", "close", "print", "append", "join", "format",
    "split", "strip", "fetchone", "fetchall", "fetchmany", "str", "int", "float",
    "dict", "list", "set", "tuple", "open", "write", "read", "items", "keys",
    "values", "encode", "decode", "commit", "sort", "sorted", "range", "enumerate",
    "isinstance", "getattr", "setattr", "hasattr", "super", "log", "add", "pop",
}


def orientation_block(repo_root=None):
    """Tier 1: deterministic ~200-token repo orientation from the code graph.

    Modules (communities, test-dominated ones excluded), top flows by criticality,
    and real project hubs (builtins/stdlib and test helpers filtered out). Pure
    SQL, no LLM. Returns None if no graph.db is present or nothing useful is found.
    """
    try:
        db_path = _find_graph_db(repo_root)
    except FileNotFoundError:
        return None
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    try:
        parts = []

        # Header stamp: coverage + freshness so the model can assess trust at
        # a glance — a stale or vendor-polluted graph gets rationally skipped,
        # a fresh covering one gets used (cairn 2336462281793).
        try:
            n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            n_files = conn.execute(
                "SELECT COUNT(DISTINCT file_path) FROM nodes").fetchone()[0]
            age_s = max(0.0, time.time() - os.path.getmtime(str(db_path)))
            if age_s < 3600:
                age = f"{int(age_s // 60)}m"
            elif age_s < 86400:
                age = f"{age_s / 3600:.1f}h"
            else:
                age = f"{age_s / 86400:.1f}d"
            parts.append(
                f"Graph: {n_nodes} nodes · {n_files} files · refreshed {age} ago")
            parts.append("")
        except (sqlite3.OperationalError, OSError):
            pass

        # Modules — architectural communities. Drop the test-dominated ones.
        try:
            comms = conn.execute(
                "SELECT name, key_symbols, size FROM community_summaries "
                "WHERE name NOT LIKE 'tests-%' AND name NOT LIKE 'test-%' "
                "ORDER BY size DESC LIMIT 5"
            ).fetchall()
        except sqlite3.OperationalError:
            comms = []
        if comms:
            mod_lines = []
            for name, key_symbols, size in comms:
                try:
                    arr = json.loads(key_symbols) if key_symbols else []
                except (json.JSONDecodeError, TypeError):
                    arr = []
                # Skip test-dominated communities — they aren't architecture.
                if arr and sum(s.startswith("test") for s in arr) > len(arr) / 2:
                    continue
                arr = [s for s in dict.fromkeys(arr) if s not in _GENERIC_HUBS][:4]
                syms = ", ".join(arr)
                mod_lines.append(f"  {name} ({size}): {syms}" if syms else f"  {name} ({size})")
            if mod_lines:
                parts.append("Modules:")
                parts.extend(mod_lines)
                parts.append("")

        # Key flows by criticality — entry-point function + its file.
        try:
            flows = conn.execute(
                "SELECT name, entry_point, criticality FROM flow_snapshots "
                "ORDER BY criticality DESC LIMIT 5"
            ).fetchall()
        except sqlite3.OperationalError:
            flows = []
        if flows:
            parts.append("Key flows (by criticality):")
            for name, entry, crit in flows:
                ep = entry.split("::")[-1] if entry and "::" in entry else (name or "?")
                fp = os.path.basename(entry.split("::")[0]) if entry and "::" in entry else ""
                crit_s = f" {crit:.2f}" if isinstance(crit, (int, float)) else ""
                parts.append(f"  {ep} ({fp}){crit_s}" if fp else f"  {ep}{crit_s}")
            parts.append("")

        # Hubs — top fan-in. Join to nodes drops stdlib/external (no node row) and
        # is_test=1 helpers; _GENERIC_HUBS drops project-defined generic names.
        try:
            hub_rows = conn.execute(
                "SELECT e.target_qualified, COUNT(*) c FROM edges e "
                "JOIN nodes n ON n.qualified_name = e.target_qualified "
                "WHERE e.kind='CALLS' AND n.is_test=0 "
                "GROUP BY e.target_qualified ORDER BY c DESC LIMIT 40"
            ).fetchall()
        except sqlite3.OperationalError:
            hub_rows = []
        hubs = []
        for tgt, c in hub_rows:
            short = tgt.split("::")[-1]
            if short in _GENERIC_HUBS:
                continue
            fp = os.path.basename(tgt.split("::")[0])
            hubs.append(f"{short} ({fp}) {c}x")
            if len(hubs) >= 6:
                break
        if hubs:
            parts.append("Hubs (high fan-in — edits ripple widely):")
            parts.append("  " + "; ".join(hubs))

        block = "\n".join(parts).strip()
        return block or None
    finally:
        conn.close()


def _graph_file_path(conn, file_path, rr):
    """Map a (possibly symlinked) absolute path to the file_path stored in the graph.

    The graph bakes in whatever absolute path the build used (e.g. /mnt/ssd/...),
    while a hook may report /home/jameo/... for the same bind-mounted checkout.
    Match by realpath, then by repo-relative tail, then basename.
    """
    for cand in (os.path.realpath(file_path), os.path.abspath(file_path)):
        if conn.execute("SELECT 1 FROM nodes WHERE file_path = ? LIMIT 1", (cand,)).fetchone():
            return cand
    try:
        rel = os.path.relpath(os.path.realpath(file_path), os.path.realpath(rr))
        if not rel.startswith(".."):
            row = conn.execute(
                "SELECT file_path FROM nodes WHERE file_path LIKE '%/' || ? LIMIT 1", (rel,)
            ).fetchone()
            if row:
                return row[0]
    except ValueError:
        pass
    row = conn.execute(
        "SELECT file_path FROM nodes WHERE file_path LIKE '%/' || ? LIMIT 1",
        (os.path.basename(file_path),)
    ).fetchone()
    return row[0] if row else None


def file_context_block(file_path, repo_root=None, max_symbols=12, risk_threshold=0.55):
    """Tier 2: structural context for one file's symbols, so the model can reason
    about it without reading it. Per-symbol signature + fan-in/out, plus a risk-tail
    callout for security-relevant / high-risk symbols. Pure SQL, no LLM. Returns
    None if no graph.db, the file isn't in the graph, or it has no symbols.
    """
    try:
        db_path = _find_graph_db(repo_root)
    except FileNotFoundError:
        return None
    rr = repo_root or str(db_path.parent.parent)
    conn = sqlite3.connect(str(db_path)); conn.execute("PRAGMA busy_timeout=3000")
    try:
        stored = _graph_file_path(conn, file_path, rr)
        if not stored:
            return None
        nodes = conn.execute(
            "SELECT qualified_name, name, kind, line_start, line_end, params, return_type "
            "FROM nodes WHERE file_path = ? AND kind IN ('Function','Class','method') "
            "AND is_test=0 ORDER BY line_start", (stored,)
        ).fetchall()
        if not nodes:
            return None

        # Per-symbol fan-in/out (CALLS) in one pass each.
        def _count(col, qn):
            return conn.execute(
                f"SELECT COUNT(*) FROM edges WHERE {col} = ? AND kind='CALLS'", (qn,)
            ).fetchone()[0]

        # Risk tail — security_relevant or high risk_score.
        risk_by_qn = {}
        try:
            for qn, rs, sec, cov in conn.execute(
                "SELECT qualified_name, risk_score, security_relevant, test_coverage FROM risk_index"
            ).fetchall():
                risk_by_qn[qn] = (rs or 0.0, sec or 0, cov or "")
        except sqlite3.OperationalError:
            pass

        try:
            rel = os.path.relpath(stored, os.path.realpath(rr))
        except ValueError:
            rel = os.path.basename(stored)

        highlights, lines = [], []
        for qn, name, kind, ls, le, params, ret in nodes[:max_symbols]:
            params = " ".join((params or "").split())  # collapse multi-line signatures
            sig = f"{name}{params}" if params else name
            if ret:
                sig += f" -> {ret}"
            cin, cout = _count("target_qualified", qn), _count("source_qualified", qn)
            fan = f"callers:{cin}" + (f" callees:{cout}" if cout else "")
            # Emit the file:line RANGE (start-end) per symbol so the agent can Read
            # exactly the symbol's span instead of over-selecting, and jump to it
            # directly — no follow-up `cairn-graph --location` tool call + turn.
            if ls and le and le > ls:
                loc = f"{rel}:{ls}-{le}"
            elif ls:
                loc = f"{rel}:{ls}"
            else:
                loc = rel
            lines.append(f"  {sig}  {loc}  [{fan}]")
            rs, sec, cov = risk_by_qn.get(qn, (0.0, 0, ""))
            if sec or rs >= risk_threshold:
                flags = []
                if sec:
                    flags.append("security-relevant")
                if rs >= risk_threshold:
                    flags.append(f"risk {rs:.2f}")
                if cov == "untested":
                    flags.append("untested")
                highlights.append(f"  ⚠ {name} {loc} ({', '.join(flags)})")

        head = f"{rel} — {len(nodes)} symbol(s)"
        if len(nodes) > max_symbols:
            head += f" (showing {max_symbols})"
        out = [head]
        if highlights:
            out.append("High-risk:")
            out.extend(highlights)
        out.extend(lines)
        return "\n".join(out)
    finally:
        conn.close()


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(USAGE.strip())
        sys.exit(1)

    cmd = sys.argv[1]
    symbol = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    try:
        if cmd == "--location" and symbol:
            print(location(symbol))
        elif cmd == "--callers" and symbol:
            print(callers(symbol))
        elif cmd == "--callees" and symbol:
            print(callees(symbol))
        elif cmd == "--tests" and symbol:
            print(tests(symbol))
        elif cmd == "--summary":
            print(summary())
        elif cmd == "--knowledge" and symbol:
            print(knowledge(symbol))
        elif cmd == "--context-pack" and symbol:
            print(context_pack(symbol))
        elif cmd == "--impact" and symbol:
            print(impact(symbol))
        elif cmd == "--orientation":
            block = orientation_block()
            print(block if block else "No graph.db found — run: code-review-graph build")
        elif cmd == "--file-context" and symbol:
            block = file_context_block(symbol)
            print(block if block else f"No graph data for file: {symbol}")
        else:
            print(USAGE.strip())
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "python3")
    if os.path.exists(_venv_python) and sys.prefix == sys.base_prefix:
        os.execv(_venv_python, [_venv_python] + sys.argv)
    main()
