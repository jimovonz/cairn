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


def callers(symbol, repo_root=None):
    """Get edges where target_qualified matches the symbol."""
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
            "SELECT file_path, line, source_qualified, kind FROM edges WHERE target_qualified = ?",
            (qn,)
        ).fetchall()
        for fp, line, src, kind in rows:
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
          AND n.name NOT LIKE '\_%' ESCAPE '\\'
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
          AND n.name NOT LIKE '\_%' ESCAPE '\\'
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
    caller_rows = conn.execute(
        "SELECT source_qualified FROM edges WHERE target_qualified = ? AND kind = 'CALLS'",
        (qn,)
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
        crows = conn.execute(
            "SELECT file_path FROM edges WHERE target_qualified = ? AND kind = 'CALLS'",
            (qn,)
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
