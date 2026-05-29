"""Tests for cairn/graph.py — query layer over .code-review-graph/graph.db."""

import os
import sqlite3
import time

import pytest

import cairn.graph as graph


NODES_SCHEMA = """
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    language TEXT,
    parent_name TEXT,
    params TEXT,
    return_type TEXT,
    modifiers TEXT,
    is_test INTEGER DEFAULT 0,
    file_hash TEXT,
    extra TEXT DEFAULT '{}',
    updated_at REAL NOT NULL
);
"""

EDGES_SCHEMA = """
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    source_qualified TEXT NOT NULL,
    target_qualified TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line INTEGER DEFAULT 0,
    extra TEXT DEFAULT '{}',
    confidence REAL DEFAULT 1.0,
    confidence_tier TEXT DEFAULT 'EXTRACTED'
);
"""


@pytest.fixture
def graph_repo(tmp_path):
    """Create a temporary repo with .code-review-graph/graph.db and sample files."""
    # Create graph DB directory
    graph_dir = tmp_path / ".code-review-graph"
    graph_dir.mkdir()
    db_path = graph_dir / "graph.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute(NODES_SCHEMA)
    conn.execute(EDGES_SCHEMA)

    now = time.time()

    # Insert sample nodes
    conn.execute(
        "INSERT INTO nodes (kind, name, qualified_name, file_path, line_start, line_end, language, is_test, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("Function", "foo", "src.main.foo", "src/main.py", 10, 25, "python", 0, now),
    )
    conn.execute(
        "INSERT INTO nodes (kind, name, qualified_name, file_path, line_start, line_end, language, is_test, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("Function", "test_foo", "tests.test_main.test_foo", "tests/test_main.py", 5, 15, "python", 1, now),
    )
    conn.execute(
        "INSERT INTO nodes (kind, name, qualified_name, file_path, line_start, line_end, language, is_test, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("class", "Bar", "src.bar.Bar", "src/bar.py", 1, 50, "python", 0, now),
    )
    conn.execute(
        "INSERT INTO nodes (kind, name, qualified_name, file_path, line_start, line_end, language, is_test, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("method", "do_thing", "src.bar.Bar.do_thing", "src/bar.py", 10, 20, "python", 0, now),
    )

    # Insert sample edges
    conn.execute(
        "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, line) "
        "VALUES (?, ?, ?, ?, ?)",
        ("CALLS", "src.bar.Bar.do_thing", "src.main.foo", "src/bar.py", 15),
    )
    conn.execute(
        "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, line) "
        "VALUES (?, ?, ?, ?, ?)",
        ("TESTED_BY", "src.main.foo", "tests.test_main.test_foo", "src/main.py", 10),
    )

    conn.commit()
    conn.close()

    # Create sample source file with enough lines for context_pack body reading
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    lines = []
    for i in range(1, 30):
        if i == 10:
            lines.append("def foo(x, y):")
        elif 10 < i <= 25:
            lines.append(f"    # line {i} of foo")
        else:
            lines.append(f"# line {i}")
    (src_dir / "main.py").write_text("\n".join(lines) + "\n")

    # Create bar.py
    bar_lines = [f"# bar line {i}" for i in range(1, 55)]
    (src_dir / "bar.py").write_text("\n".join(bar_lines) + "\n")

    # Create tests dir
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_lines = [f"# test line {i}" for i in range(1, 20)]
    (tests_dir / "test_main.py").write_text("\n".join(test_lines) + "\n")

    return tmp_path


# ---------------------------------------------------------------------------
# _find_graph_db
# ---------------------------------------------------------------------------

class TestFindGraphDb:

    def test_find_graph_db(self, graph_repo, monkeypatch):
        """Finds graph.db when cwd is the repo root."""
        monkeypatch.chdir(graph_repo)
        result = graph._find_graph_db()
        assert result.exists()
        assert result.name == "graph.db"

    def test_find_graph_db_with_repo_root(self, graph_repo):
        """Finds graph.db when repo_root is passed explicitly."""
        result = graph._find_graph_db(repo_root=str(graph_repo))
        assert result.exists()
        assert result.name == "graph.db"

    def test_find_graph_db_walks_up(self, graph_repo, monkeypatch):
        """Finds graph.db when cwd is a subdirectory of the repo root."""
        subdir = graph_repo / "src" / "deep" / "nested"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        result = graph._find_graph_db()
        assert result.exists()
        assert result.name == "graph.db"
        assert ".code-review-graph" in str(result)

    def test_find_graph_db_not_found(self, tmp_path, monkeypatch):
        """Raises FileNotFoundError when no graph.db exists."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="No .code-review-graph/graph.db found"):
            graph._find_graph_db()


# ---------------------------------------------------------------------------
# _resolve_symbol
# ---------------------------------------------------------------------------

class TestResolveSymbol:

    def test_resolve_exact_name(self, graph_repo):
        """Exact name match returns the correct node."""
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        conn = sqlite3.connect(str(db_path))
        results = graph._resolve_symbol(conn, "foo")
        conn.close()
        assert len(results) == 1
        qn, kind, fp, ls, le = results[0]
        assert qn == "src.main.foo"
        assert kind == "Function"
        assert fp == "src/main.py"
        assert ls == 10
        assert le == 25

    def test_resolve_qualified_fallback(self, graph_repo):
        """Falls back to qualified_name LIKE when no exact name match."""
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        conn = sqlite3.connect(str(db_path))
        # "Bar.do_thing" won't match any name column exactly, but will match
        # qualified_name "src.bar.Bar.do_thing" via LIKE '%Bar.do_thing'
        results = graph._resolve_symbol(conn, "Bar.do_thing")
        conn.close()
        assert len(results) == 1
        assert results[0][0] == "src.bar.Bar.do_thing"

    def test_resolve_not_found(self, graph_repo):
        """Returns empty list when symbol doesn't exist."""
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        conn = sqlite3.connect(str(db_path))
        results = graph._resolve_symbol(conn, "nonexistent_symbol")
        conn.close()
        assert results == []


# ---------------------------------------------------------------------------
# _check_freshness
# ---------------------------------------------------------------------------

class TestCheckFreshness:

    def test_check_freshness_fresh(self, graph_repo):
        """Returns True when file has not been modified since graph build."""
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        conn = sqlite3.connect(str(db_path))
        result = graph._check_freshness(conn, "src/main.py", str(graph_repo))
        conn.close()
        assert result is True

    def test_check_freshness_stale(self, graph_repo):
        """Returns False when file is newer than graph data."""
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        # Set the node's updated_at to the past
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE nodes SET updated_at = ? WHERE file_path = ?", (1000.0, "src/main.py"))
        conn.commit()
        # Touch the file to make it newer
        main_py = graph_repo / "src" / "main.py"
        main_py.write_text(main_py.read_text() + "\n# touched\n")
        result = graph._check_freshness(conn, "src/main.py", str(graph_repo))
        conn.close()
        assert result is False

    def test_check_freshness_missing_file(self, graph_repo):
        """Returns True when the file doesn't exist (can't check, so assume fresh)."""
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        conn = sqlite3.connect(str(db_path))
        # Insert a node for a file that doesn't exist on disk
        conn.execute(
            "INSERT INTO nodes (kind, name, qualified_name, file_path, line_start, line_end, language, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("Function", "ghost", "src.ghost.ghost", "src/ghost.py", 1, 5, "python", 1000.0),
        )
        conn.commit()
        result = graph._check_freshness(conn, "src/ghost.py", str(graph_repo))
        conn.close()
        assert result is True

    def test_check_freshness_unknown_path(self, graph_repo):
        """Returns True when the file_path is not in the nodes table at all."""
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        conn = sqlite3.connect(str(db_path))
        result = graph._check_freshness(conn, "completely/unknown.py", str(graph_repo))
        conn.close()
        assert result is True


# ---------------------------------------------------------------------------
# location
# ---------------------------------------------------------------------------

class TestLocation:

    def test_location_single(self, graph_repo):
        """Single match returns path:start-end."""
        result = graph.location("foo", repo_root=str(graph_repo))
        assert result == "src/main.py:10-25"

    def test_location_multiple(self, graph_repo):
        """Multiple matches lists all with qualified names."""
        # Add another node named "foo" in a different file
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO nodes (kind, name, qualified_name, file_path, line_start, line_end, language, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("Function", "foo", "src.other.foo", "src/other.py", 1, 10, "python", time.time()),
        )
        conn.commit()
        conn.close()

        result = graph.location("foo", repo_root=str(graph_repo))
        lines = result.strip().split("\n")
        assert len(lines) == 2
        # Each line should contain path:start-end, [kind], and qualified_name
        for line in lines:
            assert ":" in line
            assert "[Function]" in line

    def test_location_not_found(self, graph_repo):
        """Returns 'Symbol not found' message."""
        result = graph.location("nonexistent", repo_root=str(graph_repo))
        assert result == "Symbol not found: nonexistent"


# ---------------------------------------------------------------------------
# callers
# ---------------------------------------------------------------------------

class TestCallers:

    def test_callers(self, graph_repo):
        """Returns caller table for a symbol with callers."""
        result = graph.callers("foo", repo_root=str(graph_repo))
        assert "src.bar.Bar.do_thing" in result
        assert "src/bar.py:15" in result

    def test_callers_not_found(self, graph_repo):
        """Returns 'No callers found' when no edges target the symbol."""
        result = graph.callers("Bar", repo_root=str(graph_repo))
        assert result == "No callers found for Bar"

    def test_callers_symbol_not_found(self, graph_repo):
        """Returns 'Symbol not found' for unknown symbol."""
        result = graph.callers("nonexistent", repo_root=str(graph_repo))
        assert result == "Symbol not found: nonexistent"


# ---------------------------------------------------------------------------
# callees
# ---------------------------------------------------------------------------

class TestCallees:

    def test_callees(self, graph_repo):
        """Returns callee table for a symbol with callees."""
        result = graph.callees("do_thing", repo_root=str(graph_repo))
        assert "src.main.foo" in result
        assert "src/bar.py" in result

    def test_callees_not_found(self, graph_repo):
        """Returns 'No callees found' when symbol has no outgoing edges."""
        result = graph.callees("Bar", repo_root=str(graph_repo))
        assert result == "No callees found for Bar"

    def test_callees_symbol_not_found(self, graph_repo):
        """Returns 'Symbol not found' for unknown symbol."""
        result = graph.callees("nonexistent", repo_root=str(graph_repo))
        assert result == "Symbol not found: nonexistent"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestTests:

    def test_tests(self, graph_repo):
        """Returns TESTED_BY edges with locations."""
        result = graph.tests("foo", repo_root=str(graph_repo))
        assert "tests.test_main.test_foo" in result
        assert "tests/test_main.py:5-15" in result

    def test_tests_none(self, graph_repo):
        """Returns 'No tests found' when no TESTED_BY edges exist."""
        result = graph.tests("Bar", repo_root=str(graph_repo))
        assert result == "No tests found for Bar"

    def test_tests_symbol_not_found(self, graph_repo):
        """Returns 'Symbol not found' for unknown symbol."""
        result = graph.tests("nonexistent", repo_root=str(graph_repo))
        assert result == "Symbol not found: nonexistent"

    def test_tests_unresolved_target(self, graph_repo):
        """Handles TESTED_BY edge where the test node is missing from nodes table."""
        db_path = graph_repo / ".code-review-graph" / "graph.db"
        conn = sqlite3.connect(str(db_path))
        # Add a TESTED_BY edge pointing to a nonexistent test node
        conn.execute(
            "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, line) "
            "VALUES (?, ?, ?, ?, ?)",
            ("TESTED_BY", "src.bar.Bar", "tests.test_bar.test_bar_missing", "src/bar.py", 1),
        )
        conn.commit()
        conn.close()

        result = graph.tests("Bar", repo_root=str(graph_repo))
        assert "???" in result
        assert "tests.test_bar.test_bar_missing" in result


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:

    def test_summary(self, graph_repo):
        """Summary output contains node/edge counts and section headers."""
        result = graph.summary(repo_root=str(graph_repo))
        assert "Nodes: 4" in result
        assert "Edges: 2" in result

    def test_summary_entry_points(self, graph_repo):
        """Summary lists entry points (functions with no callers, non-test, non-private)."""
        result = graph.summary(repo_root=str(graph_repo))
        # do_thing calls foo, so do_thing has no callers and is a public non-test function
        assert "Entry points" in result
        assert "src.bar.Bar.do_thing" in result

    def test_summary_untested(self, graph_repo):
        """Summary lists untested public functions."""
        result = graph.summary(repo_root=str(graph_repo))
        assert "Untested public" in result
        # do_thing has no TESTED_BY edge, so it should appear as untested
        assert "src.bar.Bar.do_thing" in result


# ---------------------------------------------------------------------------
# impact
# ---------------------------------------------------------------------------

class TestImpact:

    def test_impact(self, graph_repo):
        """Returns callers:N tests:M files:F format."""
        result = graph.impact("foo", repo_root=str(graph_repo))
        assert result == "callers:1 tests:1 files:1"

    def test_impact_no_callers_no_tests(self, graph_repo):
        """Returns zeros when symbol has no callers or tests."""
        result = graph.impact("Bar", repo_root=str(graph_repo))
        assert result == "callers:0 tests:0 files:0"

    def test_impact_symbol_not_found(self, graph_repo):
        """Returns 'Symbol not found' for unknown symbol."""
        result = graph.impact("nonexistent", repo_root=str(graph_repo))
        assert result == "Symbol not found: nonexistent"


# ---------------------------------------------------------------------------
# context_pack
# ---------------------------------------------------------------------------

class TestContextPack:

    def test_context_pack_contains_body(self, graph_repo):
        """context_pack includes the function body from source file."""
        result = graph.context_pack("foo", repo_root=str(graph_repo))
        assert "def foo(x, y):" in result
        assert "src.main.foo" in result
        assert "[Function]" in result

    def test_context_pack_contains_callers(self, graph_repo):
        """context_pack includes callers section."""
        result = graph.context_pack("foo", repo_root=str(graph_repo))
        assert "Callers" in result
        assert "src.bar.Bar.do_thing" in result

    def test_context_pack_contains_tests(self, graph_repo):
        """context_pack includes tests section."""
        result = graph.context_pack("foo", repo_root=str(graph_repo))
        assert "Tests" in result
        assert "tests.test_main.test_foo" in result

    def test_context_pack_location(self, graph_repo):
        """context_pack includes file location."""
        result = graph.context_pack("foo", repo_root=str(graph_repo))
        assert "src/main.py:10-25" in result

    def test_context_pack_symbol_not_found(self, graph_repo):
        """Returns 'Symbol not found' for unknown symbol."""
        result = graph.context_pack("nonexistent", repo_root=str(graph_repo))
        assert result == "Symbol not found: nonexistent"

    def test_context_pack_truncation(self, graph_repo):
        """context_pack truncates output over 2KB."""
        # Create a source file with very long lines to exceed 2KB
        src_main = graph_repo / "src" / "main.py"
        lines = []
        for i in range(1, 30):
            if i == 10:
                lines.append("def foo(x, y):")
            elif 10 < i <= 25:
                lines.append("    x = " + "a" * 200 + f"  # line {i}")
            else:
                lines.append(f"# line {i}")
        src_main.write_text("\n".join(lines) + "\n")

        result = graph.context_pack("foo", repo_root=str(graph_repo))
        # If truncated, should end with truncation marker
        if len(result) > 2048:
            pytest.fail("context_pack should truncate at 2KB")


# ---------------------------------------------------------------------------
# knowledge — minimal testing (requires cairn DB)
# ---------------------------------------------------------------------------

class TestKnowledge:

    def test_knowledge_no_cairn_db(self, graph_repo, monkeypatch):
        """Returns 'Cairn DB not found' when cairn DB doesn't exist."""
        monkeypatch.setenv("CAIRN_HOME", str(graph_repo / "no-cairn-here"))
        result = graph.knowledge("foo", repo_root=str(graph_repo))
        assert "Cairn DB not found" in result

    def test_knowledge_symbol_not_found(self, graph_repo):
        """Returns 'Symbol not found' for unknown symbol."""
        result = graph.knowledge("nonexistent", repo_root=str(graph_repo))
        assert result == "Symbol not found: nonexistent"


# ---------------------------------------------------------------------------
# orientation_block (Tier 1) + file_context_block (Tier 2)
# ---------------------------------------------------------------------------

COMMUNITY_SCHEMA = """
CREATE TABLE community_summaries (
    community_id INTEGER PRIMARY KEY,
    name TEXT, purpose TEXT, key_symbols TEXT, risk TEXT,
    size INTEGER, dominant_language TEXT
);
"""
FLOW_SCHEMA = """
CREATE TABLE flow_snapshots (
    flow_id INTEGER PRIMARY KEY,
    name TEXT, entry_point TEXT, critical_path TEXT,
    criticality REAL, node_count INTEGER, file_count INTEGER
);
"""
RISK_SCHEMA = """
CREATE TABLE risk_index (
    node_id INTEGER, qualified_name TEXT, risk_score REAL,
    caller_count INTEGER, test_coverage TEXT, security_relevant INTEGER, last_computed TEXT
);
"""


@pytest.fixture
def rich_graph_repo(graph_repo):
    """Extend graph_repo with postprocess tables (communities, flows, risk)."""
    db = graph_repo / ".code-review-graph" / "graph.db"
    conn = sqlite3.connect(str(db))
    conn.execute(COMMUNITY_SCHEMA)
    conn.execute(FLOW_SCHEMA)
    conn.execute(RISK_SCHEMA)
    # A real module and a test-dominated "module" (should be filtered out).
    conn.execute("INSERT INTO community_summaries VALUES (1,'core','','[\"foo\",\"do_thing\"]','',40,'python')")
    conn.execute("INSERT INTO community_summaries VALUES (2,'suite','','[\"test_foo\",\"test_bar\",\"test_baz\"]','',99,'python')")
    conn.execute("INSERT INTO flow_snapshots VALUES (1,'foo','src/main.py::foo','[]',0.81,5,2)")
    # foo: high risk + security relevant + untested → must be flagged.
    conn.execute("INSERT INTO risk_index VALUES (1,'src.main.foo',0.7,1,'untested',1,'now')")
    conn.commit()
    conn.close()
    return graph_repo


class TestOrientationBlock:
    def test_none_when_no_graph(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert graph.orientation_block() is None

    def test_modules_flows_hubs(self, rich_graph_repo):
        block = graph.orientation_block(repo_root=str(rich_graph_repo))
        assert block is not None
        assert "Modules:" in block and "core" in block
        assert "Key flows" in block and "foo" in block
        assert "Hubs" in block  # foo has 1 caller via CALLS edge

    def test_excludes_test_dominated_community(self, rich_graph_repo):
        block = graph.orientation_block(repo_root=str(rich_graph_repo))
        # 'suite' is >50% test_* symbols → must be dropped from Modules.
        assert "suite" not in block

    def test_works_without_postprocess_tables(self, graph_repo):
        # No communities/flows/risk tables — should still emit hubs, not crash.
        block = graph.orientation_block(repo_root=str(graph_repo))
        assert block is not None and "Hubs" in block


class TestFileContextBlock:
    def test_none_when_no_graph(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert graph.file_context_block("src/main.py") is None

    def test_none_for_unknown_file(self, graph_repo):
        assert graph.file_context_block(str(graph_repo / "src" / "nope.py"),
                                        repo_root=str(graph_repo)) is None

    def test_signatures_and_fanin(self, graph_repo):
        block = graph.file_context_block(str(graph_repo / "src" / "main.py"),
                                         repo_root=str(graph_repo))
        assert block is not None
        assert "foo" in block
        assert "callers:1" in block  # do_thing CALLS foo

    def test_includes_file_line_locations(self, graph_repo):
        # Each surfaced symbol must carry its file:line so it's jump-to-able
        # without a follow-up --location call. foo is at line 10 in src/main.py.
        block = graph.file_context_block(str(graph_repo / "src" / "main.py"),
                                         repo_root=str(graph_repo))
        assert block is not None
        assert "src/main.py:10" in block

    def test_excludes_tests(self, graph_repo):
        # test_foo lives in tests/test_main.py (is_test=1) — not a target file here,
        # but ensure a file with only test nodes yields None.
        block = graph.file_context_block(str(graph_repo / "tests" / "test_main.py"),
                                         repo_root=str(graph_repo))
        assert block is None

    def test_risk_tail_highlight(self, rich_graph_repo):
        block = graph.file_context_block(str(rich_graph_repo / "src" / "main.py"),
                                         repo_root=str(rich_graph_repo))
        assert block is not None
        assert "High-risk:" in block
        assert "security-relevant" in block
        assert "foo" in block

    def test_respects_max_symbols(self, graph_repo):
        block = graph.file_context_block(str(graph_repo / "src" / "bar.py"),
                                         repo_root=str(graph_repo), max_symbols=1)
        assert block is not None
