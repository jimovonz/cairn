"""Tests for dependency graph extraction module."""

import pytest

from cairn.ast_parser import _check_available, extract_file_ast
from cairn.dep_graph import (
    build_full_graph,
    build_import_graph,
    build_inheritance_graph,
    build_symbol_index,
    graph_to_extraction_format,
    _resolve_js_import,
)


pytestmark = pytest.mark.skipif(
    not _check_available(),
    reason="tree-sitter not installed",
)


def _make_ast_results(**files):
    """Build ast_results dict from file_path=code pairs."""
    results = {}
    for path, code in files.items():
        data = extract_file_ast(path, code.encode() if isinstance(code, str) else code)
        if data:
            results[path] = data
    return results


class TestImportGraph:
    def test_python_imports(self):
        ast = _make_ast_results(**{
            "app.py": "import os\nfrom pathlib import Path\n",
        })
        edges = build_import_graph(ast)
        assert len(edges) == 2
        targets = [e["target"] for e in edges]
        assert "os" in targets
        assert "pathlib" in targets
        assert all(e["kind"] == "imports" for e in edges)

    def test_js_relative_import(self):
        ast = _make_ast_results(**{
            "src/app.js": "import { foo } from './utils';\n",
        })
        edges = build_import_graph(ast)
        assert len(edges) == 1
        assert edges[0]["target"] == "src/utils"

    def test_js_package_import(self):
        ast = _make_ast_results(**{
            "app.js": "import express from 'express';\n",
        })
        edges = build_import_graph(ast)
        assert len(edges) == 1
        assert edges[0]["target"] == "express"

    def test_go_imports(self):
        ast = _make_ast_results(**{
            "main.go": 'package main\n\nimport (\n    "fmt"\n    "os"\n)\n',
        })
        edges = build_import_graph(ast)
        assert len(edges) == 2
        targets = [e["target"] for e in edges]
        assert "fmt" in targets

    def test_rust_use(self):
        ast = _make_ast_results(**{
            "lib.rs": "use std::io;\nuse std::collections::HashMap;\n",
        })
        edges = build_import_graph(ast)
        assert len(edges) == 2
        assert all(e["kind"] == "imports" for e in edges)

    def test_c_includes(self):
        ast = _make_ast_results(**{
            "main.c": '#include <stdio.h>\n#include "mylib.h"\n',
        })
        edges = build_import_graph(ast)
        assert len(edges) == 2
        assert all(e["kind"] == "includes" for e in edges)

    def test_multi_file(self):
        ast = _make_ast_results(**{
            "a.py": "import os\n",
            "b.py": "import sys\nimport json\n",
        })
        edges = build_import_graph(ast)
        assert len(edges) == 3
        sources = set(e["source"] for e in edges)
        assert sources == {"a.py", "b.py"}


class TestInheritanceGraph:
    def test_python_inheritance(self):
        ast = _make_ast_results(**{
            "models.py": "class Animal:\n    pass\n\nclass Dog(Animal):\n    pass\n",
        })
        edges = build_inheritance_graph(ast)
        assert len(edges) == 1
        assert edges[0]["kind"] == "extends"
        assert "Dog" in edges[0]["source"]
        assert edges[0]["target"] == "Animal"

    def test_js_extends(self):
        ast = _make_ast_results(**{
            "app.js": "class Dog extends Animal {\n    bark() {}\n}\n",
        })
        edges = build_inheritance_graph(ast)
        assert len(edges) == 1
        assert edges[0]["kind"] == "extends"

    def test_rust_trait_impl(self):
        ast = _make_ast_results(**{
            "lib.rs": "trait Speak { fn speak(&self); }\nstruct Cat;\nimpl Speak for Cat { fn speak(&self) {} }\n",
        })
        edges = build_inheritance_graph(ast)
        assert len(edges) == 1
        assert edges[0]["kind"] == "implements"

    def test_no_object_base(self):
        ast = _make_ast_results(**{
            "test.py": "class Foo(object):\n    pass\n",
        })
        edges = build_inheritance_graph(ast)
        assert len(edges) == 0


class TestSymbolIndex:
    def test_function_indexed(self):
        ast = _make_ast_results(**{
            "util.py": "def helper():\n    pass\n",
        })
        index = build_symbol_index(ast)
        assert "helper" in index
        assert index["helper"][0]["file"] == "util.py"
        assert index["helper"][0]["kind"] == "function"

    def test_class_and_methods(self):
        ast = _make_ast_results(**{
            "models.py": "class Dog:\n    def bark(self):\n        pass\n",
        })
        index = build_symbol_index(ast)
        assert "Dog" in index
        assert "bark" in index
        assert index["bark"][0]["scope"] == "Dog"

    def test_go_types(self):
        ast = _make_ast_results(**{
            "main.go": "package main\n\ntype Server struct {\n    Port int\n}\n",
        })
        index = build_symbol_index(ast)
        assert "Server" in index
        assert index["Server"][0]["kind"] == "struct"

    def test_hotspot_detection(self):
        ast = _make_ast_results(**{
            "a.py": "def main():\n    pass\n",
            "b.py": "def main():\n    pass\n",
        })
        index = build_symbol_index(ast)
        assert "main" in index
        assert len(index["main"]) == 2


class TestFullGraph:
    def test_complete_graph(self):
        ast = _make_ast_results(**{
            "app.py": "import os\nfrom pathlib import Path\n\nclass App:\n    def run(self):\n        pass\n",
        })
        graph = build_full_graph(ast)
        assert "import_edges" in graph
        assert "inheritance_edges" in graph
        assert "symbol_index" in graph
        assert "stats" in graph
        assert graph["stats"]["files_parsed"] == 1
        assert graph["stats"]["import_edges"] == 2
        assert "python" in graph["stats"]["languages"]

    def test_extraction_format(self):
        ast = _make_ast_results(**{
            "a.py": "import os\nclass Foo:\n    pass\n",
            "b.py": "import sys\nclass Bar(Foo):\n    pass\n",
        })
        graph = build_full_graph(ast)
        fmt = graph_to_extraction_format(graph)
        assert "import_map" in fmt
        assert "inheritance" in fmt
        assert "hotspots" in fmt
        assert "stats" in fmt
        assert len(fmt["import_map"]) == 2


class TestResolveJsImport:
    def test_relative(self):
        assert _resolve_js_import("src/app.js", "./utils") == "src/utils"

    def test_parent_dir(self):
        assert _resolve_js_import("src/components/App.js", "../utils") == "src/utils"

    def test_package(self):
        assert _resolve_js_import("app.js", "express") == "express"

    def test_scoped_package(self):
        assert _resolve_js_import("app.js", "@org/pkg") == "@org/pkg"
