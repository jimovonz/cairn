"""Dependency graph extraction for Cairn repo ingestion.

Builds import/call/inheritance edges from AST data produced by ast_parser.
Stores edges in a lightweight format suitable for SQLite persistence.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional


def build_import_graph(ast_results: dict) -> list[dict]:
    """Build import edges from AST extraction results.

    Args:
        ast_results: dict mapping rel_path -> ast_data (from extract_repo_ast)

    Returns list of edge dicts: {source, target, kind, line}
    """
    edges = []

    for file_path, ast_data in ast_results.items():
        lang = ast_data.get("language", "")

        if lang == "python":
            for imp in ast_data.get("imports", []):
                module = imp.get("module", "")
                if module:
                    edges.append({
                        "source": file_path,
                        "target": module,
                        "kind": "imports",
                        "names": imp.get("names", []),
                        "line": imp.get("line", 0),
                    })

        elif lang in ("javascript", "typescript", "tsx"):
            for imp in ast_data.get("imports", []):
                module = imp.get("module", "")
                if module:
                    resolved = _resolve_js_import(file_path, module)
                    edges.append({
                        "source": file_path,
                        "target": resolved,
                        "kind": "imports",
                        "names": imp.get("names", []),
                        "line": imp.get("line", 0),
                    })

        elif lang == "go":
            for imp in ast_data.get("imports", []):
                module = imp.get("module", "")
                if module:
                    edges.append({
                        "source": file_path,
                        "target": module,
                        "kind": "imports",
                        "alias": imp.get("alias", ""),
                        "line": imp.get("line", 0),
                    })

        elif lang == "rust":
            for imp in ast_data.get("imports", []):
                module = imp.get("module", "")
                if module:
                    edges.append({
                        "source": file_path,
                        "target": module,
                        "kind": "imports",
                        "line": imp.get("line", 0),
                    })

        elif lang in ("c", "cpp"):
            for inc in ast_data.get("includes", []):
                path = inc.get("path", "").strip('"<>')
                if path:
                    edges.append({
                        "source": file_path,
                        "target": path,
                        "kind": "includes",
                        "line": inc.get("line", 0),
                    })

    return edges


def build_inheritance_graph(ast_results: dict) -> list[dict]:
    """Build class/struct inheritance edges from AST data."""
    edges = []

    for file_path, ast_data in ast_results.items():
        lang = ast_data.get("language", "")

        if lang == "python":
            for cls in ast_data.get("classes", []):
                bases = cls.get("bases", "")
                if bases:
                    bases = bases.strip("()")
                    for base in bases.split(","):
                        base = base.strip()
                        if base and base not in ("object",):
                            edges.append({
                                "source": f"{file_path}:{cls['name']}",
                                "target": base,
                                "kind": "extends",
                                "line": cls.get("line", 0),
                            })

        elif lang in ("javascript", "typescript", "tsx"):
            for cls in ast_data.get("classes", []):
                extends = cls.get("extends", "")
                if extends:
                    parent = extends.replace("extends", "").strip()
                    if parent:
                        edges.append({
                            "source": f"{file_path}:{cls['name']}",
                            "target": parent,
                            "kind": "extends",
                            "line": cls.get("line", 0),
                        })

        elif lang == "rust":
            for impl_block in ast_data.get("impls", []):
                trait_name = impl_block.get("trait")
                if trait_name:
                    edges.append({
                        "source": f"{file_path}:{impl_block['name']}",
                        "target": trait_name,
                        "kind": "implements",
                        "line": impl_block.get("line", 0),
                    })

    return edges


def build_symbol_index(ast_results: dict) -> dict[str, list[dict]]:
    """Build a symbol -> definitions index for cross-referencing.

    Returns dict mapping symbol name -> list of {file, line, kind, scope}.
    """
    index: dict[str, list[dict]] = {}

    for file_path, ast_data in ast_results.items():
        lang = ast_data.get("language", "")

        # Functions
        for fn in ast_data.get("functions", []):
            name = fn["name"].split(".")[-1].split("::")[-1]
            entry = {"file": file_path, "line": fn["line"], "kind": fn.get("kind", "function")}
            index.setdefault(name, []).append(entry)

        # Classes / structs / types
        for cls in ast_data.get("classes", []):
            index.setdefault(cls["name"], []).append({
                "file": file_path, "line": cls["line"], "kind": "class",
            })
            for method in cls.get("methods", []):
                method_name = method["name"].split(".")[-1].split("::")[-1]
                index.setdefault(method_name, []).append({
                    "file": file_path, "line": method["line"],
                    "kind": "method", "scope": cls["name"],
                })

        if lang == "go":
            for t in ast_data.get("types", []):
                index.setdefault(t["name"], []).append({
                    "file": file_path, "line": t["line"], "kind": t.get("kind", "type"),
                })

        if lang == "rust":
            for s in ast_data.get("structs", []):
                index.setdefault(s["name"], []).append({
                    "file": file_path, "line": s["line"], "kind": "struct",
                })
            for t in ast_data.get("traits", []):
                index.setdefault(t["name"], []).append({
                    "file": file_path, "line": t["line"], "kind": "trait",
                })
            for e in ast_data.get("enums", []):
                index.setdefault(e["name"], []).append({
                    "file": file_path, "line": e["line"], "kind": "enum",
                })
            for impl_block in ast_data.get("impls", []):
                for method in impl_block.get("methods", []):
                    method_name = method["name"].split("::")[-1]
                    index.setdefault(method_name, []).append({
                        "file": file_path, "line": method["line"],
                        "kind": "method", "scope": impl_block["name"],
                    })

        if lang in ("c", "cpp"):
            for s in ast_data.get("structs", []):
                if s["name"] != "<anonymous>":
                    index.setdefault(s["name"], []).append({
                        "file": file_path, "line": s["line"], "kind": "struct",
                    })

    return index


def build_full_graph(ast_results: dict) -> dict:
    """Build the complete dependency graph from AST results.

    Returns {
        "import_edges": [...],
        "inheritance_edges": [...],
        "symbol_index": {...},
        "stats": {...}
    }
    """
    import_edges = build_import_graph(ast_results)
    inheritance_edges = build_inheritance_graph(ast_results)
    symbol_index = build_symbol_index(ast_results)

    stats = {
        "files_parsed": len(ast_results),
        "import_edges": len(import_edges),
        "inheritance_edges": len(inheritance_edges),
        "symbols_indexed": len(symbol_index),
        "languages": sorted(set(
            d.get("language", "unknown") for d in ast_results.values()
        )),
    }

    return {
        "import_edges": import_edges,
        "inheritance_edges": inheritance_edges,
        "symbol_index": symbol_index,
        "stats": stats,
    }


def graph_to_extraction_format(graph: dict) -> dict:
    """Convert graph data to the format expected by the ingestion pipeline.

    Returns a dict suitable for inclusion in the extractions dict,
    which gets fingerprinted and sent to Haiku for distillation.
    """
    import_summary = {}
    for edge in graph.get("import_edges", []):
        source = edge["source"]
        target = edge["target"]
        import_summary.setdefault(source, []).append(target)

    inheritance_summary = []
    for edge in graph.get("inheritance_edges", []):
        inheritance_summary.append({
            "child": edge["source"],
            "parent": edge["target"],
            "kind": edge["kind"],
        })

    hotspots = []
    for name, defs in graph.get("symbol_index", {}).items():
        if len(defs) > 1:
            hotspots.append({
                "symbol": name,
                "definitions": len(defs),
                "files": list(set(d["file"] for d in defs)),
            })
    hotspots.sort(key=lambda x: x["definitions"], reverse=True)

    return {
        "import_map": dict(list(import_summary.items())[:100]),
        "inheritance": inheritance_summary[:50],
        "hotspots": hotspots[:30],
        "stats": graph.get("stats", {}),
    }


def _resolve_js_import(source_file: str, module_path: str) -> str:
    """Resolve relative JS/TS import paths to repo-relative paths."""
    if not module_path.startswith("."):
        return module_path
    source_dir = str(Path(source_file).parent)
    resolved = os.path.normpath(os.path.join(source_dir, module_path))
    return resolved.replace("\\", "/")
