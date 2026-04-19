"""Tree-sitter AST parsing for Cairn repo ingestion.

Extracts structured code elements (functions, classes, imports, exports)
from source files using tree-sitter grammars. Falls back gracefully when
tree-sitter or a language grammar is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_PARSERS: dict[str, object] = {}
_AVAILABLE = None


def _check_available():
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        from tree_sitter import Parser  # noqa: F401
        _AVAILABLE = True
    except ImportError:
        _AVAILABLE = False
    return _AVAILABLE


def _get_parser(lang_key: str):
    if not _check_available():
        return None
    if lang_key in _PARSERS:
        return _PARSERS[lang_key]

    from tree_sitter import Language, Parser

    lang_fns = {
        "python": ("tree_sitter_python", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"),
        "go": ("tree_sitter_go", "language"),
        "rust": ("tree_sitter_rust", "language"),
        "c": ("tree_sitter_c", "language"),
        "cpp": ("tree_sitter_cpp", "language"),
    }

    spec = lang_fns.get(lang_key)
    if not spec:
        _PARSERS[lang_key] = None
        return None

    mod_name, fn_name = spec
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        lang_fn = getattr(mod, fn_name)
        language = Language(lang_fn())
        parser = Parser(language)
        _PARSERS[lang_key] = parser
        return parser
    except (ImportError, AttributeError, Exception):
        _PARSERS[lang_key] = None
        return None


EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
}


def lang_for_file(path: str | Path) -> Optional[str]:
    ext = Path(path).suffix.lower()
    return EXTENSION_MAP.get(ext)


def parse_file(path: str | Path, content: Optional[bytes] = None) -> Optional[object]:
    lang = lang_for_file(path)
    if not lang:
        return None
    parser = _get_parser(lang)
    if not parser:
        return None
    if content is None:
        try:
            content = Path(path).read_bytes()
        except (OSError, PermissionError):
            return None
    return parser.parse(content)


# --- Node queries ---

def _node_text(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _find_children(node, *types):
    return [c for c in node.children if c.type in types]


def _find_first(node, *types):
    for c in node.children:
        if c.type in types:
            return c
    return None


# --- Python extractors ---

def _extract_python_function(node, scope: str = "") -> dict:
    name_node = _find_first(node, "identifier")
    name = _node_text(name_node) if name_node else "?"
    params = _find_first(node, "parameters")
    params_text = _node_text(params) if params else "()"
    ret = _find_first(node, "type")
    ret_text = f" -> {_node_text(ret)}" if ret else ""
    decorators = []
    for c in node.children:
        if c.type == "decorator":
            decorators.append(_node_text(c))
    full_name = f"{scope}.{name}" if scope else name
    return {
        "name": full_name,
        "params": params_text,
        "return_type": ret_text.strip(),
        "decorators": decorators,
        "line": node.start_point[0] + 1,
        "kind": "method" if scope else "function",
    }


def _extract_python_class(node) -> dict:
    name_node = _find_first(node, "identifier")
    name = _node_text(name_node) if name_node else "?"
    bases = _find_first(node, "argument_list")
    bases_text = _node_text(bases) if bases else ""
    body = _find_first(node, "block")
    methods = []
    if body:
        for child in body.children:
            if child.type == "function_definition":
                methods.append(_extract_python_function(child, scope=name))
            elif child.type == "decorated_definition":
                fn = _find_first(child, "function_definition")
                if fn:
                    methods.append(_extract_python_function(fn, scope=name))
    return {
        "name": name,
        "bases": bases_text,
        "methods": methods,
        "line": node.start_point[0] + 1,
    }


def _extract_python_import(node) -> list[dict]:
    imports = []
    if node.type == "import_statement":
        for child in node.children:
            if child.type == "dotted_name":
                imports.append({"module": _node_text(child), "names": [], "line": node.start_point[0] + 1})
    elif node.type == "import_from_statement":
        module_node = _find_first(node, "dotted_name", "relative_import")
        module = _node_text(module_node) if module_node else ""
        names = []
        for child in node.children:
            if child.type == "import_from_names" or child.type == "import_prefix":
                continue
            if child.type in ("dotted_name", "identifier") and child != module_node:
                names.append(_node_text(child))
            elif child.type == "aliased_import":
                name_node = _find_first(child, "identifier", "dotted_name")
                if name_node:
                    names.append(_node_text(name_node))
        imports.append({"module": module, "names": names, "line": node.start_point[0] + 1})
    return imports


def extract_python(tree) -> dict:
    root = tree.root_node
    functions = []
    classes = []
    imports = []

    for node in root.children:
        if node.type == "function_definition":
            functions.append(_extract_python_function(node))
        elif node.type == "decorated_definition":
            fn = _find_first(node, "function_definition")
            cls = _find_first(node, "class_definition")
            if fn:
                functions.append(_extract_python_function(fn))
            elif cls:
                classes.append(_extract_python_class(cls))
        elif node.type == "class_definition":
            classes.append(_extract_python_class(node))
        elif node.type in ("import_statement", "import_from_statement"):
            imports.extend(_extract_python_import(node))

    return {"functions": functions, "classes": classes, "imports": imports}


# --- JavaScript/TypeScript extractors ---

def _extract_js_function(node, scope: str = "") -> dict:
    name = "?"
    params_text = "()"
    name_node = _find_first(node, "identifier", "property_identifier")
    if name_node:
        name = _node_text(name_node)
    params = _find_first(node, "formal_parameters")
    if params:
        params_text = _node_text(params)
    full_name = f"{scope}.{name}" if scope else name
    kind = "method" if scope else "function"
    is_async = any(c.type == "async" for c in node.children)
    return {
        "name": full_name,
        "params": params_text,
        "async": is_async,
        "line": node.start_point[0] + 1,
        "kind": kind,
    }


def _extract_js_class(node) -> dict:
    name_node = _find_first(node, "identifier", "type_identifier")
    name = _node_text(name_node) if name_node else "?"
    heritage = _find_first(node, "class_heritage")
    extends = ""
    if heritage:
        extends = _node_text(heritage)
    body = _find_first(node, "class_body")
    methods = []
    if body:
        for child in body.children:
            if child.type in ("method_definition", "public_field_definition"):
                if child.type == "method_definition":
                    methods.append(_extract_js_function(child, scope=name))
    return {
        "name": name,
        "extends": extends,
        "methods": methods,
        "line": node.start_point[0] + 1,
    }


def _extract_js_import(node) -> list[dict]:
    imports = []
    source = None
    names = []
    for child in node.children:
        if child.type == "string":
            source = _node_text(child).strip("'\"")
        elif child.type == "import_clause":
            for sub in child.children:
                if sub.type == "identifier":
                    names.append(_node_text(sub))
                elif sub.type == "named_imports":
                    for spec in sub.children:
                        if spec.type == "import_specifier":
                            n = _find_first(spec, "identifier")
                            if n:
                                names.append(_node_text(n))
                elif sub.type == "namespace_import":
                    n = _find_first(sub, "identifier")
                    if n:
                        names.append(f"* as {_node_text(n)}")
    if source:
        imports.append({"module": source, "names": names, "line": node.start_point[0] + 1})
    return imports


def _extract_js_export(node) -> Optional[dict]:
    declaration = _find_first(node, "function_declaration", "class_declaration",
                              "lexical_declaration", "variable_declaration")
    if declaration:
        name_node = _find_first(declaration, "identifier", "type_identifier")
        name = _node_text(name_node) if name_node else "?"
        return {"name": name, "kind": declaration.type, "line": node.start_point[0] + 1}
    return None


def extract_javascript(tree) -> dict:
    root = tree.root_node
    functions = []
    classes = []
    imports = []
    exports = []

    for node in root.children:
        if node.type in ("function_declaration", "generator_function_declaration"):
            functions.append(_extract_js_function(node))
        elif node.type == "class_declaration":
            classes.append(_extract_js_class(node))
        elif node.type == "import_statement":
            imports.extend(_extract_js_import(node))
        elif node.type == "export_statement":
            exp = _extract_js_export(node)
            if exp:
                exports.append(exp)
            inner = _find_first(node, "function_declaration", "class_declaration")
            if inner:
                if inner.type == "function_declaration":
                    functions.append(_extract_js_function(inner))
                elif inner.type == "class_declaration":
                    classes.append(_extract_js_class(inner))
        elif node.type == "expression_statement":
            call = _find_first(node, "assignment_expression", "call_expression")
            if not call:
                continue
            # Arrow functions assigned to const/let
        elif node.type == "lexical_declaration":
            for decl in _find_children(node, "variable_declarator"):
                value = _find_first(decl, "arrow_function", "function")
                if value:
                    name_node = _find_first(decl, "identifier")
                    name = _node_text(name_node) if name_node else "?"
                    params = _find_first(value, "formal_parameters")
                    params_text = _node_text(params) if params else "()"
                    is_async = any(c.type == "async" for c in value.children)
                    functions.append({
                        "name": name,
                        "params": params_text,
                        "async": is_async,
                        "line": node.start_point[0] + 1,
                        "kind": "function",
                    })

    return {"functions": functions, "classes": classes, "imports": imports, "exports": exports}


# JS/TS share the same structure
extract_typescript = extract_javascript


# --- Go extractors ---

def _extract_go_function(node) -> dict:
    name_node = _find_first(node, "identifier", "field_identifier")
    name = _node_text(name_node) if name_node else "?"
    params = _find_first(node, "parameter_list")
    params_text = _node_text(params) if params else "()"
    result = _find_first(node, "result")
    ret = _node_text(result) if result else ""
    receiver = ""
    for child in node.children:
        if child.type == "parameter_list" and child != params:
            receiver = _node_text(child)
            break
    return {
        "name": name,
        "params": params_text,
        "return_type": ret,
        "receiver": receiver,
        "line": node.start_point[0] + 1,
        "kind": "method" if receiver else "function",
    }


def _extract_go_type(node) -> Optional[dict]:
    spec = _find_first(node, "type_spec")
    if not spec:
        return None
    name_node = _find_first(spec, "type_identifier")
    name = _node_text(name_node) if name_node else "?"
    type_node = _find_first(spec, "struct_type", "interface_type")
    kind = "struct" if type_node and type_node.type == "struct_type" else "interface"
    fields = []
    if type_node:
        field_list = _find_first(type_node, "field_declaration_list", "method_spec_list")
        if field_list:
            for child in field_list.children:
                if child.type in ("field_declaration", "method_spec"):
                    fields.append(_node_text(child))
    return {"name": name, "kind": kind, "fields": fields, "line": node.start_point[0] + 1}


def _extract_go_import(node) -> list[dict]:
    imports = []
    for child in node.children:
        if child.type == "import_spec":
            path_node = _find_first(child, "interpreted_string_literal")
            path = _node_text(path_node).strip('"') if path_node else ""
            alias_node = _find_first(child, "package_identifier", "blank_identifier", "dot")
            alias = _node_text(alias_node) if alias_node else ""
            imports.append({"module": path, "alias": alias, "line": child.start_point[0] + 1})
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    path_node = _find_first(spec, "interpreted_string_literal")
                    path = _node_text(path_node).strip('"') if path_node else ""
                    alias_node = _find_first(spec, "package_identifier", "blank_identifier", "dot")
                    alias = _node_text(alias_node) if alias_node else ""
                    imports.append({"module": path, "alias": alias, "line": spec.start_point[0] + 1})
    return imports


def extract_go(tree) -> dict:
    root = tree.root_node
    functions = []
    types = []
    imports = []

    for node in root.children:
        if node.type == "function_declaration":
            functions.append(_extract_go_function(node))
        elif node.type == "method_declaration":
            functions.append(_extract_go_function(node))
        elif node.type == "type_declaration":
            t = _extract_go_type(node)
            if t:
                types.append(t)
        elif node.type == "import_declaration":
            imports.extend(_extract_go_import(node))

    return {"functions": functions, "types": types, "imports": imports}


# --- Rust extractors ---

def _extract_rust_function(node, scope: str = "") -> dict:
    name_node = _find_first(node, "identifier")
    name = _node_text(name_node) if name_node else "?"
    params = _find_first(node, "parameters")
    params_text = _node_text(params) if params else "()"
    ret = _find_first(node, "return_type")
    ret_text = _node_text(ret) if ret else ""
    vis = _find_first(node, "visibility_modifier")
    pub = bool(vis)
    full_name = f"{scope}::{name}" if scope else name
    return {
        "name": full_name,
        "params": params_text,
        "return_type": ret_text,
        "public": pub,
        "line": node.start_point[0] + 1,
        "kind": "method" if scope else "function",
    }


def _extract_rust_struct(node) -> dict:
    name_node = _find_first(node, "type_identifier")
    name = _node_text(name_node) if name_node else "?"
    fields = []
    body = _find_first(node, "field_declaration_list")
    if body:
        for child in body.children:
            if child.type == "field_declaration":
                fields.append(_node_text(child))
    vis = _find_first(node, "visibility_modifier")
    return {
        "name": name,
        "fields": fields,
        "public": bool(vis),
        "line": node.start_point[0] + 1,
    }


def _extract_rust_impl(node) -> dict:
    type_node = _find_first(node, "type_identifier", "generic_type", "scoped_type_identifier")
    name = _node_text(type_node) if type_node else "?"
    trait_node = None
    for i, child in enumerate(node.children):
        if child.type == "for" and i > 0:
            trait_node = node.children[i - 1]
            break
    trait_name = _node_text(trait_node) if trait_node else None
    body = _find_first(node, "declaration_list")
    methods = []
    if body:
        for child in body.children:
            if child.type == "function_item":
                methods.append(_extract_rust_function(child, scope=name))
    return {
        "name": name,
        "trait": trait_name,
        "methods": methods,
        "line": node.start_point[0] + 1,
    }


def _extract_rust_import(node) -> list[dict]:
    text = _node_text(node)
    return [{"module": text, "line": node.start_point[0] + 1}]


def extract_rust(tree) -> dict:
    root = tree.root_node
    functions = []
    structs = []
    impls = []
    imports = []
    traits = []
    enums = []

    for node in root.children:
        if node.type == "function_item":
            functions.append(_extract_rust_function(node))
        elif node.type == "struct_item":
            structs.append(_extract_rust_struct(node))
        elif node.type == "impl_item":
            impls.append(_extract_rust_impl(node))
        elif node.type == "use_declaration":
            imports.extend(_extract_rust_import(node))
        elif node.type == "trait_item":
            name_node = _find_first(node, "type_identifier")
            name = _node_text(name_node) if name_node else "?"
            traits.append({"name": name, "line": node.start_point[0] + 1})
        elif node.type == "enum_item":
            name_node = _find_first(node, "type_identifier")
            name = _node_text(name_node) if name_node else "?"
            enums.append({"name": name, "line": node.start_point[0] + 1})

    return {
        "functions": functions, "structs": structs, "impls": impls,
        "imports": imports, "traits": traits, "enums": enums,
    }


# --- C/C++ extractors ---

def _extract_c_function(node, scope: str = "") -> dict:
    decl = _find_first(node, "function_declarator")
    name = "?"
    params_text = "()"
    if decl:
        name_node = _find_first(decl, "identifier", "field_identifier",
                                "destructor_name", "qualified_identifier")
        name = _node_text(name_node) if name_node else "?"
        params = _find_first(decl, "parameter_list")
        params_text = _node_text(params) if params else "()"
    ret_node = _find_first(node, "primitive_type", "type_identifier",
                           "sized_type_specifier", "template_type")
    ret = _node_text(ret_node) if ret_node else ""
    full_name = f"{scope}::{name}" if scope else name
    return {
        "name": full_name,
        "params": params_text,
        "return_type": ret,
        "line": node.start_point[0] + 1,
        "kind": "method" if scope else "function",
    }


def _extract_c_struct(node) -> dict:
    name_node = _find_first(node, "type_identifier")
    name = _node_text(name_node) if name_node else "<anonymous>"
    body = _find_first(node, "field_declaration_list")
    fields = []
    if body:
        for child in body.children:
            if child.type == "field_declaration":
                fields.append(_node_text(child).rstrip(";").strip())
    return {"name": name, "fields": fields, "line": node.start_point[0] + 1}


def extract_c(tree) -> dict:
    root = tree.root_node
    functions = []
    structs = []
    includes = []

    for node in root.children:
        if node.type == "function_definition":
            functions.append(_extract_c_function(node))
        elif node.type == "declaration":
            fn_decl = _find_first(node, "function_declarator")
            if fn_decl:
                functions.append(_extract_c_function(node))
        elif node.type in ("struct_specifier", "union_specifier"):
            structs.append(_extract_c_struct(node))
        elif node.type == "preproc_include":
            path_node = _find_first(node, "string_literal", "system_lib_string")
            if path_node:
                includes.append({"path": _node_text(path_node), "line": node.start_point[0] + 1})
        elif node.type == "type_definition":
            inner = _find_first(node, "struct_specifier", "union_specifier", "enum_specifier")
            if inner:
                structs.append(_extract_c_struct(inner))

    return {"functions": functions, "structs": structs, "includes": includes}


def extract_cpp(tree) -> dict:
    result = extract_c(tree)
    root = tree.root_node
    classes = []
    namespaces = []

    for node in root.children:
        if node.type == "class_specifier":
            name_node = _find_first(node, "type_identifier")
            name = _node_text(name_node) if name_node else "?"
            body = _find_first(node, "field_declaration_list")
            methods = []
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        methods.append(_extract_c_function(child, scope=name))
                    elif child.type == "declaration":
                        fn = _find_first(child, "function_declarator")
                        if fn:
                            methods.append(_extract_c_function(child, scope=name))
            classes.append({"name": name, "methods": methods, "line": node.start_point[0] + 1})
        elif node.type == "namespace_definition":
            name_node = _find_first(node, "identifier", "namespace_identifier")
            name = _node_text(name_node) if name_node else "<anonymous>"
            namespaces.append({"name": name, "line": node.start_point[0] + 1})

    result["classes"] = classes
    result["namespaces"] = namespaces
    return result


# --- Unified extraction entry point ---

_LANG_EXTRACTORS = {
    "python": extract_python,
    "javascript": extract_javascript,
    "typescript": extract_typescript,
    "tsx": extract_typescript,
    "go": extract_go,
    "rust": extract_rust,
    "c": extract_c,
    "cpp": extract_cpp,
}


def extract_file_ast(path: str | Path, content: Optional[bytes] = None) -> Optional[dict]:
    """Parse a file and extract structured code elements.

    Returns dict with language-specific keys (functions, classes, imports, etc.)
    or None if tree-sitter unavailable or language unsupported.
    """
    lang = lang_for_file(path)
    if not lang:
        return None
    tree = parse_file(path, content)
    if not tree:
        return None
    extractor = _LANG_EXTRACTORS.get(lang)
    if not extractor:
        return None
    result = extractor(tree)
    result["language"] = lang
    return result


def extract_repo_ast(repo_path: str | Path, walk_fn=None, max_file_size=100_000) -> dict:
    """Extract AST data from all supported files in a repo.

    Args:
        repo_path: Root directory to scan
        walk_fn: Optional file walker (yields (path, rel_path) tuples)
        max_file_size: Skip files larger than this

    Returns dict mapping relative file paths to their AST extractions.
    """
    root = Path(repo_path).resolve()
    results = {}

    if walk_fn:
        files = walk_fn(repo_path)
    else:
        files = _default_walk(root)

    for fp, rel in files:
        if not lang_for_file(fp):
            continue
        try:
            size = fp.stat().st_size
            if size > max_file_size:
                continue
            content = fp.read_bytes()
        except (OSError, PermissionError):
            continue

        ast_data = extract_file_ast(fp, content)
        if ast_data:
            has_content = False
            for key, val in ast_data.items():
                if key == "language":
                    continue
                if isinstance(val, list) and val:
                    has_content = True
                    break
            if has_content:
                results[rel] = ast_data

    return results


def _default_walk(root: Path):
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
        ".tox", ".venv", "venv", "env", ".env", "dist", "build", ".next",
        ".nuxt", "coverage", ".idea", ".vscode", "target", "vendor",
        ".terraform", ".cache", ".turbo", "out", ".output",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts) if str(rel) != "." else 0
        if depth >= 6:
            dirnames.clear()
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for f in sorted(filenames):
            fp = Path(dirpath) / f
            yield fp, str(fp.relative_to(root))
