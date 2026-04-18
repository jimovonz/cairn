#!/usr/bin/env python3
"""Repo ingestion for Cairn — two-phase architecture.

Phase 1: Mechanistic extraction (no LLM) — reads structured sources from a git
repository and produces categorised JSON.

Phase 2: LLM distillation — sends structured extracts to Haiku to produce
self-contained one-liner Cairn memory entries (fact/workflow/skill types).
"""

import argparse
import json
import os
import re
import select
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".venv", "venv", "env", ".env", "dist", "build", ".next",
    ".nuxt", "coverage", ".coverage", ".idea", ".vscode", "target",
    "vendor", ".terraform", ".cache", ".turbo", "out", ".output",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".db", ".sqlite", ".sqlite3",
    ".lock", ".map",
}

MAX_FILE_SIZE = 100_000  # 100KB per file
MAX_FILE_LINES = 500     # for content extraction


def _run_git(repo_path, *args):
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_path)] + list(args),
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _read_file(path, max_lines=MAX_FILE_LINES, ignore_size_limit=False):
    try:
        size = path.stat().st_size
        if not ignore_size_limit and size > MAX_FILE_SIZE:
            return None
        if size > MAX_FILE_SIZE:
            with open(path, errors="replace") as f:
                lines = [f.readline() for _ in range(max_lines)]
            return "".join(lines).rstrip("\n")
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return text
    except (OSError, UnicodeDecodeError):
        return None


def _is_skippable(path):
    return path.suffix.lower() in SKIP_EXTENSIONS


def _get_submodule_paths(repo_path):
    """Return set of relative paths that are git submodules."""
    root = Path(repo_path).resolve()
    gitmodules = root / ".gitmodules"
    if not gitmodules.is_file():
        return set()
    paths = set()
    content = gitmodules.read_text(errors="replace")
    for m in re.finditer(r'path\s*=\s*(.+)', content):
        paths.add(m.group(1).strip())
    return paths


def _walk_files(repo_path, max_depth=6, submodule_paths=None):
    root = Path(repo_path).resolve()
    if submodule_paths is None:
        submodule_paths = _get_submodule_paths(root)
    skip_abs = {str(root / p) for p in submodule_paths}
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts) if str(rel) != "." else 0
        if depth >= max_depth:
            dirnames.clear()
            continue
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and str(Path(dirpath) / d) not in skip_abs
        )
        for f in sorted(filenames):
            fp = Path(dirpath) / f
            if not _is_skippable(fp):
                yield fp, str(fp.relative_to(root))


# --- Extractors ---

def extract_git_info(repo_path):
    info = {"is_git": False}
    commit = _run_git(repo_path, "rev-parse", "HEAD")
    if commit is None:
        return info
    info["is_git"] = True
    info["commit"] = commit
    info["branch"] = _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    remote = _run_git(repo_path, "remote", "get-url", "origin")
    info["remote"] = remote
    info["local_only"] = remote is None
    submodules = _get_submodule_paths(repo_path)
    if submodules:
        info["submodules"] = sorted(submodules)
    return info


def extract_docs(repo_path):
    root = Path(repo_path)
    docs = []
    doc_names = [
        "README.md", "README", "README.txt", "README.rst",
        "CLAUDE.md", "ARCHITECTURE.md", "CONTRIBUTING.md",
        "DESIGN.md", "CHANGELOG.md", "API.md",
    ]
    for name in doc_names:
        p = root / name
        if p.is_file():
            content = _read_file(p)
            if content:
                docs.append({"file": name, "content": content})

    docs_dir = root / "docs"
    if docs_dir.is_dir():
        for f in sorted(docs_dir.rglob("*.md")):
            rel = str(f.relative_to(root))
            content = _read_file(f)
            if content:
                docs.append({"file": rel, "content": content})

    claude_dir = root / ".claude"
    if claude_dir.is_dir():
        for f in sorted(claude_dir.rglob("*.md")):
            rel = str(f.relative_to(root))
            content = _read_file(f)
            if content:
                docs.append({"file": rel, "content": content})

    return docs


def extract_dependencies(repo_path):
    root = Path(repo_path)
    deps = {}

    pkg_json = root / "package.json"
    if pkg_json.is_file():
        try:
            data = json.loads(pkg_json.read_text())
            deps["package.json"] = {
                "name": data.get("name"),
                "version": data.get("version"),
                "scripts": data.get("scripts", {}),
                "dependencies": data.get("dependencies", {}),
                "devDependencies": data.get("devDependencies", {}),
                "engines": data.get("engines"),
                "type": data.get("type"),
                "main": data.get("main"),
            }
        except (json.JSONDecodeError, OSError):
            pass

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        content = _read_file(pyproject)
        if content:
            deps["pyproject.toml"] = content

    setup_py = root / "setup.py"
    if setup_py.is_file():
        content = _read_file(setup_py)
        if content:
            deps["setup.py"] = content

    setup_cfg = root / "setup.cfg"
    if setup_cfg.is_file():
        content = _read_file(setup_cfg)
        if content:
            deps["setup.cfg"] = content

    cargo = root / "Cargo.toml"
    if cargo.is_file():
        content = _read_file(cargo)
        if content:
            deps["Cargo.toml"] = content

    go_mod = root / "go.mod"
    if go_mod.is_file():
        content = _read_file(go_mod)
        if content:
            deps["go.mod"] = content

    gemfile = root / "Gemfile"
    if gemfile.is_file():
        content = _read_file(gemfile)
        if content:
            deps["Gemfile"] = content

    cmake = root / "CMakeLists.txt"
    if cmake.is_file():
        content = _read_file(cmake, max_lines=200, ignore_size_limit=True)
        if content:
            deps["CMakeLists.txt"] = content

    return deps


def extract_tree(repo_path, max_depth=4):
    root = Path(repo_path).resolve()
    tree = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts) if str(rel) != "." else 0
        if depth >= max_depth:
            dirnames.clear()
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        prefix = str(rel) + "/" if str(rel) != "." else ""
        for f in sorted(filenames):
            fp = Path(dirpath) / f
            if not _is_skippable(fp):
                tree.append(prefix + f)
    return tree


def extract_config(repo_path):
    root = Path(repo_path)
    configs = []
    config_patterns = [
        ".env.example", ".env.sample", ".env.template",
        "tsconfig.json", "tsconfig.*.json",
        "docker-compose.yml", "docker-compose.yaml", "Dockerfile",
        ".dockerignore",
        "Makefile", "Justfile",
        "nginx.conf",
        ".github/workflows/*.yml", ".github/workflows/*.yaml",
        ".gitlab-ci.yml",
        "fly.toml", "render.yaml", "railway.json",
        "vercel.json", "netlify.toml",
        ".eslintrc*", ".prettierrc*", "biome.json",
        "jest.config.*", "vitest.config.*", "pytest.ini", "tox.ini",
        "bunfig.toml",
    ]
    for pattern in config_patterns:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                content = _read_file(p)
                if content:
                    configs.append({"file": str(p.relative_to(root)), "content": content})
    return configs


def extract_schemas(repo_path):
    root = Path(repo_path)
    schemas = []
    schema_patterns = [
        "**/*.prisma",
        "**/*.proto",
        "**/migrations/**/*.sql",
        "**/schema.sql",
        "**/models.py",
        "**/models/*.py",
        "**/schema.py",
        "**/schemas/*.py",
    ]
    for pattern in schema_patterns:
        for p in sorted(root.glob(pattern)):
            if p.is_file() and not any(skip in p.parts for skip in SKIP_DIRS):
                content = _read_file(p)
                if content:
                    schemas.append({"file": str(p.relative_to(root)), "content": content})

    init_db = root / "cairn" / "init_db.py"
    if not init_db.exists():
        for p in sorted(root.rglob("init_db*")):
            if p.is_file() and not any(skip in p.parts for skip in SKIP_DIRS):
                content = _read_file(p)
                if content:
                    schemas.append({"file": str(p.relative_to(root)), "content": content})
    return schemas


def extract_entrypoints(repo_path):
    root = Path(repo_path)
    entries = []
    entry_names = [
        "main.py", "app.py", "server.py", "cli.py", "run.py",
        "main.ts", "app.ts", "server.ts", "index.ts",
        "main.js", "app.js", "server.js", "index.js",
        "main.go", "main.rs", "main.c", "main.cc", "main.cpp",
        "src/main.py", "src/app.py", "src/index.ts", "src/main.ts",
        "src/index.js", "src/main.js", "src/main.go", "src/main.rs",
        "src/main.c", "src/main.cc", "src/main.cpp",
        "bin/*.ts", "bin/*.js", "bin/*.py",
        "cmd/*/main.go",
    ]
    seen = set()
    for pattern in entry_names:
        for p in sorted(root.glob(pattern)):
            if p.is_file() and str(p) not in seen:
                seen.add(str(p))
                content = _read_file(p)
                if content:
                    entries.append({"file": str(p.relative_to(root)), "content": content})
    return entries


def extract_routes(repo_path):
    route_patterns = [
        # Python Flask/FastAPI decorators
        r'@(?:app|router|blueprint)\.(get|post|put|delete|patch|route)\s*\(\s*["\']([^"\']+)',
        # Express/Hono — require path starting with /
        r'(?:app|router)\.(get|post|put|delete|patch|all|use)\s*\(\s*["\'](/[^"\']*)',
        # Go net/http
        r'(?:Handle|HandleFunc)\s*\(\s*["\'](/[^"\']+)',
        # Bun.serve / fetch handler URL matching
        r'(?:url|pathname)\s*(?:===|==|\.startsWith\(|\.match\()\s*["\'](/[^"\']+)',
        r'new\s+URL\(.*?\).*?pathname\s*(?:===|==)\s*["\'](/[^"\']+)',
    ]
    cli_patterns = [
        # Python argparse
        r'add_argument\s*\(\s*["\']([^"\']+)',
        r'add_subparsers|ArgumentParser\(\s*(?:description\s*=\s*)?["\']([^"\']*)',
        # Click/Typer
        r'@(?:click\.command|app\.command|cli\.command)\s*\(\s*(?:name\s*=\s*)?["\']?([^"\')\s]*)',
    ]

    routes = []
    cli_args = []

    for fp, rel in _walk_files(repo_path):
        if fp.suffix not in (".py", ".ts", ".js", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp"):
            continue
        content = _read_file(fp, max_lines=1000)
        if not content:
            continue

        file_routes = []
        for pat in route_patterns:
            for m in re.finditer(pat, content):
                groups = [g for g in m.groups() if g]
                file_routes.append({"match": m.group(0).strip(), "groups": groups})
        if file_routes:
            routes.append({"file": rel, "routes": file_routes})

        file_cli = []
        for pat in cli_patterns:
            for m in re.finditer(pat, content):
                groups = [g for g in m.groups() if g]
                file_cli.append({"match": m.group(0).strip(), "groups": groups})
        if file_cli:
            cli_args.append({"file": rel, "args": file_cli})

    return {"http_routes": routes, "cli_interfaces": cli_args}


def extract_exports(repo_path):
    root = Path(repo_path)
    exports = []

    # Python __init__.py with __all__
    for p in sorted(root.rglob("__init__.py")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p)
        if content and "__all__" in content:
            exports.append({"file": str(p.relative_to(root)), "content": content})

    # TypeScript index.ts with exports
    for pattern in ["**/index.ts", "**/index.js"]:
        for p in sorted(root.glob(pattern)):
            if any(skip in p.parts for skip in SKIP_DIRS):
                continue
            content = _read_file(p)
            if content and re.search(r'export\s+', content):
                exports.append({"file": str(p.relative_to(root)), "content": content})

    return exports


SIGNAL_WORDS = re.compile(
    r'\b(because|since|NB|note|warning|depends|workaround|important|assumes|'
    r'careful|must not|never|always|caveat|gotcha|tricky|subtle|'
    r'temporary|deprecated)\b',
    re.IGNORECASE,
)

INLINE_COMMENT = re.compile(r'^\s*(#|//|/?\*+)\s*')

def extract_comments(repo_path):
    comments = []
    code_exts = {
        ".py", ".ts", ".js", ".go", ".rs", ".rb", ".java", ".sh", ".bash",
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
    }

    for fp, rel in _walk_files(repo_path):
        if fp.suffix not in code_exts:
            continue
        content = _read_file(fp, max_lines=2000)
        if not content:
            continue

        file_comments = []
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if not INLINE_COMMENT.match(stripped):
                continue
            # Skip docstrings, shebangs, section dividers
            if stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith("#!"):
                continue
            if len(stripped) < 15:
                continue
            if SIGNAL_WORDS.search(stripped):
                file_comments.append({"line": i, "text": stripped})

        if file_comments:
            comments.append({"file": rel, "comments": file_comments})

    return comments


TODO_MARKER = re.compile(r'(?:#|//|/?\*+|"""|<!--)\s*.{0,20}\b(TODO|HACK|FIXME|WORKAROUND|XXX)\b', re.IGNORECASE)

def extract_todos(repo_path):
    todos = []
    code_exts = {
        ".py", ".ts", ".js", ".go", ".rs", ".rb", ".java", ".sh", ".bash",
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
    }

    for fp, rel in _walk_files(repo_path):
        if fp.suffix not in code_exts:
            continue
        content = _read_file(fp, max_lines=2000)
        if not content:
            continue

        file_todos = []
        for i, line in enumerate(content.splitlines(), 1):
            if TODO_MARKER.search(line):
                file_todos.append({"line": i, "text": line.strip()})

        if file_todos:
            todos.append({"file": rel, "todos": file_todos})

    return todos


def extract_git_log(repo_path, count=50):
    log = _run_git(repo_path, "log", "--oneline", f"-{count}")
    if not log:
        return []
    return log.splitlines()


def extract_env_vars(repo_path):
    """Extract expected environment variables from code and config."""
    env_pattern = re.compile(
        r'(?:process\.env\.|os\.environ(?:\.get)?\s*\(\s*["\']|'
        r'os\.getenv\s*\(\s*["\']|'
        r'env\s*\(\s*["\']|'
        r'Env\.\w+\s*\(\s*["\'])'
        r'([A-Z][A-Z0-9_]+)',
    )
    env_vars = {}

    for fp, rel in _walk_files(repo_path):
        if fp.suffix not in (".py", ".ts", ".js", ".go"):
            continue
        content = _read_file(fp, max_lines=1000)
        if not content:
            continue
        for m in env_pattern.finditer(content):
            var = m.group(1)
            if var not in env_vars:
                env_vars[var] = []
            env_vars[var].append(rel)

    return env_vars


def extract_protobuf(repo_path):
    """Extract protobuf/gRPC service and message definitions."""
    root = Path(repo_path)
    results = []
    for p in sorted(root.rglob("*.proto")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=500)
        if not content:
            continue
        rel = str(p.relative_to(root))
        services = re.findall(r'service\s+(\w+)\s*\{([^}]*)\}', content, re.DOTALL)
        messages = re.findall(r'message\s+(\w+)\s*\{([^}]*)\}', content, re.DOTALL)
        rpcs = []
        for svc_name, svc_body in services:
            for m in re.finditer(r'rpc\s+(\w+)\s*\((\w+)\)\s*returns\s*\((\w+)\)', svc_body):
                rpcs.append({"service": svc_name, "method": m.group(1),
                             "request": m.group(2), "response": m.group(3)})
        msg_names = [m[0] for m in messages]
        if services or messages:
            results.append({"file": rel, "services": [s[0] for s in services],
                           "rpcs": rpcs, "messages": msg_names})
    return results


def extract_cmake_flags(repo_path):
    """Extract CMake option() and ENABLE_* flags."""
    root = Path(repo_path)
    flags = []
    for p in sorted(root.rglob("CMakeLists.txt")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=2000)
        if not content:
            continue
        rel = str(p.relative_to(root))
        for m in re.finditer(r'option\s*\(\s*(\w+)\s+"([^"]*)"\s+(ON|OFF)\s*\)', content):
            flags.append({"file": rel, "flag": m.group(1), "desc": m.group(2), "default": m.group(3)})
        for m in re.finditer(r'set\s*\(\s*(ENABLE_\w+)\s+(ON|OFF)', content):
            flags.append({"file": rel, "flag": m.group(1), "desc": "", "default": m.group(2)})
    return flags


def extract_event_interfaces(repo_path):
    """Extract pub/sub, webhook, and message queue patterns."""
    root = Path(repo_path)
    patterns = [
        (r'\.(?:publish|emit|dispatch|send|produce)\s*\(\s*["\']([^"\']+)', "publish"),
        (r'\.(?:subscribe|on|listen|consume|addListener)\s*\(\s*["\']([^"\']+)', "subscribe"),
        (r'webhook[_\-]?\w*\s*[:=].*["\']([^"\']+)', "webhook"),
        (r'(?:topic|channel|queue|exchange)\s*[:=]\s*["\']([^"\']+)', "channel"),
    ]
    results = []
    for fp, rel in _walk_files(repo_path):
        if fp.suffix not in (".py", ".ts", ".js", ".go", ".rs", ".java", ".rb"):
            continue
        content = _read_file(fp, max_lines=1000)
        if not content:
            continue
        file_events = []
        for pat, kind in patterns:
            for m in re.finditer(pat, content):
                file_events.append({"kind": kind, "name": m.group(1), "match": m.group(0).strip()})
        if file_events:
            results.append({"file": rel, "events": file_events})
    return results


def extract_db_interfaces(repo_path):
    """Extract shared DB tables from migrations, SQL files, and ORM models."""
    root = Path(repo_path)
    tables = []
    sql_pattern = re.compile(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\']?(\w+)', re.IGNORECASE)
    orm_patterns = [
        re.compile(r'class\s+(\w+)\s*\(.*(?:Model|Base|DeclarativeBase|db\.Model)'),
        re.compile(r'@Entity\s*\(\s*["\']?(\w+)'),
        re.compile(r'__tablename__\s*=\s*["\'](\w+)'),
    ]

    for fp, rel in _walk_files(repo_path):
        content = None
        if fp.suffix in (".sql",):
            content = _read_file(fp, max_lines=2000)
            if content:
                for m in sql_pattern.finditer(content):
                    tables.append({"file": rel, "table": m.group(1), "source": "sql"})
        elif fp.suffix in (".py", ".ts", ".js", ".java", ".rb"):
            content = _read_file(fp, max_lines=1000)
            if content:
                for pat in orm_patterns:
                    for m in pat.finditer(content):
                        tables.append({"file": rel, "table": m.group(1), "source": "orm"})
    return tables


def extract_cpp_headers(repo_path):
    """Extract public C/C++ header class and function declarations."""
    root = Path(repo_path)
    results = []
    class_pat = re.compile(r'(?:class|struct)\s+(?:\w+\s+)?(\w+)\s*(?::\s*(?:public|private|protected)\s+\w+)?\s*\{')
    func_pat = re.compile(r'^(?:extern\s+)?(?:[\w:*&<>]+\s+)+(\w+)\s*\([^)]*\)\s*;', re.MULTILINE)

    include_dirs = ["include", "public", "api"]
    for p in sorted(root.rglob("*.h")) + sorted(root.rglob("*.hpp")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        rel_parts = p.relative_to(root).parts
        is_public = any(d in rel_parts for d in include_dirs) or len(rel_parts) <= 2
        if not is_public:
            continue
        content = _read_file(p, max_lines=500)
        if not content:
            continue
        rel = str(p.relative_to(root))
        classes = class_pat.findall(content)
        funcs = func_pat.findall(content)
        if classes or funcs:
            results.append({"file": rel, "classes": classes[:20], "functions": funcs[:20]})
    return results


def extract_ros2(repo_path):
    """Extract ROS2 interfaces: .msg/.srv/.action definitions, launch files, package.xml."""
    root = Path(repo_path)
    results = {"messages": [], "services": [], "actions": [], "launch_files": [], "packages": []}

    for p in sorted(root.rglob("*.msg")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=200)
        if content:
            results["messages"].append({"file": str(p.relative_to(root)), "content": content.strip()})

    for p in sorted(root.rglob("*.srv")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=200)
        if content:
            results["services"].append({"file": str(p.relative_to(root)), "content": content.strip()})

    for p in sorted(root.rglob("*.action")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=200)
        if content:
            results["actions"].append({"file": str(p.relative_to(root)), "content": content.strip()})

    for pattern in ["**/launch/*.py", "**/launch/*.xml", "**/launch/*.yaml", "**/*.launch.py", "**/*.launch.xml"]:
        for p in sorted(root.glob(pattern)):
            if any(skip in p.parts for skip in SKIP_DIRS):
                continue
            content = _read_file(p, max_lines=500)
            if content:
                results["launch_files"].append({"file": str(p.relative_to(root)), "content": content})

    for p in sorted(root.rglob("package.xml")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=200)
        if content:
            deps = re.findall(r'<(?:depend|exec_depend|build_depend|build_export_depend)>(\w+)</', content)
            pkg_name = re.search(r'<name>(\w+)</name>', content)
            results["packages"].append({
                "file": str(p.relative_to(root)),
                "name": pkg_name.group(1) if pkg_name else p.parent.name,
                "deps": deps,
            })

    has_content = any(v for v in results.values())
    return results if has_content else {}


def extract_dbc(repo_path):
    """Extract CAN DBC signal definitions."""
    root = Path(repo_path)
    results = []
    msg_pat = re.compile(r'BO_\s+(\d+)\s+(\w+)\s*:\s*(\d+)\s+(\w+)')
    sig_pat = re.compile(r'SG_\s+(\w+)\s*:\s*(\d+)\|(\d+)@([01])([+-])\s*\(([^,]+),([^)]+)\)\s*\[([^\]]*)\]')

    for p in sorted(root.rglob("*.dbc")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=5000)
        if not content:
            continue
        rel = str(p.relative_to(root))
        messages = []
        for m in msg_pat.finditer(content):
            msg_id = int(m.group(1))
            msg_name = m.group(2)
            msg_len = int(m.group(3))
            sender = m.group(4)
            signals = []
            msg_end = content.find("BO_", m.end())
            if msg_end == -1:
                msg_end = len(content)
            msg_block = content[m.end():msg_end]
            for s in sig_pat.finditer(msg_block):
                signals.append({
                    "name": s.group(1), "start_bit": int(s.group(2)),
                    "length": int(s.group(3)), "byte_order": "big" if s.group(4) == "0" else "little",
                    "signed": s.group(5) == "-",
                    "factor": s.group(6).strip(), "offset": s.group(7).strip(),
                    "range": s.group(8).strip(),
                })
            messages.append({"id": msg_id, "name": msg_name, "length": msg_len,
                           "sender": sender, "signals": signals})
        if messages:
            results.append({"file": rel, "messages": messages})
    return results


def extract_yocto(repo_path):
    """Extract Yocto/BitBake recipes, configs, and layer structure."""
    root = Path(repo_path)
    results = {"recipes": [], "configs": [], "layers": [], "machines": []}

    for p in sorted(root.rglob("*.bb")) + sorted(root.rglob("*.bbappend")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=500)
        if not content:
            continue
        rel = str(p.relative_to(root))
        recipe = {"file": rel}
        for field in ["DEPENDS", "RDEPENDS", "SRC_URI", "inherit", "PACKAGECONFIG", "LICENSE"]:
            matches = re.findall(rf'^{field}\s*(?:\+?=|:=)\s*"([^"]*)"', content, re.MULTILINE)
            if not matches:
                matches = re.findall(rf'^{field}\s+(.+)$', content, re.MULTILINE)
            if matches:
                recipe[field.lower()] = " ".join(matches).strip()
        if len(recipe) > 1:
            results["recipes"].append(recipe)

    for name in ["local.conf", "bblayers.conf"]:
        for p in sorted(root.rglob(name)):
            if any(skip in p.parts for skip in SKIP_DIRS):
                continue
            content = _read_file(p, max_lines=300)
            if content:
                results["configs"].append({"file": str(p.relative_to(root)), "content": content})

    for p in sorted(root.rglob("layer.conf")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=100)
        if content:
            layer_name = re.search(r'BBFILE_COLLECTIONS\s*\+?=\s*"(\w+)"', content)
            results["layers"].append({
                "file": str(p.relative_to(root)),
                "name": layer_name.group(1) if layer_name else p.parent.name,
            })

    for p in sorted(root.rglob("*.conf")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        if "machine" not in str(p.relative_to(root)).lower():
            continue
        content = _read_file(p, max_lines=200)
        if content and re.search(r'MACHINE_FEATURES|PREFERRED_PROVIDER|KERNEL_DEVICETREE', content):
            results["machines"].append({"file": str(p.relative_to(root)), "content": content})

    has_content = any(v for v in results.values())
    return results if has_content else {}


def extract_device_tree(repo_path):
    """Extract device tree source files — nodes, compatible strings, peripherals."""
    root = Path(repo_path)
    results = []
    compat_pat = re.compile(r'compatible\s*=\s*"([^"]+)"')

    for p in sorted(root.rglob("*.dts")) + sorted(root.rglob("*.dtsi")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        content = _read_file(p, max_lines=1000)
        if not content:
            continue
        rel = str(p.relative_to(root))
        compatibles = compat_pat.findall(content)
        includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', content)
        node_names = re.findall(r'(\w[\w,-]+)\s*(?:@[\da-fA-F]+)?\s*\{', content)
        results.append({
            "file": rel,
            "compatibles": list(set(compatibles)),
            "includes": includes,
            "nodes": node_names[:30],
        })
    return results


def extract_docker_ci(repo_path):
    """Extract Dockerfile, docker-compose, and CI pipeline configs."""
    root = Path(repo_path)
    results = []

    ci_patterns = [
        "Dockerfile", "Dockerfile.*", "docker-compose.yml", "docker-compose.yaml",
        ".github/workflows/*.yml", ".github/workflows/*.yaml",
        ".gitlab-ci.yml", "Jenkinsfile", ".circleci/config.yml",
        "bitbucket-pipelines.yml",
    ]
    for pattern in ci_patterns:
        for p in sorted(root.glob(pattern)):
            content = _read_file(p, max_lines=500)
            if content:
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                results.append({"file": str(p.relative_to(root)), "content": content})

    return results


def run_extraction(repo_path, project=None, verbose=False):
    root = Path(repo_path).resolve()
    if not root.is_dir():
        print(f"Error: {repo_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Extracting from {root}...", file=sys.stderr)

    git_info = extract_git_info(root)

    source_ref = {
        "path": str(root),
        "remote": git_info.get("remote"),
        "commit": git_info.get("commit"),
        "local_only": git_info.get("local_only", True),
    }

    if project is None:
        project = root.name

    extractors = [
        ("docs", lambda: extract_docs(root)),
        ("dependencies", lambda: extract_dependencies(root)),
        ("tree", lambda: extract_tree(root)),
        ("config", lambda: extract_config(root)),
        ("schemas", lambda: extract_schemas(root)),
        ("entrypoints", lambda: extract_entrypoints(root)),
        ("interfaces", lambda: extract_routes(root)),
        ("exports", lambda: extract_exports(root)),
        ("comments", lambda: extract_comments(root)),
        ("todos", lambda: extract_todos(root)),
        ("env_vars", lambda: extract_env_vars(root)),
        ("protobuf", lambda: extract_protobuf(root)),
        ("cmake_flags", lambda: extract_cmake_flags(root)),
        ("event_interfaces", lambda: extract_event_interfaces(root)),
        ("db_interfaces", lambda: extract_db_interfaces(root)),
        ("cpp_headers", lambda: extract_cpp_headers(root)),
        ("ros2", lambda: extract_ros2(root)),
        ("dbc", lambda: extract_dbc(root)),
        ("yocto", lambda: extract_yocto(root)),
        ("device_tree", lambda: extract_device_tree(root)),
        ("docker_ci", lambda: extract_docker_ci(root)),
        ("git_log", lambda: extract_git_log(root)),
    ]

    extractions = {}
    for name, fn in extractors:
        if verbose:
            print(f"  extracting {name}...", file=sys.stderr)
        try:
            extractions[name] = fn()
        except Exception as e:
            extractions[name] = {"error": str(e)}
            if verbose:
                print(f"    error: {e}", file=sys.stderr)

    return {
        "repo": source_ref,
        "project": project,
        "git": git_info,
        "extractions": extractions,
    }


def print_summary(result):
    """Print a human-readable summary to stderr."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Repo: {result['repo']['path']}", file=sys.stderr)
    print(f"Project: {result['project']}", file=sys.stderr)
    if result['git'].get('remote'):
        print(f"Remote: {result['git']['remote']}", file=sys.stderr)
    if result['git'].get('commit'):
        print(f"Commit: {result['git']['commit'][:12]}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    ex = result["extractions"]

    def _count(val):
        if isinstance(val, list):
            return len(val)
        if isinstance(val, dict):
            if "error" in val:
                return f"error: {val['error']}"
            return sum(len(v) if isinstance(v, (list, dict)) else 1 for v in val.values())
        return 0

    rows = [
        ("Docs", _count(ex.get("docs"))),
        ("Dependency files", _count(ex.get("dependencies"))),
        ("Tree entries", _count(ex.get("tree"))),
        ("Config files", _count(ex.get("config"))),
        ("Schema files", _count(ex.get("schemas"))),
        ("Entrypoints", _count(ex.get("entrypoints"))),
        ("HTTP routes", _count(ex.get("interfaces", {}).get("http_routes", []))),
        ("CLI interfaces", _count(ex.get("interfaces", {}).get("cli_interfaces", []))),
        ("Export files", _count(ex.get("exports"))),
        ("Files with signal comments", _count(ex.get("comments"))),
        ("Files with TODOs", _count(ex.get("todos"))),
        ("Env vars referenced", _count(ex.get("env_vars"))),
        ("Protobuf files", _count(ex.get("protobuf"))),
        ("CMake flags", _count(ex.get("cmake_flags"))),
        ("Event interfaces", _count(ex.get("event_interfaces"))),
        ("DB tables", _count(ex.get("db_interfaces"))),
        ("C/C++ public headers", _count(ex.get("cpp_headers"))),
        ("ROS2 interfaces", sum(len(v) for v in ex.get("ros2", {}).values() if isinstance(v, list))),
        ("DBC files", _count(ex.get("dbc"))),
        ("Yocto recipes/configs", sum(len(v) for v in ex.get("yocto", {}).values() if isinstance(v, list))),
        ("Device tree files", _count(ex.get("device_tree"))),
        ("Docker/CI configs", _count(ex.get("docker_ci"))),
        ("Git log entries", _count(ex.get("git_log"))),
    ]

    for label, count in rows:
        print(f"  {label:<30} {count}", file=sys.stderr)

    submodules = result.get("git", {}).get("submodules", [])
    if submodules:
        print(f"  {'Submodules':<30} {len(submodules)} ({', '.join(submodules)})", file=sys.stderr)
    print(file=sys.stderr)


DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")

DISTILL_PROMPT = """\
You are distilling structured repository extracts into Cairn memory entries.
Each entry must be a self-contained one-liner useful to a future AI session with zero context about this conversation.

Project: {project}
Repository: {remote_or_path}
Commit: {commit}

Below are categorised extracts from the repository. Distill them into memory entries.

Rules:
- Each entry is a single content string — no line breaks within it, but be thorough and specific
- Include concrete details: function names, file paths, config keys, exact commands, parameter names
- A future AI session will use these entries to WORK ON this codebase — vague summaries are useless
- Types allowed: fact, workflow, skill (NO decision type — decisions require human context)
- Every entry must include enough context to be useful standalone (include the project name)
- Prioritise: tech stack, build/test/deploy commands, entry points, external interfaces, cross-system connections, gotchas, conventions
- Skip: trivial facts, things obvious from the project name
- Include source_files: list the relative file paths each entry was derived from
- Aim for 20-50 entries depending on repo complexity — thoroughness over brevity
- DO NOT fabricate — only distill what is present in the extracts

Output format — one JSON array of objects:
[
  {{"type": "fact", "topic": "short-topic-slug", "content": "detailed actionable content", "keywords": ["kw1", "kw2"], "source_files": ["relative/path.py"]}},
  ...
]

Reply with ONLY the JSON array. No commentary, no markdown fences, no explanation.

=== EXTRACTS ===
{extracts}
"""


def _prepare_extracts_text(result):
    """Prepare a condensed text representation of extractions for the LLM prompt."""
    ex = result["extractions"]
    sections = []

    if ex.get("docs"):
        for doc in ex["docs"]:
            content = doc["content"]
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            sections.append(f"### Doc: {doc['file']}\n{content}")

    if ex.get("dependencies"):
        for name, content in ex["dependencies"].items():
            if isinstance(content, dict):
                content = json.dumps(content, indent=2)
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            sections.append(f"### Dependencies: {name}\n{content}")

    if ex.get("tree"):
        tree = ex["tree"]
        tree_text = "\n".join(tree[:200])
        if len(tree) > 200:
            tree_text += f"\n... ({len(tree) - 200} more entries)"
        sections.append(f"### Directory tree ({len(tree)} entries)\n{tree_text}")

    if ex.get("config"):
        for cfg in ex["config"]:
            content = cfg["content"]
            if len(content) > 1500:
                content = content[:1500] + "\n... (truncated)"
            sections.append(f"### Config: {cfg['file']}\n{content}")

    if ex.get("schemas"):
        for schema in ex["schemas"]:
            content = schema["content"]
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            sections.append(f"### Schema: {schema['file']}\n{content}")

    if ex.get("entrypoints"):
        for entry in ex["entrypoints"]:
            content = entry["content"]
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            sections.append(f"### Entrypoint: {entry['file']}\n{content}")

    interfaces = ex.get("interfaces", {})
    if interfaces.get("http_routes"):
        lines = []
        for rf in interfaces["http_routes"]:
            for r in rf["routes"]:
                lines.append(f"  {rf['file']}: {r['match']}")
        sections.append(f"### HTTP routes\n" + "\n".join(lines))

    if interfaces.get("cli_interfaces"):
        lines = []
        for cf in interfaces["cli_interfaces"]:
            lines.append(f"  {cf['file']}:")
            for a in cf["args"]:
                lines.append(f"    {a['match']}")
        sections.append(f"### CLI interfaces\n" + "\n".join(lines))

    if ex.get("exports"):
        for exp in ex["exports"][:10]:
            content = exp["content"]
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            sections.append(f"### Exports: {exp['file']}\n{content}")

    if ex.get("comments"):
        lines = []
        for cf in ex["comments"]:
            for c in cf["comments"]:
                lines.append(f"  {cf['file']}:{c['line']}: {c['text']}")
        if len(lines) > 100:
            lines = lines[:100] + [f"... ({len(lines) - 100} more)"]
        sections.append(f"### Signal comments\n" + "\n".join(lines))

    if ex.get("todos"):
        lines = []
        for tf in ex["todos"]:
            for t in tf["todos"]:
                lines.append(f"  {tf['file']}:{t['line']}: {t['text']}")
        if len(lines) > 50:
            lines = lines[:50] + [f"... ({len(lines) - 50} more)"]
        sections.append(f"### TODOs\n" + "\n".join(lines))

    if ex.get("env_vars"):
        lines = [f"  {var} (in {', '.join(files)})" for var, files in ex["env_vars"].items()]
        sections.append(f"### Environment variables\n" + "\n".join(lines))

    if ex.get("protobuf"):
        lines = []
        for pf in ex["protobuf"]:
            lines.append(f"  {pf['file']}:")
            if pf["services"]:
                lines.append(f"    services: {', '.join(pf['services'])}")
            for rpc in pf.get("rpcs", []):
                lines.append(f"    rpc {rpc['service']}.{rpc['method']}({rpc['request']}) -> {rpc['response']}")
            if pf["messages"]:
                msgs = pf["messages"]
                if len(msgs) > 20:
                    msgs = msgs[:20] + [f"... ({len(pf['messages']) - 20} more)"]
                lines.append(f"    messages: {', '.join(msgs)}")
        sections.append(f"### Protobuf definitions\n" + "\n".join(lines))

    if ex.get("cmake_flags"):
        lines = []
        for f in ex["cmake_flags"]:
            desc = f" — {f['desc']}" if f["desc"] else ""
            lines.append(f"  {f['flag']} (default: {f['default']}){desc}  [{f['file']}]")
        if len(lines) > 50:
            lines = lines[:50] + [f"... ({len(lines) - 50} more)"]
        sections.append(f"### CMake build flags\n" + "\n".join(lines))

    if ex.get("event_interfaces"):
        lines = []
        for ef in ex["event_interfaces"]:
            for e in ef["events"]:
                lines.append(f"  {ef['file']}: {e['kind']} '{e['name']}'")
        if len(lines) > 50:
            lines = lines[:50] + [f"... ({len(lines) - 50} more)"]
        sections.append(f"### Event interfaces (pub/sub, webhooks, queues)\n" + "\n".join(lines))

    if ex.get("db_interfaces"):
        lines = [f"  {t['table']} ({t['source']}) [{t['file']}]" for t in ex["db_interfaces"]]
        if len(lines) > 50:
            lines = lines[:50] + [f"... ({len(lines) - 50} more)"]
        sections.append(f"### Database tables\n" + "\n".join(lines))

    if ex.get("cpp_headers"):
        lines = []
        for hf in ex["cpp_headers"]:
            parts = []
            if hf["classes"]:
                parts.append(f"classes: {', '.join(hf['classes'][:10])}")
            if hf["functions"]:
                parts.append(f"functions: {', '.join(hf['functions'][:10])}")
            if parts:
                lines.append(f"  {hf['file']}: {'; '.join(parts)}")
        if len(lines) > 50:
            lines = lines[:50] + [f"... ({len(lines) - 50} more)"]
        sections.append(f"### C/C++ public headers\n" + "\n".join(lines))

    ros2 = ex.get("ros2", {})
    if ros2:
        lines = []
        for msg in ros2.get("messages", []):
            lines.append(f"  MSG {msg['file']}:\n    {msg['content'][:200]}")
        for srv in ros2.get("services", []):
            lines.append(f"  SRV {srv['file']}:\n    {srv['content'][:200]}")
        for act in ros2.get("actions", []):
            lines.append(f"  ACTION {act['file']}:\n    {act['content'][:200]}")
        for pkg in ros2.get("packages", []):
            lines.append(f"  package: {pkg['name']} deps: {', '.join(pkg['deps'][:15])}")
        for lf in ros2.get("launch_files", [])[:10]:
            content = lf["content"]
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            lines.append(f"  launch: {lf['file']}\n    {content}")
        if lines:
            sections.append(f"### ROS2 interfaces\n" + "\n".join(lines))

    if ex.get("dbc"):
        lines = []
        for df in ex["dbc"]:
            lines.append(f"  {df['file']}:")
            for msg in df["messages"][:20]:
                sig_names = [s["name"] for s in msg["signals"][:8]]
                lines.append(f"    0x{msg['id']:X} {msg['name']} ({msg['length']}B, {msg['sender']}): {', '.join(sig_names)}")
        if len(lines) > 80:
            lines = lines[:80] + [f"... ({len(lines) - 80} more)"]
        sections.append(f"### CAN DBC definitions\n" + "\n".join(lines))

    yocto = ex.get("yocto", {})
    if yocto:
        lines = []
        for recipe in yocto.get("recipes", [])[:30]:
            parts = [f"{k}={v[:80]}" for k, v in recipe.items() if k != "file"]
            lines.append(f"  {recipe['file']}: {'; '.join(parts)}")
        for cfg in yocto.get("configs", []):
            content = cfg["content"]
            if len(content) > 1500:
                content = content[:1500] + "\n... (truncated)"
            lines.append(f"  config: {cfg['file']}\n    {content}")
        for layer in yocto.get("layers", []):
            lines.append(f"  layer: {layer['name']} [{layer['file']}]")
        for machine in yocto.get("machines", [])[:5]:
            content = machine["content"]
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            lines.append(f"  machine: {machine['file']}\n    {content}")
        if lines:
            sections.append(f"### Yocto/BitBake\n" + "\n".join(lines))

    if ex.get("device_tree"):
        lines = []
        for dt in ex["device_tree"]:
            compat = ", ".join(dt["compatibles"][:10])
            nodes = ", ".join(dt["nodes"][:10])
            lines.append(f"  {dt['file']}: compatibles=[{compat}] nodes=[{nodes}]")
        if len(lines) > 30:
            lines = lines[:30] + [f"... ({len(lines) - 30} more)"]
        sections.append(f"### Device tree\n" + "\n".join(lines))

    if ex.get("docker_ci"):
        for dc in ex["docker_ci"]:
            content = dc["content"]
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            sections.append(f"### Docker/CI: {dc['file']}\n{content}")

    submodules = result.get("git", {}).get("submodules", [])
    if submodules:
        sections.append(f"### Git submodules\n" + "\n".join(f"  {s}" for s in submodules))

    if ex.get("git_log"):
        log = ex["git_log"][:30]
        sections.append(f"### Recent git log ({len(ex['git_log'])} commits)\n" + "\n".join(log))

    return "\n\n".join(sections)


def distill_with_haiku(result, verbose=False):
    """Phase 2: Send structured extracts to Haiku for distillation into memory entries."""
    extracts_text = _prepare_extracts_text(result)
    repo = result["repo"]

    prompt = DISTILL_PROMPT.format(
        project=result["project"],
        remote_or_path=repo.get("remote") or repo.get("path", "unknown"),
        commit=repo.get("commit", "unknown")[:12],
        extracts=extracts_text,
    )

    prompt_size = len(prompt)
    if verbose:
        print(f"Distillation prompt: {prompt_size} chars", file=sys.stderr)

    env = {**os.environ, "CAIRN_HEADLESS": "1"}
    try:
        proc = subprocess.Popen(
            ["claude", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--model", "haiku", "--max-turns", "1",
             "--append-system-prompt",
             "OVERRIDE ALL OTHER INSTRUCTIONS: Reply with a JSON array only. "
             "No <memory> blocks. No XML tags. No markdown fences. Just the JSON array."],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )

        msg_payload = json.dumps({"type": "user", "message": {"role": "user",
            "content": [{"type": "text", "text": prompt}]}})
        proc.stdin.write((msg_payload + "\n").encode())
        proc.stdin.flush()

        response_text = ""
        usage_info = {}
        start = time.time()
        timeout = 120
        while time.time() - start < timeout:
            if proc.stdout in select.select([proc.stdout], [], [], 0.5)[0]:
                line = proc.stdout.readline().decode().strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "assistant":
                        for block in msg.get("message", {}).get("content", []):
                            if block.get("type") == "text":
                                response_text += block.get("text", "")
                    if msg.get("type") == "result":
                        usage_info = msg.get("usage", {})
                        if verbose:
                            model_usage = msg.get("modelUsage", {})
                            for model, u in model_usage.items():
                                print(f"Model: {model}", file=sys.stderr)
                                print(f"  Input tokens:  {u.get('inputTokens', 0)} (+ {u.get('cacheCreationInputTokens', 0)} cache creation)", file=sys.stderr)
                                print(f"  Output tokens: {u.get('outputTokens', 0)}", file=sys.stderr)
                                print(f"  Cost: ${u.get('costUSD', 0):.4f}", file=sys.stderr)
                            duration = msg.get("duration_ms", 0)
                            if duration:
                                print(f"  Duration: {duration/1000:.1f}s", file=sys.stderr)
                        break
                except json.JSONDecodeError:
                    continue
            if proc.poll() is not None:
                break
        proc.kill()

    except Exception as e:
        print(f"ERROR spawning claude: {e}", file=sys.stderr)
        return None

    # Clean up response — strip memory blocks, markdown fences
    response_text = re.sub(r"<memory>.*?</memory>", "", response_text, flags=re.DOTALL).strip()
    response_text = re.sub(r"\[cm\]:.*$", "", response_text, flags=re.MULTILINE).strip()
    response_text = re.sub(r"^```json\s*", "", response_text).strip()
    response_text = re.sub(r"\s*```$", "", response_text).strip()

    try:
        entries = json.loads(response_text)
        if not isinstance(entries, list):
            print(f"ERROR: Expected JSON array, got {type(entries).__name__}", file=sys.stderr)
            return None
        return entries
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse Haiku response as JSON: {e}", file=sys.stderr)
        if verbose:
            print(f"Raw response:\n{response_text[:500]}", file=sys.stderr)
        return None


def insert_memories(entries, project, source_ref, session_id=None, dry_run=False):
    """Insert distilled memory entries into the Cairn database."""
    if session_id is None:
        session_id = f"ingest-{project}-{time.strftime('%Y%m%d-%H%M%S')}"

    src_ref_json = json.dumps({
        "repo": source_ref.get("remote") or source_ref.get("path"),
        "commit": source_ref.get("commit"),
        "local": source_ref.get("local_only", True),
        "path": source_ref.get("path"),
        "parent_project": source_ref.get("parent_project"),
        "parent_path": source_ref.get("parent_path"),
    })

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {len(entries)} memories would be inserted")
        print(f"Project: {project}")
        print(f"Session: {session_id}")
        print(f"Source ref: {src_ref_json}")
        print(f"{'='*60}")
        for i, entry in enumerate(entries, 1):
            print(f"  [{i}] {entry.get('type', '?')}/{entry.get('topic', '?')}")
            print(f"      {entry.get('content', '?')}")
            kw = entry.get("keywords", [])
            if kw:
                print(f"      keywords: {', '.join(kw)}")
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    inserted_ids = []

    # Register synthetic session so dashboard can find it
    repo_path = source_ref.get("path", "")
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, project, transcript_path, started_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (session_id, project, repo_path),
    )

    # Archive previous ingestion for this project (safe re-ingestion)
    prev_ingested = conn.execute(
        "SELECT id FROM memories WHERE project = ? AND session_id LIKE 'ingest-%' "
        "AND session_id != ? AND (archived_reason IS NULL OR archived_reason = '')",
        (project, session_id),
    ).fetchall()
    if prev_ingested:
        archived_reason = f"re-ingested:{session_id}"
        conn.execute(
            "UPDATE memories SET archived_reason = ?, confidence = 0, updated_at = CURRENT_TIMESTAMP "
            "WHERE project = ? AND session_id LIKE 'ingest-%' AND session_id != ? "
            "AND (archived_reason IS NULL OR archived_reason = '')",
            (archived_reason, project, session_id),
        )
        print(f"Archived {len(prev_ingested)} previous ingestion memories for {project}", file=sys.stderr)

    try:
        from cairn import embeddings as emb
        emb._load_vec(conn)
    except Exception:
        emb = None

    for entry in entries:
        mem_type = entry.get("type", "fact")
        topic = entry.get("topic", "unknown")
        content = entry.get("content", "")
        keywords = entry.get("keywords", [])

        if not content or len(content) < 20:
            continue

        embedding_blob = None
        if emb:
            try:
                search_text = f"{project} {mem_type} {topic} {content}"
                vec = emb.embed(search_text)
                if vec is not None:
                    embedding_blob = emb.to_blob(vec)
            except Exception:
                pass

        origin_id = str(uuid.uuid4())
        kw_str = ",".join(keywords) if keywords else None
        source_files = entry.get("source_files", [])
        assoc_files = json.dumps(source_files) if source_files else None

        conn.execute(
            "INSERT INTO memories (type, topic, content, embedding, session_id, project, "
            "origin_id, source_ref, keywords, depth, associated_files) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mem_type, topic, content, embedding_blob, session_id, project,
             origin_id, src_ref_json, kw_str, 0, assoc_files),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        if embedding_blob and emb:
            try:
                emb.upsert_vec_index(conn, new_id, embedding_blob)
            except Exception:
                pass

        inserted_ids.append(new_id)

    conn.commit()
    conn.close()
    return inserted_ids


def main():
    parser = argparse.ArgumentParser(description="Cairn repo ingestion — mechanistic extraction + LLM distillation")
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("--project", help="Project name (defaults to directory name)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without inserting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress on stderr")
    parser.add_argument("--output", "-o", help="Write Phase 1 JSON to file")
    parser.add_argument("--section", help="Extract only this section (e.g. docs, routes, comments)")
    parser.add_argument("--phase1-only", action="store_true", help="Run Phase 1 only, output JSON (skip distillation)")
    parser.add_argument("--save-entries", help="Save distilled entries to JSON file (reusable with --load-entries)")
    parser.add_argument("--load-entries", help="Insert entries from a previously saved JSON file (skips Phase 2)")
    parser.add_argument("--recurse-submodules", action="store_true", help="Also ingest each submodule as its own project")
    args = parser.parse_args()

    # Load-entries shortcut: skip Phase 1 + 2, just insert from file
    if args.load_entries:
        entries_data = json.loads(Path(args.load_entries).read_text())
        entries = entries_data.get("entries", entries_data)
        if isinstance(entries, dict):
            entries = entries.get("entries", [])
        project = args.project or entries_data.get("project", Path(args.repo_path).name)
        source_ref = entries_data.get("source_ref", {"path": str(Path(args.repo_path).resolve())})
        print(f"Loading {len(entries)} entries from {args.load_entries}", file=sys.stderr)
        inserted = insert_memories(entries, project=project, source_ref=source_ref, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"Inserted {len(inserted)} memories (IDs: {inserted[0]}–{inserted[-1]})", file=sys.stderr)
        return

    # Phase 1: Extract
    result = run_extraction(args.repo_path, project=args.project, verbose=args.verbose)
    print_summary(result)

    if args.section:
        section = result["extractions"].get(args.section)
        if section is None:
            print(f"Unknown section: {args.section}", file=sys.stderr)
            print(f"Available: {', '.join(result['extractions'].keys())}", file=sys.stderr)
            sys.exit(1)
        output = {"repo": result["repo"], "project": result["project"], "section": args.section, "data": section}
        print(json.dumps(output, indent=2, default=str))
        return

    if args.output:
        json_str = json.dumps(result, indent=2, default=str)
        Path(args.output).write_text(json_str)
        print(f"Phase 1 written to {args.output}", file=sys.stderr)

    if args.phase1_only:
        if not args.output:
            print(json.dumps(result, indent=2, default=str))
        return

    # Phase 2: Distill
    print("\nPhase 2: Distilling with Haiku...", file=sys.stderr)
    entries = distill_with_haiku(result, verbose=args.verbose)
    if entries is None:
        print("Distillation failed.", file=sys.stderr)
        sys.exit(1)

    print(f"Haiku produced {len(entries)} memory entries", file=sys.stderr)

    # Always save distillation output — Haiku calls are expensive
    save_data = {
        "project": result["project"],
        "source_ref": result["repo"],
        "entries": entries,
    }
    save_path = args.save_entries or os.path.join(
        "/tmp", f"cairn-ingest-{result['project']}-{int(time.time())}.json"
    )
    Path(save_path).write_text(json.dumps(save_data, indent=2))
    print(f"Entries saved to {save_path}", file=sys.stderr)

    # Insert or dry-run
    inserted = insert_memories(
        entries,
        project=result["project"],
        source_ref=result["repo"],
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\nInserted {len(inserted)} memories (IDs: {inserted[0]}–{inserted[-1]})", file=sys.stderr)

    # Recurse into submodules if requested
    if args.recurse_submodules:
        submodules = result.get("git", {}).get("submodules", [])
        root = Path(args.repo_path).resolve()
        for sub_path in submodules:
            sub_full = root / sub_path
            if not sub_full.is_dir() or not (sub_full / ".git").exists():
                print(f"\nSkipping submodule {sub_path} (not initialized)", file=sys.stderr)
                continue
            sub_project = sub_full.name
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Ingesting submodule: {sub_path} (project: {sub_project})", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            sub_result = run_extraction(str(sub_full), project=sub_project, verbose=args.verbose)
            print_summary(sub_result)
            if not args.dry_run:
                sub_entries = distill_with_haiku(sub_result, verbose=args.verbose)
                if sub_entries:
                    print(f"Haiku produced {len(sub_entries)} entries for {sub_project}", file=sys.stderr)
                    sub_save = {
                        "project": sub_project,
                        "source_ref": sub_result["repo"],
                        "entries": sub_entries,
                    }
                    sub_save_path = os.path.join(
                        "/tmp", f"cairn-ingest-{sub_project}-{int(time.time())}.json"
                    )
                    Path(sub_save_path).write_text(json.dumps(sub_save, indent=2))
                    print(f"Entries saved to {sub_save_path}", file=sys.stderr)
                    sub_source_ref = dict(sub_result["repo"])
                    sub_source_ref["parent_project"] = result["project"]
                    sub_source_ref["parent_path"] = sub_path
                    sub_inserted = insert_memories(
                        sub_entries, project=sub_project,
                        source_ref=sub_source_ref, dry_run=args.dry_run,
                    )
                    if not args.dry_run and sub_inserted:
                        print(f"Inserted {len(sub_inserted)} memories for {sub_project}", file=sys.stderr)


if __name__ == "__main__":
    main()
