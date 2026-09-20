---
name: cairn-ingest
description: Cairn repo ingestion — the 24 extractors, ingest.py/ingest_transcript.py/backfill_ingestion.py, incremental diffing, and the dependency-graph edge query.
---

## Repo Ingestion

Ingest a git repository into Cairn as portable knowledge entries:

- `.venv/bin/python ./cairn/ingest.py /path/to/repo` — extract, distill, and store (incremental if previously ingested)
- `.venv/bin/python ./cairn/ingest.py /path/to/repo --dry-run` — preview without storing
- `.venv/bin/python ./cairn/ingest.py /path/to/repo --project name` — override project name
- `.venv/bin/python ./cairn/ingest.py /path/to/repo --full` — force full re-ingestion (skip incremental diff)
- `.venv/bin/python ./cairn/ingest.py /path/to/repo --verbose` — show extraction details

24 extractors: docs, deps, tree, config, schemas, entrypoints, HTTP routes, CLI args, exports, comments, TODOs, env vars, protobuf, CMake flags, event interfaces, DB tables, C/C++ headers, ROS2 interfaces, CAN DBC, Yocto/BitBake, device tree, Docker/CI, tree-sitter AST (Python, JS, TS, TSX, Go, Rust, C, C++), dependency graph. Graph edges queryable via `.venv/bin/python ./cairn/query.py --deps <project>`.
