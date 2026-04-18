# Cairn — Persistent Memory System

This is the Cairn project — a persistent AI memory system using SQLite, Claude Code hooks, and structured self-assessment.

## Cairn Database

The memory database is at `./cairn/cairn.db`. Use `python3 ./cairn/query.py` to search it.

Query commands:
- `python3 ./cairn/query.py <search>` — full-text search
- `python3 ./cairn/query.py --semantic <query>` — semantic similarity search
- `python3 ./cairn/query.py --recent` — list recent memories
- `python3 ./cairn/query.py --today` — memories from today
- `python3 ./cairn/query.py --since <date>` — memories from date onward (ISO, today, yesterday, 3d, 2w, 1m)
- `python3 ./cairn/query.py --since <date> --until <date>` — memories in a date range
- `python3 ./cairn/query.py --type <type>` — filter by type
- `python3 ./cairn/query.py --session <id>` — filter by session
- `python3 ./cairn/query.py --chain <id>` — show session chain
- `python3 ./cairn/query.py --project <name>` — list memories for a project
- `python3 ./cairn/query.py --projects` — list all projects
- `python3 ./cairn/query.py --label <session_id> <name>` — label a session chain
- `python3 ./cairn/query.py --context <id>` — show conversation context for a memory
- `python3 ./cairn/query.py --history <id>` — show version history
- `python3 ./cairn/query.py --delete <id>` — delete a memory
- `python3 ./cairn/query.py --compact [project]` — dense cairn dump for LLM ingestion
- `python3 ./cairn/query.py --review` — surface low-confidence memories
- `python3 ./cairn/query.py --verify-sources` — analyse source_messages accuracy
- `python3 ./cairn/query.py --backfill` — generate missing embeddings
- `python3 ./cairn/query.py --stats` — database statistics

## Repo Ingestion

Ingest a git repository into Cairn as portable knowledge entries:

- `python3 ./cairn/ingest.py /path/to/repo` — dry-run (extract + distill, no DB write)
- `python3 ./cairn/ingest.py /path/to/repo --distill` — extract, distill, and store
- `python3 ./cairn/ingest.py /path/to/repo --project name` — override project name
- `python3 ./cairn/ingest.py /path/to/repo --verbose` — show extraction details

17 extractors: docs, deps, tree, config, schemas, entrypoints, HTTP routes, CLI args, exports, comments, TODOs, env vars, protobuf, CMake flags, event interfaces, DB tables, C/C++ headers, ROS2 interfaces, CAN DBC, Yocto/BitBake, device tree, Docker/CI.

## Git workflow

All changes MUST be made on feature branches, not main. Branch naming: `feature/<short-description>` or `fix/<short-description>`. Merge to main only after testing.

Before tagging a release on main:
1. Docs are up to date (README.md, ARCHITECTURE.md)
2. `install.sh` and `uninstall.sh` are verified (syntax check + review for unintended changes)
3. All tests pass (`python3 -m pytest tests/`)
4. Tag with semver: `git tag -a v0.X.Y -m "description"`

## Memory system instructions

The memory block format, context retrieval, confidence system, and all LLM behavioral rules are defined in the global rules file deployed by `install.sh`:

- `~/.claude/rules/memory-system.md` — full system documentation (single source of truth)

The project-local `.claude/rules/memory-system.md` is the source for the global copy. Edit it here, then run `./install.sh` to deploy.
