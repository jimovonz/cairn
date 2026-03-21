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
- `python3 ./cairn/query.py --history <id>` — show version history
- `python3 ./cairn/query.py --delete <id>` — delete a memory
- `python3 ./cairn/query.py --stats` — database statistics

## Memory system instructions

The memory block format, context retrieval, confidence system, and all LLM behavioral rules are defined in the global rules file deployed by `install.sh`:

- `~/.claude/CLAUDE.md` — compact memory block format and critical rules
- `~/.claude/rules/memory-system.md` — full system documentation

The project-local `.claude/rules/memory-system.md` is the source for the global copy. Edit it here, then run `./install.sh` to deploy.
