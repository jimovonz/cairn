# Cairn query.py — full CLI reference

Loaded on demand (`cat` this file); not part of always-on context.
All commands: `python3 {{CAIRN_HOME}}/cairn/query.py <args>`

## Search

| Flag | Effect |
| --- | --- |
| `<search>` | full-text (FTS5) search |
| `--semantic <query>` | semantic similarity search; `\|` splits into subqueries, run independently and merged |
| `--context <id>` | full transcript excerpt (or source files, for repo-ingested entries) behind a memory |

## Listing and filtering

| Flag | Effect |
| --- | --- |
| `--recent` | recent memories |
| `--today` | memories from today |
| `--since <date>` | from date onward (ISO, `today`, `yesterday`, `3d`, `2w`, `1m`) |
| `--since <a> --until <b>` | date range |
| `--type <type>` | filter by entry type |
| `--session <id>` | filter by session |
| `--chain <id>` | show session chain |
| `--project <name>` | memories for one project |
| `--projects` | list all projects |

## Maintenance

| Flag | Effect |
| --- | --- |
| `--label <session_id> <name>` | label a session chain |
| `--history <id>` | version history of a memory |
| `--delete <id>` | delete a memory |
| `--stats` | database statistics |

## When to use `--context`

Memories are distilled one-liners; `--context` recovers the full detail. Use it
freely, not as a last resort:

- the one-liner is ambiguous and you need the original discussion
- the user asks what exactly was decided about X
- you need to verify a memory reflects what was actually discussed
- you are about to act on a memory and want the reasoning behind it
- a repo-ingested memory references code you need to see
