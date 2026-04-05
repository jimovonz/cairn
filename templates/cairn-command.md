---
description: Query the Cairn memory system — stats, search, review, projects
---

Run the Cairn query tool based on the user's request. The argument determines which command to run.

If no argument is given, run `--stats`.

Map the argument to the appropriate command:

- `/cairn` or `/cairn stats` → `python3 {{CAIRN_HOME}}/cairn/query.py --stats`
- `/cairn recent` → `python3 {{CAIRN_HOME}}/cairn/query.py --recent`
- `/cairn projects` → `python3 {{CAIRN_HOME}}/cairn/query.py --projects`
- `/cairn review` → `python3 {{CAIRN_HOME}}/cairn/query.py --review`
- `/cairn search <term>` → `python3 {{CAIRN_HOME}}/cairn/query.py <term>`
- `/cairn semantic <query>` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/query.py --semantic <query>`
- `/cairn project <name>` → `python3 {{CAIRN_HOME}}/cairn/query.py --project <name>`
- `/cairn context <id>` → `python3 {{CAIRN_HOME}}/cairn/query.py --context <id>`
- `/cairn history <id>` → `python3 {{CAIRN_HOME}}/cairn/query.py --history <id>`
- `/cairn compact` → `python3 {{CAIRN_HOME}}/cairn/query.py --compact`
- `/cairn compact <project>` → `python3 {{CAIRN_HOME}}/cairn/query.py --compact <project>`
- `/cairn verify` → `python3 {{CAIRN_HOME}}/cairn/query.py --verify-sources`
- `/cairn backfill` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/query.py --backfill`
- `/cairn check` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/query.py --check`
- `/cairn audit` → Run `python3 {{CAIRN_HOME}}/cairn/query.py --audit` then follow the audit instructions below
- `/cairn audit-bg` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/audit_agent.py` (background audit via claude -p agent)
- `/cairn delete <id>` → `python3 {{CAIRN_HOME}}/cairn/query.py --delete <id>`
- `/cairn daemon start` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py start`
- `/cairn daemon stop` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py stop`
- `/cairn daemon status` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py status`
- `/cairn bootstrap` → `python3 {{CAIRN_HOME}}/cairn/query.py --bootstrap`
- `/cairn bootstrap <project>` → `python3 {{CAIRN_HOME}}/cairn/query.py --bootstrap <project>`
- `/cairn dashboard` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/dashboard.py &` (launches web dashboard in background, opens browser at http://127.0.0.1:8420)

Run the matching bash command and show the output to the user.

## Special: /cairn audit

When the argument is `audit`, run the audit command first, then review the output:

1. Run: `python3 {{CAIRN_HOME}}/cairn/query.py --audit`
2. For each memory listed in the output, review and ENRICH:
   - **Accurate but thin**: update with `--update <id> <richer content>` — add the *why*, alternatives considered, reasoning, or outcome
   - **Accurate and complete**: confirm with ✓
   - **Inaccurate/correctable**: update with `--update <id> <corrected content>`
   - **Superseded**: archive with `--archive <id> <reason>` — e.g. "superseded by decision to use X instead". The learning trail is preserved.
   - **Wrong/misleading**: archive with `--archive <id> <reason>` — e.g. "false — no compaction occurred". Archived memories stay in DB but drop from retrieval.
   - **Duplicate**: archive the worse copy with `--archive <id> duplicate of <other_id>`
   - **Missing context**: if the conversation covered decisions, reasoning, or facts NOT captured by any memory, add with `--add <type> <topic> <content> [--project <name>]`
3. After reviewing all memories, provide a summary:
   - Total reviewed
   - Confirmed accurate
   - Enriched (old → new)
   - Archived (with reasons)
   - New memories added (gaps filled)

User's argument: $ARGUMENTS
