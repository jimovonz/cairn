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
- `/cairn delete <id>` → `python3 {{CAIRN_HOME}}/cairn/query.py --delete <id>`
- `/cairn daemon start` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py start`
- `/cairn daemon stop` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py stop`
- `/cairn daemon status` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py status`

Run the matching bash command and show the output to the user.

## Special: /cairn audit

When the argument is `audit`, run the audit command first, then review the output:

1. Run: `python3 {{CAIRN_HOME}}/cairn/query.py --audit`
2. For each memory listed in the output, review it against what you know from this conversation:
   - **Accurate**: confirm with ✓
   - **Inaccurate/correctable**: update with `python3 {{CAIRN_HOME}}/cairn/query.py --update <id> <corrected content>`
   - **Inaccurate/unsalvageable**: delete with `python3 {{CAIRN_HOME}}/cairn/query.py --delete <id>`
   - **Stale/superseded**: delete (a newer memory covers it)
   - **Duplicate**: delete the worse copy
   - **Vague/thin**: update with richer content using `--update <id> <better content>`
3. After reviewing all memories, provide a summary:
   - Total reviewed
   - Confirmed accurate
   - Updated/corrected (with old → new)
   - Deleted (with reasons)

User's argument: $ARGUMENTS
