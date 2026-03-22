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
- `/cairn delete <id>` → `python3 {{CAIRN_HOME}}/cairn/query.py --delete <id>`
- `/cairn daemon start` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py start`
- `/cairn daemon stop` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py stop`
- `/cairn daemon status` → `{{VENV_PYTHON}} {{CAIRN_HOME}}/cairn/daemon.py status`

Run the matching bash command and show the output to the user.

User's argument: $ARGUMENTS
