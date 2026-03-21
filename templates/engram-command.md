---
description: Query the Engram memory system — stats, search, review, projects
---

Run the Engram query tool based on the user's request. The argument determines which command to run.

If no argument is given, run `--stats`.

Map the argument to the appropriate command:

- `/engram` or `/engram stats` → `python3 {{ENGRAM_HOME}}/engram/query.py --stats`
- `/engram recent` → `python3 {{ENGRAM_HOME}}/engram/query.py --recent`
- `/engram projects` → `python3 {{ENGRAM_HOME}}/engram/query.py --projects`
- `/engram review` → `python3 {{ENGRAM_HOME}}/engram/query.py --review`
- `/engram search <term>` → `python3 {{ENGRAM_HOME}}/engram/query.py <term>`
- `/engram semantic <query>` → `{{VENV_PYTHON}} {{ENGRAM_HOME}}/engram/query.py --semantic <query>`
- `/engram project <name>` → `python3 {{ENGRAM_HOME}}/engram/query.py --project <name>`
- `/engram context <id>` → `python3 {{ENGRAM_HOME}}/engram/query.py --context <id>`
- `/engram history <id>` → `python3 {{ENGRAM_HOME}}/engram/query.py --history <id>`
- `/engram compact` → `python3 {{ENGRAM_HOME}}/engram/query.py --compact`
- `/engram compact <project>` → `python3 {{ENGRAM_HOME}}/engram/query.py --compact <project>`
- `/engram verify` → `python3 {{ENGRAM_HOME}}/engram/query.py --verify-sources`
- `/engram backfill` → `{{VENV_PYTHON}} {{ENGRAM_HOME}}/engram/query.py --backfill`
- `/engram delete <id>` → `python3 {{ENGRAM_HOME}}/engram/query.py --delete <id>`
- `/engram daemon start` → `{{VENV_PYTHON}} {{ENGRAM_HOME}}/engram/daemon.py start`
- `/engram daemon stop` → `{{VENV_PYTHON}} {{ENGRAM_HOME}}/engram/daemon.py stop`
- `/engram daemon status` → `{{VENV_PYTHON}} {{ENGRAM_HOME}}/engram/daemon.py status`

Run the matching bash command and show the output to the user.

User's argument: $ARGUMENTS
