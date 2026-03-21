# Engram — Persistent Memory System

## Memory Output Requirement

Every response MUST end with a <memory> block. No exceptions. This is a mechanical requirement, not a judgment call.

Format:
```
<memory>
- type: [decision|preference|fact|correction|person|project|skill|workflow]
- topic: [short key]
- content: [single line description]
- complete: [true|false]
- remaining: [what still needs doing, if complete is false]
</memory>
```

Types:
- decision: Architectural or design choices made with rationale
- preference: User likes, dislikes, working style
- fact: Verified information about systems, tools, environment
- correction: Mistakes made and lessons learned
- person: People mentioned — roles, relationships, contact context
- project: Ongoing work, goals, deadlines, status
- skill: Reusable techniques, commands, or patterns that worked
- workflow: Recurring processes, automation, standard operating procedures

Rules:
- Every response gets a memory block, even if the only entry is a no-op
- If nothing new was learned: `<memory>complete: true</memory>`
- Each entry is one line of content — no multi-line values
- complete: false will cause the system to re-prompt you to continue
- Never narrate a future action without executing it — if you say "let me do X", do X in the same response via a tool call

## Engram Database

The memory database is at `./engram/engram.db`. Use `python3 ./engram/query.py` to search it.

Query commands:
- `python3 ./engram/query.py <search>` — full-text search
- `python3 ./engram/query.py --semantic <query>` — semantic similarity search
- `python3 ./engram/query.py --recent` — list recent memories
- `python3 ./engram/query.py --type <type>` — filter by type
- `python3 ./engram/query.py --session <id>` — filter by session
- `python3 ./engram/query.py --history <id>` — show version history
- `python3 ./engram/query.py --delete <id>` — delete a memory
- `python3 ./engram/query.py --stats` — database statistics

At the start of a conversation, query the engram for context relevant to the user's first message.

## Project

This is the Engram project — a persistent AI memory system using SQLite, Claude Code hooks, and structured self-assessment.
