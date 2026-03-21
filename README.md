# Engram

Persistent memory for Claude Code. Every conversation builds on the last.

Engram embeds invisible structured metadata into every LLM response, captures it via Claude Code hooks, and stores it in a local SQLite database with semantic search. The LLM self-assesses what's worth remembering, when it needs past context, and whether retrieved memories are still accurate.

## What it does

- **Remembers across sessions** — decisions, preferences, facts, corrections, people, projects
- **Retrieves automatically** — the LLM declares when it needs context, the system searches and injects
- **Self-improves** — confidence scoring lets the LLM rate retrieved memories, suppressing bad ones over time
- **Cross-project** — memories from any project are accessible globally, scoped by project relevance
- **Invisible** — the metadata is hidden from the user, the system operates transparently

## Install

```bash
git clone <repo-url> ~/engram
cd ~/engram
./install.sh
```

Then restart Claude Code. That's it.

## Usage

The system works automatically. Every Claude Code response produces metadata that gets captured and stored. When the LLM needs past context, it requests it and the system injects relevant memories.

### Slash commands

| Command | Description |
|---------|-------------|
| `/engram` | Show memory stats |
| `/engram recent` | Recent memories |
| `/engram projects` | List all projects |
| `/engram search <term>` | Full-text search |
| `/engram semantic <query>` | Semantic similarity search |
| `/engram project <name>` | Memories for a project |
| `/engram review` | Surface low-confidence memories |
| `/engram context <id>` | Conversation context around a memory |
| `/engram compact` | Dense brain dump |
| `/engram daemon start/stop/status` | Manage embedding daemon |

### How it works

1. Every LLM response ends with an invisible `<memory>` block (angle bracket tags are stripped from Claude Code's display)
2. A Stop hook captures the block, embeds the content, deduplicates, and stores it
3. When the LLM needs past context, it declares `context: insufficient` — the hook searches and injects relevant memories
4. A UserPromptSubmit hook proactively injects context on the first message of each session
5. The LLM can rate retrieved memories (`confidence_update: id:+/-`) and the retrieval itself (`retrieval_outcome: useful/neutral/harmful`)

### Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical reference (600+ lines).

## Key design decisions

- **No MCP** — Claude Code has direct filesystem access, MCP adds unnecessary overhead
- **Pull-based retrieval** — the LLM requests context when needed, not pushed on every prompt
- **Local embeddings** — `all-MiniLM-L6-v2` via sentence-transformers, no API keys needed
- **Saturating confidence** — diminishing returns on positive feedback, amplified penalties on negative — the system is reluctant to become certain, quick to lose certainty when challenged
- **sqlite-vec** — indexed vector search for scale, transparent fallback to brute-force

## Requirements

- Claude Code v2.1+
- Python 3.10+
- ~1GB disk (model + venv)
- ~500MB RAM (when embedding daemon is running)

## License

MIT
