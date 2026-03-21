# Cairn

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Every Claude Code response secretly writes structured metadata about itself. The user never sees it. A hook captures it. A database stores it. The next session remembers.**

Cairn exploits the gap between Claude Code's raw LLM output and its rendered display. Angle bracket tags in LLM responses are stripped from the terminal — but they're preserved in the hook system. This creates an invisible control plane where the LLM self-annotates every response with structured memory data, and the infrastructure enforces it mechanically.

No cloud. No API keys. No MCP. One SQLite file. Two hooks.

---

## What makes this different

Most LLM memory systems store conversation logs and retrieve them with RAG. Cairn does something fundamentally different:

**The LLM is the memory author.** It distills its own output into one-line structured memories — not raw transcripts, not embeddings of chat logs, but the LLM's own assessment of what matters. A 50,000-token session becomes 10 precise facts.

**The metadata is invisible.** The user sees a clean response. The hook infrastructure sees structured XML with type, topic, confidence signals, and retrieval requests. The LLM writes to a channel the user can't see.

**The LLM controls the retrieval loop.** It declares when it lacks context. A Stop hook searches the database, injects results, and re-prompts — all before the response reaches the user. The LLM also rates what it gets back, dynamically adjusting confidence scores that determine what surfaces in future sessions.

**Enforcement is mechanical, not advisory.** A Stop hook fires after every response. No memory block? Blocked and re-prompted. Says it's incomplete? Blocked and continued. Needs context? Blocked, searched, injected, continued. The LLM can't forget to participate.

**Session 1** — casual conversation in `~/temp`:
```
You:    "I see a fairly big mostly blue bird on my lawn. Solid red beak and huge feet"
Claude: "That's a pukeko — NZ Purple Swamphen..."
```

**Session 2** — different directory, days later, working on something unrelated:
```
You:    "what was on my lawn?"
Claude: "A pukeko — NZ Purple Swamphen. Large blue bird, red beak, big feet."
```

The user never asked Claude to remember the bird. Never asked it to look anything up. The memory was captured invisibly in session 1 and surfaced automatically in session 2.

## Features

- **Cross-session memory** — decisions, preferences, facts, corrections, people, projects
- **Three-layer retrieval** — proactive first-prompt push, cross-project keyword surfacing, and LLM-requested pull
- **Dynamic confidence** — saturating model prevents runaway certainty; the LLM rates retrieved memories and bad ones fade
- **Semantic search** — local embeddings via `all-MiniLM-L6-v2` with sqlite-vec indexed vector search
- **Project scoping** — memories auto-labelled by working directory, retrievable per-project or globally
- **Invisible** — metadata tags are stripped from user display; the system operates transparently
- **Quality gates** — garbage filtering, borderline suppression, relative filtering, dominance control
- **Contradiction handling** — same-topic updates suppress the old entry; negation heuristics dampen conflicting memories
- **Self-improving** — retrieval outcome feedback adaptively tightens thresholds when results are poor

## Quick start

```bash
git clone https://github.com/jimovonz/cairn.git ~/cairn
cd ~/cairn
./install.sh
```

Restart Claude Code. The system is now active in every session.

The installer:
1. Creates a Python venv and installs dependencies
2. Initializes the SQLite database
3. Deploys global hooks, instructions, and the `/cairn` slash command
4. Downloads the embedding model (~80MB, one-time)
5. Starts the embedding daemon

## Usage

The system works automatically. No manual action required.

Every Claude Code response produces invisible metadata that gets captured and stored. When the LLM needs past context, it requests it and the system injects relevant memories with project scoping, confidence scores, and recency weighting.

### Slash commands

| Command | Description |
|---------|-------------|
| `/cairn` | Memory stats, confidence distribution, drift indicators |
| `/cairn recent` | Recently stored memories |
| `/cairn projects` | List all projects with memory counts |
| `/cairn project <name>` | All memories for a project |
| `/cairn search <term>` | Full-text search |
| `/cairn semantic <query>` | Semantic similarity search |
| `/cairn review` | Surface low-confidence and suppressed memories |
| `/cairn context <id>` | Show conversation context around where a memory was recorded |
| `/cairn history <id>` | Version history for a memory |
| `/cairn compact [project]` | Dense dump suitable for LLM ingestion |
| `/cairn verify` | Analyse source location accuracy |
| `/cairn backfill` | Generate embeddings for memories stored without daemon |
| `/cairn delete <id>` | Delete a memory |
| `/cairn daemon start\|stop\|status` | Manage the embedding daemon |

## How it works

### The invisible metadata mechanism

Every LLM response ends with a `<memory>` block using angle bracket tags. Claude Code strips these from the displayed output — the user sees a clean response. But the Stop hook has full access to the structured data.

```
<memory>
- type: decision
- topic: auth-approach
- content: Use JWT for stateless auth, no server sessions
- keywords: authentication, JWT, session
- source_messages: 15-22
- complete: true
</memory>
```

### Three retrieval layers

| Layer | When | What |
|-------|------|------|
| **First-prompt push** | First message of session | Proactively injects relevant context before the LLM starts generating |
| **Keyword cross-project** | Between turns | Surfaces global knowledge based on topic keywords from the current conversation |
| **Pull-based** | When LLM identifies a gap | LLM declares `context: insufficient`, hook searches and injects |

### Confidence system

Memories start at 0.7 confidence. The LLM rates retrieved memories:
- `+` → saturating boost: `0.1 × (1 - confidence)` — hard to become certain
- `-` → scaled penalty: `0.2 × (1 + confidence)` — easy to lose certainty

No passive decay. Important but rarely accessed memories retain their confidence indefinitely.

### Quality gates

Retrieved results pass through 9 configurable gates before injection:

1. Low-information pre-filter (skip generic queries)
2. Garbage gate (reject if best similarity < 0.35)
3. Borderline gate (reject weak similarity + low score)
4. Adaptive threshold (auto-tighten if recent retrievals were poor)
5. Relative filter (drop entries far below the best match)
6. Soft confidence inclusion (high similarity overrides low confidence)
7. Dominance suppression (include runner-up if close to leader)
8. Weak-entry suppression (don't inject if top result is unreliable)
9. Hard cap (max 5 entries)

All thresholds configurable in `cairn/config.py`.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical reference (600+ lines), including:

- Database schema (memories, sessions, history, metrics)
- Composite scoring formula
- Deduplication and contradiction handling
- Embedding strategy and vector search
- Loop protection mechanisms
- Design decisions and rationale

## File structure

```
cairn/
├── install.sh              # One-command installer
├── requirements.txt        # Python dependencies
├── CLAUDE.md               # Project-local LLM instructions
├── .claude/
│   ├── settings.json       # Project-local hooks
│   └── rules/
│       └── memory-system.md  # Full system rules for the LLM
├── cairn/
│   ├── config.py           # All tunable parameters
│   ├── init_db.py          # Schema and migrations
│   ├── query.py            # CLI query tool (20+ commands)
│   ├── embeddings.py       # Embedding with daemon support + composite scoring
│   └── daemon.py           # Background embedding server (Unix socket)
├── hooks/
│   ├── stop_hook.py        # Core: capture, enforce, retrieve, confidence
│   └── prompt_hook.py      # Layer 1 + Layer 2 injection
└── templates/              # Installer templates for global config
```

## Requirements

- [Claude Code](https://claude.com/claude-code) v2.1+
- Python 3.10+
- ~1GB disk (embedding model + venv)
- ~500MB RAM (when embedding daemon is running; auto-shuts down after 30min idle)

**Platform:** Developed and tested on Ubuntu 22.04. Linux and macOS should work. Windows requires WSL — the installer is bash, and the embedding daemon uses Unix sockets. The core hooks work without the daemon (slower embedding, no daemon acceleration) but `install.sh` must run in a Unix shell.

## Configuration

All tunable parameters are in `cairn/config.py`:

- Retrieval thresholds per layer
- Composite scoring weights
- Confidence boost/penalty rates
- Quality gate thresholds
- Deduplication sensitivity
- Daemon idle timeout
- Loop protection limits

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| **No MCP** | Claude Code has direct filesystem access — MCP adds a protocol layer for capabilities already available natively |
| **Pull-based retrieval** | The LLM decides when it needs context — more token-efficient than injecting on every prompt |
| **Local embeddings** | No API keys, no network latency, no ongoing costs. `all-MiniLM-L6-v2` is 80MB and fast |
| **Saturating confidence** | Prevents runaway certainty on frequently retrieved memories without introducing passive forgetting |
| **Invisible tags** | User sees clean output; hook infrastructure sees structured metadata — no UX compromise |
| **sqlite-vec** | Indexed vector KNN search that scales, with transparent brute-force fallback |

## Limitations

**Claude Code only.** Cairn is tightly coupled to Claude Code's hook system and tag-stripping behaviour. It will not work with Cursor, VS Code agents, other LLMs, or the Claude web interface. This is by design — the architecture exploits Claude Code's specific capabilities rather than targeting a lowest common denominator.

**LLM cooperation is imperfect.** The system depends on the LLM reliably producing well-formed `<memory>` blocks and accurately declaring when it needs context. In practice, the LLM sometimes answers "I don't know" before the hook can inject memories, or produces generic memories instead of extracting specific facts. Mechanical enforcement (the Stop hook) catches most failures but adds a re-prompt turn when it does.

**Tag invisibility is behaviour-dependent.** The invisible metadata relies on Claude Code stripping angle bracket tags from rendered output. If Anthropic changes this rendering behaviour, memory blocks would become visible to users. The system would still function but the clean UX would degrade.

**Distillation is lossy.** Memories are one-line summaries. The nuance and detail of the original conversation is compressed away. The `--context` command can recover the surrounding transcript, but only if the session's transcript file still exists on disk.

**Early stage.** This project was built and iterated in a single extended session. It has limited cross-platform testing and may have edge cases around permissions, venv conflicts, or long-running daemon stability. Bug reports welcome.

## Failure modes

Things that can go wrong and how the system handles them:

| Failure | What happens | Mitigation |
|---------|-------------|------------|
| LLM forgets the `<memory>` block | Stop hook blocks the response and re-prompts "add a memory block" | User sees a brief pause; the re-prompt is invisible |
| LLM answers before checking memory | User sees "I don't know" then a correction after the hook injects context | Layer 1 (first-prompt push) proactively injects on the first message to prevent this |
| Embedding daemon not running | Memories stored without embeddings; dedup and semantic search degraded | Auto-start attempted; `--backfill` command regenerates missing embeddings |
| Hook crashes | Fail-open design: crash → exit 0 → response reaches user normally | Crash logged to metrics; no user impact |
| Retrieval returns irrelevant context | 9 quality gates filter noise; adaptive thresholds tighten if outcomes are poor | LLM can rate retrieval as `harmful`, raising thresholds automatically |
| Infinite re-prompt loop | Continuation cap (max 3) forces a stop after 3 consecutive re-prompts | Context cache prevents same query being served twice |
| Contradictory memories | Same type+topic overwrites with confidence suppression; negation heuristics dampen semantic conflicts | Old content preserved in version history |
| Database grows large | sqlite-vec provides indexed vector search; brute-force fallback for small DBs | All quality gates reduce injected volume regardless of DB size |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug fixes, retrieval improvements, test coverage, and platform compatibility contributions are especially welcome.

## Testing

156 tests across 9 test files. No embedding model required — tests use mock vectors and patched DB paths.

```bash
cd ~/cairn
python3 -m pytest tests/     # or run individually: python3 tests/test_parser.py
```

| Test file | Count | What it covers |
|-----------|-------|---------------|
| `test_parser.py` | 32 | Memory block parsing: valid, malformed, unclosed tags, code fences, special chars, stress |
| `test_scoring.py` | 18 | Composite scoring, recency decay, saturating confidence, negation heuristics |
| `test_gates.py` | 21 | All 9 quality gates: boundary conditions, interactions, diversity filter |
| `test_integration.py` | 14 | Full pipeline with in-memory DB: insert → dedup → retrieve → gate |
| `test_hook_e2e.py` | 13 | Stop hook main() with patched stdin: storage, blocking, sessions, metrics |
| `test_prompt_hook.py` | 8 | Layer 1/2: first-prompt detection, staged context, short message handling |
| `test_daemon_and_cache.py` | 16 | Daemon fallback, context cache, loop protection, fail-open, metrics |
| `test_query_cli.py` | 12 | CLI commands: search, stats, review, delete, history, compact, projects |
| `test_retrieval_pipeline.py` | 22 | Retrieval pipeline: find_nearest, insert dedup/contradiction/variant paths, retrieve_context XML, adaptive thresholds, Layer 2 cross-project, session registration, auto-labelling edge cases, negation dampening |

## License

[MIT](LICENSE)
