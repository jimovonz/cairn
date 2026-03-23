---
title: I built invisible persistent memory for Claude Code
published: true
description: Every Claude Code response secretly writes metadata about itself. A hook captures it. A database stores it. The next session remembers.
tags: claude, ai, developer-tools, open-source
cover_image: https://raw.githubusercontent.com/jimovonz/cairn/main/docs/social-preview.png
---

## The problem

If you use Claude Code heavily, you've hit this: you spend an hour explaining your project's architecture, your preferences, past decisions — and then the session ends. Next time, Claude starts from zero. You explain everything again.

Context windows are big now (200K, even 1M tokens). But they don't persist. Every new session is amnesia.

## The exploit

Claude Code strips angle bracket tags from its rendered output. If Claude writes `<anything>content</anything>` in its response, the user never sees it — but the raw response still contains it.

This is normally just a rendering detail. But Claude Code's hook system has full access to the raw response *before* it's stripped.

That gap — between what Claude writes and what you see — is an invisible control plane. And it's enough to build a persistent memory system.

## How Cairn works

Every Claude Code response ends with a hidden `<memory>` block:

```
Here's the refactored authentication module...

<memory>
- type: decision
- topic: auth-approach
- content: Use JWT for stateless auth — rejected session cookies
  because the API is consumed by mobile clients that don't persist cookies
- keywords: authentication, JWT, session
- complete: true
- context: sufficient
</memory>
```

You see: *"Here's the refactored authentication module..."*

The hook sees everything. It parses the memory block, generates a semantic embedding, deduplicates against existing memories, and stores it in a local SQLite database with confidence scoring.

Next session — even in a different project directory — when the topic comes up again, the system searches the database and injects the relevant memories back into context. Claude "remembers" what you decided and why.

## The pūkeko test

Here's the moment I knew it worked.

**Session 1** — casual conversation in `~/temp`:
```
You:    "I see a fairly big mostly blue bird on my lawn. Solid red beak
         and huge feet"
Claude: "That's a pūkeko — NZ Purple Swamphen..."
```

**Session 2** — different directory, days later:
```
You:    "what was on my lawn?"
Claude: "A pūkeko — NZ Purple Swamphen. Large blue bird, red beak,
         big feet."
```

Nobody asked Claude to remember the bird. Nobody asked it to look anything up. The memory was captured invisibly in session 1 and surfaced automatically in session 2.

## What makes this different

Most LLM memory systems store conversation logs and retrieve them with RAG. Cairn does something fundamentally different:

**The LLM is the memory author.** It distills its own output into structured one-line summaries — not raw transcripts, not embeddings of chat logs, but the LLM's own assessment of what matters. A 50,000-token session becomes 10 precise facts.

**Enforcement is mechanical.** A stop hook fires after every response. No memory block? Blocked and re-prompted. Says it's incomplete? Continued automatically. Needs past context? Searched, injected, and re-prompted — all before the response reaches you.

**Quality is actively managed.** Memories start at 0.7 confidence. The LLM rates what it retrieves — bad memories fade. 9 quality gates filter noise. Contradictions are detected. Archived memories preserve the learning trail of rejected approaches.

**It's fully local.** No cloud. No API keys. No MCP. One SQLite file, two hooks, and a lightweight embedding model (~80MB). Your memories stay on your machine.

## Quick start

```bash
git clone https://github.com/jimovonz/cairn.git ~/cairn
cd ~/cairn
./install.sh
```

Restart Claude Code. It's active in every session from that point.

The installer sets up a Python venv, initializes the database, deploys the hooks globally, downloads the embedding model, and starts a background daemon for fast embeddings. Takes about a minute.

## What you get

- **Cross-session, cross-project memory** — decisions, corrections, preferences, facts persist
- **Slash commands** — `/cairn search auth`, `/cairn audit`, `/cairn check`, `/cairn recent`
- **Memory audit system** — review, enrich, archive, and fill gaps in stored memories
- **Context recovery** — `--context <id>` recovers the full conversation behind any memory
- **Self-healing** — auto-starts embedding daemon, backfills missing embeddings, fails open on errors
- **185 tests** across 11 files — the system is mechanically reliable

## The architecture (for the curious)

The [ARCHITECTURE.md](https://github.com/jimovonz/cairn/blob/main/ARCHITECTURE.md) is 800+ lines if you want the full picture. The highlights:

- **Three retrieval layers**: proactive first-prompt push, cross-project keyword surfacing, and on-demand LLM-requested pull
- **9 quality gates**: garbage filtering, borderline suppression, relative filtering, soft confidence inclusion, dominance control, diversity dedup, adaptive thresholds, weak-entry suppression, hard cap
- **Saturating confidence model**: boosts diminish as confidence approaches 1.0, penalties scale with overconfidence. No passive decay — important but rarely retrieved memories keep their confidence
- **Modular codebase**: 7 hook modules (parser, storage, enforcement, retrieval, helpers, orchestrator, prompt hook) with type hints throughout

## Limitations (honestly)

- **Claude Code only.** This exploits Claude Code's specific hook system and tag-stripping behaviour. It won't work with Cursor, VS Code agents, or the Claude web interface.
- **LLM cooperation is imperfect.** Claude sometimes answers before the hook can inject memories, or produces generic memories instead of extracting specific facts. Mechanical enforcement catches most failures.
- **Tag invisibility is behaviour-dependent.** If Anthropic changes how angle bracket tags are rendered, the invisible channel breaks. The system would still function but memory blocks would become visible.
- **Early stage.** Tested on two machines (Ubuntu). May have edge cases with permissions, venv conflicts, or daemon stability.

## Try it

The repo is at [github.com/jimovonz/cairn](https://github.com/jimovonz/cairn). MIT licensed. Bug reports and contributions welcome.

If you're a heavy Claude Code user who's tired of re-explaining context every session, give it a try. The installer makes it painless to experiment, and there's an uninstaller if it's not for you.

---

*Cairn — a mound of stones built as a trail marker, placed one at a time by those who pass, so that those who follow can find their way.*
