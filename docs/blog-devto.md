---
title: I built "Invisible" Persistent Memory for Claude Code
published: true
description: Every Claude Code response secretly writes metadata about itself. A hook captures it. A database stores it. The next session remembers.
tags: claude, ai, developer-tools, open-source
cover_image: https://raw.githubusercontent.com/jimovonz/cairn/main/docs/social-preview.png
---

## If you use Claude Code heavily, you've hit the wall.

You spend an hour explaining your project's architecture, your library preferences, and the specific quirks of your legacy code. Then the session ends. The next time you boot up `claude`, it starts from zero. You have to re-explain everything. Even with massive context windows, every new session is a fresh case of amnesia.

I built **Cairn** to fix this. It's an open-source persistent memory system that lives entirely on your machine.

## The "Exploit": The Invisible Control Plane

I discovered a loophole in how Claude Code renders output.

Claude Code automatically strips XML-style tags (like `<example>...</example>`) from its final display. If Claude writes a tag in its response, the user never sees it, but the raw response still contains it.

Claude Code's hook system has access to that raw response *before* it's stripped. I realized this gap—between what Claude writes and what you see—is an **invisible control plane**.

## How Cairn Works

Every time Claude finishes a thought, Cairn forces it to append a hidden `<memory>` block:

```xml
Here's the refactored authentication module...

<memory>
- type: decision
- topic: auth-approach
- content: Use JWT for stateless auth — rejected session cookies
  because the API is consumed by mobile clients.
- complete: true
- context: sufficient
</memory>
```

**You see:** *"Here's the refactored authentication module..."*
**The Hook sees:** The hidden structured data.

Cairn's parser captures that block, generates a semantic embedding using a local model, and stores it in a SQLite database. In your next session—even days later or in a different directory—Cairn searches the database and injects relevant "memories" back into the prompt.

Claude "remembers" your decisions because it wrote them down for itself.

## Stopping the "I'll do that now..." Lie

We've all seen it: Claude says, *"I'll check the logs for that error,"* and then the terminal prompt just returns. It didn't check the logs. It just... stopped.

Cairn solves this using **Mechanical Enforcement**. It acts as an agentic supervisor through a **Dual-Gate Guard**:

**The Structural Gate (`complete: false`):** Inside the hidden memory block, Claude must report its own state. If Claude marks the task as `complete: false`, Cairn blocks the response from reaching you. It silently re-prompts: *"You reported this is incomplete. Continue."*

**The Semantic Gate (Trailing Intent):** Cairn analyzes the end of every response. If Claude states an intent to act (e.g., *"I will run the tests"*) but the memory block claims completion, Cairn detects the contradiction. It knows Claude is "ghosting" the task and forces it to actually execute the command.

## The "Pūkeko" Test

Here is how I knew the retrieval was working.

**Session 1** (in `~/temp`): I mentioned I saw a "big blue bird with a red beak" on my lawn. Claude identified it as a Pūkeko. I didn't save any files or notes.

**Session 2** (Three days later, in `~/projects`): I simply asked, *"What was on my lawn?"*

**The Result:** Claude answered instantly: *"A pūkeko — NZ Purple Swamphen."*

It didn't search my files. It didn't look at chat logs. It retrieved its own distilled memory from the SQLite database.

## Why this is different from "Standard RAG"

Most LLM memory systems just dump your chat logs into a vector DB. Cairn is built for the high-pressure environment of a coding agent:

**The LLM is the Memory Author:** Cairn doesn't store raw transcripts. It forces the LLM to distill facts. A 50,000-token session becomes 10 precise, high-value memories.

**Epistemic Traceability:** One-line summaries are great for context window efficiency, but if you need the "why," Cairn can help. Every memory is a pointer. Using `--context <id>`, you can instantly recover the original multi-turn conversation from your local logs.

**9 Quality Gates:** This isn't just "top-k" search. Cairn uses a sophisticated retrieval pipeline including *Saturating Confidence* (important memories stay relevant; "noise" fades) and *Adaptive Thresholds* (it learns your project's noise floor and tightens retrieval automatically).

**100% Local & Fast:** No extra API keys. No cloud privacy concerns. Cairn runs a lightweight Python daemon in the background to keep the embedding model (~80MB) in RAM, so memory injection happens in milliseconds.

## Quick Start

It takes about a minute to install. The installer sets up a Python venv, initializes the database, deploys the hooks globally, and starts the background daemon.

```bash
git clone https://github.com/jimovonz/cairn.git ~/cairn
cd ~/cairn
./install.sh
```

Restart Claude Code. It's now active in every session. You'll also get access to new slash commands:

- `/cairn search <query>`: See what Claude knows about a topic.
- `/cairn audit`: Review, edit, or delete stored memories.
- `/cairn recent`: See what was captured in the last hour.

## Limitations (Honestly)

- **Claude Code only:** This relies on the specific hook system and tag-stripping behavior of the `claude` CLI.
- **Early stage:** I've tested this extensively on Ubuntu. It's stable, but edge cases with permissions or venv conflicts might exist.
- **LLM Cooperation:** While mechanical enforcement catches most failures, Claude is still non-deterministic. Sometimes it needs a nudge to write a good memory.

## Try it out

I built this because I wanted a coding partner that actually learns my habits and project history. If you're a heavy Claude Code user, I'd love your feedback.

**GitHub Repo:** [https://github.com/jimovonz/cairn](https://github.com/jimovonz/cairn)

---

*Cairn — a mound of stones built as a trail marker, placed one at a time by those who pass, so that those who follow can find their way.*
