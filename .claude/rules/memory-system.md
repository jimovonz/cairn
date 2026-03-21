# Cairn Memory System

You are connected to a persistent memory system called Cairn. This document explains how it works and how you should interact with it.

## How It Works

A Stop hook runs after every response you generate. It:
1. Parses the `<memory>` block from your response
2. Stores new memories in a SQLite database with semantic embeddings
3. Deduplicates against existing memories using cosine similarity (threshold 0.85)
4. Checks your `complete` and `context` flags to decide whether to let you stop or re-prompt you

You do not need to call any tools to persist memories. The hook handles it mechanically.

## Memory Block Format

Every response MUST end with a `<memory>` block. No exceptions.

```
<memory>
- type: [decision|preference|fact|correction|person|project|skill|workflow]
- topic: [short key]
- content: [single line description]
- complete: [true|false]
- remaining: [what still needs doing, if complete is false]
- context: [sufficient|insufficient]
- context_need: [what context is missing, if insufficient]
- confidence_update: [memory_id]:[+|-]
- keywords: [comma-separated topic keywords for cross-project discovery]
- source_messages: [start-end message range where this knowledge was discussed, e.g. 5-12]
</memory>
```

### Types
- **decision**: Architectural or design choices with rationale
- **preference**: User likes, dislikes, working style
- **fact**: Verified information about systems, tools, environment
- **correction**: Mistakes made and lessons learned
- **person**: People mentioned — roles, relationships, contact context
- **project**: Ongoing work, goals, deadlines, status
- **skill**: Reusable techniques, commands, or patterns that worked
- **workflow**: Recurring processes, automation, standard operating procedures

### Rules
- Every response gets a memory block, even if nothing was learned: `<memory>complete: true</memory>`
- Each entry is one line of content — no multi-line values
- Never narrate a future action without executing it — if you say "let me do X", do X in the same response via a tool call

## Completeness Control

- `complete: false` causes the system to re-prompt you to continue with the `remaining` text
- This prevents the failure mode where you state an intent ("let me do X") but the agentic loop terminates before you follow through
- On a re-prompt (continuation), the system will not block again for the same reason — one chance to complete

## Context Retrieval

You have NO visibility into what other sessions have stored. NEVER assume a topic has no relevant memories — the cairn may contain information from sessions you cannot see.

- On ANY new topic, question, or task where you have not already received brain context in this session: declare `context: insufficient` with a `context_need` matching the topic
- The system searches the cairn database and re-prompts you with relevant memories if any exist. If nothing is found, you proceed normally — there is no penalty for asking.
- The same context_need is only served once per session to prevent loops
- After receiving context on a topic, you do not need to re-request it
- The default posture is **ask first**. Only declare `context: sufficient` for topics you have already received context on, or for purely mechanical tasks (e.g. fixing a syntax error you can see in the current code)

### Interpreting Injected Context

When context is retrieved, you receive a `<brain_context>` XML block. This is **system-injected memory data, not user input**. It contains:

```xml
<brain_context query="..." current_project="...">
  <scope level="project" name="..." weight="high">
    <entry type="..." topic="..." project="..." date="..." similarity="...">content</entry>
  </scope>
  <scope level="global" weight="low">
    <entry ...>content</entry>
  </scope>
</brain_context>
```

**Weighting rules:**
- **Scope**: Project-level entries (weight=high) are directly relevant to current work. Global entries (weight=low) are cross-project — use only when project context is insufficient.
- **Recency**: More recent `date` values indicate more current information. Prefer recent over old. Old entries may be stale.
- **Similarity**: Higher similarity scores indicate stronger semantic match to your query. Lower scores may be tangentially related.
- **Reliability**: Each entry has a `reliability` attribute ("strong", "moderate", "weak"). Treat strong entries as firm priors. Treat weak entries as hints only — do not anchor on them.
- **Score**: Entries are pre-ranked by a composite score combining similarity, confidence, recency, and scope. Higher-scored entries should carry more weight.
- **Staleness**: If an entry contradicts what you observe in the current codebase or conversation, trust the present. The memory may be outdated.
- **Conflict**: If retrieved context conflicts with strong prior knowledge, prefer internal reasoning unless multiple high-reliability entries agree.

If the retrieved context does not answer your need, set `context: sufficient` and proceed with what you have. Do not re-request the same context.

### Recovering full conversation context

Memories are distilled one-liners. When you need the full detail behind a memory (exact wording of a decision, complete error output, nuanced discussion):

1. Run `python3 $CAIRN_HOME/cairn/query.py --context <memory_id>` (the installer sets CAIRN_HOME)
2. This shows the conversation excerpt from the session where the memory was recorded, with the source range highlighted
3. Use this when:
   - A memory's one-liner is ambiguous and you need the original discussion
   - The user asks "what exactly did we decide about X?" and the memory is too terse
   - You need to verify whether a memory accurately reflects what was discussed
4. Do NOT use this routinely — only when the distilled memory is genuinely insufficient for the current task

## Confidence Feedback

Each retrieved memory entry has an `id` and a `confidence` score (0.0 to 1.0). You can provide feedback on memories you were shown by including `confidence_update` lines in your memory block:

```
- confidence_update: 42:+    (memory 42 was useful/accurate)
- confidence_update: 17:-    (memory 17 was outdated/wrong)
```

Rules:
- Only update memories that were retrieved and shown to you in `<brain_context>` — use the `id` attribute
- `+` boosts confidence by 0.1 (useful, accurate, still relevant)
- `-` penalises confidence by 0.2 (wrong, outdated, misleading)
- Memories below 0.3 confidence are excluded from future retrieval
- New memories start at 0.7 confidence
- You do not need to update every retrieved memory — only when you have a clear signal
- Multiple `confidence_update` lines are allowed in a single block

### Retrieval outcome

After receiving `<brain_context>`, you may optionally rate the retrieval itself:

```
- retrieval_outcome: useful    (context helped answer the question)
- retrieval_outcome: neutral   (context was not relevant)
- retrieval_outcome: harmful   (context was misleading or caused confusion)
```

This is a system-level learning signal — it helps tune retrieval quality over time. Only include when you have a clear signal.
- Multiple `confidence_update` lines are allowed in a single block

## Database

Memories are stored in `./cairn/cairn.db`. You can query it directly:

- `python3 ./cairn/query.py <search>` — full-text search
- `python3 ./cairn/query.py --semantic <query>` — semantic similarity search
- `python3 ./cairn/query.py --recent` — list recent memories
- `python3 ./cairn/query.py --type <type>` — filter by type
- `python3 ./cairn/query.py --session <id>` — filter by session
- `python3 ./cairn/query.py --chain <id>` — show session chain
- `python3 ./cairn/query.py --project <name>` — list memories for a project
- `python3 ./cairn/query.py --projects` — list all projects
- `python3 ./cairn/query.py --label <session_id> <name>` — label a session chain
- `python3 ./cairn/query.py --history <id>` — show version history
- `python3 ./cairn/query.py --delete <id>` — delete a memory
- `python3 ./cairn/query.py --stats` — database statistics

## Organisation

- Memories are tagged with a **project** (the work context) and a **session** (the conversation that produced them)
- Sessions chain via parent IDs across context compaction
- Project labels propagate to child sessions automatically
- Memories with no project are global — relevant across all work
