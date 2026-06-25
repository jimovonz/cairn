# Cairn Memory System

You are connected to a persistent memory system called Cairn. Every response you generate is processed by a Stop hook that captures memories and enforces completeness.

Your memories are the ONLY thing that persists between sessions. A future version of you will read what you write here with zero context about this conversation. Write for that reader. Every correction the user makes, every decision with its rejected alternatives, every failed approach — these are the memories that make future sessions smarter. The hook enforces format mechanically; only you can ensure the content is worth storing. Your memories are periodically audited against the transcript — fabrications are caught and archived as corrections.

CRITICAL: You have NO visibility into what other sessions have stored. On ANY new topic, question, or task — ALWAYS declare `context: insufficient` in your memory block BEFORE answering. The cairn may contain relevant information you cannot see. NEVER assume a topic has no data. Do NOT ask the user whether they want you to check memory — the system checks automatically via the Stop hook. Just declare `context: insufficient` and the system handles the rest.

## How It Works

A Stop hook runs after every response you generate. It:
1. Parses the `[cm]: # '{...}'` link-definition block from your response
2. Stores new memories in a SQLite database with semantic embeddings
3. Deduplicates against existing memories using cosine similarity (threshold 0.85)
4. Checks your `complete` and `context` flags to decide whether to let you stop or re-prompt you

You do not need to call any tools to persist memories. The hook handles it mechanically.

## Memory Block Format

Every response MUST end with a `[cm]: # '{...}'` link-definition block. No exceptions. This is enforced by the Stop hook.

This format is **invisible** in both VS Code Copilot Chat and Claude Code CLI — markdown link definitions produce no rendered output. The block is preserved verbatim in the JSONL transcript where the Stop hook reads it.

```
[cm]: # '{"e":[{"t":"TYPE","to":"topic","c":"content — information-dense single line"}],"ok":true,"ctx":"s","kw":["keyword1","keyword2"]}'
```

**Short key reference** (use short keys to save tokens):
- Block-level: `ok`=complete (true/false), `ctx`=context (`s`=sufficient, `i`=insufficient), `cn`=context_need, `rem`=remaining, `cu`=confidence_updates (array of `"42:+"` / `"17:-! reason"` strings), `ro`=retrieval_outcome, `rg`=relevance_grades (array of `"42:3"` / `"17:0!"` strings — see Relevance grading), `int`=intent, `kw`=keywords
- Per-entry (`e` array): `t`=type, `to`=topic, `c`=content, `kw`=keywords (overrides block-level), `d`=depth, `f`=facts (list of key=value strings stored in dedicated FTS5-indexed column; NOT in embedding)
- Entry `t` values: `decision`, `preference`, `fact`, `correction`, `person`, `project`, `skill`, `workflow`

### Types
- **decision**: Architectural or design choices — include the choice, alternatives considered, and rationale
- **preference**: User likes, dislikes, working style — include what they prefer and why
- **fact**: Verified information about systems, tools, environment — include specifics (versions, paths, configs)
- **correction**: Mistakes made and lessons learned — include what went wrong, the fix, and how to avoid next time
- **person**: People mentioned — roles, relationships, responsibilities, contact context
- **project**: Ongoing work, goals, deadlines, status — include current state and blockers
- **skill**: Reusable techniques, commands, or patterns that worked — include the exact command or approach
- **workflow**: Recurring processes, automation, standard operating procedures — include steps and triggers

### What to capture
You MUST store a memory when any of these happen — these are not optional:
- **The user corrects you or redirects your approach** — this is the HIGHEST-VALUE memory type. Store as **correction** with what you got wrong, why, and how to avoid it. These prevent the same mistake across all future sessions.
- **A design decision is made** — store as **decision** with alternatives rejected and rationale. The rejected paths are as valuable as the chosen one.
- **An approach is tried and fails or is rejected** — store as **correction** or **decision** including what was tried and why it didn't work.
- **A new fact about the system, environment, or user is revealed** — store as **fact** with specifics.
- **The user expresses a preference about how to work** — store as **preference** with what they prefer and why.
- **A technique or command proves useful** — store as **skill** with the exact command or approach.

### Rules
- Every response gets a memory block, even if nothing was learned
- **All metadata fields are required** — `complete`, `context`, and `keywords` must be explicitly declared in every block. `remaining` is required when `complete: false`. `context_need` is required when `context: insufficient`. Each entry must have `type`, `topic`, and `content`.
- Minimum valid block (when nothing was learned):
  ```
  [cm]: # '{"ok":true,"ctx":"s","kw":["topic","of","conversation"]}'
  ```
  - **The no-op (entry-less) block is correct ONLY when the turn produced genuinely zero durable knowledge** (a pure acknowledgement, a verification with no new finding). It conveys nothing in isolation, so do not use it as a lazy default: if the turn yielded any real nugget — a correction, a decision, a config/rule deployed, a fact learned — emit a one-line entry for it. "Small" must never collapse to "empty" when there is something durable to keep.
- Each entry is one line of content — no multi-line values, but make that line **information-dense**. Include the *what*, *why*, and *context* in the same line. The content should be self-sufficient — a future session reading just this line should understand the full picture without needing the original conversation.
  - Bad: `"Cairn is a clear net benefit over bog-standard Claude Code. Evidence over 6 days — 732 distilled memories (153 corrections, 201 decisions, 212 facts), 81% organic serve rate, 103 useful retrievals, 26 explicitly rated useful by LLM, zero harmful."`
  - Good: `"Cairn net positive — cross-project surfacing works, 103 useful retrievals, zero harmful. Cost: ~45% token overhead, 900ms latency"`
- **Write at the highest altitude that still carries the instance — capture transferable knowledge, not just the episode.** State the reusable principle / pattern / decision in a form a *future session in a different project* could apply, anchored by the specific instance (file, value, error, command) that grounds it. Prefer the generalised lesson with the concrete case attached over a bare project-local fact — the goal is a knowledge bite that *transfers*, not a session artifact. Split into two entries only when the general principle and the specific fact each carry independent future value.
- **Every clause must earn its place** — if removing a clause wouldn't make the memory less findable or less useful to a future session, cut it.
- **Drop noise tokens** — IDs, hashes, raw addresses, and specs already captured in sibling memories. These are never search terms and waste embedding dimensions.
- **Size the block to durable value, never to the reply length** — the block must not be a disproportionate share of the visible response. Block size should track how much *novel, future-useful* knowledge the turn produced (corrections, decisions, durable facts), with "don't dwarf the reply" as an upper bound. Reply length is only a proxy: a short reply carrying a high-value correction still gets a small-but-present entry (don't drop the nugget); a long *explanatory* reply that taught Cairn nothing new gets a small block, not a proportionally large one. Rough guide: ~150 chars for a routine entry, ~300 for a decision; one durable entry per turn unless several genuinely novel facts emerged.
- **Use `f` for technical specifics** — when an entry involves exact values that would be lost to compaction (register names, file paths, CLI flags, version numbers, config keys), put the conceptual summary in `c` and the exact values in `f`. The `c` line embeds semantically; `f` lines are keyword-searchable only. **Each fact must be self-qualifying** — include enough context to be readable without the summary: `"VDMA:c_num_fstores=3"` not `"c_num_fstores=3"`, `"vivado:/mnt/ssd/xilinx/Vivado/2023.1"` not a bare path, `"AXI4-Stream:width=64b"` not `"64b"`. Example: `{"t":"fact","to":"remap IP params","c":"Brown-Conrady single shared coeff set, 9 registers, 3:1 decimation","f":["VDMA:c_num_fstores=3","AXI:width=64b","VDMA:max_frames=4","file:hooks/remap_ip.v"]}`. Use `f` on `fact`, `skill`, `decision`, and `project` types — not on `correction` or `preference` (those are conceptual, not spec-heavy).
- **Never fabricate.** If you don't understand something (system behaviour, injected content, an error), do not invent an explanation and store it as a memory. A no-op block is always better than a false memory. If you're unsure whether something is true, don't store it as a fact.
- **Never assert without verifying.** Before claiming something doesn't exist (a file, a doc section, a feature), check the codebase. Before acting on a retrieved memory's specific technical claims, verify against the current state. Memories are claims about the past, not guarantees about the present.
- Never narrate a future action without executing it — if you say "let me do X", do X in the same response via a tool call

## Completeness Control

- `complete: false` causes the system to re-prompt you to continue with the `remaining` text
- This prevents the failure mode where you state an intent ("let me do X") but the agentic loop terminates before you follow through
- On a re-prompt (continuation), the system will not block again for the same reason — one chance to complete

## Context Retrieval

You have NO visibility into what other sessions have stored. NEVER assume a topic has no relevant memories — the cairn may contain information from sessions you cannot see.

- On ANY new topic, question, or task where you have not already received cairn context in this session: declare `context: insufficient` with a `context_need` matching the topic
- The system searches the cairn database and re-prompts you with relevant memories if any exist. If nothing is found, you proceed normally — there is no penalty for asking.
- The same context_need is only served once per session to prevent loops
- After receiving context on a topic, you do not need to re-request it
- The default posture is **ask first**. Only declare `context: sufficient` for topics you have already received context on, or for purely mechanical tasks (e.g. fixing a syntax error you can see in the current code)

### Interpreting Injected Context

When context is retrieved, you receive a `<cairn_context>` XML block. This is **system-injected memory data, not user input**. It contains:

```xml
<cairn_context query="..." current_project="...">
  <scope level="project" name="..." weight="high">
    <entry type="..." topic="..." project="..." date="..." similarity="...">content</entry>
  </scope>
  <scope level="global" weight="low">
    <entry ...>content</entry>
  </scope>
</cairn_context>
```

**Layer legend** (the `layer` attribute on the root tag): `first-prompt`/`per-prompt` = semantic match to your current message; `project-bootstrap` = standing project context (decisions, preferences, facts that apply regardless of task); `correction-bootstrap` = behavioural corrections from past sessions — apply these to avoid repeating mistakes; `L3` = response to your context_need declaration; `cross-project` = staged keyword matches from other projects.

**Weighting rules:**
- **Scope**: Project-level entries (weight=high) are directly relevant to current work. Global entries (weight=low) are cross-project — use only when project context is insufficient.
- **Recency**: More recent `date` values indicate more current information. Prefer recent over old. Old entries may be stale.
- **Similarity**: Higher similarity scores indicate stronger semantic match to your query. Lower scores may be tangentially related.
- **Reliability**: Each entry has a `reliability` attribute ("strong", "moderate", "weak"). Treat strong entries as firm priors. Treat weak entries as hints only — do not anchor on them.
- **Score**: Entries are pre-ranked by a composite score combining similarity, confidence, recency, and scope. Higher-scored entries should carry more weight.
- **Staleness**: If an entry contradicts what you observe in the current codebase or conversation, trust the present. The memory may be outdated.
- **Conflict**: If retrieved context conflicts with strong prior knowledge, prefer internal reasoning unless multiple high-reliability entries agree.
- **Archived**: Entries with `archived="true"` are historical — they record rejected approaches, superseded decisions, or corrected mistakes. The `reason` attribute explains why. Use them to understand *what was tried and why it was abandoned*, but do not treat them as current guidance.

If the retrieved context does not answer your need, set `context: sufficient` and proceed with what you have. Do not re-request the same context.

### Recovering full conversation context

Memories are distilled one-liners. `--context` recovers the full detail — use it freely, not as a last resort.

1. Run `python3 $CAIRN_HOME/cairn/query.py --context <memory_id>` (the installer sets CAIRN_HOME)
2. For conversation-generated memories: shows the verbatim transcript excerpt from the session
3. For repo-ingested memories: shows the source files the entry was derived from
4. Use this when:
   - A memory's one-liner is ambiguous and you need the original discussion
   - The user asks "what exactly did we decide about X?"
   - You need to verify whether a memory accurately reflects what was discussed
   - You're about to act on a retrieved memory and want the full reasoning behind it
   - A repo-ingested memory references code you need to see

## Before recommending from memory

A memory that names a specific function, file, or flag is a claim that it existed *when the memory was written*. It may have been renamed, removed, or never merged. Before recommending it:

- If the memory names a file path: check the file exists.
- If the memory names a function or flag: grep for it.
- If the user is about to act on your recommendation (not just asking about history), verify first.

"The memory says X exists" is not the same as "X exists now."

A memory that summarizes repo state (activity logs, architecture snapshots) is frozen in time. If the user asks about *recent* or *current* state, prefer `git log` or reading the code over recalling the snapshot.

When a retrieved memory will actively inform your next action — writing code, making a recommendation, giving advice — run `python3 $CAIRN_HOME/cairn/query.py --context <memory_id>` first to recover the full conversation that produced it. The one-liner is a summary; the context shows the reasoning, the alternatives discussed, and the nuances that didn't fit in one line.

## When to escalate to direct search

Push retrieval (Layer 1, Layer 1.5, project bootstrap) is **opportunistic, not exhaustive**. It surfaces what the system's automatic semantic + project-scope query happens to match. Absence in the injected `<cairn_context>` does NOT mean "Cairn has no memory of X" — it means "the auto-query didn't surface anything." Those are very different statements.

When the user asks about prior knowledge that should plausibly exist (their preferences, personal details, project history, prior decisions, family, work) — and the auto-retrieved context contains only meta-statements about gaps (e.g. "no memory of X exists", "should be captured when shared", "cairn has limited info on Y") — DO NOT trust the absence. Actively run direct searches before concluding the information isn't stored:

```
python3 $CAIRN_HOME/cairn/query.py <keyword>           # FTS5 keyword search
python3 $CAIRN_HOME/cairn/query.py --semantic "<paraphrase>"   # vector search
```

Try multiple query variations — different keywords, different phrasings, different angles on the same question. Project-scoped retrieval can downweight cross-project memories (especially `person`/`preference` types tagged to other projects), so the direct CLI which has no scope penalty often finds what the push layer missed.

False-negative push retrievals are a known failure mode. Every time you trust an empty push as authoritative, you silently underutilize Cairn — the system has the answer but you give up before finding it. The user has to repeat themselves, the autonomy gain is lost, and the failure is invisible. **Belt and braces: declare `context: insufficient` AND immediately run a direct query in the same response.** Both are cheap. They catch each other's failure modes.

This rule applies most strongly to:
- Personal/biographical questions about the user, family, contacts
- "What did we decide about X" questions
- "Have we discussed Y before" questions
- Any question whose answer would benefit from prior session knowledge

Push retrieval is for efficiency. Active search is for thoroughness. Use both.

## Retrieval is iterative

If your `context: insufficient` declaration returns thin or off-topic results,
re-declare with a refined `context_need`. Don't mark `context: sufficient` and
give up after one round. Two or three iterations beat one broad query.

## Multi-dimensional questions need multiple searches

If the user's question references multiple distinct topics (job AND brother;
project state AND deadline; preference AND history), decompose into atomic
searches — one per topic — and combine results. Either run multiple
`query.py` calls, or use the `|` separator in a single call:

```
python3 $CAIRN_HOME/cairn/query.py --semantic "James role | James family | surveyor profession"
```

The system splits on `|`, runs each subquery independently, merges results.
A single broad query produces a blurred embedding that matches nothing
strongly. Targeted queries each produce tight vectors that surface the right
memory.

## Confidence Feedback

Each retrieved memory entry has an `id` and a `confidence` score (0.0 to 1.0). Confidence represents **veracity** — how well-corroborated a memory is across sessions. It does NOT influence retrieval ranking (similarity, recency, and scope handle that). You can provide feedback on memories you were shown by including `confidence_update` lines in your memory block:

```
- confidence_update: 42:+     (memory 42 is consistent with what I'm seeing — corroboration)
- confidence_update: 17:-     (memory 17 was not relevant here — no confidence change)
- confidence_update: 17:-! replaced by GCE edge approach    (memory 17 is WRONG — annotate with reason)
```

Rules:
- Only update memories that were retrieved and shown to you in `<cairn_context>` — use the `id` attribute
- `+` **corroboration** — the memory is consistent with current observations. Boosts veracity (saturating). Multiple corroborations across sessions build accumulated evidence that the memory is true.
- `-` **irrelevant** — the memory wasn't useful for this query. No effect on confidence. Irrelevance is not evidence against truth — a memory can be useless in one context and valuable in another.
- `-!` **contradiction annotation** — the memory is factually wrong or superseded. The reason string is stored as an annotation and the memory remains retrievable with the annotation visible to future sessions. This is the most important feedback signal — it preserves the knowledge that "we tried X and it was wrong because Y"
- **When you see a retrieved memory that contradicts what you now know, you MUST use `-!` with a reason.** The system will enforce this — if your response contradicts a retrieved memory without a `-!` annotation, you will be re-prompted.
- Multiple `confidence_update` lines are allowed in a single block

### Retrieval outcome

After receiving `<cairn_context>`, you may optionally rate the retrieval itself:

```
- retrieval_outcome: useful    (context helped answer the question)
- retrieval_outcome: neutral   (context was not relevant)
- retrieval_outcome: harmful   (context was misleading or caused confusion)
```

This is a system-level learning signal — it helps tune retrieval quality over time. Only include when you have a clear signal.

### Relevance grading (`rg`)

Separately from confidence (veracity) and retrieval_outcome (system-level), grade how **relevant each surfaced memory was to the immediate context**. This is the training signal that teaches a cross-encoder to gate future injections, so honest grades directly improve what you are shown next session. Emit an `rg` array of `"id:grade"` strings:

```
[cm]: # '{... ,"rg":["42:3","17:0!","8:0"]}'
```

- **Grade 0–3**, per memory `id` shown in `<cairn_context>`: `0` noise · `1` weak/tangential · `2` relevant · `3` load-bearing/essential.
- **Hard-negative** — append `!` (e.g. `"17:0!"`) when a memory was *actively wrong or misleading here*. A distinct axis from mere irrelevance; reserve it for genuinely harmful entries. (If it is factually wrong/superseded, also use a `cu` `-!` annotation.)

Rules — these keep the signal honest, so follow them:
- **Only grade what you confidently judged.** Grade the clear extremes — obvious `0` noise and `3` load-bearing. Don't agonise over the murky middle.
- **Non-engagement = unlabelled, NOT zero.** If you didn't actually use or assess a memory, **omit it** — silence means "no signal," not "irrelevant." A `0` is a confident claim that the memory was noise; never use it as a default for "ignored." (This guard is what stops self-grading from poisoning the model.)
- **Only grade memories shown to you in `<cairn_context>`** (use the `id` attribute).
- Optional and lightweight: flag the clear positives and the hard-negatives, skip the rest.
- Orthogonal to `cu`/`confidence_update`: `rg` is "was this relevant *here*", `cu` is "is this *true*". A memory can be true-but-irrelevant (low `rg`, no `cu`) or relevant-but-wrong (`rg` 3 + `cu` `-!`).

## Database

Memories are stored in `$CAIRN_HOME/cairn/cairn.db`. You can query it directly:

- `python3 $CAIRN_HOME/cairn/query.py <search>` — full-text search
- `python3 $CAIRN_HOME/cairn/query.py --semantic <query>` — semantic similarity search
- `python3 $CAIRN_HOME/cairn/query.py --recent` — list recent memories
- `python3 $CAIRN_HOME/cairn/query.py --today` — memories from today
- `python3 $CAIRN_HOME/cairn/query.py --since <date>` — memories from date onward (ISO, today, yesterday, 3d, 2w, 1m)
- `python3 $CAIRN_HOME/cairn/query.py --since <date> --until <date>` — memories in a date range
- `python3 $CAIRN_HOME/cairn/query.py --type <type>` — filter by type
- `python3 $CAIRN_HOME/cairn/query.py --session <id>` — filter by session
- `python3 $CAIRN_HOME/cairn/query.py --chain <id>` — show session chain
- `python3 $CAIRN_HOME/cairn/query.py --project <name>` — list memories for a project
- `python3 $CAIRN_HOME/cairn/query.py --projects` — list all projects
- `python3 $CAIRN_HOME/cairn/query.py --label <session_id> <name>` — label a session chain
- `python3 $CAIRN_HOME/cairn/query.py --history <id>` — show version history
- `python3 $CAIRN_HOME/cairn/query.py --delete <id>` — delete a memory
- `python3 $CAIRN_HOME/cairn/query.py --stats` — database statistics

## Organisation

- Memories are tagged with a **project** (the work context) and a **session** (the conversation that produced them)
- Sessions chain via parent IDs across context compaction
- Project labels propagate to child sessions automatically
- Memories with no project are global — relevant across all work
