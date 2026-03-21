# Cairn — Architecture & Technical Reference

## Overview

Cairn is a persistent memory system for Claude Code. It gives the LLM long-term memory across sessions by embedding structured metadata into every response, capturing it via Claude Code hooks, and storing it in a local SQLite database with semantic search.

The key innovation is that every LLM response contains invisible structured metadata — self-assessed by the LLM, mechanically enforced by the system, and never shown to the user. This metadata serves three functions simultaneously:

1. **Memory persistence** — distilled facts, decisions, and preferences are extracted and stored
2. **Loop continuation control** — the LLM declares whether its response is complete
3. **Context retrieval** — the LLM declares whether it needs information from past sessions
4. **Confidence feedback** — the LLM rates the usefulness of retrieved memories, dynamically adjusting their future retrieval priority

## The Invisible Metadata Mechanism

### How it works

Every LLM response ends with a `<memory>` block using XML-style angle bracket tags. These tags are **invisible to the user** — Claude Code's rendering strips them from the displayed output. However, they are preserved in:

- The `last_assistant_message` field passed to Stop hooks
- The raw JSONL transcript file on disk

This means the user sees a clean response while the system infrastructure has full access to the structured data.

### Memory block format

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
- retrieval_outcome: [useful|neutral|harmful]
- keywords: [comma-separated topic keywords for cross-project discovery]
- source_messages: [start-end message range where this knowledge was discussed, e.g. 5-12]
</memory>
```

A single block can contain multiple entries (each with type/topic/content), control fields (complete, remaining, context, context_need), and confidence feedback on previously retrieved memories. All fields except `complete` are optional.

### Why the LLM produces the metadata

The LLM is instructed via `CLAUDE.md` and `.claude/rules/memory-system.md` (both auto-loaded into every conversation) to include the block on every response. A Stop hook enforces this mechanically — if the block is missing, the hook blocks the response and re-prompts the LLM to add one.

This creates a self-reinforcing loop: the instruction tells the LLM to do it, and the hook ensures it actually happens even if the LLM forgets.

## System Architecture

### Data flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER PROMPT                               │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│              UserPromptSubmit Hook (prompt_hook.py)               │
│                                                                  │
│  Layer 1: First prompt? ──yes──▶ Search brain ──▶ Inject context │
│  Layer 2: Staged data?  ──yes──▶ Inject cross-project context    │
│                                                                  │
│  Injects via additionalContext (invisible to user)               │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LLM GENERATES RESPONSE                        │
│                                                                  │
│  User-visible text + invisible <memory> block                    │
│  ┌────────────────────────────────────────────────┐              │
│  │ <memory>                                       │  ◄── hidden  │
│  │ - type: decision                               │     from     │
│  │ - topic: auth-approach                         │     user     │
│  │ - content: Use JWT for stateless auth          │              │
│  │ - keywords: authentication, JWT                │              │
│  │ - confidence_update: 42:+                      │              │
│  │ - context: sufficient                          │              │
│  │ - complete: true                               │              │
│  │ </memory>                                      │              │
│  └────────────────────────────────────────────────┘              │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Stop Hook (stop_hook.py)                         │
│                                                                  │
│  1. Register session ──▶ Auto-label project from cwd             │
│  2. Parse <memory> block                                         │
│  3. Apply confidence updates (+0.1 saturating / -0.2 scaled)     │
│  4. Store memories ──▶ Embed ──▶ Dedup ──▶ SQLite + vec index    │
│  5. Layer 2: Extract keywords ──▶ Search global ──▶ Stage        │
│  6. Evaluate control flags:                                      │
│     ├─ No block     → BLOCK, re-prompt "add memory block"        │
│     ├─ incomplete   → BLOCK, re-prompt with remaining text       │
│     ├─ need context → BLOCK, search + inject, re-prompt          │
│     └─ complete     → ALLOW STOP                                 │
│                                                                  │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   USER SEES CLEAN RESPONSE                       │
└──────────────────────────────────────────────────────────────────┘
```

### Three retrieval layers

```
┌─────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYERS                      │
│                                                         │
│  Layer 1: FIRST-PROMPT PUSH                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │ When: First message of session                    │  │
│  │ How:  Embed user message → search brain           │  │
│  │ Why:  Eliminate "I don't know" cold start         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Layer 2: KEYWORD CROSS-PROJECT                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ When: Between turns (stop hook → next prompt)     │  │
│  │ How:  Extract keywords → search global → stage    │  │
│  │ Why:  Surface knowledge LLM doesn't know to ask   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Layer 3: PULL-BASED                                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │ When: LLM declares context: insufficient          │  │
│  │ How:  Search brain → inject → re-prompt           │  │
│  │ Why:  Handle explicit gaps mid-conversation       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Confidence lifecycle

```
  NEW MEMORY
      │
      ▼
   ┌─────┐
   │ 0.7 │  ◄── default
   └──┬──┘
      │
      ├── Retrieved + LLM rates "+" ──▶ +0.1 × (1 - conf)  [saturating]
      │                                    0.7 → 0.73
      │                                    0.9 → 0.91
      │
      ├── Retrieved + LLM rates "-" ──▶ -0.2 × (1 + conf)  [amplified]
      │                                    0.7 → 0.36
      │                                    0.9 → 0.52
      │
      ├── Same type+topic, new content ──▶ old → 0.2 (suppressed)
      │                                    new → 0.7 (fresh)
      │
      ├── Negation mismatch detected ──▶ both -0.1
      │
      └── Never retrieved ──▶ NO CHANGE (no passive decay)

  RETRIEVAL INCLUSION:
      similarity ≥ 0.6  → included regardless of confidence
      confidence ≥ 0.3  → included at normal threshold
      confidence < 0.3  → only if similarity override
```

### Quality gate pipeline

```
  RAW CANDIDATES (from sqlite-vec or brute-force)
      │
      ▼
  ┌─ Low-info pre-filter ──────────┐
  │  context_need < 8 chars?       │──yes──▶ SKIP RETRIEVAL
  │  All stopwords?                │
  └────────────┬───────────────────┘
               │ no
               ▼
  ┌─ Soft confidence inclusion ────┐
  │  Keep if sim ≥ 0.6             │
  │  OR confidence ≥ 0.3           │──fail──▶ DROP
  └────────────┬───────────────────┘
               │ pass
               ▼
  ┌─ Garbage gate ─────────────────┐
  │  Best similarity < 0.35?      │──yes──▶ RETURN NOTHING
  └────────────┬───────────────────┘
               │ no
               ▼
  ┌─ Borderline gate ──────────────┐
  │  Best sim < 0.45               │
  │  AND best score < 0.50?        │──yes──▶ RETURN NOTHING
  └────────────┬───────────────────┘
               │ no
               ▼
  ┌─ Adaptive threshold ───────────┐
  │  Recent harmful/neutral > 30%? │──yes──▶ Boost thresholds +0.05-0.10
  └────────────┬───────────────────┘
               │
               ▼
  ┌─ Relative filter ──────────────┐
  │  sim < 0.7 × max_sim?         │──yes──▶ DROP
  └────────────┬───────────────────┘
               │ pass
               ▼
  ┌─ Dominance suppression ────────┐
  │  top1 - top2 < 0.05?          │──yes──▶ Include both
  └────────────┬───────────────────┘
               │
               ▼
  ┌─ Weak-entry suppression ───────┐
  │  Top score < 0.4?             │──yes──▶ RETURN NOTHING
  └────────────┬───────────────────┘
               │ pass
               ▼
  ┌─ Hard cap ─────────────────────┐
  │  Limit to top 5                │
  └────────────┬───────────────────┘
               │
               ▼
         INJECT INTO LLM
```

### Legacy text description

Three retrieval layers operate at different points in the conversation lifecycle:

```
User prompt arrives
  → Layer 1 (UserPromptSubmit hook): first prompt of session
    → Searches brain using user's message
    → Injects relevant project + global context via additionalContext
  → Layer 2 (UserPromptSubmit hook): subsequent prompts
    → Injects cross-project context staged by previous stop hook keyword search
  → LLM generates response + <memory> block (invisible to user)
  → Stop hook fires
    → Parses <memory> block (entries, control flags, keywords, confidence updates)
    → Stores memories in SQLite with embeddings
    → Layer 2 staging: searches global memories using keywords, stages for next prompt
    → Layer 3 (pull-based): if context: insufficient, searches brain and re-prompts
    → Evaluates control flags:
        ├─ No <memory> block     → blocks stop, re-prompts with format hint
        ├─ complete: false       → blocks stop, re-prompts with remaining text
        ├─ context: insufficient → searches brain, blocks stop, injects results
        └─ complete: true        → allows stop, turn ends
  → User sees clean response
```

## Components

### File structure

```
cairn/
├── CLAUDE.md                          # Concise LLM instructions (auto-loaded)
├── ARCHITECTURE.md                    # This file
├── .claude/
│   ├── settings.json                  # Hook registration
│   └── rules/
│       └── memory-system.md           # Full system docs for the LLM (auto-loaded)
├── cairn/
│   ├── cairn.db                   # SQLite database (WAL mode for concurrent access)
│   ├── db.py                          # Shared DB connection helper (WAL + busy timeout)
│   ├── config.py                      # All tunable parameters (thresholds, weights, limits)
│   ├── init_db.py                     # Schema and migrations
│   ├── query.py                       # CLI query tool
│   ├── embeddings.py                  # Sentence-transformers wrapper (daemon-aware, composite scoring)
│   ├── daemon.py                      # Background embedding server (Unix socket)
│   └── hook.log                       # Debug log
├── hooks/
│   └── stop_hook.py                   # Main hook — capture, enforce, retrieve, confidence
└── .venv/                             # Python venv with sentence-transformers
```

### Database schema

**memories** — the core table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| type | TEXT | Memory type (decision, fact, preference, etc.) |
| topic | TEXT | Short key for exact-match lookups |
| content | TEXT | One-line distilled fact |
| embedding | BLOB | 384-dim float32 vector from all-MiniLM-L6-v2 |
| session_id | TEXT | Claude Code session UUID that produced this memory |
| project | TEXT | Project label (NULL = global) |
| confidence | REAL | Dynamic confidence score 0.0–1.0 (default 0.7) |
| source_start | INTEGER | LLM-estimated conversation turn where this knowledge originated (start) |
| source_end | INTEGER | LLM-estimated conversation turn where this knowledge originated (end) |
| created_at | TIMESTAMP | First created |
| updated_at | TIMESTAMP | Last modified |

Indexes: `type`, `topic`, `project`. FTS5 virtual table on `topic` + `content`. `memories_vec` virtual table (via sqlite-vec) for indexed vector KNN search.

**memory_history** — version tracking

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| memory_id | INTEGER | FK to memories.id |
| content | TEXT | Previous content before update |
| session_id | TEXT | Session that held the old version |
| changed_at | TIMESTAMP | When the old version was replaced |

A `BEFORE UPDATE OF content` trigger on `memories` automatically snapshots the old content into `memory_history` before any update.

**sessions** — conversation tracking

| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT | Primary key — Claude Code session UUID |
| parent_session_id | TEXT | FK to parent session (set after context compaction) |
| project | TEXT | Project label (inherited from parent) |
| transcript_path | TEXT | Path to the JSONL transcript file |
| started_at | TIMESTAMP | When session was first registered |

Sessions chain via `parent_session_id`. When Claude Code compacts context and creates a new session, the hook detects the parent by comparing the first transcript entry's `sessionId` against the current `session_id`.

**metrics** — performance monitoring

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| event | TEXT | Metric name |
| session_id | TEXT | Session that generated this metric |
| detail | TEXT | Additional context |
| value | REAL | Numeric value (count, milliseconds, etc.) |
| created_at | TIMESTAMP | When recorded |

Tracked events: `hook_fired`, `memories_stored`, `missing_memory_block`, `malformed_memory_block`, `context_requested`, `context_served`, `context_empty`, `context_cache_hit`, `context_retrieval`, `retrieval_latency_ms`, `confidence_updates`, `continuation_cap_hit`, `completeness_cap_hit`, `hook_crash`.

## Concurrency

The database uses SQLite in WAL (Write-Ahead Logging) mode with a 5-second busy timeout. This is essential because Cairn operates across concurrent Claude Code sessions, cron jobs, and external integrations (e.g. Telegram).

**WAL mode** allows multiple concurrent readers with one writer. Writers queue rather than failing immediately. This is set once by `init_db.py` and persists in the database file.

**Busy timeout** (5000ms) is set on every connection. If a writer holds the lock, other writers wait up to 5 seconds before failing. Since individual write operations are small (~50ms for an INSERT + embedding), two sessions would need to complete within the same 50ms window to even queue — effectively impossible to deadlock at normal usage.

All connection points (stop hook, prompt hook, query CLI, daemon) apply the busy timeout via `PRAGMA busy_timeout`. The `cairn/db.py` module provides a shared `connect()` helper that applies both WAL and timeout automatically.

**Scaling limit**: WAL SQLite handles tens of concurrent sessions comfortably. At hundreds of concurrent writers, Postgres with connection pooling would be the upgrade path.

## Stop Hook — the core mechanism

`hooks/stop_hook.py` is a Claude Code Stop hook registered in `.claude/settings.json`. It runs as a shell command after every LLM response, receiving JSON on stdin with:

- `session_id` — current session UUID
- `transcript_path` — path to conversation JSONL
- `last_assistant_message` — the LLM's text output (including `<memory>` tags)
- `stop_hook_active` — boolean, true if this is a continuation after a previous block
- `cwd` — current working directory (used for automatic project labelling)

### Processing pipeline

1. **Session registration** — records the session in the `sessions` table, extracts parent session ID from transcript, inherits project label from parent
2. **Automatic project labelling** — if the session has no project label, derives one from the working directory basename (e.g. `/home/user/Projects/cairn` → "cairn")
3. **Continuation cap check** — if this is a re-prompted continuation, checks a per-session counter. After 3 consecutive re-prompts, forces a stop to prevent runaway loops
4. **Memory block parsing** — robust parser extracts the last `<memory>...</memory>` block (handles unclosed tags, unknown fields, unknown types). Parses key-value pairs for entries, control fields, and confidence updates. If the tag is present but content is unparseable, the hook provides specific syntax correction feedback (including a format example) on re-prompt to help the LLM fix the structure on the next turn
5. **Confidence updates** — applies `+` (boost 0.1) or `-` (penalise 0.2) adjustments to memories the LLM was shown and rated
6. **Memory persistence** — for each entry:
   - Generates a 384-dim embedding via `all-MiniLM-L6-v2` (daemon if available, auto-started on first use, direct model fallback)
   - Checks for near-identical duplicates (cosine similarity > 0.95) — updates if found
   - Falls back to exact type+topic match — if content differs, this is a **contradiction**: the old entry's confidence is dropped to 0.2 (below retrieval threshold) before overwriting. The old content is preserved in `memory_history` via the BEFORE UPDATE trigger. The new entry gets fresh default confidence (0.7)
   - Otherwise inserts new row
   - Tags with session_id, project, and default confidence (0.7)
7. **Context retrieval** (if `context: insufficient`):
   - Checks context cache — same `context_need` is only served once per session
   - Searches project-scoped memories (similarity > 0.25, up to 7 results, confidence > 0.3)
   - Searches global memories (similarity > 0.50 when project data exists, > 0.25 otherwise, up to 5 results, confidence > 0.3)
   - Falls back to FTS5 keyword search if no embedding results
   - Formats results as structured `<brain_context>` XML with scope, weight, date, similarity, id, and confidence attributes
   - Blocks the stop and injects context with a reference to weighting rules in `.claude/rules/memory-system.md`
8. **Completeness check** (if `complete: false`) — blocks stop and re-prompts with `remaining` text (subject to continuation cap)
9. **Metrics recording** — logs event counts, retrieval latency, cache hits, confidence updates, cap hits

### Error handling

The entire `main()` function is wrapped in a try/except that **fails open** — if the hook crashes for any reason, it exits with code 0 (allow stop) rather than blocking the user. Crashes are logged to both `hook.log` and the `metrics` table as `hook_crash` events.

### Loop protection

Four mechanisms prevent infinite re-prompting:

1. **Continuation cap** — a hard limit of 3 consecutive re-prompts per session. After 3 blocks (any combination of missing block, incomplete, or context injection), the hook forces a stop. The counter resets when a response completes normally.
2. **`stop_hook_active` flag** — on continuation responses, the hook still parses and stores memories but is more conservative about blocking again (only blocks for missing memory block on the first continuation).
3. **Context cache** — a JSON file (`.context_cache`) tracks which `context_need` strings have been served per session. Duplicate requests are silently skipped.
4. **Fallback instruction** — injected context includes "if this context is not sufficient, declare context: sufficient and work with what you have."

## Embedding and Deduplication

### Embedding model

`all-MiniLM-L6-v2` via sentence-transformers, running locally in a Python venv. Produces 384-dimensional normalised float32 vectors.

The embedding layer supports two modes:
- **Daemon mode** — a background process (`cairn/daemon.py`) keeps the model resident in RAM, accepting requests over a Unix socket. Eliminates cold start latency. Start with `python3 cairn/daemon.py start`.
- **Direct mode** — falls back to loading the model in-process if the daemon isn't running. ~3 second cold start on first embedding per hook invocation.

`embeddings.py` transparently tries the daemon first (auto-starting it on first failure), falling back to direct loading.

### Vector search

Similarity search uses `sqlite-vec` — a native SQLite extension providing indexed vector search via a `vec0` virtual table. The `memories_vec` table mirrors the `memories` table's embeddings and is kept in sync on every insert/update.

- **sqlite-vec available** — indexed KNN search using L2 distance, converted to cosine similarity for normalised vectors (`cosine_sim = 1 - L2² / 2`). Constant-time regardless of table size.
- **sqlite-vec unavailable** — transparent fallback to brute-force scan of all embeddings in Python. Linear in table size.

Both paths produce identical results. The fallback ensures the system works without the extension installed.

### Embedding strategy

The text embedded is `"{project} {type} {topic} {content}"` — the project name is prepended to push unrelated domains apart in vector space. This reduces cross-project bleed without changing schema or queries. Memories from "webshop" and "cairn" with similar content will have lower cosine similarity than same-project memories, naturally scoping retrieval.

### Deduplication

On every insert, the new memory's embedding is checked against existing entries via `find_nearest()` using the sqlite-vec index (with brute-force fallback).

- **Cosine similarity > 0.95** → near-identical wording, update existing row (prevents literal repetition)
- **Cosine similarity 0.6–0.95 with negation mismatch** → **soft contradiction detected** — the existing entry's confidence is reduced by 0.1. Neither is overwritten. This preserves ambiguity where it exists without requiring an NLI classifier. Negation detection uses a lightweight heuristic checking for words like "not", "never", "instead of", "replaced", etc.
- **Same type + topic, same content** → exact duplicate, update timestamp only
- **Same type + topic, different content** → **hard contradiction detected** — old entry's confidence is dropped to 0.2, then overwritten with new content at fresh 0.7 confidence. The old content is preserved in `memory_history`.
- **Neither** → insert new row

The dedup threshold was deliberately set high (0.95) to avoid accidentally overwriting distinct memories that happen to be semantically similar. Two genuinely different facts about the same domain will both be kept.

The `BEFORE UPDATE` trigger on `memories` ensures the old content is preserved in `memory_history` before any overwrite.

## Confidence System

Every memory has a `confidence` score (0.0 to 1.0) that dynamically adjusts based on LLM feedback.

### Lifecycle

1. **New memories** start at **0.7** confidence
2. When memories are retrieved and shown to the LLM via `<brain_context>`, each entry includes its `id`, `confidence`, and composite `score`
3. The LLM can provide per-memory feedback: `confidence_update: 42:+` or `confidence_update: 17:-`
4. The hook applies **saturating adjustments**:
   - `+` → `confidence += 0.1 × (1 - confidence)` — diminishing returns as confidence approaches 1.0. At 0.7: +0.03. At 0.9: +0.01.
   - `-` → `confidence -= 0.2 × (1 + confidence)` — overconfident entries fall harder. At 0.7: -0.34. At 0.9: -0.38.
5. **Saturating model prevents inflation** — the system is reluctant to become absolutely certain. Reaching 0.95 requires many more positive signals than reaching 0.7. But a single negative signal at 0.95 drops to 0.56. This prevents the dominant failure mode of feedback systems: runaway confidence on frequently retrieved entries.
6. **No passive decay** — confidence only changes via explicit LLM feedback. Important but infrequently accessed memories retain their confidence indefinitely. This is deliberate: the system should never forget high-value knowledge simply because it hasn't been asked about recently.
6. **Soft inclusion** — low-confidence entries are ranked lower via composite scoring but not hard-excluded. Entries are included if `similarity >= 0.6 OR confidence >= 0.3`.
7. **Negation dampening** — on insert, if a semantically similar entry (0.6+ similarity) has a negation mismatch (e.g. "should" vs "should not"), its confidence is reduced by 0.1. This preserves uncertainty without requiring an NLI classifier.
8. The LLM is not required to rate every retrieved memory — only when there's a clear signal

### Retrieval outcome feedback

In addition to per-memory confidence updates, the LLM can rate the retrieval itself:

```
- retrieval_outcome: useful    (context helped answer the question)
- retrieval_outcome: neutral   (context was not relevant)
- retrieval_outcome: harmful   (context was misleading or caused confusion)
```

This is a system-level learning signal recorded in the metrics table, distinct from per-memory confidence. It enables tracking retrieval quality over time and can be used to detect degradation patterns (e.g. a rising proportion of "neutral" or "harmful" outcomes).

### Why this works

The LLM is the only entity with enough context to judge whether a retrieved memory was helpful. The user doesn't see the retrieved context (it's injected into the hook's re-prompt), so user feedback isn't practical. The directional signal (`+`/`-`) is simple enough that the LLM can provide it reliably without complex reasoning.

Over time, frequently useful memories rise toward 1.0 while misleading or stale memories are deprioritised by composite scoring — they remain in the database but effectively stop influencing the LLM.

## Context Retrieval — Three-Layer Design

Three retrieval layers operate at different points, each covering a different blind spot:

| Layer | When | Trigger | What it finds | Hook |
|-------|------|---------|---------------|------|
| **1. First-prompt push** | First message of session | User submits prompt | Project context for the opening question | UserPromptSubmit |
| **2. Keyword cross-project** | Between turns | LLM outputs `keywords:` in memory block | Global knowledge surfaced by topic overlap from other projects | Stop (stages) → UserPromptSubmit (injects) |
| **3. Pull-based** | Any time LLM recognises a gap | LLM declares `context: insufficient` | Specific missing context the LLM explicitly requests | Stop |

### Layer 1: First-prompt push

On the first message of each session, the UserPromptSubmit hook (`prompt_hook.py`) embeds the user's message and searches the brain. Results are injected via `additionalContext` before the LLM starts generating. This eliminates the cold-start problem where the LLM answers "I don't know" before the Stop hook can correct it.

Uses `L1_SIM_THRESHOLD` and `L1_MAX_RESULTS` from config. Only fires once per session (tracked via `.first_prompt_done` file).

### Layer 2: Keyword cross-project

The LLM can include `keywords: authentication, JWT, session tokens` in its memory block. The Stop hook extracts these keywords and searches global memories (excluding the current project) for strong matches (`L2_SIM_THRESHOLD >= 0.60`). Results are staged in `.staged_context` and injected on the next prompt via the UserPromptSubmit hook.

This surfaces cross-project knowledge the LLM doesn't know to ask for. It only fires for cross-project results (the current project's memories are already in the LLM's context via Layer 1 or 3).

### Layer 3: Pull-based

The LLM explicitly requests context by declaring `context: insufficient` with a `context_need`. The Stop hook searches the brain, applies all quality gates, and re-prompts the LLM with results. This is the fallback for when the LLM identifies a specific gap mid-conversation.

Pull-based retrieval was retained because:

### Retrieval posture

The LLM is instructed to **always** declare `context: insufficient` on any new topic, question, or task where it hasn't already received brain context in the current session. This is the default posture — the brain decides whether there's relevant data, not the LLM. The LLM has no visibility into what other sessions stored.

Critically, the LLM must **never ask the user** whether to check memory. The Stop hook handles retrieval automatically and transparently. From the user's perspective, the LLM simply knows things from past sessions.

### Composite scoring

Results are pre-ranked hook-side using a single composite score rather than passing multiple signals for the LLM to combine:

```
score = 0.50 * similarity + 0.30 * confidence + 0.15 * recency_decay + 0.05 * scope_weight
```

Where:
- `similarity` — cosine similarity between query and memory embeddings
- `confidence` — dynamic confidence score (0.0–1.0)
- `recency_decay` — exponential decay with 30-day half-life (`e^(-0.693 * age_days / 30)`)
- `scope_weight` — 1.0 for current project, 0.3 for global

All weights are configurable in `cairn/config.py`.

### Quality gates

Before injection, results pass through multiple quality filters (all configurable):

| Gate | Default | Effect |
|------|---------|--------|
| **Low-info pre-filter** | context_need < 8 chars or all stopwords → skip | Prevents embedding generic queries like "help", "continue" |
| **Garbage gate** | max_similarity < 0.35 → reject all | Prevents injection of irrelevant context |
| **Borderline gate** | max_similarity < 0.45 AND top_score < 0.50 → reject | Eliminates weak-but-coherent matches that pass the garbage gate |
| **Adaptive threshold** | +0.05–0.10 boost if recent retrieval outcomes are poor | Self-tightening based on harmful/neutral rate over last 7 days |
| **Relative filter** | similarity < 0.7 × max_similarity → drop | Removes tail noise, keeps only the locally relevant cluster |
| **Soft confidence** | include if similarity ≥ 0.6 OR confidence ≥ 0.3 | High similarity overrides low confidence; prevents blind spots |
| **Dominance suppression** | if top1 - top2 < 0.05 → include both | Prevents false certainty from weak leaders |
| **Weak-entry suppression** | top result score < 0.4 → don't inject | Prevents single weak matches from biasing answers |
| **Hard cap** | max 5 entries | Bounds context window cost |

### Injected context format

```xml
<brain_context query="what decisions were made about X" current_project="Cairn">
  <scope level="project" name="Cairn" weight="high">
    <entry id="7" type="decision" topic="no-mcp" project="Cairn"
           date="2026-03-20 20:19:41" confidence="0.80" score="0.74"
           recency_days="0" reliability="strong" similarity="0.52">
      MCP unnecessary for Claude Code — direct filesystem and bash access to SQLite is more efficient
    </entry>
  </scope>
  <scope level="global" weight="low">
    <entry id="42" type="preference" topic="code-style" project="Other Project"
           date="2026-03-15 10:00:00" confidence="0.70" score="0.58"
           recency_days="5" reliability="moderate" similarity="0.51">
      User prefers snake_case in Python
    </entry>
  </scope>
</brain_context>
```

Each entry includes epistemic qualifiers to help the LLM calibrate trust:
- **`score`** — pre-computed composite rank (higher = more relevant overall)
- **`reliability`** — "strong" (score ≥ 0.6), "moderate" (≥ 0.4), or "weak" (< 0.4)
- **`recency_days`** — days since last update
- **`confidence`** — dynamic confidence score
- **`similarity`** — raw cosine similarity to the query

The LLM interprets these per `.claude/rules/memory-system.md`:
- Treat **strong reliability** entries as firm priors
- Treat **weak reliability** entries as hints only — do not anchor on them
- If retrieved context conflicts with strong prior knowledge, prefer internal reasoning unless multiple high-reliability entries agree

After using the context, the LLM can provide per-memory feedback (`confidence_update: 7:+`) and system-level feedback (`retrieval_outcome: useful|neutral|harmful`).

### Retrieval thresholds

All thresholds are configurable in `cairn/config.py`.

| Scenario | Project threshold | Global threshold | Rationale |
|----------|------------------|-----------------|-----------|
| Session has a project label | 0.25 | 0.50 | Include broad project matches, strict filter on global noise |
| Session has no project label | N/A | 0.25 | No project to prioritise, relax global threshold |

## Project and Session Organisation

### Projects

A project is a label applied to a session chain. All memories produced by sessions in that chain inherit the project label. This enables:

- `query.py --project "Cairn"` — all memories for a project, across all sessions
- Retrieval scoped to the current project's context
- Cross-project queries at a global level

Projects are labelled automatically from the working directory basename when a session is first registered (e.g. `/home/user/Projects/cairn` → "cairn"). They can also be labelled or overridden manually via `query.py --label <session_id> <project_name>`. Child sessions inherit the label automatically.

### Session chains

When Claude Code compacts context (near the context window limit), it may create a new session that continues the conversation. The stop hook detects this by comparing the `sessionId` in the first transcript entry against the current hook input's `session_id`. If they differ, the transcript's `sessionId` is recorded as the parent.

This creates a chain: Session A → Session B → Session C, all representing the same line of work. The project label propagates down the chain.

## Transcript Recovery

Memories are distilled one-liners — useful for retrieval but lossy. When the full detail behind a memory is needed, the system can recover the original conversation context from the session transcript.

### How it works

1. The LLM estimates which conversation turns produced each memory using `source_messages: start-end` in the memory block
2. The hook stores these as `source_start` and `source_end` on the memory row
3. The sessions table stores `transcript_path` — the path to the JSONL transcript file
4. `query.py --context <id>` reads the transcript and shows the conversation excerpt around the source range

### Turn counting

Transcripts contain many entries per conversational exchange (tool calls, tool results, system messages). The context viewer counts only **text-bearing messages** (entries with actual text content), skipping empty tool-use entries. This aligns with the LLM's perception of "message N" — it thinks in conversational turns, not JSONL entries.

For example, a 2000-entry transcript might contain only ~300 text-bearing turns. The LLM's estimate of "messages 42-55" maps to turn 42-55 in this filtered count.

### Accuracy

The LLM's source estimates are approximate. It has a sense of "this was discussed recently" vs "this was discussed early on" but cannot count precise message indices. The `--verify-sources` command measures drift between estimated and actual locations by keyword-matching memory content against the transcript.

### When the LLM uses this

Retrieved memory entries include a `has_context="true"` attribute when source range data is available. The LLM is instructed to use `--context <id>` when:

- A memory's one-liner is ambiguous and the original discussion is needed
- The user asks "what exactly did we decide about X?"
- The LLM needs to verify whether a memory accurately reflects what was discussed

It is not used routinely — only when the distilled memory is genuinely insufficient.

## Configuration

### Hook registration (`.claude/settings.json`)

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/.venv/bin/python3 /path/to/hooks/stop_hook.py",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

The hook is registered **globally** in `~/.claude/settings.json` so it fires in every Claude Code session regardless of working directory. This is essential for cross-project memory — every session in every directory captures and retrieves from the same brain. The project-local `.claude/settings.json` can add project-specific hook configuration if needed. Changes require a session restart — hooks are cached at session start.

### LLM instructions

Instructions are deployed at two levels:

**Global** (applies to all sessions in any directory):
- **`~/.claude/CLAUDE.md`** — concise format, types, rules, and the critical context retrieval posture instruction
- **`~/.claude/rules/memory-system.md`** — full mechanism explanation, weighting rules, confidence feedback, database query reference

**Project-local** (adds project-specific detail):
- **`<project>/CLAUDE.md`** — project-specific instructions that supplement the global ones
- **`<project>/.claude/rules/memory-system.md`** — can override or extend the global rules

The global CLAUDE.md opens with the most critical instruction prominently placed at the top: the LLM must ALWAYS declare `context: insufficient` on any new topic before answering, and must NEVER ask the user whether to check memory — the Stop hook handles retrieval automatically and transparently.

## Query Interface

`cairn/query.py` provides a CLI for both the user and the LLM:

| Command | Description |
|---------|-------------|
| `query.py <term>` | Full-text search via FTS5 |
| `query.py --semantic <query>` | Semantic similarity search via embeddings |
| `query.py --recent` | List recent memories |
| `query.py --type <type>` | Filter by memory type |
| `query.py --session <id>` | List memories from a session |
| `query.py --chain <id>` | Show session chain (parent/child links) |
| `query.py --project <name>` | List all memories for a project |
| `query.py --projects` | List all known projects |
| `query.py --label <id> <name>` | Label a session chain as a project |
| `query.py --context <id>` | Show conversation context around where a memory was recorded |
| `query.py --history <id>` | Show version history for a memory |
| `query.py --delete <id>` | Delete a memory and its history |
| `query.py --compact [project]` | Dense brain dump for LLM ingestion at session start |
| `query.py --review` | Surface suppressed and uncertain-confidence memories for inspection |
| `query.py --verify-sources` | Analyse accuracy of LLM source_messages estimates |
| `query.py --stats` | Database statistics, performance metrics, and retrieval latency |

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Memory block parsing | <1ms | Regex on last_assistant_message |
| Embedding generation | ~50ms | After model is loaded |
| Model cold start | ~3s | First embedding per hook invocation (eliminated by daemon) |
| sqlite-vec indexed search | ~14ms | Constant-time KNN via vec0 virtual table |
| Brute-force similarity search | ~1ms at 50 memories | Fallback if sqlite-vec unavailable; scales linearly |
| FTS5 search | <1ms | Keyword-based fallback |
| Full hook execution (no daemon) | ~4s cold / ~200ms warm | Dominated by model loading |
| Full hook execution (daemon) | ~200ms | Model already resident in daemon |

## Design Decisions and Rationale

### Why not MCP?
Claude Code has direct filesystem and bash access. MCP adds a protocol layer, a server process, and serialisation overhead for capabilities already available natively. MCP is valuable when multiple tools (Claude.ai, Cursor, ChatGPT) need shared access — for Claude Code alone, direct SQLite access is simpler and faster.

### Why a Stop hook instead of a UserPromptSubmit hook?
Push-based retrieval (injecting context on every prompt) wastes tokens on irrelevant context. Pull-based retrieval (LLM requests what it needs) is more efficient because the LLM can assess what's missing from its current context. The Stop hook serves both purposes: it captures memories (write) and retrieves context (read) in one mechanism.

### Why invisible tags?
The `<memory>` tags are invisible to the user because angle bracket tags are stripped from Claude Code's rendered output. This is a feature: the user sees a clean response, while the hook infrastructure sees the full structured metadata. There is no need for visible markers or separate output channels.

### Why LLM self-assessment for completeness?
The Claude Code agentic loop terminates when the LLM response contains no tool calls. If the LLM states an intent ("let me do X") in text but doesn't make a tool call, the loop terminates prematurely. The `complete: false` flag solves this by giving the LLM a way to signal "I'm not done" that the hook can enforce mechanically.

### Why local embeddings?
Using `all-MiniLM-L6-v2` locally avoids external API dependencies, network latency, and ongoing costs. The model is small (80MB), fast enough for per-response embedding, and produces quality embeddings for short text. The tradeoff is a ~3 second cold start per hook invocation.

### Why tiered retrieval thresholds?
When project-level data exists, global results should be held to a higher bar to avoid noise from unrelated projects. When no project data exists, the global threshold is relaxed to maximise useful results. The LLM has weighting instructions to further discriminate.

### Why dynamic confidence instead of manual curation?
The user never sees retrieved context (it's injected via the hook's re-prompt), so user feedback on memory quality isn't practical. The LLM is the only entity with enough context to judge whether a retrieved memory was helpful. Directional signals (`+`/`-`) are simple enough for reliable LLM output.

### Why saturating confidence instead of linear or decaying?
Linear `+0.1` adjustments lead to runaway confidence inflation — frequently retrieved memories saturate at 1.0 and become entrenched priors even when outdated. Passive time decay causes the opposite problem: important but infrequently accessed knowledge silently fades. The saturating model (`boost × (1 - confidence)`, `penalty × (1 + confidence)`) avoids both failure modes: it's reluctant to become certain, quick to lose certainty when challenged, and never forgets passively. A memory at 0.9 needs ~10 positive signals to reach 0.95, but a single negative drops it to 0.52.

### Why a conservative dedup threshold (0.95)?
An earlier design used 0.85, which risked overwriting distinct memories with similar wording. Two different decisions about the same domain (e.g. Python naming vs JavaScript naming) could be falsely merged. The 0.95 threshold only catches near-identical rephrasing, while the type+topic exact match handles deliberate updates to known entries.

### Why automatic contradiction handling via confidence drop?
When the same type+topic receives different content, the old fact is likely outdated. Rather than keeping both and relying on the LLM to sort it out later, the system proactively suppresses the old entry (confidence → 0.2, below retrieval threshold) while preserving it in version history. This is simpler than requiring the LLM to explicitly flag contradictions, and the old content remains recoverable via `--history` if the suppression was wrong.

### Why syntax correction feedback instead of strict validation?
When the LLM produces a malformed `<memory>` block, the hook provides specific error feedback (what's wrong + correct format example) on re-prompt. This is more effective than silent failure (LLM repeats the mistake) or strict rejection (LLM may not know what to fix). The parser is deliberately forgiving — it extracts what it can from imperfect blocks rather than rejecting everything for minor formatting issues.

### Why not PostToolUse capture?
Intermediate tool output (file reads, command results) is factual data that could be indexed. However, the LLM already distils relevant observations into the final memory block. Raw tool output is noisy and large — a single file read could be thousands of lines. Storing it would flood the database with low-value entries. The current approach (LLM distils, hook captures) trades completeness for signal quality.

### Why not time-based confidence decay?
The LLM weighting rules already instruct it to prefer recent entries over old ones. The confidence system handles "bad" memories via explicit feedback. Time decay would primarily benefit memories that are *both* old *and* never retrieved — a narrow gap that doesn't justify the operational overhead of a background cron job at current scale.

### Why hook-side composite scoring instead of LLM ranking?
LLMs are inconsistent at multi-factor ranking under token pressure. Passing similarity, confidence, recency, and scope as separate attributes and expecting the LLM to combine them leads to anchoring on whichever signal appears first or is most salient. A pre-computed composite score eliminates this: results arrive sorted, with a single scalar the LLM can trust. The weights (`0.50 similarity + 0.30 confidence + 0.15 recency + 0.05 scope`) can be tuned in `config.py` without changing any instructions.

### Why embedding augmentation with project name?
Prepending the project name to the embedding text (e.g. `"webshop decision use-postgres ..."`) pushes memories from unrelated domains apart in vector space. This reduces cross-project bleed naturally — without changing the schema, queries, or adding filtering logic. It's a zero-cost signal that improves retrieval precision as the database grows across multiple projects.

### Why negation-based contradiction dampening?
Full NLI (natural language inference) classifiers are a significant dependency. But a lightweight heuristic — checking for negation words like "not", "never", "instead of" in semantically similar entries — catches the most common contradiction pattern: "X is Y" vs "X is not Y". When detected, both entries' confidence is reduced slightly rather than one being overwritten, preserving ambiguity where it genuinely exists.

### Why retrieval outcome feedback?
Per-memory `confidence_update` signals tell the system which individual memories are good or bad. But `retrieval_outcome` tells the system whether the *query → results* mapping was useful. This is a system-level signal: if a particular type of query repeatedly yields "neutral" or "harmful" outcomes, the retrieval logic itself may need tuning. The distinction is: confidence improves individual memories, retrieval outcome improves the search process. The adaptive threshold mechanism uses this signal to automatically tighten similarity floors when poor outcomes accumulate.

### Why three retrieval layers instead of one?
Each layer covers a different blind spot: Layer 1 (first-prompt push) eliminates the cold-start "I don't know" problem. Layer 2 (keyword cross-project) surfaces knowledge the LLM doesn't know to ask for. Layer 3 (pull-based) handles explicit gaps mid-conversation. No single approach covers all three cases. The layers don't overlap — Layer 1 fires once, Layer 2 only injects cross-project data, Layer 3 only fires when the LLM explicitly declares a gap.

### Why non-linear confidence in composite scoring?
Linear confidence weighting (e.g. `0.30 × confidence`) gives similar advantage to memories at 0.4 vs 0.8 confidence. Using `confidence²` amplifies the gap: 0.4² = 0.16 vs 0.8² = 0.64. This means high-confidence memories (decisions, corrections) dominate over low-confidence entries at equivalent similarity, which matches the intended behaviour — trusted knowledge should outweigh uncertain signals.

### Why overwrite guard (similarity < 0.8)?
Same type+topic doesn't always mean same fact. "Use RTK for accuracy" and "Use GNSS fallback under canopy" could both be `decision/positioning` — they're valid under different conditions. The 0.8 similarity guard ensures only genuinely updated statements overwrite, while distinct variants coexist as separate memories.

## Configuration

All tunable parameters are centralised in `cairn/config.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `DEDUP_THRESHOLD` | 0.95 | Cosine similarity for near-identical dedup |
| `CONFIDENCE_DEFAULT` | 0.7 | Starting confidence for new memories |
| `CONFIDENCE_BOOST` / `PENALTY` | 0.1 / 0.2 | Base rates for saturating adjustments (actual: `base × (1-conf)` / `base × (1+conf)`) |
| `SCORE_W_*` | 0.50/0.30/0.15/0.05 | Composite scoring weights |
| `RECENCY_HALF_LIFE_DAYS` | 30 | Recency decay rate |
| `MIN_INJECTION_SIMILARITY` | 0.35 | Garbage gate — reject all if best < this |
| `BORDERLINE_SIM_CEILING` | 0.45 | Borderline gate similarity ceiling |
| `BORDERLINE_SCORE_FLOOR` | 0.50 | Borderline gate minimum composite score |
| `RELATIVE_FILTER_RATIO` | 0.7 | Drop entries below 70% of best similarity |
| `MAX_INJECTED_ENTRIES` | 5 | Hard cap on injected entries |
| `SOFT_SIM_OVERRIDE` | 0.60 | High similarity overrides low confidence |
| `SOFT_CONF_FLOOR` | 0.30 | Minimum confidence unless similarity override |
| `DOMINANCE_EPSILON` | 0.05 | Gap threshold for including runner-up |
| `MAX_CONTINUATIONS` | 3 | Hard cap on consecutive re-prompts |
| `DB_BUSY_TIMEOUT_MS` | 5000 | SQLite busy timeout for concurrent access |
| `L1_*` / `L2_*` / `L3_*` | various | Per-layer retrieval thresholds |

## Testing

171 tests across 10 files, runnable without the embedding model (deterministic mock vectors and patched DB paths).

```bash
python3 -m pytest tests/
```

| File | Tests | Coverage area |
|------|-------|--------------|
| `test_parser.py` | 32 | Memory block parsing — valid, malformed, unclosed tags, code fences, special chars, angle brackets in content, 10-entry stress, all-fields-populated |
| `test_scoring.py` | 18 | Composite scoring weights, recency decay curve, saturating confidence dynamics, negation/directional contradiction heuristics |
| `test_gates.py` | 21 | All 9 quality gates — boundary values, gate interactions, diversity filter word overlap, combined pass/fail paths |
| `test_integration.py` | 14 | Full pipeline with in-memory DB — insert → dedup → retrieve → gate, contradiction confidence drop, version history trigger, write throttle |
| `test_hook_e2e.py` | 13 | Stop hook `main()` with patched stdin — storage, blocking, continuation, sessions, metrics, realistic Claude output (extra text, markdown fences) |
| `test_prompt_hook.py` | 8 | Layer 1/2 — first-prompt detection, staged context loading/consumption, short message handling, empty DB |
| `test_daemon_and_cache.py` | 16 | Daemon embed fallback, allow_slow=False, stale PID, context cache semantic hit/miss, continuation counter lifecycle, fail-open crash → exit 0, metric recording |
| `test_query_cli.py` | 12 | CLI commands — search, stats, review, delete, history, compact, projects against test DB |
| `test_retrieval_pipeline.py` | 22 | Retrieval pipeline — find_nearest, insert dedup/contradiction/variant, retrieve_context, adaptive thresholds, Layer 2 cross-project staging, session registration, auto-label edge cases, negation dampening in insert |
| `test_enforcement_loop.py` | 15 | Two-pass enforcement loop, continuation cap, low-info pre-filter, retrieval outcomes, context cache, confidence updates through main() |

Tests run on every push via GitHub Actions (`.github/workflows/test.yml`).

## Limitations and Future Work

- **LLM compliance with "ask first" posture is imperfect** — despite prominent instructions, the LLM sometimes answers "I don't know" before declaring `context: insufficient`. Layer 1 (first-prompt push) mitigates the worst case by proactively injecting context before the LLM starts generating. Layer 3 (Stop hook) catches the remaining cases but the user sees the wrong answer first then the correction.
- **Stop hook output is visible to user** — when the hook blocks and re-prompts, Claude Code shows the block reason (including brain_context XML) behind a collapsible "Ran 1 stop hook (ctrl+o to expand)" element. The `reason` field is the only way to pass data to the LLM on a Stop hook block — there is no `additionalContext` support for Stop hooks. The data is collapsed by default but labelled "Stop hook error:" which can look alarming.
- **No `last_retrieved_at` tracking** — would enable smarter decay and usage-based pruning. Low-effort schema addition, deferred until decay mechanism is implemented.
- **Tag invisibility is behaviour-dependent** — relies on Claude Code stripping angle bracket tags in LLM responses only. Tags in Stop hook output are NOT stripped. If Anthropic changes LLM response rendering, memory blocks would become visible to users.
- **No cross-type contradiction detection** — contradictions are only detected within the same type+topic. A `decision` that contradicts a `fact` on a different topic would not be caught automatically.
- **No PostToolUse capture** — intermediate tool output between turns is not indexed. The LLM distils relevant observations in the final memory block, but raw tool results (file contents, command output) are not stored. This is a deliberate trade-off of completeness for signal quality.
