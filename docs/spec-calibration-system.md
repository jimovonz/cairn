# Calibration System — Design Specification

**Status:** Design complete, not yet implemented
**Bundled with:** Cairn (shares SQLite, retrieval primitives, hooks, daemon)
**Branch target:** `feature/calibration-system`

## Purpose

Capture and apply *how to interact with this user* — level, style, preferences, approach — rather than *what is known* about their domains. Complements Cairn knowledge memories: Cairn answers **what** questions; calibration shapes **how** responses are generated.

## One-line characterisation

A calibration store written by a session analyser, retrieved symmetrically against user prompts via stored "what query this answers" fields (`qf`), modulated by an effectiveness feedback loop that measures whether delivered guidance is actually followed.

## Architecture

```
Live interaction (no new burden on per-turn LLM)
    User talks → Agent responds (with calibration block injected if rows match)
        └─→ calibration_deliveries log (turn-indexed)

Session analyser (cron, fires when session JSONL idle N min)
    Input:  cairn-session-extract.py <jsonl> --signal-only   (cleaned ~12% bytes)
    Process: one LLM call, multi-dimensional structured output (Haiku class)
    Output: writes/updates calibration_rows in Cairn DB

UserPromptSubmit hook (existing infrastructure, extended)
    1. Embed user prompt
    2. Retrieve calibration_rows via (kw + qf) similarity match
    3. Filter against calibration_deliveries[session_id]  (session dedup)
    4. Inject <calibration_profile> block, log delivery
```

## Core insight: symmetric-intent retrieval

The novel piece is the `qf` field. Stored rows carry hypothetical user prompts they should fire before. The user's current prompt is itself a query in question-shape. Embedding similarity between (current prompt) ↔ (stored `qf`) is the primary applicability signal. Content keywords (`kw`) supplement for content-anchored queries.

This makes L3-quality retrieval (LLM-articulated intent — see Cairn entry 685) the default at every turn: the cognitive work happens once at write time in the analyser, every future retrieval reuses it.

## Data model

### Table: `calibration_rows`

```sql
CREATE TABLE calibration_rows (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    kw TEXT,                                -- content keywords (CSV)
    qf TEXT,                                -- query-form: JSON array of hypothetical prompts
    source TEXT NOT NULL,                   -- explicit | correction | observation | meta-assessment
    confidence REAL NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    layer TEXT NOT NULL DEFAULT 'subject',  -- subject | general
    session_scope TEXT,                     -- if set, expires at session end
    superseded_by INTEGER REFERENCES calibration_rows(id),
    archived_at TIMESTAMP,
    archive_reason TEXT,
    delivered_count INTEGER DEFAULT 0,
    followed_count INTEGER DEFAULT 0,
    ignored_count INTEGER DEFAULT 0,
    corrected_count INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB
);
CREATE INDEX idx_cal_rows_source ON calibration_rows(source);
CREATE INDEX idx_cal_rows_layer ON calibration_rows(layer);
CREATE INDEX idx_cal_rows_archived ON calibration_rows(archived_at);
```

### Table: `calibration_deliveries`

```sql
CREATE TABLE calibration_deliveries (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    row_id INTEGER NOT NULL REFERENCES calibration_rows(id),
    delivered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    similarity REAL,
    outcome TEXT,                           -- followed | ignored | corrected (set by analyser)
    outcome_evidence TEXT
);
CREATE INDEX idx_cal_deliv_session ON calibration_deliveries(session_id);
CREATE INDEX idx_cal_deliv_row ON calibration_deliveries(row_id);
CREATE INDEX idx_cal_deliv_outcome ON calibration_deliveries(outcome);
```

### Source tiers

| Source | Initial confidence | Fires immediately? | Half-life decay |
|---|---|---|---|
| explicit (user stated) | 0.85-0.95 | yes | 120 days |
| pinned | 0.95 + flag | yes | 180 days |
| correction (reactive) | 0.5-0.7 | after 1 corroboration | 60 days |
| observation (inferred) | 0.1-0.2 each | accumulates to ≥0.5 | 30 days |
| meta-assessment (self-audit) | 0.2 | cross-session corroboration | 60 days |

## Components

### 1. `cairn-session-extract.py` — transcript cleaner

Filters raw JSONL to user-text + assistant-text turns only, dropping tool calls/results, injected `<cairn_context>`/`<system-reminder>` blocks, CLAUDE.md repeats, and thinking blocks. Reduces ~110K-token session to ~13K tokens.

Flags:
- `--signal-only` (default): minimal cleaned output
- `--with-tools`: keep tool input/output for redirect detection
- `--corrections-only`: heuristic filter for likely correction turns
- `--turn-range A-B`: slice
- `--last-N-minutes N`: recent slice

Reuses `cairn/benchmark_extract.py` and `cairn/query.py --context` machinery (~150 LOC).

### 2. Session analyser (cron job)

Triggered when a session JSONL has been idle ≥ N minutes (default 15). Runs:
1. `cairn-session-extract.py` to produce cleaned transcript
2. Single LLM call (Haiku class) with multi-dimensional structured output prompt
3. Writes/updates `calibration_rows`
4. Joins delivery log against subsequent turns, scores `outcome` on existing deliveries

**Analyser prompt produces JSON with these arrays:**
- `user_observations`: inferred from terminology / level fit / style mirroring
- `explicit_instructions`: declared by user (*"I prefer..."* / *"always..."* etc.)
- `approach_assessment`: meta self-audit of agent's session-level approach
- `contradictions`: existing row vs current session evidence
- `drift_signals`: rows worth surfacing for user review
- `row_effectiveness`: per delivered row → followed | ignored | corrected + evidence

Vocabulary discipline for `qf` lives entirely in this prompt. Each emitted row carries 3-6 hypothetical user-prompt phrasings in `qf`, plus content keywords in `kw`.

### 3. UserPromptSubmit injector

Extends existing Cairn UserPromptSubmit hook (`~/.claude/hooks/userprompt-cairn.py` or equivalent). Pipeline:

1. Embed user message (reuse Cairn daemon)
2. Retrieve candidate `calibration_rows` via composite scoring on `kw` overlap + `qf` semantic similarity + recency + confidence
3. Filter:
   - Drop rows whose `archived_at IS NOT NULL`
   - Drop rows already in `calibration_deliveries WHERE session_id = current_session`
   - Drop rows whose `session_scope = current_session` AND user has muted them
4. Apply override hierarchy: pinned > recent explicit > older explicit > subject-specific > general
5. Render `<calibration_profile>` block, inject
6. Write delivery records to `calibration_deliveries`

**Injection format (distinct from `<cairn_context>`):**
```xml
<calibration_profile>
  <general weight="always">
    - <general row content>
  </general>
  <subject confidence="0.81">
    - <subject row content>
  </subject>
  <override>
    - <explicit override of conflicting general for this scope>
  </override>
</calibration_profile>
```

Tag explicitly signals **priming** (shape responses), not **facts** (cite in output).

### 4. `cairn calibration` CLI

Bash-callable subcommand of cairn. **Agent-invoked from natural-language intent, never user-typed.**

```
cairn calibration --show-profile [subject]
cairn calibration --review
cairn calibration --history <row_id>
cairn calibration add --source explicit --content "..." [--scope X] [--pin]
cairn calibration mute <row_id> [--session-only]
cairn calibration disable [--session-only]
cairn calibration mode --level <novice|expert> [--session-only]
cairn calibration delete <row_id>
cairn calibration unmute / enable
```

Session-scoped state in `hook_state` keyed by `session_id`, auto-expires.

### 5. Self-modification — three tiers

**Tier 1 — autonomous:**
- Per-row confidence updates from effectiveness scoring
- Auto-archive rows with `N>=10` deliveries and `<20%` followed
- Auto-promote rows corroborated across ≥4 independent subjects
- Recency dedup window tuning, similarity threshold tuning, decay half-life
- Bounded scalar adjustments only

**Tier 2 — surfaced for user approval via `--review`:**
- Suggested re-phrasings of low-follow rows
- `qf` vocabulary expansions for rows with high `kw` match but low `qf` match
- Promotion candidates that didn't auto-trigger
- Contradiction resolutions

**Tier 3 — manual only:**
- Analyser prompt itself
- Writing-side LLM instructions in `~/.claude/rules/`
- Retrieval scoring formula weights
- System architecture

### 6. Dashboard panels

New **Calibration** tab in cairn dashboard alongside existing tabs.

**V1 panels (ship with system):**
- **Profile**: rows by source/confidence/layer, pinned count, search/filter
- **Effectiveness**: per-row table (deliveries, follow %, ignore %, correction %), follow-rate time-series, top-impact rows, low-follow-rate flagged for archive
- **Review queue**: Tier 2 surfaced items with inline approve/dismiss/edit, badge for unreviewed count
- **Session detail extension**: per-session timeline of calibration injections alongside `cairn_context` retrievals, per-row outcome shown

**V2 panels (add after 4-6 weeks of data):**
- Coverage map (subjects clustered by `qf` vocabulary, cold-start gaps)
- Drift detail (supersession chains, contradictions, archives)
- Analyser health (last-run, processing time, failures)
- A/B causation panel (withhold-injection results)

### 7. Metric events

12 new event names in existing `metrics` table:

```
calibration_row_written         {row_id, source, confidence}
calibration_row_delivered       {row_id, session_id, turn, similarity}
calibration_row_followed        {row_id, session_id, turn, evidence}
calibration_row_ignored         {row_id, session_id, turn}
calibration_row_corrected       {row_id, session_id, turn, evidence}
calibration_row_archived        {row_id, reason}
calibration_row_promoted        {row_id, source_domains}
calibration_row_superseded      {old_row_id, new_row_id, reason}
calibration_review_surfaced     {row_id, suggestion_type}
calibration_dedup_filtered      {row_id, session_id, prev_turn}
analyser_session_processed      {session_id, rows_written, ms_duration}
analyser_session_failed         {session_id, error}
```

Flow into existing `/api/metrics` endpoint.

## What's reused vs new

**Reused (~80%):** Cairn SQLite, sentence-transformers daemon, semantic + keyword retrieval, composite scoring, dedup, supersession, confidence updates, hook_state, query.py CLI patterns, install.sh deployment, UserPromptSubmit hook, transcript JSONL parsing infrastructure (`benchmark_extract.py`, `query.py --context`).

**Genuinely new (~20%):**
1. `cairn-session-extract.py` (~150 LOC)
2. Session analyser cron job + Haiku/Sonnet integration
3. Analyser prompt template (vocabulary discipline including `qf`)
4. `qf` field semantics and population
5. `calibration_rows` + `calibration_deliveries` schema
6. `cairn calibration` CLI subcommand
7. UserPromptSubmit injector for `<calibration_profile>` block
8. Effectiveness feedback scoring
9. Dashboard panels + endpoints
10. CLAUDE.md import on install (one-time seed)

## Rejected alternatives

Recording these to prevent re-litigation in future sessions.

| Rejected | Reason |
|---|---|
| Manual CLAUDE.md maintenance as baseline comparison | User won't maintain it; baseline doesn't exist |
| Bespoke domain:task taxonomy | Cairn keyword+similarity retrieval subsumes it |
| Slash commands as user-facing UI | User prefers natural language; agent invokes tool |
| Per-turn `[cm]` schema additions for calibration | Pushes write-time burden onto live LLM; analyser does it better with full-session context |
| Two separate analyser passes (user-obs + approach-assessment) | LLMs handle multi-dimensional output in one call |
| Always-on unconditional preference injection | Subject-scoped targeted delivery is sharper |
| Full prompt-level self-modification | Unbounded drift / recursion risk |
| "Capitulate when user proposes simplification" rule | Loses the value of independent reasoning the user explicitly wants |
| Replacing per-turn `[cm]` writes with session-only analyser writes | Different shapes of memory — per-turn captures atomic facts/decisions at verbalisation moment (best write-time quality); analyser captures arcs/reconstructed-rejected-alternatives/loose-ends that no single turn can produce. The two layers harvest orthogonal signal from the same source. See Amendment 1. |
| "Cautiously add new analyser output dimensions to avoid drift" | Empirically inverted — orthogonal dimensions expand attention budget via cognitive-mode switching (cairn entry 2087); dimensions are independent within token envelope (entries 1944, 1948). Correct posture is bounded-per-dimension + envelope-aware, not minimal. See Amendment 1. |

## Honest open questions

These resolve only through deployment, not design:

1. **Magnitude of behaviour shift** — could be 10%, could be 40%. Effectiveness counters measure this in 4-6 weeks.
2. **Compliance ceiling for delivered guidance** — calibration rows may still get drifted past under conversational pressure. How often is unknown.
3. **Causation confounding** — Tier 1 auto-modification banks unearned confidence when assistant behaviour was natural anyway. Mitigations imperfect.
4. **Cold start** — new domains get no benefit until enough sessions accumulate.
5. **Drift decay calibration** — half-lives are guesses until real data tunes them.

## Amendment 1 — multi-dimensional analyser, dual write path

Added after Phase 1 landed and before Phase 2 design freeze. Three corrections to the original Components §2 design, derived from in-session review against existing cairn empirical results (entries 1841, 1944, 1948, 2007, 2087, 3048, 3302).

### A1.1 — Two output classes, two write paths

The original spec framed the analyser as writing only `calibration_rows`. That under-uses the LLM pass. The analyser now produces **two distinct classes of output**, both extracted from the same cleaned transcript in a single LLM call, written to two different tables:

| Class | Captures | Table | Distinguishing marker |
|---|---|---|---|
| Calibration rows | *How* signal — level, style, preferences, approach, override rules | `calibration_rows` | `source` in {explicit, correction, observation, meta-assessment} |
| Session-scoped memories | *What* signal that requires the arc — session arcs, reconstructed rejected alternatives, loose-ends, tool-pattern observations | existing `memories` | `source_ref="analyser-session-arc"` (new value) |

This complements per-turn `[cm]` writes; it does **not** replace them. Per-turn writes remain the primary capture path for atomic claims at verbalisation moment (entry 3302 — deliberate Cairn design choice). Session-scoped writes harvest signal that per-turn writes structurally cannot produce, because no single turn sees the arc.

Empty-session recovery is a free bonus: entry 3048 records that 62% of sessions produce zero memories under the per-turn model (compaction children, claude-assist heartbeats, monitoring crons). A session-scoped analyser pass is the only path to extract value from these — they have transcripts but never trigger the Stop hook with substantive content.

### A1.2 — Dimension count and envelope budget

Empirically (cairn entries 1944, 1948 from Isosync dimensionality testing):

- Each analyser output dimension has its own ceiling — independent breakage point per dimension
- Total output is bounded by a token envelope (~25K tokens observed for prompt+output stack on the Isosync workload)
- **Dimensions are independent within the envelope** — they do not compete for attention. 75 mental execution steps + 50 tags + both bugs + full coherence all passed simultaneously
- Entry 2087: orthogonal dimensions *expand* attention budget by activating different cognitive modes (narrative, categorical, self-eval, meta-cognitive, pattern-recognition) — mode-switching prevents autopilot
- Entry 1841: prompt quality is a multiplier on all ceilings (33% vs 100% pass on identical configs)

Design posture is therefore **bounded-per-dimension and envelope-aware**, not minimal-dimension. The analyser schema must:

1. Cap each dimension's output count explicitly (e.g. "max 6 calibration_rows", "max 3 session-arc memories", "max 10 loose-ends")
2. Sum of (per-dimension cap × avg row size) must stay below envelope
3. Each dimension is free to grow up to its own bound; dimensions do not contend with each other within envelope

### A1.3 — Expanded dimension list

Original spec §2 named 6 analyser output dimensions. Amendment expands to roughly 13. Per-dimension caps are starting estimates and become tunable in Phase 6 Tier 1.

**Calibration-row dimensions (write to `calibration_rows`):**
1. `user_observations` — inferred from terminology / level / style — cap 6
2. `explicit_instructions` — declared by user — cap 6
3. `approach_assessment` — meta self-audit of session-level approach — cap 4
4. `contradictions` — existing row vs current session evidence — cap 6
5. `drift_signals` — rows worth surfacing for user review — cap 4
6. `row_effectiveness` — per delivered row → followed | ignored | corrected + evidence — cap (= number of deliveries this session)
7. `tool_redirect_signals` — user overrode proposed tool action → correction-grade calibration — cap 6
8. `misalignment_reconvergence` — agent and user diverged then reconverged → effectiveness signal — cap 4

**Memory dimensions (write to `memories` with `source_ref="analyser-session-arc"`):**
9. `session_arc_memories` — narrative arc of the session: set-out-to-do X, hit Y, pivoted to Z, left W — cap 3
10. `decision_memories_with_alternatives` — decisions with rejected alternatives reconstructed across turns — cap 5
11. `tool_brittleness_patterns` — sequences of failed/retried tool calls → approach error signal — cap 5
12. `loose_ends` — open questions / unfinished threads for next-session bootstrap — cap 10
13. `confidence_calibration_audit` — where was the agent overconfident this session — cap 4

Additional candidates not in v1 amendment (deferred unless envelope budget allows):
- `terminology_drift` — did user adopt new vocabulary or correct agent's — cap 4
- `pacing_signal` — long deliberation vs rapid-fire — appropriate response density — cap 2
- `question_alignment` — did agent answer the actual question or a nearby one — cap 4

### A1.4 — Prompt quality is a Phase 2 work item, not a side effect

Entry 1841 establishes prompt quality as a multiplier on every dimensionality ceiling. The analyser prompt template is therefore not "draft once and ship" — Phase 2 must explicitly budget for iteration:

- Initial draft → measure per-dimension fill rate and content quality on N=20 real sessions
- Identify dimensions that consistently produce empty/garbage output → tighten prompt or reduce cap
- Identify dimensions that consistently saturate cap → consider raising cap if envelope allows
- Lock prompt v1 only after this iteration; v2+ is a Tier 2 review-queue item in Phase 6

### A1.5 — Reused infrastructure for analyser-written memories

Analyser-written rows in the `memories` table reuse the entire existing Cairn pipeline: embedding daemon, dedup against existing memories at cosine 0.85, supersession via origin_id, confidence updates, project scoping, FTS index, push retrieval. The only new bits are (a) the `source_ref="analyser-session-arc"` marker for audit/filtering and (b) the analyser write path itself. No new dedup logic, no new storage layer.

This also means analyser-written memories interact correctly with per-turn writes: if a per-turn write at turn 18 covers the same ground as a session-arc memory the analyser produces, dedup collapses them — and the higher-quality version wins on confidence accumulation.

## Implementation phases

**Phase 1: Foundation**
- `cairn-session-extract.py` with tests
- Schema migrations: `calibration_rows`, `calibration_deliveries`
- CLI scaffolding for `cairn calibration` (commands stubbed)

**Phase 2: Analyser** (see Amendment 1 for dimension list, envelope budget, dual-write rationale)
- Cron job framework (idle-detection over `~/.claude/projects/*/*.jsonl`)
- Analyser prompt template — 13 bounded dimensions (8 calibration + 5 memory), per-dimension caps enforced in schema, envelope-aware total
- Single-call multi-dimensional output handling, sectioned by output class
- **Dual write paths**: calibration rows → `calibration_rows`; session-scoped memories → existing `memories` table with `source_ref="analyser-session-arc"`
- `qf` population for calibration rows (3-6 hypothetical user-prompt phrasings per row)
- Effectiveness scoring pass over existing `calibration_deliveries`
- Reuse of existing embed/dedup/supersession/FTS pipeline for memory-table writes — no new storage logic
- Prompt iteration budget: N=20 real-session measurement pass before locking prompt v1 (per A1.4)

**Phase 3: Delivery**
- UserPromptSubmit injector
- Symmetric retrieval (`qf` + `kw` composite scoring)
- Session dedup via `calibration_deliveries` + `hook_state`
- `<calibration_profile>` block rendering

**Phase 4: CLI completion**
- mute / disable / mode / add / review / show-profile / history
- session-scoped state in `hook_state`
- Agent natural-language → CLI invocation patterns documented in CLAUDE.md

**Phase 5: Dashboard V1**
- 4 panels (Profile, Effectiveness, Review queue, Session detail extension)
- 12 metric events instrumented
- New `/api/calibration/*` endpoints

**Phase 6: Self-modification Tier 1 + 2**
- Auto-archive / auto-promote / confidence updates
- Review queue population for Tier 2 suggestions
- Tier 3 stays manual

**Phase 7: CLAUDE.md import**
- One-time install-time scan for first-person preference statements
- Seed calibration rows with high-confidence

Each phase ships independently with tests; user confirms before proceeding.

## References

- Cairn entry 3299 — HyDE-at-write-time framing (rationale for `qf` field)
- Cairn entry 685 — L3 produces highest-quality retrieval but is least used (motivates symmetric-intent default)
- Cairn entry 2805 — 33% serve rate is precision not coverage (effectiveness loop targets this)
- Cairn entry 3555 — mechanistic capture pattern (similar architectural class)
- Cairn entry 2817 — bootstrap > self-initiated retrieval (motivates push delivery)
- Cairn entry 833 — minimal compliance ceiling (constraint on expected magnitude)
- Cairn entry 3973971 — keyword overlap in composite scoring (reused infrastructure)
- Cairn entry 3302 — per-turn writes are deliberate (motivates dual-write rather than replacement) — Amendment 1.1
- Cairn entry 3048 — 62% of sessions produce zero memories under per-turn model (empty-session recovery motivation) — Amendment 1.1
- Cairn entry 2087 — orthogonal output dimensions expand attention budget via cognitive-mode switching — Amendment 1.2
- Cairn entries 1944 / 1948 — per-dimension ceilings independent within token envelope; ~25K observed — Amendment 1.2
- Cairn entry 1841 — prompt quality is a multiplier on all dimensionality ceilings — Amendment 1.4
- Cairn entry 2007 — ~100:1 transcript→memories compression empirically established (envelope sanity check)

## Out of scope

- Multi-user calibration profile sharing (future)
- Cross-user pattern aggregation (future)
- Real-time learning during a single session (sessions are the unit of distillation)
- Replacing CLAUDE.md (calibration complements; CLAUDE.md remains for static rules)
