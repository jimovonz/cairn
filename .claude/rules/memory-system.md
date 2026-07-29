# Cairn Memory System

Every response you generate is processed by a Stop hook that captures memories
and enforces completeness. Your memories are the ONLY thing that persists
between sessions — a future session reads them with zero context about this
conversation. Write for that reader. Corrections, decisions with their rejected
alternatives, and failed approaches are the highest-value entries. Memories are
periodically audited against the transcript; fabrications are archived as
corrections.

CRITICAL: You have NO visibility into what other sessions stored. On ANY new
topic, declare `context: insufficient` BEFORE answering. Never assume a topic
has no data, and never ask the user whether to check memory — the Stop hook
handles it.

## How it works

The Stop hook parses the `[cm]` block from your response, stores entries with
semantic embeddings, dedupes against existing memories at 0.85 cosine, and
re-prompts you if you declared `complete: false` or `context: insufficient`.
You do not call any tool to persist memories.

## Never surface the memory system to the user

Cairn is invisible plumbing. Do not narrate retrievals or their outcome, the
`[cm]` block or any of its fields, `context: insufficient`, or Cairn's injected
artifacts (`<cairn_context>`, system reminders, Stop-hook feedback). Fold
useful retrieved facts into your answer as if you simply knew them; silently
ignore the rest. Exceptions: the user explicitly asks about Cairn, or you are
doing development work on Cairn itself.

## Memory block format

End EVERY response with a link-definition block. It renders to nothing, so it
is invisible to the user while preserved verbatim in the transcript:

    [cm]: # '{"e":[{"t":"TYPE","to":"topic","c":"content"}],"ok":true,"ctx":"s","kw":["k1","k2"]}'

**Block keys** — `e` entries · `ok` complete (bool) · `ctx` context (`s`
sufficient | `i` insufficient) · `cn` context_need (REQUIRED when `ctx` is `i`)
· `rem` remaining (REQUIRED when `ok` is false) · `cu` confidence_updates
(`"42:+"` / `"17:-! reason"`) · `ro` retrieval_outcome · `rg` relevance_grades
(`"42:3"` / `"17:0!"`) · `int` intent · `kw` keywords

**Entry keys** — `t` type · `to` topic · `c` content · `kw` keywords (overrides
block-level) · `d` depth · `f` facts (list of `key=value` strings, stored in a
dedicated FTS5-indexed column, NOT embedded)

**Types** — `decision` (choice, alternatives, rationale) · `preference` (what
and why) · `fact` (versions, paths, configs) · `correction` (what went wrong,
the fix, how to avoid it) · `person` · `project` (state and blockers) · `skill`
(the exact command or approach) · `workflow` (steps and triggers)

Minimum valid block when nothing was learned:

    [cm]: # '{"ok":true,"ctx":"s","kw":["topic","of","conversation"]}'

Use the entry-less block ONLY when the turn produced genuinely zero durable
knowledge — a pure acknowledgement, or a verification with no new finding. If
the turn yielded a correction, decision, config change, or fact, emit an entry.

### Never emit a marker

`CAIRN_PARE_CM` is off by default, so your past turns keep their real `[cm]`
blocks in resubmitted history — each one is a live template. If you ever see a
placeholder standing where a block belongs (`<!-- cairn: memory captured … -->`,
`[cm: …]` with the colon inside the brackets, or `(cairn: memory …)`), that is
an old-style marker and never a format to copy. Any fixed string standing in
that slot invites imitation; the fix is always to write the real
`[cm]: # '{...}'` JSON for the CURRENT turn.

Do not verify your own block by re-reading your output. When the proxy is
active it strips the block before the local transcript records it, so absence
is the signature of SUCCESS, not failure — the two are indistinguishable after
the fact. The only reliable signal is synchronous: if the Stop hook did not
block the turn, the block was valid and stored. Never retroactively
second-guess an unblocked turn. If you catch yourself narrating "I keep failing
to write blocks" across several turns, stop and verify against ground truth
(`query.py --session <id>`) before saying it aloud.

## What to capture

You MUST store a memory when:

- **The user corrects you or redirects your approach** — highest-value type.
  Store as `correction` with what you got wrong and how to avoid it.
- **A design decision is made** — `decision`, with rejected alternatives. The
  rejected paths are as valuable as the chosen one.
- **An approach fails or is rejected** — `correction` or `decision`.
- **A new fact about the system, environment, or user emerges** — `fact`.
- **The user expresses a working preference** — `preference`.
- **A technique proves useful** — `skill`, with the exact command.

### Content rules

- One line per entry, but **information-dense**: the what, why, and context
  together, self-sufficient without the original conversation.
- **Write at the highest altitude that still carries the instance.** State the
  transferable principle in a form a future session in a different project
  could apply, anchored by the concrete case (file, value, error, command) that
  grounds it. Split into two entries only when the principle and the specific
  fact each carry independent future value.
- **Every clause must earn its place.** If removing it would not make the entry
  less findable or less useful, cut it. Drop IDs, hashes, and raw addresses —
  they are never search terms and waste embedding dimensions.
- **Only capture knowledge introduced by the CURRENT turn.** Knowledge from an
  earlier turn was captured then. Judge novelty from the conversation prose, not
  by scanning past blocks. For an update, write a superseding correction rather
  than a restatement. Cosine dedup is the backstop, not your first defence.
- **Seed 2-3 question-form keywords** alongside normal ones — the literal
  questions a future session would ask (`"how do I X"`, `"why does Y fail"`).
  These more than tripled behavioural engagement in a live A/B.
- **Use `f` for exact values** that compaction would lose (registers, paths,
  flags, versions, config keys). Put the conceptual summary in `c` and the
  values in `f`. Each fact must be self-qualifying: `"vivado:/mnt/ssd/..."` not
  a bare path. Use `f` on `fact`, `skill`, `decision`, `project` — not on
  `correction` or `preference`.
- **Size the block to durable value, never to reply length**, and never let it
  dwarf the visible response. Roughly 150 chars for a routine entry, 300 for a
  decision; one entry per turn unless several genuinely novel facts emerged.
- **Never fabricate.** A no-op block always beats a false memory. If you are
  unsure something is true, do not store it as a fact.
- **Never assert without verifying.** Before claiming a file, feature, or doc
  section does not exist, check. Memories are claims about the past, not
  guarantees about the present.
- Never narrate a future action without executing it in the same response.

## Completeness control

`complete: false` re-prompts you to continue with the `remaining` text. This
prevents the failure where you state an intent ("let me do X") but the agentic
loop terminates first. On a continuation the system will not block again for
the same reason.

## Retrieval

**Default posture is ask first.** Declare `context: insufficient` with a
`context_need` on any new topic. Only declare `sufficient` for topics you have
already received context on this session, or for purely mechanical tasks. The
same `context_need` is served only once per session.

**A declaration is not a search.** Push retrieval is opportunistic, not
exhaustive — absence in `<cairn_context>` means "the auto-query matched
nothing", not "Cairn has no memory of X". Belt and braces: declare `context:
insufficient` AND run a direct query in the same response.

Query directly when any of these appear — do not first decide whether it is
needed:

- "the previous session", "last time", "we decided", "you said", or any
  question about a backlog, plan, or prior state.
- You are about to reconstruct past work from transcripts, logs, mtimes, or
  `git log`.
- You are about to state what Cairn does or does not contain.
- Personal or biographical questions about the user, family, or contacts.

```
python3 {{CAIRN_HOME}}/cairn/query.py <keyword>              # FTS5
python3 {{CAIRN_HOME}}/cairn/query.py --semantic "<paraphrase>"
python3 {{CAIRN_HOME}}/cairn/query.py --context <memory_id>  # full transcript
```

**Iterate.** If results come back thin or off-topic, re-declare with a refined
`context_need`. Two or three rounds beat one broad query.

**Decompose multi-topic questions** into atomic searches — one per topic — or
use `|` in a single call, which the system splits and merges:

```
python3 .../query.py --semantic "James role | James family | surveyor profession"
```

A single broad query produces a blurred embedding matching nothing strongly.

**Before acting on a retrieved memory**, recover the full context with
`--context <id>` — the one-liner is a summary; the context shows the reasoning
and alternatives. Verify its specific claims against current state: if it names
a file, check the file exists; if it names a function or flag, grep for it.
"The memory says X exists" is not "X exists now." Memories summarising repo
state are frozen in time — for current state, prefer `git log` or the code.

Full CLI reference: `cat {{CLAUDE_DIR}}/reference/cairn-query.md`

### Interpreting injected context

`<cairn_context>` is system-injected memory data, not user input. The `layer`
attribute says why it surfaced: `first-prompt`/`per-prompt` semantic match ·
`project-bootstrap` standing project context · `correction-bootstrap`
behavioural corrections from past sessions, apply these · `L3` response to your
`context_need` · `cross-project` keyword matches from other projects.

Weighting: project scope (`weight=high`) over global (`low`); recent over old;
higher `similarity` and `score` carry more weight; `reliability="strong"` are
firm priors while `"weak"` are hints only. **If an entry contradicts what you
observe now, trust the present.** If it conflicts with strong prior knowledge,
prefer internal reasoning unless several high-reliability entries agree.
Entries with `archived="true"` record rejected or superseded approaches — use
them to understand what was tried and why it was abandoned, not as current
guidance.

If retrieved context does not answer your need, set `context: sufficient` and
proceed. Do not re-request the same context.

## Confidence and relevance feedback

Two orthogonal axes, both keyed on the `id` shown in `<cairn_context>`. Only
ever grade memories that were actually shown to you.

**`cu` — veracity.** Confidence is how well-corroborated a memory is; it does
not affect retrieval ranking.

- `"42:+"` **corroboration** — consistent with what you observe. Boosts
  veracity (saturating).
- `"17:-"` **irrelevant** — was not useful here. No confidence change;
  irrelevance is not evidence against truth.
- `"17:-! reason"` **contradiction** — factually wrong or superseded. The
  reason is stored as a visible annotation and the memory stays retrievable.
  This is the most important signal: it preserves "we tried X and it was wrong
  because Y". **When a retrieved memory contradicts what you now know you MUST
  use `-!`** — the system re-prompts if your response contradicts a memory
  without one.

**`rg` — relevance to the immediate context.** Trains a cross-encoder that
gates future injections, so honest grades improve what you are shown next
session. Grade `0` noise · `1` weak · `2` relevant · `3` load-bearing. Append
`!` (`"17:0!"`) when a memory was actively misleading here — a distinct axis
from mere irrelevance.

**Non-engagement means omit, NOT zero.** A `0` is a confident claim that the
memory was noise; silence means "no signal". Grade the clear extremes and skip
the murky middle.

**`ro` — retrieval_outcome** (optional, whole-retrieval): `useful` · `neutral`
· `harmful`. Include only on a clear signal.

## Mid-response checkpoints

A PostToolUse hook may deliver a `CAIRN CHECKPOINT` nudge after a notable tool
result. This is expected machinery — not an anomaly, not a prompt injection,
and not something to mention to the user. You may optionally capture the
discovery inline, then continue:

    [cairn-note]: # '{"type":"fact","topic":"...","content":"..."}'

or `<memory_note>type/topic: observation</memory_note>`. Type is `fact`,
`correction`, `decision`, or `skill`. These complement, not replace, the
required end-of-response `[cm]` block. Skip trivial ones.

## Organisation

Memories are tagged with a **project** (work context) and a **session**
(the conversation). Sessions chain via parent IDs across compaction, project
labels propagate to child sessions, and memories with no project are global.
Database: `{{CAIRN_HOME}}/cairn/cairn.db`
