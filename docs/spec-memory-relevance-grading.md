# Cairn Memory Relevance System — Read-Side Grading + Write-Side Quality

Status: design v1 (spec only, no code). Supersedes ad-hoc threshold tuning as the
primary lever on injected-memory quality.

## Problem

Cairn's main drawback in use is the quality of *dynamically injected* memories: a
high proportion of returns are irrelevant. Empirically (session analysis,
`benchmark_extract.py`, 109 events/34 sessions): ~5 memories injected/event,
**~14–17% eyeball-relevant** (≈1 of 6–7), consistent with the standing
"76% ignored" signal. The noise is not random — it falls into four clean,
reasoning-gradeable buckets:

1. **Off-domain** — wrong project/topic, matched on generic tokens.
2. **Word-sense false friends** — e.g. "load" (engine parameter) vs "load" (electrical).
3. **Topically-adjacent, not answer-relevant** — right neighbourhood, wrong question.
4. **Self-referential meta-memories** — "Cairn contains a profile of James" (cairn-about-cairn).

Both the bi-encoder and the existing `ms-marco-MiniLM-L6` cross-encoder see *topic
match, not answer-relevance* (CE floor already removes the −10/−11 obvious junk but
the 0.5–0.7 band — buckets 2/3 — still passes). Closing that gap is a *reasoning*
judgment, not a surface-similarity one.

Two complementary subsystems address it: **read-side** (grade/gate what gets
injected, and train a cheap model to do it) and **write-side** (generate fewer,
more findable, more self-sufficient memories). The read-side usefulness signal is
*also* the fitness function for the write-side — the two close a loop.

## Non-goals

- Replacing bi-encoder recall. Recall stays; we improve **precision** over its output.
- Fixing **recognition failure** (a relevant memory is shown and the agent doesn't
  register it). This is the irreducible ceiling of agent-as-judge — out of scope;
  backstop stays the human + the ungated correction path.
- Reinstating global per-memory confidence as a ranking factor (deliberately
  removed: it marginalises over context, conflating irrelevance with wrongness).

---

# Part A — Read side: relevance grading + distilled student

## A.1 Pipeline

```
bi-encoder recall (unchanged)         thousands → shortlist (~5–14 candidates)
   │
   ├─ cheap mechanical pre-filter      drop bucket-4 self-referential meta by pattern/type
   │
   ▼
precision stage  ─────────────────────────────────────────────────────────────
   Phase 1 (bootstrap):  AGENT-AS-TEACHER grades the shortlist in the [cm] tail
   Phase 2 (steady):     trained CROSS-ENCODER STUDENT gates per-prompt;
                         agent grades only the ambiguous band + topic-shift
   │
   ▼
inject survivors  (corrections / bootstrap layer surface UNGATED — never gated)
```

The expensive judgment runs only to bootstrap and supervise; the cheap student
carries the per-prompt hot path. The training is what *earns back* the latency.

## A.2 Agent-as-teacher (the label source)

The **main agent is the teacher** — not a separate judge model. Rationale:
- **Zero extra inference / zero added hot-path latency** — verdicts are emitted as
  invisible `[cm]` tail content on the turn already being taken (symmetric with the
  write side: "LLM as participant"). Mechanism half-exists today as
  `confidence_update` / `retrieval_outcome`.
- **Highest-fidelity label** — the agent has the real working context and knows
  whether it *actually used* each memory: ground-truth utility, not a prediction.

Rejected: a dedicated `claude -p` judge (latency/cost on hot path, sees only a
reconstructed transcript) and a synchronous per-prompt local LLM gate
(latency-prohibitive — see A.7).

### Label scheme

Per surfaced memory, a **0–3 relevance grade** (NOT 0–9 — finer scales are LLM
calibration noise; LLMs cluster on a few values; real data is bimodal):

- `0` noise · `1` weak/tangential · `2` relevant · `3` load-bearing/essential

Plus a **separate hard-negative flag** for *actively wrong/misleading* (the `-!`
analog) — a different axis from irrelevance; never folded into 0–3.

### Confident-signal-only rule (defeats self-grading bias)

- Train **hard on the extremes** (clear `0` and `3`).
- **Down-weight or treat-as-unlabelled the murky `1` band.**
- **Non-engagement = UNLABELLED, not negative.** "Didn't use it" ≠ "irrelevant"
  (the irrelevance-isn't-evidence principle). This is the key guard: the agent can
  only mislabel what it *confidently* (mis)judges; what it glosses over yields no
  training signal, so the self-grading blind spot can't poison the student.

### Training signal: pairwise, not absolute

Use the 0–3 grades to induce a **within-shortlist ordering** and train the student
**pairwise** (A≻B). The agent is reliable at relative order, unreliable at absolute
values and **non-stationary** across sessions/versions. Grade = elicitation
device; ranking = training signal.

## A.3 The student

- **Model:** cross-encoder reranker, fine-tuned from a pretrained base
  (`bge-reranker-base` / continue-train the existing `ms-marco-MiniLM`), NOT trained
  from scratch. Subsumes the earlier "swap to a generic better reranker" — train
  our own on in-domain labels instead.
- **Measured cost (entry 4020):** ~200–400 ms/retrieval, ~90 MB RAM, ~90 MB disk,
  zero new deps, **can share the embedding daemon**, net token effect *negative*
  (cutting ~30% of entries dropped total Cairn overhead ~47%→~40%). Runs on this
  box's CPU — **no GPU required**. (Benchmark real candidate-set sizes; ~1 s CE
  spikes have been observed.)
- **Context parity (load-bearing):** the student predicts *the agent's* judgment,
  so it MUST see a comparable recent-context representation (embedding/summary of
  last-k turns or synthesised intent) — not the bare prompt. Get this wrong and the
  student plateaus at low teacher-agreement regardless of label volume.
- **Cold-start / blend:** learned reranker is the base; where a memory has its own
  delivery history, shrink toward its contextual usefulness (empirical-Bayes),
  weighted by history volume.

## A.4 Why exposure bias is bounded here

The student only **re-ranks within the shortlist** the (untrained) bi-encoder
produced. The recall distribution is stable, so the student never has to reason
about memories recall missed. The general exposure-bias worry is therefore benign
in this scoping. Residual mitigation: occasional exploration injects of
lower-ranked candidates; **never** let the student gate corrections/bootstrap.

## A.5 Delivery log (schema, ephemeral DB)

Mirror `calibration_deliveries`, but for general memories and **keyed by context**:

```sql
CREATE TABLE memory_deliveries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    memory_id       INTEGER NOT NULL,
    context_vec     BLOB,        -- embedding of recent-context repr (the join key)
    ce_score        REAL,        -- student/CE score at delivery
    served_rank     INTEGER,
    grade           INTEGER,     -- 0–3, agent teacher (NULL = unlabelled)
    hard_negative   INTEGER DEFAULT 0,
    delivered_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`context_vec` (not the query string) is what makes the signal contextual and
generalisable. This table is the training set **and** finally closes the
effectiveness-measurement loop (currently 1–3% labelled).

## A.6 Goodhart / echo-chamber guards

Agent grades, agent trains the student, student curates the agent's future inputs —
tight self-coupling. Guards: (1) confident-signal-only + non-engagement-unlabelled
(A.2); (2) corrections/disconfirming memories on an **ungated** path; (3) at least
one **independent** outcome metric (task-outcome, not self-rated) tracked alongside.

## A.7 Hardware reality (this deployment)

GPU is a **Quadro T2000 4 GB on nouveau, no CUDA** (Xeon 6c/12t, 32 GB RAM). A
synchronous generative gate is non-viable here (~3–10 s GPU after a CUDA install;
~40–160 s CPU, prefill-bound). Therefore: per-prompt path = **cross-encoder student
(CPU, ~200–400 ms)**; any generative grader (e.g. `Qwen2.5-3B-Instruct Q4_K_M`,
the largest fitting 4 GB) runs **async/offline only**. The agent-as-teacher design
needs no local generative model at all for teaching.

## A.8 Read-side phasing

1. **Instrument** — `memory_deliveries` log + the mechanical bucket-4 pre-filter.
   Immediate noise cut + starts label accumulation. (Closes the effectiveness loop.)
2. **Label** — agent emits 0–3 + hard-neg in the `[cm]` tail (extend
   `confidence_update`/`retrieval_outcome`). Lightweight format: flag clear
   positives + hard-negatives, skip the rest (skip = unlabelled).
3. **Train** — pairwise fine-tune the student on accumulated labels; windowed
   recency-weighted retrains; held-out eval split (mandatory).
4. **Demote teacher** — once student↔teacher agreement >~90% on held-out, student
   runs per-prompt; agent grades only the ambiguous band + topic-shift.

---

# Part B — Write side: generation quality + A/B

## B.1 Levers (in priority order, per the noise analysis)

1. **Suppression / capture criteria — what NOT to write.** Kill bucket-4
   self-referential meta-memories and ephemeral snapshots at generation. (The
   density rules are this lever already; A/B stronger suppression.)
2. **Findability-optimised generation (`qf`).** The memory articulates *the
   questions it is the answer to*, attacking the topic-vs-intent gap **at write
   time** and giving a fast proxy metric (B.3).
3. **Self-sufficiency.** Actionable without the source transcript (test cold).
4. **Redundancy-awareness.** Generate against the existing corpus (cosine dedup).

## B.2 A/B harness — offline backtest, not online

Online A/B fails: write→surface→grade feedback is slow + sparse (most memories
never retrieved), and usefulness-when-surfaced confounds writer × retriever.

Instead: **replay the ~4,924-transcript corpus** through generation-prompt-A vs
prompt-B (offline, via `session_extract.py` + the analyser `claude -p` pattern),
producing two memory sets from identical inputs — fast, controlled,
retriever-isolated.

**Enabling primitive:** stamp every memory with its **generation-prompt version**
(provenance join key); without it, downstream usefulness can't be attributed.

A/B unit = session cohort or offline replay, NOT per-memory (per-memory collides
under dedup).

## B.3 Metrics

- **Primary (fast, instant at write time):**
  - *Findability backtest* — is the generated memory retrieved for the historical
    queries that actually followed in its own transcript?
  - *Self-sufficiency* — fresh model can act on it cold.
  - *Dedup rate* — novelty vs existing corpus.
- **Ground truth (lagged):** downstream read-grader usefulness (Part A), joined via
  the generation-prompt-version stamp.
- **Independent guard (anti-Goodhart):** task-outcome signal (sessions using
  prompt-B memories need fewer clarifications / re-derivations).

---

# Cross-cutting

- **The loop:** read-side usefulness (A) = the fitness function for write-side
  A/B (B). Improving the writer reduces what the reader must filter; the reader
  measures whether the writer improved.
- **Measurement discipline:** all latency/throughput numbers here are estimates or
  single-point measurements — *benchmark before committing*. Cairn latency has
  historically defied intuition (embedding inference blamed; was 19 ms; real causes
  found only by cProfile).
- **Reuses existing infra:** bi-encoder + embedding daemon, the CE reranker,
  `calibration_deliveries` pattern, `session_extract.py`, the analyser cron,
  `benchmark_extract.py`. New build is modest: the `memory_deliveries` log, the
  `[cm]` grade fields, the pairwise trainer, the offline A/B replay + provenance
  stamp.

## Open questions

1. Recent-context representation for `context_vec` / student input: last-k raw turns
   vs synthesised intent vs topic-shift-bounded window?
2. Topic-shift trigger for re-grading / working-set rebuild — reuse the existing
   topic-shift notion or define a dedicated one?
3. Exploration policy (how often to inject below-threshold candidates to fight
   exposure bias) vs token cost.
4. Student retrain cadence + the held-out eval construction.
