# Cairn Remediation Programme — 2026-07

Origin: third-party architecture assessment (2026-07-26) plus self-audit of the
same. This programme is **longitudinal** — several stages cannot be validated
without accumulating Cairn delivery data, so gates are keyed to **data volume,
not dates**.

Read this file before proposing new Cairn work. Update the Status table in the
same commit as any stage change.

## Status

| Stage | Title | State | Gate to enter |
|-------|-------|-------|---------------|
| 0 | Gate + make the surface legible | **done** — 0.1–0.4 | — |
| 1 | Truth & correctness needing no measurement | **done** — 1.1–1.10 | — (Amendment 2) |
| 2F | Fast lane — enforcement cost | **2F.1 done; 2F.2 BLOCKED on data** | needs ≥200 `hook_fired` after 2026-07-26 |
| 2S | Slow lane — label validity → reranker verdict | **2S.1/2/3/5/6/7 done (F1, F2, F3); 2S.4 BLOCKED on data** | — (Amendment 2) |
| 3 | Act on measurement | **3.3 done; 3.1 unblocked (expensive LLM pass); 3.2 blocked on 2S.5 data** | 2S.1 done |
| 4 | New capability | **4.1 done** — flags, does not archive (F4) | — (Amendment 1) |

**Blocked on accumulating data (not on work):**
- **2F.2** — needs ≥200 `hook_fired` after 2026-07-26, when `enforcement_block`
  instrumentation landed. Pre-instrumentation rows cannot substitute: the old
  cause markers fire on both blocking and non-blocking paths.
- **2S.4** — needs `gate_status` rows to accumulate. Historical deliveries are
  NULL on that column, and excluding by the NULL-`reranker_model` proxy is
  exactly the approximation F1 flagged as insufficient.

**Open and implementable now** (no data dependency): 2S.5 (randomised A/B —
needs the daemon to hold and randomly assign two cross-encoders), 2S.6
(deferred-value window), 3.3 (per-document fingerprints), 4.1
(verification-on-retrieval). 3.1 is unblocked by 2S.1 but is a long-running,
expensive Opus vote pass (~4,455 votes outstanding).

**Gate (as amended by Amendment 1): write paths only.** Read-side work —
thresholds, rerankers, retrieval, flags — ships freely with no gate. Write-path
work (schema, corpus writes, archive/delete, replication) lands only when its
writes are attributable via `source_ref` and retractable in bulk. The original
blanket freeze on new subsystems is RETRACTED.

## Baseline snapshot — 2026-07-26

Captured before any remediation, so future sessions measure drift rather than
re-deriving. Reproduce with the acceptance commands in each stage.

**Enforcement (14d, ephemeral `metrics`):** `hook_fired` 261 successful stores
against 81 enforcement events (`missing_memory_block` 46, `question_before_cairn`
35, plus blocked variants) — approximately 24% of stop events, upper bound,
because hard blocks and staged nudges are not yet distinguished.

**Engagement label strata (`memory_deliveries`, all time):**

| `engaged_method` | rows | engaged=1 | engaged=0 | NULL |
|---|---|---|---|---|
| (untagged) | 21,101 | 1,699 | **0** | 19,402 |
| lexical | 1,198 | 179 | 1,019 | 0 |
| semantic-backfill | 26 | 26 | **0** | 0 |

**The central finding:** 94.6% of deliveries have **no negative class**. Untagged
rows recorded only positives; non-engagement was never distinguished from
never-scored. Only the lexical stratum carries a usable base rate (14.9%).
Any engagement rate computed across strata is invalid, and the usable label
pool is ~1,198 rows — not ~22,000.

## Stage 0 — Freeze, and make the freeze legible

Labelling the surface is the enforcement mechanism for the freeze, not a
documentation afterthought.

- **0.1** Freeze policy recorded in `CLAUDE.md` with a pointer to this file.
- **0.2** Subsystem tiering: every subsystem marked *supported* / *experimental*
  / *frozen*, with per-tier guarantees.
- **0.3** Multi-node sync relabelled *experimental*, stating the mismatch
  plainly: LWW merge, no ACL, no inter-peer trust provenance, on a design whose
  every other assumption is single-user.

Rejected alternative: reducing subsystem count by deletion — costs more than
labelling and discards working code.

## Stage 1 — Truth & correctness needing no measurement

- **1.1** `README.md:413` "Pull-based retrieval" contradicts `README.md:57`
  "Automatic context injection". The pull-only row predates first-prompt /
  per-prompt / project-bootstrap push injection. Rewrite as push+pull.
- **1.2** Comparison table → two plain differentiators (zero-extra-call
  distillation, mechanical enforcement). Date-stamp the 30-repo survey and state
  it was not independently audited.
- **1.3** `ingest.py` `source_sections` is an LLM-declared cache-invalidation key
  (`ingest.py:1134`, `1601`). Derive attribution mechanically; fail safe
  (archive on any section change) when attribution is unavailable.
  NOTE: removal detection is **not** broken — `_fingerprint_section`
  (`ingest.py:70`) hashes the whole section payload, so deletions do flip the
  hash. The defect is the archival key, not the fingerprint.
- **1.4** Tests for the five fingerprint functions (currently zero), including
  the removal case, so 1.3's scope is settled by test rather than argument.
- **1.5** Document the ms-marco fallback: `training_data/` is gitignored
  (`.gitignore:62`), so the student that beat the incumbent 67.0% vs 39.6% is a
  per-machine artifact. Other nodes silently fall back to pretrained ms-marco.
- **1.6** Upstream contract canaries in CI asserting the hook-input shape,
  `agent_id`, and transcript JSONL layout Cairn depends on. Ordered here
  deliberately: undetected upstream drift during Stage 2 would silently
  invalidate every number it collects.
- **1.7** `train_reranker.py:137` filters `engaged IS NOT NULL`, which is
  insufficient — the 1,699 untagged positives pass that filter and carry no
  negatives, so `--engagement` trains on ~1,904 positives vs 1,019 negatives
  with 1,725 positives drawn from strata that cannot emit a negative. Restrict
  engagement pairs to strata containing both classes, or stratify explicitly.

Acceptance: `.venv/bin/python -m pytest tests/ -x` passes; README self-consistent.

## Stage 2F — Fast lane: enforcement cost

No dependencies. Shortest path from measurement to a real improvement.

- **2F.1** Split enforcement metrics into hard-block vs staged-nudge — they are
  conflated in the 24% baseline. Report block rate, turns-to-resolution, and
  wall-clock cost.
- **2F.2** Reduce enforcement cost, targeted by 2F.1. The hard-block floor
  (~15% of turns paying a full extra inference) is the largest known latency tax
  and is currently unaddressed.

Gate: 2F.1 needs ≥200 `hook_fired` events after the split lands.

Acceptance: README's "brief pause" claim replaced with a measured distribution.

## Stage 2S — Slow lane: label validity, then reranker verdict

Strictly sequential. Each step is unsound before the one above it.

- **2S.1** Segment all engagement reporting by `engaged_method`. Never compute a
  rate across strata. Treat untagged rows as unlabelled, not as negatives.
- **2S.2** Explain the engagement/grade anti-correlation (ungated scores show
  highest engagement, worst grades). **Leading hypothesis:** it is a stratum
  artifact — if ungated deliveries skew untagged, their rate is inflated toward
  100% by construction. Test this first; it may dissolve the finding entirely.
- **2S.3** Mark gate-unavailable deliveries distinctly.
- **2S.4** Re-run student vs ms-marco excluding gate-unavailable rows.
- **2S.5** Randomised per-delivery reranker A/B. Every current live verdict is
  flag-day / time-confounded, so this is the only sound comparison.
- **2S.6** Deferred-value-aware per-layer window. Do NOT compare layers on a
  single-turn window.

Gate: 2S.5 needs ≥500 randomised deliveries per arm before any verdict.

If this lane sprawls, cut 2S.6 first, then 2S.5. Do not cut 2S.1 or 2S.2 —
everything in Stage 3 hangs off them.

## Stage 3 — Act on measurement

Scope undecidable until 2F and 2S report.

- **3.1** Resume denoise vote pass (~2006/6461, resumable) + retrain. Gated on
  2S.1 — retraining against an invalid label wastes the pass.
- **3.2** Ship the student as a release asset. Gated on 2S.5.
- **3.3** Per-document fingerprint granularity in `ingest.py`, so one changed
  docstring stops re-distilling the whole `docs` section. Cost, not correctness.
  Dependency-free — pull forward if Stage 2 stalls.

## Stage 4 — New capability, only after characterisation

- **4.1** Verification-on-retrieval for memories asserting current repo state.
  This is the remedy for the staleness axis. It is a new feature, so the freeze
  bars it until Stage 3 is committed.

## Non-goals / rejected

| Rejected | Reason |
|---|---|
| Passive confidence decay | `SCORE_W_CONFIDENCE = 0.0`, `SCORE_W_RECENCY = 0.0` — confidence is already decoupled from ranking, so decay would relabel without changing retrieval. Staleness handled by 4.1 |
| First-prompt suppression | Category error: single-turn engagement mismeasures a layer whose value horizon is the whole session. Suppression maximises invisible misses |
| Student floor recalibration | Already shipped per-model in `floor.txt`. Further recalibration contradicts the settled budgets-not-thresholds decision, and a floor on uncalibrated pairwise logits is unstable |
| Semantic engagement **threshold tuning** | Dropped 2026-07-25 — low recall is the intended precision behaviour. Note 2S.1 (segmented *reporting*) is a distinct, retained item |
| Reducing subsystem count by deletion | Tiering (0.2) achieves the goal at lower cost |
| Claiming removal detection in ingest is broken | Disproved — section payload hashing catches deletions |

## Amendment log

Append numbered amendments here rather than rewriting stages in place, so the
programme's evolution stays traceable.

### Amendment 1 — 2026-07-26: freeze narrowed to write paths

Source: revised third-party position, plus the stated constraint that Cairn is
self-hosting, so a deferred effectiveness feature forfeits compounding return on
every subsequent day of work. That is a real return, not a rationalisation.

**The blanket freeze in Stage 0 is retracted.** Velocity was the wrong target.
The line that matters is reversibility, and it runs read-side vs write-side:

- **Read-side errors are bounded** by the time they were live. A miscalibrated
  threshold, a weak reranker, an expensive fan-out — all revert by changing a
  constant. Default-off flags (`CAIRN_SYNC_ENABLED`, `RERANKER_BGE_ENABLED`,
  `CAIRN_GRAPH_WATCH`, `RELEVANCE_PREFILTER_ENABLED`) already handle this well.
  **Ship these as fast as desired. No gate.**
- **Write-side errors accumulate in the one artefact that cannot be rebuilt.**
  A flag flipped off does not retract the writes made while it was on. So the
  gate applies only to: schema changes, corpus writes, anything that archives or
  deletes, anything replicated.

**Replacement gate (Stage 0.4):** a write-path feature lands only when its
writes are attributable via `source_ref` AND retractable in bulk. This costs
roughly two lines per subsystem and no velocity.

The velocity argument extends one step to support this: because Cairn gates the
effectiveness of the work it is used to build, corpus noise taxes all future
feature work. Past some noise level, shipping faster makes you slower. The
crossing point will not be noticed by intuition — it presents as the assistant
being vaguely less useful, not as a failure. **Marginal-entry engagement is the
metric that would show it** (restored as 2S.7 below; it was dropped in error
during the original re-staging).

**Corrected attribution of the ingest defect.** The revised critique again cites
"undetected removals". That remains disproved — `_fingerprint_section`
(`ingest.py:70`) hashes the whole section payload, so deletions flip the hash.
The write-side defect is 1.3: archival keys on `source_sections`, which the
distilling LLM self-reports. This changes the invariant to record for
`ingest.py`: not "assumes a fixed non-shrinking section namespace" (true of
`diff_sections`, but not the bug) but **"trusts LLM-declared provenance as an
invalidation key"**. Recording the wrong invariant would leave the actual defect
undocumented.

Also non-load-bearing in the revised critique: stale entries persisting "at 0.7
confidence with no passive decay". `SCORE_W_CONFIDENCE = 0.0` — confidence feeds
nothing, and archival is driven by `archived_reason`, not confidence.

### Amendment 1 — new items

- **0.4** Replace the blanket freeze with the write-path gate above. Read-side
  work is explicitly unblocked, effective immediately.
- **1.8** **Bulk retraction verb** — `query.py:604` `archive_memory` is single-id
  only; there is no archive-by-`source_ref`. Stamping without retraction is
  inert, so this lands **first**, with `--dry-run` and a count before it
  writes. Until it exists, "a bad experiment is a query away from being
  archived" is false.
- **1.9** **Generalise version stamping to every write path.** Currently only the
  agent-memory path carries a version (`GENERATION_PROMPT_VERSION` /
  `AB_ARM_VERSIONS`). Unversioned: `review_writeback.py:200`
  (`"review-writeback"`), the analyser (`"analyser-session-arc"`), and
  `ingest.py`, which writes a rich source_ref dict but no extractor version
  despite `EXTRACTOR_VERSIONS` existing and feeding only the fingerprint. Also
  covers `ingest_transcript.py` and `backfill_ingestion.py`.
  **Census 2026-07-26** (measured once 1.8 made it queryable): 10,501 memories
  carry a NULL source_ref — the majority of the corpus is unattributable and
  therefore unretractable. `analyser-session-arc` holds 2,227 rows under one
  unversioned stamp, so a bad analyser change can only be retracted wholesale,
  never per-version. Generation arms (`genA-*`/`genB-*`) total 1,209 and are the
  only correctly versioned writes. Note `ingest.py` stamps the whole source_ref
  JSON dict as the value, so ingest retraction requires `--like` on a path
  substring; exact match cannot address it.
- **1.10** **Per-subsystem input-domain invariant.** One line per write-path
  subsystem stating what it assumes about its input domain. The two ingest
  defects came from transplanting an invariant into a domain that violates it,
  not from haste — slowing down would not have caught them. Cheapest available
  intervention, zero velocity cost.
- **2S.7** **Marginal-entry engagement** — restored. Gated on 2S.1 (segmented
  reporting), since it is an engagement measure and inherits the stratum problem.
- **4.1 reclassified.** Verification-on-retrieval archives stale entries, so it
  is a write path, not a read path. Under the new gate its prerequisite is
  1.8 + 1.9 (attributable and retractable), not "Stage 3 committed". Applying
  the amendment consistently unblocks it earlier than the original staging did.

Net effect: write-path features ship as fast as read-path ones, because a bad
experiment becomes attributable and retractable rather than permanently mixed
into the corpus. This also makes the engagement instrumentation usable
per-experiment rather than per-tier — measurement currently generated and not
consumed.

### Finding F1 — 2026-07-26: 2S.2 resolved, the anti-correlation was an artifact

Closed by 2S.1's stratified reporting, against existing data — no new
collection was needed, which is why Amendment 2 moved it forward.

Engagement measured WITHIN the two-class stratum, by reranker:

**What is established.** Untagged deliveries are 100% engaged *by construction*
(1,699 positives, 0 negatives), so any group's blended rate rises with its
untagged share. The prior finding — "ungated scores the HIGHEST engagement
(61.6%) with the WORST grades", which was blocking trust in engagement as a
referee — is fully explained by this: ungated rows are 97.3% untagged. That is
arithmetic, not inference. **The anti-correlation was an artifact; there is no
anomaly to explain, and engagement is usable as a referee provided strata are
respected.** 2S.2 is closed.

**What is NOT established — no ranker verdict follows.** Two errors must be
avoided when reading the per-reranker numbers, and an earlier draft of this
finding made both:

1. **A NULL `reranker_model` is not an "ungated" arm.** On a gated layer it
   means the gate was *unavailable* (daemon down), so those 428 rows must be
   excluded from any reranker comparison. Including them manufactures
   ungated-vs-reranker conclusions in whichever direction the artifact happens
   to point.
2. **Engagement rate is a precision proxy, not the ranker's objective.** The
   ranker's job is maximum relevant data against least noise, so a model that
   suppresses aggressively can post a high rate while losing recall. Rate must
   always be read beside delivered relevant volume.

Gate-available rows only, lexical stratum, volume beside rate:

| reranker | lexical n | engaged | rate | engaged/session | sessions | avg grade |
|---|---|---|---|---|---|---|
| student (local) | 449 | 146 | 32.5% | 0.78 | 188 | 1.47 |
| cross-encoder/ms-marco | 225 | 0 | 0.0% | 0.00 | 142 | 1.33 |
| BAAI/bge-reranker-base | 102 | 0 | 0.0% | 0.00 | 603 | 1.20 |

This looks like strict dominance and must not be read as one. **Zero engaged out
of 225 is not what a mediocre ranker looks like — it is what a confound looks
like.** The student is the currently deployed model, so its rows concentrate in
recent interactive sessions, while bge's 102 rows are spread over 603 mostly
historical sessions. The comparison is time-confounded exactly as 2S.5 assumes.

Consequences:
- 2S.2 closed; dependents (2S.4, 2S.7, 3.1) unblocked on the label-validity axis.
- **No promotion decision may be taken from this table.** 2S.5 (randomised
  per-delivery A/B) remains strictly required — a large effect measured badly is
  still measured badly.
- 2S.3 is upgraded from bookkeeping to a correctness prerequisite: until
  gate-unavailable rows are marked explicitly rather than inferred from NULL,
  every reranker comparison silently re-admits them.
- Do NOT read the layer table's first-prompt figure (5.2%) as a verdict on
  first-prompt: single-turn engagement mismeasures a session-horizon layer, per
  the standing rejection of first-prompt suppression.

### 2S.5 — how to run the randomised reranker A/B

Shipped and OFF by default; each resident arm costs daemon memory. To start it:

```bash
export CAIRN_RERANKER_AB=1
export CAIRN_RERANKER_AB_ARMS="cross-encoder/ms-marco-MiniLM-L-6-v2,<student-path>"
cairn-daemon restart
```

Assignment is deterministic hash-bucketing on the rerank query, so a retry keeps
its treatment — switching arms on retry would contaminate the comparison the
experiment exists to make clean. Balance measured at 1001/999 over 2,000 keys.

No new column: `memory_deliveries.reranker_model` already records which model
scored, and the daemon reports the ARM that actually scored rather than the
default model name — mislabelling the treatment would silently invalidate the
experiment. A failed arm degrades to the default model rather than dropping the
gate, and is labelled accordingly.

Read the result with `query.py --delivery-stats` (stratified per 2S.1) and
`--marginal-engagement`. Per Finding F2, judge on **rank-ordering quality**, not
only aggregate engagement: a model that lifts the average while leaving the head
flat has not done the ranker's job. Gate: ≥500 randomised deliveries per arm.

### Finding F4 — 2026-07-26: file-absence is weak evidence of staleness

4.1 was specified as verification-on-retrieval that archives memories asserting
repo state that no longer holds. Built and run, the premise does not survive
contact with the corpus, so the design changed from ARCHIVE to FLAG.

Two false-positive classes, both found by running it:

1. **Ephemeral paths.** The first live pass flagged 52 true memories about CAN
   register findings solely because the `/tmp` file they were harvested into had
   been cleaned up. The absence of a scratch file is evidence about the
   filesystem, not about the claim. Now excluded by prefix.
2. **Knowledge derived from a file.** After excluding ephemeral paths it still
   flagged entries like `forscan-license-architecture`, whose content is what
   was LEARNED from a binary, not an assertion that the binary exists. Deleting
   the artefact does not falsify the finding.

The general shape: a memory citing a path is usually asserting something learned
*from* it, not asserting its existence. So the checkable subset is much smaller
than 4.1 assumed, and archiving on file-absence would destroy true knowledge —
the precise failure the no-passive-decay decision was protecting against.

`cairn/verify_stale.py` therefore flags to the annotation log by default and
archives only under an explicit `--archive`, with a versioned `source_ref` so a
bad pass is retractable in one command. It also refuses to judge relative paths
when the repo root is absent on this machine, since "not checked out here" is
otherwise indistinguishable from "deleted".

This does NOT reopen passive decay, which remains rejected for the original
reason: confidence is decoupled from ranking, so decaying it would relabel
without changing what is retrieved.

### Finding F3 — 2026-07-26: every layer carries substantial deferred value

`cairn/deferred_engagement.py`, validated at **82.1% agreement** with live
verdicts at window=1 before reporting.

| layer | w=1 | w=3 | w=5 | w=10 | n |
|---|---|---|---|---|---|
| per-prompt | 41.7% | 72.0% | 79.9% | **88.9%** | 314 |
| first-prompt | **19.8%** | 33.9% | 38.5% | **45.3%** | 192 |
| correction-bootstrap | 64.5% | 75.2% | 76.9% | 78.5% | 121 |
| project-bootstrap | 57.9% | 80.7% | 86.0% | 87.7% | 114 |

**Single-turn engagement understates every layer, and understates first-prompt
worst.** first-prompt has the lowest same-turn rate of any layer (19.8%) and
more than doubles by ten turns. That is the quantitative confirmation of the
standing rejection of first-prompt suppression: ranking layers on a single-turn
window scores a session-horizon layer as noise, and the layer that looked
weakest on the old metric is the one whose value is most deferred.

Consequence: any layer-level decision (suppression, cap, budget) must state its
window. A comparison at w=1 is not wrong, but it is a claim about immediate use,
not about value.

**The self-validation gate paid for itself on first use.** An initial run
reported 76-90% at w=1 — implausible against a live rate near 15%. The gate
withheld the layer table, which forced diagnosis rather than publication. The
cause was an argument-order bug in this tool (`score_engagement` takes
response, memory, prompt — memory was being passed first), not the response
reconstruction that had been hypothesised. Without the gate, a confident and
entirely false "deferred value does not exist" finding would have shipped.

### Finding F2 — 2026-07-26: the ranker is not ordering within its own head

From `query.py --marginal-engagement`, lexical stratum only (n=1,204).

| served rank | n | engaged | rate |
|---|---|---|---|
| 0 | 328 | 51 | 15.5% |
| 1 | 254 | 41 | 16.1% |
| 2 | 215 | 38 | **17.7%** |
| 3 | 126 | 21 | 16.7% |
| 4 | 84 | 12 | 14.3% |
| 5 | 53 | 5 | 9.4% |
| 6 | 45 | 3 | 6.7% |

Two readings, one actionable and one not:

1. **Ordering within the head is flat, and slightly inverted.** If the ranker
   were ordering well, engagement would fall monotonically with rank. Rank 0
   (15.5%) is not better than rank 2 (17.7%). Across the top five positions the
   ranker is not separating useful from less useful at all — a direct
   measurement of ranking quality that an aggregate rate cannot express, and it
   is independent of which model is deployed.
2. **Marginal value does fall off after rank ~4** (9.4%, 6.7%), so the tail of
   each injection is materially weaker than the head. This is the honest basis
   for a cap discussion; the head-flatness above is the honest basis for
   doubting that reranking currently earns its cost.

**The by-turn-size table is confounded and must not be used to set a cap.**
Turns with few injected memories are turns where retrieval found few candidates,
so size correlates with match quality: the 2.7% rate at size 1 measures weak
prompts, not an oversized cap. The readout now says so in-band, because the
number is otherwise an inviting way to justify whatever cap one already wanted.

Consequence: 2S.5's randomised A/B should be judged on rank-ordering quality,
not only on aggregate engagement — a model that raises the average while leaving
the head flat has not done the ranker's job.

### Amendment 2 — 2026-07-26: schedule by clock-start, not by stage

The Stage 1 → 2 → 3 ordering sequenced work by certainty. With data-volume
gates that is wrong: for any item gated on accumulating data, the instrumentation
is what starts the clock, so deferring it behind untimed work burns calendar for
zero benefit. **Stage numbers no longer imply execution order.** Schedule by
class:

- **Class A — clock-starters.** 1.9 (stamping: every unstamped write is
  permanently unattributable, cf. the 10,501 NULL rows), 2F.1 (enforcement
  split), 2S.1 (segmented reporting), 2S.3 (mark gate-unavailable deliveries),
  2S.5 (randomisation switch). Land these first regardless of stage. Every day
  deferred is a day of data permanently lost.
- **Class B — analysis on data that already exists.** 2S.2, 2S.4, 2S.7. No
  waiting whatsoever. In particular **2S.2 requires no new data** — the
  stratum-artifact hypothesis is testable against historical rows immediately,
  and it was previously parked behind a gate that does not exist.
- **Class C — untimed correctness and docs.** 1.1–1.7, 1.10, 0.2, 0.3, 3.3.
  Do these while Class A accumulates.
- **Class D — genuinely gated.** 2F.2 (needs 2F.1's distribution), 3.1 (needs
  valid labels), 3.2 (needs the A/B verdict), 4.1 (needs 1.8 + 1.9).

The stage headings remain as topical grouping and dependency record. The
Status table tracks stages; execution follows classes.
