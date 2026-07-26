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
| 0 | Freeze + make the freeze legible | **not started** | — |
| 1 | Truth & correctness needing no measurement | not started | Stage 0 committed |
| 2F | Fast lane — enforcement cost | not started | Stage 1 committed |
| 2S | Slow lane — label validity → reranker verdict | not started | Stage 1 committed |
| 3 | Act on measurement | blocked | 2F and 2S report |
| 4 | New capability | blocked | Stage 3 committed |

**Freeze is IN EFFECT while any of Stages 0–2 is unfinished.** No new
subsystems. Bug fixes, docs, and tests are always permitted.

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

- *(none yet)*
