---
name: cairn-relevance-grading
description: Read-side relevance grading (memory_deliveries, rg grades, engagement labels, reranker/student, buckets) and the write-side A/B experiment (arms, AB_ARM_VERSIONS, offline replay).
---

## Read-side relevance grading & write-side A/B (docs/spec-memory-relevance-grading.md)

**Ranking weights — check these before reasoning about ranking.** `config.SCORE_W_*` is the composite scorer's only input: similarity 0.50, keywords 0.15, scope 0.05, and **confidence 0.0 and recency 0.0 — both deliberately disabled** ("veracity is not a ranking signal"; "age is not a usefulness signal; obsolescence handled by supersession"). `confidence` is still computed, stored, updated by `cu` corroboration / `-!` contradiction, and surfaced as the `reliability` attribute — it simply carries no ranking weight. `_recency_decay()` and `RECENCY_HALF_LIFE_DAYS` likewise still compute, and nothing consumes them. Reading either code path without checking its weight gives the wrong answer about how retrieval ranks.

**Read side — what gets injected, graded, and measured.** Every injected memory is logged to `memory_deliveries` (ephemeral DB) keyed by a cleaned recent-context window (`cairn/relevance.py:build_context_window`), with ranking provenance (`reranker_model`, `score_components`, `layer`, `scope`). Two labels accumulate per delivery:
- **Agent-as-teacher grades** — the `rg` field in the `[cm]` block (`"id:grade"`, 0–3 + trailing `!` for hard-negative) is written back by `apply_relevance_grades`. (Behaviour already specified in the memory rules.)
- **Behavioural engagement** (the PRIMARY, non-circular label) — at Stop time `apply_engagement` mechanically detects whether the response actually USED each delivered memory (distinctive-term overlap minus prompt terms) → `engaged` / `engaged_score`. Agent `rg` supplements it.

A mechanical **bucket-4 prefilter** (`is_self_referential_meta`) drops self-referential meta-memories (gated by `RELEVANCE_PREFILTER_ENABLED`, **ON** since the 2026-07-02 review; corrections exempt). The cross-encoder reranker is device-aware (`config.resolve_reranker`): `BAAI/bge-reranker-base` (floor **0.015**, recalibrated 2026-07-07 from 3733 Opus `rg` labels — the earlier 0.10, calibrated 2026-07-02 from 9k `memory_deliveries`, dropped 56% of grade-3 load-bearing memories) on CUDA when `RERANKER_BGE_ENABLED` **and** the GPU has >= `RERANKER_MIN_VRAM_GB` (6 GB), else `cross-encoder/ms-marco-MiniLM-L-6-v2` (floor −3.0); the daemon owns the model and scores the recent-context window, so the hot hook path never imports torch. A **trained student** (`CROSS_ENCODER_STUDENT_PATH`, a local per-machine dir) overrides both when present — see "Phase 3 — trained cross-encoder student" below. Phases 1–3 are live; teacher-demotion (Phase 4) is still future work. **Intended automation for Phase 4**: reuse the `calibration_selfmod.py` / `ab_selfmod.py` self-mod cron pattern — label-volume-gated retrain, then a promotion gate before swapping the deployed model file. Note the deploy gate that actually shipped is **beat the incumbent**, not the spec's >~90% held-out agreement (`docs/spec-memory-relevance-grading.md` A.8), which is demoted to an `auto-promote-safe` flag.

**Write side — generation quality.** Each agent-written memory is stamped with `config.GENERATION_PROMPT_VERSION` (the live value is machine-managed — `ab_selfmod` rewrites it on every promotion, so read `config.GENERATION_PROMPT_VERSION` rather than trusting a number quoted here) in `memories.source_ref` (the provenance join key; precedence `entry["source_ref"] > call param > NULL`). The live generation rules carry a dual-altitude **transferability** lever (generalised principle + specific anchor) and an **in-session duplicate-suppression** rule. Memory keywords **union** (not overwrite) on dedup, so re-encounters enrich findability.

**Two A/B paths exist — do not conflate them:**
- **LIVE per-prompt A/B** (`config.AB_TEST_ENABLED`, ON) — the real experiment. Each user prompt is randomly assigned **arm A** (control = current live rules) or **arm B** (control + one speculative variable from `config.AB_B_INSTRUCTION`, injected that turn). The prompt hook records the arm in `hook_state`; the Stop hook stamps that turn's memories with the arm's `source_ref` version (`config.AB_ARM_VERSIONS` — also machine-managed: `ab_selfmod` promotes B into A and queues a new candidate, so read the symbol, not a number quoted here). Subagents excluded; arm-B injection is post-cache (no cache disturbance). Compare arms with `.venv/bin/python cairn/query.py --delivery-stats` (engagement/grade grouped by generation version + reranker). To stop it: `AB_TEST_ENABLED=False`. To change what B tests: edit `AB_B_INSTRUCTION`.
- **OFFLINE replay harness** (`cairn/ab_writeside.py`) — an analysis tool (not the live experiment, never run on the corpus yet): replays transcripts through prompt-A vs prompt-B → Opus 4.8 judge, BLIND + position-swapped + pairwise (findability / self-sufficiency / fitness), A/B unit = session cohort; metrics dedup rate / findability backtest / self-sufficiency cold-read. CLI: `.venv/bin/python cairn/ab_writeside.py replay|ab --limit N [--dry-run]`.

### Phase 3 — trained cross-encoder student (shipped 2026-07-07)

The read-side gate now has a trained student, not just instrumentation:

- **`cairn/train_reranker.py`** — PAIRWISE fine-tune of a pretrained cross-encoder
  (default `ms-marco-MiniLM`) on `training_data/relevance_silver.jsonl` via
  `MarginRankingLoss` over `(query,memory)` logits; grades induce within-query order.
  Held-out split is **by query** (`--min-gap 2` = train/eval on clear pairs only, since
  adjacent grades dilute agreement to chance). The deploy gate is **beat the incumbent**
  (`resolve_reranker()` scored on the SAME held-out pairs) by `--deploy-margin`, NOT an
  arbitrary 90% (that's demoted to an `auto-promote-safe` flag). Saves to
  `training_data/reranker-student/` only if it beats the incumbent.
- **`cairn/calibrate_bge_floor.py`** — calibrates a suppression floor from the rg labels
  (keep grade-3, drop grade-0). This surfaced and fixed a live bug: the bge floor `0.10`
  was dropping 56% of load-bearing memories (now `0.015`).
- **Deployment** — `config.resolve_reranker()` returns `CROSS_ENCODER_STUDENT_PATH`
  (a LOCAL dir, offline-safe under `HF_HUB_OFFLINE`) when it exists, overriding the
  pretrained base on any device. `training_data/` is gitignored, so the student is a
  **per-machine** artifact — other nodes fall back to ms-marco gracefully until they have
  their own. The student is PAIRWISE-trained (good ordering, compressed absolute scores),
  so its floor is OFF (`CROSS_ENCODER_STUDENT_FLOOR=-100`) — it re-ranks via the
  normalised blend without hard-suppressing; floor calibration is a follow-up.
- **Status** — the first student beats the incumbent decisively (66.0% vs 39.6% held-out
  pairwise agreement) and is live on this machine. It is NOT auto-promote-grade (<90%);
  the lever to improve is MORE LABELS (per-delivery pool + other cairn instances), not
  more epochs — 5 epochs (66.0%) did not beat 3 (67.0%): the student is data-limited.
- **Engagement weak labels** — `train_reranker.py --engagement` merges behavioural
  engagement observations (`memory_deliveries.engaged`) into the training pairs as
  0/3 pseudo-grades, on top of the agent `rg` labels. Merged AFTER `split_by_query`,
  so held-out stays pure agent-rg and the beat-the-incumbent deploy gate is never
  judged on weak labels. Tunable via `--engagement-max-pairs` / `--engagement-min-pos`.
  `ENGAGEMENT_MIN_POS_DEFAULT = 0.2`: the lexical overlap ratio runs median 0.110 /
  p90 0.262, so the obvious-looking 0.5 admits ~1% of positives and starves the pool.
  Note `engaged_score` is bimodal — semantic-second-chance rows store a cosine
  (>= 0.55 by construction), lexical rows a much smaller overlap ratio.
- **Semantic engagement backfill** — `.venv/bin/python cairn/backfill_semantic_engagement.py`
  (`--dry-run`, `--limit`, `--json`) retro-scores historical deliveries, recovering the
  response text from the session transcript since it is not stored on the delivery row.
  Idempotent via the `engaged_method` tag (`lexical` / `semantic` / `semantic-backfill`) —
  that column exists so the two measurement bases stay separable; **a rate computed across
  untagged and tagged rows is invalid**. First run (2026-07-25): 981 examined, 26 rescued
  (2.7%). The rescues look precise but low-recall — rescued rows average agent grade 2.17
  vs 0.84 for non-rescued (n=6, too small to act on). Thresholds are NOT yet calibrated
  against the grade labels; do that the way `calibrate_bge_floor.py` does, not by taste.
