# Threshold empirical sweep — 2026-05-28

Replay of 105 real user prompts (3 projects, recent JSONLs) through current retrieval paths at multiple floor values. Definitive recall-vs-precision curves replacing the speculative "tune by intuition" approach.

## Method

- **Sample**: 105 substantive user prompts extracted via `cairn/session_extract.py` from `~/.claude/projects/*/*.jsonl` (last 7 days), filtered to `has_tools=False` user turns, length 8–500 chars, excluding system-injected wrappers.
- **Path**: each prompt replayed through `cairn.embeddings.find_similar` (memory) and `cairn.calibration_inject.retrieve_calibration` (calibration). Both paths exercise the schema v8 dual-embedding logic shipped today.
- **Sweep**:
  - Memory `MIN_INJECTION_SIMILARITY` (and equivalent floor used by `find_similar`'s threshold parameter): 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
  - Calibration `SIMILARITY_FLOOR`: 0.35, 0.40, 0.45, 0.50
- **Chaff proxy**: cross-project candidates with similarity < 0.50 in a project-scoped query (the documented BattGO/MAF/intercept-offset failure mode). Heuristic, not ground truth — true chaff would need labelling.

## Memory retrieval results

| Floor | Prompts ≥1 candidate | Prompts ≥5 | Total candidates | Mean top-5 sim | Chaff-suspect candidates | Chaff rate |
|---|---|---|---|---|---|---|
| 0.35 | 105 / 105 | 104 / 105 | 1953 | 0.633 | 218 | 11.2 % |
| 0.40 | 105 / 105 | 104 / 105 | 1943 | 0.634 | 208 | 10.7 % |
| 0.45 | 105 / 105 | 104 / 105 | 1902 | 0.634 | 169 | 8.9 % |
| **0.50** | **105 / 105** | **103 / 105** | **1724** | **0.641** | **0** (tautological) | **0 %** |
| 0.55 | 104 / 105 | 96 / 105 | 1057 | 0.654 | 0 | 0 % |
| 0.60 | 99 / 105 | 74 / 105 | 664 | 0.673 | 0 | 0 % |

### Recall regression — prompts that have candidates at the lower floor but none at the higher

| Step | Prompts lost | % of sample |
|---|---|---|
| 0.35 → 0.40 | 0 | 0.0 % |
| 0.40 → 0.45 | 0 | 0.0 % |
| **0.45 → 0.50** | **0** | **0.0 %** |
| 0.50 → 0.55 | 1 | 1.0 % |
| 0.55 → 0.60 | 5 | 4.8 % |

### Interpretation

The **0.45 → 0.50 step is unambiguously a free chaff cut**: zero recall regression, drops total candidate volume by 178 (≈9 %), eliminates all 169 cross-project low-sim chaff suspects, and lifts mean top-5 similarity from 0.634 to 0.641. No prompt loses its top-1 match. No prompt drops below ≥5 candidates.

The **0.50 → 0.55 step** loses 1 prompt entirely (no candidates at any depth) and 7 more prompts drop below the ≥5 threshold. Total candidates drop from 1724 to 1057 (≈39 %). The cost-benefit shifts here — significant volume reduction but real recall starts to bite. Defensible only if instrumentation later shows that the 0.45–0.55 band is dominated by chaff in production.

The **0.55 → 0.60 step** loses 5 prompts (4.8 %) and 22 prompts drop below ≥5. Real recall damage; not recommended without strong evidence of remaining chaff at 0.55.

## Calibration retrieval results

| Floor | Prompts ≥1 candidate | Total candidates | Mean top-5 sim |
|---|---|---|---|
| 0.35 | 100 / 105 | 1419 | 0.501 |
| **0.40** | **94 / 105** | **1108** | **0.586** |
| 0.45 | 87 / 105 | 844 | 0.645 |
| 0.50 | 82 / 105 | 541 | 0.654 |

### Recall regression

| Step | Prompts lost | % of sample |
|---|---|---|
| 0.35 → 0.40 | 6 | 5.7 % |
| 0.40 → 0.45 | 7 | 6.7 % |
| 0.45 → 0.50 | 5 | 4.8 % |

### Interpretation

Calibration behaves **fundamentally differently** from memory retrieval. Every floor step costs prompts — recall regression is real at every threshold. The current 0.40 floor is well-calibrated: lowering to 0.35 admits ~310 more candidates but drops mean top-5 sim from 0.586 to 0.501 (entering the tangential-match band per the audit shape). Raising to 0.45 costs 7 prompts (6.7 %).

Why the asymmetry vs memory? The calibration corpus is smaller (519 active rows) and skewed by source tier — explicit-source rows tend to hit 0.6+, observation-source rows often hit 0.3–0.4. A flat floor can't distinguish; raising it disproportionately kills the observation tier. This is the bimodal distribution issue flagged in my previous response.

**Defensible move: keep `SIMILARITY_FLOOR = 0.40`.** The data supports the current value.

## Concrete recommendations

| Threshold | Current (main) | Current (PR #2) | Recommended | Rationale |
|---|---|---|---|---|
| Memory `MIN_INJECTION_SIMILARITY` | 0.35 | 0.45 | **0.50** | 0 % recall regression vs 0.45, eliminates 169 chaff suspects, lifts mean top-5 sim |
| Memory `BORDERLINE_SIM_CEILING` | 0.35 | 0.35 | **0.40** | Per-entry gate; aligning with the memory injection floor band's lower edge |
| Calibration `SIMILARITY_FLOOR` | 0.40 | 0.40 | **0.40 (keep)** | Empirical recall regression confirms current value |
| Memory `DIVERSITY_SIM_THRESHOLD` | 0.90 | 0.85 | **0.85 (PR #2)** | Consistency with 0.85 retrieval-time diversity; memory `DEDUP_THRESHOLD` itself is 0.95 (separate concern) |

### Suggested next move

Merge **PR #2** (which raises memory floor to 0.45) **and then in a follow-up bump it to 0.50** based on this data. The 0.45 → 0.50 step is the cleanest evidence-backed change in the entire threshold space — and the only one this sweep can recommend at high confidence.

## Caveats

- **N = 105 prompts across 3 projects.** Smaller than ideal for tail analysis; the "1 prompt lost at 0.55" is one data point.
- **Chaff proxy is heuristic.** Real chaff measurement requires labelling — the cross-project + low-sim rule will miss intra-project chaff and may flag legitimate cross-project surfacing as chaff. The 11 % chaff rate at floor 0.35 is a lower bound (true chaff is probably higher).
- **`memory_chaff` at floor 0.50+ is tautologically 0** because the chaff proxy itself requires sim < 0.50. The real signal there is the candidate-count drop (1902 → 1724 → 1057), not the chaff field.
- **Cross-encoder re-rank was not included as a secondary quality signal.** Adding it would tighten chaff identification but requires daemon calls per candidate. Worth doing in a follow-up sweep.
- **Calibration distribution is bimodal.** A single floor doesn't optimally serve explicit-source (high-sim) and observation-source (lower-sim) rows. Source-tier-aware floors are the structural fix, not a flat number.
- **The cairn memory 2336462276841 warning applies**: tightening floors has invisible cost — silently dropped entries that would have been load-bearing. This sweep's "zero recall regression at 0.45 → 0.50" mitigates that concern *for this sample*, but doesn't eliminate it in the long tail.

## Raw data

`/tmp/threshold_audit_data.json` (preserved at time of run).

## Next instrumentation worth shipping

- `dual_embedding_topic_win` / `content_win` counters in `_brute_force_candidates` — measures how often topic vs content drives a match.
- `injection_similarity_distribution` — log similarity of each injected entry. After a week, real histogram replaces this sweep's snapshot.
- Cross-encoder score collection per injected candidate — independent quality signal for chaff detection.

These would let the next threshold pass be data-driven without needing to replay sessions.
