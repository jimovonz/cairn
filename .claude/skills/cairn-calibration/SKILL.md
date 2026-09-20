---
name: cairn-calibration
description: Cairn calibration system — schema (calibration_rows, calibration_deliveries), the session analyser, the prompt injector, self-modification, the dashboard tab, and the cairn-calibration CLI. Load when working on calibration rows or deliveries, the analyser, or when the user expresses an interaction preference (expert/novice), asks to mute/delete a rule, or asks to see their profile.
---

## Calibration system (Phases 1–7)

Phase 1 shipped scaffolding (schema, extractor, stubbed CLI). Phase 2 ships the analyser: a single LLM pass per session over a cleaned transcript produces sectioned JSON across 13 bounded dimensions, writing 8 dimensions to `calibration_rows` and 5 to the existing `memories` table with `source_ref="analyser-session-arc"`. A post-pass scores effectiveness on prior `calibration_deliveries`. See `docs/spec-calibration-system.md` (especially Amendment 1) for the dimension list and design rationale.

Phase 2 commands:
- `cairn-calibration-analyser analyse <jsonl> [--dry-run]` — analyse a single session
- `cairn-calibration-analyser cron [--idle-minutes N] [--limit K]` — one cron pass: walks `~/.claude/projects/*/*.jsonl`, picks idle un-analysed sessions (idle ≥ `--idle-minutes`, default 15, so active sessions are left alone), runs the analyser on up to K of them (oldest first), per-session try/except so one failure doesn't block the rest
- `cairn-calibration-analyser list-idle` — print idle un-analysed session paths

The analyser invokes `claude -p` with `CAIRN_MODE=read-only` so the analyser pass doesn't itself trigger the Stop hook capture path; it also sets `CAIRN_NO_INJECT=1` to skip prompt-hook context injection (recall is noise for an analysis pass, and these passes dominated first-prompt injection volume). Defaults to `claude-sonnet-4-6` — the 13-dim sectioned output benefits from a mode-switching-capable model (per cairn entry 2087), and per-call cost is amortised across many future retrievals. Override via `--model` flag or `CAIRN_ANALYSER_MODEL` env var.

**No per-dimension count caps** — per cairn entry 623 (format enforced mechanically, content enforced editorially), the analyser does not cap how many items each dimension emits. The model's editorial judgment governs quantity, guided by the prompt's "Quality > quantity. Never pad" rule. Two structural guards remain: (a) `ENVELOPE_CHARS_MAX` (60K chars) — `analyser_envelope_exceeded` metric fires above this; claude -p truncates upstream of us anyway (cairn entry 1734); (b) cosine-0.85 dedup at insert filters near-duplicates regardless of count.

**Subagent filter** — sessions are skipped unless they have at least `MIN_SUBSTANTIVE_TURNS` (4) substantive turns AND `MIN_CLEANED_CHARS` (500) of cleaned content. This avoids spending Sonnet calls on heartbeat / compaction-child / subagent transcripts. `--force` bypasses.

**Incremental analysis** — per-session state is recorded in `hook_state` under key `calibration_analyser_state` (last_turn_count, last_analysed_at, first_analysed_at). A previously-analysed session is re-eligible only when its turn count has grown by `INCREMENTAL_TURN_THRESHOLD` (10) turns. Re-runs pass the prior calibration rows to the LLM via the prompt ("PRIOR CALIBRATION ROWS FROM THIS SESSION — do NOT re-emit") and additionally apply mechanical cosine-similarity dedup at 0.85 against existing rows. Long-running multi-day sessions get periodic top-ups without paying per appended turn.

**Dedup** — both write paths apply cosine-0.85 dedup before INSERT. `write_calibration_rows` dedups against the full non-archived `calibration_rows` set. `write_session_memories` dedups against prior analyser-written rows only (filtered by `source_ref="analyser-session-arc"`) — per-turn writes have write-time priority per cairn entry 3302 and are never blocked by an analyser duplicate.

**Per-qf symmetric retrieval (schema v7)** — calibration retrieval scores each row as `max_i cos(prompt_embedding, qf_i_embedding)` using the `calibration_qf_embeddings` sidecar table. The previous single-vector design joined `content+kw+qf` into one row embedding, conflating third-person content with first-person qf phrasings — empirically this clustered prompt similarities at 0.20-0.36, below the 0.40 floor. Per-qf retrieval embeds each qf string individually at write time (analyser `write_calibration_rows`) and stores them in the sidecar (PK row_id+qf_index, FK ON DELETE CASCADE). Rows without sidecar entries fall back to the legacy single-vector cosine — graceful migration, no flag day. Backfill for existing rows: `python3 cairn/calibration_qf_backfill.py` (idempotent, local embedder, no LLM cost).

**Anti-hedge prompt** — the analyser prompt forbids "may or may not", "possibly", "unclear if", and similar hedge phrasings. Emitting an empty array for a dimension is preferred over a hedged row.

Effectiveness scoring updates `calibration_deliveries.outcome` AND bumps the corresponding counter (`followed_count` / `ignored_count` / `corrected_count`) on `calibration_rows`. Metrics: `analyser_session_processed` on success, `analyser_session_failed` with error preview on failure.

**Phase 4 — agent natural-language → CLI patterns.** The CLI is *agent-invoked from intent, never user-typed*. When the user says something like the LHS column, invoke the RHS command:

| User says | Agent invokes |
|---|---|
| "treat me as an expert in X" / "stop explaining basics" | `cairn-calibration mode --level expert` |
| "I'm new to X, give more context" | `cairn-calibration mode --level novice` |
| "forget that thing about X" / "stop reminding me about X" | `cairn-calibration mute <row_id>` (look up id via `--show-profile X`) |
| "actually that rule is wrong" / "I never said that" | `cairn-calibration delete <row_id>` |
| "for this session only, ..." | append `--session-only` to mute/disable/mode |
| "turn off calibration" / "stop the priming" | `cairn-calibration disable` |
| "what do you think I prefer?" / "show my profile" | `cairn-calibration --show-profile` |
| "I prefer X" / "always do Y" / "never Z" | `cairn-calibration add --source explicit --content "..."` |
| "anything you want me to review?" / "what's flagged?" | `cairn-calibration --review` |

**Phase 5 — Calibration dashboard tab** (`http://localhost:5174/`) with 4 V1 panels: Profile (rows by source/confidence/pinned, follow rate per row), Effectiveness (per-row deliveries/follow%/ignore/correct, low-follow flagged), Review Queue (Tier 2 surfaced items with type/detail/age), Summary cards (total rows, deliveries, follow rate, review-queue count, flagged count). Endpoints: `/api/calibration/profile|effectiveness|review-queue|session/<id>`. All 12 metric events from spec §7 instrumented (`calibration_row_{written,delivered,followed,ignored,corrected,archived,promoted,superseded}`, `calibration_review_surfaced`, `calibration_dedup_filtered`, `analyser_session_{processed,failed}`).

**Phase 6 — self-modification** (`cairn-calibration-selfmod`): Tier 1 autonomous — `auto_archive_low_follow` (≥10 deliv, <20% followed), `auto_promote_corroborated` (≥80% follow + ≥3 distinct sessions), `decay_unused` (multiplicative half-life decay per source tier). Tier 2 surfaced into `calibration_review_queue` — low-follow rephrase candidates (40–60% band), promotion candidates that missed auto-threshold. Tier 3 (analyser prompt, retrieval weights, system architecture) stays manual by design.

**Phase 7 — CLAUDE.md import** (`cairn-calibration-import-claude-md [path]`): one-shot scanner for first-person preference statements ("I prefer X", "Always/Never Y", "Stop Z"). Idempotent via SHA tracking in `hook_state`. Seeds rows as pinned `explicit` with confidence 0.90.

Foundation (Phase 1, still current):

Calibration captures *how to interact with this user* (level, style, preferences) — complementing Cairn knowledge which captures *what is known*. Phase 1 laid the foundation — schema, transcript extractor, CLI — and the analyser, injector, self-modification, and dashboard are all now shipped (Phases 2–7 above). See `docs/spec-calibration-system.md` for the full design.

Schema (created by `init_db.init` / `init_db.init_ephemeral`):
- `calibration_rows` (durable DB) — id, content, kw, qf, source, confidence, pinned, layer, session_scope, supersession, archived_at, effectiveness counters, embedding
- `calibration_deliveries` (ephemeral DB) — turn-indexed log of which rows were injected into which session/turn, with outcome scoring fields

CLI commands (all implemented; agent-invoked from natural-language intent, never user-typed):
- `python3 ./cairn/session_extract.py <jsonl>` — clean a session JSONL to user/assistant text only, dropping tool blocks, thinking, `<cairn_context>`, `<system-reminder>`, and `[cm]` link-defs. Flags: `--with-tools`, `--corrections-only`, `--turn-range A-B`, `--last-N-minutes N`, `--json`.
- `cairn-calibration --show-profile [subject]` — show calibration profile
- `cairn-calibration --review` — Tier 2 review queue
- `cairn-calibration --history <row_id>` — supersession/archive history
- `cairn-calibration add --source <explicit|correction|observation|meta-assessment> --content "..." [--scope X] [--pin]`
- `cairn-calibration mute <row_id> [--session-only]` / `unmute <row_id>`
- `cairn-calibration disable [--session-only]` / `enable`
- `cairn-calibration mode --level <novice|expert> [--session-only]`
- `cairn-calibration delete <row_id>`

The CLI is **agent-invoked from natural-language intent**, never user-typed.
