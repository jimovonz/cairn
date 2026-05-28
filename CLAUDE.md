# Cairn — Persistent Memory System

This is the Cairn project — a persistent AI memory system using SQLite, Claude Code hooks, and structured self-assessment.

## Cairn Database

The memory database is at `./cairn/cairn.db`. Use `python3 ./cairn/query.py` to search it.

Query commands:
- `python3 ./cairn/query.py <search>` — full-text search
- `python3 ./cairn/query.py --semantic <query>` — semantic similarity search
- `python3 ./cairn/query.py --recent` — list recent memories
- `python3 ./cairn/query.py --today` — memories from today
- `python3 ./cairn/query.py --since <date>` — memories from date onward (ISO, today, yesterday, 3d, 2w, 1m)
- `python3 ./cairn/query.py --since <date> --until <date>` — memories in a date range
- `python3 ./cairn/query.py --type <type>` — filter by type
- `python3 ./cairn/query.py --session <id>` — filter by session
- `python3 ./cairn/query.py --chain <id>` — show session chain
- `python3 ./cairn/query.py --project <name>` — list memories for a project
- `python3 ./cairn/query.py --projects` — list all projects
- `python3 ./cairn/query.py --label <session_id> <name>` — label a session chain
- `python3 ./cairn/query.py --context <id>` — show conversation context for a memory
- `python3 ./cairn/query.py --history <id>` — show version history
- `python3 ./cairn/query.py --delete <id>` — delete a memory
- `python3 ./cairn/query.py --compact [project]` — dense cairn dump for LLM ingestion
- `python3 ./cairn/query.py --review` — surface low-confidence memories
- `python3 ./cairn/query.py --verify-sources` — analyse source_messages accuracy
- `python3 ./cairn/query.py --backfill` — generate missing embeddings
- `python3 ./cairn/query.py --stats` — database statistics

## Repo Ingestion

Ingest a git repository into Cairn as portable knowledge entries:

- `python3 ./cairn/ingest.py /path/to/repo` — extract, distill, and store (incremental if previously ingested)
- `python3 ./cairn/ingest.py /path/to/repo --dry-run` — preview without storing
- `python3 ./cairn/ingest.py /path/to/repo --project name` — override project name
- `python3 ./cairn/ingest.py /path/to/repo --full` — force full re-ingestion (skip incremental diff)
- `python3 ./cairn/ingest.py /path/to/repo --verbose` — show extraction details

24 extractors: docs, deps, tree, config, schemas, entrypoints, HTTP routes, CLI args, exports, comments, TODOs, env vars, protobuf, CMake flags, event interfaces, DB tables, C/C++ headers, ROS2 interfaces, CAN DBC, Yocto/BitBake, device tree, Docker/CI, tree-sitter AST (Python, JS, TS, TSX, Go, Rust, C, C++), dependency graph. Graph edges queryable via `python3 ./cairn/query.py --deps <project>`.

## Calibration system (Phase 1 + 2)

Phase 1 shipped scaffolding (schema, extractor, stubbed CLI). Phase 2 ships the analyser: a single LLM pass per session over a cleaned transcript produces sectioned JSON across 13 bounded dimensions, writing 8 dimensions to `calibration_rows` and 5 to the existing `memories` table with `source_ref="analyser-session-arc"`. A post-pass scores effectiveness on prior `calibration_deliveries`. See `docs/spec-calibration-system.md` (especially Amendment 1) for the dimension list and design rationale.

Phase 2 commands:
- `cairn-calibration-analyser analyse <jsonl> [--dry-run]` — analyse a single session
- `cairn-calibration-analyser cron [--idle-minutes N] [--limit K]` — one cron pass: walks `~/.claude/projects/*/*.jsonl`, picks idle un-analysed sessions, runs the analyser on up to K of them (oldest first), per-session try/except so one failure doesn't block the rest
- `cairn-calibration-analyser list-idle` — print idle un-analysed session paths

The analyser invokes `claude -p` with `CAIRN_MODE=read-only` so the analyser pass doesn't itself trigger the Stop hook capture path. Defaults to `claude-sonnet-4-6` — the 13-dim sectioned output benefits from a mode-switching-capable model (per cairn entry 2087), and per-call cost is amortised across many future retrievals. Override via `--model` flag or `CAIRN_ANALYSER_MODEL` env var.

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

Phase 1 scaffolding (still applies):

Calibration captures *how to interact with this user* (level, style, preferences) — complementing Cairn knowledge which captures *what is known*. Phase 1 lands foundation only: schema, transcript extractor, and stubbed CLI. The analyser, injector, and dashboard come in later phases. See `docs/spec-calibration-system.md` for the full design.

Schema (created by `init_db.init` / `init_db.init_ephemeral`):
- `calibration_rows` (durable DB) — id, content, kw, qf, source, confidence, pinned, layer, session_scope, supersession, archived_at, effectiveness counters, embedding
- `calibration_deliveries` (ephemeral DB) — turn-indexed log of which rows were injected into which session/turn, with outcome scoring fields

Phase 1 commands (stubs return exit 2 — wiring is verified, no behaviour yet):
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

## Git workflow

All changes MUST be made on feature branches, not main. Branch naming: `feature/<short-description>` or `fix/<short-description>`. Merge to main only after testing.

Before tagging a release on main:
1. Docs are up to date (README.md, ARCHITECTURE.md)
2. `install.sh` and `uninstall.sh` are verified (syntax check + review for unintended changes)
3. All tests pass (`python3 -m pytest tests/`)
4. Tag with semver: `git tag -a v0.X.Y -m "description"`

## Memory system instructions

The memory block format, context retrieval, confidence system, and all LLM behavioral rules are defined in the global rules file deployed by `install.sh`:

- `~/.claude/rules/memory-system.md` — full system documentation (single source of truth)

The project-local `.claude/rules/memory-system.md` is the source for the global copy. Edit it here, then run `./install.sh` to deploy.
