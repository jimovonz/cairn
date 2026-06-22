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

## Subagent & review memory capture

Two complementary paths capture knowledge that would otherwise be lost.

**SubagentStop capture (general).** Subagents (Task tool) emit `[cm]` memory blocks into their *own* transcripts, which the main-session `Stop` hook never sees. A `SubagentStop` hook (registered in `templates/global-settings.json`, no matcher → all agent types) routes the subagent's final message into `hooks/stop_hook.py`. The existing `is_subagent` branch (detected via `agent_id` **or** `hook_event_name=="SubagentStop"`) stores volunteered entries opportunistically and **skips enforcement** (no re-prompting a subagent). It reads the subagent's own transcript via `agent_transcript_path` (`own_transcript`) for storage + `--context` excerpts, while `session_id`/`transcript_path` stay the parent session so memories chain to it. Entries dedup at cosine 0.85 like any other.

**Review write-back (code-attached).** `cairn-review-writeback` (`cairn/review_writeback.py`) persists **durable review rationale** — the *why* that survives the fix (intentional couplings, accepted trade-offs, justified decisions captured at merge), **not raw transient bug findings** (a "PR has bug X" claim becomes false once fixed+merged and self-invalidates in cairn; default `type` is `decision` to nudge this) — keyed to the *target repo* and *changed file/symbol*, so `cairn-graph --knowledge SYMBOL` surfaces it later — it joins on `associated_files LIKE '%path%'` **and** an FTS `MATCH` on the symbol (FTS indexes topic/content/keywords/facts). Each finding sets explicit `associated_files=[abs, rel]` (a per-entry override added to `hooks/storage.insert_memories`, taking precedence over transcript-derived files) and carries the symbol in `content`/`keywords`/`facts`. It registers a synthetic `review-<project>-<commit>` session tagged to the target project (so project-scoped retrieval also surfaces findings), batches under `MAX_MEMORIES_PER_RESPONSE`, and dedups at cosine 0.85 (re-running a review is idempotent — no archiving).

Input is JSON on stdin (or `--file`): `{repo, project?, commit?, findings:[{file, symbol?, line?, type?, topic?, content, severity?, pr?, keywords?}]}`. Flags: `--dry-run`, `--json`. Invoke it from a review's end-step, e.g.:

| Situation | Agent invokes |
|---|---|
| A review surfaced a *durable* reason worth keeping (intentional coupling, justified trade-off, decision at merge) | pipe rationale JSON to `cairn-review-writeback` (resolve via `.venv/bin/`) |
| Raw transient bug findings on an open PR | do NOT write back — they self-invalidate once the bug is fixed |
| Want to preview the keyed memories first | add `--dry-run` |

## Code graph navigation (cairn-graph)

`cairn-graph` is a zero-cost, no-LLM query layer over `.code-review-graph/graph.db` (built by the `code-review-graph` tool). **Prefer it over grep/file-reads for structural questions** — it's faster and structurally aware:

- `cairn-graph --location SYMBOL` — where a symbol is defined (replaces grep)
- `cairn-graph --callers SYMBOL` / `--callees SYMBOL` — who calls it / what it calls
- `cairn-graph --impact SYMBOL` — one-line blast radius (`callers:N tests:M files:F`)
- `cairn-graph --context-pack SYMBOL` — body + callers + tests + related cairn memories
- `cairn-graph --tests SYMBOL` — tests covering a symbol
- `cairn-graph --summary` / `--orientation` — repo-level modules/flows/hubs
- `cairn-graph --file-context FILE` — a file's symbols, signatures, fan-in/out, risk tail

This data is also surfaced automatically into sessions: a repo orientation block at session start (Tier 1, prompt hook) and per-file structural context on Read/Edit (Tier 2, pretool hook, deduped once-per-file). The Tier 2 hook also recovers file paths from `Bash` commands (`cat`/`sed`/`head` + `cch-edit.py`/`cch-write.py`), so it still fires in environments where Read/Edit are routed through Bash helpers. Both are gated by `GRAPH_ORIENTATION_ENABLED` / `GRAPH_FILE_CONTEXT_ENABLED` and fail open if no graph is built.

### Fleet — keeping every repo graph-ready

`code-review-graph` installs inside cairn's venv (`code-review-graph` is **not** on PATH; resolve via `.venv/bin/` or `cairn.repo_discovery._resolve_crg`). Freshness is **not** driven by git hooks — git-ai (and other git proxies) own the native hook path and don't chain repo hooks, so `.git/hooks/post-commit` is unreliable. Instead:

- **`cairn/graph_fleet.py`** discovers every git repo under the configured roots (`CAIRN_GRAPH_ROOTS`, colon-separated; default = parent of `CAIRN_HOME`), **builds** any missing graph and incrementally **`update`s** existing ones. Run `python3 -m cairn.graph_fleet` (sweep) or `--status`.
- An **hourly cron** runs this sweep — the freshness backbone, daemon-independent. `install.sh` kicks an initial background bootstrap that builds all repos.
- The **prompt hook** (`repo_discovery.kick_graph_build`) also build/updates the current repo's graph on first contact, as a per-session fast path.
- **HEAD-change detection** (`repo_discovery.kick_graph_update_if_head_changed`, fired per-prompt from the prompt hook) catches a *mid-session* branch switch / pull / rebase / commit: it compares the repo's current HEAD against a per-cwd sentinel in `hook_state` and kicks a background `crg update` when it moved. This is the portable path (no native git hooks) — without it, a branch switch would leave the graph stale until the next hourly sweep or new session.
- **Optional real-time layer:** set `CAIRN_GRAPH_WATCH=1` to also register repos with the `code-review-graph` watch daemon (`crg daemon`, 2s poll) for sub-hour freshness. Off by default — the daemon doesn't reliably persist when spawned outside a login shell and churns on volatile files (e.g. cairn's own ephemeral DB), so the cron sweep is the dependable mechanism.

So every repo is graph-ready before first contact, independent of whether cairn has been active in it.

## API proxy (artifact-free injection)

`cairn/proxy/` is an opt-out bidirectional HTTP proxy between Claude Code and the Anthropic API. It injects retrieved context into outbound requests **without disturbing the cacheable prefix** (Anthropic prompt cache stays byte-exact — a prompt-cache integrity guard verifies this) and strips every Cairn artifact (`<memory>`/`[cm]` blocks, `<cairn_context>`, system reminders) from inbound responses, capturing the stripped artifacts via `sidecar.py` for the hook pipeline. It is the artifact-hiding alternative to tag-stripping: capture/injection keep working even if Claude Code changes its tag rendering.

- `server.py` runs a detached daemon (`start`/`stop`/`restart`, port-specific PID file) on `127.0.0.1:8789` (`CAIRN_PROXY_PORT`). It `dup2`s fd0←/dev/null and fd1/fd2←log so it never holds an inherited stdout pipe open.
- `install.sh` enables it **by default** (opt out with `CAIRN_PROXY_ENABLED=0`), installs a `c` shell launcher (marked, idempotent rc block — `c` routes through the proxy, bare `claude` stays direct), and a `*/5` keep-alive cron (`start` is idempotent).
- Context is injected only on agentic requests.

## Dev-container support

The daemon exposes a **TCP listener on port 47390** alongside its Unix socket so container shims can dial the host daemon via `cairn_recall` / `cairn_remember` opcodes. `cairn/container_injector.py` injects context inside the container, with an extension auto-installer and VSIX staging, so a containerised session reaches the same host cairn as the native session.

## Calibration system (Phases 1–7)

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
