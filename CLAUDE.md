# Cairn — Persistent Memory System

This is the Cairn project — a persistent AI memory system using SQLite, Claude Code hooks, and structured self-assessment.

## Cairn Database

The memory database is at `./cairn/cairn.db`. Use `.venv/bin/python ./cairn/query.py` to search it (all commands below require the venv interpreter — system `python3` lacks pysqlite3, which cairn now requires).

Query commands:
- `.venv/bin/python ./cairn/query.py <search>` — full-text search (hyphenated terms are auto-quoted on FTS5 syntax errors — see `cairn/ftsquery.py`)
- `.venv/bin/python ./cairn/query.py --semantic <query>` — semantic similarity search
- `.venv/bin/python ./cairn/query.py --recent` — list recent memories
- `.venv/bin/python ./cairn/query.py --today` — memories from today
- `.venv/bin/python ./cairn/query.py --since <date>` — memories from date onward (ISO, today, yesterday, 3d, 2w, 1m)
- `.venv/bin/python ./cairn/query.py --since <date> --until <date>` — memories in a date range
- `.venv/bin/python ./cairn/query.py --type <type>` — filter by type
- `.venv/bin/python ./cairn/query.py --session <id>` — filter by session
- `.venv/bin/python ./cairn/query.py --chain <id>` — show session chain
- `.venv/bin/python ./cairn/query.py --project <name>` — list memories for a project
- `.venv/bin/python ./cairn/query.py --projects` — list all projects
- `.venv/bin/python ./cairn/query.py --label <session_id> <name>` — label a session chain
- `.venv/bin/python ./cairn/query.py --context <id>` — show conversation context for a memory
- `.venv/bin/python ./cairn/query.py --history <id>` — show version history
- `.venv/bin/python ./cairn/query.py --delete <id>` — delete a memory
- `.venv/bin/python ./cairn/query.py --compact [project]` — dense cairn dump for LLM ingestion
- `.venv/bin/python ./cairn/query.py --review` — surface low-confidence memories
- `.venv/bin/python ./cairn/query.py --verify-sources` — analyse source_messages accuracy
- `.venv/bin/python ./cairn/query.py --backfill` — generate missing embeddings
- `.venv/bin/python ./cairn/query.py --heal-vec` — index embedded rows missing from the `memories_vec` ANN index (heals the silent vec0-write gap; `--stats` reports the gap)
- `.venv/bin/python ./cairn/query.py --stats` — database statistics

## Repo Ingestion

Ingesting a git repo into Cairn as portable knowledge entries (24 extractors, incremental diff, dependency-graph edges).

Detail: **`/skill:cairn-ingest`** — load it when ingesting a repository or working on the extractors/ingestion pipeline.
## Subagent & review memory capture

How subagent and code-review write-back capture memories (SubagentStop path, review_writeback.py, associated_files overrides, 0.85 dedup).

Detail: **`/skill:cairn-subagent-capture`** — load it when working on how subagent or review findings are captured.
## Code graph navigation (cairn-graph)

`cairn-graph` is a zero-cost, no-LLM query layer over `.code-review-graph/graph.db` (built by the `code-review-graph` tool). **Prefer it over grep/file-reads for structural questions** — it's faster and structurally aware:

- `cairn-graph --location SYMBOL` — where a symbol is defined (replaces grep)
- `cairn-graph --callers SYMBOL` / `--callees SYMBOL` — who calls it / what it calls
- `cairn-graph --impact SYMBOL` — one-line blast radius (`callers:N tests:M files:F`)
- `cairn-graph --context-pack SYMBOL` — body + callers + tests + related cairn memories
- `cairn-graph --tests SYMBOL` — tests covering a symbol
- `cairn-graph --summary` / `--orientation` — repo-level modules/flows/hubs
- `cairn-graph --file-context FILE` — a file's symbols, signatures, fan-in/out, risk tail

This data is also surfaced automatically into sessions: a repo orientation block at session start (Tier 1, prompt hook) and per-file structural context on Read/Edit (Tier 2, pretool hook, deduped once-per-file). Tier 2 resolves the graph from the **accessed file's own location** (not cwd), so it still surfaces when cwd is a parent dir (e.g. `~/Projects`) and the file lives in a subrepo. The Tier 2 hook also recovers file paths from `Bash` commands (`cat`/`sed`/`head` + `cch-edit.py`/`cch-write.py`), so it still fires in environments where Read/Edit are routed through Bash helpers. Both are gated by `GRAPH_ORIENTATION_ENABLED` / `GRAPH_FILE_CONTEXT_ENABLED` and fail open if no graph is built.

### Fleet — keeping every repo graph-ready

`code-review-graph` installs inside cairn's venv (`code-review-graph` is **not** on PATH; resolve via `.venv/bin/` or `cairn.repo_discovery._resolve_crg`). Freshness is **not** driven by git hooks — git-ai (and other git proxies) own the native hook path and don't chain repo hooks, so `.git/hooks/post-commit` is unreliable. Instead:

- **`cairn/graph_fleet.py`** discovers every git repo under the configured roots (`CAIRN_GRAPH_ROOTS`, colon-separated; default = parent of `CAIRN_HOME`), **builds** any missing graph and incrementally **`update`s** existing ones. Run `python3 -m cairn.graph_fleet` (sweep) or `--status`.
- An **hourly cron** runs this sweep — the freshness backbone, daemon-independent. `install.sh` kicks an initial background bootstrap that builds all repos.
- The **prompt hook** (`repo_discovery.kick_graph_build`) also build/updates the current repo's graph on first contact, as a per-session fast path.
- **HEAD-change detection** (`repo_discovery.kick_graph_update_if_head_changed`, fired per-prompt from the prompt hook) catches a *mid-session* branch switch / pull / rebase / commit: it compares the repo's current HEAD against a per-cwd sentinel in `hook_state` and kicks a background `crg update` when it moved. This is the portable path (no native git hooks) — without it, a branch switch would leave the graph stale until the next hourly sweep or new session.
- **Optional real-time layer:** set `CAIRN_GRAPH_WATCH=1` to also register repos with the `code-review-graph` watch daemon (`crg daemon`, 2s poll) for sub-hour freshness. Off by default — the daemon doesn't reliably persist when spawned outside a login shell and churns on volatile files (e.g. cairn's own ephemeral DB), so the cron sweep is the dependable mechanism.

So every repo is graph-ready before first contact, independent of whether cairn has been active in it.

## API proxy (artifact-free injection)

The opt-out bidirectional HTTP proxy that injects context and strips Cairn artifacts without disturbing the prompt cache.

Detail: **`/skill:cairn-proxy`** — load it when working on the proxy or artifact injection/stripping.
## Dev-container support

How containerised sessions reach the host daemon (TCP listener + container injector).

Detail: **`/skill:cairn-devcontainer`** — load it when working on dev-container support.
## Multi-node sync (v2)

Opt-in peer-to-peer LAN replication (Ed25519 pairing, TLS pinning, Lamport LWW, port defaults).

Detail: **`/skill:cairn-sync`** — load it when working on multi-node sync.
## Calibration system (Phases 1–7)

Calibration captures *how to interact with this user* (level, style, preferences), complementing Cairn knowledge which captures *what is known*. The schema, analyser, injector, self-modification, dashboard, and CLI are detailed in `docs/spec-calibration-system.md` and the **`/skill:cairn-calibration`** skill.

**Load `/skill:cairn-calibration` when** working on `calibration_rows`/`calibration_deliveries`, the analyser, or the `cairn-calibration` CLI — or when the user asks to be treated as expert/novice, to mute/delete a calibration rule, to disable calibration, or to show their profile (the natural-language→command mapping lives in the skill).
## Read-side relevance grading & write-side A/B (docs/spec-memory-relevance-grading.md)

Read-side relevance grading (deliveries, rg grades, engagement) and the write-side A/B experiment.

Detail: **`/skill:cairn-relevance-grading`** — load it when reasoning about retrieval ranking, relevance labels, or the write-side A/B.
## Time handling (UTC storage, local display)

Storage is **always UTC** (SQLite `CURRENT_TIMESTAMP` / `datetime('now')`); display
and day-bucketing are **always local**. `cairn/timeutil.py` is the single source of
truth (`fmt_local`, `since_bound_utc` / `until_bound_utc`, `now_local`); the local
zone is `CAIRN_TZ` (IANA name) or the auto-detected system zone.

- **Never** read a stored timestamp as local — a `created_at` of `2026-06-25
  21:04:34` is UTC (e.g. `2026-06-26 09:04:34 NZST`). `query.py` already renders
  local and `--today/--since/--until` already bucket by the local day; the
  dashboard localises every `*_at` field. So trust those outputs as local.
- When querying the DB directly, compare against UTC: use
  `created_at >= datetime('now','-N minutes')` (UTC-relative) or
  `timeutil.since_bound_utc(...)`, **not** a hand-typed local time. (This is the
  exact trap behind the false "no memories today" outage.)

## SQLite library discipline (corruption prevention)

ALL processes that open a cairn DB (`cairn.db`, `cairn-ephemeral.db`) MUST use a
single SQLite library — **pysqlite3** — via the guard at the top of
`cairn/ingest.py` (`try: import pysqlite3 as sqlite3` / `except ImportError`,
falling back to stdlib only under explicit `CAIRN_ALLOW_STDLIB_SQLITE=1`). Mixing
the system stdlib `sqlite3` (an older SQLite) with pysqlite3 on the same WAL file
is a documented cause of `database disk image is malformed` corruption (commit
be91366). Enforced by `tests/test_sqlite_guard.py` (no exemptions — every `cairn/`
and `hooks/` module) and an `install.sh` post-install assertion that cairn actually
resolves `sqlite3 -> pysqlite3`.

- **External writers in OTHER repos** (e.g. `jira/confluence_ingest.py`) that
  import cairn and write `cairn.db` are NOT covered by the in-repo test — copy the
  guard into them manually.
- **Never inspect a live cairn WAL DB with stdlib `sqlite3`** (including a bare
  `import sqlite3` in a throwaway script while the daemon is running) — use
  pysqlite3 or `query.py`. Concurrent stdlib access can corrupt the WAL.

## Active remediation programme (2026-07) — write-path gate

The 2026-07 write-path remediation programme and its gates.

Detail: **`/skill:cairn-remediation`** — load it when proposing or starting Cairn work that touches the write path.
## Git workflow

All changes MUST be made on feature branches, not main. Branch naming: `feature/<short-description>` or `fix/<short-description>`. Merge to main only after testing.

Before tagging a release on main:
1. Docs are up to date (README.md, ARCHITECTURE.md)
2. **Bump the version files to the new version** — they do NOT auto-update and drift silently: `pyproject.toml` (`version = "..."`) and `cairn/__init__.py` (`__version__ = "..."`). Both must match the tag.
3. `install.sh` and `uninstall.sh` are verified (syntax check + review for unintended changes)
4. All tests pass (`.venv/bin/python -m pytest tests/`)
5. Tag with semver, writing a substantive annotation body (it becomes the release notes): `git tag -a v0.X.Y -m "description"`

**After tagging, the "release file" is GitHub Releases** (not a local CHANGELOG) — publish each tag as a release so the page stays complete:
6. `git push origin v0.X.Y` then `gh release create v0.X.Y --verify-tag --title "<tag subject>" --notes-from-tag` (add `--latest` for the newest; `--latest=false` when backfilling older tags).
7. Reconcile periodically — every tag should have a release. This should print nothing:
   `comm -23 <(git tag --sort=version:refname | grep '^v' | sort) <(gh release list --limit 100 --json tagName -q '.[].tagName' | sort)`

## Memory system instructions

The memory block format, context retrieval, confidence system, and all LLM behavioral rules are defined in the global rules file deployed by `install.sh`:

- `~/.claude/rules/memory-system.md` — full system documentation (single source of truth)

The project-local `.claude/rules/memory-system.md` is the source for the global copy. Edit it here, then run `./install.sh` to deploy.
