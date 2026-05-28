# Cairn v0.14 — Implementation Spec

Handover document for implementation by a smaller model. Each section is self-contained. Execute in the declared order — later changes assume earlier ones landed.

## Ground rules

- **Branch per change.** Name `feature/<slug>` or `fix/<slug>`. Merge to `main` only after tests pass.
- **One branch can land multiple closely-related changes** (e.g. §5 + §10 are one-liners, bundle them).
- **Do not** refactor outside the stated scope. No "while I'm here" cleanups.
- **Tests first for correctness changes** (§1, §3, §7). Test after for scoring/UX changes.
- **Verify before landing:** `python3 -m pytest tests/ -x` passes. `python3 cairn/query.py --stats` works.
- **Commit style:** matches recent log (see `git log --oneline -10`) — short imperative subject, optional body, no attribution footer unless user requests.
- **Do not skip hooks** on commit — the Stop hook is part of the system under test.

---

## Change order & dependency graph

```
§1  Ephemeral DB split ──┬── §6  Systemic failure alerting (uses ephemeral counters)
                         └── §4  Retention dashboard (touches dashboard.py same area)
§3  Re-embed on edit ────── (independent, small)
§5  Recency weight → 0 ──── (independent, one-line)
§10 Doc sync ────────────── (independent, doc only)
§8  Delete dead code ────── (independent)
§2  Project dominance ───── (independent, retrieval change)
§7  -! reason audit trail ─ (independent, small schema add)
§9  XML chrome trim ─────── (do after §7 to avoid rework)
§4  Retention dashboard ─── (largest, lands last)
```

Recommended landing order: §10, §5, §8, §3, §1, §6, §2, §7, §9, §4.

---

# §1 — Ephemeral DB split

## Why

Seven SQLite corruption events in the past week (`cairn/cairn.db.corrupt-*`). Pattern in `cairn/corruption.log`: `hook_state` and `metrics` tables — the hot-path ephemeral writers — corrupt first, then cascade to `memories`. Main memory table usually survives. Isolating ephemeral writes into a separate DB file contains the blast radius — corruption there only costs session state, not knowledge.

## Scope

Move these tables from `cairn.db` → new `cairn-ephemeral.db`:
- `hook_state`
- `metrics`
- `pair_assessments` (offline compute cache — safe to regenerate)

Keep in `cairn.db`:
- `memories`, `memory_history`, `memories_fts`, `memories_vec`
- `sessions`, `correction_triggers`, `schema_version`

## Files

| File | Change |
|------|--------|
| `cairn/config.py` | Add `EPHEMERAL_DB_PATH` |
| `cairn/init_db.py` | Split schema; add `init_ephemeral_db()` |
| `hooks/hook_helpers.py` | Add `get_ephemeral_conn()`; keep `get_conn()` for main |
| `hooks/storage.py` | Route metrics + hook_state writes through ephemeral conn |
| `cairn/daemon.py` | Any state writes → ephemeral |
| `cairn/dashboard.py` | Read from both (use `ATTACH DATABASE`) |
| `cairn/query.py` | `--stats` reads both |
| `cairn/recover.py` | Handle both DBs; ephemeral recovery = drop & recreate |
| `install.sh` | Ensure `cairn-ephemeral.db` created on install |
| `uninstall.sh` | Remove `cairn-ephemeral.db` |
| `.gitignore` | Add `cairn-ephemeral.db*` |

## Implementation steps

1. **Add config** (`cairn/config.py`):
   ```python
   import os
   CAIRN_DIR = os.path.join(os.path.dirname(__file__))
   EPHEMERAL_DB_PATH = os.environ.get("CAIRN_EPHEMERAL_DB_PATH",
                                       os.path.join(CAIRN_DIR, "cairn-ephemeral.db"))
   ```

2. **Split schema** (`cairn/init_db.py`):
   - Rename the existing `init_db()` to take a `path` argument defaulting to `DB_PATH`.
   - Split the CREATE TABLE statements into two functions:
     - `init_main_db(path)`: memories, memory_history, sessions, correction_triggers, FTS/vec, schema_version
     - `init_ephemeral_db(path)`: metrics, hook_state, pair_assessments
   - Keep existing migration code in `init_main_db`; ephemeral has no migration (drop & recreate is cheap).
   - Export both.

3. **Add connection accessor** (`hooks/hook_helpers.py`):
   ```python
   from cairn.config import EPHEMERAL_DB_PATH

   def get_ephemeral_conn() -> sqlite3.Connection:
       conn = sqlite3.connect(EPHEMERAL_DB_PATH)
       conn.execute("PRAGMA busy_timeout=5000")
       conn.execute("PRAGMA journal_mode=WAL")
       return conn
   ```
   Do NOT change `get_conn()`.

4. **Route writes** (`hooks/storage.py`):
   - Find every call that writes to `metrics` or `hook_state`. Grep: `INSERT INTO metrics`, `INSERT INTO hook_state`, `UPDATE hook_state`, `DELETE FROM hook_state`, `DELETE FROM metrics`.
   - Replace the `conn = get_conn()` in those functions with `conn = get_ephemeral_conn()`.
   - Reads of metrics/hook_state → also ephemeral.
   - Confidence updates, memory inserts, memory_history writes → stay on `get_conn()`.

5. **Dashboard reads** (`cairn/dashboard.py`):
   - Where dashboard queries JOIN memories + metrics/hook_state, use `ATTACH DATABASE`:
     ```python
     conn = sqlite3.connect(DB_PATH)
     conn.execute(f"ATTACH DATABASE '{EPHEMERAL_DB_PATH}' AS ephem")
     # queries reference `ephem.metrics`, `ephem.hook_state`
     ```
   - Or query each DB independently and merge in Python where cleaner.

6. **One-time migration script** (`cairn/migrate_ephemeral.py`, new file):
   ```python
   """Migrate metrics, hook_state, pair_assessments from cairn.db to cairn-ephemeral.db.
   Runs once. Idempotent — safe to re-run."""
   # 1. Create cairn-ephemeral.db if missing via init_ephemeral_db()
   # 2. For each table: SELECT * FROM main.<table>, INSERT OR REPLACE into ephemeral
   # 3. After verification, DROP <table> from main cairn.db
   # 4. VACUUM main
   ```
   Invoke from `install.sh` during upgrade.

7. **Recover** (`cairn/recover.py`):
   - `recover_main()` — existing logic, unchanged.
   - `recover_ephemeral()` — new: rename corrupt file to `.corrupt-<ts>`, call `init_ephemeral_db()` on fresh path. No need to preserve data.

## Tests

- `tests/test_storage.py` — verify `save_hook_state` writes to ephemeral path, not main.
- `tests/test_e2e_pipeline.py` — end-to-end memory capture still works with split DBs.
- `tests/test_integration.py` — dashboard query that joins both DBs returns correct data.
- New: `tests/test_ephemeral_split.py` — corrupt ephemeral (truncate the file); verify main memories still queryable and retrievable.

## Acceptance

- `python3 -m pytest tests/ -x` passes.
- `python3 cairn/query.py --stats` shows correct counts (reads from both).
- Dashboard loads (`python3 cairn/dashboard.py` then visit localhost).
- Deleting `cairn-ephemeral.db` and running any hook recreates it.
- `ls cairn/*.db` shows both files.

---

# §2 — Project dominance softening

## Why

`hooks/retrieval.py:174-201` gates global-scope results behind a threshold ONLY when project has zero results. A cross-project memory at sim=0.60 is dropped in favour of a project memory at sim=0.25. This directly blocks the system's stated "seamless benefit of prior sessions" goal — valuable cross-project knowledge is invisibly suppressed.

## Scope

Always include global results above a hard floor, regardless of project hit count.

## Files

| File | Change |
|------|--------|
| `cairn/config.py` | Add `GLOBAL_HARD_FLOOR = 0.50` |
| `hooks/retrieval.py` | Modify `hybrid_search` scope filter |

## Implementation

1. **Add config:**
   ```python
   # cairn/config.py
   GLOBAL_HARD_FLOOR = 0.50  # Global-scope results above this similarity always surface,
                             # even when project scope has results. Prevents cross-project blind spot.
   ```

2. **Modify `hooks/retrieval.py:174-201`** (the scope-splitting block in `hybrid_search`):
   Current logic (paraphrased):
   ```python
   if project_results:
       # use only project, apply higher threshold to global
   else:
       # use global with project threshold
   ```
   New logic:
   ```python
   # Always include project results above project threshold
   # Always include global results above max(global_threshold, GLOBAL_HARD_FLOOR)
   # Union, dedup by id, keep per-scope limits
   ```
   Preserve the existing `limit` / per-scope caps — soften the gate, not the size budget.

## Tests

- `tests/test_retrieval_quality.py` — add case: project has one result at sim=0.30, global has one at sim=0.60. Assert both returned.
- `tests/test_retrieval_benchmark.py` — verify recall@5 does not regress on existing benchmark.

## Acceptance

- New test passes.
- `test_retrieval_quality.py` and `test_retrieval_benchmark.py` pass overall.
- Manual: in a ProjectA session, query for a ProjectB topic you know exists at high similarity → memory surfaces.

---

# §3 — Re-embed on content edit

## Why

`hooks/storage.py` updates memory `content` and bumps `updated_at` but never invalidates the `embedding` BLOB. After any correction/audit, semantic search uses a vector pointing at the pre-edit content. Silent correctness bug.

## Scope

When a memory's content changes, clear its embedding. Backfill is already implemented (`query.py --backfill`); add an on-write trigger so NULL embeddings accumulate and get refilled by the daemon.

## Files

| File | Change |
|------|--------|
| `cairn/init_db.py` | Add AFTER UPDATE trigger on memories |
| `hooks/storage.py` | Null embedding on content-change update paths |
| `cairn/query.py` | `--update` path nulls embedding |
| `hooks/stop_hook.py` | After applying confidence updates or rewrites, kick off `backfill_null_embeddings()` if embedder available |
| `cairn/embeddings.py` | Helper `backfill_null_embeddings(limit=N)` — refill NULL rows using daemon |

## Implementation

1. **Schema trigger** (`cairn/init_db.py`, inside `init_main_db`):
   ```sql
   CREATE TRIGGER IF NOT EXISTS null_embedding_on_content_edit
   AFTER UPDATE OF content ON memories
   FOR EACH ROW
   WHEN NEW.content != OLD.content
   BEGIN
       UPDATE memories SET embedding = NULL WHERE id = NEW.id;
   END;
   ```
   Add a schema_version bump so existing DBs get the trigger on next init.

2. **Helper** (`cairn/embeddings.py`):
   ```python
   def backfill_null_embeddings(conn, limit: int = 20) -> int:
       """Find up to `limit` memories with NULL embedding, compute via daemon, write back.
       Returns count embedded. No-op if daemon unavailable."""
   ```
   Reuse daemon path only (no slow in-process fallback — don't block hooks).

3. **Trigger on write paths** (`hooks/stop_hook.py`):
   After `apply_confidence_updates` or any memory rewrite, call `backfill_null_embeddings(conn, limit=20)`. Wrap in try/except — failure must not block the hook.

## Tests

- `tests/test_storage.py` — update a memory's content, assert embedding is now NULL in the DB.
- `tests/test_e2e_pipeline.py` — update content, run backfill, verify semantic search returns updated result (old content no longer matches, new content does).

## Acceptance

- Trigger fires on every content change.
- `SELECT count(*) FROM memories WHERE embedding IS NULL` trends to 0 within a minute of edit (assuming daemon up).
- Tests pass.

---

# §4 — Retention & storage dashboard

## Why

Session JSONLs (Claude Code transcripts) are user-controlled resources. Once purged, `query.py --context <id>` silently degrades — the one-liner memory persists but verbatim recovery is gone. User has no visibility of this erosion. Also no way to triage low-value sessions before purging.

## Scope

Add:
1. **Retention panel** (read-only) — active retention window, session disk usage, age histogram, at-risk sessions, context-recovery coverage.
2. **Session triage view** — per-session value score, filters, drill-in.
3. **Snapshot action** — capture source excerpts for a session's memories into a side table, so `--context` survives JSONL purge.
4. **Selective removal action** — delete JSONL for a session under confirmation.
5. **Capture window** — snapshot all transcripts in a date range.

Ship API + frontend in the same PR (memory id:3573 flags the anti-pattern of API-first-UI-lagging).

## Files

| File | Change |
|------|--------|
| `cairn/init_db.py` | Add `memory_source_excerpt` table |
| `hooks/stop_hook.py` | On memory insert, snapshot source excerpt if transcript available |
| `cairn/query.py` | `--context` reads excerpt side table first, transcript as fallback |
| `cairn/dashboard.py` | New endpoints: `/api/retention`, `/api/session-triage`, `/api/snapshot-session/<id>`, `/api/purge-session/<id>`, `/api/capture-window` |
| `cairn/static/index.html` | New "Retention" tab with panels + actions |

## Schema

```sql
CREATE TABLE IF NOT EXISTS memory_source_excerpt (
    memory_id INTEGER PRIMARY KEY,
    session_id TEXT,
    transcript_path TEXT,
    excerpt TEXT NOT NULL,
    context_before TEXT,
    context_after TEXT,
    captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_source_excerpt_session ON memory_source_excerpt(session_id);
```

## API endpoints

### `GET /api/retention`
Returns:
```json
{
  "transcripts_dir": "/home/james/.claude/projects/",
  "retention_note": "Retention is user-controlled — see Claude Code settings for cleanup policy.",
  "total_sessions": 917,
  "sessions_with_transcript": 412,
  "sessions_transcript_purged": 505,
  "total_transcript_bytes": 458612384,
  "age_histogram": [{"bucket": "0-7d", "count": 42}, {"bucket": "7-30d", "count": 180}, ...],
  "at_risk_count": 38,
  "excerpt_coverage_pct": 23.4,
  "memories_recoverable_via_excerpt": 988,
  "memories_recoverable_via_transcript": 1800,
  "memories_no_recovery": 1433
}
```

### `GET /api/session-triage?sort=value_score&order=asc&limit=50&offset=0`
Per-session value score = `memories_retrieved_elsewhere / max(memories_generated, 1)`. Filterable by: no_retrievals=true, max_age_days=N, min_size_bytes=N.
Returns list of `{session_id, label, created_at, memory_count, retrieved_count, value_score, transcript_exists, transcript_bytes, excerpt_coverage}`.

### `POST /api/snapshot-session/<session_id>`
For each memory in the session, extract ±N lines of transcript context around the memory's `source_ref` (or created_at timestamp) and insert into `memory_source_excerpt`. Idempotent. Returns `{snapshotted: N, skipped: M}`.

### `POST /api/purge-session/<session_id>`
Requires `confirm=true` query param. Deletes the JSONL file under `~/.claude/projects/…`. Does NOT delete memories. Returns `{deleted: path, freed_bytes: N}`.

### `POST /api/capture-window`
Body `{"since": "2026-03-01", "until": "2026-03-15"}`. Iterates sessions in range, calls snapshot logic for each. Returns counts.

## Stop hook excerpt capture

In `hooks/stop_hook.py`, after `insert_memories` returns successfully:
- For each inserted memory ID, extract transcript lines around the response that produced it (session_id + timestamp or source_ref column).
- Store excerpt of ±15 lines (config-driven: `SOURCE_EXCERPT_LINES_BEFORE`, `SOURCE_EXCERPT_LINES_AFTER`).
- Write to `memory_source_excerpt`.
- Wrap in try/except — excerpt failure must not block memory capture.

## `--context` fallback order

In `cairn/query.py` `--context <id>`:
1. Try `memory_source_excerpt.excerpt` — if present, display with a `[from snapshot]` header.
2. Fallback to existing transcript-reading code.
3. If neither available, show `[no context available: transcript purged, no snapshot]`.

## UI (cairn/static/index.html)

Add a "Retention" nav tab. Panels:
- **Stats row**: cards for total sessions, transcripts-on-disk, coverage %, at-risk count.
- **Age histogram**: reuse existing Chart.js instance or render simple HTML bars.
- **At-risk table**: `session_id | age | size | memories | coverage | actions[snapshot, purge]`.
- **Triage tab**: sortable table per-session with value score + filters.
- **Capture window form**: date range inputs + submit, reports progress.

Follow existing dashboard patterns: 4-import Flask (memory id:3066), sortable tables via `makeSortable()` (memory id:3050).

## Tests

- `tests/test_query_cli.py` — `--context <id>` reads excerpt when transcript file removed.
- `tests/test_dashboard.py` (new if absent) — each new endpoint returns valid JSON.
- `tests/test_e2e_pipeline.py` — full cycle: create memory, snapshot, purge transcript, verify `--context` still works.

## Acceptance

- Retention tab loads with real data from your current Cairn DB.
- Snapshot a session, delete its JSONL manually, `--context` for one of its memories still works.
- `python3 -m pytest tests/` passes.

---

# §5 — Drop SCORE_W_RECENCY to 0

## Why

User-stated preference: age does not correlate with usefulness. Supersession + `-!` annotations handle genuine obsolescence. The 5% recency weight is residual bias against old-but-valid memories. Single-line change.

## Files

- `cairn/config.py:46`

## Implementation

```python
SCORE_W_RECENCY = 0.0   # Disabled — age is not a usefulness signal.
                        # Obsolescence is handled via supersession (archived_reason) and -! annotations.
```

Optional: redistribute the 0.05 to `SCORE_W_SIMILARITY` (0.50 → 0.55) or leave total < 1.0 (scoring is relative).

Decision: **leave total sub-unity** (simpler; no composite rebalance).

## Tests

- `tests/test_scoring.py` — update any assertions that depend on recency tiebreaker.
- `tests/test_retrieval_benchmark.py` — verify recall@5 does not regress.

## Acceptance

- Tests pass.
- Two memories with identical similarity but different ages rank in insertion order (or arbitrary), not by age.

---

# §6 — Systemic failure alerting

## Why

Current failure mode: embedder or DB silently becomes unavailable → enforcement gates no-op, memory capture drops writes, user unaware. System delivers no benefit and no-one knows. User explicitly wants immediate alert for systemic (persistent) failures, distinct from transient hiccups.

## Scope

- Detect: daemon unreachable for ≥ N consecutive calls, integrity_check != ok, embedding fail rate > threshold over window, hook crash rate > threshold.
- Signal: sentinel file + desktop notification + dashboard pill + Claude-visible banner.
- Auto-clear: next healthy op of the same kind removes the sentinel.

## Files

| File | Change |
|------|--------|
| `cairn/config.py` | Failure thresholds |
| `hooks/health.py` (new) | Detection + sentinel management |
| `hooks/hook_helpers.py` | Record failure counters to ephemeral DB |
| `hooks/prompt_hook.py` | Inject IMPAIRED warning when sentinel present (first-prompt only) |
| `cairn/dashboard.py` | `GET /api/health` + pill in UI |
| `cairn/static/index.html` | Red/green health pill in header |

## Config

```python
# cairn/config.py
SENTINEL_PATH = os.path.join(CAIRN_DIR, ".impaired")
DAEMON_FAIL_THRESHOLD = 5            # consecutive unreachable calls
EMBEDDING_FAIL_WINDOW = 50           # most recent N calls
EMBEDDING_FAIL_RATE_THRESHOLD = 0.5  # > 50% failures in window → systemic
HOOK_CRASH_WINDOW_MINUTES = 10
HOOK_CRASH_THRESHOLD = 3             # ≥ 3 hook crashes in 10 min → systemic
```

## Sentinel format

`.impaired` JSON:
```json
{
  "reason": "daemon_unreachable",
  "since": "2026-04-19T12:00:00Z",
  "last_error": "Connection refused",
  "count": 7
}
```

## hooks/health.py (new)

```python
"""Detect systemic Cairn failures, manage sentinel file, emit alerts."""
import json, os, subprocess, time
from hooks.hook_helpers import get_ephemeral_conn, log_warning
from cairn.config import SENTINEL_PATH, DAEMON_FAIL_THRESHOLD, ...

def record_failure(kind: str, detail: str = "") -> None:
    """Insert failure event into ephemeral metrics."""

def record_success(kind: str) -> None:
    """Insert success event; if sentinel matches kind, clear it."""

def check_systemic() -> Optional[dict]:
    """Evaluate thresholds. Return dict if systemic, None otherwise."""

def write_sentinel(info: dict) -> None:
    """Write .impaired atomically. Call notify-send (best-effort)."""

def clear_sentinel() -> None:
    """Remove .impaired if present."""

def sentinel_info() -> Optional[dict]:
    """Read .impaired, return parsed dict or None."""
```

Call `record_failure("daemon")` from `embeddings.py:_daemon_embed` on failure; `record_success("daemon")` on success. Same pattern for DB ops.

## prompt_hook.py injection

At the top of first-prompt context assembly:
```python
from hooks.health import sentinel_info
info = sentinel_info()
if info:
    impaired_notice = f"⚠️ Cairn IMPAIRED: {info['reason']} since {info['since']} — memory capture/retrieval may be degraded. See dashboard."
    # Prepend to additionalContext
```

## Desktop notify

`hooks/health.py::write_sentinel` calls:
```python
try:
    subprocess.Popen(["notify-send", "-u", "critical", "Cairn impaired", info["reason"]])
except FileNotFoundError:
    pass  # notify-send not installed — sentinel + dashboard still work
```

## Dashboard

- `GET /api/health` reads `.impaired` if present, returns `{impaired: bool, info: ...}`.
- UI: green "Cairn OK" pill in header, red "Cairn impaired: <reason>" if sentinel exists. Click → modal with detail.

## Tests

- `tests/test_health.py` (new) — simulate 5 daemon failures → sentinel written. Subsequent success → sentinel cleared. prompt_hook injects warning when sentinel present.

## Acceptance

- Stop the daemon manually (`cairn/daemon.py stop`). Run 5 sessions. `.impaired` appears. Restart daemon, run one session — `.impaired` clears.
- Dashboard pill reflects state.
- Tests pass.

---

# §7 — `-!` reason audit trail + resurface

## Why

Current behaviour (`hooks/storage.py:apply_confidence_updates`): first `-!` archives the memory with its reason into `archived_reason`. Already surfaced in retrieval as `<entry superseded="true" reason="...">`. But:
- Subsequent `-!` on an already-archived memory is lost (no second opinion recorded).
- `+` corroborations adjust confidence but the event itself is not logged — we can't trace how confidence evolved.
- User notes "articulating reasons has value in itself" — so we want the audit trail preserved even when it doesn't change retrieval behaviour.

## Scope

Add `memory_annotation_log` table capturing every `+`/`-`/`-!` with reason, session, timestamp. No retrieval change needed for `-!` (already surfaced via `archived_reason`). Optional: aggregate query shows evolution per-memory in the dashboard.

## Files

| File | Change |
|------|--------|
| `cairn/init_db.py` | Add `memory_annotation_log` table |
| `hooks/storage.py` | `apply_confidence_updates` inserts log row for every update |
| `cairn/query.py` | `--history <id>` includes annotations (already shows memory_history; extend) |
| `cairn/dashboard.py` | `/api/memory/<id>/annotations` endpoint |

## Schema

```sql
CREATE TABLE IF NOT EXISTS memory_annotation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    direction TEXT NOT NULL,    -- '+', '-', '-!'
    reason TEXT,
    session_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_ann_memory ON memory_annotation_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_ann_session ON memory_annotation_log(session_id);
```

## Storage change

In `hooks/storage.py::apply_confidence_updates`, before/after the existing branches, INSERT into `memory_annotation_log` regardless of branch. One INSERT per update. Zero change to retrieval.

## Dashboard

Memory detail view gains an "Annotation history" list:
```
2026-04-12  session abc123  +  corroborates Apr 9 observation
2026-04-18  session def456  -! reason: replaced by GCE edge approach
```

## Tests

- `tests/test_storage.py` — apply three updates, verify three rows in `memory_annotation_log`.
- Verify `-!` still archives memory and writes annotation row (unchanged behaviour + new log).

## Acceptance

- Every confidence update leaves a log row.
- Dashboard shows per-memory annotation history.
- Retrieval behaviour unchanged (no new tokens).

---

# §8 — Delete dead code

## Why

`hooks/query_expansion.py` contains `corpus_prf` (line 169) and `neighbor_blend` (line 226) strategies that are loaded but never invoked. `find_similar` uses only type-prefix fan-out (benchmarked best; memory id:3056 era). Dead code pollutes search and increases cognitive load.

## Files

- `hooks/query_expansion.py`

## Implementation

1. Remove `corpus_prf` function (and imports it alone uses).
2. Remove `neighbor_blend` function (and imports it alone uses).
3. Keep `combined_expansion` only if still called. Verify with `grep -rn "combined_expansion\|corpus_prf\|neighbor_blend" /home/james/Projects/cairn/`. Delete any that have zero callers outside the file itself.

## Tests

- `tests/test_query_expansion.py` — remove tests of deleted functions.
- `python3 -m pytest tests/` passes.

## Acceptance

- `grep -rn "corpus_prf\|neighbor_blend" /home/james/Projects/cairn/` returns zero matches.
- Tests pass.

---

# §9 — Retrieval XML chrome trim

## Why

Every retrieval injection includes:
- `<instruction>Before acting on any entry below, run: python3 .../query.py --context <id> …</instruction>` (~120 tokens) — static text, belongs in the rules file the LLM already receives.
- Redundant `query="…"` and `current_project="…"` attributes on every scope block — duplicates the root.
- Full `reliability="strong"`, verbose day counts, multiple attributes per entry.

Net overhead: ~150 tokens per retrieval × many retrievals per session × many sessions.

## Scope

- Remove `<instruction>` tag from XML. Add the equivalent sentence to `.claude/rules/memory-system.md` (source of global rules) — the LLM sees it once per session, not every retrieval.
- Drop `query=` and `current_project=` from inner `<scope>` tags; keep on root `<cairn_context>` only.
- Shorten per-entry attributes to short keys:
  - `reliability="strong"` → `r="s"` (with legend in rules file)
  - Keep `id`, `type`, `topic`, `project`, `date`, `similarity` on the root entry; move `score`, `recency_days`, `confidence` behind a compact `meta="..."` attribute or drop.

## Files

| File | Change |
|------|--------|
| `hooks/retrieval.py` | Simplify XML emission (~lines 330-380 area) |
| `.claude/rules/memory-system.md` | Add the `--context <id>` reminder once; add short-key legend |
| `README.md` / `ARCHITECTURE.md` | Update XML sample if shown |

## Implementation

1. Emit root tag with query + current_project + layer.
2. Scope tags carry only `level` + `weight`.
3. Entries emit: `<entry id="123" type="fact" topic="..." project="..." days="5" sim="0.62">content</entry>`. Short attrs: `sim` (not `similarity`), `days` (not `recency_days`), drop `score` and `reliability` in favour of explicit `sim`.
4. Archived entries keep `superseded="true" reason="..."`.

Measure before & after: instrument `hooks/hook_helpers.py` to log token estimate per injection. Target: -100 tokens median.

## Rules-file addition

Append to `.claude/rules/memory-system.md`:
```
### Retrieval XML conventions

Every `<cairn_context>` block contains one or more `<entry>` elements with:
- `id` — memory ID; use with `query.py --context <id>` to recover full context
- `sim` — semantic similarity 0.0–1.0
- `days` — age in days
- `superseded="true"` + `reason="..."` — memory is archived; use for history, not current guidance

Before acting on a high-stakes retrieved memory, run `python3 $CAIRN_HOME/cairn/query.py --context <id>` to recover the original conversation.
```

Then run `./install.sh` to propagate.

## Tests

- `tests/test_rrf_and_gotcha.py` (or nearest) — update XML assertion.
- `tests/test_retrieve_context_rrf*.py` — update format assertions.

## Acceptance

- `python3 -m pytest tests/` passes.
- Manual inspection of `hook.log` shows the new compact XML.
- Token estimate (can be crude `len(xml.split())`) drops measurably in logs.

---

# §10 — Doc sync

## Why

`memory-system.md` and `ARCHITECTURE.md` imply confidence influences retrieval ranking. Code reality: `config.py:44 SCORE_W_CONFIDENCE = 0.0` (confidence is veracity-only). Misleads the LLM. Simple doc update.

## Files

| File | Change |
|------|--------|
| `.claude/rules/memory-system.md` | Source file — update confidence section |
| `ARCHITECTURE.md` | Match |
| `README.md` | Match any summary claim |

## Changes

In the "Confidence Feedback" section of `.claude/rules/memory-system.md`, change the intro to explicitly state:
```
Confidence represents veracity — how well-corroborated a memory is across sessions.
It does NOT influence retrieval ranking. Retrieval uses similarity, keyword overlap, and scope
(see config.py SCORE_W_* weights). Confidence is recorded for audit and can gate archival when
multiple sessions mark a memory wrong.
```

Remove or correct any line that says "confidence affects ranking" / "high-confidence memories surface more often."

Run `./install.sh` to deploy the updated rules to `~/.claude/rules/memory-system.md`.

## Tests

None (doc-only).

## Acceptance

- `grep -n "confidence" .claude/rules/memory-system.md` returns only correct claims.
- After `./install.sh`, `~/.claude/rules/memory-system.md` matches source.

---

# Post-implementation

After all sections land:

1. Update `README.md` and `ARCHITECTURE.md` sections affected by §4 (retention) and §6 (impairment) — describe the new features briefly.
2. Bump version in `pyproject.toml` to `0.14.0`.
3. Verify:
   - `python3 -m pytest tests/ -v` — all pass
   - `python3 cairn/query.py --stats` — works, reads both DBs
   - `python3 cairn/dashboard.py` — loads, shows Retention tab
   - Stop and restart daemon: `.impaired` appears, then clears
   - One full Claude Code session: hooks log cleanly, no warnings
4. Tag: `git tag -a v0.14.0 -m "ephemeral DB split, retention dashboard, systemic alerting, retrieval polish"`.

## Non-goals (do NOT implement)

- Memory citation in Claude responses (user doesn't require visible benefit signal)
- Age-based pruning / lifecycle policies (user wants all data retained)
- Daemon systemd / @reboot autostart (already autostarts on-demand)
- Real-time contradiction surfacing (3:30 AM cron is acceptable)
- Confidence as a retrieval-ranking signal (deliberately disabled)

---

## Appendix A — Verification commands

```bash
cd /home/james/Projects/cairn

# Baseline
python3 -m pytest tests/ -x
python3 cairn/query.py --stats

# After §1
ls cairn/cairn.db cairn/cairn-ephemeral.db
python3 -c "import sqlite3; c=sqlite3.connect('cairn/cairn.db'); print(c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall())"
python3 -c "import sqlite3; c=sqlite3.connect('cairn/cairn-ephemeral.db'); print(c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall())"

# After §3
python3 -c "from hooks.storage import get_conn; c=get_conn(); print(c.execute('SELECT count(*) FROM memories WHERE embedding IS NULL').fetchone())"

# After §5
grep SCORE_W_RECENCY cairn/config.py

# After §6
ls -la cairn/.impaired 2>/dev/null || echo "sentinel absent (healthy)"
curl -s localhost:5000/api/health | python3 -m json.tool

# After §7
python3 -c "import sqlite3; c=sqlite3.connect('cairn/cairn.db'); print(c.execute('SELECT count(*) FROM memory_annotation_log').fetchone())"

# After §8
grep -rn "corpus_prf\|neighbor_blend" hooks/ cairn/ tests/

# After §10
grep -in "confidence.*ranking\|ranking.*confidence" .claude/rules/memory-system.md ARCHITECTURE.md README.md
```

## Appendix B — Rollback

If a change breaks production, rollback path:

- §1: Copy ephemeral tables back into main DB via one-off script; revert config + routing; delete ephemeral DB.
- §3: Drop trigger; backfill NULL embeddings one final time.
- §5: Restore `SCORE_W_RECENCY = 0.05`.
- §6: Delete `.impaired` manually; comment out sentinel injection in prompt_hook.
- §7: `DROP TABLE memory_annotation_log;` — no retrieval dependency.
- §9: Revert `hooks/retrieval.py`; re-run install.sh.

All changes are additive except §1 (requires data migration) and §9 (changes XML shape). Commit per section to keep rollback surgical.
