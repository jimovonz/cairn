# Cairn code analysis — 2026-05-28

Multi-agent architectural review of all Cairn modules. Each finding is tagged **P0** (critical: correctness / data loss / silent failure), **P1** (material: known fragility / functional gap), or **P2** (worth fixing eventually).

Module groups reviewed:
1. Hook entry points (`hooks/prompt_hook.py`, `hooks/stop_hook.py`, `hooks/pretool_hook.py`, `hooks/posttool_hook.py`, `hooks/enforcement.py`)
2. Core memory plumbing (`hooks/storage.py`, `hooks/retrieval.py`, `hooks/parser.py`, `hooks/hook_helpers.py`)
3. Embedding + daemon (`cairn/embeddings.py`, `cairn/daemon.py`, `cairn/keywords.py`, `hooks/query_expansion.py`)
4. Calibration system (`cairn/analyser.py`, `cairn/calibration_inject.py`, `cairn/calibration_selfmod.py`, `cairn/calibration.py`, `cairn/calibration_import_claude_md.py`, backfills, `cairn/session_extract.py`)
5. Surrounding infrastructure (`cairn/init_db.py`, `cairn/query.py`, `cairn/dashboard.py`, `cairn/consolidate.py`, `cairn/ingest.py`, `cairn/graph.py`, `cairn/repo_discovery.py`, `install.sh`)

---

## Executive summary — prioritised action list

### P0 — ship soon

| # | Finding | Module | Cost to fix |
|---|---|---|---|
| 1 | **`hooks/query_expansion.py` is dead code; tests assert behaviour the production path no longer exhibits.** Production fan-out is inlined in `embeddings.find_similar:L588+` with z-score normalisation; the module's `type_prefix_fanout` uses max-sim and a different `composite_score` signature. Tests pass against an algorithm nothing calls. | Embedding | Small — delete module + rewrite the two test files OR refactor `find_similar` to call into it |
| 2 | **Topic-embedding supplement scan is unbounded and stacks with fan-out scan.** `_topic_candidates_brute` is a full-table scan on `topic_embedding` (no ANN index); `find_similar` fan-out is another unconditional full scan over all embeddings. Two O(N) passes per query on top of the vec index. ~50ms today, scales linearly. | Embedding | Medium — add a `memories_topic_vec` mirror table, or gate fan-out by row count |
| 3 | **Write-side dedup is content-only while reads are dual.** `insert_memories` calls `find_nearest` which scores `embedding` alone. A new row whose topic embedding closely matches an existing row's `topic_embedding` (but content diverges) passes the 0.95 `DEDUP_THRESHOLD` and is inserted as a duplicate — inflating the corpus along the very axis the read path now privileges. | Core plumbing | Medium — extend `find_nearest` to apply the same `max(content_sim, topic_sim)` rule |
| 4 | **`_snapshot_excerpts` in `stop_hook.py` is dead-and-broken.** Defined and references `hook_helpers.get_conn` but the module imports symbols not the alias — `NameError` on call. Never invoked. | Stop hook | Small — delete or fix import + wire in |
| 5 | **`install.sh` pipe-grep conflates real failures with no-match.** `set -eo pipefail` plus piped `pip install … \| grep` returns non-zero if grep filter matches nothing; the `\|\|` rescue then prints ERROR and exits. Real pip failures are invisible. Backfills run with `>/dev/null 2>&1 \|\| echo "skipped"` — broken v8 migration is silently swallowed. | Install | Small — replace pipes with `tee` + post-hoc inspection; surface backfill failures |

### P1 — material fragility, fix this iteration

| Finding | Module |
|---|---|
| **6.** `correction_bootstrap()` defined in `prompt_hook.py` but never invoked from `main()` — dead layer | Prompt hook |
| **7.** `.staged_context/*.txt` files never GC'd unless the same session resumes; on-disk reminder leaks forever | Prompt hook |
| **8.** Enforcement cascade order in `stop_hook.main()` has no integration test — 8 sequential gates with `sys.exit` between each, any refactor can silently reorder behaviour | Stop hook |
| **9.** Continuation cap reset only on clean exit; density-fail / validation-fail branches increment without bounding inside the branch — pathological format-fail loops can still produce N=MAX re-prompts | Stop hook |
| **10.** PreTool `_is_high_signal_edit` always returns `True` — every file modification fires a checkpoint nudge at cooldown boundaries regardless of value | PostTool |
| **11.** PreTool runs **two full-table scans per tool invocation** (corrections + everything) with no `WHERE associated_files LIKE` prefilter or index; basename-only fallback collides across projects (cross-project `utils.py` matches inject noise) | PreTool |
| **12.** Module-global embedding caches in `enforcement.py` are useless in the per-Stop-subprocess invocation model — every Stop event re-pays full embedder cost | Enforcement |
| **13.** `check_correction_triggers` loads every trigger row + blob and computes cosine in Python; no vec index, no top-K, no LIMIT — scales linearly with the correction-trigger table (the design's core feedback loop) | Enforcement |
| **14.** `hybrid_search` project pass adds IDs to `seen_ids` before global pass — borderline same-project hits just below `effective_threshold` are dropped entirely instead of falling through to global scope | Retrieval |
| **15.** `_keyword_match_search` is LIKE-on-CSV-string with full-table substring match then post-filter — O(N) Layer 2 hot path; no inverted index on the `keywords` column | Retrieval |
| **16.** Context cache `is_context_cached` re-embeds the query at full embedding latency just to test cache hit — inverts the optimisation Layer 1.5 was built to enable | Retrieval |
| **17.** `parser.parse_memory_block` silently falls from `_parse_linkdef` to `_parse_verbose` on JSON error — masks LLM format violations with no metric to record the fallback | Parser |
| **18.** `strip_seen_entries` regex assumes `id` is first `<entry>` attribute — any future formatter change zeros out the dedup gate without firing `strip_seen_leaked` (it would record 0 strips, not catch the failure) | Hook helpers |
| **19.** Daemon recv loop assumes client `shutdown(WR)` — third-party / async callers will hang; no length-prefix or newline framing | Daemon |
| **20.** Daemon `similarity` action opens a new sqlite connection per request with no `PRAGMA busy_timeout` / WAL — can collide with writers under hook fan-in | Daemon |
| **21.** Daemon auto-start race: `flock` lock-loser path does `sleep(3)` then proceeds — if model load exceeds 3s the loser falls through to per-call ST inference | Embedding |
| **22.** `_topic_candidates_brute` silently drops rows on `from_blob` exceptions (bare `except: continue`); no metric, no log — corrupted blobs would mask retrieval misses | Embedding |
| **23.** `init_db.py` FTS rebuild path: DROP TABLE + CREATE + rebuild not wrapped in a transaction — crash mid-migration leaves search-blind window until next `init()` | Init DB |
| **24.** Dashboard has no auth; anyone on localhost (or misconfigured bind) reads the entire memory store. `get_conn()` opens durable DB per request (no pooling, no `mode=ro`) — dashboard contends with hooks on WAL writer lock during heavy sessions | Dashboard |
| **25.** `ingest.py` has no cost cap, no Phase-2 skip env knob, single 120s `claude -p` timeout with **no retry** — transient failure = whole run lost. JSON `strict=False` fix is good but other failure modes (truncated array on output cap, trailing prose, partial markdown) remain | Ingest |
| **26.** `consolidate.py` sets `PRAGMA busy_timeout=5000` but no retry on `SQLITE_BUSY` further down; long writers from the embedding daemon can knock the daily cron sideways | Consolidate |
| **27.** `repo_discovery._has_ingest_record` uses unindexed `source_ref LIKE '%"<path>"%'` — substring scan on every session start; scales with `memories` table size | Repo discovery |
| **28.** Concurrent-cron race in `analyser._get/set_analyser_state`: non-transactional RMW across separate connections; two crons can both see "no prior state" and double-bill the same session. No advisory lock equivalent to `.drain.lock` | Calibration analyser |
| **29.** `_load_prior_rows_for_session` query is `created_at >= first_analysed_at` with no project/session filter — returns rows from unrelated sessions, telling the LLM "don't re-emit" against unrelated content | Calibration analyser |
| **30.** `calibration_inject.inject_for_prompt` swallows all exceptions silently with no metric — a regression in embeddings or schema degrades silently to 0 deliveries (the exact symptom that motivated today's per-qf fix) | Calibration inject |
| **31.** CLI `calibration add` and `calibration_import_claude_md` may not write `calibration_qf_embeddings` sidecar entries — if so, rows added via these paths are delivery-starved on the new per-qf retrieval | Calibration CLI |
| **32.** `extract_associated_files` issues per-entry SELECT+UPDATE with `ORDER BY updated_at DESC LIMIT 1` — racy when two entries share `(type, topic)` in the same response; should use captured `last_insert_rowid()` | Storage |
| **33.** `inline_backfill` doesn't backfill `topic_embedding` — leaves rows half-dual; only the out-of-band `memory_topic_embedding_backfill.py` script converges them | Storage |

### P2 — worth fixing eventually

| Finding | Module |
|---|---|
| **34.** `_is_empty_memory` is substring-only; doesn't reject hedge phrasings the calibration analyser already forbids ("may or may not", "possibly") | Storage |
| **35.** Six staged-reminder branches in `prompt_hook.main()` are structurally identical — refactor to table-driven loop | Prompt hook |
| **36.** PostTool `ERROR_PATTERNS` substring-matches success messages containing "error" (e.g. `0 errors found`) — false positives | PostTool |
| **37.** PostTool `exit_code` falsy check treats exit 0 and missing-metadata identically | PostTool |
| **38.** `keywords.py` `_naive_fallback` stopword list duplicates the one in `embeddings.extract_query_terms`; can drift | Keywords |
| **39.** Daemon has no max-threads limit — burst (e.g. backfill) could OOM on small VMs | Daemon |
| **40.** `init_db.py` v1–v8 sentinels are advisory; code does not gate migrations on them. Acceptable today but as migrations grow expensive (e.g. v8 backfill) gating becomes necessary | Init DB |
| **41.** `init_db.py` swallows `sqlite_vec` import/load errors silently — install can fall back to brute-force similarity without warning | Init DB |
| **42.** `query.py` is 1499 LOC, no module split — accretion target | Query CLI |
| **43.** `graph.py` is silent if `graph.db` missing — no hint to run `code-review-graph build` | Graph CLI |
| **44.** `repo_discovery` price string `"$0.10-0.50"` is hard-coded, not measured | Repo discovery |
| **45.** `_distinct_sessions_for_row` (selfmod) counts only `outcome='followed'` sessions; deserves a comment + boundary test | Selfmod |
| **46.** `decay_unused` only on `delivered_count = 0`; spec says "rows older than half-life with low usage" — single-delivery cold rows escape | Selfmod |
| **47.** Tier 2 surfaces never re-surface after `resolution='dismissed'` even if row stats change | Selfmod |
| **48.** Hook helpers `load_injected_ids` / `save_injected_ids` round-trip JSON on every call; unbounded list growth, O(N) per injection | Hook helpers |
| **49.** No `dual_embedding_topic_win` metric in `_brute_force_candidates` when `topic_sim > sim` — the architectural v8 bet has no feedback signal | Embedding |
| **50.** Continuation counter in `enforcement` does 3 SQLite open/commit/close cycles per increment; should be a single transaction | Enforcement |

---

## Cross-cutting findings

### Subprocess invocation model + module-level caches don't compose
Every Stop / UserPromptSubmit / PreTool / PostTool hook is invoked as a fresh Python subprocess. Module-global lazy caches in `enforcement.py`, `prompt_hook.py`, and elsewhere are recomputed on every fire, defeating their purpose. Structural fix is either a persistent helper process or an on-disk cache (mmap'd numpy file for embedding refs, sqlite kv for detector triggers). The embedding daemon already follows this pattern — extend to the rest.

### Ad-hoc staged-reminder file IPC
Six different `.staged_context/<session>_*.txt` file types are read+delete with bespoke try/except blocks across two modules (`prompt_hook.py` reads, `stop_hook.py` and `enforcement.py` write). No GC for orphaned files when a session never resumes. Should be a single ephemeral-DB table (`staged_reminders(session_id, kind, payload, created_at)`) or a single dispatched-reminders abstraction with a retention sweep.

### Schema v8 / dual-embedding ROI unmeasured
The architectural decision (separate topic embedding, max-cos retrieval) was driven by an audit showing +189% candidate lift. But the running system has no metric distinguishing "topic embedding fired and won" from "content embedding fired and won". Without it the ongoing ROI of v8 is invisible — and the next architectural decision will be similarly blind. Add `dual_embedding_topic_win` and `dual_embedding_content_win` counters in `_brute_force_candidates` and the supplement merge.

### Write-side / read-side asymmetry
This recurs across the codebase:
- Memory dedup writes content-only, reads dual (finding #3)
- Calibration CLI/import may write rows without qf-sidecar embeddings, but reads via per-qf max-cos (finding #31)
- `inline_backfill` fills `embedding` but not `topic_embedding` (finding #33)

Pattern is the same each time: a read-side optimization landed without auditing every write path. A startup invariant check could catch all three: `SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL AND topic_embedding IS NULL` should trend to zero; same shape for calibration_qf_embeddings.

### 13 deleted tautological tests (memory 394) not yet replaced
Concepts that were tested (saturating confidence, gate boundaries, diversity filtering, pre-filter) are worth validating — but through real code paths against controlled embedder fixtures. The "Tier 2" pattern documented in memory 1786 (real pipeline logic with fake embeddings, ~5 line mock, uses existing `fresh_db()` pattern) is the right shape. Currently a coverage hole.

### Inconsistent project scoping in cross-cutting paths
- `pretool_hook` basename-only fallback matches across projects (finding #11)
- `_load_prior_rows_for_session` ignores project (finding #29)
- `repo_discovery._has_ingest_record` doesn't tag matches by project

Cairn is fundamentally project-scoped (per CLAUDE.md), and most leaks of "wrong project content" trace back to one of these paths.

---

## Per-module detail

### Hook entry points

**`prompt_hook.py`** — Layered UserPromptSubmit injection. 200-line linear pipeline with 6 near-identical staged-file reader blocks; `correction_bootstrap` defined but unused. P0/P1 findings #6, #7. Cross-cutting: cold-start gap (no default `context: insufficient` seed) is still open per gotcha 60.

**`stop_hook.py`** — 660 lines of sequential gates with `sys.exit(0)` between each. Order matters but no integration test pins it (finding #8). `_snapshot_excerpts` dead-and-broken (finding #4). Async/sync write decision is `PYTEST_CURRENT_TEST`-gated — brittle if subprocess tests lose the env var.

**`pretool_hook.py`** — File-context injection. Two full table scans per tool call with no indexing or project scoping (finding #11). Corrections path has no similarity/recency ranking — DB-scan order wins.

**`posttool_hook.py`** — Mid-response nudges. `_is_high_signal_edit` always `True` (finding #10). No total-nudges-per-session cap; cooldown only spaces them out.

**`enforcement.py`** — Detector library. Module-global embedding caches useless in subprocess model (finding #12). `check_correction_triggers` is O(N) linear cosine (finding #13).

### Core memory plumbing

**`hooks/storage.py`** — Insert/update with dedup + contradiction detection. Dedup is content-only (finding #3). Race in `extract_associated_files` (finding #32). `inline_backfill` doesn't touch `topic_embedding` (finding #33).

**`hooks/retrieval.py`** — Layer 1/1.5/2 + thin-detection. Project/global scope leak on borderline matches (finding #14). Layer 2 keyword path O(N) LIKE (finding #15). Cache hit-test re-embeds the query (finding #16). SQL exclude_ids gate is correctly threaded.

**`hooks/parser.py`** — Memory-block parsing. Silent format fallback on link-def JSON error (finding #17).

**`hooks/hook_helpers.py`** — Shared utilities. Regex-fragile `strip_seen_entries` (finding #18). `_metric_buffer` is module-global, not session-scoped.

### Embedding + daemon

**`cairn/embeddings.py`** — The retrieval hot path. Topic supplement scan is unbounded (finding #2). Fan-out path bypasses sqlite-vec at scale (finding #2 same root). `_topic_candidates_brute` swallows blob-decode errors (finding #22). Dual-embedding ROI unmeasured (finding #49).

**`cairn/daemon.py`** — Resident MiniLM + cross-encoder + NLI server. Recv assumes client `shutdown(WR)` (finding #19). Per-request sqlite connection in `similarity` action (finding #20). No max-threads back-pressure (finding #39).

**`cairn/keywords.py`** — YAKE phrase extraction. Stopword duplication with `embeddings.extract_query_terms` (finding #38). Returns `set` so caller ordering is lost.

**`hooks/query_expansion.py`** — Dead in production (finding #1). Algorithm and `composite_score` signature both drifted from the inlined `find_similar` version.

### Calibration system

**`cairn/analyser.py`** — Cron-driven LLM pass. Concurrent-cron race (finding #28). `_load_prior_rows_for_session` cross-session leak (finding #29). Envelope check post-truncation (P2).

**`cairn/calibration_inject.py`** — Per-qf max-cos retrieval. Silent exception path (finding #30). N×M qf-decode in the hot path (P1, today's empirical 31 deliveries validates the design but the cost path will dominate at 10× row growth).

**`cairn/calibration_selfmod.py`** — Tier 1 + Tier 2. `decay_unused` excludes single-delivery cold rows (finding #46). Tier 2 dismiss is permanent (finding #47).

**`cairn/calibration.py` / `calibration_import_claude_md.py`** — CLI / CLAUDE.md scanner. Possible qf-sidecar parity gap (finding #31) — needs verification.

**`session_extract.py`** — Stable, well-tested. No findings.

**Backfills (`calibration_qf_backfill.py`, `memory_topic_embedding_backfill.py`)** — Idempotent, local embedder, no LLM cost. Should be invariant-checked at startup (cross-cutting #4) so the question "are all rows backfilled?" doesn't depend on remembering to run the scripts.

### Surrounding infrastructure

**`cairn/init_db.py`** — Schema v1→v8. FTS rebuild non-atomic (finding #23). `sqlite_vec` failure silently swallowed (finding #41).

**`cairn/query.py`** — CLI surface. 1499 LOC, accreting (finding #42). `--semantic` daemon-down fallback is silent.

**`cairn/dashboard.py`** — Web UI. No auth + per-request conn open (finding #24). All advertised panels appear to be wired.

**`cairn/consolidate.py`** — Daily cron. No `SQLITE_BUSY` retry (finding #26). O(n²) per-pair commits in the cluster loop.

**`cairn/ingest.py`** — Repo ingestion. No cost cap, no retry (finding #25). JSON `strict=False` fix landed today; other failure modes remain.

**`cairn/graph.py`** — `cairn-graph` CLI. Silent when graph.db missing (finding #43).

**`cairn/repo_discovery.py`** — Two-tier session-start. Unindexed LIKE on `source_ref` (finding #27). Hard-coded price string (finding #44).

**`install.sh`** — Installer. Pipe-grep failure conflation (finding #5). Backfills suppress real errors. No `--dry-run`.

---

## Suggested order for addressing P0s

1. **#4** — Delete `_snapshot_excerpts` or fix the import (~5 minutes, removes a NameError landmine)
2. **#1** — Decide on `hooks/query_expansion.py`: delete + rewrite tests OR refactor `find_similar` to call into it. Currently the test suite is asserting an algorithm that doesn't run. Choose one and align (~1 hour)
3. **#5** — Fix `install.sh` pipe-grep + surface backfill errors. Real install failures are currently invisible (~30 minutes)
4. **#3** — Extend `find_nearest` write-side dedup to apply `max(content_sim, topic_sim)`. This is the single most impactful invariant-restoring change (~1 hour + tests)
5. **#2** — Add `memories_topic_vec` mirror table OR gate fan-out + topic supplement by row count threshold. Buys headroom before the O(N) scans become a real problem (~2-3 hours including testing)

After these five, the P1 list is the natural backlog. P2s are best handled opportunistically when touching the surrounding code.

---

## Note on review confidence

- The five subagent reviews each examined their assigned modules with code reads + `cairn-graph` callers/tests queries.
- Findings were cross-checked against today's session work (per-qf sidecar, dual embedding, SQL dedup gate, repo discovery) to avoid flagging shipped fixes as outstanding.
- Some findings reference behaviour I have not personally executed (e.g. "fan-out scan is the dominant cost at scale") — these are architectural claims based on the code shape, not measured. Where claims are measured (e.g. today's 31 calibration deliveries, audit's +189% candidate lift), the basis is stated explicitly.
- The cross-cutting findings are mine, drawn from patterns across the per-module reports.
