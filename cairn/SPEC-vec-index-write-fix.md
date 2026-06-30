# SPEC: Fix silent vec-index write failures (`no such module: vec0`)

## Problem
Memories are written with embeddings but their rows are frequently NOT inserted
into the `memories_vec` ANN index, so they become keyword/FTS-findable but
invisible to `--semantic` retrieval. A whole-DB backfill on 2026-06-28 found
**1587** such rows. This is a primary cause of false-negative push retrieval.

## Root cause (confirmed)
The vector index requires the `sqlite_vec` extension loaded on the *connection*.
- READ path loads it: `embeddings._load_vec(conn)` (embeddings.py:345) is called
  by `find_similar` (embeddings.py:973) -> reads work.
- WRITE path does NOT: `embeddings.upsert_vec_index` (embeddings.py:585) issues
  `INSERT INTO memories_vec ...` but never calls `_load_vec`. Its callers all open
  a bare `sqlite3.connect(DB_PATH)`:
    - query.py:545  add_memory
    - ingest.py:1574 insert_memories (upsert at 1677)
    - the Stop-hook memory writer (same pattern)
  => `INSERT` raises `OperationalError: no such module: vec0`.
- The error is swallowed twice: `upsert_vec_index` try/excepts to a warning log;
  `add_memory` wraps the call in `except Exception: pass`. => silent, cumulative.

## Fix

### 1. Primary — make the index write self-sufficient (single choke point)
In `embeddings.upsert_vec_index`, ensure the extension is loaded before use:

    def upsert_vec_index(conn, memory_id, embedding_blob):
        if not _load_vec(conn):          # NEW: idempotent; returns False if truly unavailable
            _log_embed(f"vec index unavailable, skipping upsert for {memory_id}", "warning")
            return False
        try:
            conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
            conn.execute("INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                         (memory_id, embedding_blob))
            return True
        except Exception as e:
            _log_embed(f"vec upsert failed {memory_id}: {type(e).__name__}: {e}", "warning")
            return False

`_load_vec` is idempotent (loading twice on one conn is harmless), so this fixes
ALL write callers without touching them. This is the minimal correct fix.

### 2. Defence in depth — one canonical connection helper
Add `embeddings.connect_db(path=DB_PATH, load_vec=True)` that opens the conn, sets
`PRAGMA busy_timeout`, and calls `_load_vec` when `load_vec`. Replace the bare
`sqlite3.connect(DB_PATH)` in: query.py add_memory/backfill_embeddings,
ingest.py insert_memories, and the Stop-hook writer. Prevents recurrence if a new
writer is added that bypasses upsert_vec_index.

### 3. Stop silent failure
- Remove the blanket `except Exception: pass` around the upsert in add_memory;
  let it use the boolean return.
- `add_memory` print: distinguish `with embedding+index` / `embedding only (INDEX
  FAILED)` so a failure is visible at the call site.

### 4. Heal existing + future gaps (CLI)
Extend backfill to cover the present-embedding-but-missing-vec case (current
`--backfill` only handles `embedding IS NULL`):
- New: `query.py --heal-vec` -> for every row WHERE embedding IS NOT NULL AND
  deleted_at IS NULL AND id NOT IN (SELECT memory_id FROM memories_vec): load_vec
  then INSERT. Report count healed + remaining. (One-shot heal already run manually;
  this makes it a supported command.)

### 5. Observability / regression guard
- `query.py --stats`: add a line `vec index: K/E embedded rows indexed (gap G)`,
  computed as embedded-and-not-deleted vs present in memories_vec. Warn if G>0
  with the `--heal-vec` hint (mirrors the existing `--backfill` warning).
- This converts a silent failure into a visible, monitored metric.

## Tests
- `test_upsert_loads_vec`: on a fresh conn from bare `sqlite3.connect`, calling
  `upsert_vec_index` succeeds (proves it self-loads).
- `test_add_memory_is_semantically_retrievable`: `add_memory(...)` then assert the
  id is in `memories_vec` AND `find_similar` returns it.
- `test_upsert_idempotent`: upsert twice -> exactly one vec row.
- `test_heal_vec`: insert a row with embedding but no vec row (simulate the bug),
  run heal, assert gap -> 0.
- `test_stats_reports_gap`: with an artificial gap, `--stats` reports G>0.

## Acceptance criteria
- After `query.py --add ...` (and Stop-hook capture, and ingest), the new id is
  present in `memories_vec` and returned by `--semantic`. No `no such module: vec0`.
- `--stats` shows `vec index gap 0` on a healthy DB; non-zero triggers a warning.
- `--heal-vec` drives the gap to 0 and is idempotent.
- Existing data healed (already done manually: 1587 -> 0; keep as the migration).

## Notes / risks
- If `sqlite_vec` is genuinely not installed in some runtime (e.g. headless cron
  without the wheel), `_load_vec` returns False and writes degrade to embedding-only
  with a WARNING (not silent) + are recoverable later via `--heal-vec`. That is the
  correct, observable degradation.
- The CUDA warnings from torch in the Bash env are unrelated (CPU embedding works);
  do not conflate them with the vec0 fault.
