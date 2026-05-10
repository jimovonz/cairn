# Cairn Multi-Node Sync — Design

Status: design v1, branch `feature/multi-node-sync`.

## Problem

Today every cairn install is a single-node SQLite database. We want a 20-engineer team to share memories across machines while:

1. **Working offline.** A laptop on a train must keep reading and writing locally; sync resumes when network returns.
2. **Surviving any peer being down.** No central authoritative server.
3. **Preserving authorship.** Who wrote it, who corroborated it, who contradicted it — all must travel with the row.
4. **Allowing private scratchpads.** Some memories are personal; some are team-shared.
5. **Avoiding embedding/FTS sync cost.** Derived data is regenerated locally on receipt.

## Non-goals

- Strong consistency. We accept eventual convergence; conflict resolution is deterministic but not "linearizable."
- Real-time push. Pull-based, polled. A sync interval of 30–300s is acceptable.
- Hiding nodes from each other. Peers are explicitly listed in `~/.cairn/peers.json`; no autodiscovery in v1.

## Identity model

Three IDs, each with one job.

| Column | Scope | Semantics | Mutability |
|---|---|---|---|
| `memories.id` (existing INTEGER PK) | local only | Auto-increment row counter. **Never synced.** Stays as the FK target for local-only tables (`memory_history`, `memory_annotation_log`, `correction_triggers`, `memory_source_excerpt`, `memories_vec`). | local |
| `memories.origin_id` (existing TEXT) | global | The row's cross-node identity. UUIDv4 generated at first insert; never changes. **This is the sync key.** Already populated for all existing rows by `init_db.init()` backfill. | immutable |
| `memories.created_by_node` (NEW TEXT) | global | The node UUID that first inserted this row. Provenance for "who wrote this." | immutable |
| `memories.created_by_user` (existing `user_id`) | global | The user identifier (`$USER@$hostname` or LDAP handle) at the creating node. | immutable |
| `memories.updated_by_node` (NEW TEXT) | global | Last node to mutate the row. Used for LWW conflict tiebreak. | mutable |
| `memories.updated_by_user` (existing `updated_by` repurposed) | global | Last user to mutate. | mutable |
| `memories.lamport` (NEW INTEGER) | global | Per-node Lamport clock value at last mutation. Used as primary LWW ordering key. Beats `updated_at` because wall clocks lie. | mutable |

The local `id` and the global `origin_id` are decoupled. When node A's row crosses to node B, B inserts it with a fresh local `id` but preserves `origin_id`. Node B's local FK tables (`memory_history` etc.) reference B's local `id`. History rows themselves are synced separately (see below) and re-anchored to the local `id` via `origin_id` lookup.

### Node ID

`~/.cairn/node_id` — a single UUIDv4 written on first run. Survives reinstall. **Not** derived from hostname/MAC (which leak info and aren't stable). If the file is deleted, a new identity is generated and the node looks like a new peer to everyone else — that's a feature, not a bug, because the previous node's identity may still be active elsewhere.

### User ID

`~/.cairn/user_id` overrides; otherwise `f"{os.getenv('USER')}@{socket.gethostname()}"`. For team rollout this becomes an LDAP/email handle.

## What syncs, what doesn't

| Table / column | Synced? | Notes |
|---|---|---|
| `memories` (canonical fields) | YES | `type, topic, content, confidence, archived_reason, deleted_at, keywords, project, depth, associated_files, origin_id, created_by_*, updated_by_*, lamport, visibility, embedding_model_version` |
| `memories.embedding` | NO | Regenerated locally on receipt. Avoids ~1.5KB/row + model-version coupling. |
| `memories.id, created_at, updated_at` | NO | Local counters and wall-clock timestamps. `lamport` is the cross-node ordering key. |
| `memories_fts`, `memories_vec` | NO | Derived; rebuilt by triggers / regen. |
| `memory_history` | YES | Re-anchored to local `id` via `origin_id` lookup on receipt. |
| `memory_annotation_log` | YES | Confidence votes must accumulate across the team. |
| `correction_triggers` | YES | Re-anchored via `origin_id`. |
| `pair_assessments` | YES | LLM judgments are expensive — share to dedupe work. Re-anchored via origin_id on both sides. |
| `ingested_repos` | YES | Per memory 1378. |
| `sessions, hook_state, metrics, ingestion_cache` | NO | Local operational state. |
| `memory_source_excerpt` | OPTIONAL (default NO) | Transcripts may be private. Per-team opt-in via `peers.json`. |
| Visibility = `private` | NEVER | Filtered out at extraction time. |

## Conflict resolution

Per-column rules (deterministic, hash-able, no human in the loop):

- **`content`, `topic`, `type`, `keywords`, `project`, `depth`, `archived_reason`** — Last-Writer-Wins ordered by `(lamport DESC, updated_by_node ASC)`. Lamport beats wall clocks; node ID lex-ordering is the deterministic tiebreaker for simultaneous edits.
- **`confidence`** — Counter, not LWW. The column becomes a **derived view** computed from `confidence_log` entries (see below). Two nodes voting `+` independently both stick; `-!` annotations both stick.
- **`deleted_at`** — Tombstone. Once non-null on any node, it becomes non-null on all (earliest wins). Hard-delete is local-only and never propagates.
- **`memory_history`** — Pure append. On sync, union by `(origin_id, lamport, changed_by_node)`.
- **`memory_annotation_log`** — Pure append. Union by row UUID (`annotation_uuid`, NEW).

### Confidence as a counter

The current `confidence` column is mutated in-place by `apply_confidence_updates()`. This breaks under multi-node — two nodes both seeing a `+` would either double-boost (if naive sum) or one would clobber the other (if LWW).

Replace with:

```
CREATE TABLE confidence_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,   -- local
    log_uuid      TEXT NOT NULL UNIQUE,                -- global
    memory_origin TEXT NOT NULL,                       -- FK to memories.origin_id
    direction     TEXT NOT NULL,                       -- '+', '-', '-!'
    reason        TEXT,
    node_id       TEXT NOT NULL,
    user_id       TEXT,
    session_id    TEXT,
    lamport       INTEGER NOT NULL,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_conf_log_memory ON confidence_log(memory_origin);
```

`memories.confidence` is **recomputed deterministically** from `confidence_log` entries for that memory: starting from `CONFIDENCE_DEFAULT (0.7)`, apply `+` boosts in `(lamport, log_uuid)` order using the existing saturating formula. Recompute is cheap (mean ~3 votes per memory, max ~50) and runs on insert + on changeset apply.

This makes confidence convergent: any two nodes with the same `confidence_log` set produce the same `confidence` value.

`memory_annotation_log` and `confidence_log` are nearly the same data — `confidence_log` is the synced canonical source; `memory_annotation_log` becomes derived/local-only and may eventually be dropped. For v1 we maintain both during transition.

## Lamport clocks

Every mutating operation:

```python
node_lamport = max(node_lamport, peer_lamport_seen) + 1
row.lamport = node_lamport
```

The per-node clock lives in `node_state.lamport` (a single-row table). When applying a peer changeset, we bump the local clock past the highest received lamport, so subsequent local edits are causally after observed peer edits.

## Sync mechanism

**Hand-rolled, pull-based, peer-to-peer.** No central authority.

### Why hand-rolled (rejecting cr-sqlite)

- The `memories` table already has `origin_id, user_id, updated_by, team_id, source_ref, deleted_at, synced_at` columns and a `schema_version` table — the previous design intended hand-rolled sync.
- cr-sqlite would give per-column LWW free, but: (a) it's a loadable extension every node must deploy, (b) embedding regeneration still needs an out-of-band trigger, (c) confidence-as-counter doesn't fit the LWW model and would need bypass machinery anyway.
- A JSON-over-HTTPS payload is auditable, easy to authenticate, and easy to diff in tests.

### Wire format

Pull request (B asks A "what's new since I last asked?"):

```http
POST /sync HTTP/1.1
Authorization: Bearer <peer-token>
X-Cairn-Node: <B's node_id>
X-Cairn-Schema-Version: 4
Content-Type: application/json

{
  "since_lamport_by_node": {
    "<node-A-uuid>": 18432,
    "<node-C-uuid>": 9921
  },
  "max_rows": 1000,
  "include_excerpts": false
}
```

Response:

```json
{
  "schema_version": 4,
  "node_id": "<A's node_id>",
  "lamport_now": 18450,
  "memories":            [ { "origin_id": ..., "lamport": ..., ... }, ... ],
  "memory_history":      [ { "history_uuid": ..., "memory_origin": ..., ... }, ... ],
  "confidence_log":      [ { "log_uuid": ..., "memory_origin": ..., ... }, ... ],
  "correction_triggers": [ ... ],
  "pair_assessments":    [ ... ],
  "ingested_repos":      [ ... ],
  "tombstones":          [ { "origin_id": ..., "deleted_at": ..., "by_node": ... }, ... ]
}
```

The requesting node:
1. Verifies `schema_version` matches (else aborts with clear error).
2. Applies rows in dependency order: `memories` first, then `*_log` and `*_history` (which FK by `origin_id`).
3. For each `memories` row: lookup by `origin_id`. If absent, INSERT with fresh local `id`. If present, per-column LWW merge.
4. After insert, regenerate embedding locally and upsert to `memories_vec`. FTS triggers auto-fire.
5. Recompute `confidence` from the union of local + received `confidence_log` entries.
6. Update `sync_state` vector clock.

### `sync_state` and `sync_peers`

```sql
CREATE TABLE sync_peers (
    peer_node_id    TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    bearer_token    TEXT NOT NULL,
    label           TEXT,
    schema_version  INTEGER,
    include_excerpts INTEGER DEFAULT 0,
    last_attempted_at TEXT,
    last_succeeded_at TEXT,
    last_error      TEXT
);

CREATE TABLE sync_state (
    peer_node_id    TEXT NOT NULL,
    source_node_id  TEXT NOT NULL,           -- a row originated by this node
    last_lamport    INTEGER NOT NULL,        -- highest lamport seen for that source via this peer
    PRIMARY KEY (peer_node_id, source_node_id)
);

CREATE TABLE node_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
-- key='node_id', key='user_id', key='lamport', key='embedding_model_version'
```

The vector-clock-per-peer (`sync_state`) is what makes pull-based gossip converge in O(N) bandwidth instead of O(N²) — when B asks A, B says "give me everything past the lamports I've already seen, regardless of which node originated them," and A filters accordingly.

### Schema version handshake

Every request and response carries `X-Cairn-Schema-Version`. Mismatch → 409 with the body `{"error": "schema_version_mismatch", "server": 4, "client": 3}`. The client logs and skips that peer until upgraded. No partial sync across schema versions.

## Embedding model versioning

`embedding_model_version TEXT` (new column on `memories`, also stored in `node_state`). Format: `<model-name>@<sha256-prefix>` e.g. `sentence-transformers/all-MiniLM-L6-v2@dca72b`. On receipt:

- If incoming row's model matches local: copy embedding (future optimization; v1 always regenerates).
- If different: regenerate locally with the local model. The local copy carries the local `embedding_model_version`; the canonical row carries the *creator's* model version unchanged.

The vector index (`memories_vec`) is always populated from local embeddings.

## Visibility

`visibility TEXT NOT NULL DEFAULT 'team'` ∈ `{private, team, public}`.

- **private** — never enters any sync payload. Personal scratchpad.
- **team** — synced to listed peers (default).
- **public** — flagged for hypothetical future cross-team sharing. Treated as `team` in v1.

Visibility is mutable; flipping `team → private` does NOT retract previously-synced copies (impossible). The UI must warn.

## Migration plan (v3 → v4)

`init_db.init()` already does ALTER TABLE-then-pass migrations. Add v4:

```python
# Schema v4 — multi-node sync
for col, coltype in [
    ("created_by_node", "TEXT"),
    ("updated_by_node", "TEXT"),
    ("lamport", "INTEGER DEFAULT 0"),
    ("visibility", "TEXT DEFAULT 'team'"),
    ("embedding_model_version", "TEXT"),
]:
    try: conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {coltype}")
    except sqlite3.OperationalError: pass

# new tables: confidence_log, sync_peers, sync_state, node_state
# backfill: created_by_node = local node_id, lamport = id (preserves order),
#           visibility = 'team', embedding_model_version = current model id
```

Backfill rationale:
- `created_by_node = <local node_id>` — every existing memory was authored on this machine.
- `lamport = id` — preserves total order; no two existing rows clash.
- `visibility = 'team'` — opt-out, not opt-in; the user can re-mark privately afterwards.
- `embedding_model_version` — read from the live embedder.

Migration is **idempotent** — re-running `init()` is a no-op after v4.

## Rollout

1. **Solo (current):** v4 schema lands; node_id generated; nothing else changes for single-node users. No regressions, no opt-in needed.
2. **Two-node test bed:** host + lubuntu-browse VM. Both run v4. Pull-only sync VM ← host. Validates row replay, embedding regen, FTS rebuild, no duplicate-by-content.
3. **Bidirectional + offline conflict:** both sides edit same memory's `confidence_log` while VM offline; sync; verify deterministic merge.
4. **Three-node simulation:** third local cairn at `$CAIRN_HOME=/tmp/cairn-node3` to validate gossip convergence.
5. **LAN team rollout:** rendezvous node (any one machine) reduces N² gossip; everyone syncs against it on a 5-min cron, plus pairwise as fallback.

## Open questions deferred to post-v1

- Auth strong enough for prod: bearer tokens are fine on a trusted LAN; need OIDC / mTLS for internet-facing.
- Per-author confidence weighting (memory 581) — defer until we have multi-node data showing it matters.
- Encryption at rest of the synced payload — the sync transport is HTTPS, but local DBs are plaintext.
- Selective sync ("don't pull memories from project X") — possible via filter in the request body, deferred.
