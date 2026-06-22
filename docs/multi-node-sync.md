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

---

# v2 — Discovery, public-key pairing, and dashboard authorization

Status: design v2. Supersedes two v1 decisions: the "no autodiscovery" non-goal,
and bearer-token auth. The v1 sync *core* (changeset extract/apply, Lamport
clock, `sync_state` vector clock, LWW, visibility, embedding regen) is unchanged —
v2 only replaces **how peers find each other** and **how trust is established**.

## Motivation

v1 required hand-editing a peer registry and sharing a symmetric bearer token
out-of-band per peer. That doesn't scale to ad-hoc sharing and a leaked token is
a silent compromise. v2 makes any cairn node able to **advertise itself on the
LAN**, lets another node **request to sync**, and requires the host to
**explicitly authorize each interconnection from the dashboard** — the same trust
model as Syncthing (device = keypair, local discovery, approve-this-device,
then mutual-auth transport). There is still **no central server**: pairing is
peer-to-peer and every node remains a complete offline-first local DB.

## Identity becomes a keypair

Each node generates an **Ed25519 keypair** on first run, stored alongside the
existing identity files:

- `~/.cairn/node_key`      — private key, `chmod 600`, never leaves the machine.
- `~/.cairn/node_key.pub`  — public key.
- `~/.cairn/node_id`       — **derived** as `base32(sha256(pubkey))[:52]` (a
  Syncthing-style device fingerprint) instead of a bare UUIDv4. Stable, globally
  unique, and *self-certifying*: the ID is a commitment to the public key, so a
  peer can't impersonate a node_id without its private key.

`user_id` is unchanged (`$USER@host` or override) — it is a human label, not an
auth principal. The **public key is the auth principal**; `created_by_node` (the
device fingerprint) becomes cryptographically attributable rather than
self-declared.

## Discovery (advertising)

A node opts into discovery (`CAIRN_SYNC_ADVERTISE=1`). When on, it broadcasts a
small beacon on the local segment — **mDNS/DNS-SD** service type
`_cairn-sync._tcp` (fall back to UDP broadcast on `47391` where mDNS is blocked):

```
{ "node_id": "<device-fingerprint>", "user_id": "alice@thinkpad",
  "url": "https://192.168.1.42:8787", "schema_version": 10, "proto": 2 }
```

Discovery only makes a node *visible and dialable*; it grants **no access**.
Beacons are unauthenticated by design (they carry only public info). A node that
doesn't advertise can still pair by dialing a known URL — discovery is
convenience, not a security boundary.

## Pairing handshake + dashboard authorization

Trust is established once per peer pair, interactively:

1. **Request.** Node B (the requester) dials A's `/pair` endpoint with a signed
   request: `{node_id_B, pubkey_B, user_id_B, url_B, nonce, ts}` signed by B's
   private key. A verifies the signature proves possession of `pubkey_B` and that
   `node_id_B == fingerprint(pubkey_B)` (self-certifying check).
2. **Queue.** A stores it in `pairing_requests` with `status='pending'`. No sync
   access is granted yet. A returns `202 Accepted`.
3. **Authorize.** A's **dashboard** shows a pending-pairing queue with B's
   `user_id`, fingerprint (displayed for out-of-band verification, exactly like
   reading a Syncthing device ID aloud), and source IP. The host clicks
   **Approve** or **Deny**.
4. **Pin.** On approve, B's pubkey is pinned into `sync_peers` (`peer_public_key`,
   `status='approved'`). From then on, all `/sync` requests from B are
   authenticated by signature against the pinned key.

Bidirectional sync requires **mutual approval** — A approving B lets B pull from
A; for A to pull from B, B must approve A symmetrically. The dashboard surfaces
both directions.

Trust is **TOFU** (trust-on-first-approve): the fingerprint shown at approval
time is the commitment. To defend against a discovery-time MITM, the host can
verify the displayed fingerprint against B over a side channel before approving —
optional on a trusted LAN, recommended otherwise.

## Auth on the sync path (replaces bearer token)

`sync_peers.bearer_token` is **removed**; `peer_public_key` replaces it. Each
`/sync` request carries:

```
X-Cairn-Node: <node_id>
X-Cairn-Timestamp: <unix>
X-Cairn-Nonce: <random>
X-Cairn-Signature: ed25519( method | path | sha256(body) | ts | nonce )
```

The server (`cairn/sync/server.py::_authorized`) looks up `peer_public_key` by
`X-Cairn-Node`, verifies the signature, rejects if `status != 'approved'`,
rejects stale timestamps (±300s window) and replayed nonces (short-lived seen-set).
This is transport-agnostic: it works over plain HTTP, and composes with TLS
(self-signed certs pinned at pairing time) for confidentiality without a CA.

## Schema delta (v10)

```sql
-- sync_peers: drop bearer_token, add the pinned key + approval state
ALTER TABLE sync_peers ADD COLUMN peer_public_key TEXT;     -- pinned at approval
ALTER TABLE sync_peers ADD COLUMN status TEXT DEFAULT 'approved'; -- approved|revoked
ALTER TABLE sync_peers ADD COLUMN approved_at TEXT;
-- (bearer_token retained as nullable for one migration cycle, then dropped)

CREATE TABLE pairing_requests (
    id              INTEGER PRIMARY KEY,
    peer_node_id    TEXT NOT NULL,          -- = fingerprint(peer_public_key)
    peer_public_key TEXT NOT NULL,
    user_id         TEXT,
    url             TEXT,
    source_ip       TEXT,
    direction       TEXT DEFAULT 'inbound', -- inbound (they asked) | outbound (we asked)
    status          TEXT DEFAULT 'pending', -- pending|approved|denied
    requested_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    decided_at      TEXT
);
```

`node_state` gains nothing new — the keypair lives in `~/.cairn/` so it survives
DB rebuild (same rationale as `node_id` in v1). Bump `SCHEMA_VERSION` to 10 in
`cairn/sync/__init__.py`; the v1 handshake (`X-Cairn-Schema-Version`, 409 on
mismatch) already gates cross-version sync, so v1 and v2 nodes simply refuse each
other until upgraded — acceptable, no production peers exist.

## Dashboard endpoints

- `GET  /api/sync/pairing-requests`            — pending queue (+ recent decided).
- `POST /api/sync/pairing-requests/<id>/approve` — pin pubkey → `sync_peers`.
- `POST /api/sync/pairing-requests/<id>/deny`    — mark denied; no access granted.
- `GET  /api/sync/peers`                        — approved peers, last sync, errors.
- `POST /api/sync/peers/<node_id>/revoke`        — set `status='revoked'`.

## Trust revocation

Revoking sets `sync_peers.status='revoked'` — future `/sync` requests from that
key are rejected. Memories already pulled are **retained with attribution**
(deleting them would lose corroboration history and is unenforceable anyway, like
the `team→private` case). A revoked peer must be re-approved (fresh pairing) to
resume.

## Threat model (v2, trusted-LAN scope)

- **Impersonation** — prevented: `node_id` is a commitment to the pubkey;
  signatures prove key possession.
- **Unauthorized sync** — prevented: no pinned, approved key ⇒ 401.
- **Discovery MITM** — mitigated by out-of-band fingerprint verification at
  approval (TOFU); residual risk acceptable on a trusted LAN.
- **Token leakage** — eliminated: there is no shared secret.
- **Replay** — mitigated: timestamp window + nonce seen-set.
- **At-rest confidentiality** — unchanged from v1: local DBs are plaintext
  (still an open question); the keypair only protects the wire + identity.

## Open questions (v2)

- Outbound auto-pairing UX: when *we* discover a peer, do we one-click request, or
  require typing/scanning their fingerprint first?
- mDNS reliability across managed switches / VLANs — may need the UDP-broadcast
  fallback or a manual URL path as the dependable default (mirrors the v1
  graph-watch-daemon lesson: cron/manual is the dependable backbone, real-time is
  the optimization).
- Key rotation: rotating a node's keypair changes its `node_id`/fingerprint, so it
  reads as a new peer and must be re-approved. Acceptable, but document it.

---

# v2.1 — as implemented

The v2 design above is implemented with these concrete choices:

- **Transport: HTTPS, always.** The sync server wraps its socket in TLS using a
  self-signed EC P-256 cert (`~/.cairn/tls_cert.pem`). The client
  (`http.client.HTTPSConnection`) pins the peer's cert SHA-256 — captured TOFU at
  pairing, stored in `sync_peers.peer_cert_fingerprint`, verified on every pull.
  No CA. Ed25519 request signatures still authenticate the *client* to the server;
  TLS authenticates the *server* to the client and encrypts everything.
- **Discovery: our own UDP broadcast, no mDNS.** Beacons (incl. cert fingerprint)
  on udp/47391. The **daemon** runs advertiser + listener threads when
  `CAIRN_SYNC_ENABLED=1`, so `discovered_peers` populates with zero user action.
  "Online" = a beacon seen within `CAIRN_SYNC_ONLINE_WINDOW` (90s).
- **Approval round-trip.** Requesters record an `outbound` row and poll the peer's
  signed `/pair/status`; on approval the peer is auto-promoted into `sync_peers`
  and pulling begins. The dashboard shows: online users (request access),
  incoming requests (approve/deny), my requests (waiting/approved), connected peers.
- **Schema v11** adds the cert-fingerprint columns. Wire `SCHEMA_VERSION = 11`.
- **Config**: `CAIRN_SYNC_ENABLED` (opt-in), `CAIRN_SYNC_PORT` (8787),
  `CAIRN_SYNC_BIND`, `CAIRN_SYNC_DISCOVERY_PORT` (47391),
  `CAIRN_SYNC_ADVERTISE_INTERVAL` (15s), `CAIRN_SYNC_ONLINE_WINDOW` (90s).

Deferred: mDNS/cross-subnet discovery; X25519 payload encryption (TLS covers the
wire); key/cert rotation UX.
