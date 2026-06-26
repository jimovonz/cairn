# Cairn Multi-Node Sync — Design & As-Built

Status: **implemented (v2)** — peer-to-peer, offline-first, autodiscovery +
public-key pairing + HTTPS cert-pinned transport. Code lives in `cairn/sync/`
(`identity.py`, `discovery.py`, `pairing.py`, `server.py`, `client.py`,
`changeset.py`, `ephemeral.py`, `service.py`, `cli.py`).

> This document records both the design rationale (why the system is shaped the
> way it is) and the **as-built** protocol. Where the early v1 design and the
> shipped code diverge, the as-built sections are authoritative — the v1 design
> narrative is retained below for context because the *sync core* it describes
> (changeset extract/apply, Lamport clock, per-peer vector clock, LWW,
> visibility, embedding regen) is exactly what shipped. Only **how peers find
> each other** and **how trust is established** changed.

## Problem

Today every cairn install is a single-node SQLite database. We want a 20-engineer team to share memories across machines while:

1. **Working offline.** A laptop on a train must keep reading and writing locally; sync resumes when network returns.
2. **Surviving any peer being down.** No central authoritative server.
3. **Preserving authorship.** Who wrote it, who corroborated it, who contradicted it — all must travel with the row.
4. **Allowing private scratchpads.** Some memories are personal; some are team-shared.
5. **Avoiding embedding/FTS sync cost.** Derived data is regenerated locally on receipt.

## Non-goals

- Strong consistency. We accept eventual convergence; conflict resolution is deterministic but not "linearizable."
- Real-time push. Pull-based, polled. The shipped pull interval is `CAIRN_SYNC_PULL_INTERVAL` (default 120s).
- ~~Hiding nodes from each other. Peers are explicitly listed; no autodiscovery.~~ **Superseded:** v2 ships LAN autodiscovery (UDP broadcast). Discovery makes a node *visible*, not *trusted* — access still requires explicit pairing approval.

## Identity model

Two identity namespaces, one per concern:

- **Data-provenance identity** — `node_id`, a UUIDv4 in `~/.cairn/node_id`
  (generated on first run, survives DB rebuild). This is what is written to
  `memories.created_by_node` / `updated_by_node` and what the Lamport vector
  clock is keyed on. **It is NOT derived from the public key.**
- **Auth identity** — the Ed25519 **public-key fingerprint**,
  `base32(sha256(raw_pubkey))[:52]` (`identity.fingerprint`). This is
  self-certifying (a commitment to the public key) and is the value used as
  `X-Cairn-Node` on the wire, advertised in beacons, and pinned in
  `sync_peers`/`pairing_requests`. `service.start_sync_services` and the CLI use
  the fingerprint as the network node id.

These are two separate IDs by design — the provenance UUID stays stable for the
memory rows regardless of key rotation, while the cryptographic fingerprint is
the auth principal.

The per-row column model (from the v1 design, unchanged):

| Column | Scope | Semantics |
|---|---|---|
| `memories.id` (INTEGER PK) | local only | Auto-increment row counter. **Never synced.** FK target for local-only tables. |
| `memories.origin_id` (TEXT) | global | The row's cross-node identity (UUIDv4 at first insert; immutable). **The sync key.** |
| `memories.created_by_node` (TEXT) | global | Node UUID that first inserted the row. Provenance + vector-clock key. Immutable. |
| `memories.created_by_user` (`user_id`) | global | User identifier at the creating node. Immutable. |
| `memories.updated_by_node` (TEXT) | global | Last node to mutate the row. LWW tiebreak. Mutable. |
| `memories.lamport` (INTEGER) | global | Per-node Lamport clock at last mutation. Primary LWW ordering key. Mutable. |
| `memories.embedding_model_version` (TEXT) | global | Creator's embedding-model id, carried so a receiver knows whether it must regen. |

When node A's row crosses to node B, B inserts it with a fresh local `id` but
preserves `origin_id`. Local FK tables re-anchor to B's local `id` via
`origin_id` lookup on receipt.

### Node ID (data provenance)

`~/.cairn/node_id` — a single UUIDv4 written on first run (`ensure_node_id`).
Survives reinstall. **Not** derived from hostname/MAC. If deleted, a new
identity is generated and the node looks like a new peer.

### User ID

`~/.cairn/user_id` overrides; otherwise `f"{USER}#{first6hex(node_id)}"`
(`get_user_id`) — e.g. `alice#5f3a9c`. The short per-node suffix disambiguates
two people sharing a username on the LAN with zero setup. This is a human label
only; the auth principal is the public key.

### Keypair (auth)

Each node generates an **Ed25519 keypair** on first run (`ensure_keypair`):

- `~/.cairn/node_key`      — raw 32-byte private seed, `chmod 600`, never leaves the machine.
- `~/.cairn/node_key.pub`  — base64 public key.

The auth fingerprint is `fingerprint(pubkey)` (see Identity model). Keys survive
DB rebuild like `node_id`.

### TLS cert (transport)

Each node also generates a long-lived **self-signed EC P-256 cert**
(`ensure_tls_cert` → `~/.cairn/tls_cert.pem`, `tls_key.pem`). Peers pin its
SHA-256 fingerprint (TOFU) at pairing time; there is no CA.

## What syncs, what doesn't

| Table / column | Synced? | Notes |
|---|---|---|
| `memories` (canonical fields) | YES | `type, topic, content, confidence, archived_reason, deleted_at, keywords, project, depth, associated_files, origin_id, created_by_node, updated_by_node, lamport, visibility, embedding_model_version` (see `SYNCED_MEMORY_COLS` in `changeset.py`). |
| `memories.embedding` | NO | Regenerated locally on receipt. Avoids per-row payload + model-version coupling. |
| `memories.id, created_at, updated_at` | NO | Local counters / wall-clock. `lamport` is the cross-node ordering key. |
| `memories_fts`, `memories_vec` | NO | Derived; rebuilt by triggers / regen. |
| `memory_history` | YES | Re-anchored to local `id` via `origin_id` lookup on receipt. |
| `confidence_log` | YES | Confidence votes accumulate across the team (synced canonical source). |
| `correction_triggers` | YES | Re-anchored via `origin_id`. |
| `pair_assessments` | YES | LLM judgments are expensive — shared to dedupe work. Re-anchored via origin_id. |
| `ingested_repos` | YES | Per memory 1378. |
| `tombstones` | YES | Deletions propagate (earliest-wins). |
| `sessions, hook_state, metrics, ingestion_cache` | NO | Local operational state. |
| `discovered_peers, sync_state` | NO (and never synced) | Operational; live in the **ephemeral** DB (see below). |
| `sync_peers, pairing_requests` | NO | Local trust registry. |
| session excerpts | OPT-IN (`CAIRN_SYNC_SHARE_SESSIONS`) | Not bulk-synced — pulled lazily, one at a time, via `/session` (own-data-only, never for private memories). |
| Visibility = `private` | NEVER | Filtered out unconditionally at extraction time. |

### Own-data-only extraction

A node serves **only rows it authored** (`created_by_node == self_node`) from
`/sync` — there is no relay of other nodes' data. A full team picture therefore
requires pairing with each source directly; the per-source Lamport vector clock
still applies on top. (`extract_changeset`.)

## Conflict resolution

- **`content`, `topic`, `type`, `keywords`, `project`, `depth`,
  `archived_reason`** — Last-Writer-Wins. Incoming wins iff
  `(incoming_lamport, incoming_node) > (local_lamport, local_node)` compared
  lexicographically (`_apply_memory_row`). Lamport beats wall clocks; the
  node-id string comparison is the deterministic tiebreaker for equal lamports
  (the lexically-greater node id wins).
- **`confidence`** — Counter, not LWW. Recomputed deterministically from the
  union of `confidence_log` entries, so two nodes voting independently both
  stick and any two nodes with the same log converge to the same value.
- **`deleted_at`** — Tombstone (earliest-wins). Hard-delete is local-only and
  never propagates.
- **`memory_history`, `confidence_log`** — Pure append; union by row UUID.

### Lamport clock

```python
new = max(local_lamport, observed) + 1   # identity.bump_lamport
```

The per-node clock lives in `node_state` (key `lamport`). When applying a peer
changeset, `apply_changeset` tracks the highest received lamport and bumps the
local clock past it, so subsequent local edits are causally after observed peer
edits.

### Per-peer vector clock (`sync_state`)

`since_lamport_by_node` maps each *source node UUID* → the highest lamport this
node has already pulled for that source via a given peer. The pull request sends
it; `extract_changeset` returns only rows with `lamport > threshold` for their
`created_by_node`. This keeps gossip bandwidth O(N) instead of O(N²).
`sync_state` is stored in the ephemeral DB (`eph.sync_state`, PK
`(peer_node_id, source_node_id)`).

## Sync mechanism

**Hand-rolled, pull-based, peer-to-peer.** No central authority. JSON over
HTTPS.

### Transport: HTTPS, always

The sync server wraps its socket in TLS using the node's self-signed cert. The
client (`http.client.HTTPSConnection`) pins the peer's cert SHA-256 — captured
TOFU at pairing, stored in `sync_peers.peer_cert_fingerprint`, verified on
every request (`_https_post` raises `CertPinError` on mismatch). On the first
successful pull the fingerprint is pinned if it was not already. Ed25519 request
signatures authenticate the *client* to the server; TLS + cert pinning
authenticate the *server* to the client and encrypt everything. No CA.

### Endpoints

The server (`server.py`, stdlib `http.server`) exposes exactly four POST
endpoints:

- `POST /sync` — changeset pull (authorized, signed).
- `POST /pair` — queue a signed, self-certifying pairing request.
- `POST /pair/status` — a requester asks whether it has been approved yet.
- `POST /session` — serve the raw session excerpt behind one of *our* memories
  (own-data-only, never private; gated by `CAIRN_SYNC_SHARE_SESSIONS`).

(There is no `/changeset` or `/pull` endpoint — the pull endpoint is `/sync`.)

### Auth on every signed request

Each authorized request carries:

```
X-Cairn-Node:        <our auth fingerprint>
X-Cairn-Timestamp:   <unix seconds>
X-Cairn-Nonce:       <random>
X-Cairn-Signature:   base64( ed25519( METHOD \n PATH \n sha256(body) \n ts \n nonce ) )
X-Cairn-Schema-Version: <SCHEMA_VERSION>
```

The server's `_authorized` looks up `peer_public_key` by `X-Cairn-Node`,
rejects if `status != 'approved'`, rejects stale timestamps (`±300s`,
`_TS_TOLERANCE_SEC`) and replayed nonces (in-memory seen-set pruned at the same
tolerance), then verifies the Ed25519 signature over the canonical message.
**There is no bearer-token fallback** — it was removed (the `bearer_token`
column was dropped in durable schema v13) once no pre-v2 peers remained.

### Pull wire format

Request body to `/sync`:

```json
{
  "since_lamport_by_node": { "<source-node-uuid>": 18432, "...": 9921 },
  "max_rows": 5000,
  "include_excerpts": false
}
```

Response (`extract_changeset`):

```json
{
  "schema_version": 11,
  "node_id": "<server fingerprint>",
  "lamport_now": 18450,
  "memories":            [ ... ],
  "memory_history":      [ ... ],
  "confidence_log":      [ ... ],
  "correction_triggers": [ ... ],
  "pair_assessments":    [ ... ],
  "ingested_repos":      [ ... ],
  "tombstones":          [ ... ]
}
```

`max_rows` defaults to 5000. `include_excerpts` is per-peer
(`sync_peers.include_excerpts`).

The requesting node (`apply_changeset`):
1. Applies `memories` first, then `*_log`/`*_history`/tombstones (FK by `origin_id`).
2. Per `memories` row: lookup by `origin_id`; INSERT with fresh local `id` if absent, else per-column LWW merge.
3. Regenerates the embedding locally and upserts `memories_vec`; FTS triggers auto-fire.
4. Recomputes `confidence` from the union of local + received `confidence_log`.
5. Bumps the local Lamport clock past the highest received lamport.
6. Updates `sync_state` high-water marks for this peer.

### Schema-version handshake

Every request carries `X-Cairn-Schema-Version`. A peer below
`MIN_COMPATIBLE_SCHEMA_VERSION` is refused with `409` and a body
`{"error":"schema_version_mismatch","server":...,"min_compatible":...,"client":...}`.
Wire `SCHEMA_VERSION = 11`; `MIN_COMPATIBLE_SCHEMA_VERSION = 11`
(`cairn/sync/__init__.py`). `apply_changeset` reads a fixed column set via
`rec.get()`, so *additive* schema changes are tolerated across the floor —
newer fields are ignored, missing fields default to NULL. The compatibility
floor is bumped only for a non-additive/breaking change.

## Discovery (LAN autodiscovery)

Our own **UDP broadcast** — no mDNS (deferred). Beacons go out on
`CAIRN_SYNC_DISCOVERY_PORT` (default 47391) carrying public info only:

```json
{ "magic": "cairn-sync-beacon/1", "node_id": "<fingerprint>",
  "user_id": "alice#5f3a9c", "url": "https://192.168.1.42:8787",
  "public_key": "<b64>", "schema_version": 11, "cert_fingerprint": "<sha256>" }
```

Beacons are unauthenticated by design — discovery grants **no access**.
Listeners record beacons into `eph.discovered_peers`. To keep beacon churn off
the durable write lock, `record_beacon` throttles writes: it skips a write when
a fresh identical row exists newer than `DISCOVERY_WRITE_THROTTLE_SEC` (20s).
It also self-heals an approved peer's pull URL on IP change — but never touches
the pinned cert, so a lying beacon can't redirect a pull to a host that can't
present the pinned cert.

"Online" in the dashboard = a beacon seen within `CAIRN_SYNC_ONLINE_WINDOW`
(90s).

## Pairing handshake + dashboard authorization

Trust is established once per peer pair, interactively:

1. **Request.** Requester B dials A's `/pair` with a signed body
   `{node_id, public_key, user_id, url, cert_fingerprint, nonce/ts in headers}`.
   A verifies the signature proves possession of the presented key and that the
   claimed `node_id == fingerprint(public_key)` (self-certifying check). B also
   records a local `outbound` `pairing_requests` row.
2. **Queue.** A inserts an `inbound` `pairing_requests` row with
   `status='pending'` and returns `202`. No access granted.
3. **Authorize.** A's dashboard Sync tab shows the pending queue with B's
   `user_id`, fingerprint (for out-of-band verification, Syncthing-style), and
   source IP. The host clicks **Approve** or **Deny**.
4. **Pin.** On approve (`approve_pairing`), B's public key **and** cert
   fingerprint are pinned into `sync_peers` (`status='approved'`). From then on
   B's `/sync` requests authenticate by signature against the pinned key, over
   TLS pinned by cert fingerprint.

**Approval round-trip for the requester:** B polls A's signed `/pair/status`;
when A approves, `refresh_outbound` promotes the `outbound` row into
`sync_peers` and pulling begins. This runs automatically in the daemon's
listener thread and can also be triggered via the dashboard `/api/sync/refresh`.

Bidirectional sync requires **mutual approval** — A approving B lets B pull
from A; for A to pull from B, B must approve A symmetrically (and own-data-only
extraction means each side only ever serves its own rows).

Trust is **TOFU**: the fingerprint shown at approval is the commitment. Revoking
(`revoke_peer`) sets `sync_peers.status='revoked'`; future requests from that
key are rejected, but already-pulled memories are retained with attribution. A
revoked peer must be re-approved to resume.

## Operational tables on the ephemeral DB

`discovered_peers` (beacon cache) and `sync_state` (pull high-water vector
clock) are high-churn and fully recoverable (beacons re-arrive; a lost
high-water just triggers an idempotent re-pull), so they live in the
**ephemeral** DB, not the durable memory DB. `ephemeral.attach_ephemeral(conn)`
attaches it as alias `eph` and ensures the two tables exist; all sync SQL
references `eph.discovered_peers` / `eph.sync_state`. The path is derived
per-connection (the configured ephemeral DB for the real durable DB, otherwise
a `<stem>-ephemeral` sibling), giving each node — and each test node — its own.
This keeps sync off the durable write lock except for actual synced memories
plus the local `sync_peers`/`pairing_requests` trust registry.

## The daemon service

When `CAIRN_SYNC_ENABLED=1`, `service.start_sync_services` launches four
background threads (no user action beyond enabling sync):

1. **HTTPS sync server** (`/sync`, `/pair`, `/pair/status`, `/session`) on `CAIRN_SYNC_PORT`.
2. **Advertiser** — broadcasts a beacon every `CAIRN_SYNC_ADVERTISE_INTERVAL` (15s).
3. **Listener** — records beacons and periodically runs `refresh_outbound` to promote approved outbound pairings.
4. **Pull loop** — `pull_all` from every approved peer every `CAIRN_SYNC_PULL_INTERVAL` (120s).

`start_sync_services` advertises `https://<lan_ip>:<CAIRN_SYNC_PORT>` and the
auth **fingerprint** as the node id.

## CLI (`cairn-sync`)

`cli.py` exposes these subcommands (all over the same HTTPS/signature path):

| Subcommand | Purpose |
|---|---|
| `id` | show this node's sync identity (fingerprint, user_id, pubkey) |
| `serve` | run the sync HTTPS server (`--bind`, `--db`) |
| `advertise` | broadcast LAN discovery beacons (`--url`, `--sync-port`, `--port`, `--interval`) |
| `discover` | listen for peer beacons (`--timeout`, `--port`, `--watch`) |
| `pair <url>` | send a pairing request to a peer URL (`--my-url`, `--sync-port`) |
| `requests` | list pairing requests (`--all` includes decided) |
| `approve <id>` | approve a pending request (`--label`) → pin pubkey + cert |
| `deny <id>` | deny a pending request |
| `peers` | list paired peers |
| `revoke <node_id>` | revoke a paired peer |
| `pull` | pull changesets from peers (`--peer` for one) |
| `session <origin_id>` | fetch the raw session behind a synced memory |

## Dashboard endpoints

The dashboard Sync tab (`dashboard.py`) reads `discovered_peers` /
`pairing_requests` / `sync_peers`:

- `GET  /api/sync/identity`                       — this node's identity.
- `GET  /api/sync/pairing-requests`              — pending + recent decided.
- `GET  /api/sync/peers`                          — approved peers, last sync, errors.
- `GET  /api/sync/discovered`                     — raw beacon cache.
- `GET  /api/sync/online`                         — peers seen within `CAIRN_SYNC_ONLINE_WINDOW`, annotated with relationship state (connected / requested / incoming / available / revoked).
- `POST /api/sync/pairing-requests/<id>/approve`  — pin pubkey → `sync_peers`.
- `POST /api/sync/pairing-requests/<id>/deny`     — mark denied.
- `POST /api/sync/peers/<node_id>/revoke`          — set `status='revoked'`.
- `POST /api/sync/refresh`                         — poll outbound-pending peers and promote approved ones.
- `POST /api/sync/pair`                            — send an outbound pairing request from the dashboard.

## Embedding model versioning

`embedding_model_version` is a `<model-name>@<sha256-prefix>` string
(`get_embedding_model_version`, e.g.
`sentence-transformers/all-MiniLM-L6-v2@<8hex>`). The hash is name-derived (an
"advertised model version", honestly not a verified weight digest). The
canonical row carries the *creator's* version; the receiver always regenerates
the embedding locally with its own model and the local `memories_vec` is always
populated from local embeddings.

## Visibility

`visibility ∈ {private, team, public}`:

- **private** — never enters any sync payload (filtered unconditionally in
  `extract_changeset`); never served via `/session`.
- **team** — synced to approved peers (default).
- **public** — treated as `team`.

Flipping `team → private` does NOT retract previously-synced copies (impossible).

## Config flags (`cairn/config.py`)

| Flag | Default | Meaning |
|---|---|---|
| `CAIRN_SYNC_ENABLED` | **off** (`""`) | Master opt-in; gates the daemon services. |
| `CAIRN_SYNC_PORT` | `8787` | HTTPS sync server port. |
| `CAIRN_SYNC_BIND` | `0.0.0.0` | Sync server bind address. |
| `CAIRN_SYNC_DISCOVERY_PORT` | `47391` | UDP discovery broadcast port. |
| `CAIRN_SYNC_ADVERTISE_INTERVAL` | `15` (s) | Beacon broadcast interval. |
| `CAIRN_SYNC_PULL_INTERVAL` | `120` (s) | Pull-loop interval. |
| `CAIRN_SYNC_ONLINE_WINDOW` | `90` (s) | Beacon freshness window for "online". |
| `CAIRN_SYNC_SHARE_SESSIONS` | **off** | Allow serving raw session excerpts via `/session`. |

## Schema versions

- **Wire** `SCHEMA_VERSION = 11` (HTTPS + cert pinning; 10 = pubkey
  pairing/signatures; 9 = bearer). `MIN_COMPATIBLE_SCHEMA_VERSION = 11`.
- **Durable** migrations advanced past the wire version for changes that don't
  touch the wire: **v12** moved `discovered_peers`/`sync_state` to the ephemeral
  DB; **v13** dropped `sync_peers.bearer_token`. Neither bumps the wire version
  because those tables were never synced and `sync_peers` is local-only.

## Open questions / deferred

- mDNS / cross-subnet discovery (UDP broadcast is the shipped default).
- X25519 payload encryption (TLS already covers the wire).
- Key/cert rotation UX — rotating a keypair changes the fingerprint, so the node
  reads as a new peer and must be re-approved.
- At-rest confidentiality — local DBs are still plaintext.
- Per-author confidence weighting (memory 581) — deferred until multi-node data shows it matters.
- Selective sync ("don't pull project X") — deferred.
