---
name: cairn-sync
description: Opt-in peer-to-peer LAN replication: Ed25519 pairing, TLS cert pinning, Lamport-clock LWW, changeset replication, session sharing.
---

## Multi-node sync (v2)

`cairn/sync/` is opt-in peer-to-peer LAN replication, **off by default** — set `CAIRN_SYNC_ENABLED=1` per node to opt in (wired into `install.sh`). When enabled the daemon runs the HTTPS sync server, UDP-broadcasts a LAN discovery beacon, and pulls from approved peers. Identity is an Ed25519 keypair; pairing is dashboard-authorized by public key; transport is signed + cert-pinned; replication is changeset-based with Lamport-clock last-write-wins. **Only a node's own memories are shared**, and raw session transcripts are never synced (a node may opt in to serving them behind its memories to approved peers via `CAIRN_SYNC_SHARE_SESSIONS=1`). Ports: HTTPS `CAIRN_SYNC_PORT=8787`, discovery `CAIRN_SYNC_DISCOVERY_PORT=47391`. See the Multi-User Architecture section of `ARCHITECTURE.md`.
