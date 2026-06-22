"""Cairn multi-node sync — peer-to-peer offline-first replication.

See docs/multi-node-sync.md for the design. Public surface:

    from cairn.sync.identity import (
        ensure_node_id, get_user_id, get_embedding_model_version,
        bump_lamport, peek_lamport,
    )
    from cairn.sync.changeset import extract_changeset, apply_changeset
    from cairn.sync.client import pull_from_peer
    from cairn.sync.server import build_app
"""

SCHEMA_VERSION = 11  # v11 = HTTPS + cert pinning; v10 = pubkey pairing/signatures; 9 = bearer
# The oldest wire version this node still interoperates with. Sync is accepted
# from any peer >= this floor (apply_changeset reads a fixed column set via
# rec.get(), so additive schema changes are tolerated — newer fields ignored,
# missing fields default to NULL). DISCIPLINE: bump this ONLY for a
# non-additive/breaking change; additive columns/tables must NOT raise it.
MIN_COMPATIBLE_SCHEMA_VERSION = 11
