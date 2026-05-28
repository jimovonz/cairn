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

SCHEMA_VERSION = 4
