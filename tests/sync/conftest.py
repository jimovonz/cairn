"""Per-test isolated cairn nodes.

Each `node` fixture provides a fully initialised cairn DB with its own
node_id, user_id, and Lamport clock — simulating an independent install.
Embeddings are mocked deterministically (seed-based normalised float32).

Use `make_node()` factory to spawn N nodes in a single test.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Callable

import pytest

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeEmbedder:
    """Deterministic embedder — embeds by hashing text → seed → normalised vec."""

    def __init__(self) -> None:
        import numpy as np
        self.np = np

    def embed(self, text: str, allow_slow: bool = False):
        seed = abs(hash(text)) % (2**31)
        rng = self.np.random.RandomState(seed)
        v = rng.randn(384).astype(self.np.float32)
        return v / self.np.linalg.norm(v)

    def to_blob(self, vec) -> bytes:
        return vec.tobytes()

    def cosine_similarity(self, a, b) -> float:
        return float((a @ b) / (self.np.linalg.norm(a) * self.np.linalg.norm(b)))

    def upsert_vec_index(self, conn, mem_id: int, blob: bytes) -> None:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                (mem_id, blob),
            )
        except Exception:
            pass

    def find_nearest(self, conn, text: str, limit: int = 1):
        return []


class Node:
    """A self-contained cairn install."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state_dir = root / "state"
        self.db_path = str(root / "cairn.db")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Pre-write a stable node_id so identity helpers don't race
        self.node_id = str(uuid.uuid4())
        (self.state_dir / "node_id").write_text(self.node_id + "\n")
        (self.state_dir / "user_id").write_text(f"user-{self.node_id[:8]}\n")
        self._init_db()

    def _init_db(self) -> None:
        # Activate this node's identity for the duration of init
        old_state = os.environ.get("CAIRN_STATE_DIR")
        os.environ["CAIRN_STATE_DIR"] = str(self.state_dir)
        try:
            from cairn import init_db as init_module
            init_module.DB_PATH = self.db_path
            # Clear identity caches so it re-reads
            from cairn.sync import identity
            identity._MODEL_VERSION_CACHE.clear()
            init_module.init()
            # Pin the node_id into node_state so sync code (which reads from
            # the DB, not env) sees the test-assigned identity even after
            # CAIRN_STATE_DIR is restored.
            conn = self.conn()
            conn.execute(
                "INSERT INTO node_state (key, value) VALUES ('node_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (self.node_id,),
            )
            conn.commit()
            conn.close()
        finally:
            if old_state is None:
                os.environ.pop("CAIRN_STATE_DIR", None)
            else:
                os.environ["CAIRN_STATE_DIR"] = old_state

    def conn(self):
        try:
            import pysqlite3 as sqlite3
        except ImportError:
            import sqlite3
        return sqlite3.connect(self.db_path)

    def activate(self) -> None:
        """Make this node the 'current' one for storage.py operations."""
        os.environ["CAIRN_STATE_DIR"] = str(self.state_dir)
        os.environ["CAIRN_DB_PATH"] = self.db_path
        from cairn.sync import identity
        identity._MODEL_VERSION_CACHE.clear()

    def insert_memory(self, *, type_: str, topic: str, content: str,
                      project: str = "test", origin_id=None) -> int:
        """Direct insert for test setup — bypasses storage.py dedup."""
        from cairn.sync.identity import bump_lamport
        conn = self.conn()
        lam = bump_lamport(conn)
        embedder = FakeEmbedder()
        vec = embedder.embed(f"{type_} {topic} {content}")
        blob = embedder.to_blob(vec)
        oid = origin_id or str(uuid.uuid4())
        conn.execute(
            "INSERT INTO memories (type, topic, content, embedding, project, origin_id, "
            "created_by_node, updated_by_node, user_id, updated_by, lamport, visibility, embedding_model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'team', 'test-model@v1')",
            (type_, topic, content, blob, project, oid,
             self.node_id, self.node_id, f"user-{self.node_id[:8]}", f"user-{self.node_id[:8]}", lam),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        embedder.upsert_vec_index(conn, new_id, blob)
        conn.commit()
        conn.close()
        return new_id

    def confidence_vote(self, memory_origin: str, direction: str,
                        reason=None) -> None:
        from cairn.sync.identity import bump_lamport
        from cairn.sync.changeset import _recompute_confidence
        conn = self.conn()
        lam = bump_lamport(conn)
        conn.execute(
            "INSERT INTO confidence_log (log_uuid, memory_origin, direction, reason, "
            "node_id, user_id, lamport) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), memory_origin, direction, reason, self.node_id,
             f"user-{self.node_id[:8]}", lam),
        )
        _recompute_confidence(conn, memory_origin)
        conn.commit()
        conn.close()


@pytest.fixture
def make_node(tmp_path) -> Callable[[str], Node]:
    nodes: list[Node] = []
    def _factory(label: str = "node") -> Node:
        d = tmp_path / f"{label}-{len(nodes)}"
        d.mkdir(parents=True, exist_ok=True)
        n = Node(d)
        nodes.append(n)
        return n
    yield _factory
    # Clean up node-state caches
    from cairn.sync import identity
    identity._MODEL_VERSION_CACHE.clear()


@pytest.fixture
def node(make_node) -> Node:
    return make_node("solo")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
