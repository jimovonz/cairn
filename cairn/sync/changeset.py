"""Changeset extraction and application — the core merge engine.

A changeset is a JSON-serialisable dict containing all rows from synced tables
that have a `lamport` value greater than the requesting peer's vector-clock entry
for the originating node.

Extraction respects visibility (`private` rows are filtered out unconditionally).
Application is per-column LWW with Lamport-clock primary ordering and node-id
lexicographic tiebreak. Confidence is convergent: rebuilt deterministically from
the union of `confidence_log` entries.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore

from cairn.sync import SCHEMA_VERSION, MIN_COMPATIBLE_SCHEMA_VERSION
from cairn.sync.identity import bump_lamport, get_embedding_model_version, node_id_for_conn


# ---- Synced columns on memories ----
# Anything not in this list is local-only (id, created_at, updated_at, embedding,
# session_id, anchor_line, source_start, source_end, source_ref).
SYNCED_MEMORY_COLS = (
    "origin_id",          # cross-node row identity
    "type",
    "topic",
    "content",
    "confidence",         # carried but recomputed from confidence_log on apply
    "archived_reason",
    "deleted_at",
    "keywords",
    "project",
    "depth",
    "associated_files",
    "created_by_node",
    "created_by_user",    # column name on disk: user_id
    "updated_by_node",
    "updated_by_user",    # column name on disk: updated_by
    "lamport",
    "visibility",
    "embedding_model_version",
)

# Map wire-name -> on-disk column name for the few that differ
_MEM_DISK_COL = {
    "created_by_user": "user_id",
    "updated_by_user": "updated_by",
}

def _disk_col(wire: str) -> str:
    return _MEM_DISK_COL.get(wire, wire)


# =====================================================================
# Extraction
# =====================================================================

def extract_changeset(
    conn,
    since_lamport_by_node: dict[str, int],
    *,
    include_excerpts: bool = False,
    max_rows: int = 5000,
) -> dict[str, Any]:
    """Return a JSON-serialisable changeset for the requesting peer.

    `since_lamport_by_node` maps source-node UUIDs to the highest lamport
    the peer has already seen for that source. Rows are filtered to those
    where `lamport > since_lamport_by_node.get(created_by_node, 0)`.
    Visibility = 'private' rows are never included.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "lamport_now": _peek_lamport(conn),
        "memories": [],
        "memory_history": [],
        "confidence_log": [],
        "correction_triggers": [],
        "pair_assessments": [],
        "ingested_repos": [],
        "tombstones": [],
    }

    # Own-data-only: a node shares ONLY rows it authored (created_by_node == self).
    # Peers get a person's memories by pairing with them directly — no transitive
    # relay of other nodes' data. The per-source lamport vector clock still applies
    # (now effectively just this node's own watermark).
    self_node = node_id_for_conn(conn)

    # --- memories ---
    cols_sql = ", ".join(_disk_col(c) for c in SYNCED_MEMORY_COLS)
    rows = conn.execute(
        f"SELECT {cols_sql} FROM memories "
        f"WHERE (visibility != 'private' OR visibility IS NULL) AND created_by_node = ?",
        (self_node,),
    ).fetchall()
    count = 0
    for row in rows:
        rec = dict(zip(SYNCED_MEMORY_COLS, row))
        cb_node = rec.get("created_by_node")
        if not cb_node:
            # Pre-v4 row that wasn't backfilled. Skip — we can't attribute it.
            continue
        threshold = since_lamport_by_node.get(cb_node, 0)
        if (rec.get("lamport") or 0) <= threshold:
            continue
        # Tombstones travel separately for compactness, but the row also goes if other columns changed
        payload["memories"].append(rec)
        count += 1
        if count >= max_rows:
            break

    # --- confidence_log ---
    log_rows = conn.execute(
        "SELECT log_uuid, memory_origin, direction, reason, node_id, user_id, "
        "session_id, lamport, created_at FROM confidence_log WHERE node_id = ?",
        (self_node,),
    ).fetchall()
    for r in log_rows:
        log_uuid, mem_origin, direction, reason, node_id, user_id, sid, lam, created_at = r
        threshold = since_lamport_by_node.get(node_id, 0)
        if (lam or 0) <= threshold:
            continue
        payload["confidence_log"].append({
            "log_uuid": log_uuid, "memory_origin": mem_origin,
            "direction": direction, "reason": reason,
            "node_id": node_id, "user_id": user_id, "session_id": sid,
            "lamport": lam, "created_at": created_at,
        })

    # --- memory_history ---
    hist_rows = conn.execute(
        "SELECT history_uuid, memory_origin, content, session_id, changed_at, "
        "changed_by_node, lamport FROM memory_history "
        "WHERE history_uuid IS NOT NULL AND changed_by_node = ?",
        (self_node,),
    ).fetchall()
    for r in hist_rows:
        h_uuid, mem_origin, content, sid, changed_at, by_node, lam = r
        threshold = since_lamport_by_node.get(by_node, 0)
        if (lam or 0) <= threshold:
            continue
        payload["memory_history"].append({
            "history_uuid": h_uuid, "memory_origin": mem_origin,
            "content": content, "session_id": sid, "changed_at": changed_at,
            "changed_by_node": by_node, "lamport": lam,
        })

    # --- correction_triggers ---
    # These FK to local memory_id; re-key to origin_id for transport.
    ct_rows = conn.execute(
        "SELECT ct.id, m.origin_id, ct.trigger, ct.created_at "
        "FROM correction_triggers ct JOIN memories m ON ct.memory_id = m.id "
        "WHERE (m.visibility != 'private' OR m.visibility IS NULL) AND m.created_by_node = ?",
        (self_node,),
    ).fetchall()
    for ct_id, mem_origin, trig, created_at in ct_rows:
        # No native UUID — synthesise a deterministic one
        synth_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"ct:{mem_origin}:{ct_id}:{trig[:80]}"))
        payload["correction_triggers"].append({
            "trigger_uuid": synth_uuid, "memory_origin": mem_origin,
            "trigger": trig, "created_at": created_at,
        })

    # --- pair_assessments (re-keyed via origin_id) ---
    pa_rows = conn.execute(
        "SELECT ma.origin_id, mb.origin_id, pa.mode, pa.verdict, pa.reason, pa.assessed_at "
        "FROM pair_assessments pa "
        "JOIN memories ma ON pa.memory_id_a = ma.id "
        "JOIN memories mb ON pa.memory_id_b = mb.id "
        "WHERE (ma.visibility != 'private' OR ma.visibility IS NULL) "
        "AND (mb.visibility != 'private' OR mb.visibility IS NULL) "
        "AND ma.created_by_node = ? AND mb.created_by_node = ?",
        (self_node, self_node),
    ).fetchall()
    for oa, ob, mode, verdict, reason, at in pa_rows:
        payload["pair_assessments"].append({
            "memory_origin_a": oa, "memory_origin_b": ob,
            "mode": mode, "verdict": verdict, "reason": reason, "assessed_at": at,
        })

    # --- tombstones (deletions only) ---
    tomb_rows = conn.execute(
        "SELECT origin_id, deleted_at, updated_by_node, lamport "
        "FROM memories WHERE deleted_at IS NOT NULL AND updated_by_node = ?",
        (self_node,),
    ).fetchall()
    for origin, deleted_at, by_node, lam in tomb_rows:
        threshold = since_lamport_by_node.get(by_node, 0)
        if (lam or 0) <= threshold:
            continue
        payload["tombstones"].append({
            "origin_id": origin, "deleted_at": deleted_at,
            "updated_by_node": by_node, "lamport": lam,
        })

    # --- ingested_repos (table may not exist in all installs) ---
    try:
        ir_rows = conn.execute("SELECT * FROM ingested_repos").fetchall()
        ir_cols = [d[0] for d in conn.execute("SELECT * FROM ingested_repos LIMIT 0").description]
        payload["ingested_repos"] = [dict(zip(ir_cols, r)) for r in ir_rows]
    except sqlite3.OperationalError:
        pass

    return payload


# =====================================================================
# Application
# =====================================================================

class ApplyResult:
    def __init__(self) -> None:
        self.memories_inserted = 0
        self.memories_updated = 0
        self.memories_skipped_lww = 0
        self.confidence_log_added = 0
        self.history_added = 0
        self.tombstones_applied = 0
        self.embeddings_regenerated = 0
        self.errors: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "memories_inserted": self.memories_inserted,
            "memories_updated": self.memories_updated,
            "memories_skipped_lww": self.memories_skipped_lww,
            "confidence_log_added": self.confidence_log_added,
            "history_added": self.history_added,
            "tombstones_applied": self.tombstones_applied,
            "embeddings_regenerated": self.embeddings_regenerated,
            "errors": self.errors,
        }


def apply_changeset(
    conn,
    payload: dict[str, Any],
    *,
    embedder=None,
    regen_embeddings: bool = True,
) -> ApplyResult:
    """Apply a peer changeset to the local DB.

    Per-column LWW: a row's column is updated iff incoming
    (lamport, updated_by_node) > local (lamport, updated_by_node) lex-ordered.
    Confidence is recomputed from the union of confidence_log entries.
    Embeddings are regenerated locally on insert/content-change unless disabled.
    """
    # Accept payloads from any peer >= our compatibility floor; apply reads a
    # fixed column set so additive drift is tolerated. Only a below-floor
    # (breaking) payload is rejected.
    _pv = payload.get("schema_version") or 0
    if _pv < MIN_COMPATIBLE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version too old: payload={payload.get('schema_version')} "
            f"min_compatible={MIN_COMPATIBLE_SCHEMA_VERSION}"
        )
    result = ApplyResult()

    max_observed_lamport = 0

    # --- memories ---
    # Commit per row so the write lock is released between rows: embedding
    # regeneration (the slow part, inside _apply_memory_row) then happens with NO
    # write lock held, and the memory-capture drain can interleave. Holding one
    # big transaction across every row's embed previously starved capture
    # ("database is locked"). Safe to commit incrementally — sync is idempotent
    # (per-column LWW; a re-pull re-applies), so partial application never corrupts.
    for rec in payload.get("memories", []):
        try:
            updated, inserted, lam = _apply_memory_row(conn, rec, embedder, regen_embeddings, result)
            if inserted:
                result.memories_inserted += 1
            elif updated:
                result.memories_updated += 1
            else:
                result.memories_skipped_lww += 1
            max_observed_lamport = max(max_observed_lamport, lam or 0)
            conn.commit()
        except Exception as e:  # surface but don't abort
            result.errors.append(f"memory {rec.get('origin_id')}: {e}")
            try:
                conn.rollback()
            except Exception:
                pass

    # --- confidence_log (idempotent on log_uuid) ---
    affected_origins: set[str] = set()
    for entry in payload.get("confidence_log", []):
        try:
            existing = conn.execute(
                "SELECT 1 FROM confidence_log WHERE log_uuid = ?", (entry["log_uuid"],)
            ).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO confidence_log (log_uuid, memory_origin, direction, reason, "
                "node_id, user_id, session_id, lamport, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry["log_uuid"], entry["memory_origin"], entry["direction"],
                 entry.get("reason"), entry["node_id"], entry.get("user_id"),
                 entry.get("session_id"), entry["lamport"], entry.get("created_at"))
            )
            result.confidence_log_added += 1
            affected_origins.add(entry["memory_origin"])
            max_observed_lamport = max(max_observed_lamport, entry.get("lamport") or 0)
        except Exception as e:
            result.errors.append(f"confidence_log {entry.get('log_uuid')}: {e}")

    # --- history (idempotent on history_uuid) ---
    for h in payload.get("memory_history", []):
        try:
            existing = conn.execute(
                "SELECT 1 FROM memory_history WHERE history_uuid = ?", (h["history_uuid"],)
            ).fetchone()
            if existing:
                continue
            local_id = _local_id_for_origin(conn, h["memory_origin"])
            conn.execute(
                "INSERT INTO memory_history (memory_id, content, session_id, changed_at, "
                "history_uuid, memory_origin, changed_by_node, lamport) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (local_id, h["content"], h.get("session_id"), h.get("changed_at"),
                 h["history_uuid"], h["memory_origin"], h.get("changed_by_node"), h["lamport"])
            )
            result.history_added += 1
            max_observed_lamport = max(max_observed_lamport, h.get("lamport") or 0)
        except Exception as e:
            result.errors.append(f"history {h.get('history_uuid')}: {e}")

    # --- tombstones (earliest non-null wins) ---
    for t in payload.get("tombstones", []):
        try:
            local = conn.execute(
                "SELECT id, deleted_at FROM memories WHERE origin_id = ?", (t["origin_id"],)
            ).fetchone()
            if not local:
                continue
            local_id, local_deleted = local
            if local_deleted is None or t["deleted_at"] < local_deleted:
                conn.execute(
                    "UPDATE memories SET deleted_at = ?, updated_by_node = ?, lamport = ? "
                    "WHERE id = ?",
                    (t["deleted_at"], t.get("updated_by_node"), t["lamport"], local_id)
                )
                result.tombstones_applied += 1
            max_observed_lamport = max(max_observed_lamport, t.get("lamport") or 0)
        except Exception as e:
            result.errors.append(f"tombstone {t.get('origin_id')}: {e}")

    # --- correction_triggers (idempotent on trigger_uuid; we don't store the uuid yet,
    # so dedup by (memory_origin, trigger text)) ---
    for ct in payload.get("correction_triggers", []):
        try:
            local_id = _local_id_for_origin(conn, ct["memory_origin"])
            if local_id is None:
                continue
            existing = conn.execute(
                "SELECT 1 FROM correction_triggers WHERE memory_id = ? AND trigger = ?",
                (local_id, ct["trigger"])
            ).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO correction_triggers (memory_id, trigger, created_at) VALUES (?, ?, ?)",
                (local_id, ct["trigger"], ct.get("created_at"))
            )
        except Exception as e:
            result.errors.append(f"trigger: {e}")

    # --- pair_assessments (re-key origin_id back to local id) ---
    for pa in payload.get("pair_assessments", []):
        try:
            ida = _local_id_for_origin(conn, pa["memory_origin_a"])
            idb = _local_id_for_origin(conn, pa["memory_origin_b"])
            if ida is None or idb is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO pair_assessments (memory_id_a, memory_id_b, mode, verdict, reason, assessed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ida, idb, pa["mode"], pa["verdict"], pa.get("reason"), pa.get("assessed_at"))
            )
        except Exception as e:
            result.errors.append(f"pair_assessment: {e}")

    # --- recompute confidence for affected memories ---
    for origin in affected_origins:
        try:
            _recompute_confidence(conn, origin)
        except Exception as e:
            result.errors.append(f"confidence recompute {origin}: {e}")

    # --- bump local Lamport past observed peer values ---
    if max_observed_lamport:
        bump_lamport(conn, observed=max_observed_lamport)

    conn.commit()
    return result


# =====================================================================
# Per-row LWW merge for a memory
# =====================================================================

def _apply_memory_row(conn, rec: dict, embedder, regen_embeddings: bool, result: ApplyResult):
    """Returns (updated: bool, inserted: bool, lamport: int)."""
    origin = rec.get("origin_id")
    if not origin:
        raise ValueError("memory missing origin_id")
    local = conn.execute(
        "SELECT id, lamport, updated_by_node, content FROM memories WHERE origin_id = ?",
        (origin,),
    ).fetchone()

    incoming_lamport = rec.get("lamport") or 0
    incoming_node = rec.get("updated_by_node") or rec.get("created_by_node") or ""

    if local is None:
        # Insert. embedding regenerated locally if embedder available.
        embedding_blob, model_version = _maybe_regen_embedding(rec, embedder, regen_embeddings)
        if embedding_blob is not None:
            result.embeddings_regenerated += 1
        cols = list(SYNCED_MEMORY_COLS) + ["embedding"]
        disk_cols = [_disk_col(c) for c in SYNCED_MEMORY_COLS] + ["embedding"]
        vals = [rec.get(c) for c in SYNCED_MEMORY_COLS]
        # If we regenerated an embedding under a different model, advertise the local model
        if embedding_blob is not None and model_version:
            idx = SYNCED_MEMORY_COLS.index("embedding_model_version")
            vals[idx] = model_version
        vals.append(embedding_blob)
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(disk_cols)
        conn.execute(f"INSERT INTO memories ({col_list}) VALUES ({placeholders})", vals)
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        if embedding_blob is not None and embedder is not None:
            try:
                embedder.upsert_vec_index(conn, new_id, embedding_blob)
            except Exception:
                pass
        return False, True, incoming_lamport

    local_id, local_lamport, local_node, local_content = local
    local_lamport = local_lamport or 0
    local_node = local_node or ""

    # LWW: incoming wins iff (lamport, node_id) > local (lex)
    if (incoming_lamport, incoming_node) <= (local_lamport, local_node):
        return False, False, incoming_lamport

    # Per-column update — simplest correct: update all synced columns to incoming values.
    # Trigger memories_version snapshots old content automatically; the snapshot will get
    # synced to peers via memory_history on next pull (with NULL history_uuid → it gets
    # backfilled on next init). For determinism we set updated_at via SQLite default.
    set_clause = ", ".join(f"{_disk_col(c)} = ?" for c in SYNCED_MEMORY_COLS)
    vals = [rec.get(c) for c in SYNCED_MEMORY_COLS]
    content_changed = (rec.get("content") != local_content)
    embedding_blob, model_version = (None, None)
    if content_changed and regen_embeddings:
        embedding_blob, model_version = _maybe_regen_embedding(rec, embedder, regen_embeddings)
        if embedding_blob is not None:
            result.embeddings_regenerated += 1
            idx = SYNCED_MEMORY_COLS.index("embedding_model_version")
            vals[idx] = model_version or vals[idx]
    if embedding_blob is not None:
        set_clause += ", embedding = ?"
        vals.append(embedding_blob)
    vals.append(local_id)
    conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", vals)
    if embedding_blob is not None and embedder is not None:
        try:
            embedder.upsert_vec_index(conn, local_id, embedding_blob)
        except Exception:
            pass
    return True, False, incoming_lamport


def _maybe_regen_embedding(rec: dict, embedder, regen: bool):
    """Regenerate embedding locally for an incoming row. Returns (blob, model_version)."""
    if not regen or embedder is None:
        return None, None
    try:
        project = rec.get("project") or ""
        text = f"{project + ' ' if project else ''}{rec.get('type','')} {rec.get('topic','')} {rec.get('content','')}"
        vec = embedder.embed(text, allow_slow=False)
        if vec is None:
            return None, None
        return embedder.to_blob(vec), get_embedding_model_version()
    except Exception:
        return None, None


def _local_id_for_origin(conn, origin_id: str) -> Optional[int]:
    if not origin_id:
        return None
    row = conn.execute("SELECT id FROM memories WHERE origin_id = ?", (origin_id,)).fetchone()
    return row[0] if row else None


def _peek_lamport(conn) -> int:
    row = conn.execute("SELECT value FROM node_state WHERE key = 'lamport'").fetchone()
    return int(row[0]) if row else 0


# =====================================================================
# Confidence recomputation (deterministic across nodes)
# =====================================================================

def _recompute_confidence(conn, memory_origin: str) -> None:
    """Recompute memories.confidence from the union of confidence_log entries.

    Deterministic algorithm:
      start = CONFIDENCE_DEFAULT
      for entry in sorted(log_entries, key=(lamport, log_uuid)):
          if direction == '+': c = min(c + BOOST*(1-c), CONFIDENCE_MAX)
          if direction == '-':  no-op
          if direction == '-!': annotate archived_reason (last write wins)
    """
    from cairn.config import CONFIDENCE_DEFAULT, CONFIDENCE_BOOST, CONFIDENCE_MAX
    rows = conn.execute(
        "SELECT log_uuid, direction, reason, lamport FROM confidence_log "
        "WHERE memory_origin = ? ORDER BY lamport, log_uuid",
        (memory_origin,)
    ).fetchall()
    c = CONFIDENCE_DEFAULT
    annotation: Optional[str] = None
    for log_uuid, direction, reason, lamport in rows:
        if direction == "+":
            c = min(c + CONFIDENCE_BOOST * (1 - c), CONFIDENCE_MAX)
        elif direction == "-!":
            annotation = reason or "contradicted by later session"
    update_sql = "UPDATE memories SET confidence = ?"
    params: list = [c]
    if annotation is not None:
        update_sql += ", archived_reason = ?"
        params.append(annotation)
    update_sql += " WHERE origin_id = ?"
    params.append(memory_origin)
    conn.execute(update_sql, params)
