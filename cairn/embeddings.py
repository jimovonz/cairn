#!/usr/bin/env python3
"""Embedding utilities for Cairn."""

from __future__ import annotations

import numpy as np
import os
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import struct
from datetime import datetime
from typing import Any, Optional

_model: Any = None
_metrics_conn: Optional[sqlite3.Connection] = None


def _log_embed(msg: str, level: str = "info") -> None:
    """Log from embeddings module. Uses cairn logger if available, falls back to stderr."""
    try:
        import logging
        logger = logging.getLogger("cairn")
        if logger.handlers:
            getattr(logger, level, logger.info)(f"[embeddings] {msg}")
            return
    except Exception:
        pass
    import sys
    print(f"[cairn/embeddings] {msg}", file=sys.stderr)


def _record_embed_metric(event: str, value: float) -> None:
    """Record an embedding performance metric to the DB. Best-effort, never raises."""
    global _metrics_conn
    try:
        if _metrics_conn is None:
            from cairn.config import EPHEMERAL_DB_PATH
            _metrics_conn = sqlite3.connect(EPHEMERAL_DB_PATH)
            _metrics_conn.execute("PRAGMA busy_timeout=2000")
            _metrics_conn.execute("PRAGMA journal_mode=WAL")
            try:
                _metrics_conn.execute("SELECT 1 FROM metrics LIMIT 0")
            except sqlite3.OperationalError:
                from cairn.init_db import init_ephemeral
                init_ephemeral(EPHEMERAL_DB_PATH)
        _metrics_conn.execute(
            "INSERT INTO metrics (event, value) VALUES (?, ?)",
            (event, value)
        )
        _metrics_conn.commit()
    except Exception as exc:
        _log_embed(f"metric write failed ({event}): {type(exc).__name__}: {exc}", "warning")
        _metrics_conn = None


_daemon_start_attempted: bool = False


def _daemon_embed(text: str) -> Optional[np.ndarray]:
    """Try embedding via the daemon. Auto-starts daemon on first failure."""
    global _daemon_start_attempted
    try:
        from cairn.daemon import send_request, is_running, SOCKET_PATH
        import subprocess

        # Auto-start daemon if not running (once per process, with file lock to prevent races)
        if not is_running() and not _daemon_start_attempted:
            _daemon_start_attempted = True
            daemon_path = os.path.join(os.path.dirname(__file__), "daemon.py")
            venv_python = os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "python3")
            lock_path = os.path.join(os.path.dirname(__file__), ".daemon.lock")
            if os.path.exists(venv_python):
                import fcntl, time
                try:
                    lock_fd = open(lock_path, "w")
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Double-check after acquiring lock
                    if not is_running():
                        subprocess.Popen([venv_python, daemon_path, "start"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        time.sleep(2)
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except (IOError, OSError):
                    # Another process holds the lock — daemon is being started
                    import time
                    time.sleep(3)

        resp = send_request({"action": "embed", "text": text})
        if resp and "vector" in resp:
            try:
                from hooks.health import record_success
                record_success("daemon")
            except Exception:
                pass
            return np.frombuffer(bytes.fromhex(resp["vector"]), dtype=np.float32)
    except (ConnectionError, TimeoutError, OSError, ValueError) as e:
        _log_embed(f"daemon embed unavailable: {type(e).__name__}: {e}", "warning")
        try:
            from hooks.health import record_failure
            record_failure("daemon", str(e))
        except Exception:
            pass
    except Exception as e:
        _log_embed(f"daemon embed unexpected error: {type(e).__name__}: {e}", "error")
    return None


def _daemon_embed_batch(texts: list[str]) -> Optional[list[np.ndarray]]:
    """Embed many texts in one daemon round-trip. Returns vectors or None."""
    try:
        from cairn.daemon import send_request
        resp = send_request({"action": "embed_batch", "texts": texts})
        if resp and resp.get("vectors") is not None:
            return [np.frombuffer(bytes.fromhex(h), dtype=np.float32)
                    for h in resp["vectors"]]
    except (ConnectionError, TimeoutError, OSError, ValueError) as e:
        _log_embed(f"daemon embed_batch unavailable: {type(e).__name__}: {e}", "warning")
    except Exception as e:
        _log_embed(f"daemon embed_batch unexpected error: {type(e).__name__}: {e}", "error")
    return None


def embed_batch(texts: list[str], allow_slow: bool = True) -> Optional[list[np.ndarray]]:
    """Embed a list of texts, preferring one batched daemon call.

    Falls back to per-text embed() when the running daemon predates the
    embed_batch action or is down. Returns None if any text fails to embed.

    Test seam: the suite patches embed() (patch.object(emb, "embed", ...));
    when that global has been replaced, route per-text through it instead of
    the daemon so mocked vectors are honored.
    """
    if not texts:
        return []
    if embed is not _EMBED_ORIGINAL:
        out_mocked: list[np.ndarray] = []
        for t in texts:
            v = embed(t, allow_slow=allow_slow)
            if v is None:
                return None
            out_mocked.append(v)
        return out_mocked
    import time as _time
    t0 = _time.perf_counter()
    vecs = _daemon_embed_batch(texts)
    if vecs is not None and len(vecs) == len(texts):
        _record_embed_metric("embed_batch_ms", (_time.perf_counter() - t0) * 1000)
        return vecs
    out: list[np.ndarray] = []
    for t in texts:
        v = embed(t, allow_slow=allow_slow)
        if v is None:
            return None
        out.append(v)
    return out


def _conn_is_main_db(conn) -> bool:
    """True when conn points at the production cairn.db — the only DB the
    daemon-resident matrices mirror. Callers passing any other connection
    (tests, alternate DBs) must take the local scoring path."""
    try:
        main_db = os.path.realpath(os.path.join(os.path.dirname(__file__), "cairn.db"))
        for _, name, file in conn.execute("PRAGMA database_list").fetchall():
            if name == "main":
                return bool(file) and os.path.realpath(file) == main_db
    except Exception:
        pass
    return False


def _daemon_vector_search(texts: list[str], n_base: int, min_sim: float,
                          top_k: int = 300) -> Optional[list[dict]]:
    """Score query variants against the daemon-resident memory matrices.

    Returns scored candidate rows (metadata + similarity) or None when the
    daemon is down or predates the vector_search action — callers fall back
    to the local fetch-and-score path.
    """
    try:
        from cairn.daemon import send_request
        resp = send_request({"action": "vector_search", "texts": texts,
                             "n_base": n_base, "min_sim": min_sim, "top_k": top_k})
        if resp and resp.get("rows") is not None:
            return resp["rows"]
    except (ConnectionError, TimeoutError, OSError, ValueError) as e:
        _log_embed(f"daemon vector_search unavailable: {type(e).__name__}: {e}", "warning")
    except Exception as e:
        _log_embed(f"daemon vector_search unexpected error: {type(e).__name__}: {e}", "error")
    return None


def _daemon_rerank(query: str, candidates: list[str]):
    """Re-rank candidates via the daemon's cross-encoder. Returns
    (scores, score_floor, model_name) or None. The daemon resolves the
    device-appropriate model + floor (bge on CUDA, ms-marco on CPU) and reports
    both (+ the loaded model name, for delivery provenance) so this process
    needn't import torch."""
    try:
        from cairn.daemon import send_request
        resp = send_request({"action": "rerank", "query": query, "candidates": candidates})
        if resp and resp.get("scores") is not None:
            return resp["scores"], resp.get("score_floor"), resp.get("model")
    except (ConnectionError, TimeoutError, OSError) as e:
        _log_embed(f"daemon rerank unavailable: {type(e).__name__}: {e}", "warning")
    except Exception as e:
        _log_embed(f"daemon rerank error: {type(e).__name__}: {e}", "error")
    return None


def _daemon_nli(pairs: list[list[str]]) -> Optional[list[list[float]]]:
    """Score (premise, hypothesis) pairs via the daemon's NLI model.

    Returns a list of [contradiction, entailment, neutral] score triples, or None.
    """
    try:
        from cairn.daemon import send_request
        resp = send_request({"action": "nli", "pairs": pairs})
        if resp and resp.get("scores") is not None:
            return resp["scores"]
    except (ConnectionError, TimeoutError, OSError) as e:
        _log_embed(f"daemon NLI unavailable: {type(e).__name__}: {e}", "warning")
    except Exception as e:
        _log_embed(f"daemon NLI error: {type(e).__name__}: {e}", "error")
    return None


def find_clusters(
    conn: sqlite3.Connection,
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_cluster_size: int = 10,
) -> list[list[dict]]:
    """Find clusters of semantically similar active memories using bi-encoder embeddings.

    Returns a list of clusters, each a list of memory dicts sorted by recency.
    """
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, confidence, session_id "
        "FROM memories WHERE embedding IS NOT NULL AND (archived_reason IS NULL OR archived_reason = '') AND deleted_at IS NULL"
    ).fetchall()

    entries = []
    for row in rows:
        vec = from_blob(row[4])
        entries.append({
            "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
            "vec": vec, "updated_at": row[5], "project": row[6], "confidence": row[7] or 0.7,
            "session_id": row[8],
        })

    assigned: set[int] = set()
    clusters: list[list[dict]] = []

    for i, a in enumerate(entries):
        if a["id"] in assigned:
            continue
        cluster = [a]
        assigned.add(a["id"])
        for j in range(i + 1, len(entries)):
            b = entries[j]
            if b["id"] in assigned:
                continue
            if len(cluster) >= max_cluster_size:
                break
            sim = cosine_similarity(a["vec"], b["vec"])
            if sim >= similarity_threshold:
                cluster.append(b)
                assigned.add(b["id"])
        if len(cluster) >= min_cluster_size:
            cluster.sort(key=lambda x: x["updated_at"] or "", reverse=True)
            clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


def get_model() -> Any:
    global _model
    if _model is None:
        import torch
        # Force CPU if CUDA is available but incompatible (avoids kernel launch failures)
        device = "cpu"
        if torch.cuda.is_available():
            try:
                # Test with a small tensor to verify CUDA actually works
                torch.zeros(1, device="cuda")
                device = "cuda"
            except RuntimeError:
                pass  # Incompatible GPU — stay on CPU
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _model


def embed(text: str, allow_slow: bool = True) -> Optional[np.ndarray]:
    """Return embedding vector. Uses daemon if available.
    If allow_slow=False and daemon unavailable, returns None instead of blocking for model load."""
    import time as _time
    t0 = _time.perf_counter()
    vec = _daemon_embed(text)
    if vec is not None:
        _record_embed_metric("embed_daemon_ms", (_time.perf_counter() - t0) * 1000)
        return vec
    if not allow_slow:
        return None
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    _record_embed_metric("embed_local_ms", (_time.perf_counter() - t0) * 1000)
    return vec


# Original reference for the embed_batch test seam — embed_batch compares the
# module global against this to detect a monkeypatched embed().
_EMBED_ORIGINAL = embed


def to_blob(vector: np.ndarray) -> bytes:
    """Convert numpy array to bytes for SQLite storage."""
    return vector.astype(np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Convert SQLite blob back to numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Assumes normalized vectors."""
    return float(np.dot(a, b))


def _load_vec(conn: sqlite3.Connection) -> bool:
    """Try to load sqlite-vec extension. Returns True if available."""
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except (ImportError, OSError):
        return False


def _recency_decay(updated_at_str: str) -> float:
    """Compute recency decay factor (0-1). Recent = 1, old = approaching 0."""
    from cairn.config import RECENCY_HALF_LIFE_DAYS
    try:
        updated = datetime.strptime(updated_at_str[:19], "%Y-%m-%d %H:%M:%S")
        age_days = (datetime.now() - updated).total_seconds() / 86400
        import math
        return math.exp(-0.693 * age_days / RECENCY_HALF_LIFE_DAYS)  # 0.693 = ln(2)
    except (ValueError, TypeError):
        return 0.5  # Unknown age, neutral


def _scope_weight(
    project: Optional[str],
    current_project: Optional[str],
    mem_type: Optional[str] = None,
) -> float:
    """Return scope weight: 1.0 for project match, 0.3 for global.

    Memory types in SCOPE_BIAS_EXEMPT_TYPES (e.g. person, preference) always
    receive full weight regardless of project — biographical/cross-cutting
    facts about the user apply universally, not just in the project where
    they were captured.
    """
    from cairn.config import SCOPE_BIAS_EXEMPT_TYPES
    if mem_type and mem_type in SCOPE_BIAS_EXEMPT_TYPES:
        return 1.0
    if current_project and project == current_project:
        return 1.0
    return 0.3


def extract_query_terms(text: str) -> set[str]:
    """Extract meaningful terms from query text for keyword matching.

    Delegates to `cairn.keywords.prompt_keywords` (YAKE phrase
    extraction + compound-identifier preservation). Signature
    preserved for callers; falls back internally if yake import fails.
    """
    from cairn.keywords import prompt_keywords
    return prompt_keywords(text)


def keyword_overlap(query_terms: set[str], keywords_csv: Optional[str]) -> float:
    """Compute overlap ratio between query terms and a memory's keywords (0.0-1.0)."""
    if not query_terms or not keywords_csv:
        return 0.0
    mem_keywords = {k.strip().lower() for k in keywords_csv.split(",") if k.strip()}
    if not mem_keywords:
        return 0.0
    matches = sum(1 for mk in mem_keywords if any(qt in mk or mk in qt for qt in query_terms))
    return matches / len(mem_keywords)


def composite_score(
    similarity: float,
    confidence: float,
    updated_at_str: Optional[str] = None,
    project: Optional[str] = None,
    current_project: Optional[str] = None,
    mem_type: Optional[str] = None,
    kw_overlap: float = 0.0,
) -> float:
    """Compute a single scalar retrieval score combining all signals.
    Reduces cognitive load on the LLM by pre-ranking results."""
    from cairn.config import SCORE_W_SIMILARITY, SCORE_W_CONFIDENCE, SCORE_W_RECENCY, SCORE_W_SCOPE, SCORE_W_KEYWORDS
    recency = _recency_decay(updated_at_str) if updated_at_str else 0.5
    scope = _scope_weight(project, current_project, mem_type) if project is not None else 0.5
    return (
        SCORE_W_SIMILARITY * similarity +
        SCORE_W_CONFIDENCE * (confidence ** 2) +
        SCORE_W_KEYWORDS * kw_overlap +
        SCORE_W_RECENCY * recency +
        SCORE_W_SCOPE * scope
    )


def _vec_candidates(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    k: int,
    current_project: Optional[str] = None,
    query_terms: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Get top-k candidates from sqlite-vec index."""
    query_blob = to_blob(query_vec)
    rows = conn.execute("""
        SELECT v.memory_id, v.distance, m.type, m.topic, m.content, m.updated_at, m.project, m.confidence, m.depth, m.archived_reason, m.session_id, m.keywords
        FROM memories_vec v
        JOIN memories m ON v.memory_id = m.id
        WHERE v.embedding MATCH ?
          AND k = ?
          AND m.deleted_at IS NULL
    """, (query_blob, k)).fetchall()

    results: list[dict[str, Any]] = []
    qt = query_terms or set()
    for row in rows:
        if row[9]:  # archived_reason — exclude from main search
            continue
        confidence = row[7] if row[7] is not None else 0.7
        l2_dist = row[1]
        sim = 1.0 - (l2_dist * l2_dist / 2.0)
        kw_ov = keyword_overlap(qt, row[11])
        results.append({
            "id": row[0],
            "type": row[2],
            "topic": row[3],
            "content": row[4],
            "similarity": sim,
            "updated_at": row[5],
            "project": row[6],
            "confidence": confidence,
            "session_id": row[10],
            "depth": row[8],
            "archived_reason": row[9],
            "keywords": row[11],
            "score": composite_score(sim, confidence, row[5], row[6], current_project, row[2], kw_ov)
        })
    return results


_BRUTE_FORCE_WARN_THRESHOLD = 10000

def _brute_force_candidates(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    k: int,
    current_project: Optional[str] = None,
    query_terms: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Get top-k candidates via brute-force scan.

    Dual-embedding (schema v8): scores row as max(cos(prompt, embedding),
    cos(prompt, topic_embedding)). Topic embedding lives in a prompt-shaped
    region of vector space — short queries match short topics far better
    than long content+kw mashup. Rows without topic_embedding (legacy,
    pre-backfill) fall back to content-only cosine.
    """
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, confidence, depth, archived_reason, session_id, keywords, topic_embedding "
        "FROM memories WHERE embedding IS NOT NULL AND (archived_reason IS NULL OR archived_reason = '') AND deleted_at IS NULL"
    ).fetchall()
    if len(rows) >= _BRUTE_FORCE_WARN_THRESHOLD:
        _log_embed(
            f"brute-force scan over {len(rows)} memories — consider enabling sqlite-vec index",
            "warning",
        )

    results: list[dict[str, Any]] = []
    qt = query_terms or set()
    for row in rows:
        confidence = row[7] if row[7] is not None else 0.7
        row_vec = from_blob(row[4])
        sim = float(cosine_similarity(query_vec, row_vec))
        if row[12] is not None:
            try:
                topic_vec = from_blob(row[12])
                topic_sim = float(cosine_similarity(query_vec, topic_vec))
                if topic_sim > sim:
                    sim = topic_sim
            except Exception:
                pass
        kw_ov = keyword_overlap(qt, row[11])
        results.append({
            "id": row[0],
            "type": row[1],
            "topic": row[2],
            "content": row[3],
            "similarity": sim,
            "updated_at": row[5],
            "project": row[6],
            "confidence": confidence,
            "session_id": row[10],
            "depth": row[8],
            "archived_reason": row[9],
            "keywords": row[11],
            "score": composite_score(sim, confidence, row[5], row[6], current_project, row[1], kw_ov)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


def _topic_candidates_brute(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    k: int,
    current_project: Optional[str] = None,
    query_terms: Optional[set[str]] = None,
    min_sim: float = 0.30,
) -> list[dict[str, Any]]:
    """Brute-force scan over topic_embedding (schema v8) — supplements vec-index
    content lookup to surface rows whose topic is a strong prompt match even
    when the content embedding is weak. Returns rows scoring >= min_sim on
    topic similarity.
    """
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, confidence, depth, archived_reason, session_id, keywords, topic_embedding "
        "FROM memories WHERE topic_embedding IS NOT NULL AND (archived_reason IS NULL OR archived_reason = '') AND deleted_at IS NULL"
    ).fetchall()
    results: list[dict[str, Any]] = []
    qt = query_terms or set()
    for row in rows:
        try:
            topic_vec = from_blob(row[12])
        except Exception:
            continue
        topic_sim = float(cosine_similarity(query_vec, topic_vec))
        if topic_sim < min_sim:
            continue
        confidence = row[7] if row[7] is not None else 0.7
        kw_ov = keyword_overlap(qt, row[11])
        results.append({
            "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
            "similarity": topic_sim,
            "updated_at": row[5], "project": row[6], "confidence": confidence,
            "session_id": row[10], "depth": row[8], "archived_reason": row[9],
            "keywords": row[11],
            "score": composite_score(topic_sim, confidence, row[5], row[6], current_project, row[1], kw_ov),
            "_topic_only": True,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


def upsert_vec_index(conn: sqlite3.Connection, memory_id: int, embedding_blob: bytes) -> None:
    """Insert or update a vector in the vec index."""
    try:
        conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
        conn.execute("INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                     (memory_id, embedding_blob))
    except Exception as e:
        _log_embed(f"vec index upsert failed for memory {memory_id}: {type(e).__name__}: {e}", "warning")


def find_similar(
    conn: sqlite3.Connection,
    text: str,
    threshold: Optional[float] = None,
    limit: Optional[int] = None,
    current_project: Optional[str] = None,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """Find memories similar to the given text with full quality filtering.

    Applies:
    - Soft confidence inclusion (high similarity overrides low confidence)
    - Relative filtering (drop entries far below the best match)
    - Dominance suppression (include runner-up if close to leader)
    - Garbage gate (don't return anything if best match is too weak)
    - Composite scoring (pre-ranked by similarity + confidence + recency + scope)

    Multi-query decomposition: if text contains a `|` separator, each part is
    treated as an independent subquery. Results from all subqueries are merged
    keeping the best score per memory. This is the right approach for
    multi-dimensional questions ("user's job AND brother") where a single
    embedding would blur multiple distinct concepts.

    Single-pass design: all query variants (base, unprefixed, type-prefix
    fan-out) are embedded in ONE daemon round-trip and scored against ONE
    fetch of the memory table via vectorized matrix products. The unprefixed
    variant is part of the fan-out set, so callers never need a second
    unprefixed pass. Archived rows ride the same fetch and surface as
    negative knowledge ("we tried X and abandoned it") — by design they are
    delivered, not suppressed.

    rerank=False skips the cross-encoder pass — used by latency-critical
    every-prompt callers (L1.5); quality-critical callers (L1/L3) keep it."""
    from cairn.config import (SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR, RELATIVE_FILTER_RATIO,
                        MIN_INJECTION_SIMILARITY, MAX_INJECTED_ENTRIES, DOMINANCE_EPSILON,
                        BORDERLINE_SIM_CEILING, BORDERLINE_SCORE_FLOOR)

    import time as _time
    import numpy as _np

    if threshold is None:
        threshold = 0.15  # Permissive floor — quality gates handle the rest
    if limit is None:
        limit = MAX_INJECTED_ENTRIES

    # Multi-query decomposition: split on | and merge results
    if "|" in text:
        subqueries = [s.strip() for s in text.split("|") if s.strip()]
        if len(subqueries) > 1:
            merged: dict[int, dict[str, Any]] = {}
            for sq in subqueries:
                sub_results = find_similar(conn, sq, threshold=threshold,
                                           limit=limit, current_project=current_project,
                                           rerank=rerank)
                for r in sub_results:
                    existing = merged.get(r["id"])
                    if not existing or r["score"] > existing["score"]:
                        merged[r["id"]] = r
            ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
            return ranked[:limit]

    from cairn.config import QUERY_EXPANSION_FANOUT

    # Extract query terms once for keyword overlap scoring across all candidates
    qt = extract_query_terms(text)

    # --- Query variants, embedded in ONE daemon round-trip ---
    # Memories are embedded as "{project} {type} {topic} {content}". The
    # variant set covers: project-prefixed base, bare base (replaces the old
    # second unprefixed find_similar pass in hybrid_search), and type-prefix
    # fan-out in both forms. Max-over-variants with per-variant z-score
    # normalization picks the best alignment per memory.
    _FANOUT_TYPES = ["fact", "decision", "correction", "skill", "preference", "project", "workflow"]
    variant_texts: list[str] = []
    if current_project:
        variant_texts.append(f"{current_project} {text}")
        variant_texts.append(text)
    else:
        variant_texts.append(text)
    if QUERY_EXPANSION_FANOUT:
        for mtype in _FANOUT_TYPES:
            if current_project:
                variant_texts.append(f"{current_project} {mtype} {text}")
            variant_texts.append(f"{mtype} {text}")
    # Topic embeddings live in a prompt-shaped region — they are scored against
    # the base variants only (prefixed + bare), not the type fan-out.
    n_base = 2 if current_project else 1

    # --- Candidate generation: daemon-resident search first ---
    # The long-lived daemon caches the normalized embedding matrices, so one
    # socket round-trip replaces the per-process blob fetch + matrix rebuild.
    # Skipped when embed() is monkeypatched (test seam) so mocked vectors are
    # honored; falls back to the local path when the daemon is down or stale.
    t_fanout = _time.perf_counter()
    k = limit * 5  # Over-fetch for post-filtering
    candidates: list[dict[str, Any]] = []
    archived_pool: list[dict[str, Any]] = []
    daemon_rows = None
    if embed is _EMBED_ORIGINAL and _conn_is_main_db(conn):
        floor = min(threshold, MIN_INJECTION_SIMILARITY)
        daemon_rows = _daemon_vector_search(variant_texts, n_base, floor, top_k=k * 6)

    if daemon_rows is not None:
        _record_embed_metric("search_daemon_ms", (_time.perf_counter() - t_fanout) * 1000)
        for r in daemon_rows:
            sim = r["similarity"]
            is_archived = bool(r.get("archived_reason"))
            confidence = r["confidence"] if r["confidence"] is not None else 0.7
            if is_archived:
                if sim < MIN_INJECTION_SIMILARITY:
                    continue
                archived_pool.append({
                    "id": r["id"], "type": r["type"], "topic": r["topic"], "content": r["content"],
                    "similarity": sim, "updated_at": r["updated_at"], "project": r["project"],
                    "confidence": confidence, "depth": r["depth"],
                    "score": 0.0,
                    "archived": True, "archived_reason": r["archived_reason"],
                })
                continue
            if sim < threshold:
                continue
            kw_ov = keyword_overlap(qt, r.get("keywords"))
            candidates.append({
                "id": r["id"], "type": r["type"], "topic": r["topic"], "content": r["content"],
                "similarity": sim, "updated_at": r["updated_at"], "project": r["project"],
                "confidence": confidence, "session_id": r["session_id"], "depth": r["depth"],
                "archived_reason": r["archived_reason"], "keywords": r.get("keywords"),
                "score": composite_score(sim, confidence, r["updated_at"], r["project"],
                                         current_project, r["type"], kw_ov),
            })
    else:
        vecs = embed_batch(variant_texts)
        if vecs is None:
            return []
        V = _np.stack(vecs).astype(_np.float32)
        v_norms = _np.linalg.norm(V, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1.0
        V = V / v_norms

        # --- Single fetch of the memory table (active + archived together) ---
        t_search = _time.perf_counter()
        rows = conn.execute(
            "SELECT id, type, topic, content, embedding, updated_at, project, confidence, "
            "depth, archived_reason, session_id, keywords, topic_embedding "
            "FROM memories WHERE embedding IS NOT NULL AND deleted_at IS NULL"
        ).fetchall()
        if not rows:
            return []

        dim = V.shape[1]
        mem_vecs: list[_np.ndarray] = []
        valid_rows: list[Any] = []
        for row in rows:
            try:
                v = _np.frombuffer(row[4], dtype=_np.float32)
            except (ValueError, TypeError):
                continue
            if v.shape[0] != dim:
                continue
            mem_vecs.append(v)
            valid_rows.append(row)
        if not valid_rows:
            return []

        M = _np.stack(mem_vecs)
        m_norms = _np.linalg.norm(M, axis=1, keepdims=True)
        m_norms[m_norms == 0] = 1.0
        M = M / m_norms

        # All-variant similarity in one matrix product: [N, n_variants]
        S = M @ V.T
        if S.shape[1] > 1:
            # Z-score normalize within each variant column, then take each row's
            # best raw similarity at its best-aligned variant. Without this,
            # variants producing systematically higher raw cosines would dominate.
            mu = S.mean(axis=0)
            sd = _np.maximum(S.std(axis=0), 1e-6)
            Z = (S - mu) / sd
            best_idx = Z.argmax(axis=1)
            best_sim = S[_np.arange(S.shape[0]), best_idx].astype(_np.float64)
        else:
            best_sim = S[:, 0].astype(_np.float64)

        # Dual-embedding supplement (schema v8): max in topic-embedding similarity
        # against the base variants for rows that have one.
        t_idx = [i for i, r in enumerate(valid_rows) if r[12] is not None]
        if t_idx:
            t_vecs: list[_np.ndarray] = []
            t_keep: list[int] = []
            for i in t_idx:
                try:
                    tv = _np.frombuffer(valid_rows[i][12], dtype=_np.float32)
                except (ValueError, TypeError):
                    continue
                if tv.shape[0] != dim:
                    continue
                t_vecs.append(tv)
                t_keep.append(i)
            if t_keep:
                T = _np.stack(t_vecs)
                t_norms = _np.linalg.norm(T, axis=1, keepdims=True)
                t_norms[t_norms == 0] = 1.0
                TS = (T / t_norms) @ V[:n_base].T
                t_best = TS.max(axis=1)
                keep_arr = _np.array(t_keep)
                best_sim[keep_arr] = _np.maximum(best_sim[keep_arr], t_best)

        _record_embed_metric("search_matrix_ms", (_time.perf_counter() - t_search) * 1000)

        for i, row in enumerate(valid_rows):
            sim = float(best_sim[i])
            is_archived = bool(row[9])
            if is_archived:
                if sim < MIN_INJECTION_SIMILARITY:
                    continue
            elif sim < threshold:
                continue
            confidence = row[7] if row[7] is not None else 0.7
            if is_archived:
                archived_pool.append({
                    "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
                    "similarity": sim, "updated_at": row[5], "project": row[6],
                    "confidence": confidence, "depth": row[8],
                    "score": 0.0,
                    "archived": True, "archived_reason": row[9],
                })
                continue
            kw_ov = keyword_overlap(qt, row[11])
            candidates.append({
                "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
                "similarity": sim, "updated_at": row[5], "project": row[6],
                "confidence": confidence, "session_id": row[10], "depth": row[8],
                "archived_reason": row[9], "keywords": row[11],
                "score": composite_score(sim, confidence, row[5], row[6], current_project, row[1], kw_ov),
            })
    _record_embed_metric("fanout_ms", (_time.perf_counter() - t_fanout) * 1000)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:k * 2]

    # Filter by similarity threshold only — confidence no longer gates retrieval
    filtered = [r for r in candidates if r["similarity"] >= threshold]

    if not filtered:
        return []

    # Sort by composite score
    filtered.sort(key=lambda x: x["score"], reverse=True)

    # Garbage gate: if the best match is too weak, return nothing
    if filtered[0]["similarity"] < MIN_INJECTION_SIMILARITY:
        return []

    # Borderline gate: weak similarity + low score → skip
    if (filtered[0]["similarity"] < BORDERLINE_SIM_CEILING
            and filtered[0]["score"] < BORDERLINE_SCORE_FLOOR):
        return []

    # Relative filtering: drop entries far below the best match
    max_sim = filtered[0]["similarity"]
    filtered = [r for r in filtered if r["similarity"] >= RELATIVE_FILTER_RATIO * max_sim]

    # Dominance suppression: if top two are close, ensure both are included
    if len(filtered) >= 2:
        gap = filtered[0]["score"] - filtered[1]["score"]
        if gap < DOMINANCE_EPSILON:
            # Both are close — keep at least 2
            limit = max(limit, 2)

    # Diversity filter: greedily drop near-duplicates from results
    from cairn.config import DIVERSITY_SIM_THRESHOLD
    diverse: list[dict[str, Any]] = []
    for r in filtered:
        is_dup = False
        for selected in diverse:
            # Same type+topic = duplicate
            if r.get("topic") == selected.get("topic") and r.get("type") == selected.get("type"):
                is_dup = True
                break
            # High content word overlap = likely duplicate
            r_words = set(r.get("content", "").lower().split())
            s_words = set(selected.get("content", "").lower().split())
            if r_words and s_words:
                overlap = len(r_words & s_words) / max(len(r_words | s_words), 1)
                if overlap > DIVERSITY_SIM_THRESHOLD:
                    is_dup = True
                    break
        if not is_dup:
            diverse.append(r)

    # Archived candidates (negative knowledge) gated relative to the best
    # active match — same rules as before, sourced from the single fetch.
    best_active_sim = diverse[0]["similarity"] if diverse else 0.0
    active_ids = {r["id"] for r in diverse}
    archived_candidates = [
        r for r in archived_pool
        if r["id"] not in active_ids
        and r["similarity"] >= RELATIVE_FILTER_RATIO * best_active_sim
    ]
    archived_candidates.sort(key=lambda x: x["similarity"], reverse=True)

    # Cross-encoder re-ranking: ONE combined daemon call scores active and
    # archived candidates jointly — half the round-trips of the previous
    # two-call design at the same quality. Caps keep CE latency bounded
    # (cost is linear in pair count).
    from cairn.config import (CROSS_ENCODER_ENABLED, CROSS_ENCODER_MIN_CANDIDATES,
                              CROSS_ENCODER_WEIGHT, CROSS_ENCODER_SCORE_FLOOR,
                              CROSS_ENCODER_MAX_CANDIDATES, CROSS_ENCODER_MAX_ARCHIVED)
    ce_active = CROSS_ENCODER_ENABLED and rerank and len(diverse) >= CROSS_ENCODER_MIN_CANDIDATES
    ce_archived = CROSS_ENCODER_ENABLED and rerank and bool(archived_candidates)
    if ce_active or ce_archived:
        t_rerank = _time.perf_counter()
        if ce_active:
            diverse = diverse[:CROSS_ENCODER_MAX_CANDIDATES]
        archived_candidates = archived_candidates[:CROSS_ENCODER_MAX_ARCHIVED]
        active_for_ce = diverse if ce_active else []
        ce_pool = active_for_ce + (archived_candidates if ce_archived else [])
        candidate_texts = [f"{r.get('type', '')} {r.get('topic', '')}: {r.get('content', '')}" for r in ce_pool]
        ce_out = _daemon_rerank(text, candidate_texts)
        ce_scores, ce_floor, ce_model = ce_out if ce_out else (None, None, None)
        floor = ce_floor if ce_floor is not None else CROSS_ENCODER_SCORE_FLOOR
        if ce_scores and len(ce_scores) == len(ce_pool):
            for i, r in enumerate(ce_pool):
                r["ce_score"] = ce_scores[i]
                r["reranker_model"] = ce_model  # provenance for memory_deliveries
            if ce_active:
                pre_filter = len(diverse)
                above_floor = [r for r in diverse if r["ce_score"] >= floor]
                diverse = above_floor if above_floor else diverse[:1]
                ce_min = min(r["ce_score"] for r in diverse) if diverse else 0
                ce_max = max(r["ce_score"] for r in diverse) if diverse else 1
                ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                for r in diverse:
                    ce_norm = (r["ce_score"] - ce_min) / ce_range
                    r["score"] = (1 - CROSS_ENCODER_WEIGHT) * r["score"] + CROSS_ENCODER_WEIGHT * ce_norm
                diverse.sort(key=lambda x: x["score"], reverse=True)
                _record_embed_metric("rerank_filtered", pre_filter - len(diverse))
            if ce_archived:
                archived_candidates = [r for r in archived_candidates
                                       if r["ce_score"] >= floor]
        _record_embed_metric("rerank_ms", (_time.perf_counter() - t_rerank) * 1000)

    results = diverse[:limit]

    # Archived memory enrichment: related negative knowledge rides along for
    # the learning trail — deliberately delivered, never suppressed.
    if results and archived_candidates:
        results.extend(archived_candidates[:limit])

    return results

def find_nearest(conn: sqlite3.Connection, text: str, limit: int = 1) -> list[dict[str, Any]]:
    """Find the single nearest memory by raw similarity. Used for deduplication.

    Dual-embedding (schema v8): score row as max(cos(query, embedding),
    cos(query, topic_embedding)). Without this, dedup is blind on the
    topic axis — a new memory whose topic vector closely matches an
    existing row but whose content diverges would slip past the 0.85
    floor and be inserted as a duplicate. Mirrors find_similar''s
    read-side dual-embedding behaviour.

    Uses vec index if available, supplemented by a brute-force topic
    scan so topic-only matches surface as candidates. No confidence or
    threshold filtering.
    """
    # Daemon-resident fast path (same matrices as find_similar) — one socket
    # round-trip replaces the per-process scan. ~118ms -> ~10ms per entry,
    # which the stop hook pays once per stored memory.
    if embed is _EMBED_ORIGINAL and _conn_is_main_db(conn):
        rows = _daemon_vector_search([text], 1, 0.0, top_k=max(limit * 5, 25))
        if rows is not None:
            return [{
                "id": r["id"], "type": r["type"], "topic": r["topic"],
                "content": r["content"], "similarity": r["similarity"],
                "updated_at": r["updated_at"], "project": r["project"],
                "confidence": r["confidence"] if r["confidence"] is not None else 0.7,
            } for r in rows[:limit]]

    query_vec = embed(text)

    vec_candidates: list[dict[str, Any]] = []
    if _load_vec(conn):
        try:
            vec_candidates = _vec_candidates(conn, query_vec, limit * 5)
        except Exception as e:
            _log_embed(f"vec nearest search failed, falling back to brute-force: {type(e).__name__}: {e}", "warning")

    # Topic supplement: scan topic_embedding column and merge by id (max sim).
    # On the vec path, this surfaces topic-only candidates that the content
    # index missed. On the brute fallback, we fold this into the loop below.
    if vec_candidates:
        try:
            topic_rows = conn.execute(
                "SELECT id, topic_embedding FROM memories "
                "WHERE topic_embedding IS NOT NULL AND deleted_at IS NULL"
            ).fetchall()
        except Exception:
            topic_rows = []
        by_id = {c["id"]: c for c in vec_candidates}
        for rid, blob in topic_rows:
            try:
                topic_vec = from_blob(blob)
            except Exception:
                continue
            t_sim = float(cosine_similarity(query_vec, topic_vec))
            existing = by_id.get(rid)
            if existing is not None and t_sim > existing["similarity"]:
                existing["similarity"] = t_sim
            elif existing is None and t_sim >= 0.30:
                # Fetch full row metadata for topic-only matches above a coarse floor.
                # 0.30 chosen as the lowest plausible dedup-relevant signal; callers
                # apply their own DEDUP_THRESHOLD on top.
                row = conn.execute(
                    "SELECT id, type, topic, content, updated_at, project, confidence "
                    "FROM memories WHERE id = ?", (rid,)
                ).fetchone()
                if row:
                    by_id[rid] = {
                        "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
                        "similarity": t_sim, "updated_at": row[4], "project": row[5],
                        "confidence": row[6] if row[6] is not None else 0.7,
                    }
        merged = list(by_id.values())
        merged.sort(key=lambda x: x["similarity"], reverse=True)
        return merged[:limit]

    # Brute-force fallback — dual-score inline.
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, confidence, topic_embedding "
        "FROM memories WHERE embedding IS NOT NULL AND deleted_at IS NULL"
    ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        row_vec = from_blob(row[4])
        sim = float(cosine_similarity(query_vec, row_vec))
        if row[8] is not None:
            try:
                topic_vec = from_blob(row[8])
                t_sim = float(cosine_similarity(query_vec, topic_vec))
                if t_sim > sim:
                    sim = t_sim
            except Exception:
                pass
        results.append({
            "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
            "similarity": sim, "updated_at": row[5], "project": row[6],
            "confidence": row[7] if row[7] is not None else 0.7
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]
