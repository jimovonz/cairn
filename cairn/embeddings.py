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


def _record_embed_metric(event: str, value: float) -> None:
    """Record an embedding performance metric to the DB. Best-effort, never raises."""
    global _metrics_conn
    try:
        if _metrics_conn is None:
            db_path = os.path.join(os.path.dirname(__file__), "cairn.db")
            if not os.path.exists(db_path):
                return
            _metrics_conn = sqlite3.connect(db_path)
            _metrics_conn.execute("PRAGMA busy_timeout=2000")
        _metrics_conn.execute(
            "INSERT INTO metrics (event, value) VALUES (?, ?)",
            (event, value)
        )
        _metrics_conn.commit()
    except Exception:
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
            return np.frombuffer(bytes.fromhex(resp["vector"]), dtype=np.float32)
    except (ConnectionError, TimeoutError, OSError, ValueError):
        pass
    except Exception as e:
        # Unexpected error — log for debugging but don't crash
        import sys
        print(f"[cairn] daemon embed error: {type(e).__name__}: {e}", file=sys.stderr)
    return None


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
    """Extract meaningful terms from query text for keyword matching."""
    import re
    _STOPWORDS = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "its", "was", "are", "be",
        "has", "had", "have", "do", "did", "does", "will", "can", "could",
        "would", "should", "may", "might", "not", "no", "what", "when",
        "where", "who", "how", "why", "that", "this", "these", "those",
        "my", "your", "his", "her", "our", "their", "me", "you", "him",
        "them", "about", "into", "over", "after", "before", "between",
        "other", "some", "any", "all", "just", "also", "than", "then",
        "very", "too", "here", "there", "been", "being", "were",
    })
    words = re.findall(r'\w+', text.lower())
    meaningful = {w for w in words if len(w) > 2 and w not in _STOPWORDS}
    return meaningful if meaningful else {w for w in words if len(w) > 2}


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


def _brute_force_candidates(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    k: int,
    current_project: Optional[str] = None,
    query_terms: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Get top-k candidates via brute-force scan."""
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, confidence, depth, archived_reason, session_id, keywords "
        "FROM memories WHERE embedding IS NOT NULL AND (archived_reason IS NULL OR archived_reason = '')"
    ).fetchall()

    results: list[dict[str, Any]] = []
    qt = query_terms or set()
    for row in rows:
        confidence = row[7] if row[7] is not None else 0.7
        row_vec = from_blob(row[4])
        sim = cosine_similarity(query_vec, row_vec)
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


def upsert_vec_index(conn: sqlite3.Connection, memory_id: int, embedding_blob: bytes) -> None:
    """Insert or update a vector in the vec index."""
    try:
        conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
        conn.execute("INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                     (memory_id, embedding_blob))
    except Exception:
        pass  # Vec table may not exist


def find_similar(
    conn: sqlite3.Connection,
    text: str,
    threshold: Optional[float] = None,
    limit: Optional[int] = None,
    current_project: Optional[str] = None,
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

    Uses sqlite-vec index if available, falls back to brute-force."""
    from cairn.config import (SOFT_SIM_OVERRIDE, SOFT_CONF_FLOOR, RELATIVE_FILTER_RATIO,
                        MIN_INJECTION_SIMILARITY, MAX_INJECTED_ENTRIES, DOMINANCE_EPSILON,
                        BORDERLINE_SIM_CEILING, BORDERLINE_SCORE_FLOOR)

    import time as _time

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
                                           limit=limit, current_project=current_project)
                for r in sub_results:
                    existing = merged.get(r["id"])
                    if not existing or r["score"] > existing["score"]:
                        merged[r["id"]] = r
            ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
            return ranked[:limit]

    from cairn.config import QUERY_EXPANSION_FANOUT

    # Extract query terms once for keyword overlap scoring across all candidates
    qt = extract_query_terms(text)

    # Prefix query with project to match how stored embeddings are augmented
    query_text = f"{current_project} {text}" if current_project else text
    query_vec = embed(query_text)
    k = limit * 5  # Over-fetch for post-filtering

    t_search = _time.perf_counter()
    candidates: list[dict[str, Any]] = []
    search_method = "brute"
    if _load_vec(conn):
        try:
            candidates = _vec_candidates(conn, query_vec, k, current_project, qt)
            search_method = "vec"
        except Exception:
            pass

    if not candidates:
        candidates = _brute_force_candidates(conn, query_vec, k, current_project, qt)
        search_method = "brute"
    _record_embed_metric(f"search_{search_method}_ms", (_time.perf_counter() - t_search) * 1000)

    # Type-prefix fan-out: search with each memory type prefix, keep max similarity per memory.
    # Memories are embedded as "{project} {type} {topic} {content}" — a bare query misses the
    # type prefix. Fan-out closes this gap with ~7x more dot products (no model inference).
    if QUERY_EXPANSION_FANOUT and candidates:
        t_fanout = _time.perf_counter()
        _FANOUT_TYPES = ["fact", "decision", "correction", "skill", "preference", "project", "workflow"]
        fanout_vecs = []
        for mtype in _FANOUT_TYPES:
            ft = f"{current_project} {mtype} {text}" if current_project else f"{mtype} {text}"
            fanout_vecs.append(embed(ft))

        # Fetch all embeddings once for fan-out scoring
        all_rows = conn.execute(
            "SELECT id, embedding, confidence, updated_at, project FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()
        fanout_best: dict[int, float] = {}
        for row in all_rows:
            mem_vec = from_blob(row[1])
            max_sim = float(cosine_similarity(query_vec, mem_vec))
            for fvec in fanout_vecs:
                sim = float(cosine_similarity(fvec, mem_vec))
                if sim > max_sim:
                    max_sim = sim
            fanout_best[row[0]] = max_sim

        # Update candidates with fan-out similarities where they improved
        for c in candidates:
            mid = c["id"]
            if mid in fanout_best and fanout_best[mid] > c["similarity"]:
                c["similarity"] = fanout_best[mid]
                kw_ov = keyword_overlap(qt, c.get("keywords"))
                c["score"] = composite_score(
                    fanout_best[mid], c["confidence"],
                    c.get("updated_at"), c.get("project"), current_project, c.get("type"), kw_ov
                )
        _record_embed_metric("fanout_ms", (_time.perf_counter() - t_fanout) * 1000)

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

    results = diverse[:limit]

    # Archived memory enrichment: find related archived memories for learning trail
    if results:
        active_ids = {r["id"] for r in results}
        best_active_sim = results[0]["similarity"] if results else 0
        archived_candidates: list[dict[str, Any]] = []
        try:
            archived = conn.execute(
                "SELECT id, type, topic, content, embedding, updated_at, project, confidence, "
                "depth, archived_reason "
                "FROM memories WHERE archived_reason IS NOT NULL AND embedding IS NOT NULL"
            ).fetchall()
            for row in archived:
                if row[0] in active_ids:
                    continue
                row_vec = from_blob(row[4])
                sim = cosine_similarity(query_vec, row_vec)
                if sim >= RELATIVE_FILTER_RATIO * best_active_sim and sim >= MIN_INJECTION_SIMILARITY:
                    archived_candidates.append({
                        "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
                        "similarity": sim, "updated_at": row[5], "project": row[6],
                        "confidence": row[7], "depth": row[8],
                        "score": 0.0,
                        "archived": True, "archived_reason": row[9]
                    })
        except sqlite3.OperationalError:
            pass  # archived_reason column not yet added
        archived_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        results.extend(archived_candidates[:limit])

    return results


def find_nearest(conn: sqlite3.Connection, text: str, limit: int = 1) -> list[dict[str, Any]]:
    """Find the single nearest memory by raw similarity. Used for deduplication.
    Uses vec index if available. No confidence or threshold filtering."""
    query_vec = embed(text)

    if _load_vec(conn):
        try:
            candidates = _vec_candidates(conn, query_vec, limit)
            if candidates:
                candidates.sort(key=lambda x: x["similarity"], reverse=True)
                return candidates[:limit]
        except Exception:
            pass

    # Brute-force fallback
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, confidence FROM memories WHERE embedding IS NOT NULL"
    ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        row_vec = from_blob(row[4])
        sim = cosine_similarity(query_vec, row_vec)
        results.append({
            "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
            "similarity": sim, "updated_at": row[5], "project": row[6],
            "confidence": row[7] if row[7] is not None else 0.7
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]
