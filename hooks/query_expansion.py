"""Query expansion strategies for Cairn retrieval.

Mechanistic approaches to improve retrieval by mutating queries into
semantically close variants that better match stored memory embeddings.

Strategies:
1. Type-prefix fan-out — search with each memory type prefix, take max similarity
2. Corpus-aware PRF — use FTS hits to expand the query with corpus-specific terms
3. Nearest-neighbor blending — blend query vector with top-K neighbor average
"""

from __future__ import annotations

import re
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
from typing import Any, Optional

import numpy as np


MEMORY_TYPES = ["fact", "decision", "correction", "skill", "preference", "project", "workflow"]


def combined_expansion(
    conn: sqlite3.Connection,
    query: str,
    embed_fn,
    current_project: Optional[str] = None,
    limit: int = 10,
    blend_k: int = 3,
    query_weight: float = 0.7,
) -> list[dict[str, Any]]:
    """Combined strategy: type-prefix fan-out + neighbor blend, deduped by max score.

    Runs both strategies independently, then merges by keeping the highest score
    per memory ID. Explores two orthogonal dimensions:
    - Fan-out: type-prefix space (matches how memories are embedded)
    - Blend: local neighborhood (pulls toward relevant cluster)
    """
    fanout_results = type_prefix_fanout(conn, query, embed_fn,
                                         current_project=current_project, limit=limit * 2)
    blend_results = neighbor_blend(conn, query, embed_fn,
                                    current_project=current_project, limit=limit * 2,
                                    blend_k=blend_k, query_weight=query_weight)

    # RRF-style rank fusion — same principle as the FTS+semantic fusion in retrieval.py
    # Each method contributes 1/(k + rank) per memory it found.
    k = 60  # Standard RRF constant

    fanout_ranked = {r["id"]: (i, r) for i, r in enumerate(fanout_results)}
    blend_ranked = {r["id"]: (i, r) for i, r in enumerate(blend_results)}

    all_ids = set(fanout_ranked.keys()) | set(blend_ranked.keys())
    fused: list[dict[str, Any]] = []

    for mid in all_ids:
        rrf_score = 0.0
        result_dict = None

        if mid in fanout_ranked:
            rank, r = fanout_ranked[mid]
            rrf_score += 1.0 / (k + rank)
            result_dict = r

        if mid in blend_ranked:
            rank, r = blend_ranked[mid]
            rrf_score += 1.0 / (k + rank)
            if result_dict is None or r["score"] > result_dict["score"]:
                result_dict = r

        if result_dict is not None:
            # Boost the composite score by normalised RRF contribution
            max_rrf = 2.0 / (k + 1)  # Both methods, rank 0
            rrf_boost = rrf_score * (0.20 / max_rrf)
            result_dict = dict(result_dict)  # Copy to avoid mutating originals
            result_dict["score"] = result_dict["score"] + rrf_boost
            fused.append(result_dict)

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:limit]

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


def type_prefix_fanout(
    conn: sqlite3.Connection,
    query: str,
    embed_fn,
    current_project: Optional[str] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Strategy 1: Search with each type prefix, merge by max similarity per memory.

    Embeddings are stored as "{project} {type} {topic} {content}". A bare query
    misses the type prefix, reducing similarity. Fan out across all types and
    take the best similarity per memory.
    """
    from cairn.embeddings import composite_score, from_blob, cosine_similarity

    # Fetch all embeddings once
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, "
        "confidence, session_id, depth, archived_reason "
        "FROM memories WHERE embedding IS NOT NULL"
    ).fetchall()

    if not rows:
        return []

    # Build prefixed queries
    prefixes = []
    if current_project:
        for mtype in MEMORY_TYPES:
            prefixes.append(f"{current_project} {mtype} {query}")
        # Also try bare project prefix (no type)
        prefixes.append(f"{current_project} {query}")
    else:
        for mtype in MEMORY_TYPES:
            prefixes.append(f"{mtype} {query}")
        prefixes.append(query)

    # Embed all prefixed queries
    query_vecs = [embed_fn(p) for p in prefixes]

    # Score each memory against all prefixed queries, keep max
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        mem_id = row[0]
        mem_vec = from_blob(row[4])
        confidence = row[7] if row[7] is not None else 0.7

        max_sim = 0.0
        for qvec in query_vecs:
            sim = float(cosine_similarity(qvec, mem_vec))
            if sim > max_sim:
                max_sim = sim

        score = composite_score(max_sim, confidence, row[5], row[6], current_project)

        if mem_id not in best or score > best[mem_id]["score"]:
            best[mem_id] = {
                "id": mem_id, "type": row[1], "topic": row[2], "content": row[3],
                "updated_at": row[5], "project": row[6], "session_id": row[8],
                "confidence": confidence, "depth": row[9],
                "archived_reason": row[10],
                "similarity": max_sim, "score": score,
            }

    results = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return results[:limit]


def corpus_prf(
    conn: sqlite3.Connection,
    query: str,
    embed_fn,
    current_project: Optional[str] = None,
    limit: int = 10,
    prf_top_k: int = 3,
) -> list[dict[str, Any]]:
    """Strategy 2: Corpus-aware pseudo-relevance feedback.

    1. Run FTS5 on the original query to find keyword matches
    2. Extract distinctive terms from top-K FTS hits
    3. Augment the query with those terms
    4. Re-embed and search semantically with the expanded query
    """
    from cairn.embeddings import find_similar

    # Step 1: FTS keyword search
    words = re.findall(r'\w+', query.lower())
    meaningful = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
    if not meaningful:
        meaningful = words
    fts_query = " OR ".join(f'"{w}"' for w in meaningful) if meaningful else query

    try:
        fts_rows = conn.execute("""
            SELECT m.topic, m.content
            FROM memories_fts f JOIN memories m ON f.rowid = m.id
            WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?
        """, (fts_query, prf_top_k)).fetchall()
    except Exception:
        fts_rows = []

    if not fts_rows:
        # Fallback to standard search
        return find_similar(conn, query, limit=limit, current_project=current_project)

    # Step 2: Extract distinctive terms from FTS hits
    expansion_words: list[str] = []
    query_words = set(meaningful)
    for topic, content in fts_rows:
        for word in re.findall(r'\w+', f"{topic} {content}".lower()):
            if len(word) > 3 and word not in _STOPWORDS and word not in query_words:
                expansion_words.append(word)

    # Deduplicate, take top-5 by frequency
    from collections import Counter
    term_counts = Counter(expansion_words)
    top_terms = [t for t, _ in term_counts.most_common(5)]

    # Step 3: Augment query
    expanded = query + " " + " ".join(top_terms)

    # Step 4: Search with expanded query
    return find_similar(conn, expanded, limit=limit, current_project=current_project)


def neighbor_blend(
    conn: sqlite3.Connection,
    query: str,
    embed_fn,
    current_project: Optional[str] = None,
    limit: int = 10,
    blend_k: int = 3,
    query_weight: float = 0.7,
) -> list[dict[str, Any]]:
    """Strategy 3: Nearest-neighbor vector blending.

    1. Embed the query
    2. Find top-K nearest memory vectors
    3. Blend: new_vec = query_weight * query_vec + (1 - query_weight) * avg(neighbor_vecs)
    4. Re-search with the blended vector
    """
    from cairn.embeddings import composite_score, from_blob, cosine_similarity

    query_text = f"{current_project} {query}" if current_project else query
    query_vec = embed_fn(query_text)

    # Fetch all embeddings
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, "
        "confidence, session_id, depth, archived_reason "
        "FROM memories WHERE embedding IS NOT NULL"
    ).fetchall()

    if not rows:
        return []

    # Find top-K neighbors by similarity
    scored = []
    for row in rows:
        mem_vec = from_blob(row[4])
        sim = float(cosine_similarity(query_vec, mem_vec))
        scored.append((sim, mem_vec, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Blend with top-K neighbors
    top_k = scored[:blend_k]
    if top_k:
        neighbor_vecs = np.array([s[1] for s in top_k])
        avg_neighbor = neighbor_vecs.mean(axis=0)
        avg_neighbor = avg_neighbor / np.linalg.norm(avg_neighbor)
        blended = query_weight * query_vec + (1 - query_weight) * avg_neighbor
        blended = blended / np.linalg.norm(blended)
    else:
        blended = query_vec

    # Re-score all memories against blended vector
    results: list[dict[str, Any]] = []
    for row in rows:
        mem_vec = from_blob(row[4])
        sim = float(cosine_similarity(blended, mem_vec))
        confidence = row[7] if row[7] is not None else 0.7
        score = composite_score(sim, confidence, row[5], row[6], current_project)
        results.append({
            "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
            "updated_at": row[5], "project": row[6], "session_id": row[8],
            "confidence": confidence, "depth": row[9],
            "archived_reason": row[10],
            "similarity": sim, "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]
