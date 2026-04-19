"""Query expansion strategies for Cairn retrieval.

Type-prefix fan-out — search with each memory type prefix, take max similarity.
"""

from __future__ import annotations

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
from typing import Any, Optional


MEMORY_TYPES = ["fact", "decision", "correction", "skill", "preference", "project", "workflow"]




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
    from cairn.embeddings import composite_score, from_blob, cosine_similarity, extract_query_terms, keyword_overlap

    qt = extract_query_terms(query)

    # Fetch all embeddings once
    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, "
        "confidence, session_id, depth, archived_reason, keywords "
        "FROM memories WHERE embedding IS NOT NULL AND deleted_at IS NULL"
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

        kw_ov = keyword_overlap(qt, row[11])
        score = composite_score(max_sim, confidence, row[5], row[6], current_project, kw_overlap=kw_ov)

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


