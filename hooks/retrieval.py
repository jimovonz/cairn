"""Context retrieval — Layer 2 (cross-project) and Layer 3 (on-demand pull)."""

from __future__ import annotations

import json
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
from typing import Any, Optional

import hooks.hook_helpers as hook_helpers
from hooks.hook_helpers import (
    log, get_conn, get_ephemeral_conn, get_session_project, record_metric,
    recency_days as _recency_days, reliability_label, record_layer_delivery,
    format_entry as _shared_format_entry, build_context_xml, save_hook_state,
)
import re

from cairn.config import (L3_PROJECT_SIM_THRESHOLD, L3_GLOBAL_SIM_WITH_PROJECT,
                     L3_GLOBAL_SIM_WITHOUT_PROJECT, L3_PROJECT_QUALITY_FLOOR,
                     L3_MAX_PROJECT_RESULTS, L3_MAX_GLOBAL_RESULTS,
                     WEAK_ENTRY_SCORE_FLOOR, RRF_K, GLOBAL_HARD_FLOOR)


CONTEXT_CACHE_SIM_THRESHOLD: float = 0.9


def hybrid_search(
    query: str,
    conn,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    threshold: float = 0.30,
    limit: int = 10,
    use_adaptive: bool = True,
    exclude_project: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Hybrid semantic + FTS5 search with RRF fusion.

    Returns (project_results, global_results, threshold_boost).
    Used by all retrieval layers for consistent search quality.
    """
    from cairn.config import (
        MIN_INJECTION_SIMILARITY, L3_PROJECT_QUALITY_FLOOR,
        SCOPE_BIAS_EXEMPT_TYPES, WEAK_ENTRY_SCORE_FLOOR,
    )
    from cairn.embeddings import extract_query_terms, keyword_overlap as _kw_overlap

    emb = hook_helpers.get_embedder()
    semantic_ranked: dict[int, tuple[int, dict[str, Any]]] = {}
    fts_ranked: dict[int, tuple[int, dict[str, Any]]] = {}

    threshold_boost = get_adaptive_threshold_boost() if use_adaptive else 0.0
    effective_threshold = threshold + threshold_boost

    # --- Semantic search ---
    if emb:
        try:
            all_results = emb.find_similar(conn, query, threshold=threshold,
                                           limit=limit * 2, current_project=project)
            if project and not exclude_project:
                unprefixed = emb.find_similar(conn, query, threshold=threshold,
                                              limit=limit * 2, current_project=None)
                by_id = {r["id"]: r for r in all_results}
                for r in unprefixed:
                    if r["id"] not in by_id or r["score"] > by_id[r["id"]]["score"]:
                        by_id[r["id"]] = r
                all_results = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)

            rank = 0
            for r in all_results:
                if session_id and r.get("session_id") == session_id:
                    continue
                if exclude_project and project and r.get("project") == project:
                    continue
                semantic_ranked[r["id"]] = (rank, r)
                rank += 1
        except Exception as e:
            log(f"Hybrid search semantic error: {e}")

    # --- FTS5 keyword search ---
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
    _query_terms = extract_query_terms(query)

    try:
        words = re.findall(r'\w+', query.lower())
        meaningful = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
        if not meaningful:
            meaningful = [w for w in words if len(w) > 2]
        if not meaningful:
            meaningful = words
        fts_query = " OR ".join(f'"{w}"' for w in meaningful) if meaningful else query
        rows = conn.execute("""
            SELECT m.id, m.type, m.topic, m.content, m.updated_at, m.project,
                   m.session_id, m.confidence, m.depth, m.archived_reason, rank, m.keywords
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.id
            WHERE memories_fts MATCH ? AND m.deleted_at IS NULL
            ORDER BY rank LIMIT ?
        """, (fts_query, limit * 3)).fetchall()
        rank = 0
        for r in rows:
            if session_id and r[6] == session_id:
                continue
            if exclude_project and project and r[5] == project:
                continue
            confidence = r[7] if r[7] is not None else 0.7
            entry: dict[str, Any] = {
                "id": r[0], "type": r[1], "topic": r[2], "content": r[3],
                "updated_at": r[4], "project": r[5], "session_id": r[6],
                "confidence": confidence, "depth": r[8],
                "archived_reason": r[9], "bm25_rank": -r[10],
                "keywords": r[11],
            }
            fts_ranked[r[0]] = (rank, entry)
            rank += 1
    except Exception as e:
        log(f"Hybrid search FTS error: {e}")

    # --- RRF Fusion ---
    all_ids = set(semantic_ranked.keys()) | set(fts_ranked.keys())
    fused: list[dict[str, Any]] = []

    for mid in all_ids:
        rrf_score = 0.0
        result_dict: Optional[dict[str, Any]] = None

        if mid in semantic_ranked:
            sem_rank, sem_result = semantic_ranked[mid]
            rrf_score += 1.0 / (RRF_K + sem_rank)
            result_dict = sem_result

        if mid in fts_ranked:
            fts_rank_pos, fts_result = fts_ranked[mid]
            rrf_score += 1.0 / (RRF_K + fts_rank_pos)
            if result_dict is None:
                from cairn.embeddings import composite_score as _cscore
                fts_sim = 0.35
                fts_result["similarity"] = fts_sim
                fts_kw_ov = _kw_overlap(_query_terms, fts_result.get("keywords"))
                fts_result["score"] = _cscore(
                    fts_sim, fts_result["confidence"],
                    fts_result.get("updated_at"), fts_result.get("project"), project,
                    kw_overlap=fts_kw_ov
                )
                result_dict = fts_result

        if result_dict is None:
            continue

        rrf_boost = rrf_score * (0.20 / (2.0 / (RRF_K + 1)))
        result_dict["rrf_score"] = rrf_score
        result_dict["score"] = result_dict.get("score", 0.30) + rrf_boost

        in_sem = mid in semantic_ranked
        in_fts = mid in fts_ranked
        result_dict["_source"] = "both" if (in_sem and in_fts) else ("semantic" if in_sem else "fts")

        fused.append(result_dict)

    fused.sort(key=lambda x: x["score"], reverse=True)

    # --- Split by scope ---
    project_results: list[dict[str, Any]] = []
    global_results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for r in fused:
        mid = r["id"]
        if mid in seen_ids:
            continue
        sim = r.get("similarity", 0)
        if project and r.get("project") == project and sim >= effective_threshold:
            project_results.append(r)
            seen_ids.add(mid)

    global_threshold = effective_threshold
    for r in fused:
        mid = r["id"]
        if mid in seen_ids:
            continue
        sim = r.get("similarity", 0)
        is_exempt = r.get("type") in SCOPE_BIAS_EXEMPT_TYPES
        eff_t = effective_threshold if is_exempt else global_threshold
        above_hard_floor = sim >= GLOBAL_HARD_FLOOR
        if above_hard_floor or sim >= eff_t or (r.get("_source") == "fts" and r["score"] >= WEAK_ENTRY_SCORE_FLOOR):
            global_results.append(r)
            seen_ids.add(mid)

    project_results = project_results[:limit]
    global_results = global_results[:limit]

    both_count = sum(1 for r in fused if r.get("_source") == "both")
    sem_only = sum(1 for r in fused if r.get("_source") == "semantic")
    fts_only = sum(1 for r in fused if r.get("_source") == "fts")
    log(f"Hybrid search: {len(fused)} candidates ({both_count} both, {sem_only} sem, {fts_only} fts) "
        f"→ {len(project_results)} project + {len(global_results)} global")

    return project_results, global_results, threshold_boost


def _is_thin_retrieval(entries: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Detect whether retrieved entries are too thin to be a useful answer.

    Two principled signals (no hand-curated meta-pattern lists):

    1. Score-based: max similarity below THIN_TOP_SIM threshold means the
       embedding model didn't find a strong match for the query.
    2. Count-based: fewer than THIN_MIN_ENTRIES entries means there's not
       enough material to be confident.

    A thin flag triggers stop-hook escalation on the next turn — forces the
    LLM to run query.py directly or re-declare context: insufficient with a
    refined need before proceeding.
    """
    from cairn.config import THIN_RETRIEVAL_MIN_ENTRIES, THIN_RETRIEVAL_TOP_SIM_THRESHOLD
    if not entries:
        return True, {"reason": "empty", "count": 0}
    if len(entries) < THIN_RETRIEVAL_MIN_ENTRIES:
        return True, {"reason": "too_few", "count": len(entries)}
    similarities = [e.get("similarity", 0) for e in entries]
    max_sim = max(similarities) if similarities else 0
    if max_sim < THIN_RETRIEVAL_TOP_SIM_THRESHOLD:
        return True, {
            "reason": "top_too_weak",
            "count": len(entries),
            "max_sim": round(max_sim, 3),
        }
    return False, {
        "reason": "ok",
        "count": len(entries),
        "max_sim": round(max_sim, 3),
    }


def get_adaptive_threshold_boost() -> float:
    """Check recent retrieval outcomes. If harmful/neutral rate is high, boost the similarity floor."""
    try:
        conn = get_ephemeral_conn()
        recent = conn.execute("""
            SELECT event, COUNT(*) FROM metrics
            WHERE event IN ('retrieval_useful', 'retrieval_neutral', 'retrieval_harmful')
            AND created_at > datetime('now', '-7 days')
            GROUP BY event
        """).fetchall()
        conn.close()

        counts = {r[0]: r[1] for r in recent}
        total = sum(counts.values())
        if total < 5:
            return 0.0

        harmful_rate = (counts.get("retrieval_harmful", 0) + counts.get("retrieval_neutral", 0)) / total
        if harmful_rate > 0.5:
            return 0.10
        elif harmful_rate > 0.3:
            return 0.05
        return 0.0
    except (sqlite3.Error, OSError):
        return 0.0


def retrieve_context(context_need: str, session_id: Optional[str] = None, max_per_scope: Optional[int] = None) -> Optional[str]:
    """Search the cairn for memories matching the context need. Returns structured XML context.

    Args:
        max_per_scope: Override max results per scope (project/global). Used by bootstrap
                       to limit noise. If None, uses L3_MAX_PROJECT_RESULTS/L3_MAX_GLOBAL_RESULTS.
    """
    import time
    from datetime import datetime as dt
    start = time.time()
    conn = get_conn()
    project = get_session_project(conn, session_id)

    max_project = max_per_scope or L3_MAX_PROJECT_RESULTS
    max_global = max_per_scope or L3_MAX_GLOBAL_RESULTS

    project_results, global_results, threshold_boost = hybrid_search(
        context_need, conn, project=project, session_id=session_id,
        threshold=L3_PROJECT_SIM_THRESHOLD, limit=max(max_project, max_global),
    )

    # L3-specific: filter already-served entries
    from hooks.hook_helpers import load_injected_ids
    already_served: set[int] = load_injected_ids(session_id) if session_id else set()
    if already_served:
        project_results = [r for r in project_results if r["id"] not in already_served]
        global_results = [r for r in global_results if r["id"] not in already_served]

    project_results = project_results[:max_project]
    global_results = global_results[:max_global]

    conn.close()

    elapsed_ms: float = (time.time() - start) * 1000
    total = len(project_results) + len(global_results)
    record_metric(session_id, "context_retrieval", context_need[:100], total)
    record_metric(session_id, "retrieval_latency_ms", context_need[:50], elapsed_ms)
    log(f"Retrieval: {len(project_results)} project + {len(global_results)} global in {elapsed_ms:.0f}ms")

    # Thin-retrieval detection — record a flag if results are too few or all weak.
    # Stop hook will use this on the next turn to force escalation if the LLM
    # doesn't run query.py or re-declare with a refined need.
    is_thin, thin_diag = _is_thin_retrieval(project_results + global_results)
    if is_thin:
        from datetime import datetime
        from hooks.hook_helpers import save_hook_state
        save_hook_state(session_id, "pending_thin_retrieval", json.dumps({
            "timestamp": datetime.now().isoformat(),
            "context_need": context_need[:200],
            "diagnostics": thin_diag,
        }))
        record_metric(session_id, "thin_retrieval_detected", context_need[:100], thin_diag.get("count", 0))
        log(f"Thin retrieval flagged: {thin_diag}")
    else:
        # Healthy retrieval — clear any prior pending flag
        from hooks.hook_helpers import delete_hook_state
        delete_hook_state(session_id, "pending_thin_retrieval")

    if not project_results and not global_results:
        record_metric(session_id, "context_empty", context_need[:100])
        return None

    # L3 uses a compact entry format (fewer attributes = fewer tokens in block reason)
    def format_entry(r: dict[str, Any]) -> str:
        score = r.get("score", r.get("confidence", 0.7))
        rel_label = reliability_label(score)
        days = _recency_days(r.get("updated_at", ""))
        sim = r.get("similarity", 0)
        reason = r.get("archived_reason")
        if r.get("archived") or reason:
            return (
                f'  <entry id="{r["id"]}" superseded="true" reason="{reason}" days="{days}">'
                f'{r["content"]}</entry>'
            )
        return (
            f'  <entry id="{r["id"]}" days="{days}" sim="{sim:.2f}">'
            f'{r["content"]}</entry>'
        )

    lines: list[str] = ['<cairn_context query="{}" current_project="{}" layer="L3">'.format(
        context_need.replace('"', '&quot;'),
        project or "none"
    )]

    project_cap = max_per_scope if max_per_scope is not None else L3_MAX_PROJECT_RESULTS
    global_cap = max_per_scope if max_per_scope is not None else L3_MAX_GLOBAL_RESULTS

    if project_results:
        project_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        lines.append(f'  <scope level="project" name="{project}" weight="high">')
        for r in project_results[:project_cap]:
            lines.append("  " + format_entry(r))
        lines.append("  </scope>")

    if global_results:
        global_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        lines.append('  <scope level="global" weight="low">')
        for r in global_results[:global_cap]:
            lines.append("  " + format_entry(r))
        lines.append("  </scope>")

    lines.append("</cairn_context>")
    return "\n".join(lines)


def _keyword_match_search(conn, keywords_list: list[str], project: Optional[str],
                          limit: int) -> list[dict]:
    """Find cross-project memories that share exact keywords.

    Uses the persisted keywords column for precise matching — no embedding needed.
    Scores by keyword overlap count. Complements semantic search by catching
    cross-project connections that embedding similarity misses (different domains,
    same concept).
    """
    if not keywords_list:
        return []

    # Build LIKE conditions for each keyword (case-insensitive, comma-separated field)
    conditions = []
    params = []
    for kw in keywords_list:
        kw_clean = kw.strip().lower()
        if kw_clean:
            conditions.append("LOWER(keywords) LIKE ?")
            params.append(f"%{kw_clean}%")

    if not conditions:
        return []

    # Find memories matching at least one keyword, from OTHER projects
    where_kw = " OR ".join(conditions)
    project_filter = "AND project != ?" if project else "AND project IS NOT NULL"
    if project:
        params.append(project)
    params.append(limit * 3)  # Over-fetch for scoring

    try:
        rows = conn.execute(f"""
            SELECT id, type, topic, content, updated_at, project, confidence, keywords
            FROM memories
            WHERE keywords IS NOT NULL
            AND ({where_kw})
            AND (archived_reason IS NULL OR archived_reason = '')
            AND deleted_at IS NULL
            {project_filter}
            ORDER BY updated_at DESC
            LIMIT ?
        """, params).fetchall()
    except Exception as e:
        log(f"Keyword match search error: {type(e).__name__}: {e}")
        return []

    # Score by keyword overlap count
    kw_set = {k.strip().lower() for k in keywords_list if k.strip()}
    results = []
    for r in rows:
        mem_id, mem_type, topic, content, updated_at, mem_project, confidence, mem_keywords = r
        mem_kw_set = {k.strip().lower() for k in (mem_keywords or "").split(",") if k.strip()}
        overlap = len(kw_set & mem_kw_set)
        if overlap < 2:
            continue  # Require ≥2 keyword overlap to reduce cross-project noise
        results.append({
            "id": mem_id, "type": mem_type, "topic": topic, "content": content,
            "updated_at": updated_at, "project": mem_project,
            "confidence": confidence if confidence is not None else 0.7,
            "similarity": 0.0,  # Not semantic
            "score": overlap / max(len(kw_set), 1),  # Overlap ratio
            "keyword_overlap": overlap,
        })

    results.sort(key=lambda x: (-x["keyword_overlap"], x["updated_at"] or ""), reverse=False)
    results.sort(key=lambda x: x["keyword_overlap"], reverse=True)
    return results[:limit]


def layer2_cross_project_search(keywords_list: list[str], session_id: Optional[str] = None) -> None:
    """Layer 2: Cross-project search using hybrid search (semantic + FTS5 + RRF).

    Also includes exact keyword matching on the persisted keywords column for
    connections that embeddings miss (same concept, different domain language).
    Stages results for next UserPromptSubmit injection.
    """
    from cairn.config import L2_SIM_THRESHOLD, L2_MAX_RESULTS

    if not keywords_list:
        return

    conn = get_conn()
    project = get_session_project(conn, session_id)

    # Exact keyword matching (no embedder needed) — catches cross-project by shared keywords
    keyword_results = _keyword_match_search(conn, keywords_list, project, L2_MAX_RESULTS)
    keyword_ids = {r["id"] for r in keyword_results}

    if keyword_results:
        log(f"Layer 2 keyword match: {len(keyword_results)} cross-project hits")
        record_metric(session_id, "layer2_keyword_matches", None, len(keyword_results))

    # Hybrid search (semantic + FTS5 + RRF) — cross-project only
    query = " ".join(keywords_list)
    _, hybrid_results, _ = hybrid_search(
        query, conn, project=project, session_id=session_id,
        threshold=L2_SIM_THRESHOLD, limit=L2_MAX_RESULTS,
        exclude_project=True,
    )

    # Merge: keyword matches + hybrid results, deduplicated
    cross_project = list(keyword_results)
    for r in hybrid_results:
        if r["id"] not in keyword_ids:
            cross_project.append(r)
    cross_project = cross_project[:L2_MAX_RESULTS]

    if not cross_project:
        log(f"Layer 2: no cross-project matches for keywords: {' '.join(keywords_list)[:50]}")
        conn.close()
        return

    staged_xml = build_context_xml(
        f"cross-project keywords: {' '.join(keywords_list)[:60]}", project, "cross-project",
        [], cross_project
    )

    save_hook_state(session_id, "staged_context", staged_xml)
    conn.close()

    log(f"Layer 2: staged {len(cross_project)} cross-project entries "
        f"({len(keyword_results)} keyword, {len(hybrid_results)} hybrid)")
    record_metric(session_id, "layer2_staged", " ".join(keywords_list)[:100], len(cross_project))
    result_ids = [r["id"] for r in cross_project if "id" in r]
    if result_ids:
        import json as _json
        record_metric(session_id, "layer_delivery", _json.dumps({"layer": "L2", "ids": result_ids}))


# --- Context cache ---

def load_context_cache(session_id: str) -> list[dict[str, Any]]:
    conn = get_ephemeral_conn()
    row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'context_cache'",
        (session_id,)
    ).fetchone()
    conn.close()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return []
    return []


def save_context_cache(session_id: str, served_needs: list[dict[str, Any]]) -> None:
    conn = get_ephemeral_conn()
    conn.execute(
        "INSERT OR REPLACE INTO hook_state (session_id, key, value) VALUES (?, 'context_cache', ?)",
        (session_id, json.dumps(served_needs))
    )
    conn.commit()
    conn.close()


def is_context_cached(context_need: str, served_needs: list[dict[str, Any]], emb: Any) -> bool:
    """Check if a semantically similar context_need has already been served."""
    if not emb or not served_needs:
        return any(s.get("text") == context_need for s in served_needs)
    try:
        import numpy as np
        query_vec = emb.embed(context_need)
        for cached in served_needs:
            cached_vec = np.frombuffer(bytes.fromhex(cached["embedding_hex"]), dtype=np.float32)
            sim = emb.cosine_similarity(query_vec, cached_vec)
            if sim >= CONTEXT_CACHE_SIM_THRESHOLD:
                return True
    except Exception as e:
        log(f"Context cache semantic check failed, falling back to exact match: {type(e).__name__}: {e}")
        return any(s.get("text") == context_need for s in served_needs)
    return False


def add_to_context_cache(context_need: str, served_needs: list[dict[str, Any]], emb: Any) -> list[dict[str, Any]]:
    """Add a context_need to the cache with its embedding."""
    entry: dict[str, Any] = {"text": context_need}
    if emb:
        try:
            vec = emb.embed(context_need)
            entry["embedding_hex"] = emb.to_blob(vec).hex()
        except Exception as e:
            log(f"Context cache embedding failed: {type(e).__name__}: {e}")
    served_needs.append(entry)
    return served_needs
