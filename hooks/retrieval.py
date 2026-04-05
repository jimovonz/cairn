"""Context retrieval — Layer 2 (cross-project) and Layer 3 (on-demand pull)."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

import hooks.hook_helpers as hook_helpers
from hooks.hook_helpers import log, get_conn, get_session_project, record_metric
import re

from cairn.config import (L3_PROJECT_SIM_THRESHOLD, L3_GLOBAL_SIM_WITH_PROJECT,
                     L3_GLOBAL_SIM_WITHOUT_PROJECT, L3_PROJECT_QUALITY_FLOOR,
                     L3_MAX_PROJECT_RESULTS, L3_MAX_GLOBAL_RESULTS,
                     WEAK_ENTRY_SCORE_FLOOR, RRF_K)


CONTEXT_CACHE_SIM_THRESHOLD: float = 0.9


def get_adaptive_threshold_boost() -> float:
    """Check recent retrieval outcomes. If harmful/neutral rate is high, boost the similarity floor."""
    try:
        conn = get_conn()
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

    emb = hook_helpers.get_embedder()
    project_results: list[dict[str, Any]] = []
    global_results: list[dict[str, Any]] = []

    threshold_boost = get_adaptive_threshold_boost()
    if threshold_boost > 0:
        log(f"Adaptive threshold boost: +{threshold_boost:.2f}")

    # --- Collect candidates from both search methods ---
    # semantic_ranked: {memory_id: (rank_position, result_dict)}
    # fts_ranked:      {memory_id: (rank_position, result_dict)}
    semantic_ranked: dict[int, tuple[int, dict[str, Any]]] = {}
    fts_ranked: dict[int, tuple[int, dict[str, Any]]] = {}

    if emb:
        try:
            # Primary search: project-prefixed query (biased toward project-local matches)
            all_results = emb.find_similar(conn, context_need, current_project=project)

            # Mitigation 3: Unprefixed search for cross-project matches
            if project:
                unprefixed_results = emb.find_similar(conn, context_need, current_project=None)
                # Merge unprefixed results (keep best score per id)
                by_id = {r["id"]: r for r in all_results}
                for r in unprefixed_results:
                    if r["id"] not in by_id or r["score"] > by_id[r["id"]]["score"]:
                        by_id[r["id"]] = r
                all_results = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)

            # Build ranked list (excluding same-session)
            rank = 0
            for r in all_results:
                if session_id and r.get("session_id") == session_id:
                    continue
                semantic_ranked[r["id"]] = (rank, r)
                rank += 1

        except (ConnectionError, TimeoutError, OSError) as e:
            log(f"Context retrieval embedding unavailable: {e}")
        except Exception as e:
            log(f"Context retrieval error ({type(e).__name__}): {e}")

    # FTS5 keyword search — runs alongside semantic for hybrid fusion
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
    try:
        words = re.findall(r'\w+', context_need.lower())
        meaningful = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
        if not meaningful:
            meaningful = [w for w in words if len(w) > 2]
        if not meaningful:
            meaningful = words
        fts_query = " OR ".join(f'"{w}"' for w in meaningful) if meaningful else context_need
        rows = conn.execute("""
            SELECT m.id, m.type, m.topic, m.content, m.updated_at, m.project,
                   m.session_id, m.confidence, m.depth, m.archived_reason, rank
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.id
            WHERE memories_fts MATCH ?
            ORDER BY rank LIMIT 20
        """, (fts_query,)).fetchall()
        rank = 0
        for r in rows:
            if session_id and r[6] == session_id:
                continue
            confidence = r[7] if r[7] is not None else 0.7
            entry: dict[str, Any] = {
                "id": r[0], "type": r[1], "topic": r[2], "content": r[3],
                "updated_at": r[4], "project": r[5], "session_id": r[6],
                "confidence": confidence, "depth": r[8],
                "archived_reason": r[9], "bm25_rank": -r[10],  # FTS5 rank is negative (lower = better)
            }
            fts_ranked[r[0]] = (rank, entry)
            rank += 1
    except Exception as e:
        log(f"FTS search error: {e}")

    # --- RRF Fusion ---
    # Merge both result sets. Memories found by both methods get boosted.
    all_ids = set(semantic_ranked.keys()) | set(fts_ranked.keys())
    fused: list[dict[str, Any]] = []

    for mid in all_ids:
        # RRF score: sum of 1/(k + rank) across each method that found this memory
        rrf_score = 0.0
        result_dict: Optional[dict[str, Any]] = None

        if mid in semantic_ranked:
            sem_rank, sem_result = semantic_ranked[mid]
            rrf_score += 1.0 / (RRF_K + sem_rank)
            result_dict = sem_result

        if mid in fts_ranked:
            fts_rank_pos, fts_result = fts_ranked[mid]
            rrf_score += 1.0 / (RRF_K + fts_rank_pos)
            # If we didn't get this from semantic, use the FTS result
            if result_dict is None:
                # FTS-only result needs a composite score computed from its metadata
                from cairn.embeddings import composite_score as _cscore
                fts_sim = 0.35  # FTS doesn't produce a vector similarity; use baseline
                fts_result["similarity"] = fts_sim
                fts_result["score"] = _cscore(
                    fts_sim, fts_result["confidence"],
                    fts_result.get("updated_at"), fts_result.get("project"), project
                )
                result_dict = fts_result

        if result_dict is None:
            continue

        # Blend: use the original composite score but boost by RRF contribution
        # RRF score range for k=60: single-method top rank = 1/61 ≈ 0.016,
        # dual-method top rank = 2/61 ≈ 0.033. Normalise to a meaningful boost.
        # Max possible RRF = 2/61 ≈ 0.033. Scale so dual-top gives ~0.20 boost.
        rrf_boost = rrf_score * (0.20 / (2.0 / (RRF_K + 1)))
        result_dict["rrf_score"] = rrf_score
        result_dict["score"] = result_dict.get("score", 0.30) + rrf_boost

        # Tag source method for logging
        in_sem = mid in semantic_ranked
        in_fts = mid in fts_ranked
        result_dict["_source"] = "both" if (in_sem and in_fts) else ("semantic" if in_sem else "fts")

        fused.append(result_dict)

    # Sort by fused score and split into project/global
    fused.sort(key=lambda x: x["score"], reverse=True)

    seen_ids: set[int] = set()
    project_threshold: float = L3_PROJECT_SIM_THRESHOLD + threshold_boost
    quality_project_count = 0

    for r in fused:
        mid = r["id"]
        if mid in seen_ids:
            continue
        sim = r.get("similarity", 0)

        if project and r.get("project") == project and sim >= project_threshold:
            project_results.append(r)
            seen_ids.add(mid)
            if sim >= L3_PROJECT_QUALITY_FLOOR:
                quality_project_count += 1
        # Defer global results until we know the threshold

    global_threshold: float = (L3_GLOBAL_SIM_WITH_PROJECT if quality_project_count > 0
                               else L3_GLOBAL_SIM_WITHOUT_PROJECT) + threshold_boost

    for r in fused:
        mid = r["id"]
        if mid in seen_ids:
            continue
        sim = r.get("similarity", 0)
        # FTS-only results (sim=0.35) pass through if their fused score is strong enough
        # Use a relaxed threshold for FTS-only: if RRF boosted score >= global threshold, allow
        if sim >= global_threshold or (r.get("_source") == "fts" and r["score"] >= WEAK_ENTRY_SCORE_FLOOR):
            global_results.append(r)
            seen_ids.add(mid)

    # Log RRF fusion stats
    both_count = sum(1 for r in fused if r.get("_source") == "both")
    sem_only = sum(1 for r in fused if r.get("_source") == "semantic")
    fts_only = sum(1 for r in fused if r.get("_source") == "fts")
    log(f"RRF fusion: {len(fused)} candidates ({both_count} both, {sem_only} sem-only, {fts_only} fts-only)")

    conn.close()

    elapsed_ms: float = (time.time() - start) * 1000
    total = len(project_results) + len(global_results)
    record_metric(session_id, "context_retrieval", context_need[:100], total)
    record_metric(session_id, "retrieval_latency_ms", context_need[:50], elapsed_ms)
    log(f"Retrieval: {len(project_results)} project + {len(global_results)} global in {elapsed_ms:.0f}ms")

    if not project_results and not global_results:
        record_metric(session_id, "context_empty", context_need[:100])
        return None

    def recency_days(updated_at_str: str) -> int:
        try:
            updated = dt.strptime(updated_at_str[:19], "%Y-%m-%d %H:%M:%S")
            return max(0, (dt.now() - updated).days)
        except Exception:
            return -1

    def reliability(r: dict[str, Any]) -> float:
        return r.get("score", r.get("confidence", 0.7))

    def format_entry(r: dict[str, Any]) -> str:
        rel = reliability(r)
        rel_label = "strong" if rel >= 0.6 else "moderate" if rel >= 0.4 else "weak"
        days = recency_days(r.get("updated_at", ""))
        has_source = r.get("depth") is not None
        source_attr = ' ctx="y"' if has_source else ""
        reason = r.get("archived_reason")
        if r.get("archived") or reason:
            return (
                f'  <entry id="{r["id"]}" superseded="true" reason="{reason}" days="{days}">'
                f'{r["content"]}</entry>'
            )
        return (
            f'  <entry id="{r["id"]}" reliability="{rel_label}" days="{days}"{source_attr}>'
            f'{r["content"]}</entry>'
        )

    lines: list[str] = ['<cairn_context query="{}" current_project="{}">'.format(
        context_need.replace('"', '&quot;'),
        project or "none"
    )]
    lines.append('  <instruction>Before acting on any entry below, run: python3 /home/james/Projects/cairn/cairn/query.py --context &lt;id&gt; to recover the full conversation behind it.</instruction>')

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


def layer2_cross_project_search(keywords_list: list[str], session_id: Optional[str] = None) -> None:
    """Layer 2: Search global memories for cross-project relevance using keywords.
    Stages results for the next UserPromptSubmit hook injection."""
    from cairn.config import L2_SIM_THRESHOLD, L2_MAX_RESULTS

    if not keywords_list:
        return

    emb = hook_helpers.get_embedder()
    if not emb:
        return

    conn = get_conn()
    project = get_session_project(conn, session_id)

    query = " ".join(keywords_list)
    try:
        results = emb.find_similar(conn, query, threshold=L2_SIM_THRESHOLD,
                                   limit=L2_MAX_RESULTS * 2, current_project=project)
    except Exception as e:
        log(f"Layer 2 search error: {e}")
        conn.close()
        return

    cross_project = [r for r in results
                     if r.get("project") != project
                     and r["similarity"] >= L2_SIM_THRESHOLD][:L2_MAX_RESULTS]

    if not cross_project:
        log(f"Layer 2: no cross-project matches for keywords: {query[:50]}")
        conn.close()
        return

    from datetime import datetime as dt
    lines: list[str] = [f'<cairn_context query="cross-project keywords: {query[:60]}" current_project="{project or "none"}" layer="cross-project">']
    lines.append('  <instruction>Before acting on any entry below, run: python3 /home/james/Projects/cairn/cairn/query.py --context &lt;id&gt; to recover the full conversation behind it.</instruction>')
    lines.append('  <scope level="global" weight="low">')
    for r in cross_project:
        proj: str = r.get("project") or "global"
        conf: float = r.get("confidence", 0.7)
        score: float = r.get("score", conf)
        days: int = 0
        try:
            updated = dt.strptime(r["updated_at"][:19], "%Y-%m-%d %H:%M:%S")
            days = max(0, (dt.now() - updated).days)
        except Exception:
            pass
        rel: str = "strong" if score >= 0.6 else "moderate" if score >= 0.4 else "weak"
        lines.append(
            f'    <entry id="{r["id"]}" type="{r["type"]}" topic="{r["topic"]}" '
            f'project="{proj}" date="{r["updated_at"]}" confidence="{conf:.2f}" '
            f'score="{score:.2f}" recency_days="{days}" reliability="{rel}" similarity="{r["similarity"]:.2f}">'
            f'{r["content"]}</entry>'
        )
    lines.append('  </scope>')
    lines.append('</cairn_context>')

    staged_xml: str = "\n".join(lines)

    conn.execute(
        "INSERT OR REPLACE INTO hook_state (session_id, key, value) VALUES (?, 'staged_context', ?)",
        (session_id, staged_xml)
    )
    conn.commit()
    conn.close()

    log(f"Layer 2: staged {len(cross_project)} cross-project entries for next prompt")
    record_metric(session_id, "layer2_staged", query[:100], len(cross_project))
    result_ids = [r["id"] for r in cross_project if "id" in r]
    if result_ids:
        import json as _json
        record_metric(session_id, "layer_delivery", _json.dumps({"layer": "L2", "ids": result_ids}))


# --- Context cache ---

def load_context_cache(session_id: str) -> list[dict[str, Any]]:
    conn = get_conn()
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
    conn = get_conn()
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
    except Exception:
        return any(s.get("text") == context_need for s in served_needs)
    return False


def add_to_context_cache(context_need: str, served_needs: list[dict[str, Any]], emb: Any) -> list[dict[str, Any]]:
    """Add a context_need to the cache with its embedding."""
    entry: dict[str, Any] = {"text": context_need}
    if emb:
        try:
            vec = emb.embed(context_need)
            entry["embedding_hex"] = emb.to_blob(vec).hex()
        except Exception:
            pass
    served_needs.append(entry)
    return served_needs
