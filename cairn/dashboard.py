#!/usr/bin/env python3
"""Cairn Dashboard — Web UI for monitoring and managing the memory system.

Zero external dependencies — uses Python stdlib http.server only.
"""

import json
import os
import re
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err
import sys
import webbrowser
from collections import Counter
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Re-exec under venv python if not already in the venv
_venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "python3")
if os.path.exists(_venv_python) and sys.prefix == sys.base_prefix:
    os.execv(_venv_python, [_venv_python] + sys.argv)

from cairn import config

DB_PATH = os.path.join(os.path.dirname(__file__), "cairn.db")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row):
    return dict(row) if row else None


def rows_to_list(rows):
    return [dict(r) for r in rows]


# ─── Route handlers ─────────────────────────────────────────────────────────
# Each returns (data_dict, status_code). The request handler serialises to JSON.

def api_stats(params):
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    with_emb = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL").fetchone()[0]
    history_count = conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0]
    types = conn.execute("SELECT type, COUNT(*) as c FROM memories GROUP BY type ORDER BY c DESC").fetchall()
    mem_sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM memories WHERE session_id IS NOT NULL").fetchone()[0]
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conf_dist = conn.execute("""
        SELECT
            COUNT(CASE WHEN confidence < 0.3 THEN 1 END) as suppressed,
            COUNT(CASE WHEN confidence >= 0.3 AND confidence < 0.5 THEN 1 END) as low,
            COUNT(CASE WHEN confidence >= 0.5 AND confidence < 0.7 THEN 1 END) as medium,
            COUNT(CASE WHEN confidence >= 0.7 AND confidence < 0.9 THEN 1 END) as high,
            COUNT(CASE WHEN confidence >= 0.9 THEN 1 END) as very_high,
            AVG(confidence) as avg_confidence
        FROM memories
    """).fetchone()
    contradiction_count = 0
    try:
        contradiction_count = conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE event = 'contradiction_detected'"
        ).fetchone()[0]
    except Exception:
        pass
    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    growth = rows_to_list(conn.execute("""
        SELECT DATE(created_at, 'localtime') as date, COUNT(*) as count
        FROM memories GROUP BY DATE(created_at, 'localtime') ORDER BY date
    """).fetchall())
    conn.close()
    return {
        "total_memories": total, "with_embeddings": with_emb, "history_entries": history_count,
        "memory_sessions": mem_sessions, "tracked_sessions": session_count, "db_size_bytes": db_size,
        "types": {t["type"]: t["c"] for t in types},
        "confidence": {
            "suppressed": conf_dist["suppressed"], "low": conf_dist["low"],
            "medium": conf_dist["medium"], "high": conf_dist["high"],
            "very_high": conf_dist["very_high"],
            "average": round(conf_dist["avg_confidence"] or 0, 3),
        },
        "contradictions": contradiction_count, "growth": growth,
    }, 200


def api_memories(params):
    q = params.get("q", "").strip()
    mem_type = params.get("type", "").strip()
    project = params.get("project", "").strip()
    session_id = params.get("session", "").strip()
    limit = min(int(params.get("limit", 50)), 200)
    offset = int(params.get("offset", 0))
    mode = params.get("mode", "fts")
    sort = params.get("sort", "updated_at")
    order = params.get("order", "desc")

    allowed_sorts = {"id", "type", "topic", "confidence", "updated_at", "created_at", "project", "served_count"}
    if sort not in allowed_sorts:
        sort = "updated_at"
    order_dir = "ASC" if order.lower() == "asc" else "DESC"

    conn = get_conn()

    if q and mode == "semantic":
        try:
            from cairn import embeddings as emb
            results = emb.find_similar(conn, q, threshold=0.3, limit=limit)
            conn.close()
            memories = [{
                "id": r["id"], "type": r.get("type", ""), "topic": r.get("topic", ""),
                "content": r.get("content", ""), "similarity": round(r.get("similarity", 0), 3),
                "confidence": r.get("confidence", 0.7), "project": r.get("project", ""),
                "updated_at": r.get("updated_at", ""), "keywords": r.get("keywords", ""),
            } for r in results]
            return {"memories": memories, "total": len(memories)}, 200
        except Exception as e:
            conn.close()
            return {"error": str(e)}, 500

    where, sql_params = [], []
    if q:
        fts_order = "rank" if sort == "updated_at" and order == "desc" else f"m.{sort} {order_dir}"
        rows = conn.execute(f"""
            SELECT m.id, m.type, m.topic, m.content, m.confidence, m.project,
                   m.session_id, m.updated_at, m.created_at, m.archived_reason, m.keywords, m.facts
            FROM memories_fts f JOIN memories m ON f.rowid = m.id
            WHERE memories_fts MATCH ? ORDER BY {fts_order} LIMIT ? OFFSET ?
        """, (q, limit, offset)).fetchall()
        total_row = conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?", (q,)).fetchone()
        total = total_row[0] if total_row else len(rows)
    else:
        if mem_type:
            where.append("type = ?"); sql_params.append(mem_type)
        if project:
            where.append("project = ?"); sql_params.append(project)
        if session_id:
            where.append("session_id LIKE ?"); sql_params.append(f"{session_id}%")
        where_clause = " WHERE " + " AND ".join(where) if where else ""
        total = conn.execute(f"SELECT COUNT(*) FROM memories{where_clause}", sql_params).fetchone()[0]
        if sort == "served_count":
            # Special path: served_count is computed from layer_delivery
            # metric events, not stored on memories. Fetch all matching
            # IDs, score them, sort, page, then load full rows.
            id_rows = conn.execute(
                f"SELECT id FROM memories{where_clause}", sql_params
            ).fetchall()
            ids_all = [r["id"] for r in id_rows]
            served = _memory_served_counts(ids_all) if ids_all else {}
            reverse = order_dir == "DESC"
            sorted_ids = sorted(
                ids_all,
                key=lambda i: (served.get(i, 0), i),
                reverse=reverse,
            )
            page_ids = sorted_ids[offset:offset + limit]
            if not page_ids:
                rows = []
            else:
                placeholders = ",".join("?" * len(page_ids))
                rows = conn.execute(f"""
                    SELECT id, type, topic, content, confidence, project,
                           session_id, updated_at, created_at, archived_reason, keywords, facts
                    FROM memories WHERE id IN ({placeholders})
                """, page_ids).fetchall()
                # Preserve sort order from page_ids
                by_id = {r["id"]: r for r in rows}
                rows = [by_id[i] for i in page_ids if i in by_id]
        else:
            rows = conn.execute(f"""
                SELECT id, type, topic, content, confidence, project,
                       session_id, updated_at, created_at, archived_reason, keywords, facts
                FROM memories{where_clause} ORDER BY {sort} {order_dir} LIMIT ? OFFSET ?
            """, sql_params + [limit, offset]).fetchall()

    memories = [{
        "id": r["id"], "type": r["type"], "topic": r["topic"], "content": r["content"],
        "confidence": r["confidence"], "project": r["project"], "session_id": r["session_id"],
        "updated_at": r["updated_at"], "created_at": r["created_at"], "archived_reason": r["archived_reason"],
        "keywords": r["keywords"], "facts": r["facts"],
    } for r in rows]
    conn.close()
    # Enrich with served counts — how many times each memory has been
    # surfaced via layer_delivery metric events. Lets the UI render a
    # visual "served" indicator distinguishing live-used memories.
    if memories:
        ids = [m["id"] for m in memories]
        served = _memory_served_counts(ids)
        for m in memories:
            m["served_count"] = served.get(m["id"], 0)
    return {"memories": memories, "total": total, "sort": sort,
            "order": order_dir.lower()}, 200


def _memory_served_counts(memory_ids: list[int]) -> dict[int, int]:
    """Count how many times each given memory_id appears in
    layer_delivery metric events. One scan of the metric table; cheap
    for typical page sizes (50 memories)."""
    if not memory_ids:
        return {}
    eph = _get_eph_conn()
    counts: dict[int, int] = {}
    try:
        # layer_delivery detail is JSON like {"layer": "L1.5", "ids": [..]}
        rows = eph.execute(
            "SELECT detail FROM metrics WHERE event = 'layer_delivery'"
        ).fetchall()
    finally:
        eph.close()
    wanted = set(memory_ids)
    for (detail,) in rows:
        if not detail:
            continue
        try:
            data = json.loads(detail)
            for mid in data.get("ids", []):
                if mid in wanted:
                    counts[mid] = counts.get(mid, 0) + 1
        except (ValueError, TypeError):
            continue
    return counts


def api_memory_detail(params, memory_id):
    conn = get_conn()
    row = conn.execute("""
        SELECT id, type, topic, content, confidence, project, session_id,
               updated_at, created_at, archived_reason, depth, associated_files, keywords, facts,
               source_ref
        FROM memories WHERE id = ?
    """, (memory_id,)).fetchone()
    if not row:
        conn.close()
        return {"error": "Not found"}, 404
    history = rows_to_list(conn.execute("""
        SELECT id, content, session_id, changed_at
        FROM memory_history WHERE memory_id = ? ORDER BY changed_at DESC
    """, (memory_id,)).fetchall())
    conn.close()
    return {"memory": row_to_dict(row), "history": history}, 200


def api_memory_context(params, memory_id):
    margin = int(params.get("margin", 8))
    conn = get_conn()
    row = conn.execute(
        "SELECT id, session_id, created_at, depth FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    if not row:
        conn.close()
        return {"error": "Not found"}, 404
    session_id, created_at, depth = row["session_id"], row["created_at"], row["depth"]
    if not session_id:
        conn.close()
        return {"messages": [], "note": "No session ID"}, 200
    session = conn.execute(
        "SELECT transcript_path FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if not session or not session["transcript_path"]:
        return {"messages": [], "note": "No transcript path"}, 200
    transcript_path = session["transcript_path"]
    if not os.path.exists(transcript_path):
        return {"messages": [], "note": "Transcript not found"}, 200
    try:
        mem_time = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return {"messages": [], "note": "Cannot parse timestamp"}, 200
    lookback = depth if depth and depth > 0 else margin
    messages = _parse_transcript(transcript_path)
    if not messages:
        return {"messages": [], "note": "No messages in transcript"}, 200
    anchor_idx = len(messages) - 1
    for i, msg in enumerate(messages):
        if msg["ts"] and msg["ts"] > mem_time:
            anchor_idx = max(0, i - 1)
            break
    start_idx = max(0, anchor_idx - lookback)
    end_idx = min(len(messages), anchor_idx + 3)
    context_messages = messages[start_idx:end_idx]
    for i, msg in enumerate(context_messages):
        msg["is_anchor"] = (start_idx + i == anchor_idx)
        msg["ts"] = msg["ts"].isoformat() if msg["ts"] else None
    return {"messages": context_messages, "anchor_idx": anchor_idx - start_idx}, 200


def api_sessions(params):
    limit = min(int(params.get("limit", 50)), 200)
    offset = int(params.get("offset", 0))
    sort = params.get("sort", "started_at")
    order = params.get("order", "desc")
    hide_empty = params.get("hide_empty", "false").lower() in ("1", "true", "yes")
    allowed_sorts = {"started_at", "memory_count", "project"}
    if sort not in allowed_sorts:
        sort = "started_at"
    order_dir = "ASC" if order.lower() == "asc" else "DESC"
    conn = get_conn()
    having_clause = "HAVING memory_count > 0" if hide_empty else ""
    total_sql = f"""
        SELECT COUNT(*) FROM (
            SELECT s.session_id, COUNT(m.id) as memory_count
            FROM sessions s LEFT JOIN memories m ON m.session_id = s.session_id
            GROUP BY s.session_id {having_clause}
        )
    """
    total = conn.execute(total_sql).fetchone()[0]
    sessions = conn.execute(f"""
        SELECT s.session_id, s.parent_session_id, s.project, s.started_at, s.transcript_path,
               COUNT(m.id) as memory_count,
               (SELECT COUNT(*) FROM sessions c WHERE c.parent_session_id = s.session_id) as child_count
        FROM sessions s
        LEFT JOIN memories m ON m.session_id = s.session_id
        GROUP BY s.session_id
        {having_clause}
        ORDER BY {sort} {order_dir} LIMIT ? OFFSET ?
    """, (limit, offset)).fetchall()
    result = []
    for s in sessions:
        d = dict(s)
        d["cwd"] = _extract_cwd(s["transcript_path"])
        d["session_dir"] = _extract_session_dir(s["transcript_path"], cwd=d["cwd"])
        d["session_type"] = "agent" if s["parent_session_id"] else ("parent" if s["child_count"] > 0 else "session")
        d["interaction_count"] = 0
        result.append(d)
    conn.close()
    return {"sessions": result, "total": total}, 200


def api_session_detail(params, session_id):
    conn = get_conn()
    session = conn.execute(
        "SELECT session_id, parent_session_id, project, transcript_path, started_at FROM sessions WHERE session_id LIKE ?",
        (f"{session_id}%",)
    ).fetchone()
    if not session:
        conn.close()
        return {"error": "Session not found"}, 404
    full_id = session["session_id"]
    memories = rows_to_list(conn.execute("""
        SELECT id, type, topic, content, confidence, updated_at, created_at
        FROM memories WHERE session_id = ? ORDER BY created_at
    """, (full_id,)).fetchall())
    chain = rows_to_list(conn.execute("""
        WITH RECURSIVE chain(sid) AS (
            SELECT session_id FROM sessions WHERE session_id = ?
            UNION ALL
            SELECT s.session_id FROM sessions s JOIN chain c ON s.parent_session_id = c.sid
        )
        SELECT s.session_id, s.parent_session_id, s.started_at,
               (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.session_id) as memory_count
        FROM chain c JOIN sessions s ON s.session_id = c.sid ORDER BY s.started_at
    """, (full_id,)).fetchall())
    _eph = _get_eph_conn()
    retrieved_row = _eph.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'retrieved_ids'", (full_id,)
    ).fetchone()
    _eph.close()
    consumed_ids = []
    if retrieved_row and retrieved_row[0]:
        try:
            consumed_ids = json.loads(retrieved_row[0])
        except Exception:
            pass
    consumed = []
    if consumed_ids:
        placeholders = ",".join("?" * len(consumed_ids))
        consumed = rows_to_list(conn.execute(f"""
            SELECT id, type, topic, content, confidence, project, session_id, updated_at
            FROM memories WHERE id IN ({placeholders})
        """, consumed_ids).fetchall())
    conn.close()
    _eph2 = _get_eph_conn()
    layer_rows = _eph2.execute(
        "SELECT detail FROM metrics WHERE session_id = ? AND event = 'layer_delivery' AND detail IS NOT NULL",
        (full_id,)
    ).fetchall()
    _eph2.close()
    layer_detail = []
    _id_layers: dict = {}  # id -> list of layers (from layer_delivery metrics)
    for r in layer_rows:
        try:
            d = json.loads(r[0])
            layer_detail.append(d)
            lname = d.get("layer", "")
            for mid in d.get("ids", []):
                if mid not in _id_layers:
                    _id_layers[mid] = []
                if lname not in _id_layers[mid]:
                    _id_layers[mid].append(lname)
        except Exception:
            pass
    # Parse sim scores from transcript cairn_context blocks (sim attr on each <entry>)
    import re as _re
    _id_scores: dict = {}
    transcript_path = session.get("transcript_path", "") if isinstance(session, dict) else (session["transcript_path"] if session else "")
    if transcript_path:
        try:
            with open(transcript_path) as _tf:
                for _line in _tf:
                    if "cairn_context" not in _line:
                        continue
                    for _m in _re.finditer(r'<entry[^>]+?id=\\"(\d+)\\"[^>]*?sim=\\"([0-9.]+)\\"', _line):
                        _eid, _sim = int(_m.group(1)), float(_m.group(2))
                        if _eid not in _id_scores:
                            _id_scores[_eid] = _sim
                    # Also try unescaped form (direct string content)
                    for _m in _re.finditer(r'id="(\d+)"[^>]*?sim="([0-9.]+)"', _line):
                        _eid, _sim = int(_m.group(1)), float(_m.group(2))
                        if _eid not in _id_scores:
                            _id_scores[_eid] = _sim
        except Exception:
            pass
    # Attach sim score and layer info to consumed entries
    for m in consumed:
        mid = m.get("id") if isinstance(m, dict) else None
        if mid is not None:
            m["sim"] = _id_scores.get(mid)
            m["layers"] = _id_layers.get(mid, [])
    # Sort consumed by sim desc (None/missing scores sort last)
    consumed.sort(key=lambda m: (m.get("sim") is None, -(m.get("sim") or 0)))
    is_ingestion = full_id.startswith("ingest-")
    tokens = _estimate_tokens(session["transcript_path"]) if not is_ingestion else {}
    source_info = None
    if is_ingestion and memories:
        try:
            first_src = memories[0].get("source_ref") if isinstance(memories[0], dict) else None
            if not first_src:
                mc = get_conn()
                sr_row = mc.execute("SELECT source_ref FROM memories WHERE session_id = ? AND source_ref IS NOT NULL LIMIT 1", (full_id,)).fetchone()
                mc.close()
                if sr_row:
                    first_src = sr_row["source_ref"]
            if first_src:
                source_info = json.loads(first_src) if isinstance(first_src, str) else first_src
        except Exception:
            pass
    session_dict = row_to_dict(session)
    session_dict["session_dir"] = _extract_session_dir(session_dict.get("transcript_path"))
    return {
        "session": session_dict, "memories": memories, "consumed": consumed,
        "layer_detail": layer_detail, "chain": chain, "tokens": tokens,
        "is_ingestion": is_ingestion, "source_info": source_info,
    }, 200


def api_session_transcript(params, session_id):
    conn = get_conn()
    session = conn.execute(
        "SELECT transcript_path FROM sessions WHERE session_id LIKE ?", (f"{session_id}%",)
    ).fetchone()
    conn.close()
    if not session or not session["transcript_path"]:
        return {"messages": [], "note": "No transcript"}, 200

    full_id = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id LIKE ?", (f"{session_id}%",)
    ).fetchone()
    if full_id and full_id["session_id"].startswith("ingest-"):
        sr_row = conn.execute(
            "SELECT source_ref, associated_files FROM memories WHERE session_id = ? AND source_ref IS NOT NULL LIMIT 1",
            (full_id["session_id"],)
        ).fetchone()
        conn.close()
        if sr_row:
            try:
                src_ref = json.loads(sr_row["source_ref"]) if sr_row["source_ref"] else {}
                assoc = json.loads(sr_row["associated_files"]) if sr_row["associated_files"] else []
            except Exception:
                src_ref, assoc = {}, []
            repo_path = src_ref.get("path", "")
            source_files = []
            for rel_path in assoc[:20]:
                full_path = os.path.join(repo_path, rel_path) if repo_path else rel_path
                content = ""
                if os.path.exists(full_path):
                    try:
                        with open(full_path, errors="replace") as f:
                            content = f.read(10000)
                    except Exception:
                        content = "(error reading file)"
                source_files.append({"path": rel_path, "content": content})
            return {"messages": [], "source_files": source_files, "source_ref": src_ref, "note": "Ingested from repository"}, 200
        return {"messages": [], "note": "Ingestion session — no source files linked"}, 200

    transcript_path = session["transcript_path"]
    if not os.path.exists(transcript_path):
        return {"messages": [], "note": "Transcript file not found"}, 200
    messages = _parse_transcript(transcript_path)
    for msg in messages:
        msg["ts"] = msg["ts"].isoformat() if msg["ts"] else None
        if msg["role"] == "assistant" and "<memory>" in msg["text"]:
            match = re.search(r"<memory>(.*?)</memory>", msg["text"], re.DOTALL)
            if match:
                msg["memory_block"] = match.group(1).strip()
                msg["text_before_memory"] = msg["text"][:match.start()].strip()
            else:
                msg["memory_block"] = None
                msg["text_before_memory"] = msg["text"]
        else:
            msg["memory_block"] = None
            msg["text_before_memory"] = msg["text"]
    return {"messages": messages}, 200


def _metrics_conn():
    """Connection for reading metrics across BOTH the live ephemeral DB and the
    frozen historical metrics in the main DB.

    Metrics writes moved to the ephemeral DB on 2026-05-27, freezing the main
    `metrics` table. The dashboard previously read only the main DB via get_conn(),
    so it showed stale pre-cutover data and *no* live events — including the
    graph_orientation_injected / graph_file_context_injected metrics. This unions
    both tables behind a TEMP VIEW `all_metrics` so existing aggregate SQL is
    unchanged and history is preserved. When ephemeral == main (test fixtures
    point both at one file) we skip the union to avoid double-counting."""
    eph_path = getattr(config, "EPHEMERAL_DB_PATH", None) or \
        os.path.join(os.path.dirname(DB_PATH), "cairn-ephemeral.db")
    conn = sqlite3.connect(eph_path)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    cols = "event, session_id, detail, value, created_at"
    same = os.path.realpath(eph_path) == os.path.realpath(DB_PATH)
    union_main = False
    if not same and os.path.exists(DB_PATH):
        conn.execute("ATTACH DATABASE ? AS maindb", (DB_PATH,))
        # The main metrics table only exists on installs that predate the
        # 2026-05-27 move to the ephemeral DB; a fresh install / test DB has none.
        union_main = conn.execute(
            "SELECT 1 FROM maindb.sqlite_master WHERE type='table' AND name='metrics'"
        ).fetchone() is not None
    if union_main:
        conn.execute(
            f"CREATE TEMP VIEW all_metrics AS "
            f"SELECT {cols} FROM metrics "
            f"UNION ALL SELECT {cols} FROM maindb.metrics"
        )
    else:
        conn.execute(f"CREATE TEMP VIEW all_metrics AS SELECT {cols} FROM metrics")
    return conn


def api_metrics(params):
    conn = _metrics_conn()
    summary = rows_to_list(conn.execute("""
        SELECT event, COUNT(*) as count, AVG(value) as avg_value,
               MIN(value) as min_value, MAX(value) as max_value
        FROM all_metrics GROUP BY event ORDER BY count DESC
    """).fetchall())
    latency_series = rows_to_list(conn.execute("""
        SELECT DATE(created_at, 'localtime') as date, AVG(value) as avg_ms, COUNT(*) as count
        FROM all_metrics WHERE event = 'retrieval_latency_ms'
        GROUP BY DATE(created_at, 'localtime') ORDER BY date
    """).fetchall())
    layer_series = rows_to_list(conn.execute("""
        SELECT DATE(created_at, 'localtime') as date, event, COUNT(*) as count
        FROM all_metrics WHERE event LIKE 'layer%' OR event LIKE 'retrieval_%'
        GROUP BY DATE(created_at, 'localtime'), event ORDER BY date
    """).fetchall())
    embed_events = ("embed_daemon_ms", "embed_local_ms", "search_vec_ms", "search_brute_ms", "fanout_ms")
    placeholders = ",".join("?" * len(embed_events))
    embed_stats = rows_to_list(conn.execute(f"""
        SELECT event, COUNT(*) as count, AVG(value) as avg_ms,
               MIN(value) as min_ms, MAX(value) as max_ms,
               (SELECT AVG(sub.value) FROM (
                   SELECT value FROM all_metrics m2 WHERE m2.event = all_metrics.event
                   ORDER BY m2.created_at DESC LIMIT 20
               ) sub) as recent_avg_ms
        FROM all_metrics WHERE event IN ({placeholders}) GROUP BY event
    """, embed_events).fetchall())
    embed_series = rows_to_list(conn.execute(f"""
        SELECT DATE(created_at, 'localtime') as date, event, AVG(value) as avg_ms, COUNT(*) as count
        FROM all_metrics WHERE event IN ({placeholders})
        GROUP BY DATE(created_at, 'localtime'), event ORDER BY date
    """, embed_events).fetchall())
    # Graph-injection metrics surfaced explicitly so the frontend can render them
    # alongside gotcha_injected / file_context_injected without scanning `summary`.
    graph_events = ("graph_orientation_injected", "graph_file_context_injected")
    gph = ",".join("?" * len(graph_events))
    graph_injection = rows_to_list(conn.execute(f"""
        SELECT event, COUNT(*) as count
        FROM all_metrics WHERE event IN ({gph}) GROUP BY event
    """, graph_events).fetchall())
    conn.close()
    return {
        "summary": summary, "latency_series": latency_series, "layer_series": layer_series,
        "embed_stats": embed_stats, "embed_series": embed_series,
        "graph_injection": graph_injection,
    }, 200


def api_enforcement(params):
    """Enforcement metrics — surfaces which mechanical guards are firing in real use.

    Returns counts and ratios for the enforcement events captured by the hook layer.
    Most useful number: thin_retrieval satisfied/staged ratio — directly answers
    "is the LLM actually escalating when prompted, or ignoring reminders?"
    """
    conn = get_conn()
    ENFORCEMENT_EVENTS = (
        # L5 thin-retrieval escalation
        "thin_retrieval_detected",
        "thin_retrieval_escalation_staged",
        "thin_retrieval_escalation_satisfied",
        "thin_retrieval_escalation_abandoned",
        # L2 query-quality
        "context_phoned_in",
        # Active bootstrap trigger
        "active_bootstrap_triggered",
        # Existing bootstrap mechanisms (for comparison)
        "context_bootstrap_triggered",
        "context_requested",
        "question_before_cairn",
        # Trailing intent enforcement
        "trailing_intent_blocked",
        "trailing_intent_detected",
        "trailing_intent_clear",
        "trailing_intent_skipped",
        "trailing_intent_resolved_escape",
        # PostToolUse checkpoints
        "checkpoint_nudge",
        "memory_notes_stored",
        # Context recovery (LLM digs into full conversation behind a memory)
        "context_recovery_invoked",
    )
    placeholders = ",".join("?" * len(ENFORCEMENT_EVENTS))

    # Per-event totals
    totals = rows_to_list(conn.execute(f"""
        SELECT event, COUNT(*) as count
        FROM metrics WHERE event IN ({placeholders})
        GROUP BY event ORDER BY count DESC
    """, ENFORCEMENT_EVENTS).fetchall())
    totals_map = {r["event"]: r["count"] for r in totals}

    # Daily series for the last 14 days
    series = rows_to_list(conn.execute(f"""
        SELECT DATE(created_at, 'localtime') as date, event, COUNT(*) as count
        FROM metrics
        WHERE event IN ({placeholders})
        AND created_at >= datetime('now', '-14 days')
        GROUP BY DATE(created_at, 'localtime'), event ORDER BY date
    """, ENFORCEMENT_EVENTS).fetchall())

    # L5 satisfaction ratio (the key health metric)
    detected = totals_map.get("thin_retrieval_detected", 0)
    staged = totals_map.get("thin_retrieval_escalation_staged", 0)
    satisfied = totals_map.get("thin_retrieval_escalation_satisfied", 0)
    abandoned = totals_map.get("thin_retrieval_escalation_abandoned", 0)
    l5_ratio = round(satisfied / (satisfied + abandoned), 3) if (satisfied + abandoned) > 0 else None

    # Recent abandoned escalations (with context_need so user can see what got ignored)
    abandoned_recent = rows_to_list(conn.execute("""
        SELECT detail, value, created_at
        FROM metrics WHERE event = 'thin_retrieval_escalation_abandoned'
        ORDER BY created_at DESC LIMIT 20
    """).fetchall())

    # Recent phoned-in detections
    phoned_in_recent = rows_to_list(conn.execute("""
        SELECT detail, created_at
        FROM metrics WHERE event = 'context_phoned_in'
        ORDER BY created_at DESC LIMIT 20
    """).fetchall())

    # Recent active bootstrap triggers
    active_bootstrap_recent = rows_to_list(conn.execute("""
        SELECT detail, created_at
        FROM metrics WHERE event = 'active_bootstrap_triggered'
        ORDER BY created_at DESC LIMIT 20
    """).fetchall())

    # Trailing intent health
    intent_blocked = totals_map.get("trailing_intent_blocked", 0)
    intent_detected = totals_map.get("trailing_intent_detected", 0)
    intent_clear = totals_map.get("trailing_intent_clear", 0)
    intent_skipped = totals_map.get("trailing_intent_skipped", 0)
    intent_escaped = totals_map.get("trailing_intent_resolved_escape", 0)
    intent_total_checked = intent_detected + intent_clear
    intent_detection_rate = round(intent_detected / intent_total_checked, 3) if intent_total_checked > 0 else None

    # Recent trailing intent blocks
    intent_recent = rows_to_list(conn.execute("""
        SELECT detail, value, created_at
        FROM metrics WHERE event = 'trailing_intent_blocked'
        ORDER BY created_at DESC LIMIT 20
    """).fetchall())

    # Checkpoint health
    nudge_count = totals_map.get("checkpoint_nudge", 0)
    notes_count = totals_map.get("memory_notes_stored", 0)
    nudge_by_tool = rows_to_list(conn.execute("""
        SELECT SUBSTR(detail, 1, INSTR(detail, ':') - 1) as tool,
               COUNT(*) as count
        FROM metrics WHERE event = 'checkpoint_nudge'
        GROUP BY tool ORDER BY count DESC
    """).fetchall()) if nudge_count > 0 else []

    # Context recovery — LLM proactively requesting full conversation behind memories
    recovery_count = totals_map.get("context_recovery_invoked", 0)
    recovery_recent = rows_to_list(conn.execute("""
        SELECT detail as memory_id, created_at
        FROM metrics WHERE event = 'context_recovery_invoked'
        ORDER BY created_at DESC LIMIT 20
    """).fetchall()) if recovery_count > 0 else []

    conn.close()
    return {
        "totals": totals_map,
        "series": series,
        "l5_health": {
            "detected": detected,
            "staged": staged,
            "satisfied": satisfied,
            "abandoned": abandoned,
            "satisfaction_ratio": l5_ratio,
        },
        "trailing_intent": {
            "blocked": intent_blocked,
            "detected": intent_detected,
            "clear": intent_clear,
            "skipped": intent_skipped,
            "escaped": intent_escaped,
            "detection_rate": intent_detection_rate,
            "recent": intent_recent,
        },
        "checkpoints": {
            "nudges": nudge_count,
            "notes_stored": notes_count,
            "by_tool": nudge_by_tool,
        },
        "context_recovery": {
            "count": recovery_count,
            "recent": recovery_recent,
        },
        "recent": {
            "abandoned": abandoned_recent,
            "phoned_in": phoned_in_recent,
            "active_bootstrap": active_bootstrap_recent,
        },
    }, 200


_token_cache: dict = {}


def api_token_stats(params):
    conn = get_conn()
    sessions = conn.execute("SELECT session_id, transcript_path, project FROM sessions").fetchall()
    conn.close()
    total_user, total_assistant = 0, 0
    by_project: dict = {}
    for s in sessions:
        sid = s["session_id"]
        if sid not in _token_cache:
            _token_cache[sid] = _estimate_tokens(s["transcript_path"])
        t = _token_cache[sid]
        total_user += t["user_tokens"]
        total_assistant += t["assistant_tokens"]
        proj = s["project"] or "global"
        if proj not in by_project:
            by_project[proj] = {"user_tokens": 0, "assistant_tokens": 0, "sessions": 0}
        by_project[proj]["user_tokens"] += t["user_tokens"]
        by_project[proj]["assistant_tokens"] += t["assistant_tokens"]
        by_project[proj]["sessions"] += 1
    return {
        "total_user_tokens": total_user, "total_assistant_tokens": total_assistant,
        "total_tokens": total_user + total_assistant, "by_project": by_project,
        "note": "Estimated from transcript text (~4 chars/token). Excludes system prompts, tool metadata, and thinking tokens.",
    }, 200


def api_memory_usage(params):
    conn = get_conn()
    generated = rows_to_list(conn.execute("""
        SELECT s.session_id, s.project, s.started_at, COUNT(m.id) as generated
        FROM sessions s LEFT JOIN memories m ON m.session_id = s.session_id
        GROUP BY s.session_id HAVING generated > 0 ORDER BY s.started_at DESC
    """).fetchall())
    retrieved_rows = conn.execute("""
        SELECT session_id, value FROM hook_state
        WHERE key = 'retrieved_ids' AND value IS NOT NULL AND value != ''
    """).fetchall()
    retrieval_by_session: dict = {}
    memory_retrieval_count: Counter = Counter()
    for r in retrieved_rows:
        try:
            ids = json.loads(r["value"])
            if ids:
                retrieval_by_session[r["session_id"]] = len(ids)
                for mid in ids:
                    memory_retrieval_count[mid] += 1
        except Exception:
            pass
    for s in generated:
        s["retrieved"] = retrieval_by_session.get(s["session_id"], 0)
    top_retrieved = []
    if memory_retrieval_count:
        top_ids = memory_retrieval_count.most_common(20)
        id_list = [mid for mid, _ in top_ids]
        placeholders = ",".join("?" * len(id_list))
        mem_rows = conn.execute(f"""
            SELECT id, type, topic, content, confidence, project
            FROM memories WHERE id IN ({placeholders})
        """, id_list).fetchall()
        mem_map = {r["id"]: dict(r) for r in mem_rows}
        for mid, count in top_ids:
            m = mem_map.get(mid, {"id": mid, "type": "?", "topic": "(deleted)", "content": "", "confidence": 0, "project": ""})
            m["retrieval_count"] = count
            top_retrieved.append(m)
    delivery_rows = conn.execute("""
        SELECT detail FROM metrics WHERE event = 'layer_delivery' AND detail IS NOT NULL
    """).fetchall()
    layer_counts: dict = {}
    for row in delivery_rows:
        try:
            d = json.loads(row["detail"])
            layer = d.get("layer", "?")
            layer_counts[layer] = layer_counts.get(layer, 0) + len(d.get("ids", []))
        except Exception:
            pass
    if not layer_counts:
        legacy_layers = conn.execute("""
            SELECT event, COUNT(*) as count, COALESCE(SUM(value), 0) as total FROM metrics
            WHERE event IN ('layer1_5_injected','project_bootstrap_injected','context_served','layer2_staged')
            GROUP BY event
        """).fetchall()
        legacy_map = {"layer1_5_injected": "L1.5", "project_bootstrap_injected": "bootstrap",
                      "context_served": "L3", "layer2_staged": "L2"}
        for r in legacy_layers:
            layer_counts[legacy_map.get(r["event"], r["event"])] = int(r["total"]) or r["count"]
    all_ids_set = set(memory_retrieval_count.keys())
    never_count = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id NOT IN ({})".format(
            ",".join("?" * len(all_ids_set)) if all_ids_set else "NULL"),
        list(all_ids_set) if all_ids_set else []
    ).fetchone()[0]
    total_memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    total_retrieved = len(all_ids_set)
    total_retrieval_events = sum(memory_retrieval_count.values())
    outcomes = rows_to_list(conn.execute("""
        SELECT event, COUNT(*) as count FROM metrics
        WHERE event IN ('retrieval_useful', 'retrieval_neutral', 'retrieval_harmful') GROUP BY event
    """).fetchall())

    # Retrieval quality funnel: context_need → requested → served → outcome
    funnel_events = (
        "context_requested", "context_served", "context_cache_hit",
        "context_prefiltered", "context_weak_suppressed",
        "retrieval_useful", "retrieval_neutral", "retrieval_harmful",
        "gotcha_injected", "file_context_injected", "project_bootstrap_injected",
    )
    funnel_rows = conn.execute("""
        SELECT event, COUNT(*) as count FROM metrics
        WHERE event IN ({}) GROUP BY event
    """.format(",".join("?" * len(funnel_events))), funnel_events).fetchall()
    funnel = {r["event"]: r["count"] for r in funnel_rows}

    conn.close()
    return {
        "summary": {
            "total_memories": total_memories, "unique_retrieved": total_retrieved,
            "never_retrieved": never_count,
            "retrieval_rate": round(total_retrieved / total_memories * 100, 1) if total_memories else 0,
            "total_retrieval_events": total_retrieval_events,
            "outcomes": {o["event"].replace("retrieval_", ""): o["count"] for o in outcomes},
        },
        "top_retrieved": top_retrieved, "layer_counts": layer_counts, "sessions": generated[:100],
        "funnel": funnel,
    }, 200


def api_projects(params):
    conn = get_conn()
    projects = rows_to_list(conn.execute("""
        SELECT project, COUNT(*) as memory_count, MAX(updated_at) as last_updated
        FROM memories WHERE project IS NOT NULL AND project != ''
        GROUP BY project ORDER BY memory_count DESC
    """).fetchall())
    conn.close()
    return {"projects": projects}, 200


def api_config_get(params):
    # Each setting's override key is the EXACT env var it reads (config derives
    # this map by parsing its own environ.get calls); fall back to the attr name.
    env_keys = getattr(config, "CONFIG_ENV_KEYS", {})
    params_out = {}
    for name in sorted(dir(config)):
        if name.startswith("_") or not name.isupper() or name == "CONFIG_ENV_KEYS":
            continue
        val = getattr(config, name)
        # Scalars shown as-is; string lists rendered as CSV; None as empty string.
        if isinstance(val, (int, float, bool, str)):
            disp, typ = val, type(val).__name__
        elif isinstance(val, list) and all(isinstance(x, str) for x in val):
            disp, typ = ",".join(val), "csv"
        elif val is None:
            disp, typ = "", "str"
        else:
            continue
        params_out[name] = {
            "value": disp, "type": typ,
            "env_var": env_keys.get(name, name),
            "env_overridable": name in env_keys,
        }
    overrides = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    overrides[key.strip()] = val.strip()
    for name, info in params_out.items():
        env_key = info["env_var"]
        info["is_overridden"] = env_key in overrides
        if info["is_overridden"]:
            info["override"] = overrides[env_key]
    return {"config": params_out}, 200


def api_config_update(body):
    data = body
    if not data or "updates" not in data:
        return {"error": "Missing 'updates' field"}, 400
    existing = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    existing[key.strip()] = val.strip()
    env_keys = getattr(config, "CONFIG_ENV_KEYS", {})
    for name, value in data["updates"].items():
        env_key = env_keys.get(name, name)
        if value is None:
            existing.pop(env_key, None)
        else:
            existing[env_key] = str(value)
    with open(ENV_PATH, "w") as f:
        f.write("# Cairn config overrides — managed by dashboard\n")
        f.write("# These override defaults in cairn/config.py\n")
        for key, val in sorted(existing.items()):
            f.write(f"{key}={val}\n")
    return {"status": "saved", "overrides": existing}, 200


# ─── Helpers ────────────────────────────────────────────────────────────────

def _count_interactions(transcript_path):
    if not transcript_path or not os.path.exists(transcript_path):
        return 0
    from hooks.transcript_adapter import iter_normalized_entries
    count = 0
    for entry in iter_normalized_entries(transcript_path):
        msg = entry.get("message", {})
        if isinstance(msg, dict) and msg.get("role") == "user":
            count += 1
    return count


def _estimate_tokens(transcript_path):
    if not transcript_path or not os.path.exists(transcript_path):
        return {"user_tokens": 0, "assistant_tokens": 0}
    from hooks.transcript_adapter import iter_normalized_entries
    user_chars, assistant_chars = 0, 0
    for entry in iter_normalized_entries(transcript_path):
        msg = entry.get("message", {})
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            chars = sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
        elif isinstance(content, str):
            chars = len(content)
        else:
            continue
        if role == "user":
            user_chars += chars
        elif role == "assistant":
            assistant_chars += chars
    return {"user_tokens": user_chars // 4, "assistant_tokens": assistant_chars // 4}


def _extract_session_dir(transcript_path, cwd=None):
    """Return the human-readable project dir name.
    Prefers basename of already-decoded cwd; falls back to encoded dir parsing."""
    if cwd:
        return os.path.basename(cwd.rstrip('/')) or None
    if not transcript_path:
        return None
    # Decode the ~/.claude/projects/<encoded> directory name
    m = re.search(r'/\.claude/projects/([^/]+)/', transcript_path)
    if not m:
        return None
    encoded = m.group(1)  # e.g. -home-jameo-Projects-cairn or -home-x-Projects-tezuka-fw
    # Strip common path prefixes to recover project name (preserves hyphens in name)
    parts = [p for p in encoded.split('-') if p]
    # Drop known prefix segments: home/mnt + username + common dirs (Projects, work, src)
    skip = {'home', 'mnt', 'root', 'var', 'usr', 'opt', 'srv', 'projects', 'project',
            'work', 'src', 'dev', 'repos', 'code', 'git'}
    # Keep stripping: path components, then username, then common dirs
    while len(parts) > 1:
        p = parts[0].lower()
        if p in skip:
            parts = parts[1:]
        elif p.isalpha() and len(p) <= 12 and parts[1].lower() in skip | {'projects'}:
            parts = parts[1:]  # skip username
        else:
            break
    return '-'.join(parts) if parts else encoded


def _extract_cwd(transcript_path):
    if not transcript_path:
        return None
    # Copilot transcripts live under workspaceStorage/<hash>/GitHub.copilot-chat/
    # and don't encode cwd in the path. Try the session.start record's
    # data.context.cwd field (populated when Copilot supplies workspace context).
    if "GitHub.copilot-chat" in transcript_path or "/transcripts/" in transcript_path:
        try:
            with open(transcript_path, encoding="utf-8") as f:
                first = f.readline().strip()
            if first:
                rec = json.loads(first)
                if rec.get("type") == "session.start":
                    ctx = (rec.get("data") or {}).get("context") or {}
                    cwd = ctx.get("cwd") if isinstance(ctx, dict) else None
                    if cwd:
                        return cwd
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        # Copilot transcript with no cwd context — nothing to derive
        if "/.claude/projects/" not in transcript_path:
            return None
    m = re.search(r'/\.claude/projects/([^/]+)/', transcript_path)
    if not m:
        return None
    encoded = m.group(1)
    projects_dir = os.path.expanduser("~/.claude/projects/")
    if os.path.isdir(os.path.join(projects_dir, encoded)):
        if not encoded.startswith("-"):
            return encoded
        parts = encoded[1:].split("-")
        path = "/"
        i = 0
        while i < len(parts):
            found = False
            for j in range(len(parts), i, -1):
                candidate = "-".join(parts[i:j])
                test_path = os.path.join(path, candidate)
                if os.path.exists(test_path):
                    path = test_path
                    i = j
                    found = True
                    break
            if not found:
                path = os.path.join(path, parts[i])
                i += 1
        return path
    if encoded.startswith("-"):
        return "/" + encoded[1:].replace("-", "/")
    return encoded.replace("-", "/")


def _parse_transcript(transcript_path):
    from hooks.transcript_adapter import iter_normalized_entries
    messages = []
    for line_num, entry in enumerate(iter_normalized_entries(transcript_path)):
        try:
            ts_str = entry.get("timestamp", "")
            msg = entry.get("message", entry)
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(b.get("text", "") for b in content
                                    if isinstance(b, dict) and b.get("type") == "text")
            elif not isinstance(content, str):
                continue
            if not content or not content.strip():
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, AttributeError):
                ts = None
            messages.append({"line": line_num, "role": role, "text": content, "ts": ts})
        except Exception:
            continue
    return messages


# ─── Retention & Health APIs ──────────────────────────────────────────────


def api_retention(params):
    conn = get_conn()
    total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    sessions_with_path = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE transcript_path IS NOT NULL AND transcript_path != ''"
    ).fetchone()[0]
    transcripts_on_disk = 0
    total_bytes = 0
    rows = conn.execute("SELECT transcript_path FROM sessions WHERE transcript_path IS NOT NULL").fetchall()
    for r in rows:
        p = r[0]
        if p and os.path.exists(p):
            transcripts_on_disk += 1
            total_bytes += os.path.getsize(p)

    total_memories = conn.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL").fetchone()[0]
    try:
        with_excerpt = conn.execute("SELECT COUNT(*) FROM memory_source_excerpt").fetchone()[0]
    except Exception:
        with_excerpt = 0
    excerpt_pct = round(with_excerpt / max(total_memories, 1) * 100, 1)

    age_buckets = {"0-7d": 0, "7-30d": 0, "30-90d": 0, "90d+": 0}
    age_rows = conn.execute(
        "SELECT julianday('now') - julianday(started_at) AS age FROM sessions"
    ).fetchall()
    for r in age_rows:
        age = r[0] or 0
        if age <= 7:
            age_buckets["0-7d"] += 1
        elif age <= 30:
            age_buckets["7-30d"] += 1
        elif age <= 90:
            age_buckets["30-90d"] += 1
        else:
            age_buckets["90d+"] += 1
    conn.close()

    return {
        "total_sessions": total_sessions,
        "sessions_with_transcript": transcripts_on_disk,
        "sessions_transcript_purged": sessions_with_path - transcripts_on_disk,
        "total_transcript_bytes": total_bytes,
        "age_histogram": [{"bucket": k, "count": v} for k, v in age_buckets.items()],
        "excerpt_coverage_pct": excerpt_pct,
        "memories_with_excerpt": with_excerpt,
        "total_memories": total_memories,
    }, 200


def api_session_triage(params):
    conn = get_conn()
    limit = int(params.get("limit", "50"))
    offset = int(params.get("offset", "0"))
    sort = params.get("sort", "value_score")
    order = params.get("order", "asc")

    rows = conn.execute("""
        SELECT s.session_id, s.project, s.started_at, s.transcript_path,
               COUNT(m.id) AS memory_count
        FROM sessions s
        LEFT JOIN memories m ON m.session_id = s.session_id AND m.deleted_at IS NULL
        GROUP BY s.session_id
    """).fetchall()

    results = []
    for r in rows:
        sid = r[0]
        path = r[3]
        exists = bool(path and os.path.exists(path))
        size = os.path.getsize(path) if exists else 0
        try:
            excerpt_count = conn.execute(
                "SELECT COUNT(*) FROM memory_source_excerpt WHERE session_id = ?", (sid,)
            ).fetchone()[0]
        except Exception:
            excerpt_count = 0
        mem_count = r[4]
        results.append({
            "session_id": sid,
            "project": r[1],
            "created_at": r[2],
            "memory_count": mem_count,
            "value_score": 0,
            "transcript_exists": exists,
            "transcript_bytes": size,
            "excerpt_coverage": round(excerpt_count / max(mem_count, 1) * 100, 1),
        })

    rev = order == "desc"
    results.sort(key=lambda x: x.get(sort, 0) or 0, reverse=rev)
    conn.close()
    return {"sessions": results[offset:offset + limit], "total": len(results)}, 200


def _get_eph_conn():
    """Open ephemeral DB connection for calibration_deliveries / metrics."""
    eph_path = getattr(config, "EPHEMERAL_DB_PATH", None) or \
        os.path.join(os.path.dirname(DB_PATH), "cairn-ephemeral.db")
    conn = sqlite3.connect(eph_path)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def api_calibration_profile(params):
    """List active calibration rows for the Profile panel.

    Query params:
        limit (int, default 50)
        offset (int, default 0)
        sort (str, one of: pinned, source, id, delivered_count, confidence,
              updated_at; default "pinned")
        order (asc|desc, default desc)
    """
    sort_col = params.get("sort", "pinned")
    if sort_col not in ("pinned", "source", "id", "delivered_count",
                         "confidence", "updated_at", "created_at",
                         "follow_rate"):
        sort_col = "pinned"
    order = params.get("order", "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    try:
        limit = max(1, min(500, int(params.get("limit", 50))))
    except (TypeError, ValueError):
        limit = 50
    try:
        offset = max(0, int(params.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0

    # follow_rate is computed; emulate via expression in ORDER BY
    if sort_col == "follow_rate":
        order_expr = ("CASE WHEN delivered_count > 0 THEN "
                      "followed_count * 1.0 / delivered_count ELSE -1 END "
                      + order.upper())
    elif sort_col == "pinned":
        # Pin-first then by delivered_count desc, id asc as tiebreaker
        order_expr = (f"pinned {order.upper()}, delivered_count DESC, "
                       "source, id")
    else:
        order_expr = f"{sort_col} {order.upper()}, id ASC"

    conn = get_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM calibration_rows "
            "WHERE archived_at IS NULL AND superseded_by IS NULL"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT id, content, kw, qf, source, confidence, pinned, layer, "
            "delivered_count, followed_count, ignored_count, corrected_count, "
            "created_at, updated_at "
            "FROM calibration_rows "
            "WHERE archived_at IS NULL AND superseded_by IS NULL "
            f"ORDER BY {order_expr} "
            "LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        by_source = dict(conn.execute(
            "SELECT source, COUNT(*) FROM calibration_rows "
            "WHERE archived_at IS NULL AND superseded_by IS NULL "
            "GROUP BY source"
        ).fetchall())
    finally:
        conn.close()
    out = []
    for r in rows:
        d = {"id": r[0], "content": r[1], "kw": r[2], "qf": r[3],
             "source": r[4], "confidence": r[5], "pinned": bool(r[6]),
             "layer": r[7], "delivered_count": r[8], "followed_count": r[9],
             "ignored_count": r[10], "corrected_count": r[11],
             "created_at": r[12], "updated_at": r[13]}
        d["follow_rate"] = (r[9] / r[8]) if r[8] else None
        out.append(d)
    return {"rows": out, "total": total, "by_source": by_source,
            "limit": limit, "offset": offset,
            "sort": sort_col, "order": order}, 200


def api_calibration_effectiveness(params):
    """Per-row effectiveness table + aggregate counts."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, content, source, delivered_count, followed_count, "
            "ignored_count, corrected_count "
            "FROM calibration_rows WHERE archived_at IS NULL "
            "AND delivered_count > 0 "
            "ORDER BY delivered_count DESC LIMIT 200"
        ).fetchall()
    finally:
        conn.close()
    per_row = []
    total_d = total_f = total_i = total_c = 0
    flagged_low = []
    for rid, content, source, dc, fc, ic, cc in rows:
        rate = fc / dc if dc else 0.0
        per_row.append({"id": rid, "content": content, "source": source,
                        "delivered": dc, "followed": fc, "ignored": ic,
                        "corrected": cc, "follow_rate": round(rate, 3)})
        total_d += dc; total_f += fc; total_i += ic; total_c += cc
        if dc >= 10 and rate < 0.20:
            flagged_low.append(rid)
    return {
        "per_row": per_row,
        "aggregate": {
            "delivered": total_d, "followed": total_f,
            "ignored": total_i, "corrected": total_c,
            "overall_follow_rate": round(total_f / total_d, 3) if total_d else None,
        },
        "flagged_low_follow": flagged_low,
    }, 200


def api_calibration_review_queue(params):
    """Unresolved Tier 2 review-queue items for the Review panel."""
    conn = get_conn()
    try:
        try:
            rows = conn.execute(
                "SELECT rq.id, rq.row_id, rq.suggestion_type, rq.detail, "
                "rq.surfaced_at, cr.content "
                "FROM calibration_review_queue rq "
                "LEFT JOIN calibration_rows cr ON cr.id = rq.row_id "
                "WHERE rq.resolved_at IS NULL "
                "ORDER BY rq.surfaced_at DESC"
            ).fetchall()
        except sqlite3.OperationalError:
            return {"items": [], "total": 0}, 200
    finally:
        conn.close()
    out = [{"item_id": r[0], "row_id": r[1], "suggestion_type": r[2],
            "detail": r[3], "surfaced_at": r[4], "content": r[5]}
           for r in rows]
    return {"items": out, "total": len(out)}, 200


def api_calibration_row(params, row_id):
    """Detail view for a single calibration_row — full content, all
    metadata, and the calibration_deliveries history grouped by session."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id, content, kw, qf, source, confidence, pinned, layer, "
            "session_scope, superseded_by, archived_at, archive_reason, "
            "delivered_count, followed_count, ignored_count, "
            "corrected_count, created_at, updated_at, origin_session_id "
            "FROM calibration_rows WHERE id = ?", (row_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {"error": "not found"}, 404
    detail = {
        "id": row[0], "content": row[1], "kw": row[2], "qf": row[3],
        "source": row[4], "confidence": row[5], "pinned": bool(row[6]),
        "layer": row[7], "session_scope": row[8], "superseded_by": row[9],
        "archived_at": row[10], "archive_reason": row[11],
        "delivered_count": row[12], "followed_count": row[13],
        "ignored_count": row[14], "corrected_count": row[15],
        "created_at": row[16], "updated_at": row[17],
        "origin_session_id": row[18],
    }
    detail["follow_rate"] = (row[13] / row[12]) if row[12] else None

    # Deliveries grouped by session
    eph = _get_eph_conn()
    try:
        rows = eph.execute(
            "SELECT session_id, turn_index, delivered_at, similarity, "
            "outcome, outcome_evidence "
            "FROM calibration_deliveries WHERE row_id = ? "
            "ORDER BY delivered_at DESC LIMIT 200",
            (row_id,)
        ).fetchall()
    finally:
        eph.close()
    deliveries_by_session = {}
    for sid, turn, at, sim, outcome, evidence in rows:
        deliveries_by_session.setdefault(sid, []).append({
            "turn": turn, "delivered_at": at, "similarity": sim,
            "outcome": outcome, "outcome_evidence": evidence,
        })
    detail["deliveries_by_session"] = [
        {"session_id": sid, "deliveries": d}
        for sid, d in deliveries_by_session.items()
    ]
    return detail, 200


def api_calibration_session(params, session_id):
    """Per-session calibration delivery timeline + outcomes."""
    eph = _get_eph_conn()
    try:
        deliveries = eph.execute(
            "SELECT id, turn_index, row_id, delivered_at, similarity, "
            "outcome, outcome_evidence "
            "FROM calibration_deliveries WHERE session_id = ? "
            "ORDER BY turn_index, id",
            (session_id,),
        ).fetchall()
    finally:
        eph.close()
    if not deliveries:
        return {"session_id": session_id, "deliveries": [], "total": 0}, 200
    row_ids = list({d[2] for d in deliveries})
    placeholders = ",".join("?" * len(row_ids))
    conn = get_conn()
    try:
        crows = conn.execute(
            f"SELECT id, content, source, confidence FROM calibration_rows "
            f"WHERE id IN ({placeholders})", row_ids,
        ).fetchall()
    finally:
        conn.close()
    row_info = {c[0]: {"content": c[1], "source": c[2],
                       "confidence": c[3]} for c in crows}
    timeline = []
    for did, turn, rid, at, sim, outcome, evidence in deliveries:
        timeline.append({
            "delivery_id": did, "turn": turn, "row_id": rid,
            "delivered_at": at, "similarity": sim, "outcome": outcome,
            "outcome_evidence": evidence,
            "row": row_info.get(rid, {}),
        })
    return {"session_id": session_id, "deliveries": timeline,
            "total": len(timeline)}, 200


def api_health(params):
    from hooks.health import sentinel_info
    info = sentinel_info()
    return {
        "impaired": info is not None,
        "info": info,
        "topic_embedding": _topic_embedding_coverage(),
        "exact_duplicates": _exact_duplicate_groups(),
    }, 200


# Topic name used by the cairn self-test; excluded from duplicate detection so a
# repeatedly-run smoke test doesn't masquerade as a double-insert regression.
_SMOKE_TEST_TOPIC = "cairn-smoke-test"


def _topic_embedding_coverage():
    """Dual-embedding (v8 topic_embedding) coverage over live memories.

    Drift below 100% means the topic_embedding backfill hasn't kept up with
    content edits / inserts; <95% is flagged so symmetric topic retrieval doesn't
    silently degrade."""
    conn = get_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"
        ).fetchone()[0]
        with_topic = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL AND topic_embedding IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    pct = (with_topic / total * 100.0) if total else 100.0
    return {"with_topic_embedding": with_topic, "total": total,
            "pct": round(pct, 1), "warn": pct < 95.0}


def _exact_duplicate_groups():
    """Count groups of live memories sharing identical (topic, content).

    Non-zero is a defensive signal for double-insert-class regressions (e.g. the
    dual-embedding + sync-merge double-insert fixed in a871775). The smoke-test
    topic is excluded so self-test churn doesn't inflate the count."""
    conn = get_conn()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM memories "
            "WHERE deleted_at IS NULL AND topic != ? "
            "GROUP BY topic, content HAVING COUNT(*) > 1)",
            (_SMOKE_TEST_TOPIC,)
        ).fetchone()[0]
    finally:
        conn.close()
    return {"count": count, "warn": count > 0}


def api_snapshot_session(params, session_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT m.id, m.session_id, s.transcript_path FROM memories m "
        "JOIN sessions s ON s.session_id = m.session_id "
        "WHERE m.session_id = ? AND m.deleted_at IS NULL "
        "AND m.id NOT IN (SELECT memory_id FROM memory_source_excerpt)",
        (session_id,)
    ).fetchall()
    if not rows:
        conn.close()
        return {"snapshotted": 0, "skipped": 0, "reason": "no memories without excerpts"}, 200

    transcript_path = rows[0][2] if rows else None
    excerpt = ""
    if transcript_path and os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r") as f:
                lines = f.readlines()
            excerpt = "".join(lines[-60:]) if len(lines) > 60 else "".join(lines)
        except Exception:
            pass

    if not excerpt:
        conn.close()
        return {"snapshotted": 0, "skipped": len(rows), "reason": "transcript not available"}, 200

    count = 0
    for r in rows:
        conn.execute(
            "INSERT OR IGNORE INTO memory_source_excerpt (memory_id, session_id, transcript_path, excerpt) "
            "VALUES (?, ?, ?, ?)",
            (r[0], session_id, transcript_path or "", excerpt[:10000])
        )
        count += 1
    conn.commit()
    conn.close()
    return {"snapshotted": count, "skipped": 0}, 200


# ─── Code-graph fleet + explorer ──────────────────────────────────────────────
# Observability + navigation over the per-repo code-review-graph databases that
# cairn keeps fresh across the whole repo fleet (see cairn/graph_fleet.py).

import subprocess as _subprocess


def _discovered_repos():
    """All git repos the fleet manages, as realpaths. Empty list if the fleet
    module / graph tooling isn't importable (fail-open)."""
    try:
        from cairn import graph_fleet
        return graph_fleet.discover_repos()
    except Exception:
        return []


def _resolve_fleet_repo(repo):
    """Map a repo path/alias from the client to a managed repo realpath, or None.

    Guards the visualize/explorer endpoints against running tooling on arbitrary
    paths: the input must resolve to a repo the fleet actually manages."""
    if not repo:
        return None
    repos = _discovered_repos()
    target = os.path.realpath(os.path.expanduser(repo))
    if target in repos:
        return target
    # Allow addressing by basename alias (unique match only).
    by_alias = [r for r in repos if os.path.basename(r) == repo]
    return by_alias[0] if len(by_alias) == 1 else None


def _git_head_sha(repo):
    try:
        r = _subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, _subprocess.SubprocessError):
        return None


def _repo_graph_stats(repo):
    """Read a repo's .code-review-graph/graph.db: nodes, edges, languages,
    last_updated, recorded HEAD sha. Returns None if no graph DB present."""
    db = os.path.join(repo, ".code-review-graph", "graph.db")
    if not os.path.exists(db):
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
        conn.execute("PRAGMA busy_timeout=2000")
        nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        langs = [r[0] for r in conn.execute(
            "SELECT DISTINCT language FROM nodes WHERE language IS NOT NULL ORDER BY language"
        ).fetchall()]
        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        conn.close()
        return {
            "nodes": nodes, "edges": edges, "languages": langs,
            "last_updated": meta.get("last_updated"),
            "graph_head_sha": meta.get("git_head_sha"),
            "db_mtime": os.path.getmtime(db),
        }
    except Exception:
        return {"nodes": None, "edges": None, "languages": [],
                "last_updated": None, "graph_head_sha": None, "locked": True}


def _repo_state(repo, stats):
    """fresh | stale | missing for a repo's graph.

    Stale = the graph was built against a different HEAD than the repo is on now
    (cheap, precise). Falls back to db-mtime vs HEAD commit time when sha is
    unavailable."""
    if stats is None:
        return "missing"
    if stats.get("locked"):
        return "stale"
    head = _git_head_sha(repo)
    graph_head = stats.get("graph_head_sha")
    if head and graph_head:
        return "fresh" if head == graph_head else "stale"
    # Fallback: compare graph mtime to HEAD commit time.
    try:
        r = _subprocess.run(["git", "-C", repo, "log", "-1", "--format=%ct"],
                            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            head_time = float(r.stdout.strip())
            return "fresh" if (stats.get("db_mtime") or 0) >= head_time else "stale"
    except (OSError, _subprocess.SubprocessError, ValueError):
        pass
    return "fresh"


def api_graph_fleet(params):
    """Fleet observability: coverage + per-repo graph stats + last sweep + daemon."""
    repos = _discovered_repos()
    per_repo = []
    counts = {"with_graph": 0, "missing": 0, "stale": 0, "fresh": 0}
    for repo in repos:
        stats = _repo_graph_stats(repo)
        state = _repo_state(repo, stats)
        counts[state if state in counts else "fresh"] += 1
        if stats is not None:
            counts["with_graph"] += 1
        per_repo.append({
            "alias": os.path.basename(repo), "path": repo,
            "nodes": (stats or {}).get("nodes"),
            "edges": (stats or {}).get("edges"),
            "languages": (stats or {}).get("languages", []),
            "last_updated": (stats or {}).get("last_updated"),
            "state": state,
        })
    per_repo.sort(key=lambda r: ({"stale": 0, "missing": 1, "fresh": 2}.get(r["state"], 3),
                                 r["alias"]))

    # Last sweep summary written by graph_fleet.sweep() into ephemeral hook_state.
    last_sweep = None
    try:
        from hooks.hook_helpers import load_hook_state
        raw = load_hook_state("graph_fleet", "graph_fleet_last_sweep")
        if raw:
            last_sweep = json.loads(raw)
    except Exception:
        pass

    # Daemon status only when the opt-in real-time layer is enabled.
    daemon = "disabled"
    if os.environ.get("CAIRN_GRAPH_WATCH", "0").lower() in ("1", "true", "yes"):
        try:
            from cairn.repo_discovery import _resolve_crg
            crg = _resolve_crg()
            if crg:
                r = _subprocess.run([crg, "daemon", "status"],
                                    capture_output=True, text=True, timeout=15)
                daemon = (r.stdout + r.stderr).strip()[:500] or "unknown"
        except Exception:
            daemon = "error"

    coverage = {
        "total": len(repos), "with_graph": counts["with_graph"],
        "missing": counts["missing"], "stale": counts["stale"], "fresh": counts["fresh"],
    }
    return {"coverage": coverage, "repos": per_repo,
            "last_sweep": last_sweep, "daemon": daemon}, 200


def api_graph_explorer(params):
    """Per-repo graph navigator backed by cairn.graph.

    Params: repo (required, path or alias). Optional symbol — when given, returns
    a context pack (body + callers + callees + tests + cairn knowledge); otherwise
    a repo orientation block (modules / flows / hubs)."""
    repo = _resolve_fleet_repo(params.get("repo", ""))
    if not repo:
        return {"error": "unknown or unmanaged repo"}, 404
    if not os.path.exists(os.path.join(repo, ".code-review-graph", "graph.db")):
        return {"error": "no graph for this repo", "repo": repo}, 404
    try:
        from cairn import graph
        symbol = (params.get("symbol") or "").strip()
        if symbol:
            return {"repo": repo, "symbol": symbol,
                    "context_pack": graph.context_pack(symbol, repo_root=repo)}, 200
        return {"repo": repo, "orientation": graph.orientation_block(repo_root=repo)}, 200
    except Exception as e:
        return {"error": str(e), "repo": repo}, 500


def _generate_graph_viz(repo):
    """Regenerate the community-mode interactive HTML for a repo and return its
    path. Community mode is the only sane default — full mode never settles in
    the browser at fleet scale. Returns None on failure."""
    try:
        from cairn.repo_discovery import _resolve_crg
        crg = _resolve_crg()
        if not crg:
            return None
        r = _subprocess.run(
            [crg, "visualize", "--repo", repo, "--mode", "community", "--format", "html"],
            capture_output=True, text=True, timeout=120)
        html = os.path.join(repo, ".code-review-graph", "graph.html")
        return html if (r.returncode == 0 and os.path.exists(html)) else None
    except (OSError, _subprocess.SubprocessError):
        return None


# ─── HTTP Server ────────────────────────────────────────────────────────────

# ─── Sync (v2 peer-to-peer) ───────────────────────────────────────────────────
# Management surface for the Sync dashboard tab: identity, discovered peers,
# pairing-request queue (approve/deny), paired peers (revoke), outbound pairing.

def api_sync_identity(params):
    from cairn.sync import identity, SCHEMA_VERSION
    return {
        "node_id": identity.get_node_fingerprint(),
        "user_id": identity.get_user_id(),
        "public_key": identity.get_public_key_b64(),
        "schema_version": SCHEMA_VERSION,
    }, 200


def api_sync_pairing_requests(params):
    from cairn.sync import pairing
    conn = get_conn()
    try:
        return {"requests": pairing.list_pairing_requests(conn, pending_only=False)}, 200
    finally:
        conn.close()


def api_sync_peers(params):
    from cairn.sync import pairing
    conn = get_conn()
    try:
        return {"peers": pairing.list_peers(conn)}, 200
    finally:
        conn.close()


def api_sync_discovered(params):
    from cairn.sync import discovery
    conn = get_conn()
    try:
        return {"discovered": discovery.list_discovered(conn)}, 200
    finally:
        conn.close()


def api_sync_approve(params, request_id):
    from cairn.sync import pairing
    conn = get_conn()
    try:
        res = pairing.approve_pairing(conn, int(request_id))
        return res, (200 if res.get("ok") else 404)
    finally:
        conn.close()


def api_sync_deny(params, request_id):
    from cairn.sync import pairing
    conn = get_conn()
    try:
        res = pairing.deny_pairing(conn, int(request_id))
        return res, (200 if res.get("ok") else 404)
    finally:
        conn.close()


def api_sync_revoke(params, node_id):
    from cairn.sync import pairing
    conn = get_conn()
    try:
        res = pairing.revoke_peer(conn, node_id)
        return res, (200 if res.get("ok") else 404)
    finally:
        conn.close()


def api_sync_online(params):
    """Discovered peers seen within the online window, annotated with my pairing
    relationship (connected / requested / incoming / available / revoked)."""
    from cairn.sync import discovery, SCHEMA_VERSION, MIN_COMPATIBLE_SCHEMA_VERSION
    from cairn import config
    conn = get_conn()
    try:
        disc = discovery.list_discovered(conn, max_age_sec=config.CAIRN_SYNC_ONLINE_WINDOW)
        peers = {r[0]: (r[1] or "approved")
                 for r in conn.execute("SELECT peer_node_id, status FROM sync_peers")}
        out_pending = {r[0] for r in conn.execute(
            "SELECT peer_node_id FROM pairing_requests WHERE direction='outbound' "
            "AND status IN ('pending','already_paired')")}
        in_pending = {r[0] for r in conn.execute(
            "SELECT peer_node_id FROM pairing_requests WHERE direction='inbound' AND status='pending'")}
        for d in disc:
            nid = d["node_id"]
            sv = d.get("schema_version")
            # Incompatible if their advertised version is below our floor — we'd
            # refuse them. (Newer-than-us is fine: apply tolerates additive drift.)
            d["compatible"] = sv is not None and sv >= MIN_COMPATIBLE_SCHEMA_VERSION
            if not d["compatible"]:
                d["state"] = "incompatible"
            elif peers.get(nid) == "approved":
                d["state"] = "connected"
            elif peers.get(nid) == "revoked":
                d["state"] = "revoked"
            elif nid in out_pending:
                d["state"] = "requested"
            elif nid in in_pending:
                d["state"] = "incoming"
            else:
                d["state"] = "available"
        return {"online": disc, "our_schema": SCHEMA_VERSION,
                "min_compatible": MIN_COMPATIBLE_SCHEMA_VERSION}, 200
    finally:
        conn.close()


def api_sync_refresh(params):
    """Poll outbound-pending peers for approval and promote approved ones."""
    from cairn.sync import client
    conn = get_conn()
    try:
        return {"refreshed": client.refresh_outbound(conn)}, 200
    finally:
        conn.close()


def api_sync_pair(body):
    """Initiate an outbound pairing request to a peer URL (dashboard one-click).
    Records the request locally so it shows under 'my requests' until approved."""
    from cairn.sync import client
    from cairn.sync.service import lan_ip
    from cairn import config
    url = (body or {}).get("url")
    if not url:
        return {"ok": False, "error": "url required"}, 400
    my_url = (body or {}).get("my_url") or f"https://{lan_ip()}:{config.CAIRN_SYNC_PORT}"
    conn = get_conn()
    try:
        resp = client.send_pairing_request(url, my_url=my_url, conn=conn)
        return resp, (200 if resp.get("ok") else 502)
    finally:
        conn.close()


# ─── HTTP Server ──────────────────────────────────────────────────────────────

# Route table: (method, pattern) -> handler
# Patterns use {name} for path params, converted to regex groups
def _looks_like_timestamp(v: str) -> bool:
    """A stored UTC datetime like '2026-06-25 21:04:34' (not a date-only bucket)."""
    return (len(v) >= 16 and v[4] == "-" and v[7] == "-"
            and v[10] in " T" and ":" in v[11:16])


def _localize_timestamps(obj):
    """Recursively convert any '*_at' UTC timestamp field to a local-time display
    string so the dashboard shows local time everywhere. Storage is untouched —
    presentation only (cairn/timeutil; single source of truth). Date-only chart
    buckets are left alone (handled by DATE(...,'localtime') in their queries)."""
    from cairn import timeutil
    if isinstance(obj, dict):
        return {k: (timeutil.fmt_local(v) if isinstance(v, str) and k.endswith("_at")
                    and _looks_like_timestamp(v) else _localize_timestamps(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_localize_timestamps(x) for x in obj]
    return obj


_ROUTES: list[tuple[str, str, callable]] = [
    ("GET", "/api/stats", lambda p, **kw: api_stats(p)),
    ("GET", "/api/memories", lambda p, **kw: api_memories(p)),
    ("GET", "/api/memory/{memory_id}/context", lambda p, **kw: api_memory_context(p, int(kw["memory_id"]))),
    ("GET", "/api/memory/{memory_id}", lambda p, **kw: api_memory_detail(p, int(kw["memory_id"]))),
    ("GET", "/api/sessions", lambda p, **kw: api_sessions(p)),
    ("GET", "/api/session/{session_id}/transcript", lambda p, **kw: api_session_transcript(p, kw["session_id"])),
    ("GET", "/api/session/{session_id}", lambda p, **kw: api_session_detail(p, kw["session_id"])),
    ("GET", "/api/metrics", lambda p, **kw: api_metrics(p)),
    ("GET", "/api/enforcement", lambda p, **kw: api_enforcement(p)),
    ("GET", "/api/token-stats", lambda p, **kw: api_token_stats(p)),
    ("GET", "/api/memory-usage", lambda p, **kw: api_memory_usage(p)),
    ("GET", "/api/projects", lambda p, **kw: api_projects(p)),
    ("GET", "/api/config", lambda p, **kw: api_config_get(p)),
    ("GET", "/api/retention", lambda p, **kw: api_retention(p)),
    ("GET", "/api/session-triage", lambda p, **kw: api_session_triage(p)),
    ("GET", "/api/health", lambda p, **kw: api_health(p)),
    ("GET", "/api/calibration/profile", lambda p, **kw: api_calibration_profile(p)),
    ("GET", "/api/calibration/effectiveness", lambda p, **kw: api_calibration_effectiveness(p)),
    ("GET", "/api/calibration/review-queue", lambda p, **kw: api_calibration_review_queue(p)),
    ("GET", "/api/calibration/row/{row_id}", lambda p, **kw: api_calibration_row(p, int(kw["row_id"]))),
    ("GET", "/api/calibration/session/{session_id}", lambda p, **kw: api_calibration_session(p, kw["session_id"])),
    ("GET", "/api/graph-fleet", lambda p, **kw: api_graph_fleet(p)),
    ("GET", "/api/graph-explorer", lambda p, **kw: api_graph_explorer(p)),
    ("GET", "/api/sync/identity", lambda p, **kw: api_sync_identity(p)),
    ("GET", "/api/sync/pairing-requests", lambda p, **kw: api_sync_pairing_requests(p)),
    ("GET", "/api/sync/peers", lambda p, **kw: api_sync_peers(p)),
    ("GET", "/api/sync/discovered", lambda p, **kw: api_sync_discovered(p)),
    ("GET", "/api/sync/online", lambda p, **kw: api_sync_online(p)),
]

# Compile route patterns to regex
_COMPILED_ROUTES = []
for method, pattern, handler in _ROUTES:
    regex = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', pattern) + '$'
    _COMPILED_ROUTES.append((method, re.compile(regex), handler))


def _match_route(method, path):
    for route_method, regex, handler in _COMPILED_ROUTES:
        if method == route_method:
            m = regex.match(path)
            if m:
                return handler, m.groupdict()
    return None, {}


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logging

    def _send_json(self, data, status=200):
        body = json.dumps(_localize_timestamps(data), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_error(404)

    def _serve_graph_viz(self, params):
        repo = _resolve_fleet_repo(params.get("repo", ""))
        if not repo:
            self.send_error(404, "unknown or unmanaged repo")
            return
        html_path = _generate_graph_viz(repo)
        if not html_path:
            self.send_error(500, "could not generate visualization")
            return
        self._send_file(html_path, "text/html; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        # Static files
        if path == "/" or path == "/index.html":
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
            return

        # Interactive code-graph visualization (raw HTML, not JSON) — embedded
        # in an iframe by the Graph Explorer view.
        if path == "/api/graph-viz":
            self._serve_graph_viz(params)
            return

        # API routes
        handler, kwargs = _match_route("GET", path)
        if handler:
            try:
                data, status = handler(params, **kwargs)
                self._send_json(data, status)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

        if path == "/api/config":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            try:
                data, status = api_config_update(body)
                self._send_json(data, status)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        m = re.match(r"/api/snapshot-session/([^/]+)$", path)
        if m:
            try:
                data, status = api_snapshot_session(params, m.group(1))
                self._send_json(data, status)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        # Sync (v2) management POST routes
        sync_post = [
            (r"/api/sync/pairing-requests/(\d+)/approve$", api_sync_approve),
            (r"/api/sync/pairing-requests/(\d+)/deny$", api_sync_deny),
            (r"/api/sync/peers/([^/]+)/revoke$", api_sync_revoke),
        ]
        for pat, fn in sync_post:
            mm = re.match(pat, path)
            if mm:
                try:
                    data, status = fn(params, mm.group(1))
                    self._send_json(data, status)
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
                return
        if path == "/api/sync/refresh":
            try:
                data, status = api_sync_refresh(params)
                self._send_json(data, status)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return
        if path == "/api/sync/pair":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            try:
                data, status = api_sync_pair(body)
                self._send_json(data, status)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        self.send_error(404)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cairn Dashboard")
    parser.add_argument("--port", type=int, default=8420, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    if not args.no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    print(f"Cairn Dashboard: http://{args.host}:{args.port}")
    server = HTTPServer((args.host, args.port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
