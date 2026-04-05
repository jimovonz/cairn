#!/usr/bin/env python3
"""Cairn Dashboard — Web UI for monitoring and managing the memory system.

Zero external dependencies — uses Python stdlib http.server only.
"""

import json
import os
import re
import sqlite3
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

import config

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
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM memories GROUP BY DATE(created_at) ORDER BY date
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

    allowed_sorts = {"id", "type", "topic", "confidence", "updated_at", "created_at", "project"}
    if sort not in allowed_sorts:
        sort = "updated_at"
    order_dir = "ASC" if order.lower() == "asc" else "DESC"

    conn = get_conn()

    if q and mode == "semantic":
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            import embeddings as emb
            results = emb.find_similar(conn, q, threshold=0.3, limit=limit)
            conn.close()
            memories = [{
                "id": r["id"], "type": r.get("type", ""), "topic": r.get("topic", ""),
                "content": r.get("content", ""), "similarity": round(r.get("similarity", 0), 3),
                "confidence": r.get("confidence", 0.7), "project": r.get("project", ""),
                "updated_at": r.get("updated_at", ""),
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
                   m.session_id, m.updated_at, m.created_at, m.archived_reason
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
        rows = conn.execute(f"""
            SELECT id, type, topic, content, confidence, project,
                   session_id, updated_at, created_at, archived_reason
            FROM memories{where_clause} ORDER BY {sort} {order_dir} LIMIT ? OFFSET ?
        """, sql_params + [limit, offset]).fetchall()

    memories = [{
        "id": r["id"], "type": r["type"], "topic": r["topic"], "content": r["content"],
        "confidence": r["confidence"], "project": r["project"], "session_id": r["session_id"],
        "updated_at": r["updated_at"], "created_at": r["created_at"], "archived_reason": r["archived_reason"],
    } for r in rows]
    conn.close()
    return {"memories": memories, "total": total}, 200


def api_memory_detail(params, memory_id):
    conn = get_conn()
    row = conn.execute("""
        SELECT id, type, topic, content, confidence, project, session_id,
               updated_at, created_at, archived_reason, depth, associated_files
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
    empty_where = "WHERE (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.session_id) > 0" if hide_empty else ""
    total = conn.execute(f"SELECT COUNT(*) FROM sessions s {empty_where}").fetchone()[0]
    sessions = conn.execute(f"""
        SELECT s.session_id, s.parent_session_id, s.project, s.started_at, s.transcript_path,
               (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.session_id) as memory_count,
               (SELECT COUNT(*) FROM sessions c WHERE c.parent_session_id = s.session_id) as child_count
        FROM sessions s {empty_where} ORDER BY {sort} {order_dir} LIMIT ? OFFSET ?
    """, (limit, offset)).fetchall()
    result = []
    for s in sessions:
        d = dict(s)
        d["cwd"] = _extract_cwd(s["transcript_path"])
        d["session_type"] = "agent" if s["parent_session_id"] else ("parent" if s["child_count"] > 0 else "session")
        d["interaction_count"] = _count_interactions(s["transcript_path"])
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
    retrieved_row = conn.execute(
        "SELECT value FROM hook_state WHERE session_id = ? AND key = 'retrieved_ids'", (full_id,)
    ).fetchone()
    consumed_ids = []
    if retrieved_row and retrieved_row["value"]:
        try:
            consumed_ids = json.loads(retrieved_row["value"])
        except Exception:
            pass
    consumed = []
    if consumed_ids:
        placeholders = ",".join("?" * len(consumed_ids))
        consumed = rows_to_list(conn.execute(f"""
            SELECT id, type, topic, content, confidence, project, session_id, updated_at
            FROM memories WHERE id IN ({placeholders})
        """, consumed_ids).fetchall())
    layer_rows = conn.execute(
        "SELECT detail FROM metrics WHERE session_id = ? AND event = 'layer_delivery' AND detail IS NOT NULL",
        (full_id,)
    ).fetchall()
    layer_detail = []
    for r in layer_rows:
        try:
            layer_detail.append(json.loads(r["detail"]))
        except Exception:
            pass
    conn.close()
    tokens = _estimate_tokens(session["transcript_path"])
    return {
        "session": row_to_dict(session), "memories": memories, "consumed": consumed,
        "layer_detail": layer_detail, "chain": chain, "tokens": tokens,
    }, 200


def api_session_transcript(params, session_id):
    conn = get_conn()
    session = conn.execute(
        "SELECT transcript_path FROM sessions WHERE session_id LIKE ?", (f"{session_id}%",)
    ).fetchone()
    conn.close()
    if not session or not session["transcript_path"]:
        return {"messages": [], "note": "No transcript"}, 200
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


def api_metrics(params):
    conn = get_conn()
    summary = rows_to_list(conn.execute("""
        SELECT event, COUNT(*) as count, AVG(value) as avg_value,
               MIN(value) as min_value, MAX(value) as max_value
        FROM metrics GROUP BY event ORDER BY count DESC
    """).fetchall())
    latency_series = rows_to_list(conn.execute("""
        SELECT DATE(created_at) as date, AVG(value) as avg_ms, COUNT(*) as count
        FROM metrics WHERE event = 'retrieval_latency_ms'
        GROUP BY DATE(created_at) ORDER BY date
    """).fetchall())
    layer_series = rows_to_list(conn.execute("""
        SELECT DATE(created_at) as date, event, COUNT(*) as count
        FROM metrics WHERE event LIKE 'layer%' OR event LIKE 'retrieval_%'
        GROUP BY DATE(created_at), event ORDER BY date
    """).fetchall())
    embed_events = ("embed_daemon_ms", "embed_local_ms", "search_vec_ms", "search_brute_ms", "fanout_ms")
    placeholders = ",".join("?" * len(embed_events))
    embed_stats = rows_to_list(conn.execute(f"""
        SELECT event, COUNT(*) as count, AVG(value) as avg_ms,
               MIN(value) as min_ms, MAX(value) as max_ms,
               (SELECT AVG(sub.value) FROM (
                   SELECT value FROM metrics m2 WHERE m2.event = metrics.event
                   ORDER BY m2.created_at DESC LIMIT 20
               ) sub) as recent_avg_ms
        FROM metrics WHERE event IN ({placeholders}) GROUP BY event
    """, embed_events).fetchall())
    embed_series = rows_to_list(conn.execute(f"""
        SELECT DATE(created_at) as date, event, AVG(value) as avg_ms, COUNT(*) as count
        FROM metrics WHERE event IN ({placeholders})
        GROUP BY DATE(created_at), event ORDER BY date
    """, embed_events).fetchall())
    conn.close()
    return {
        "summary": summary, "latency_series": latency_series, "layer_series": layer_series,
        "embed_stats": embed_stats, "embed_series": embed_series,
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
    params_out = {}
    for name in sorted(dir(config)):
        if name.startswith("_") or not name.isupper():
            continue
        val = getattr(config, name)
        if not isinstance(val, (int, float, bool, str)):
            continue
        params_out[name] = {"value": val, "type": type(val).__name__, "env_var": f"CAIRN_{name}"}
    overrides = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    overrides[key.strip()] = val.strip()
    for name, info in params_out.items():
        env_key = f"CAIRN_{name}"
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
    for name, value in data["updates"].items():
        env_key = f"CAIRN_{name}"
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
    count = 0
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                if '"role":"user"' in line or '"role": "user"' in line:
                    count += 1
    except Exception:
        pass
    return count


def _estimate_tokens(transcript_path):
    if not transcript_path or not os.path.exists(transcript_path):
        return {"user_tokens": 0, "assistant_tokens": 0}
    user_chars, assistant_chars = 0, 0
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
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
    except Exception:
        pass
    return {"user_tokens": user_chars // 4, "assistant_tokens": assistant_chars // 4}


def _extract_cwd(transcript_path):
    if not transcript_path:
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
    messages = []
    with open(transcript_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
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


# ─── HTTP Server ────────────────────────────────────────────────────────────

# Route table: (method, pattern) -> handler
# Patterns use {name} for path params, converted to regex groups
_ROUTES: list[tuple[str, str, callable]] = [
    ("GET", "/api/stats", lambda p, **kw: api_stats(p)),
    ("GET", "/api/memories", lambda p, **kw: api_memories(p)),
    ("GET", "/api/memory/{memory_id}/context", lambda p, **kw: api_memory_context(p, int(kw["memory_id"]))),
    ("GET", "/api/memory/{memory_id}", lambda p, **kw: api_memory_detail(p, int(kw["memory_id"]))),
    ("GET", "/api/sessions", lambda p, **kw: api_sessions(p)),
    ("GET", "/api/session/{session_id}/transcript", lambda p, **kw: api_session_transcript(p, kw["session_id"])),
    ("GET", "/api/session/{session_id}", lambda p, **kw: api_session_detail(p, kw["session_id"])),
    ("GET", "/api/metrics", lambda p, **kw: api_metrics(p)),
    ("GET", "/api/token-stats", lambda p, **kw: api_token_stats(p)),
    ("GET", "/api/memory-usage", lambda p, **kw: api_memory_usage(p)),
    ("GET", "/api/projects", lambda p, **kw: api_projects(p)),
    ("GET", "/api/config", lambda p, **kw: api_config_get(p)),
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
        body = json.dumps(data, default=str).encode("utf-8")
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

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        # Static files
        if path == "/" or path == "/index.html":
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
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

        if path == "/api/config":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            try:
                data, status = api_config_update(body)
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
