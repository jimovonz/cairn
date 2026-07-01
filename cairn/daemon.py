#!/usr/bin/env python3
"""
Cairn Embedding Daemon.

Keeps the sentence-transformers model resident in RAM, accepting requests
over a Unix socket. Eliminates the ~3s cold start per hook invocation.

Usage:
  Start:  python3 cairn/daemon.py start
  Stop:   python3 cairn/daemon.py stop
  Status: python3 cairn/daemon.py status
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
from typing import Any

HOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks")
VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python3")
HOOK_ROUTES = {
    "userpromptsubmit": os.path.join(HOOK_DIR, "prompt_hook.py"),
    "stop": os.path.join(HOOK_DIR, "stop_hook.py"),
    "pretool": os.path.join(HOOK_DIR, "pretool_hook.py"),
    "posttool": os.path.join(HOOK_DIR, "posttool_hook.py"),
}
# Persistent stash for container-sourced transcripts so query.py --context
# can recover their conversation history after the originating container
# session ends. One file per session_id, overwritten on each hook call
# (shim always sends the full current transcript body).
CONTAINER_TRANSCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "transcripts", "container"
)


def _safe_session_filename(session_id: str) -> str:
    """Allow only alnum, dash, underscore in stashed filename to avoid path traversal."""
    return "".join(c for c in session_id if c.isalnum() or c in "-_")[:200] or "unknown"

SOCKET_PATH = os.path.join(os.path.dirname(__file__), ".daemon.sock")
PID_PATH = os.path.join(os.path.dirname(__file__), ".daemon.pid")
CAIRN_DIR = os.path.dirname(__file__)

# cairn package is on sys.path via pip install -e .


_cross_encoder: Any = None
_cross_encoder_name: Any = None  # name of the loaded reranker — provenance for memory_deliveries
_cross_encoder_floor: Any = None  # score floor matched to the loaded model
_force_cpu_reranker: bool = False  # set after a GPU predict fault — pins reload to ms-marco/CPU
_nli_model: Any = None


def _get_nli_model() -> Any:
    """Lazily load the NLI cross-encoder model. Returns None if disabled or unavailable."""
    global _nli_model
    if _nli_model is not None:
        return _nli_model
    try:
        from cairn.config import NLI_ENABLED, NLI_MODEL
        if not NLI_ENABLED:
            return None
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(NLI_MODEL)
        return _nli_model
    except Exception:
        return None


def _get_cross_encoder() -> Any:
    """Lazily load the cross-encoder model. Returns None if disabled or unavailable."""
    global _cross_encoder, _cross_encoder_name, _cross_encoder_floor
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from cairn.config import (CROSS_ENCODER_ENABLED, resolve_reranker,
                                  CROSS_ENCODER_MODEL, CROSS_ENCODER_SCORE_FLOOR)
        if not CROSS_ENCODER_ENABLED:
            return None
        from sentence_transformers import CrossEncoder
        if _force_cpu_reranker:
            # Post-fault: pin to the small ms-marco model on CPU so the gate keeps
            # working (weak > dead) until a restart reloads the best GPU model.
            model_name, _floor = CROSS_ENCODER_MODEL, CROSS_ENCODER_SCORE_FLOOR
            _cross_encoder = CrossEncoder(model_name, device="cpu")
        else:
            model_name, _floor = resolve_reranker()  # device-aware: bge on CUDA, ms-marco on CPU
            _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_name = model_name
        _cross_encoder_floor = _floor
        return _cross_encoder
    except Exception:
        return None


# --- Daemon-resident vector search -----------------------------------------
# The daemon is long-lived; hook processes are not. Caching the normalized
# embedding matrices here means each retrieval costs one socket round-trip
# instead of a per-process 35MB blob fetch + matrix rebuild (~150-400ms).
# Invalidation: a cheap stamp query (count, max id, max updated_at) per
# request — any memory write changes the stamp and triggers a reload.

_search_cache: dict = {"stamp": None, "meta": [], "M": None, "T": None, "t_pos": None}


def _db_conn():
    try:
        import pysqlite3 as sqlite3  # type: ignore[import-untyped]
    except ImportError as _pysqlite_err:  # pragma: no cover
        import os as _os
        if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
            import sqlite3  # type: ignore[no-redef]
        else:
            raise ImportError(
                "cairn requires pysqlite3 (WAL checkpoint-race fixes); stdlib sqlite3 can "
                "corrupt WAL DBs under concurrent multi-version access. Set "
                "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
            ) from _pysqlite_err
    c = sqlite3.connect(os.path.join(CAIRN_DIR, "cairn.db"))
    c.execute("PRAGMA busy_timeout=5000")
    return c


def _ensure_search_cache() -> dict:
    """Load (or revalidate) the in-memory search matrices."""
    import numpy as np
    conn = _db_conn()
    stamp = tuple(conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(id),0), COALESCE(MAX(updated_at),'') "
        "FROM memories WHERE embedding IS NOT NULL AND deleted_at IS NULL"
    ).fetchone())
    try:
        stamp = stamp + tuple(conn.execute(
            "SELECT COUNT(*) FROM memory_qf_embeddings").fetchone())
    except Exception:
        stamp = stamp + (0,)  # pre-v14 DB — no sidecar
    if _search_cache["stamp"] == stamp:
        conn.close()
        return _search_cache

    rows = conn.execute(
        "SELECT id, type, topic, content, embedding, updated_at, project, confidence, "
        "depth, archived_reason, session_id, keywords, topic_embedding "
        "FROM memories WHERE embedding IS NOT NULL AND deleted_at IS NULL"
    ).fetchall()
    # qf sidecar (schema v14): one row per question-form keyword embedding.
    try:
        qf_rows = conn.execute(
            "SELECT memory_id, embedding FROM memory_qf_embeddings").fetchall()
    except Exception:
        qf_rows = []
    conn.close()

    metas: list = []
    vecs: list = []
    t_pos: list = []
    t_vecs: list = []
    dim = None
    for row in rows:
        try:
            v = np.frombuffer(row[4], dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if dim is None:
            dim = v.shape[0]
        if v.shape[0] != dim:
            continue
        idx = len(vecs)
        vecs.append(v)
        metas.append({
            "id": row[0], "type": row[1], "topic": row[2], "content": row[3],
            "updated_at": row[5], "project": row[6], "confidence": row[7],
            "depth": row[8], "archived_reason": row[9], "session_id": row[10],
            "keywords": row[11],
        })
        if row[12] is not None:
            try:
                tv = np.frombuffer(row[12], dtype=np.float32)
            except (ValueError, TypeError):
                continue
            if tv.shape[0] == dim:
                t_pos.append(idx)
                t_vecs.append(tv)

    # qf matrix: rows map many-to-one onto memory indices via q_pos (a memory can
    # have several question-form keywords; each is its own row).
    id_to_idx = {m["id"]: i for i, m in enumerate(metas)}
    q_pos: list = []
    q_vecs: list = []
    for mem_id, qblob in qf_rows:
        idx = id_to_idx.get(mem_id)
        if idx is None:
            continue
        try:
            qv = np.frombuffer(qblob, dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if dim is not None and qv.shape[0] == dim:
            q_pos.append(idx)
            q_vecs.append(qv)

    M = T = tp = Q = qp = None
    if vecs:
        M = np.stack(vecs)
        n = np.linalg.norm(M, axis=1, keepdims=True)
        n[n == 0] = 1.0
        M = M / n
        if t_vecs:
            T = np.stack(t_vecs)
            tn = np.linalg.norm(T, axis=1, keepdims=True)
            tn[tn == 0] = 1.0
            T = T / tn
            tp = np.array(t_pos)
        if q_vecs:
            Q = np.stack(q_vecs)
            qn = np.linalg.norm(Q, axis=1, keepdims=True)
            qn[qn == 0] = 1.0
            Q = Q / qn
            qp = np.array(q_pos)
    _search_cache.update({"stamp": stamp, "meta": metas, "M": M, "T": T, "t_pos": tp,
                          "Q": Q, "q_pos": qp})
    return _search_cache


def _vector_search(emb, texts, n_base, min_sim, top_k):
    """Score all memories against the query variants; return top rows.

    Mirrors find_similar's candidate scoring: per-variant z-score
    normalization with max-over-variants raw similarity, plus the
    dual-embedding topic supplement against the base variants only.
    """
    import numpy as np
    cache = _ensure_search_cache()
    if cache["M"] is None:
        return []
    model = emb.get_model()
    V = np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
    if V.ndim == 1:
        V = V[None, :]
    S = cache["M"] @ V.T
    if S.shape[1] > 1:
        mu = S.mean(axis=0)
        sd = np.maximum(S.std(axis=0), 1e-6)
        Z = (S - mu) / sd
        best_idx = Z.argmax(axis=1)
        best = S[np.arange(S.shape[0]), best_idx].astype(np.float64)
    else:
        best = S[:, 0].astype(np.float64)
    if cache["T"] is not None:
        TS = cache["T"] @ V[:max(1, n_base)].T
        t_best = TS.max(axis=1)
        tp = cache["t_pos"]
        best[tp] = np.maximum(best[tp], t_best)
    if cache.get("Q") is not None:
        # Per-qf symmetric supplement (schema v14): a memory scores as its best
        # question-form keyword match. maximum.at handles the many-to-one qf->memory
        # mapping (plain fancy-index assignment would keep only the LAST qf row).
        QS = cache["Q"] @ V[:max(1, n_base)].T
        q_best = QS.max(axis=1)
        np.maximum.at(best, cache["q_pos"], q_best)

    order = np.argsort(-best)
    out = []
    for i in order[:top_k]:
        s = float(best[i])
        if s < min_sim:
            break
        m = dict(cache["meta"][i])
        m["similarity"] = s
        out.append(m)
    return out


def handle_client(conn, emb):
    """Handle a single client request."""
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk

        request = json.loads(data.decode())
        action = request.get("action", "")

        if action == "embed":
            text = request["text"]
            # Use model directly — do NOT call emb.embed() which would recurse into _daemon_embed
            model = emb.get_model()
            vec = model.encode(text, normalize_embeddings=True)
            response = {"vector": emb.to_blob(vec).hex()}

        elif action == "embed_batch":
            texts = request["texts"]
            model = emb.get_model()
            vecs = model.encode(texts, normalize_embeddings=True)
            response = {"vectors": [emb.to_blob(v).hex() for v in vecs]}

        elif action == "vector_search":
            rows = _vector_search(
                emb,
                request["texts"],
                int(request.get("n_base", 1)),
                float(request.get("min_sim", 0.15)),
                int(request.get("top_k", 300)),
            )
            response = {"rows": rows}

        elif action == "rerank":
            query = request["query"]
            candidates = request["candidates"]
            ce = _get_cross_encoder()
            if ce is None:
                response = {"scores": None}
            else:
                pairs = [(query, c) for c in candidates]
                try:
                    scores = ce.predict(pairs).tolist()
                except Exception as ce_err:
                    # A GPU fault (e.g. cudaErrorLaunchFailure) poisons the cached
                    # GPU model so every predict throws. Evict it, pin to ms-marco
                    # on CPU, and retry once — the gate degrades to working-weak
                    # immediately instead of staying dead until the restart cron.
                    global _cross_encoder, _force_cpu_reranker
                    _cross_encoder = None
                    _force_cpu_reranker = True
                    ce = _get_cross_encoder()
                    if ce is None:
                        raise ce_err
                    scores = ce.predict(pairs).tolist()
                response = {"scores": scores, "score_floor": _cross_encoder_floor,
                            "model": _cross_encoder_name}

        elif action == "nli":
            pairs = request["pairs"]
            nli = _get_nli_model()
            if nli is None:
                response = {"scores": None}
            else:
                scores = nli.predict(pairs).tolist()
                response = {"scores": scores}

        elif action == "similarity":
            text = request["text"]
            threshold = request.get("threshold", 0.5)
            limit = request.get("limit", 10)
            try:
                import pysqlite3 as sqlite3  # type: ignore[import-untyped]
            except ImportError as _pysqlite_err:  # pragma: no cover
                import os as _os
                if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
                    import sqlite3
                else:
                    raise ImportError(
                        "cairn requires pysqlite3 (WAL checkpoint-race fixes); stdlib sqlite3 "
                        "can corrupt WAL DBs under concurrent multi-version access. Set "
                        "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
                    ) from _pysqlite_err
            DB_PATH = os.path.join(CAIRN_DIR, "cairn.db")
            conn_db = sqlite3.connect(DB_PATH)
            results = emb.find_similar(conn_db, text, threshold=threshold, limit=limit)
            conn_db.close()
            response = {"results": results}

        elif action == "ping":
            response = {"status": "ok"}

        elif action == "sync_start":
            # Live-start P2P sync services (dashboard toggle / pairing auto-enable).
            # force=True so it starts even if the daemon booted with sync disabled.
            try:
                from cairn.sync.service import start_sync_services, sync_running
                start_sync_services(force=True)
                response = {"ok": True, "running": sync_running()}
            except Exception as e:  # noqa: BLE001
                response = {"ok": False, "error": str(e)}

        elif action == "sync_stop":
            try:
                from cairn.sync.service import stop_sync_services, sync_running
                stop_sync_services()
                response = {"ok": True, "running": sync_running()}
            except Exception as e:  # noqa: BLE001
                response = {"ok": False, "error": str(e)}

        elif action == "sync_status":
            try:
                from cairn.sync.service import sync_running
                response = {"ok": True, "running": sync_running()}
            except Exception as e:  # noqa: BLE001
                response = {"ok": False, "error": str(e)}

        elif action == "hook":
            route = request.get("route", "")
            payload = request.get("payload", {}) or {}
            hook_path = HOOK_ROUTES.get(route)
            if not hook_path:
                response = {"error": f"Unknown hook route: {route}"}
            else:
                body = payload.pop("_transcript_body", None)
                if body:
                    session_id = payload.get("session_id") or payload.get("sessionId") or ""
                    fname = _safe_session_filename(session_id) + ".jsonl"
                    os.makedirs(CONTAINER_TRANSCRIPTS_DIR, exist_ok=True)
                    stash_path = os.path.join(CONTAINER_TRANSCRIPTS_DIR, fname)
                    # Write atomically so a concurrent reader never sees a partial file.
                    tmp_write = stash_path + ".tmp"
                    with open(tmp_write, "w") as f:
                        f.write(body)
                    os.replace(tmp_write, stash_path)
                    payload["transcript_path"] = stash_path
                try:
                    result = subprocess.run(
                        [VENV_PYTHON, hook_path],
                        input=json.dumps(payload),
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    response = {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.returncode,
                    }
                except subprocess.TimeoutExpired as e:
                    response = {"stdout": "", "stderr": f"hook timeout: {e}", "exit_code": 124}

        elif action == "cairn_recall":
            # Semantic-search the cairn DB for entries matching `text`.
            # Used by container-side clients (e.g. copilot-human-loop extension)
            # that have no direct query.py access.
            text = request.get("text", "") or ""
            limit = int(request.get("limit", 10))
            threshold = float(request.get("threshold", 0.3))
            if not text.strip():
                response = {"results": []}
            else:
                from cairn import query as _query
                results = _query.semantic_search(text, limit=limit, threshold=threshold) or []
                response = {"results": results}

        elif action == "cairn_remember":
            # Insert a memory via the same path as `query.py --add`.
            mem_type = request.get("type") or "fact"
            topic = request.get("topic") or ""
            content = request.get("content") or ""
            project = request.get("project")
            session_id = request.get("session_id")
            if not topic or not content:
                response = {"error": "topic and content required"}
            else:
                from cairn import query as _query
                _query.add_memory(mem_type, topic, content, project=project, session_id=session_id)
                response = {"ok": True}

        else:
            response = {"error": f"Unknown action: {action}"}

        conn.sendall(json.dumps(response).encode())
    except Exception as e:
        try:
            conn.sendall(json.dumps({"error": str(e)}).encode())
        except Exception:
            pass
    finally:
        conn.close()


def _detect_docker0_ip() -> str:
    """Return the docker0 bridge IP if present, else fall back to 0.0.0.0."""
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "docker0"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout:
            # Line shape: "3: docker0    inet 172.17.0.1/16 ..."
            for token in out.stdout.split():
                if "/" in token and token.replace(".", "").replace("/", "").isdigit():
                    return token.split("/", 1)[0]
    except (subprocess.SubprocessError, OSError):
        pass
    return "0.0.0.0"


def _start_tcp_listener(emb, port: int) -> None:
    """Spawn a daemon thread that accepts TCP connections and dispatches to handle_client.

    Same JSON-over-stream protocol as the Unix socket — same handle_client function.
    """
    bind_ip = _detect_docker0_ip()

    def serve():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((bind_ip, port))
        except OSError as e:
            print(f"TCP listener bind {bind_ip}:{port} failed: {e}")
            return
        srv.listen(16)
        print(f"TCP listener bound to {bind_ip}:{port}")
        while True:
            try:
                conn, _addr = srv.accept()
                t = threading.Thread(target=handle_client, args=(conn, emb), daemon=True)
                t.start()
            except OSError:
                break

    threading.Thread(target=serve, name="cairn-tcp-listener", daemon=True).start()


def run_server():
    """Start the daemon server."""
    # Clean up stale socket
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    # Load model eagerly
    print("Loading embedding model...")
    from cairn import embeddings as emb
    emb.get_model()
    print("Model loaded. Daemon ready.")

    # Optional: dev-container extension auto-injector
    try:
        from cairn.config import CONTAINER_AUTO_INSTALL_ENABLED, CONTAINER_AUTO_INSTALL_VSIX_DIR
        if CONTAINER_AUTO_INSTALL_ENABLED:
            from cairn.container_injector import start_in_background
            start_in_background(CONTAINER_AUTO_INSTALL_VSIX_DIR)
            print(f"Container injector watching {CONTAINER_AUTO_INSTALL_VSIX_DIR}")
    except Exception as e:  # noqa: BLE001 — never let injector failure kill daemon
        print(f"Container injector not started: {e}")

    # Optional: TCP listener so container-side shims can reach the daemon
    # without bind-mounting the Unix socket. Bound to docker0 bridge IP
    # (containers' default gateway) when available, falling back to 0.0.0.0.
    try:
        from cairn.config import CAIRN_TCP_LISTENER_ENABLED, CAIRN_TCP_PORT
        if CAIRN_TCP_LISTENER_ENABLED:
            _start_tcp_listener(emb, CAIRN_TCP_PORT)
    except Exception as e:  # noqa: BLE001
        print(f"TCP listener not started: {e}")

    # Optional: always-on peer-to-peer sync (HTTPS server + LAN discovery).
    # Opt-in via CAIRN_SYNC_ENABLED; the dashboard Sync tab drives pairing.
    try:
        from cairn.sync.service import start_sync_services
        start_sync_services()
    except Exception as e:  # noqa: BLE001 — never let sync failure kill the daemon
        print(f"Sync services not started: {e}")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)

    # Write PID
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    def shutdown(signum, frame):
        print("\nShutting down daemon...")
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        if os.path.exists(PID_PATH):
            os.unlink(PID_PATH)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    IDLE_TIMEOUT = 1800  # 30 minutes
    server.settimeout(IDLE_TIMEOUT)

    while True:
        try:
            conn, _ = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, emb))
            thread.daemon = True
            thread.start()
        except socket.timeout:
            print("Idle timeout reached — shutting down daemon")
            break
        except OSError:
            break

    shutdown(None, None)


def send_request(request):
    """Send a request to the daemon. Returns response dict or None if daemon not running."""
    if not os.path.exists(SOCKET_PATH):
        return None
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(10)
        client.connect(SOCKET_PATH)
        client.sendall(json.dumps(request).encode())
        client.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
        client.close()
        return json.loads(data.decode())
    except (ConnectionRefusedError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def is_running():
    """Check if daemon is running."""
    if not os.path.exists(PID_PATH):
        return False
    try:
        with open(PID_PATH, encoding="utf-8") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        # Stale PID file
        if os.path.exists(PID_PATH):
            os.unlink(PID_PATH)
        return False


def _rerank_healthy() -> bool:
    """True if the daemon's cross-encoder rerank actually works — not just that
    the process answers ping. Distinguishes a responsive-but-poisoned daemon (a
    GPU-resident model that now throws cudaErrorLaunchFailure on every predict
    after a GPU fault) from a healthy one: the model is cached, so the process
    stays up while the relevance gate is silently dead. Returns True when
    reranking is legitimately disabled (nothing to heal)."""
    try:
        from cairn.config import CROSS_ENCODER_ENABLED
    except Exception:
        return True
    if not CROSS_ENCODER_ENABLED:
        return True
    try:
        resp = send_request({"action": "rerank", "query": "health probe",
                             "candidates": ["health probe candidate"]})
    except Exception:
        return False
    if not resp or resp.get("error"):
        return False
    scores = resp.get("scores")
    return isinstance(scores, list) and len(scores) == 1


def _stop_daemon() -> None:
    """SIGTERM the running daemon and wait for it to exit (best-effort)."""
    import time
    try:
        with open(PID_PATH, encoding="utf-8") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return
    for _ in range(20):
        time.sleep(0.5)
        if not is_running():
            return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: daemon.py [start|stop|status|healthcheck]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "start":
        if is_running():
            if _rerank_healthy():
                print("Daemon already running.")
                sys.exit(0)
            # Responsive but the rerank gate is dead (e.g. GPU fault poisoned the
            # cached cross-encoder). Restart to reload the model on a healthy
            # device — ping-liveness alone would leave the gate silently off.
            print("Daemon running but rerank unhealthy — restarting to reload model.")
            _stop_daemon()
        # Launch as detached subprocess
        import subprocess
        cairn_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen(
            [sys.executable, "-c",
             f"import os; os.chdir({repr(cairn_dir)}); "
             "from cairn.daemon import run_server; run_server()"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        import time
        # Wait for daemon to become responsive (model load can take 10s+ on slow machines)
        for _ in range(20):
            time.sleep(1)
            if is_running():
                resp = send_request({"action": "ping"})
                if resp and resp.get("status") == "ok":
                    print("Daemon started.")
                    sys.exit(0)
        if is_running():
            print("Daemon started (not yet responding to ping).")
        else:
            print("Daemon starting (model loading)...")
        sys.exit(0)

    elif cmd == "stop":
        if not is_running():
            print("Daemon not running.")
            sys.exit(0)
        with open(PID_PATH, encoding="utf-8") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        print("Daemon stopped.")

    elif cmd == "status":
        if is_running():
            resp = send_request({"action": "ping"})
            if resp and resp.get("status") == "ok":
                print("Daemon running and responsive.")
            else:
                print("Daemon PID exists but not responding.")
        else:
            print("Daemon not running.")

    elif cmd == "healthcheck":
        # Cron entry point: verify the daemon is up AND reranking works; restart
        # if the gate is dead. Idempotent and quiet on the happy path.
        if is_running() and _rerank_healthy():
            sys.exit(0)
        if is_running():
            print("rerank unhealthy — restarting daemon.")
            _stop_daemon()
        import subprocess, time
        cairn_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen(
            [sys.executable, "-c",
             f"import os; os.chdir({repr(cairn_dir)}); "
             "from cairn.daemon import run_server; run_server()"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(20):
            time.sleep(1)
            if is_running() and _rerank_healthy():
                print("daemon healthy.")
                sys.exit(0)
        print("daemon restarted (rerank not yet confirmed).")
        sys.exit(0)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
