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
import tempfile
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

SOCKET_PATH = os.path.join(os.path.dirname(__file__), ".daemon.sock")
PID_PATH = os.path.join(os.path.dirname(__file__), ".daemon.pid")
CAIRN_DIR = os.path.dirname(__file__)

# cairn package is on sys.path via pip install -e .


_cross_encoder: Any = None
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
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from cairn.config import CROSS_ENCODER_ENABLED, CROSS_ENCODER_MODEL
        if not CROSS_ENCODER_ENABLED:
            return None
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return _cross_encoder
    except Exception:
        return None


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

        elif action == "rerank":
            query = request["query"]
            candidates = request["candidates"]
            ce = _get_cross_encoder()
            if ce is None:
                response = {"scores": None}
            else:
                pairs = [(query, c) for c in candidates]
                scores = ce.predict(pairs).tolist()
                response = {"scores": scores}

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
            except ImportError:
                import sqlite3
            DB_PATH = os.path.join(CAIRN_DIR, "cairn.db")
            conn_db = sqlite3.connect(DB_PATH)
            results = emb.find_similar(conn_db, text, threshold=threshold, limit=limit)
            conn_db.close()
            response = {"results": results}

        elif action == "ping":
            response = {"status": "ok"}

        elif action == "hook":
            route = request.get("route", "")
            payload = request.get("payload", {}) or {}
            hook_path = HOOK_ROUTES.get(route)
            if not hook_path:
                response = {"error": f"Unknown hook route: {route}"}
            else:
                tmp_path = None
                body = payload.pop("_transcript_body", None)
                if body:
                    fd, tmp_path = tempfile.mkstemp(prefix="cairn-transcript-", suffix=".jsonl")
                    with os.fdopen(fd, "w") as f:
                        f.write(body)
                    payload["transcript_path"] = tmp_path
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
                finally:
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

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


def _start_tcp_listener(emb, port: int) -> None:
    """Spawn a daemon thread that accepts TCP connections and dispatches to handle_client.

    Binds 0.0.0.0 so containers on ANY Docker bridge network can reach the
    daemon via their default gateway — not just containers on the default
    docker0 bridge. Compose-created networks have their own bridge with a
    different gateway IP (e.g. 172.18.0.1 instead of docker0's 172.17.0.1),
    so binding only docker0 misses them.

    Same JSON-over-stream protocol as the Unix socket — same handle_client.
    """
    bind_ip = "0.0.0.0"

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: daemon.py [start|stop|status]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "start":
        if is_running():
            print("Daemon already running.")
            sys.exit(0)
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

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
