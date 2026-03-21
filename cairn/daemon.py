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
import sys
import threading

SOCKET_PATH = os.path.join(os.path.dirname(__file__), ".daemon.sock")
PID_PATH = os.path.join(os.path.dirname(__file__), ".daemon.pid")
CAIRN_DIR = os.path.dirname(__file__)

sys.path.insert(0, CAIRN_DIR)


def handle_client(conn, model):
    """Handle a single client request."""
    import embeddings as emb

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
            vec = emb.embed(text)
            response = {"vector": emb.to_blob(vec).hex()}

        elif action == "similarity":
            text = request["text"]
            threshold = request.get("threshold", 0.5)
            limit = request.get("limit", 10)
            import sqlite3
            DB_PATH = os.path.join(CAIRN_DIR, "cairn.db")
            conn_db = sqlite3.connect(DB_PATH)
            results = emb.find_similar(conn_db, text, threshold=threshold, limit=limit)
            conn_db.close()
            response = {"results": results}

        elif action == "ping":
            response = {"status": "ok"}

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


def run_server():
    """Start the daemon server."""
    # Clean up stale socket
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    # Load model eagerly
    print("Loading embedding model...")
    import embeddings as emb
    emb.get_model()
    print("Model loaded. Daemon ready.")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)

    # Write PID
    with open(PID_PATH, "w") as f:
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
    except Exception:
        return None


def is_running():
    """Check if daemon is running."""
    if not os.path.exists(PID_PATH):
        return False
    try:
        with open(PID_PATH) as f:
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
             f"import sys,os; sys.path.insert(0,{repr(cairn_dir)}); "
             f"os.chdir({repr(cairn_dir)}); "
             "from daemon import run_server; run_server()"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        import time
        time.sleep(5)
        if is_running():
            print("Daemon started.")
        else:
            print("Daemon starting (model loading)...")
        sys.exit(0)

    elif cmd == "stop":
        if not is_running():
            print("Daemon not running.")
            sys.exit(0)
        with open(PID_PATH) as f:
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
