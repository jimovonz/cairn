"""
Cairn hook API server.

Listens on a Unix socket and exposes the four cairn hook event endpoints
(/userpromptsubmit, /stop, /pretool, /posttool) so container-side shims
can relay hook payloads to the host without each container needing a
full cairn install.

Each endpoint subprocesses the matching hooks/*.py script with the
posted JSON payload on stdin, captures stdout/stderr/exit-code, and
returns them in a JSON envelope the shim replays to the calling tool.

If the request includes a `_transcript_body` field, the body is written
to a host-side temp file and `transcript_path` in the payload is
rewritten to point at it — so the host-side hook can read transcript
content that originally lived inside the container filesystem.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import UnixStreamServer

CAIRN_HOME = Path(__file__).resolve().parent.parent
HOOK_DIR = CAIRN_HOME / "hooks"
VENV_PYTHON = CAIRN_HOME / ".venv" / "bin" / "python3"
SOCKET_PATH = CAIRN_HOME / "cairn" / ".api.sock"

ROUTES = {
    "/userpromptsubmit": HOOK_DIR / "prompt_hook.py",
    "/stop": HOOK_DIR / "stop_hook.py",
    "/pretool": HOOK_DIR / "pretool_hook.py",
    "/posttool": HOOK_DIR / "posttool_hook.py",
}


def _materialize_transcript(payload: dict) -> str | None:
    body = payload.pop("_transcript_body", None)
    if not body:
        return None
    fd, tmp_path = tempfile.mkstemp(prefix="cairn-transcript-", suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        f.write(body)
    payload["transcript_path"] = tmp_path
    return tmp_path


def _run_hook(hook_path: Path, payload: dict) -> dict:
    tmp = _materialize_transcript(payload)
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(hook_path)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return {"stdout": "", "stderr": f"hook timeout: {e}", "exit_code": 124}
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


class UnixHTTPServer(UnixStreamServer):
    """HTTPServer over a Unix-domain socket."""

    address_family = socket.AF_UNIX
    allow_reuse_address = True

    def get_request(self):
        request, _ = super().get_request()
        return request, ["local", 0]

    def server_bind(self):
        path = self.server_address
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        super().server_bind()
        os.chmod(path, 0o600)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        hook_path = ROUTES.get(self.path)
        if not hook_path:
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"invalid json: {e}".encode())
            return

        envelope = _run_hook(hook_path, payload)
        body = json.dumps(envelope).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence default access log
        pass


def serve(socket_path: Path = SOCKET_PATH) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    server = UnixHTTPServer(str(socket_path), Handler)
    print(f"cairn api server listening on {socket_path}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    serve()
