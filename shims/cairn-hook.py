#!/usr/bin/env python3
"""
Cairn hook shim for use inside a dev container.

Relays a hook payload from stdin to the host's cairn daemon over a
mounted Unix socket, replays stdout/stderr/exit-code from the response.

Inlines the contents of `transcript_path` (so the host-side hook can
read transcripts that live in the container filesystem) up to a
configurable byte cap.

Invoke as: cairn-hook.py <route>
  route: userpromptsubmit | stop | pretool | posttool

Env:
  CAIRN_SOCK   path to host's cairn .daemon.sock (default /run/cairn/.daemon.sock)
  CAIRN_TRANSCRIPT_INLINE_MAX  max bytes to inline (default 2_000_000)
"""

import json
import os
import socket
import sys

SOCK = os.environ.get("CAIRN_SOCK", "/run/cairn/.daemon.sock")
MAX_INLINE = int(os.environ.get("CAIRN_TRANSCRIPT_INLINE_MAX", "2000000"))


def _default_gateway() -> str | None:
    """Return the default-route gateway IP from inside the container.

    Reads /proc/net/route directly to avoid relying on `ip` or `route` CLIs
    (absent in minimal container images like cpp-school's Dockerfile.dev).
    Format: header line, then per-route rows where Destination=00000000
    means the default route. Gateway is 8 hex chars in little-endian byte
    order — e.g. 010011AC → 172.17.0.1.
    """
    try:
        with open("/proc/net/route") as f:
            next(f, None)  # header
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "00000000":
                    gw_hex = parts[2]
                    if len(gw_hex) == 8:
                        return ".".join(str(int(gw_hex[i:i + 2], 16))
                                        for i in range(6, -1, -2))
    except (OSError, ValueError):
        pass
    return None


def _connect():
    """Return a connected socket to the cairn daemon, or None on failure."""
    if SOCK and os.path.exists(SOCK):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(120)
        s.connect(SOCK)
        return s
    host = TCP_HOST_OVERRIDE or _default_gateway()
    if not host:
        return None
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(120)
    s.connect((host, TCP_PORT))
    return s


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: cairn-hook.py <userpromptsubmit|stop|pretool|posttool>", file=sys.stderr)
        return 2
    route = sys.argv[1]
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as e:
        print(f"invalid hook payload json: {e}", file=sys.stderr)
        return 2

    tpath = payload.get("transcript_path")
    if tpath and os.path.isfile(tpath):
        try:
            if os.path.getsize(tpath) <= MAX_INLINE:
                with open(tpath, encoding="utf-8", errors="replace") as f:
                    payload["_transcript_body"] = f.read()
        except OSError:
            pass

    request = {"action": "hook", "route": route, "payload": payload}
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(120)
        client.connect(SOCK)
        client.sendall(json.dumps(request).encode())
        client.shutdown(socket.SHUT_WR)
        data = b""
        while chunk := client.recv(8192):
            data += chunk
        client.close()
    except (FileNotFoundError, ConnectionRefusedError, TimeoutError, OSError) as e:
        # Fail open — don't block the user's prompt if the host daemon is down.
        print(f"cairn-hook: daemon unreachable ({e}); skipping", file=sys.stderr)
        return 0

    try:
        resp = json.loads(data.decode())
    except json.JSONDecodeError as e:
        print(f"cairn-hook: invalid daemon response ({e})", file=sys.stderr)
        return 0

    if "error" in resp:
        print(f"cairn-hook: daemon error: {resp['error']}", file=sys.stderr)
        return 0

    sys.stdout.write(resp.get("stdout", ""))
    sys.stderr.write(resp.get("stderr", ""))
    return int(resp.get("exit_code", 0))


if __name__ == "__main__":
    sys.exit(main())
