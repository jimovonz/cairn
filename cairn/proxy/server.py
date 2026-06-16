#!/usr/bin/env python3
"""cairn-proxy — reverse proxy between Claude Code and the Anthropic API.

Strips every Cairn artifact from the response stream and injects Cairn context
into the request, so the user sees a clean conversation while Cairn keeps
working and the Anthropic prompt cache stays byte-exact.

Fail-open by design: any error in the Cairn rewrite path degrades to a byte-exact
passthrough; if the daemon is down, Claude Code talks to Anthropic directly.

Commands: serve | start | stop | status | restart
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cairn import config
from cairn.proxy.response_filter import CairnResponseFilter
from cairn.proxy.request_inject import reinject_cm, inject_bootstrap, inject_prompt_context
from cairn.proxy import sidecar

UPSTREAM = config.PROXY_UPSTREAM
_PROXY_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(os.path.dirname(_PROXY_DIR), "proxy.log")       # cairn/proxy.log


def _pid_file(port: int) -> str:
    # Port-specific so daemons on different ports (e.g. test serve instances)
    # never clobber each other's PID tracking. cairn/.proxy-<port>.pid
    return os.path.join(os.path.dirname(_PROXY_DIR), f".proxy-{port}.pid")

logger = logging.getLogger("cairn-proxy")


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_session_id(headers: dict) -> str:
    for k, v in headers.items():
        if k.lower() == "x-claude-code-session-id":
            return v
    return ""


# --- UsageTracker: passthrough token accounting (proves cache_read) ----------
class UsageTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_create = 0
        self.buffer = b""

    def process_chunk(self, chunk: bytes) -> None:
        self.buffer += chunk
        while b"\n\n" in self.buffer:
            end = self.buffer.find(b"\n\n") + 2
            self._parse(self.buffer[:end]); self.buffer = self.buffer[end:]

    def flush(self) -> None:
        if self.buffer:
            self._parse(self.buffer); self.buffer = b""

    def _parse(self, event: bytes) -> None:
        try:
            text = event.decode("utf-8")
        except UnicodeDecodeError:
            return
        for line in text.split("\n"):
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except (json.JSONDecodeError, ValueError):
                continue
            if data.get("type") == "message_start":
                u = data.get("message", {}).get("usage", {})
                self.cache_read = u.get("cache_read_input_tokens", 0)
                self.cache_create = u.get("cache_creation_input_tokens", 0)
                self.input_tokens = u.get("input_tokens", 0) + self.cache_read + self.cache_create
            elif data.get("type") == "message_delta":
                self.output_tokens = data.get("usage", {}).get("output_tokens", self.output_tokens)


def _rewrite_request(body: bytes, session_id: str) -> bytes:
    """Apply Cairn request injection. Fail-open: return original body on any error."""
    if not (body and session_id and config.PROXY_REWRITE):
        return body
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body
    if not isinstance(data, dict) or "messages" not in data:
        return body
    try:
        # Re-injecting verbatim [cm] into assistant turns is harmless on any
        # request (it only matches turns this session generated).
        reinject_cm(data, sidecar.load_cm_map(session_id))
        # Cairn context must land on the REAL agentic turn, not on Claude Code's
        # auxiliary calls (title/topic generation), which also hit /v1/messages
        # but carry no tool set. Gating on `tools` stops an auxiliary request
        # from consuming the staged context meant for the user turn. Only consume
        # the one-shot per-prompt sidecar when we are actually going to inject it.
        if data.get("tools"):
            inject_bootstrap(data, sidecar.read_bootstrap(session_id))
            inject_prompt_context(data, sidecar.consume_prompt_context(session_id))
        return json.dumps(data).encode("utf-8")
    except Exception as exc:  # fail-open
        logger.warning("request rewrite failed, passing through: %s", exc)
        return body


def run_proxy(port: int, debug: bool) -> None:
    import aiohttp
    from aiohttp import web

    class CairnProxy:
        def __init__(self):
            self.app = web.Application(client_max_size=200 * 1024 * 1024)
            self.app.router.add_route("*", "/{path:.*}", self.handle)
            self.session = None

        async def _ensure_session(self):
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()

        async def handle(self, request):
            await self._ensure_session()
            path = request.path
            if request.query_string:
                path = f"{path}?{request.query_string}"
            target = f"{UPSTREAM}{path}"
            session_id = get_session_id(dict(request.headers))
            body = await request.read()

            skip = {"host", "content-length", "transfer-encoding", "connection"}
            headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}

            # Cairn request injection (only message-create calls; fail-open).
            is_messages = request.method == "POST" and "/v1/messages" in request.path
            if is_messages:
                body = _rewrite_request(body, session_id)

            try:
                async with self.session.request(
                    method=request.method, url=target, headers=headers, data=body, ssl=True,
                ) as up:
                    ctype = up.headers.get("content-type", "")
                    streaming = "text/event-stream" in ctype
                    resp_headers = {
                        k: v for k, v in up.headers.items()
                        if k.lower() not in {"transfer-encoding", "content-encoding", "content-length"}
                    }
                    logger.info("%s %s -> %s (streaming=%s session=%s)",
                                request.method, request.path, up.status, streaming, session_id[:8])

                    if not streaming:
                        return web.Response(status=up.status, headers=resp_headers, body=await up.read())

                    response = web.StreamResponse(status=up.status, headers=resp_headers)
                    await response.prepare(request)
                    usage = UsageTracker()
                    do_filter = is_messages and config.PROXY_REWRITE
                    filt = CairnResponseFilter() if do_filter else None

                    async for chunk in up.content.iter_any():
                        usage.process_chunk(chunk)
                        if filt is not None:
                            try:
                                out = filt.process_chunk(chunk)
                            except Exception as exc:  # fail-open mid-stream
                                logger.warning("response filter error, passthrough rest: %s", exc)
                                filt = None
                                await response.write(chunk)
                                continue
                            if out:
                                await response.write(out)
                        else:
                            await response.write(chunk)

                    if filt is not None:
                        try:
                            tail = filt.flush()
                            if tail:
                                await response.write(tail)
                            rec = filt.finalize_record()
                            if session_id:
                                sidecar.append_capture(session_id, rec)
                        except Exception as exc:
                            logger.warning("response finalize error: %s", exc)
                    usage.flush()
                    await response.write_eof()
                    if usage.input_tokens:
                        logger.info("usage session=%s input=%d cache_read=%d cache_create=%d output=%d",
                                    session_id[:8], usage.input_tokens, usage.cache_read,
                                    usage.cache_create, usage.output_tokens)
                    return response

            except aiohttp.ClientError as exc:
                logger.error("upstream error: %s", exc)
                return web.Response(status=502, text=f"cairn-proxy upstream error: {exc}")

        async def _cleanup(self, app):
            if self.session and not self.session.closed:
                await self.session.close()

        def run(self):
            self.app.on_cleanup.append(self._cleanup)
            logger.info("cairn-proxy on %s:%d -> %s", config.PROXY_HOST, port, UPSTREAM)
            web.run_app(self.app, host=config.PROXY_HOST, port=port,
                        print=lambda *_: None)

    CairnProxy().run()


# --- daemon management -------------------------------------------------------
def _read_pid(port: int):
    try:
        return int(open(_pid_file(port)).read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_pid(port: int):
    with open(_pid_file(port), "w") as fh:
        fh.write(str(os.getpid()))


def _remove_pid(port: int):
    try:
        os.unlink(_pid_file(port))
    except FileNotFoundError:
        pass


def is_running(port: int) -> bool:
    pid = _read_pid(port)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        _remove_pid(port)
        return False


def cmd_serve(args):
    setup_logging(args.debug)
    _write_pid(args.port)
    signal.signal(signal.SIGINT, lambda *_: (_remove_pid(args.port), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (_remove_pid(args.port), sys.exit(0)))
    try:
        run_proxy(args.port, args.debug)
    finally:
        _remove_pid(args.port)


def cmd_start(args):
    if is_running(args.port):
        print(f"cairn-proxy already running (PID {_read_pid(args.port)})")
        return
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("Error: aiohttp required. Install: pip install aiohttp")
        sys.exit(1)
    if os.fork() > 0:
        print(f"Started cairn-proxy on {config.PROXY_HOST}:{args.port}")
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdin.close()
    out = open(LOG_FILE, "a")
    sys.stdout = out
    sys.stderr = out
    setup_logging(args.debug)
    _write_pid(args.port)
    signal.signal(signal.SIGTERM, lambda *_: (_remove_pid(args.port), sys.exit(0)))
    try:
        run_proxy(args.port, args.debug)
    finally:
        _remove_pid(args.port)


def cmd_stop(args):
    pid = _read_pid(args.port)
    if pid is None:
        print("cairn-proxy not running")
        return
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped cairn-proxy (PID {pid})")
    except OSError as exc:
        print(f"Error stopping: {exc}")
    _remove_pid(args.port)


def cmd_status(args):
    if is_running(args.port):
        print(f"cairn-proxy running (PID {_read_pid(args.port)}) on {config.PROXY_HOST}:{args.port}")
        print(f"  log: {LOG_FILE}")
    else:
        print("cairn-proxy not running")
        sys.exit(1)


def cmd_restart(args):
    if is_running(args.port):
        cmd_stop(args)
        import time
        time.sleep(1)
    cmd_start(args)


def main():
    p = argparse.ArgumentParser(description="cairn-proxy")
    sub = p.add_subparsers(dest="command")
    for name, fn in (("serve", cmd_serve), ("start", cmd_start), ("restart", cmd_restart)):
        sp = sub.add_parser(name)
        sp.add_argument("--port", type=int, default=config.PROXY_PORT)
        sp.add_argument("--debug", action="store_true")
        sp.set_defaults(func=fn)
    for name, fn in (("stop", cmd_stop), ("status", cmd_status)):
        sp = sub.add_parser(name)
        sp.add_argument("--port", type=int, default=config.PROXY_PORT)
        sp.set_defaults(func=fn)
    args = p.parse_args()
    if not args.command:
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
