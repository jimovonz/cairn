"""Headless end-to-end check of the live cairn proxy against a fake upstream.

Not auto-collected (no test_ prefix); run directly:
    python3 tests/proxy_e2e_check.py
Proves the real daemon strips [cm] from the streamed response, the client sees
clean text, and the capture sidecar receives the verbatim block.
"""
import json, os, sys, time, threading, subprocess, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cairn.proxy import sidecar

UP_PORT = 8911
PROXY_PORT = 8912
SESSION = "e2e-check-session"

BODY = "Here is the streamed answer, line one.\nLine two of the reply."
CM = "\n\n[cm]: # '{\"e\":[{\"t\":\"fact\",\"to\":\"e2e\",\"c\":\"end to end works\"}],\"ok\":true,\"ctx\":\"s\",\"kw\":[\"e2e\"]}'"


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def start_fake_upstream():
    import asyncio
    from aiohttp import web

    async def handler(request):
        await request.read()
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(_sse("message_start", {"type": "message_start",
                         "message": {"usage": {"input_tokens": 5, "cache_read_input_tokens": 0}}}))
        await resp.write(_sse("content_block_start", {"type": "content_block_start", "index": 0,
                         "content_block": {"type": "text", "text": ""}}))
        for piece in (BODY[:20], BODY[20:], CM):
            await resp.write(_sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": piece}}))
        await resp.write(_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
        await resp.write(_sse("message_delta", {"type": "message_delta",
                         "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}))
        await resp.write(_sse("message_stop", {"type": "message_stop"}))
        await resp.write_eof()
        return resp

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", UP_PORT)
    loop.run_until_complete(site.start())
    loop.run_forever()


def main():
    # clean sidecar
    for suf in ("_cm_capture.jsonl", "_inject_prompt.txt", "_inject_bootstrap.txt"):
        try: os.unlink(sidecar._path(SESSION, suf))
        except FileNotFoundError: pass

    threading.Thread(target=start_fake_upstream, daemon=True).start()
    time.sleep(1.0)

    env = dict(os.environ, CAIRN_PROXY_ENABLED="1", CAIRN_PROXY_PORT=str(PROXY_PORT),
               CAIRN_PROXY_UPSTREAM=f"http://127.0.0.1:{UP_PORT}")
    proxy = subprocess.Popen([sys.executable, "-m", "cairn.proxy.server", "serve",
                              "--port", str(PROXY_PORT)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2.0)
        req = urllib.request.Request(
            f"http://127.0.0.1:{PROXY_PORT}/v1/messages",
            data=json.dumps({"model": "x", "max_tokens": 10,
                             "messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json", "X-Claude-Code-Session-Id": SESSION},
            method="POST")
        client_text = ""
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode()
        for blk in raw.split("\n\n"):
            for line in blk.split("\n"):
                if line.startswith("data: "):
                    try: d = json.loads(line[6:])
                    except Exception: continue
                    if d.get("type") == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta":
                        client_text += d["delta"]["text"]

        ok = True
        if "[cm]" in client_text:
            print("FAIL: client saw [cm]"); ok = False
        if client_text != BODY:
            print(f"FAIL: client text != body\n  got: {client_text!r}"); ok = False
        cm_map = sidecar.load_cm_map(SESSION)
        if not any("[cm]" in v for v in cm_map.values()):
            print("FAIL: capture sidecar missing cm"); ok = False
        print("client_text:", repr(client_text))
        print("captured cm:", repr(next(iter(cm_map.values()), "")[:60]))
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        proxy.terminate()
        try: proxy.wait(timeout=5)
        except Exception: proxy.kill()
        for suf in ("_cm_capture.jsonl",):
            try: os.unlink(sidecar._path(SESSION, suf))
            except FileNotFoundError: pass


if __name__ == "__main__":
    sys.exit(main())
