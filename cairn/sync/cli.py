"""cairn-sync — operate the v2 peer-to-peer sync layer from the terminal.

    cairn-sync id                      show this node's sync identity
    cairn-sync serve --bind H:P        run the sync HTTP server (/sync, /pair)
    cairn-sync advertise --url URL     broadcast a LAN discovery beacon (loop)
    cairn-sync discover [--watch]      listen for peers; record to discovered_peers
    cairn-sync pair URL                send a pairing request to a peer
    cairn-sync requests [--all]        list pairing requests (pending by default)
    cairn-sync approve ID              approve a pairing request (pin its pubkey)
    cairn-sync deny ID                 deny a pairing request
    cairn-sync peers                   list paired peers
    cairn-sync revoke NODE_ID          revoke a paired peer
    cairn-sync pull [--peer NODE_ID]   pull changesets from approved peers

The CLI is also driven by the dashboard Sync tab — both call the same helpers.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore


def _db_path() -> str:
    import os
    if os.environ.get("CAIRN_DB_PATH"):
        return os.environ["CAIRN_DB_PATH"]
    from cairn import init_db
    return init_db.DB_PATH


def _conn():
    c = sqlite3.connect(_db_path())
    c.execute("PRAGMA busy_timeout=5000")
    return c


def _lan_ip() -> str:
    """Best-effort primary LAN IPv4 (no traffic actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def cmd_id(args) -> int:
    from cairn.sync import identity
    print(json.dumps({
        "node_id": identity.get_node_fingerprint(),
        "user_id": identity.get_user_id(),
        "public_key": identity.get_public_key_b64(),
        "provenance_node_id": identity.ensure_node_id(),
    }, indent=2))
    return 0


def cmd_serve(args) -> int:
    from cairn.sync.server import build_server
    host, port = args.bind.rsplit(":", 1)
    httpd = build_server(host, int(port), args.db)
    from cairn.sync import identity
    print(f"cairn-sync server on {args.bind} (node {identity.get_node_fingerprint()[:12]})",
          flush=True)
    httpd.serve_forever()
    return 0


def cmd_advertise(args) -> int:
    from cairn.sync import discovery, identity
    url = args.url or f"http://{_lan_ip()}:{args.sync_port}"
    beacon = discovery.build_beacon(
        identity.get_node_fingerprint(), identity.get_user_id(), url,
        identity.get_public_key_b64(),
        __import__("cairn.sync", fromlist=["SCHEMA_VERSION"]).SCHEMA_VERSION,
        identity.get_tls_cert_fingerprint())
    print(f"advertising {url} on udp/{args.port} every {args.interval}s (ctrl-c to stop)")
    try:
        while True:
            discovery.advertise_once(beacon, port=args.port)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmd_discover(args) -> int:
    from cairn.sync import discovery, identity
    self_fp = identity.get_node_fingerprint()
    conn = _conn()
    def _store(beacon, addr):
        if discovery.record_beacon(conn, beacon, self_node_id=self_fp):
            print(f"  + {beacon['node_id'][:12]} {beacon.get('user_id')} {beacon.get('url')}")
    if args.watch:
        print(f"watching udp/{args.port} for beacons (ctrl-c to stop)")
        try:
            while True:
                discovery.listen(duration=10.0, port=args.port, on_beacon=_store)
        except KeyboardInterrupt:
            return 0
    found = discovery.listen(duration=args.timeout, port=args.port, on_beacon=_store)
    print(json.dumps([{"node_id": b["node_id"], "user_id": b.get("user_id"),
                       "url": b.get("url")} for b in found], indent=2))
    return 0


def cmd_pair(args) -> int:
    from cairn.sync import client, identity
    my_url = args.my_url or f"http://{_lan_ip()}:{args.sync_port}"
    resp = client.send_pairing_request(args.url, my_url=my_url)
    print(json.dumps(resp, indent=2))
    if resp.get("ok"):
        print(f"\nYour fingerprint (read this to the host to verify):\n  {identity.get_node_fingerprint()}")
    return 0 if resp.get("ok") else 1


def cmd_requests(args) -> int:
    from cairn.sync import pairing
    reqs = pairing.list_pairing_requests(_conn(), pending_only=not args.all)
    for r in reqs:
        print(f"[{r['id']}] {r['status']:8} {r['direction']:8} {r['user_id'] or '?':20} "
              f"{r['peer_node_id'][:16]}  {r['url'] or ''}  ({r['source_ip'] or ''})")
    if not reqs:
        print("(no pairing requests)")
    return 0


def cmd_approve(args) -> int:
    from cairn.sync import pairing
    res = pairing.approve_pairing(_conn(), args.id, label=args.label)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


def cmd_deny(args) -> int:
    from cairn.sync import pairing
    res = pairing.deny_pairing(_conn(), args.id)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


def cmd_peers(args) -> int:
    from cairn.sync import pairing
    for p in pairing.list_peers(_conn()):
        print(f"{(p['status'] or 'approved'):9} {p['peer_node_id'][:16]} {p['label'] or '':20} "
              f"{p['url'] or '':28} last_ok={p['last_succeeded_at'] or '-'}")
    return 0


def cmd_revoke(args) -> int:
    from cairn.sync import pairing
    res = pairing.revoke_peer(_conn(), args.node_id)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


def cmd_session(args) -> int:
    from cairn.sync import client
    res = client.fetch_session(_conn(), args.origin_id)
    if not res.get("ok"):
        print(json.dumps(res, indent=2))
        return 1
    src = "local cache" if res.get("cached") else "peer"
    print(f"--- raw session for {args.origin_id} (from {src}) ---")
    if res.get("context_before"):
        print(res["context_before"])
    print(res.get("excerpt", ""))
    if res.get("context_after"):
        print(res["context_after"])
    return 0


def cmd_pull(args) -> int:
    from cairn.sync import client
    try:
        from cairn import embeddings
        embedder = embeddings
    except Exception:
        embedder = None
    conn = _conn()
    if args.peer:
        results = [client.pull_from_peer(conn, args.peer, embedder=embedder)]
    else:
        results = client.pull_all(conn, embedder=embedder)
    for r in results:
        status = "ok" if r.ok else f"FAIL: {r.error}"
        print(f"{r.peer_node_id[:16]}  {status}  rows={r.row_counts}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cairn-sync", description="Cairn peer-to-peer sync (v2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("id", help="show this node's sync identity").set_defaults(fn=cmd_id)

    sp = sub.add_parser("serve", help="run the sync HTTP server")
    sp.add_argument("--bind", default="0.0.0.0:8787")
    sp.add_argument("--db", default=None)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("advertise", help="broadcast LAN discovery beacon")
    sp.add_argument("--url", default=None, help="this node's reachable sync URL")
    sp.add_argument("--sync-port", type=int, default=8787)
    sp.add_argument("--port", type=int, default=47391, help="UDP discovery port")
    sp.add_argument("--interval", type=float, default=15.0)
    sp.set_defaults(fn=cmd_advertise)

    sp = sub.add_parser("discover", help="listen for peer beacons")
    sp.add_argument("--timeout", type=float, default=5.0)
    sp.add_argument("--port", type=int, default=47391)
    sp.add_argument("--watch", action="store_true", help="keep listening")
    sp.set_defaults(fn=cmd_discover)

    sp = sub.add_parser("pair", help="send a pairing request to a peer URL")
    sp.add_argument("url")
    sp.add_argument("--my-url", default=None)
    sp.add_argument("--sync-port", type=int, default=8787)
    sp.set_defaults(fn=cmd_pair)

    sp = sub.add_parser("requests", help="list pairing requests")
    sp.add_argument("--all", action="store_true", help="include decided")
    sp.set_defaults(fn=cmd_requests)

    sp = sub.add_parser("approve", help="approve a pairing request")
    sp.add_argument("id", type=int)
    sp.add_argument("--label", default=None)
    sp.set_defaults(fn=cmd_approve)

    sp = sub.add_parser("deny", help="deny a pairing request")
    sp.add_argument("id", type=int)
    sp.set_defaults(fn=cmd_deny)

    sub.add_parser("peers", help="list paired peers").set_defaults(fn=cmd_peers)

    sp = sub.add_parser("revoke", help="revoke a paired peer")
    sp.add_argument("node_id")
    sp.set_defaults(fn=cmd_revoke)

    sp = sub.add_parser("pull", help="pull changesets from peers")
    sp.add_argument("--peer", default=None)
    sp.set_defaults(fn=cmd_pull)

    sp = sub.add_parser("session", help="fetch the raw session behind a synced memory")
    sp.add_argument("origin_id")
    sp.set_defaults(fn=cmd_session)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
