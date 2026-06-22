"""Always-on sync services, launched by the cairn daemon when CAIRN_SYNC_ENABLED.

Three background threads:
  1. HTTPS sync server (/sync, /pair, /pair/status) on CAIRN_SYNC_PORT.
  2. Discovery advertiser — broadcasts a LAN beacon every interval.
  3. Discovery listener — records peer beacons into discovered_peers and
     periodically promotes approved outbound pairings (so a node starts pulling
     as soon as the peer approves it).

No user action beyond enabling sync: discovery is automatic, and the dashboard
Sync tab reads discovered_peers / pairing_requests / sync_peers.
"""

from __future__ import annotations

import socket
import threading
import time

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore


def lan_ip() -> str:
    """Best-effort primary LAN IPv4 (no packets actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _db_path() -> str:
    import os
    if os.environ.get("CAIRN_DB_PATH"):
        return os.environ["CAIRN_DB_PATH"]
    from cairn import init_db
    return init_db.DB_PATH


def start_sync_services():
    """Start the sync server + discovery threads. Returns the HTTP server (or
    None if disabled / failed). Safe to call once at daemon startup."""
    from cairn import config
    if not config.CAIRN_SYNC_ENABLED:
        return None
    from cairn.sync import SCHEMA_VERSION, identity, discovery
    from cairn.sync import server as sync_server
    from cairn.sync import client as sync_client

    db = _db_path()
    self_fp = identity.get_node_fingerprint()
    my_url = f"https://{lan_ip()}:{config.CAIRN_SYNC_PORT}"

    # 1. HTTPS sync server
    httpd = sync_server.build_server(config.CAIRN_SYNC_BIND, config.CAIRN_SYNC_PORT, db, tls=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # 2. Advertiser
    def _advertise():
        beacon = discovery.build_beacon(
            self_fp, identity.get_user_id(), my_url,
            identity.get_public_key_b64(), SCHEMA_VERSION,
            identity.get_tls_cert_fingerprint())
        while True:
            try:
                discovery.advertise_once(beacon, port=config.CAIRN_SYNC_DISCOVERY_PORT)
            except OSError:
                pass
            time.sleep(config.CAIRN_SYNC_ADVERTISE_INTERVAL)
    threading.Thread(target=_advertise, daemon=True).start()

    # 3. Listener + periodic outbound-approval promotion
    def _listen():
        conn = sqlite3.connect(db)
        def _store(beacon, addr):
            try:
                discovery.record_beacon(conn, beacon, self_node_id=self_fp)
            except Exception:
                pass
        while True:
            try:
                discovery.listen(duration=float(config.CAIRN_SYNC_ADVERTISE_INTERVAL),
                                 port=config.CAIRN_SYNC_DISCOVERY_PORT, on_beacon=_store)
            except OSError:
                time.sleep(5)
            try:
                sync_client.refresh_outbound(conn)
            except Exception:
                pass
    threading.Thread(target=_listen, daemon=True).start()

    print(f"cairn-sync services up: server {my_url}, discovery udp/{config.CAIRN_SYNC_DISCOVERY_PORT}",
          flush=True)
    return httpd
