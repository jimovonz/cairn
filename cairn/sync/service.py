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
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err


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


# Runtime control state — lets the dashboard toggle sync on/off live (and lets
# pairing auto-enable it) without restarting the daemon. Holds the running httpd
# + a stop Event the loops poll so stop_sync_services can tear them down.
_SYNC_STATE: dict = {"httpd": None, "stop": None}


def sync_running() -> bool:
    """True if sync services are currently up in this process."""
    return _SYNC_STATE.get("httpd") is not None


def stop_sync_services() -> bool:
    """Tear down running sync services (server + advertise/listen/pull loops).
    Returns True if something was stopped. Idempotent."""
    stop = _SYNC_STATE.get("stop")
    httpd = _SYNC_STATE.get("httpd")
    if stop is None and httpd is None:
        return False
    if stop is not None:
        stop.set()
    if httpd is not None:
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass
    _SYNC_STATE["httpd"] = None
    _SYNC_STATE["stop"] = None
    print("cairn-sync services stopped", flush=True)
    return True


def set_sync_enabled(enabled: bool, *, session_only: bool = False) -> dict:
    """Toggle sync on/off and (unless session_only) persist to cairn/.env so the
    choice survives a daemon restart. Signals the running daemon to start/stop
    sync services immediately via the sync_start/sync_stop opcodes. Single control
    point behind the dashboard toggle and pairing auto-enable.
    Returns {enabled, persisted, signalled, running}."""
    from cairn import config
    persisted = False
    if not session_only:
        try:
            config.set_env_kv("CAIRN_SYNC_ENABLED", "1" if enabled else "0")
            persisted = True
        except Exception:
            pass
    import os
    os.environ["CAIRN_SYNC_ENABLED"] = "1" if enabled else "0"
    signalled = running = False
    try:
        from cairn import daemon
        if daemon.is_running():
            resp = daemon.send_request(
                {"action": "sync_start" if enabled else "sync_stop"})
            signalled = bool(resp and resp.get("ok"))
            running = bool(resp and resp.get("running"))
        else:
            # No daemon: reflect the in-process services directly.
            if enabled:
                start_sync_services(force=True)
            else:
                stop_sync_services()
            running = sync_running()
            signalled = True
    except Exception:
        pass
    return {"enabled": enabled, "persisted": persisted,
            "signalled": signalled, "running": running}


def start_sync_services(force: bool = False):
    """Start the sync server + discovery/pull threads. Returns the HTTP server
    (or None if disabled / failed). Idempotent — a second call while already
    running is a no-op. `force=True` starts even when CAIRN_SYNC_ENABLED is unset
    (used when the dashboard / pairing enables sync live at runtime)."""
    from cairn import config
    if not force and not config.CAIRN_SYNC_ENABLED:
        return None
    if sync_running():
        return _SYNC_STATE["httpd"]
    from cairn.sync import SCHEMA_VERSION, identity, discovery
    from cairn.sync import server as sync_server
    from cairn.sync import client as sync_client

    db = _db_path()
    self_fp = identity.get_node_fingerprint()
    my_url = f"https://{lan_ip()}:{config.CAIRN_SYNC_PORT}"
    stop = threading.Event()

    # 1. HTTPS sync server
    httpd = sync_server.build_server(config.CAIRN_SYNC_BIND, config.CAIRN_SYNC_PORT, db, tls=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # 2. Advertiser
    def _advertise():
        beacon = discovery.build_beacon(
            self_fp, identity.get_user_id(), my_url,
            identity.get_public_key_b64(), SCHEMA_VERSION,
            identity.get_tls_cert_fingerprint())
        while not stop.is_set():
            try:
                discovery.advertise_once(beacon, port=config.CAIRN_SYNC_DISCOVERY_PORT)
            except OSError:
                pass
            stop.wait(config.CAIRN_SYNC_ADVERTISE_INTERVAL)
    threading.Thread(target=_advertise, daemon=True).start()

    # 3. Listener + periodic outbound-approval promotion
    def _listen():
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA busy_timeout=5000")
        def _store(beacon, addr):
            try:
                discovery.record_beacon(conn, beacon, self_node_id=self_fp)
            except Exception:
                pass
        while not stop.is_set():
            try:
                discovery.listen(duration=float(config.CAIRN_SYNC_ADVERTISE_INTERVAL),
                                 port=config.CAIRN_SYNC_DISCOVERY_PORT, on_beacon=_store)
            except OSError:
                stop.wait(5)
            if stop.is_set():
                break
            try:
                sync_client.refresh_outbound(conn)
            except Exception:
                pass
    threading.Thread(target=_listen, daemon=True).start()

    # 4. Periodic pull — actually sync memories from every approved peer on an
    # interval (this is what makes "periodically sync with approved peers" real;
    # discovery/approval above only establish trust). Each peer serves only its
    # own data (extract_changeset is own-data-only), so a full picture requires
    # pairing with each source directly. pull_all presence-gates so an offline
    # peer is skipped, never dialled.
    def _pull():
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            from cairn import embeddings as _emb
        except Exception:
            _emb = None
        while not stop.is_set():
            stop.wait(config.CAIRN_SYNC_PULL_INTERVAL)
            if stop.is_set():
                break
            try:
                sync_client.pull_all(conn, embedder=_emb)
            except Exception:
                pass
    threading.Thread(target=_pull, daemon=True).start()

    _SYNC_STATE["httpd"] = httpd
    _SYNC_STATE["stop"] = stop
    print(f"cairn-sync services up: server {my_url}, discovery udp/{config.CAIRN_SYNC_DISCOVERY_PORT}",
          flush=True)
    return httpd
