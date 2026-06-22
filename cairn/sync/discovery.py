"""LAN peer discovery via UDP broadcast.

A node opts in (CAIRN_SYNC_ADVERTISE=1) and periodically broadcasts a small JSON
beacon on the discovery port. Listeners record beacons into `discovered_peers`
so the dashboard can offer one-click outbound pairing. Beacons are
unauthenticated and carry only public info — discovery makes a node *visible*,
not *trusted* (no access without pairing approval). mDNS is a future enhancement;
UDP broadcast is the dependable default. See docs/multi-node-sync.md v2.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Callable, Optional

DISCOVERY_PORT = 47391
BEACON_MAGIC = "cairn-sync-beacon/1"


def build_beacon(node_id: str, user_id: str, url: str, public_key: str,
                 schema_version: int, cert_fingerprint: str = "") -> bytes:
    return json.dumps({
        "magic": BEACON_MAGIC,
        "node_id": node_id,
        "user_id": user_id,
        "url": url,
        "public_key": public_key,
        "schema_version": schema_version,
        "cert_fingerprint": cert_fingerprint,
    }).encode("utf-8")


def parse_beacon(data: bytes) -> Optional[dict]:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict) or obj.get("magic") != BEACON_MAGIC:
        return None
    if not obj.get("node_id"):
        return None
    return obj


def advertise_once(beacon: bytes, *, port: int = DISCOVERY_PORT,
                   addr: str = "255.255.255.255") -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(beacon, (addr, port))
    finally:
        s.close()


def record_beacon(conn, beacon: dict, *, self_node_id: Optional[str] = None) -> bool:
    """Upsert a discovered peer. Ignores our own beacon. Returns True if stored."""
    nid = beacon.get("node_id")
    if not nid or nid == self_node_id:
        return False
    conn.execute(
        "INSERT INTO discovered_peers "
        "(node_id, user_id, url, public_key, schema_version, cert_fingerprint, last_seen) "
        "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(node_id) DO UPDATE SET user_id = excluded.user_id, "
        "url = excluded.url, public_key = excluded.public_key, "
        "schema_version = excluded.schema_version, "
        "cert_fingerprint = excluded.cert_fingerprint, last_seen = CURRENT_TIMESTAMP",
        (nid, beacon.get("user_id"), beacon.get("url"), beacon.get("public_key"),
         beacon.get("schema_version"), beacon.get("cert_fingerprint")),
    )
    conn.commit()
    return True


def listen(duration: float = 5.0, *, port: int = DISCOVERY_PORT,
           on_beacon: Optional[Callable[[dict, tuple], None]] = None) -> list[dict]:
    """Listen for beacons for `duration` seconds; return the parsed beacons.

    If `on_beacon` is given it is called per beacon as (beacon, addr)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    found: list[dict] = []
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        s.bind(("", port))
        s.settimeout(0.5)
        end = time.time() + duration
        while time.time() < end:
            try:
                data, addr = s.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            beacon = parse_beacon(data)
            if beacon:
                found.append(beacon)
                if on_beacon:
                    on_beacon(beacon, addr)
    finally:
        s.close()
    return found


def list_discovered(conn, *, max_age_sec: Optional[int] = None) -> list[dict]:
    cols = ["node_id", "user_id", "url", "public_key", "schema_version",
            "cert_fingerprint", "first_seen", "last_seen"]
    q = f"SELECT {', '.join(cols)} FROM discovered_peers"
    if max_age_sec:
        q += f" WHERE last_seen >= datetime('now', '-{int(max_age_sec)} seconds')"
    q += " ORDER BY last_seen DESC"
    return [dict(zip(cols, r)) for r in conn.execute(q).fetchall()]
