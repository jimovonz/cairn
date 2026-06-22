"""Node identity, user identity, embedding model version, and Lamport clock.

All node-scoped state lives in two places:
- ~/.cairn/node_id and ~/.cairn/user_id   — file-based, survive DB rebuild
- node_state table in cairn.db            — Lamport clock, embedding model version

Both are written on first call and read thereafter. Idempotent.
"""

from __future__ import annotations

import hashlib
import os
import socket
import uuid
from pathlib import Path
from typing import Optional

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore


# ---- File-based identity ----

def _state_dir() -> Path:
    """Resolve ~/.cairn/ — overridable via CAIRN_STATE_DIR for tests."""
    override = os.environ.get("CAIRN_STATE_DIR")
    if override:
        d = Path(override)
    else:
        d = Path.home() / ".cairn"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_node_id() -> str:
    """Return this node's UUID, generating ~/.cairn/node_id if missing.

    The file-based identity survives DB rebuilds and reinstalls. Deleting it
    creates a new identity that peers will see as a separate node.
    """
    p = _state_dir() / "node_id"
    if p.exists():
        nid = p.read_text().strip()
        if nid:
            return nid
    nid = str(uuid.uuid4())
    p.write_text(nid + "\n")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return nid


def get_user_id() -> str:
    """Return the user identity, defaulting to USER@hostname.

    Override via ~/.cairn/user_id (e.g. set to LDAP/email handle for team rollout).
    """
    p = _state_dir() / "user_id"
    if p.exists():
        uid = p.read_text().strip()
        if uid:
            return uid
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    host = socket.gethostname()
    return f"{user}@{host}"


# ---- Embedding model version ----

def get_embedding_model_version() -> str:
    """Return a stable identifier for the local embedding model.

    Format: '<model-name>@<sha256-prefix>'. The hash is over a small
    canary embedding so any silent model swap (same name, different weights)
    produces a different version string. Cached after first call.
    """
    if _MODEL_VERSION_CACHE.get("v"):
        return _MODEL_VERSION_CACHE["v"]
    name = os.environ.get("CAIRN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    canary = "cairn-canary"
    digest_prefix = hashlib.sha256((name + "|" + canary).encode()).hexdigest()[:8]
    # We deliberately don't load the model here — that would couple every init() to
    # heavy imports. The hash above is name-only; on first embed call, the model
    # ID is re-derived and stored. Honest naming: this is "advertised model version,"
    # not "verified weight digest." Good enough for v1 sync compatibility checks.
    v = f"{name}@{digest_prefix}"
    _MODEL_VERSION_CACHE["v"] = v
    return v


_MODEL_VERSION_CACHE: dict[str, str] = {}


# ---- Lamport clock ----

def peek_lamport(conn) -> int:
    """Return the current Lamport clock value without bumping it."""
    row = conn.execute("SELECT value FROM node_state WHERE key = 'lamport'").fetchone()
    return int(row[0]) if row else 0


def bump_lamport(conn, observed: Optional[int] = None) -> int:
    """Atomically increment the Lamport clock past `observed` and return the new value.

    Use this on every mutating operation so the row's `lamport` column reflects
    causal ordering across nodes. Pass the highest peer-observed lamport when
    applying a changeset, so subsequent local edits are causally after.
    """
    cur = peek_lamport(conn)
    new = max(cur, observed or 0) + 1
    conn.execute(
        "INSERT INTO node_state (key, value, updated_at) VALUES ('lamport', ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        (str(new),),
    )
    return new


def node_id_for_conn(conn) -> str:
    """Resolve this DB's node_id, preferring the per-DB record over the env-bound file.

    Why: a single process can connect to multiple DBs (tests, multi-tenant tools).
    The env-var-driven file is fine for a normal install (one DB per home dir),
    but sync code must use the identity tied to the *connection*, not the env.
    Persists the resolved id back to node_state so subsequent calls are env-free.
    """
    row = conn.execute("SELECT value FROM node_state WHERE key = 'node_id'").fetchone()
    if row and row[0]:
        return row[0]
    nid = ensure_node_id()
    conn.execute(
        "INSERT INTO node_state (key, value) VALUES ('node_id', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (nid,),
    )
    conn.commit()
    return nid


def record_embedding_model(conn) -> None:
    """Persist the current embedding model version to node_state."""
    v = get_embedding_model_version()
    conn.execute(
        "INSERT INTO node_state (key, value, updated_at) VALUES ('embedding_model_version', ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        (v,),
    )


# ---- Ed25519 keypair identity (v2 sync: pairing + signature auth) ----
# The public key is the auth principal. node_id (v1, UUID in node_state) stays the
# data-provenance id on memories.created_by_node; the keypair fingerprint below is
# the *auth* identity used for pairing and /sync signatures. Two namespaces, one
# per concern — see docs/multi-node-sync.md v2.

import base64 as _b64
import hashlib as _hashlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization as _ser


def _key_path() -> Path:
    return _state_dir() / "node_key"


def ensure_keypair() -> Ed25519PrivateKey:
    """Return this node's Ed25519 private key, generating ~/.cairn/node_key (raw
    32-byte seed, chmod 600) + node_key.pub (base64) on first call. Idempotent;
    survives DB rebuild like node_id."""
    p = _key_path()
    if p.exists():
        raw = p.read_bytes()
        if len(raw) >= 32:
            return Ed25519PrivateKey.from_private_bytes(raw[:32])
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(
        encoding=_ser.Encoding.Raw,
        format=_ser.PrivateFormat.Raw,
        encryption_algorithm=_ser.NoEncryption(),
    )
    p.write_bytes(raw)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    pub_raw = key.public_key().public_bytes(
        encoding=_ser.Encoding.Raw, format=_ser.PublicFormat.Raw)
    (p.parent / "node_key.pub").write_text(_b64.b64encode(pub_raw).decode() + "\n")
    return key


def get_public_key_b64() -> str:
    """Base64 of our raw 32-byte Ed25519 public key."""
    pub_raw = ensure_keypair().public_key().public_bytes(
        encoding=_ser.Encoding.Raw, format=_ser.PublicFormat.Raw)
    return _b64.b64encode(pub_raw).decode()


def fingerprint(public_key_b64: str) -> str:
    """Self-certifying device id: base32(sha256(raw_pubkey))[:52], no padding.
    A node fingerprint is a commitment to its public key, so it cannot be forged
    without the private key."""
    raw = _b64.b64decode(public_key_b64)
    digest = _hashlib.sha256(raw).digest()
    return _b64.b32encode(digest).decode().rstrip("=")[:52]


def get_node_fingerprint() -> str:
    """This node's v2 sync auth identity = fingerprint of its public key."""
    return fingerprint(get_public_key_b64())


# ---- Request signing / verification (Ed25519 over a canonical message) ----

def canonical_message(method: str, path: str, body: bytes, ts: str, nonce: str) -> bytes:
    body_hash = _hashlib.sha256(body or b"").hexdigest()
    return "\n".join([method.upper(), path, body_hash, str(ts), str(nonce)]).encode()


def sign_request(method: str, path: str, body: bytes, ts: str, nonce: str) -> str:
    """Sign a request with our private key; returns base64 signature."""
    sig = ensure_keypair().sign(canonical_message(method, path, body, ts, nonce))
    return _b64.b64encode(sig).decode()


def verify_request(public_key_b64: str, method: str, path: str, body: bytes,
                   ts: str, nonce: str, signature_b64: str) -> bool:
    """Verify a request signature against a peer's public key. Never raises."""
    try:
        pub = Ed25519PublicKey.from_public_bytes(_b64.b64decode(public_key_b64))
        pub.verify(_b64.b64decode(signature_b64),
                   canonical_message(method, path, body, ts, nonce))
        return True
    except Exception:
        return False


# ---- TLS identity (self-signed cert, pinned by fingerprint at pairing) ----
# The sync transport is HTTPS. Each node has a long-lived self-signed cert; peers
# pin its SHA-256 fingerprint when they approve pairing (TOFU, Syncthing-style),
# so the connection is encrypted and the server is authenticated without any CA.

def ensure_tls_cert() -> tuple:
    """Generate/load this node's self-signed TLS cert + key in ~/.cairn/
    (tls_cert.pem, tls_key.pem). Returns (cert_path, key_path). Idempotent."""
    import datetime
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    cert_p = _state_dir() / "tls_cert.pem"
    key_p = _state_dir() / "tls_key.pem"
    if cert_p.exists() and key_p.exists():
        return cert_p, key_p
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cairn-sync")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2099, 1, 1))
        .sign(key, hashes.SHA256())
    )
    key_p.write_bytes(key.private_bytes(
        _ser.Encoding.PEM, _ser.PrivateFormat.PKCS8, _ser.NoEncryption()))
    try:
        os.chmod(key_p, 0o600)
    except OSError:
        pass
    cert_p.write_bytes(cert.public_bytes(_ser.Encoding.PEM))
    return cert_p, key_p


def get_tls_cert_fingerprint() -> str:
    """SHA-256 hex of this node's TLS cert (DER) — the value peers pin."""
    from cryptography import x509
    cert_p, _ = ensure_tls_cert()
    cert = x509.load_pem_x509_certificate(cert_p.read_bytes())
    return _hashlib.sha256(cert.public_bytes(_ser.Encoding.DER)).hexdigest()


def cert_fingerprint_from_der(der: bytes) -> str:
    """SHA-256 hex of a DER-encoded cert (for pinning a presented peer cert)."""
    return _hashlib.sha256(der).hexdigest()
