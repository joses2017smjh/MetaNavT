"""Operator-issued capabilities. Issuance is never registered as an agent tool.

The host protects the key and SQLite ledger outside every agent-readable root.
Anyone able to read that key or execute arbitrary code as the operator is outside
this boundary. A token authorizes exactly one immutable review in one session.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import time


class ApprovalError(PermissionError):
    pass


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def review_hash(review: dict) -> str:
    return hashlib.sha256(canonical(review)).hexdigest()


def create_key(path: Path) -> None:
    """Explicit operator setup; refuses to overwrite an existing key."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, secrets.token_bytes(32))
        os.fsync(fd)
    finally:
        os.close(fd)


def _key(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
            raise ApprovalError("operator key must be a private regular file (mode 0600)")
        key = os.read(fd, 4096)
    finally:
        os.close(fd)
    if len(key) != 32:
        raise ApprovalError("operator key must contain exactly 32 random bytes")
    return key


def sign_review(review: dict, key_file: Path, *, actor: str, ttl_seconds: float = 120,
                now: float | None = None) -> str:
    """Trusted operator path, separate from FilesystemTools and both MCP servers.

    The operator must inspect the complete review before invoking this function.
    It signs reviewed data; it does not execute actions or resolve file paths.
    """
    if not actor or not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= 3600:
        raise ApprovalError("actor and an expiry between 0 and 3600 seconds are required")
    if review.get("schema") != "metanavit-reviewed-action-v1":
        raise ApprovalError("unsupported review schema")
    issued = time.time() if now is None else now
    body = canonical({"review_sha256": review_hash(review), "grant_id": secrets.token_hex(24),
                      "issued_at": issued, "expires_at": issued + ttl_seconds, "actor": actor})
    sig = hmac.new(_key(Path(key_file)), body, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(body).decode().rstrip("=") + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")


class ApprovalVerifier:
    """Verifies capabilities and durably consumes grant IDs before mutation.

    SQLite UNIQUE + BEGIN IMMEDIATE makes replay consumption transactional across
    worker threads/processes using this ledger. Failed mutations burn the grant.
    """
    def __init__(self, root: Path, key_file: Path, ledger_file: Path):
        root = Path(root).resolve()
        for path in (Path(key_file), Path(ledger_file)):
            resolved = path.resolve()
            if resolved == root or root in resolved.parents:
                raise ApprovalError("approval key and ledger must be outside the corpus root")
            if path.is_symlink():
                raise ApprovalError("approval state must not be a symlink")
        self._key = _key(Path(key_file))
        self.ledger = Path(ledger_file)
        if not self.ledger.exists():
            fd = os.open(self.ledger, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        if self.ledger.stat().st_mode & 0o077:
            raise ApprovalError("approval ledger must be private (mode 0600)")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS grants (grant_id TEXT PRIMARY KEY, review_sha256 TEXT NOT NULL, actor TEXT NOT NULL, consumed_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, plan_id TEXT NOT NULL, action TEXT NOT NULL, result TEXT NOT NULL, detail TEXT)")

    def _connect(self):
        return sqlite3.connect(self.ledger, timeout=10)

    def consume(self, token: str, review: dict, *, now: float | None = None) -> dict:
        try:
            if not isinstance(token, str) or len(token) > 8192:
                raise ValueError("invalid token type/length")
            encoded, signature = token.split(".")
            body = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            sig = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
            if not hmac.compare_digest(sig, hmac.new(self._key, body, hashlib.sha256).digest()):
                raise ApprovalError("invalid approval signature")
            grant = json.loads(body)
            instant = time.time() if now is None else now
            if not isinstance(grant, dict) or grant["review_sha256"] != review_hash(review):
                raise ApprovalError("approval is bound to a different reviewed action")
            expires, issued = float(grant["expires_at"]), float(grant["issued_at"])
            if not all(map(math.isfinite, (instant, expires, issued))) or not 0 < expires-issued <= 3600:
                raise ApprovalError("invalid approval lifetime")
            if instant >= expires or issued > instant + 5:
                raise ApprovalError("approval expired or issued in the future")
            if not isinstance(grant["grant_id"], str) or len(grant["grant_id"]) != 48 or not isinstance(grant["actor"], str):
                raise ApprovalError("invalid grant identity")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ApprovalError("malformed approval capability") from exc
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("INSERT INTO grants VALUES (?, ?, ?, ?)", (grant["grant_id"], grant["review_sha256"], grant["actor"], instant))
        except sqlite3.IntegrityError as exc:
            raise ApprovalError("approval already consumed") from exc
        return grant

    def audit(self, plan_id: str, action: str, result: str, detail: str = "") -> None:
        with self._connect() as db:
            db.execute("INSERT INTO audit(timestamp,plan_id,action,result,detail) VALUES (?,?,?,?,?)", (time.time(), plan_id, action, result, detail[:500]))


def verifier_from_env(root: Path) -> ApprovalVerifier | None:
    key = os.getenv("METANAVIT_APPROVAL_KEY_FILE")
    ledger = os.getenv("METANAVIT_APPROVAL_LEDGER_FILE")
    if bool(key) != bool(ledger):
        raise ApprovalError("configure both approval key and ledger paths")
    return ApprovalVerifier(root, Path(key), Path(ledger)) if key else None
