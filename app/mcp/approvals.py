"""Operator-issued capabilities. Issuance is never registered as an agent tool.

The operator signs with an Ed25519 private key (app/mcp/approve.py sign) and the
servers verify with the public key alone, so a process that verifies
capabilities cannot mint one from what it loads. The host keeps the private
key, the public key and the SQLite ledger outside every agent-readable root.
Anyone able to read the private key, replace the public key or the ledger, or
execute arbitrary code as the operator is outside this boundary. A token
authorizes exactly one immutable review, which records the session its plan
was proposed in.

The grant ledger is a SQLite file (METANAVIT_APPROVAL_LEDGER_FILE), single use
among the processes on one host that share it; with
METANAVIT_APPROVAL_LEDGER=postgres the API keeps it in its database instead
(app/mcp/postgres_ledger.py), shared by every replica on it. Whoever can write
those tables can replay grants, as whoever can replace the file can.

Token: base64url(body) "." base64url(signature), unpadded. The body is the
canonical JSON of {review_sha256, grant_id, issued_at, expires_at, actor}; the
signature is the 64-byte Ed25519 signature over those body bytes. Key files are
PEM: PKCS#8 for the private key (mode 0600), SubjectPublicKeyInfo for the
public key (not writable by group or others).
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


# Every refusal at the tool boundary names the check that refused it, so a
# client, the audit trail and the approval benchmark can say why without
# reading exception text (which can carry server paths). Closed set.
REFUSAL_CHECKS = {
    "capability_required": "no operator capability was presented, or none can be verified here",
    "approval_argument": "a caller's approval flag never authorizes a change",
    "malformed": "the capability is not a well-formed token",
    "signature": "the capability's signature does not verify",
    "binding": "the capability was issued for a different reviewed action",
    "lifetime": "the capability's lifetime is invalid",
    "expiry": "the capability has expired or was issued in the future",
    "grant_identity": "the capability's grant identity is invalid",
    "ledger": "the capability was already consumed; a grant is single use",
    "action_mismatch": "the reviewed plan is for a different action",
    "staleness": "the plan or the reviewed files changed since the review; propose again",
    "symlink": "the path is or passes through a symlink",
    "expression_gate": "imports and attribute access are unavailable in the MCP expression tool",
}


class ApprovalError(PermissionError):
    """A refused or misconfigured capability; `check` (a REFUSAL_CHECKS key) names the refusing check."""

    def __init__(self, message: str, *, check: str | None = None):
        super().__init__(message)
        self.check = check


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def review_hash(review: dict) -> str:
    return hashlib.sha256(canonical(review)).hexdigest()


def create_key(private_path: Path, public_path: Path) -> None:
    """Explicit operator setup: a new Ed25519 key pair. Refuses to overwrite either file.

    The private key (mode 0600) is read only by sign_review; servers are given
    the public key. Neither file is left behind if the other cannot be created.
    """
    private_path, public_path = Path(private_path), Path(public_path)
    if private_path.absolute() == public_path.absolute():
        raise ApprovalError("the private and the public key need two different paths")
    key = Ed25519PrivateKey.generate()
    files = ((private_path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption()), 0o600),
             (public_path, key.public_key().public_bytes(serialization.Encoding.PEM,
                                                         serialization.PublicFormat.SubjectPublicKeyInfo), 0o644))
    created = []
    try:
        for path, data, mode in files:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
            created.append(path)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException:
        for path in created:
            path.unlink()
        raise


def _key_bytes(path: Path, *, private: bool) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ApprovalError("an approval key must be a regular file")
        if private and st.st_mode & 0o077:
            raise ApprovalError("the operator private key must be private (mode 0600)")
        if not private and st.st_mode & 0o022:
            raise ApprovalError("the approval public key must not be writable by group or others")
        return os.read(fd, 4096)
    finally:
        os.close(fd)


def _private_key(path: Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(_key_bytes(path, private=True), password=None)
    except (ValueError, TypeError) as exc:
        raise ApprovalError("the operator key must be a PEM Ed25519 private key (python -m app.mcp.approve init)") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ApprovalError("the operator key must be an Ed25519 private key")
    return key


def _public_key(path: Path) -> Ed25519PublicKey:
    data = _key_bytes(path, private=False)
    if b"PRIVATE KEY" in data:
        raise ApprovalError("a verifier is given the operator's public key, never the private key")
    try:
        key = serialization.load_pem_public_key(data)
    except (ValueError, TypeError) as exc:  # including a pre-Ed25519 32-byte HMAC key
        raise ApprovalError("the approval public key must be a PEM Ed25519 public key (python -m app.mcp.approve init)") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ApprovalError("the approval public key must be an Ed25519 key")
    return key


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def sign_review(review: dict, key_file: Path, *, actor: str, ttl_seconds: float = 120,
                now: float | None = None) -> str:
    """Trusted operator path, separate from FilesystemTools and both MCP servers.

    `key_file` is the operator's Ed25519 private key. The operator must inspect
    the complete review before invoking this function. It signs reviewed data;
    it does not execute actions or resolve file paths.
    """
    if not actor or not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= 3600:
        raise ApprovalError("actor and an expiry between 0 and 3600 seconds are required")
    if review.get("schema") != "metanavit-reviewed-action-v1":
        raise ApprovalError("unsupported review schema")
    issued = time.time() if now is None else now
    body = canonical({"review_sha256": review_hash(review), "grant_id": secrets.token_hex(24),
                      "issued_at": issued, "expires_at": issued + ttl_seconds, "actor": actor})
    return _b64(body) + "." + _b64(_private_key(Path(key_file)).sign(body))


class ApprovalVerifier:
    """Verifies capabilities with the operator's public key and durably consumes grant IDs before mutation.

    It never loads the private key, so it cannot issue a capability.
    SQLite UNIQUE + BEGIN IMMEDIATE makes replay consumption transactional across
    worker threads/processes using this ledger. Failed mutations burn the grant.
    `ledger` (instead of `ledger_file`) is a database ledger with the same two
    calls (consume, audit): app/mcp/postgres_ledger.PostgresLedger.
    """
    def __init__(self, root: Path, public_key_file: Path, ledger_file: Path | None = None, *, ledger=None):
        root = Path(root).resolve()
        if (ledger_file is None) == (ledger is None):
            raise ApprovalError("configure exactly one grant ledger: a SQLite file or a database ledger")
        for path in (Path(public_key_file),) + (() if ledger_file is None else (Path(ledger_file),)):
            resolved = path.resolve()
            if resolved == root or root in resolved.parents:
                raise ApprovalError("approval public key and ledger must be outside the corpus root")
            if path.is_symlink():
                raise ApprovalError("approval state must not be a symlink")
        self.public_key = _public_key(Path(public_key_file))
        self.database_ledger = ledger  # None: the SQLite file below
        self.ledger = None if ledger_file is None else Path(ledger_file)
        if self.ledger is None:
            return
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
            try:
                self.public_key.verify(sig, body)
            except InvalidSignature:
                raise ApprovalError("invalid approval signature", check="signature") from None
            grant = json.loads(body)
            instant = time.time() if now is None else now
            if not isinstance(grant, dict) or grant["review_sha256"] != review_hash(review):
                raise ApprovalError("approval is bound to a different reviewed action", check="binding")
            expires, issued = float(grant["expires_at"]), float(grant["issued_at"])
            if not all(map(math.isfinite, (instant, expires, issued))) or not 0 < expires-issued <= 3600:
                raise ApprovalError("invalid approval lifetime", check="lifetime")
            if instant >= expires or issued > instant + 5:
                raise ApprovalError("approval expired or issued in the future", check="expiry")
            if not isinstance(grant["grant_id"], str) or len(grant["grant_id"]) != 48 or not isinstance(grant["actor"], str):
                raise ApprovalError("invalid grant identity", check="grant_identity")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ApprovalError("malformed approval capability", check="malformed") from exc
        if self.database_ledger is not None:
            self.database_ledger.consume(grant, instant)  # ApprovalError (check "ledger") when already consumed
            return grant
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("INSERT INTO grants VALUES (?, ?, ?, ?)", (grant["grant_id"], grant["review_sha256"], grant["actor"], instant))
        except sqlite3.IntegrityError as exc:
            raise ApprovalError("approval already consumed", check="ledger") from exc
        return grant

    def audit(self, plan_id: str, action: str, result: str, detail: str = "") -> None:
        if self.database_ledger is not None:
            self.database_ledger.audit(plan_id, action, result, detail)
            return
        with self._connect() as db:
            db.execute("INSERT INTO audit(timestamp,plan_id,action,result,detail) VALUES (?,?,?,?,?)", (time.time(), plan_id, action, result, detail[:500]))


LEDGERS = ("sqlite", "postgres")  # METANAVIT_APPROVAL_LEDGER


def verifier_from_env(root: Path, *, database=None) -> ApprovalVerifier | None:
    """The verifier the environment configures, or None (no approval state: every write is refused).

    METANAVIT_APPROVAL_LEDGER picks the grant ledger: sqlite (unset: the default) is the file
    METANAVIT_APPROVAL_LEDGER_FILE names; postgres is `database` (the API's vector store manager), so every
    replica on that database shares one ledger. A process without a database (the stdio MCP servers) refuses
    postgres rather than fall back to a file, and so does a configuration that names a file as well: with two
    ledgers one capability could be consumed once in each.
    """
    if os.getenv("METANAVIT_APPROVAL_KEY_FILE"):
        # Refused even beside the new setting, so a stale shared key is noticed and removed.
        raise ApprovalError("METANAVIT_APPROVAL_KEY_FILE named a shared HMAC key, which is no longer supported: unset "
                            "METANAVIT_APPROVAL_KEY_FILE, create an Ed25519 key pair with `python -m app.mcp.approve "
                            "init` and give servers only the public key through METANAVIT_APPROVAL_PUBLIC_KEY_FILE")
    public = os.getenv("METANAVIT_APPROVAL_PUBLIC_KEY_FILE")
    ledger = os.getenv("METANAVIT_APPROVAL_LEDGER_FILE")
    kind = (os.getenv("METANAVIT_APPROVAL_LEDGER") or "sqlite").strip().lower()
    if kind not in LEDGERS:
        raise ApprovalError("METANAVIT_APPROVAL_LEDGER must be sqlite or postgres")
    if kind == "postgres":
        if ledger:
            raise ApprovalError("METANAVIT_APPROVAL_LEDGER=postgres keeps grants in the database: unset METANAVIT_APPROVAL_LEDGER_FILE")
        if database is None:
            raise ApprovalError("METANAVIT_APPROVAL_LEDGER=postgres needs the API's database; the MCP stdio servers keep a "
                                "SQLite ledger: set METANAVIT_APPROVAL_LEDGER_FILE and unset METANAVIT_APPROVAL_LEDGER")
        if not public:
            raise ApprovalError("METANAVIT_APPROVAL_LEDGER=postgres also needs METANAVIT_APPROVAL_PUBLIC_KEY_FILE")
        from app.mcp.postgres_ledger import PostgresLedger

        return ApprovalVerifier(root, Path(public), ledger=PostgresLedger(database))
    if bool(public) != bool(ledger):
        raise ApprovalError("configure both the approval public key and ledger paths")
    return ApprovalVerifier(root, Path(public), Path(ledger)) if public else None
