"""Named API keys: add one to the key file the API reads (METANAVIT_API_KEYS_FILE, app/api/security.py).

    python -m app.api.keys add --file /srv/metanavit/api_keys alice

prints a new random key once, on stdout, and appends "alice:<SHA-256 of the key, hex>" to the file (created with
mode 0600 when missing). The file never holds a key; give the key to its holder and keep no other copy. The name is
the holder's identity on the API: the actor of their plan decisions and their request budget. Revoke a key by
deleting its line; a running API reads the file again when it changes. The file must not be under DATA_DIR or the
output directory, which /api/files serves.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys
from pathlib import Path

from app.api.security import KEY_NAME, SHARED_KEY_IDENTITY, KeyFileError, parse_key_file


def add_key(path: Path, name: str) -> str:
    """Append a new key for `name` to the file and return the key. Refuses an invalid, reserved or existing name and a
    file that does not parse (nothing is written then)."""
    if not KEY_NAME.fullmatch(name):
        raise KeyFileError("a name is 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
    if name == SHARED_KEY_IDENTITY:
        raise KeyFileError(f"{SHARED_KEY_IDENTITY} is reserved for METANAVIT_API_KEY")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if name in parse_key_file(text):
        raise KeyFileError(f"{name} already has a key; delete its line first to replace it")
    key = secrets.token_urlsafe(32)
    line = f"{name}:{hashlib.sha256(key.encode('utf-8')).hexdigest()}\n"
    if text and not text.endswith("\n"):  # a last line written without a newline: the new line must not join it
        line = "\n" + line
    parse_key_file(text + line)  # the file the API will read: never write one it would refuse
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return key


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="create a key for NAME: its hash goes to the file, the key to stdout")
    add.add_argument("--file", type=Path, required=True, help="the key file (METANAVIT_API_KEYS_FILE)")
    add.add_argument("name")
    args = parser.parse_args(argv)
    try:
        key = add_key(args.file, args.name)
    except (KeyFileError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    sys.stdout.write(key + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
