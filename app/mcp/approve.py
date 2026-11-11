"""Trusted operator CLI; never an MCP tool.

Create an Ed25519 key pair. The private key stays with the operator, outside
indexed/agent-readable roots; servers are given only the public key:
  python -m app.mcp.approve init --key /private/operator.key --public-key /srv/metanavit/operator.pub
Inspect the entire proposal's review JSON, then issue a short-lived capability with the private key:
  python -m app.mcp.approve sign --key /private/operator.key --review review.json --actor operator
Configure servers with METANAVIT_APPROVAL_PUBLIC_KEY_FILE and METANAVIT_APPROVAL_LEDGER_FILE.
The host must protect these paths from every connected agent, not only this MCP:
the private key from being read, the public key and the ledger from being
replaced (whoever can replace the public key can mint capabilities).
"""
import argparse
import json
from pathlib import Path
from app.mcp.approvals import create_key, sign_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--key", type=Path, required=True, help="private key to create (mode 0600)")
    init.add_argument("--public-key", type=Path, required=True, help="public key to create, for the servers")
    sign = commands.add_parser("sign")
    sign.add_argument("--key", type=Path, required=True, help="the operator's private key")
    sign.add_argument("--review", type=Path, required=True)
    sign.add_argument("--actor", required=True)
    sign.add_argument("--ttl", type=float, default=120)
    args = parser.parse_args(argv)
    if args.command == "init":
        create_key(args.key, args.public_key)
    else:
        print(sign_review(json.loads(args.review.read_text()), args.key, actor=args.actor, ttl_seconds=args.ttl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
