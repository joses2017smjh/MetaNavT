"""Trusted operator CLI; never an MCP tool.

Create a private key outside indexed/agent-readable roots:
  python -m app.mcp.approve init --key /private/operator.key
Inspect the entire proposal's review JSON, then issue a short-lived capability:
  python -m app.mcp.approve sign --key /private/operator.key --review review.json --actor operator
Configure servers with METANAVIT_APPROVAL_KEY_FILE and METANAVIT_APPROVAL_LEDGER_FILE.
The host must protect these paths from every connected agent, not only this MCP.
"""
import argparse
import json
from pathlib import Path
from app.mcp.approvals import create_key, sign_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--key", type=Path, required=True)
    sign = commands.add_parser("sign")
    sign.add_argument("--key", type=Path, required=True)
    sign.add_argument("--review", type=Path, required=True)
    sign.add_argument("--actor", required=True)
    sign.add_argument("--ttl", type=float, default=120)
    args = parser.parse_args(argv)
    if args.command == "init":
        create_key(args.key)
    else:
        print(sign_review(json.loads(args.review.read_text()), args.key, actor=args.actor, ttl_seconds=args.ttl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
