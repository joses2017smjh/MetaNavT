"""Source fingerprints for reproducible M4 deterministic regressions."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess


def provenance():
    root = Path(__file__).resolve().parents[2]
    files = (
        "app/mcp/approvals.py", "app/mcp/approve.py", "app/mcp/secure_io.py",
        "app/mcp/filesystem.py", "app/mcp/server.py", "app/agent/citation_verify.py",
        "app/artifacts/templates.py",
        "app/agent/evidence_checks.py", "app/eval/index_loader.py",
        "app/eval/trusted_actions.py", "app/eval/evidence_consistency.py",
        "app/eval/m4_provenance.py",
    )
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {"run_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": head,
            "working_tree_dirty": dirty,
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
