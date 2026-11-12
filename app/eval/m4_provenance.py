"""Source fingerprints for reproducible M4 deterministic regressions."""
from datetime import datetime, timezone

from app.eval.provenance import run_provenance

# The files whose sha256 the M4 regressions record (trusted_actions, evidence_consistency, agent_injection).
SOURCES = (
    "app/mcp/approvals.py", "app/mcp/approve.py", "app/mcp/secure_io.py",
    "app/mcp/filesystem.py", "app/mcp/server.py", "app/agent/citation_verify.py",
    "app/artifacts/templates.py",
    "app/agent/evidence_checks.py", "app/eval/index_loader.py",
    "app/eval/trusted_actions.py", "app/eval/evidence_consistency.py",
    "app/eval/m4_provenance.py",
)


def provenance():
    """When, the full commit, whether the tree had uncommitted changes, and the sha256 of SOURCES (app.eval.provenance)."""
    return {"run_at_utc": datetime.now(timezone.utc).isoformat(), **run_provenance(sources=SOURCES)}
