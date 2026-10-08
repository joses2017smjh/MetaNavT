"""Explicit disposable operator fixture for demos, not a production issuer.

Demo callers opt into this fixture; the host creates a private test key outside
its temporary corpus, inspects its own generated manifest, and signs it. Actual
user workflows require external operator issuance with an existing private key.
"""
from contextlib import contextmanager
from pathlib import Path
import tempfile

from app.mcp.approvals import ApprovalVerifier, create_key, sign_review
from app.mcp.filesystem import FilesystemTools


@contextmanager
def operator_fixture(*, index=None):
    with tempfile.TemporaryDirectory(prefix="metanavit-operator-demo-") as temporary:
        base = Path(temporary)
        root, private = base / "corpus", base / "operator-private"
        root.mkdir(); private.mkdir(mode=0o700)
        key = private / "key"
        create_key(key)
        tools = FilesystemTools(root=root, index=index, approval_verifier=ApprovalVerifier(root, key, private / "ledger.sqlite"))

        def issue(plan_id, **options):
            review = tools.review_plan(plan_id, **options)
            return sign_review(review, key, actor="disposable-demo-operator")
        yield tools, issue
