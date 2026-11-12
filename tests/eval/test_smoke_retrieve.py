"""scripts/smoke_retrieve.py, the check CI's parity and docker-smoke jobs run against a started API (no API here).

The smoke compares the table's rows (/health n_nodes) with what the ingest manifest accounts for. It read that count
from the start's own indexing report, which only an API with INDEX_ON_START has; since Compose and CI run the index
step before the API, /health carries the manifest the start read (indexing.manifest_at_start) instead, and the
comparison was skipped in both jobs.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("smoke_retrieve", ROOT / "scripts" / "smoke_retrieve.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)

MANIFEST = {"files": 61, "chunks": 71, "pipelines": ["app.chunking/auto/v2|hash-256"]}


@pytest.mark.parametrize("indexing,n_nodes,passes", [
    ({"indexed": True, "n_nodes": 71}, 71, True),  # a start that indexed (INDEX_ON_START): its own report
    ({"indexed": True, "n_nodes": 71}, 72, False),
    ({"indexed": False, "reason": "INDEX_ON_START not set", "manifest_at_start": MANIFEST}, 71, True),  # the index step ran first
    ({"indexed": False, "reason": "INDEX_ON_START not set", "manifest_at_start": MANIFEST}, 75, False),
    ({"indexed": False, "reason": "INDEX_ON_START not set"}, 71, True),  # nothing to compare with
])
def test_the_smoke_compares_the_rows_with_what_the_manifest_accounts_for(monkeypatch, capsys, indexing, n_nodes, passes):
    health = {"status": "ok", "n_nodes": n_nodes, "indexing": indexing}
    monkeypatch.setattr(smoke, "wait_for_health", lambda base, timeout: health)
    monkeypatch.setattr(smoke, "_post", lambda url, payload: (200, {"hits": [{"score": 0.5, "path": "configs/run_047.yaml"}], "counts": {"bm25": 1}}))
    monkeypatch.setattr(sys, "argv", ["smoke_retrieve.py", "--require-bm25"])

    assert smoke.main() == (0 if passes else 1)
    out = capsys.readouterr().out
    if passes:
        assert "SMOKE OK" in out
    else:
        assert f"table has {n_nodes} rows but the ingest manifest accounts for 71 (stray rows?)" in out
