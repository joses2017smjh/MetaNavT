"""A38: fixture v1's gold questions are pinned by digest, and the results headers name the gold file they scored.

bench/corpus/MANIFEST.json lists corpus files only, so before this pin an edited bench/gold/questions.jsonl ran and
scored without any check noticing. The digest is of the file's bytes as committed; .gitattributes keeps git from
rewriting its line endings on checkout (core.autocrlf), as it does for bench/corpus.
"""

import hashlib
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval import parity
from app.eval.harness import DEFAULT_CONFIGS, run_bench
from app.eval.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "bench" / "gold" / "questions.jsonl"
# sha256 of bench/gold/questions.jsonl as committed (fixture v1, frozen; unchanged since 106552e). Change it only
# together with a new fixture version, never to make this test pass.
GOLD_SHA256 = "06c2421220fa2569c87cfa5fc57902476da06ac28f9ea41a5320623d64e984de"


def test_the_gold_question_file_is_fixture_v1s():
    assert hashlib.sha256(GOLD.read_bytes()).hexdigest() == GOLD_SHA256
    assert sha256_file(GOLD) == GOLD_SHA256


def test_git_checks_the_gold_file_out_byte_for_byte():
    """No end-of-line conversion under bench/gold, whatever core.autocrlf says: the pin is of the committed bytes."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "check-attr", "text", "--", "bench/gold/questions.jsonl"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    assert out.splitlines() == ["bench/gold/questions.jsonl: text: unset"]


def test_a_bench_results_header_names_the_gold_file_it_scored(tmp_path):
    blob = run_bench(configs=[replace(DEFAULT_CONFIGS[1])], out=tmp_path / "bm25.json")  # bm25_only: the fastest row
    assert blob["gold_sha256"] == GOLD_SHA256


def test_a_parity_results_header_names_the_gold_file_it_scored(monkeypatch):
    def fake_run_config(cfg, gold, files_root, n_files=None, **kwargs):
        from app.eval.metrics import aggregate_retrieval

        per_query = [{"id": q.id, "retrieved": q.relevant_ids(), "relevant": q.relevant_ids(), "category": q.category} for q in gold]
        retrieval = aggregate_retrieval(per_query, n_files=n_files)
        return {"config": cfg.name, "settings": {"name": cfg.name}, "retrieval": retrieval.as_dict(), "latency": {},
                "_per_query_scores": retrieval.per_query}

    monkeypatch.setattr(parity, "wait_for_health", lambda url: {"status": "ok"})
    monkeypatch.setattr(parity, "run_config", fake_run_config)
    monkeypatch.setattr(parity, "post_retrieve", lambda url, query, k=50, timeout=60.0: {"route": "semantic", "hits": [{"path": "README.md"}]})
    assert parity.run("http://127.0.0.1:9", root=ROOT)["gold_sha256"] == GOLD_SHA256
