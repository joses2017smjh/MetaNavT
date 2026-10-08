"""TREC replay must retain zero-hit queries and reject altered evidence."""
import json
from pathlib import Path

import pytest

pytest.importorskip("snowballstemmer")
pytest.importorskip("pytrec_eval")

from app.eval import beir
from app.eval.beir_rescore import rescore


def artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "nfcorpus"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text('\n'.join(json.dumps({"_id": did, "title": "", "text": text}) for did, text in [("d1", "cells divide"), ("d2", "stars form")]) + '\n')
    (root / "queries.jsonl").write_text('\n'.join(json.dumps({"_id": qid, "text": text}) for qid, text in [("q1", "unmatchedword"), ("q2", "cells")]) + '\n')
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td2\t2\nq2\td1\t1\n")
    # Hash/overlap models keep this evidence-replay test deterministic and tiny.
    configs = [beir.BeirConfig(name="bm25", mode="bm25"), beir.BeirConfig(name="hybrid@bge-small", mode="hybrid", embedder="hash"), beir.BeirConfig(name=beir.RERANK_FP32_512, mode="hybrid", embedder="hash", reranker="overlap", parent="unmeasured")]
    blob = beir.run(tmp_path, configs, dataset="nfcorpus", download=False, n_boot=25, device="cpu", trec_dir=tmp_path / "trec", require_trec_eval=True)
    # Simulate the historical absent parent that this migration must repair.
    blob["results"][-1]["parent"] = "unmeasured"
    blob["results"][-1]["settings"]["parent"] = "unmeasured"
    blob["results"][-1]["delta_vs_parent"] = None
    path = tmp_path / "result.json"
    path.write_text(json.dumps(blob))
    return path


@pytest.mark.parametrize("legacy", [False, True])
def test_rescore_zero_hit_queries_parent_and_independent_metrics(tmp_path, legacy):
    path = artifacts(tmp_path)
    before = json.loads(path.read_text())
    assert "q1 Q0" not in Path(before["results"][0]["trec"]["run"]).read_text()
    if legacy:
        for row in before["results"]:
            row["trec"].pop("query_order")
            row["trec"].pop("query_order_sha256")
        path.write_text(json.dumps(before))
    after = rescore(path)
    child = after["results"][-1]
    assert child["parent"] == "hybrid@bge-small"
    assert child["delta_vs_parent"]["reference"] == "hybrid@bge-small"
    assert child["metric_cross_check"]["n_queries"] == 2
    assert child["metric_cross_check"]["max_absolute_difference"] == 0
    assert child["latency"] == before["results"][-1]["latency"]
    assert after["results"][0]["ci"] == before["results"][0]["ci"]


def test_rescore_refuses_modified_trec_rankings(tmp_path):
    path = artifacts(tmp_path)
    blob = json.loads(path.read_text())
    run = Path(blob["results"][0]["trec"]["run"])
    run.write_text(run.read_text() + "q1 Q0 d1 1 1 changed\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        rescore(path)
