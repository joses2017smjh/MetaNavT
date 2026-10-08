"""Recompute paired statistics from preserved BEIR TREC rankings, without inference.

python -m app.eval.beir_rescore --input RESULT.json --out RESCORED.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from app.eval import beir
from app.eval.metrics import mrr_at_k, recall_at_k


def rescore(path: Path) -> dict:
    blob = json.loads(path.read_text())
    names = {row["config"] for row in blob["results"]}
    query_order = None
    for row in blob["results"]:
        provenance = row["trec"]
        run_path = Path(provenance["run"])
        if beir.sha256_of(run_path) != provenance["run_sha256"]:
            raise RuntimeError("preserved TREC run/qrels checksum mismatch")
        if provenance.get("query_order"):
            order_path = Path(provenance["query_order"])
            if beir.sha256_of(order_path) != provenance["query_order_sha256"]:
                raise RuntimeError("preserved TREC query-order checksum mismatch")
            query_order = json.loads(order_path.read_text())
            break
        # Compatibility for runs recorded before the query-order sidecar:
        # another measured configuration that retrieves for every query
        # preserves the complete original iteration order.
        candidate = list(dict.fromkeys(line.split()[0] for line in run_path.read_text().splitlines()))
        if len(candidate) == blob["n_queries"]:
            query_order = candidate
            break
    if query_order is None or len(query_order) != blob["n_queries"] or len(set(query_order)) != len(query_order):
        raise RuntimeError("TREC artifacts do not preserve a complete unique query order")
    for row in blob["results"]:
        provenance = row["trec"]
        run_path, qrels_path = Path(provenance["run"]), Path(provenance["qrels"])
        if beir.sha256_of(run_path) != provenance["run_sha256"] or beir.sha256_of(qrels_path) != provenance["qrels_sha256"]:
            raise RuntimeError("preserved TREC run/qrels checksum mismatch")
        qrels: dict[str, dict[str, int]] = {}
        for line in qrels_path.read_text().splitlines():
            qid, _, did, grade = line.split()
            qrels.setdefault(qid, {})[did] = int(grade)
        ranked: dict[str, list[tuple[int, str]]] = {}
        for line in run_path.read_text().splitlines():
            qid, _, did, rank, _, _ = line.split()
            ranked.setdefault(qid, []).append((int(rank), did))
        # Existing runs process queries in the order used for the paired
        # bootstrap. Keep that insertion order instead of sorting ids.
        if set(ranked) - set(query_order) or set(query_order) - set(qrels):
            raise RuntimeError("TREC query ids disagree with preserved query order/qrels")
        rankings = {qid: [did for _, did in sorted(ranked.get(qid, []))] for qid in query_order}
        per_query = [{"id": qid, "ndcg@10": beir.graded_ndcg_at_k(ids, qrels[qid], 10), "recall@10": recall_at_k(ids, qrels[qid], 10), "recall@100": recall_at_k(ids, qrels[qid], 100), "mrr@10": mrr_at_k(ids, qrels[qid], 10)} for qid, ids in rankings.items()]
        if len(per_query) != blob["n_queries"]:
            raise RuntimeError("TREC run does not cover the reported query count")
        for metric in ("ndcg@10", "recall@10", "recall@100", "mrr@10"):
            if round(sum(p[metric] for p in per_query) / len(per_query), 4) != row["retrieval"][metric]:
                raise RuntimeError(f"stored retrieval average disagrees for {row['config']} {metric}")
        row["_per_query"] = per_query
        row["metric_cross_check"] = beir.cross_check_metrics(rankings, qrels, per_query, required=True)
        if row["config"] == beir.RERANK_FP32_512 and row["parent"] not in names and "hybrid@bge-small" in names:
            row["comparison_note"] = "Uncapped reranker was not measured; bounded capped reranker is compared as a component addition against measured hybrid retrieval. Precision row retains identical cap/depth parent."
            row["parent"] = "hybrid@bge-small"
            row["settings"]["parent"] = "hybrid@bge-small"
    settings = blob["bootstrap"]
    beir.attach_confidence(blob["results"], blob["n_queries"], seed=settings["seed"], n_boot=settings["n_boot"], published=blob.get("published"))
    blob["rescored"] = {"input": str(path), "input_sha256": beir.sha256_of(path), "runner_sha256": beir.sha256_of(Path(__file__)), "timestamp": datetime.now(timezone.utc).isoformat(), "method": "Preserved checksummed TREC rankings; independent metric verification and paired bootstrap recomputed; no retrieval/inference rerun"}
    return blob


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    blob = rescore(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2) + "\n")
    print(beir.markdown_table(blob))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
