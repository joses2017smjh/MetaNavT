"""make parity's report (app/eval/parity.py), with the API and the in-memory row faked: no server, no models.

M17 (B14): the API row counts the router's route per question, and carries the paired gap per gold
category against the in-memory row, so the exact_path questions (the lexical_path route on fixture v1)
are reported for both rows.
"""

from pathlib import Path

from app.eval import parity
from app.eval.gold import load_gold
from app.eval.metrics import aggregate_retrieval

ROOT = Path(__file__).resolve().parents[2]


def fake_run_config(cfg, gold, files_root, n_files=None, **kwargs):
    """The in-memory row with every question's relevant files ranked first."""
    per_query = [{"id": q.id, "retrieved": q.relevant_ids(), "relevant": q.relevant_ids(), "category": q.category} for q in gold]
    retrieval = aggregate_retrieval(per_query, n_files=n_files)
    return {"config": cfg.name, "settings": {"name": cfg.name}, "retrieval": retrieval.as_dict(), "latency": {},
            "_per_query_scores": retrieval.per_query}


def test_the_api_row_counts_routes_and_has_a_paired_gap_per_category(monkeypatch):
    gold = {q.question: q for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")}

    def fake_post(url, query, k=50, timeout=60.0):
        """The API misses every exact_path question and answers every other one like the in-memory row."""
        q = gold[query]
        exact = q.category == "exact_path"
        paths = ["elsewhere/none.txt"] if exact else q.relevant_ids()
        return {"route": "lexical_path" if exact else "semantic", "hits": [{"path": p} for p in paths], "latency_ms": {}}

    monkeypatch.setattr(parity, "wait_for_health", lambda url: {"status": "ok"})
    monkeypatch.setattr(parity, "run_config", fake_run_config)
    monkeypatch.setattr(parity, "post_retrieve", fake_post)
    blob = parity.run("http://127.0.0.1:9", root=ROOT)

    mem, api = blob["results"]
    assert api["routes"] == {"lexical_path": 8, "semantic": 128}
    by_category = {block["category"]: block for block in api["delta_vs_parent_by_category_all"]}
    assert by_category["exact_path"]["reference"] == mem["config"]
    assert by_category["exact_path"]["n"] == 8 and by_category["exact_path"]["small_n"]
    assert by_category["exact_path"]["ndcg@10"] == {"delta": -1.0, "lo": -1.0, "hi": -1.0}
    assert by_category["simple_factual"]["ndcg@10"]["delta"] == 0.0
    assert api["retrieval"]["by_category"]["exact_path"]["ndcg@10"] == 0.0 and mem["retrieval"]["by_category"]["exact_path"]["ndcg@10"] == 1.0
    assert blob["api_ok"]
