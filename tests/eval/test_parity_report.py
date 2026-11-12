"""make parity's report (app/eval/parity.py), with the API and the in-memory row faked: no server, no models.

M17 (B14): the API row counts the router's route per question, and carries the paired gap per gold
category against the in-memory row, so the exact_path questions (the lexical_path route on fixture v1)
are reported for both rows.
"""

from pathlib import Path

import pytest

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


# ---------------------------------------------------------------- A19: a named in-memory configuration, both sides' settings

NEURAL = "in_memory:hybrid+bge-rerank+router+staleness@bge-small/fp16-512/top20/staleness-first"


def test_the_configuration_is_chosen_by_name(monkeypatch):
    chosen = []

    def recording_run_config(cfg, *args, **kwargs):
        chosen.append(cfg)
        return fake_run_config(cfg, *args, **kwargs)

    gold = {q.question: q for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")}
    monkeypatch.setattr(parity, "wait_for_health", lambda url: {"status": "ok"})
    monkeypatch.setattr(parity, "run_config", recording_run_config)
    monkeypatch.setattr(parity, "post_retrieve", lambda url, query, k=50, timeout=60.0: {"route": "semantic", "hits": [{"path": p} for p in gold[query].relevant_ids()]})
    assert parity.run("http://127.0.0.1:9", root=ROOT)["in_memory_config"] == parity.IN_MEMORY.name  # CI's, by default
    blob = parity.run("http://127.0.0.1:9", root=ROOT, in_memory=NEURAL)
    assert blob["in_memory_config"] == NEURAL and chosen[-1] is parity.IN_MEMORY_NEURAL
    assert (chosen[-1].embedder, chosen[-1].reranker, chosen[-1].rerank_depth, chosen[-1].rerank_precision, chosen[-1].rerank_max_length,
            chosen[-1].enable_router, chosen[-1].staleness_tier1, chosen[-1].staleness_before_rerank) == (
        "st:BAAI/bge-small-en-v1.5", "BAAI/bge-reranker-v2-m3", 20, "fp16", 512, True, True, True)
    with pytest.raises(ValueError, match="unknown in-memory configuration"):
        parity.run("http://127.0.0.1:9", root=ROOT, in_memory="in_memory:nope")


def _served_health(instruction=""):
    """/health of an API started with bge-small, the served reranker settings and EMBEDDING_QUERY_INSTRUCTION."""
    return {"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_query_instruction": instruction, "dense_search": {"depth": 50},
            "reranker": {"configured": True, "loaded": True, "model": "BAAI/bge-reranker-v2-m3", "precision": "fp16", "max_length": 512, "depth": 20}}


def test_both_sides_settings_side_by_side():
    mem = {"reranker_loaded": True, "models": {"embedder": {"model": "BAAI/bge-small-en-v1.5", "neural": True, "dim": 384}}}
    api = {"routes": {"semantic": 95, "aggregation_query": 33, "lexical_path": 8}, "settings": {"staleness": {"enabled": True}}}
    settings = parity.side_by_side(parity.IN_MEMORY_NEURAL, mem, api, _served_health(""))
    assert [k for k, v in settings.items() if v["in_memory"] != v["api"]] == []
    assert settings["rerank_depth"] == {"in_memory": 20, "api": 20} and settings["staleness"]["api"] == "before the rerank"

    prefixed = parity.side_by_side(parity.IN_MEMORY_NEURAL, mem, api, _served_health("Represent this question for searching relevant passages: "))
    assert [k for k, v in prefixed.items() if v["in_memory"] != v["api"]] == ["query_instruction"]
    every = parity.side_by_side(parity.RERANK_DEPTH_CONFIGS[1], mem, api, _served_health(""))  # neural.json's row: every candidate, fp32, after
    assert {k for k, v in every.items() if v["in_memory"] != v["api"]} == {"router", "rerank_depth", "rerank_precision", "rerank_max_length", "staleness"}


def test_the_hash_run_has_nothing_to_report_apart():
    mem = {"reranker_loaded": False, "models": {"embedder": {"model": "hash", "neural": False, "dim": 256}}}
    api = {"routes": {"semantic": 95, "aggregation_query": 33, "lexical_path": 8}, "settings": {"staleness": {"enabled": True}}}
    health = {"embedding_model": "hash-256", "embedding_query_instruction": "", "dense_search": {"depth": 50},
              "reranker": {"configured": False, "loaded": False}}
    settings = parity.side_by_side(parity.IN_MEMORY, mem, api, health)
    assert [k for k, v in settings.items() if v["in_memory"] != v["api"]] == []
    assert settings["staleness"] == {"in_memory": "on, no reranker", "api": "on, no reranker"}


def test_the_readme_block_renders_both_runs_with_the_settings(tmp_path):
    """report.parity_table: the hash run and the model-backed run, each with its gap and exact_path slice, and the
    model-backed run's settings side by side (derived from the file, not typed)."""
    import json

    from app.eval.report import PARITY_FILE, PARITY_NEURAL_FILE, parity_table

    hash_run = json.loads((ROOT / PARITY_FILE).read_text())
    neural = json.loads(json.dumps(hash_run))
    neural["in_memory_config"] = NEURAL
    neural["results"][0]["config"] = NEURAL
    neural["api_health"].update(embedding_provider="huggingface", embedding_model="BAAI/bge-small-en-v1.5", embedding_query_instruction="")
    neural["api_health"]["reranker"].update(loaded=True, model="BAAI/bge-reranker-v2-m3", precision="fp16")
    neural["settings"] = {"embedding_model": {"in_memory": "BAAI/bge-small-en-v1.5", "api": "BAAI/bge-small-en-v1.5"},
                          "query_instruction": {"in_memory": "", "api": ""}, "rerank_depth": {"in_memory": 20, "api": 20}}
    neural["settings_differ"] = []
    block = parity_table(hash_run, neural)
    assert block.count("| hash embedder, no reranker |") == 2 and block.count("| BAAI/bge-small-en-v1.5, BAAI/bge-reranker-v2-m3 |") == 2
    assert PARITY_FILE in block and PARITY_NEURAL_FILE in block and f"PARITY_CONFIG={NEURAL}" in block
    assert "query instruction none / none; rerank depth 20 / 20" in block and "Settings that differ: none." in block
    assert block.count("exact_path questions") == 2 and "is missing" not in block
    assert block.count("EMBEDDING_QUERY_INSTRUCTION=none (the bare query") == 1  # the model-backed API's non-default setting, said
    assert "is missing" in parity_table(hash_run)  # the model-backed file not recorded yet: said, not hidden


def test_a_hash_run_that_names_its_embedding_model_is_still_labelled_the_hash_embedder():
    """Since M17 /health names the embedding model, "hash-256" for the hash adapter: a hash run recorded with it keeps
    the fallback label the block has always given that run, and a real model is named by its model id."""
    from app.eval.report import _parity_stack

    health = {"embedding_provider": "hash", "embed_model": "HashEmbedding", "embedding_model": "hash-256", "reranker": {"loaded": False}}
    assert _parity_stack({"api_health": health}) == "hash embedder, no reranker"
    served = {**health, "embedding_provider": "huggingface", "embed_model": "HuggingFaceEmbedding", "embedding_model": "BAAI/bge-small-en-v1.5"}
    assert _parity_stack({"api_health": served}) == "BAAI/bge-small-en-v1.5, no reranker"


def test_the_readme_block_names_every_category_whose_interval_excludes_zero():
    """An overall interval that covers zero can hide a category whose interval does not (a loser kept in view): each
    run's paragraph names, from the file, every gold category whose paired interval excludes zero, and only those."""
    import json

    from app.eval.report import PARITY_FILE, _parity_run_text

    blob = json.loads((ROOT / PARITY_FILE).read_text())
    blocks = blob["results"][1]["delta_vs_parent_by_category_all"]
    for block in blocks:
        block["ndcg@10"] = {"delta": 0.01, "lo": -0.01, "hi": 0.03}
    assert "Every gold category's paired interval covers zero." in _parity_run_text(blob, PARITY_FILE)
    simple = next(b for b in blocks if b["category"] == "simple_factual")
    simple["ndcg@10"] = {"delta": -0.0448, "lo": -0.0726, "hi": -0.0188}
    text = _parity_run_text(blob, PARITY_FILE)
    assert f"Gold categories whose paired interval excludes zero (API minus in-memory, nDCG@10): simple_factual -0.0448 [-0.0726, -0.0188] (n = {simple['n']})." in text
    assert "covers zero." not in text.split("Gold categories whose")[1].split("(n = ")[0]
