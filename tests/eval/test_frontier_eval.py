"""M7: the prototype arms are evaluated without gold fields reaching retrieval."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from app.eval.harness import BenchConfig, _retrieve
from app.eval.latency import StageTimer
from app.graph.hipporag import triples_from_chunks
from app.retrieval.hybrid import InMemoryHybridIndex
from app.retrieval.router import RouteType
from app.retrieval.types import Chunk


def _tiny_index() -> InMemoryHybridIndex:
    chunks = [
        Chunk("a", "configs/run_040.yaml", "run_id: 40 encoder: resnet50 learning_rate: 1e-4", 0, 60),
        Chunk("b", "configs/run_047.yaml", "run_id: 47 encoder: dinov2 learning_rate: 3e-4", 0, 60),
        Chunk("c", "slurm/run_047.sbatch", "run_id: 47 sbatch job for dinov2", 0, 40),
    ]
    return InMemoryHybridIndex(chunks, retrieve_k=5, enable_rerank=False, enable_router=False)


HIPPO = BenchConfig(name="hybrid+hipporag", mode="hybrid", enable_rerank=False, enable_router=False,
                    staleness_tier1=False, hipporag=True, e2e=False, log_triples=False)


def test_retrieve_has_no_gold_category_parameter():
    # The harness used to forward the gold category of a question into the PPR gate.
    assert "category" not in inspect.signature(_retrieve).parameters
    index = _tiny_index()
    with pytest.raises(TypeError):
        _retrieve(index, "which slurm job launched run 47", HIPPO, StageTimer(), {}, None, category="multi_hop")


def test_hipporag_gate_in_the_harness_is_the_router_not_the_gold_label(monkeypatch):
    import app.eval.harness as harness

    index = _tiny_index()
    triples = triples_from_chunks(index.chunks)
    seen: list[dict] = []
    real = harness.apply_hipporag

    def spy(query, hits, triples, **kwargs):
        seen.append(kwargs)
        return real(query, hits, triples, **kwargs)

    monkeypatch.setattr(harness, "apply_hipporag", spy)
    # Gold calls this kind of question multi_hop; the router does not call it aggregation.
    query = "which slurm job launched run 47"
    assert index.router.route(query).route != RouteType.AGGREGATION
    base, _ = _retrieve(index, query, replace(HIPPO, hipporag=False), StageTimer(), {}, None)
    boosted, _ = _retrieve(index, query, HIPPO, StageTimer(), {}, None, hippo_triples=triples)
    assert boosted == base, "PPR must not fire on a question the router does not route to aggregation"
    assert seen and all(not kw.get("category") for kw in seen)
    assert all(kw.get("router") is index.router for kw in seen)


# ---- the frontier run: parents in the same file, one switch per arm ----------------------------

def test_frontier_configs_put_parents_first_and_name_a_parent_for_every_arm():
    from app.eval.harness import DEFAULT_CONFIGS, FRONTIER_NOT_EVALUATED, frontier_configs

    cpu = frontier_configs()
    names = [c.name for c in cpu]
    assert names[:3] == ["hybrid", "hybrid+rerank", "dense_only"] and len(names) == len(set(names))
    defaults = {c.name: c for c in DEFAULT_CONFIGS}
    for parent in cpu[:3]:  # the parents are the default bench's rows, switch for switch
        d = defaults[parent.name]
        assert (parent.mode, parent.embedder, parent.reranker, parent.enable_rerank, parent.enable_router, parent.staleness_tier1,
                parent.retrieve_k, parent.rerank_n, parent.chunk_strategy) == (
                d.mode, d.embedder, d.reranker, d.enable_rerank, d.enable_router, d.staleness_tier1, d.retrieve_k, d.rerank_n, d.chunk_strategy)
    for arm in cpu[3:]:
        assert arm.parent in names[:3], arm.name
        assert arm.enable_router is False and arm.staleness_tier1 is False, arm.name  # same as the parents
        assert arm.enable_rerank == (arm.parent == "hybrid+rerank") or arm.reranker == "rankgpt", arm.name
        assert arm.mode == ("dense" if arm.parent == "dense_only" else "hybrid"), arm.name
    assert not set(FRONTIER_NOT_EVALUATED) & set(names)  # no token embeddings on the hash embedder
    for wanted in ("hybrid+multiquery", "hybrid+hipporag", "hybrid+hops=1", "hybrid+fixed-chunks", "hybrid+contextual",
                   "hybrid+rerank+hyde", "hybrid+rerank+decompose", "hybrid+rerank+corrective", "hybrid+rerank+hyde+corrective",
                   "hybrid+rerank+adaptive", "hybrid+rerank+graphrag", "hybrid+hyde", "dense_only+hyde", "hybrid+hyde+rerank"):
        assert wanted in names
    stages = {c.name: c.hyde_stage for c in cpu if c.hyde}
    assert stages == {"hybrid+hyde": "first_stage", "dense_only+hyde": "first_stage", "hybrid+hyde+rerank": "first_stage",
                      "hybrid+rerank+hyde": "after_rerank", "hybrid+rerank+hyde+corrective": "after_rerank"}


def test_neural_frontier_has_its_own_parents_and_real_models():
    from app.eval.harness import BGE_RERANKER, BGE_SMALL, frontier_configs

    neural = frontier_configs(neural=True)
    names = [c.name for c in neural]
    assert names[:3] == ["hybrid@bge-small", "hybrid+bge-rerank@bge-small", "dense_only@bge-small"]
    assert "hybrid+late-chunking@bge-small" in names and "hybrid+bge-rerank+graphrag@bge-small" in names
    assert {"hybrid+hyde@bge-small", "dense_only+hyde@bge-small", "hybrid+hyde+bge-rerank@bge-small", "hybrid+bge-rerank+hyde@bge-small"} <= set(names)
    assert all(c.embedder == BGE_SMALL for c in neural)
    assert all(c.parent in names[:3] for c in neural[3:])
    assert all(c.reranker == BGE_RERANKER for c in neural if c.enable_rerank and "rankgpt" not in c.name)


def test_graphrag_arm_with_one_community_only_splits_members_from_non_members():
    from app.graph.graphrag import answer_global, boost_by_communities, build_graphrag, community_stats, match_communities
    from app.retrieval.types import RetrievalHit

    chunks = [
        Chunk("a", "configs/run_040.yaml", "run_id: 40 encoder: dinov2", 0, 30),
        Chunk("b", "configs/run_047.yaml", "run_id: 47 encoder: dinov2", 0, 30),
        Chunk("n", "paper/notes.md", "an overview with no run id", 0, 30),
    ]
    communities = build_graphrag(chunks)
    stats = community_stats(communities, n_corpus_files=3)
    assert stats["n_communities"] == 1 and stats["discriminates"] is False and stats["n_member_files"] == 2
    assert match_communities("zzz qqq", communities) == []  # retrieval gets no default community ...
    assert answer_global("zzz qqq", communities).startswith("[community 0]")  # ... the global answer keeps its fallback
    matched = match_communities("which run used dinov2", communities)
    assert [c.community_id for c in matched] == [0]
    hits = [RetrievalHit(chunks[2], 3.0, 1), RetrievalHit(chunks[1], 2.0, 2), RetrievalHit(chunks[0], 1.0, 3)]
    out = boost_by_communities(hits, matched)
    # members move ahead of the non-member, their own order is untouched, nothing is added
    assert [h.chunk.path for h in out] == ["configs/run_047.yaml", "configs/run_040.yaml", "paper/notes.md"]
    assert [h.rank for h in out] == [1, 2, 3]


def test_adaptive_arm_skips_only_when_the_classifier_says_skip():
    index = _tiny_index()
    base = BenchConfig(mode="hybrid", enable_rerank=False, enable_router=False, staleness_tier1=False, e2e=False, log_triples=False)
    fired: dict[str, int] = {}
    paths, payload = _retrieve(index, "hello there", replace(base, adaptive=True), StageTimer(), {}, None, fired=fired)
    assert paths == [] and payload == [] and fired == {"adaptive_skip": 1}
    query = "what learning rate did run 47 use"
    expected, _ = _retrieve(index, query, base, StageTimer(), {}, None)
    got, _ = _retrieve(index, query, replace(base, adaptive=True), StageTimer(), {}, None, fired=fired)
    assert got == expected and fired["adaptive_retrieve"] == 1


# ---- HyDE: where its dense list enters the pipeline ------------------------------------------

class _AxisEmbedder:
    """One axis per keyword (alpha, beta, gamma, delta), so every cosine is 0 or 1 and the dense order is easy to read."""

    name, dim = "axis", 4
    WORDS = ("alpha", "beta", "gamma", "delta")

    def encode(self, texts):
        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for j, word in enumerate(self.WORDS):
                if word in (text or "").lower():
                    out[i, j] = 1.0
        return out


def _hyde_index(rerank_fn):
    chunks = [
        Chunk("a", "a.md", "alpha one", 0, 9),
        Chunk("b", "b.md", "alpha two", 0, 9),
        Chunk("c", "c.md", "gamma three", 0, 11),
        Chunk("d", "d.md", "delta four", 0, 10),
    ]
    return InMemoryHybridIndex(chunks, embedder=_AxisEmbedder(), retrieve_k=2, rerank_fn=rerank_fn,
                               enable_rerank=rerank_fn is not None, enable_router=False)


class _SpyReranker:
    """Records the candidates it is given and returns them in reverse order (an order RRF would never produce)."""

    def __init__(self):
        self.seen: list[list[str]] = []
        self.returned: list[list[str]] = []

    def __call__(self, query, pairs):
        self.seen.append([c.chunk_id for c, _ in pairs])
        out = [(c, float(i)) for i, (c, _) in enumerate(reversed(pairs), start=1)]
        self.returned.append([c.chunk_id for c, _ in out])
        return out


def _gamma_hyde(monkeypatch):
    """The HyDE vector points at chunk c, which neither BM25 nor the raw query's dense search ranks in the top 2."""
    import numpy as np

    import app.eval.harness as harness

    monkeypatch.setattr(harness, "hyde_embed", lambda *args, **kwargs: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))


_HYDE = BenchConfig(name="hyde-test", mode="hybrid", retrieve_k=2, rerank_n=2, enable_router=False, staleness_tier1=False,
                    e2e=False, log_triples=False)


def test_first_stage_hyde_lets_the_reranker_order_candidates_that_include_hyde_hits(monkeypatch):
    _gamma_hyde(monkeypatch)
    spy = _SpyReranker()
    index = _hyde_index(spy)
    fired: dict[str, int] = {}
    paths, result = _retrieve(index, "alpha", replace(_HYDE, enable_rerank=True, hyde=True, hyde_stage="first_stage"),
                              StageTimer(), {}, None, fired=fired)
    assert spy.seen == [["a", "c"]]  # BM25 top 2 (a, b) fused with the HyDE dense top 2 (c, a): c is a candidate
    assert [h.chunk.chunk_id for h in result.hits] == spy.returned[0] == ["c", "a"]  # the reranker has the last word
    assert paths == ["c.md", "a.md"] and fired == {"hyde": 1}
    plain, _ = _retrieve(index, "alpha", replace(_HYDE, enable_rerank=True), StageTimer(), {}, None)
    assert "c.md" not in plain  # without HyDE the raw query never reaches c


def test_after_rerank_hyde_is_kept_and_still_overrides_the_reranker(monkeypatch):
    # The wiring recorded before M18 as hybrid+rerank+hyde: rerank first, then RRF with the HyDE dense list and re-sort.
    _gamma_hyde(monkeypatch)
    spy = _SpyReranker()
    index = _hyde_index(spy)
    paths, result = _retrieve(index, "alpha", replace(_HYDE, enable_rerank=True, hyde=True, hyde_stage="after_rerank"),
                              StageTimer(), {}, None)
    assert spy.seen == [["a", "b"]] and spy.returned == [["b", "a"]]  # the raw query's candidates only
    assert [h.chunk.chunk_id for h in result.hits] == ["a", "b"] != spy.returned[0]  # re-sorted after the reranker
    assert "c.md" not in paths  # a HyDE hit outside the reranked list is never added


def test_first_stage_hyde_without_a_reranker_and_in_dense_mode_uses_the_hyde_vector(monkeypatch):
    _gamma_hyde(monkeypatch)
    index = _hyde_index(None)
    fired: dict[str, int] = {}
    hybrid, _ = _retrieve(index, "alpha", replace(_HYDE, enable_rerank=False, hyde=True, hyde_stage="first_stage"),
                          StageTimer(), {}, None, fired=fired)
    assert hybrid == ["a.md", "c.md"]  # a: BM25 rank 1 + HyDE rank 2; c: HyDE rank 1
    dense, _ = _retrieve(index, "alpha", replace(_HYDE, mode="dense", enable_rerank=False, hyde=True, hyde_stage="first_stage"),
                         StageTimer(), {}, None, fired=fired)
    assert dense == ["c.md", "a.md"] and fired == {"hyde": 2}


def test_hyde_needs_a_stage_and_dense_mode_has_only_the_first():
    index = _hyde_index(None)
    with pytest.raises(ValueError, match="hyde_stage"):
        _retrieve(index, "alpha", replace(_HYDE, hyde=True), StageTimer(), {}, None)
    with pytest.raises(ValueError, match="hyde_stage"):
        _retrieve(index, "alpha", replace(_HYDE, mode="dense", hyde=True, hyde_stage="after_rerank"), StageTimer(), {}, None)


def test_the_frontier_block_and_the_evidence_name_each_hyde_wiring(tmp_path):
    import json

    from app.eval.evidence import _frontier_claims
    from app.eval.harness import frontier_configs, run_bench
    from app.eval.report import frontier_table

    wanted = {"hybrid", "hybrid+rerank", "hybrid+hyde+rerank", "hybrid+rerank+hyde"}
    out = tmp_path / "bench" / "results" / "frontier.json"
    run_bench(configs=[c for c in frontier_configs() if c.name in wanted], out=out, frontier=True)
    blob = json.loads(out.read_text())
    notes = {line.split("`")[1]: line for line in frontier_table(blob).splitlines() if line.startswith("- `")}
    assert "HyDE after the reranker" in notes["hybrid+rerank+hyde"] and "order does not survive" in notes["hybrid+rerank+hyde"]
    assert "HyDE in the first stage" in notes["hybrid+hyde+rerank"] and "the reranker, which has the last word" in notes["hybrid+hyde+rerank"]
    claims = {c["claim"].split("`")[1]: c["not_shown"] for c in _frontier_claims(tmp_path)}
    assert "HyDE after the reranker" in claims["hybrid+rerank+hyde"] and "HyDE in the first stage" in claims["hybrid+hyde+rerank"]


def test_switches_added_after_main_json_stay_out_of_the_settings_of_rows_that_do_not_use_them():
    import json

    from app.eval.harness import DEFAULT_CONFIGS, bench_settings, project_root

    main = {r["config"]: r for r in json.loads((project_root() / "bench" / "results" / "main.json").read_text())["results"]}
    for cfg in DEFAULT_CONFIGS:  # make bench-check compares settings field by field
        assert bench_settings(cfg) == main[cfg.name]["settings"], cfg.name
    hyde = replace(DEFAULT_CONFIGS[2], hyde=True, hyde_stage="first_stage")
    assert bench_settings(hyde)["hyde_stage"] == "first_stage"


def test_contextual_chunks_index_a_prefix_and_keep_the_cited_bytes():
    from app.eval.harness import project_root
    from app.eval.index_loader import load_chunks

    root = project_root() / "bench" / "corpus" / "files"
    plain = {c.chunk_id: c for c in load_chunks(root)}
    situated = {c.chunk_id: c for c in load_chunks(root, contextual=True)}
    assert plain.keys() == situated.keys()
    for cid, chunk in situated.items():
        assert chunk.text.startswith("Experiment source path: " + chunk.path)
        assert chunk.text.endswith(plain[cid].text)
        assert chunk.metadata["evidence_raw"] == plain[cid].metadata["evidence_raw"]
        assert chunk.metadata["text_is_verbatim"] is False


def test_corrective_refine_returns_chunks_instead_of_crashing():
    # `make bench-frontier` could not finish before M7: the refine branch built a Chunk without its byte span.
    from app.agent.corrective import RelevanceLabel, EvaluatedChunk, determine_correction

    chunk = Chunk("c", "logs/run_047.out", "val_rmse 0.04 at epoch 9. unrelated line", 3, 43, mtime=5.0)
    action = determine_correction("val_rmse of run 47", [EvaluatedChunk(chunk, 0.4, RelevanceLabel.AMBIGUOUS)] * 2
                                  + [EvaluatedChunk(chunk, 0.0, RelevanceLabel.INCORRECT)])
    assert action.strategy == "refine" and action.refined_chunks
    refined = action.refined_chunks[0]
    assert (refined.path, refined.start_byte, refined.end_byte, refined.mtime) == ("logs/run_047.out", 3, 43, 5.0)
    assert "val_rmse" in refined.text and "unrelated" not in refined.text


def test_frontier_run_records_what_an_arm_needs_next_to_its_delta(tmp_path):
    import json

    from app.eval.harness import frontier_configs, run_bench

    wanted = {"hybrid", "hybrid+rerank", "hybrid+rerank+adaptive", "hybrid+rerank+graphrag", "hybrid+hipporag"}
    out = tmp_path / "frontier.json"
    run_bench(configs=[c for c in frontier_configs() if c.name in wanted], out=out, frontier=True)
    blob = json.loads(out.read_text())
    for key in ("git_sha", "timestamp", "command", "device", "corpus_sha256", "n_gold", "n_files", "bootstrap", "results", "frontier"):
        assert key in blob
    assert blob["frontier"]["parents"] == ["hybrid", "hybrid+rerank"]
    assert blob["frontier"]["not_evaluated"][0]["arm"] == "hybrid+late-chunking"
    rows = {r["config"]: r for r in blob["results"]}
    assert all("_retrieved" not in r and "_per_query_scores" not in r for r in rows.values())
    assert rows["hybrid"]["fallbacks"] == ["hash embedder (no model)"]
    assert "overlap reranker (no model)" in rows["hybrid+rerank"]["fallbacks"]
    for name in wanted - {"hybrid"}:
        row = rows[name]
        parent = row["settings"]["parent"]
        assert row["delta_vs_previous"]["reference"] == parent and row["vs_parent"]["parent"] == parent
        assert {"delta", "lo", "hi"} <= set(row["delta_vs_previous"]["ndcg@10"])
        cats = row["delta_vs_parent_by_category_all"]
        assert sum(c["n"] for c in cats) == blob["n_gold"] and all(c["reference"] == parent for c in cats)
    # An arm that ties its parent says why: the gate never said skip.
    adaptive = rows["hybrid+rerank+adaptive"]
    assert adaptive["fired"] == {"adaptive_retrieve": blob["n_gold"]}
    assert adaptive["vs_parent"]["n_list_differs"] == 0
    assert adaptive["delta_vs_previous"]["ndcg@10"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0}
    # The community step collapses the fixture: recorded, not hidden.
    g = rows["hybrid+rerank+graphrag"]
    assert g["graphrag"]["n_communities"] == 1 and g["graphrag"]["discriminates"] is False
    assert any("connected components" in label for label in g["fallbacks"])
    # The PPR gate is the router: the count of questions it fired on is in the row.
    assert 0 < rows["hybrid+hipporag"]["fired"]["hipporag"] == rows["hybrid+hipporag"]["vs_parent"]["n_list_differs"]


def test_default_rows_do_not_gain_frontier_keys():
    from app.eval.gold import load_gold
    from app.eval.harness import DEFAULT_CONFIGS, project_root, run_config

    root = project_root()
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")[:5]
    row = run_config(DEFAULT_CONFIGS[2], gold, root / "bench" / "corpus" / "files")
    assert not {"fired", "_retrieved", "fallbacks", "vs_parent", "graphrag", "late_chunking"} & set(row)


def test_late_chunking_pools_document_tokens_only_for_files_with_several_chunks(tmp_path):
    import numpy as np

    from app.chunking.late import late_chunk_index
    from app.retrieval.embedders import HashEmbedder

    class TokenEmbedder(HashEmbedder):
        """One-hot vector per whitespace token position, so pooled spans are easy to read."""

        def token_embeddings(self, text, max_tokens=510):
            import re

            offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            mat = np.zeros((len(offsets), self.dim), dtype=np.float32)
            for i in range(len(offsets)):
                mat[i, i] = 1.0
            return mat, offsets

    (tmp_path / "two.txt").write_bytes("alpha beta gamma delta".encode())
    (tmp_path / "one.txt").write_bytes("solo file".encode())
    chunks = [
        Chunk("t1", "two.txt", "alpha beta", 0, 10),
        Chunk("t2", "two.txt", "gamma delta", 11, 22),
        Chunk("o1", "one.txt", "solo file", 0, 9),
    ]
    index = InMemoryHybridIndex(chunks, embedder=TokenEmbedder(dim=8), enable_rerank=False, enable_router=False)
    before = index.doc_mat.copy()
    stats = late_chunk_index(index, tmp_path)
    assert stats["n_chunks_late_pooled"] == 2 and stats["n_files_with_several_chunks"] == 1 and stats["n_chunks"] == 3
    half = np.float32(1 / np.sqrt(2))
    assert np.allclose(index.doc_mat[0][:4], [half, half, 0, 0]) and np.allclose(index.doc_mat[1][:4], [0, 0, half, half])
    assert np.array_equal(index.doc_mat[2], before[2])  # a one-chunk file keeps its ordinary vector
    with pytest.raises(ValueError):
        late_chunk_index(InMemoryHybridIndex(chunks, enable_rerank=False, enable_router=False), tmp_path)


# ---- the committed results files -------------------------------------------------------------

@pytest.mark.parametrize("rel", ["bench/results/frontier.json", "bench/results/frontier_neural.json"])
def test_committed_frontier_file_is_self_contained(rel):
    import json

    from app.eval.harness import project_root

    path = project_root() / rel
    if not path.exists():
        pytest.skip(f"{rel} not produced yet")
    blob = json.loads(path.read_text())
    for key in ("git_sha", "timestamp", "command", "device", "corpus_sha256", "n_gold", "n_files", "bootstrap"):
        assert blob.get(key), key
    assert "--frontier" in blob["command"]
    rows = {r["config"]: r for r in blob["results"]}
    parents, arms = blob["frontier"]["parents"], blob["frontier"]["arms"]
    assert 2 <= len(parents) <= 3 and set(parents) <= set(rows) and arms
    for name in arms:
        row = rows[name]
        parent = row["settings"]["parent"]
        assert parent in parents, name
        delta = row["delta_vs_previous"]
        assert delta["reference"] == parent
        for metric in ("ndcg@10", "recall@10", "recall@50"):
            assert delta[metric]["lo"] <= delta[metric]["delta"] <= delta[metric]["hi"], (name, metric)
        assert row["vs_parent"]["n_questions"] == blob["n_gold"]
        assert row["fallbacks"] is not None and row["delta_vs_parent_by_category_all"]
        # an arm whose list never differs from its parent's must have a zero-width delta, and the row says what fired
        if row["vs_parent"]["n_list_differs"] == 0:
            assert delta["ndcg@10"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0} and row.get("fired")
    graph = [r for n, r in rows.items() if "graphrag" in n]
    assert graph and all("n_communities" in r["graphrag"] and "discriminates" in r["graphrag"] for r in graph)


# ---- reporting: generated only ---------------------------------------------------------------

def test_verdict_word_comes_from_the_interval_alone():
    from app.eval.report import verdict

    assert verdict({"delta": 0.05, "lo": 0.01, "hi": 0.09}) == "above zero"
    assert verdict({"delta": 0.05, "lo": -0.01, "hi": 0.09}) == "covers zero"
    assert verdict({"delta": -0.05, "lo": -0.09, "hi": -0.01}) == "below zero"
    assert verdict({"delta": 0.0, "lo": 0.0, "hi": 0.0}) == "covers zero"  # an arm identical to its parent
    assert verdict(None) == "n/a"


def test_frontier_block_has_one_row_per_arm_with_parent_verdict_and_stand_ins():
    import json

    from app.eval.harness import project_root
    from app.eval.report import FRONTIER_FILE, frontier_table, verdict

    blob = json.loads((project_root() / FRONTIER_FILE).read_text())
    text = frontier_table(blob)
    assert blob["git_sha"] in text and "make bench-frontier" in text
    rows = {r["config"]: r for r in blob["results"]}
    for name in blob["frontier"]["arms"]:
        line = next(l for l in text.splitlines() if l.startswith(f"| {name} |"))
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[1] == rows[name]["settings"]["parent"]
        assert cells[4] == verdict(rows[name]["delta_vs_previous"]["ndcg@10"])
        assert "hash embedder (no model)" in cells[-1]  # every stand-in is named in the row
    assert "1 community" in text and "cannot tell one file from another" in text
    assert "hybrid+late-chunking`: not evaluated in this run" in text
    assert "is missing; run `make bench-frontier`" in frontier_table(None)


def test_evidence_table_has_a_row_for_every_frontier_arm():
    import json

    from app.eval.evidence import FILES, claims
    from app.eval.harness import project_root

    rows = claims()
    for key in ("frontier", "frontier_neural"):
        path = project_root() / FILES[key]
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        for name in blob["frontier"]["arms"]:
            mine = [r for r in rows if f"`{name}`" in r["claim"] and FILES[key] in r["source"]]
            assert len(mine) == 1, name
            assert "verdict from the interval" in mine[0]["not_shown"] and "stand-ins" in mine[0]["not_shown"]


# ---- reranker prototypes against the cross-encoder ---------------------------------------------

def test_student_training_rows_use_serving_features_not_the_teacher_rank():
    from app.retrieval.distill import DistilledReranker, serving_features, triples_from_hits, triples_from_pairs
    from app.retrieval.types import RetrievalHit

    index = _tiny_index()
    query = "what learning rate did run 47 use"
    pairs = [(h.chunk, h.score) for h in index.retrieve(query, k=5).hits]
    teacher = {chunk.chunk_id: float(10 - i) for i, (chunk, _) in enumerate(reversed(pairs))}  # the teacher reverses the list
    rows = triples_from_pairs(query, pairs, teacher)
    assert [r.features for r in rows] == serving_features(query, pairs)
    assert [r.features["rrf_rank"] for r in rows] == [1.0, 2.0, 3.0]  # the fused rank, whatever the teacher said
    assert all(r.features["dense_cosine"] == 0.0 and r.features["bm25_score"] == 0.0 for r in rows)  # absent when serving
    assert [r.ce_score for r in rows] == [teacher[c.chunk_id] for c, _ in pairs]
    # The bench log is a different thing: after a rerank its rank feature is the teacher's own order.
    reranked = [RetrievalHit(c, teacher[c.chunk_id], rank, rerank=teacher[c.chunk_id]) for rank, (c, _) in enumerate(reversed(pairs), start=1)]
    assert [t.features["rrf_rank"] for t in triples_from_hits(query, reranked)] == [1.0, 2.0, 3.0]
    student = DistilledReranker().fit(rows * 3)
    assert {c.chunk_id for c, _ in student(query, pairs)} == {c.chunk_id for c, _ in pairs}


def test_rerank_arm_split_alternates_inside_each_category_and_scores_pairs():
    from app.eval.gold import load_gold
    from app.eval.harness import project_root
    from app.eval.rerank_arms import score_rows, split_questions

    gold = load_gold(project_root() / "bench" / "gold" / "questions.jsonl")
    train, test = split_questions(gold)
    assert not set(train) & set(test) and sorted(train + test) == list(range(len(gold)))
    cats = lambda ids: {c: sum(1 for i in ids if gold[i].category == c) for c in {q.category for q in gold}}  # noqa: E731
    assert all(abs(cats(train)[c] - cats(test)[c]) <= 1 for c in cats(train))
    perfect = [list(q.relevant_ids()) for q in gold]
    empty = [[] for _ in gold]
    rows, boot = score_rows({"good": perfect, "bad": empty}, gold, test, {"good": [1.0] * len(gold)}, 61, [("bad", "good")])
    by = {r["config"]: r for r in rows}
    assert by["good"]["n"] == len(test) == boot["n_queries"] and by["good"]["retrieval"]["ndcg@10"] == 1.0
    assert by["good"]["rerank_latency"]["n"] == len(test)
    (delta,) = by["bad"]["deltas"]
    assert delta["reference"] == "good" and delta["ndcg@10"] == {"delta": -1.0, "lo": -1.0, "hi": -1.0}
    assert set(delta) == {"reference", "ndcg@10", "recall@10", "recall@50"}


def test_rerank_arms_block_reads_verdicts_off_the_file():
    import json

    from app.eval.harness import project_root
    from app.eval.report import RERANK_ARMS_FILE, rerank_arms_table, verdict

    assert "make bench-rerank-arms" in rerank_arms_table(None)
    path = project_root() / RERANK_ARMS_FILE
    if not path.exists():
        pytest.skip(f"{RERANK_ARMS_FILE} not produced yet")
    blob = json.loads(path.read_text())
    text = rerank_arms_table(blob)
    student = next(r for r in blob["test_split"]["results"] if "student" in r["config"])
    line = next(l for l in text.splitlines() if l.startswith(f"| {student['config']} |"))
    vs_teacher = next(b for b in student["deltas"] if b["reference"].endswith("bge-rerank@bge-small"))["ndcg@10"]
    assert f"| {verdict(vs_teacher)} |" in line and str(student["rerank_latency"]["p50_ms"]) in line
    if student["student"]["fallback"]:
        assert student["student"]["fallback"] in line  # the stand-in is named in the row
    assert set(blob["split"]["train_ids"]).isdisjoint(blob["split"]["test_ids"])
    if not blob["listwise"]["evaluated"]:
        assert "not evaluated in this run" in text
