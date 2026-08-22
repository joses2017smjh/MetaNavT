"""Phase 6–10 frontier plan: jury, RankGPT, Deep Research, HippoRAG, Search-R1."""

from app.agent.deep_research import (
    DeepResearchAgent,
    Scratchpad,
    expand_queries,
    rrf_union_paths,
)
from app.eval.harness import DEFAULT_CONFIGS, FRONTIER_CONFIGS, BenchConfig
from app.eval.judge import (
    HeuristicJudge,
    exact_match_label,
    pairwise_ab_ba,
    parse_score,
    self_consistency_vote,
)
from app.eval.jury import (
    KAPPA_GATE,
    Jury,
    cohens_kappa,
    kappa_vs_gold,
    select_best_of_n,
)
from app.eval.index_loader import build_index
from app.graph.hipporag import apply_hipporag, personalized_pagerank, query_seeds, triples_from_chunks
from app.graph.graphrag import Triple, extract_triples
from app.retrieval.hybrid import InMemoryHybridIndex
from app.retrieval.rankgpt import RankGPTReranker, bge_reranker, parse_permutation as parse_perm
from app.retrieval.rerank import CrossEncoderReranker, OverlapReranker
from app.retrieval.types import Chunk, RetrievalHit
from app.rl.grpo import dummy_step, dummy_train_on_index, grpo_advantages, masked_nll
from app.rl.reward import search_r1_reward
from app.rl.search_r1 import SearchR1Env, parse_trajectory, token_train_mask


def test_parse_score_json_and_label():
    assert parse_score('{"score": 0.8, "label": "correct"}') == 0.8
    assert parse_score("SCORE: 7") == 0.7
    assert parse_score("wrong") == 0.0


def test_pairwise_ab_ba_averages_and_reports_position_gap():
    def always_first(prompt: str) -> str:
        return "A"

    biased = pairwise_ab_ba(always_first, "q", "good", "bad")
    assert biased["p_a"] == 0.5
    assert biased["position_gap"] == 1.0

    def prefer_good(prompt: str) -> str:
        if "Answer A:\ngood" in prompt:
            return "A"
        if "Answer B:\ngood" in prompt:
            return "B"
        return "TIE"

    fair = pairwise_ab_ba(prefer_good, "q", "good", "bad")
    assert fair["p_a"] == 1.0
    assert fair["position_gap"] == 0.0

    def tie(_prompt: str) -> str:
        return "TIE"

    tied = pairwise_ab_ba(tie, "q", "a", "b")
    assert tied["p_a"] == 0.5
    assert tied["position_gap"] == 0.0


def test_self_consistency_majority():
    replies = iter(["correct", "wrong", "correct"])

    def complete(_prompt: str) -> str:
        return next(replies)

    vote = self_consistency_vote(complete, "grade this", k=3)
    assert vote["label"] == "correct"
    assert vote["agreement"] == 2 / 3
    assert vote["votes"]["correct"] == 2


def test_cohens_kappa_perfect_and_gate():
    labels = ["correct", "partial", "wrong", "correct"]
    assert cohens_kappa(labels, labels) == 1.0
    disagree = ["wrong", "wrong", "wrong", "wrong"]
    assert cohens_kappa(labels, disagree) < KAPPA_GATE


def test_jury_tie_is_partial_and_majority_wins():
    class Fixed:
        def __init__(self, label: str):
            self.label = label

        def judge_answer(self, *a, **k):
            from app.eval.judge import PointwiseVerdict

            return PointwiseVerdict(0.5, 0.5, 0.5, self.label, {})

    tied = Jury([("a", Fixed("correct")), ("b", Fixed("wrong"))]).vote("q", "a", ["c"], "g")
    assert tied.label == "partial"
    maj = Jury(
        [("a", Fixed("correct")), ("b", Fixed("correct")), ("c", Fixed("wrong"))]
    ).vote("q", "a", ["c"], "g")
    assert maj.label == "correct"


def test_kappa_gate_heuristic_on_simple_factual():
    golds = ["3e-4", "dinov2", "oak", "0.05"]
    answers = [
        "learning_rate: 3e-4 in run 47",
        "encoder dinov2",
        "bark_type: oak",
        "val_rmse 0.05",
    ]
    contexts = answers
    judge = HeuristicJudge()
    labels = [judge.judge_answer("q", a, [c], g).label for a, c, g in zip(answers, contexts, golds)]
    report = kappa_vs_gold(answers, golds, labels)
    gold_labels = [exact_match_label(a, g) for a, g in zip(answers, golds)]
    assert gold_labels == ["correct"] * 4
    assert report.kappa >= KAPPA_GATE
    assert report.gated is True


def test_select_best_of_n_by_gold_path_citations():
    traces = [
        {"id": "a", "paths": ["other.yaml"], "n_search_calls": 1},
        {"id": "b", "paths": ["configs/run_047.yaml", "src/fusion.py"], "n_search_calls": 2},
        {"id": "c", "paths": ["configs/run_047.yaml"], "n_search_calls": 1},
    ]
    best = select_best_of_n(traces, ["configs/run_047.yaml", "src/fusion.py"])
    assert best["id"] == "b"


def test_rankgpt_permutation_and_heuristic_rerank():
    assert parse_perm("[3] > [1] > [2]", 3) == [2, 0, 1]
    chunks = [
        Chunk("a", "noise.txt", "zzzz", 0, 4),
        Chunk("b", "configs/run_047.yaml", "learning_rate 3e-4 dinov2", 0, 40),
    ]
    ranked = RankGPTReranker()("what learning rate dinov2", [(chunks[0], 0.9), (chunks[1], 0.1)])
    assert ranked[0][0].chunk_id == "b"


def test_bge_reranker_labels_fallback_when_model_missing():
    fn, loaded = bge_reranker("definitely-not-a-real-model-xxxxx")
    assert loaded is False
    assert isinstance(fn, CrossEncoderReranker)
    pairs = [(Chunk("c", "a.yaml", "hello world", 0, 11), 0.5)]
    out = fn("hello", pairs)
    assert out[0][0].chunk_id == "c"


def test_index_loader_records_honest_fallback(tmp_path):
    (tmp_path / "a.yaml").write_text("learning_rate: 3e-4")
    idx = build_index(tmp_path, embedder_name="hash", reranker="BAAI/bge-reranker-v2-m3")
    if idx.reranker_loaded:
        assert not idx.reranker_fallback
    else:
        assert idx.reranker_fallback == "overlap"
    gpt = build_index(tmp_path, embedder_name="hash", reranker="rankgpt")
    assert gpt.reranker_fallback == "overlap-listwise"


def test_expand_queries_and_scratchpad_budget():
    qs = expand_queries("what learning rate did run 47 use", n=3)
    assert len(qs) == 3
    assert qs[0].startswith("what learning")
    pad = Scratchpad(token_budget=2)
    fat = RetrievalHit(
        Chunk("c1", "configs/run_047.yaml", "learning_rate 3e-4 " * 40, 0, 80), 1.0, 1
    )
    pad.add_hits([fat])
    assert pad.can_respond() is False or pad.tokens_used <= 2


def test_deep_research_cites_and_refuses_empty_budget():
    chunks = [
        Chunk("c1", "configs/run_047.yaml", "run_id: 47 learning_rate: 3e-4 dinov2", 0, 40),
        Chunk("c2", "src/fusion.py", "fusion concatenates per-view features", 0, 40),
    ]
    idx = InMemoryHybridIndex(chunks, retrieve_k=5, rerank_n=3, rerank_fn=OverlapReranker())
    agent = DeepResearchAgent(idx, n_queries=3, token_budget=2048)
    ans = agent.run("what learning rate did run 47 use")
    assert not ans.failed
    assert ans.citations
    assert ans.citations[0].path
    tight = DeepResearchAgent(idx, n_queries=2, token_budget=1, grade_threshold=0.99)
    empty = tight.run("unrelated zzzyx tokens")
    assert empty.failed
    assert empty.fail_reason == "empty_scratchpad"


def test_rrf_union_merges_paraphrases():
    chunks = [
        Chunk("c1", "configs/run_047.yaml", "run_id: 47 learning_rate: 3e-4", 0, 40),
        Chunk("c2", "logs/run_047.out", "val_rmse 0.04 dinov2", 0, 30),
    ]
    idx = InMemoryHybridIndex(chunks, retrieve_k=5, enable_rerank=False)
    hits = rrf_union_paths(idx, ["learning rate run 47", "dinov2 rmse run 47"])
    assert {h.chunk.path for h in hits} >= {"configs/run_047.yaml"}


def test_hipporag_boosts_aggregation_not_simple_factual():
    chunks = [
        Chunk("a", "configs/run_040.yaml", "run_id: 40 encoder: resnet50 learning_rate: 1e-4", 0, 60),
        Chunk("b", "configs/run_047.yaml", "run_id: 47 encoder: dinov2 learning_rate: 3e-4", 0, 60),
    ]
    triples = triples_from_chunks(chunks)
    assert any(t.relation == "uses_encoder" and "dinov2" in t.dst for t in triples)
    hits = [
        RetrievalHit(chunks[0], 1.0, 1),
        RetrievalHit(chunks[1], 0.2, 2),
    ]
    boosted = apply_hipporag("which run used dinov2", hits, triples, category="aggregation")
    assert boosted[0].chunk.path == "configs/run_047.yaml"
    unchanged = apply_hipporag(
        "what learning rate did run 40 use", hits, triples, category="simple_factual"
    )
    assert unchanged[0].chunk.path == "configs/run_040.yaml"
    seeds = query_seeds("which run used dinov2")
    assert any("dinov2" in s for s in seeds)
    ppr = personalized_pagerank({"encoder:dinov2": {"run:47": 1.0}, "run:47": {"encoder:dinov2": 1.0}}, seeds)
    assert ppr["encoder:dinov2"] > 0


def test_search_r1_masks_retrieved_tokens_and_rewards():
    raw = (
        "<search>lr run 47</search>\n"
        "<information>configs/run_047.yaml learning_rate 3e-4</information>\n"
        "<answer>[configs/run_047.yaml] 3e-4</answer>"
    )
    traj = parse_trajectory(raw)
    assert traj.n_search == 1
    assert traj.answer.endswith("3e-4")
    mask = token_train_mask(raw)
    info_tokens = token_train_mask("<information>configs/run_047.yaml learning_rate 3e-4</information>")
    assert info_tokens and not any(info_tokens)
    assert len(mask) == len(traj.tokens)
    assert False in mask and True in mask
    traj.retrieved_paths = ["configs/run_047.yaml"]
    traj.cited = True
    good = search_r1_reward(traj, gold="3e-4", gold_paths=["configs/run_047.yaml"])
    assert good["em"] == 1.0
    assert good["ndcg@10"] == 1.0
    assert good["jury_applied"] is False
    empty = parse_trajectory("<search>q</search>\n<answer></answer>")
    empty.retrieved_paths = []
    bad = search_r1_reward(empty, gold="3e-4", gold_paths=["configs/run_047.yaml"])
    assert bad["uncited_or_empty"] == 1.0
    assert bad["reward"] < good["reward"]


def test_grpo_advantages_and_dummy_cpu_step():
    adv = grpo_advantages([3.0, 2.0, 1.0])
    assert abs(sum(adv)) < 1e-9
    assert adv[0] > 0 > adv[2]
    assert masked_nll([-1.0, -2.0], [True, False]) == 1.0
    t_good = parse_trajectory(
        "<search>q</search><information>secret retrieved</information><answer>ok</answer>"
    )
    t_bad = parse_trajectory("<search>q</search><answer></answer>")
    step = dummy_step([t_good, t_bad], [1.0, -1.0])
    assert step["n"] == 2
    assert step["backend"] == "numpy-dummy"
    assert len(step["advantages"]) == 2


def test_search_r1_env_heuristic_rollout():
    chunks = [
        Chunk("c1", "configs/run_047.yaml", "run_id: 47 learning_rate: 3e-4", 0, 40),
    ]
    idx = InMemoryHybridIndex(chunks, retrieve_k=3, enable_rerank=False)
    env = SearchR1Env(idx)
    traj = env.rollout("what learning rate did run 47 use")
    assert traj.n_search == 1
    assert traj.informations
    assert "run_047" in traj.answer or "3e-4" in traj.answer
    blob = dummy_train_on_index(
        idx, "what learning rate did run 47 use", "3e-4", ["configs/run_047.yaml"]
    )
    assert blob["n"] == 4
    assert "rewards" in blob


def test_frontier_configs_do_not_replace_default_six():
    assert len(DEFAULT_CONFIGS) == 6
    names = [c.name for c in FRONTIER_CONFIGS]
    assert names == [
        "hybrid+bge-rerank",
        "hybrid+rankgpt",
        "hybrid+multiquery",
        "hybrid+hipporag",
    ]
    assert DEFAULT_CONFIGS[0].jury is False
    bge = FRONTIER_CONFIGS[0]
    assert isinstance(bge, BenchConfig)
    assert bge.reranker == "BAAI/bge-reranker-v2-m3"


def test_extract_triples_for_hipporag_seed():
    t = extract_triples("configs/run_047.yaml", "run_id: 47 encoder: dinov2")
    assert any(isinstance(x, Triple) and x.dst == "encoder:dinov2" for x in t)
