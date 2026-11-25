"""M48 (app/eval/adapted_embedders.py, app/eval/embedder_train.py): what was declared before any adapted embedder was
trained or measured, and the code that trains, loads, scores and re-scores them.

The declarations are pinned value by value: a bar, a recipe or an arm list that could change quietly after a number
exists would not be a declaration. Tests that need torch skip where it is not installed (the login node's suites) and
run under the GPU environment (recorded in milestones2/m48/NOTES.md).
"""

import builtins
import gzip
import hashlib
import json
import math
import pathlib
from pathlib import Path

import numpy as np
import pytest

from app.eval import adapted_embedders as ae
from app.eval import embedder_train as et

ROOT = Path(__file__).resolve().parents[2]


def test_the_bar_is_one_named_row_on_three_datasets():
    assert ae.BAR_ARM == "bge-small-marginmse"
    assert ae.BAR["row"] == "hybrid@bge-small-marginmse" and ae.BAR["parent"] == "hybrid@bge-small"
    assert ae.BAR["metric"] == "ndcg@10"
    assert ae.BAR["datasets"] == {"fixture": 68, "scifact": 300, "fiqa": 648}
    for needle in ("strictly above zero", "a stored 0.0 is not above zero", "held-out half (68 questions)", "SciFact test (300)",
                   "FiQA test (648)"):
        assert needle in ae.BAR["passes"]
    assert "seed 0 alone" in ae.BAR["one_seed"]
    assert "never read from the shared cache" in ae.BAR["parent_rule"]
    assert "RRF, k = 60, equal weights" in ae.BAR["hybrid"]
    assert "split_questions" in ae.BAR["held_out"] and "no choice of any kind is made on the tune half" in ae.BAR["held_out"]
    assert ae.BAR["read_from"] == ("bench/results/adapted_embedders.json (scope held_out)", "bench/results/beir_scifact_adapted_embedders.json",
                                   "bench/results/beir_fiqa_adapted_embedders.json")
    assert "first complete eval run per dataset at the C3" in ae.BAR["which_run"] and "never replaces it" in ae.BAR["which_run"]
    assert "2000 resamples, seed 0" in ae.BAR["interval"]
    assert tuple(ae.BAR["outcomes"]) == ae.OUTCOMES == ("passes", "does not pass", "open", "not run")
    assert "intersection-union" in ae.BAR["multiple_testing"] and "2.5%" in ae.BAR["multiple_testing"]
    assert "no metric uses an LLM judge" in ae.BAR["llm_judge"]
    assert ae.ATTEMPT == 1


def test_the_arms_are_declared_in_table_order():
    assert list(ae.ARMS) == ["bge-small-marginmse", "bge-small-marginmse-nofixture", "bge-small-marginmse-s1", "bge-small-marginmse-s2",
                             "bge-small-mnrl", "bge-base-marginmse"]
    shape = {t: (a["init"], a["loss"], a["training_set"], a["seed"], a["role"]) for t, a in ae.ARMS.items()}
    assert shape == {
        "bge-small-marginmse": ("bge-small", "marginmse", "all", 0, "bar"),
        "bge-small-marginmse-nofixture": ("bge-small", "marginmse", "beir", 0, "control"),
        "bge-small-marginmse-s1": ("bge-small", "marginmse", "all", 1, "seed"),
        "bge-small-marginmse-s2": ("bge-small", "marginmse", "all", 2, "seed"),
        "bge-small-mnrl": ("bge-small", "mnrl", "all", 0, "loss"),
        "bge-base-marginmse": ("bge-base", "marginmse", "all", 0, "base"),
    }
    assert ae.TRAINING_SETS == {"all": ("fixture", "scifact", "fiqa"), "beir": ("scifact", "fiqa")}
    assert ae.SCALE_KEYS == (("bge-small", "all"), ("bge-small", "beir"), ("bge-base", "all"))
    assert ae.SEED_ARMS == ("bge-small-marginmse", "bge-small-marginmse-s1", "bge-small-marginmse-s2")
    assert ae.CONTROL_ARM == "bge-small-marginmse-nofixture" and ae.MNRL_ARM == "bge-small-mnrl" and ae.BASE_ARM == "bge-base-marginmse"
    assert "confounded" in ae.ARMS[ae.CONTROL_ARM]["about"]


def test_the_inits_are_pinned_and_encode_as_every_committed_bge_row():
    assert ae.INITS["bge-small"] == {"model": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                                     "licence": "mit", "embedder": "st:BAAI/bge-small-en-v1.5"}
    assert ae.INITS["bge-base"]["revision"] == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a" and ae.INITS["bge-base"]["optional"] is True
    assert ae.ENCODING == {"pooling": "cls", "normalize": True, "max_seq_length": 512, "do_lower_case": True, "dtype": "float32",
                           "query_prompt": None, "document_prompt": None}
    from app.retrieval.embedders import QUERY_PROMPT_NAMES

    assert not any(init["model"] in QUERY_PROMPT_NAMES for init in ae.INITS.values())  # bare queries, as every bge row


def test_the_recipe_is_m43s_with_a_fixed_scale():
    from app.eval import small_rerankers as sr

    numbers = {k: v for k, v in ae.RECIPE.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    assert numbers == {"lists_per_step": 8, "epochs": 4, "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_fraction": 0.1, "max_grad_norm": 1.0}
    for key in numbers:  # M43's optimisation recipe, unchanged
        assert sr.RECIPE[key] == numbers[key], key
    assert ae.RECIPE["corpus_mixing"] == sr.RECIPE["corpus_mixing"]
    assert ae.RECIPE["train_lists"] == {"fixture": 691, "scifact": 4267, "fiqa": 5129}
    assert ae.RECIPE["validation_lists"] == {"fixture": 42, "scifact": 223, "fiqa": 270}
    assert "std with ddof 0 in float64" in ae.RECIPE["scale"] and "every pair i < j" in ae.RECIPE["scale"]
    assert "mean over i < j of ((s_i - s_j) - (t_i - t_j))^2" in ae.RECIPE["loss"]
    assert "s * cos(E(q), E(d))" in ae.RECIPE["student_score"] and "preprocess()" in ae.RECIPE["student_score"]
    assert "allow_tf32 False" in ae.RECIPE["precision"] and "'highest'" in ae.RECIPE["precision"]
    assert "never a hyper-parameter" in ae.RECIPE["memory"]
    assert "never a gold question, a tune-half question or a BEIR query" in ae.RECIPE["validation"]
    assert "ties go to the earlier epoch" in ae.RECIPE["selection"] and "not used" in ae.RECIPE["selection"]
    assert ae.RECIPE["prompts"] is None


def test_the_mnrl_arm_rules_are_fixed():
    m = ae.RECIPE["mnrl"]
    assert (m["scale"], m["hard_negatives"], m["queries_per_batch"]) == (20.0, 7, 32)
    assert "10 lowest-scored candidates of a 20- or 21-candidate list" in m["hard_negative_rule"]
    assert "the first 7 of those in the list's order" in m["hard_negative_rule"]
    assert "teacher ranks the source document first" in m["lists"]
    assert "-inf, except the anchor's own positive column" in m["mask"]
    assert "never dropped" in m["batches"] and "square root" in m["batches"]
    assert "batches built once with seed 0" in m["selection"]
    assert "40 GB or more" in m["hardware"]


def test_the_twin_rule_and_the_files_are_named():
    assert ae.TWINS["thresholds"] == (0.80, 0.85) and ae.TWINS["fixture_reading_threshold"] == 0.85
    assert (ae.TWINS["assert_below"], ae.TWINS["assert_tolerance"], ae.TWINS["min_subgroup"]) == (0.90, 0.001, 10)
    assert ae.FIXTURE_READINGS == ("adaptation beyond templated twins", "not separable from templated twins")
    assert ae.CUTOFF == "2026-12-06" and ae.CPU_BAND == (0.90, 1.10)
    assert ae.RESULTS == {"fixture": "bench/results/adapted_embedders.json", "scifact": "bench/results/beir_scifact_adapted_embedders.json",
                          "fiqa": "bench/results/beir_fiqa_adapted_embedders.json"}
    assert all(p.endswith("adapted_embedders_base.json") for p in ae.BASE_RESULTS.values())
    assert (ae.SCALE_FILE, ae.TWINS_FILE, ae.MANIFEST_FILE) == ("bench/distill/embedder_scale.json", "bench/distill/embedder_twins.json",
                                                                "bench/distill/embedders.json")
    assert ae.SIDECAR_SCHEMA == "adapted-embedders-per-query-v1"
    assert ae.embedder_name(ae.BAR_ARM) == "ft:bge-small-marginmse"
    assert ae.row_names(ae.BAR_ARM, "fixture") == ("dense_only@bge-small-marginmse", "hybrid@bge-small-marginmse")
    assert ae.row_names("bge-small", "fiqa") == ("dense@bge-small", "hybrid@bge-small")


def test_the_declared_list_counts_are_the_committed_files():
    meta = json.loads((ROOT / "bench" / "distill" / "queries.meta.json").read_text())
    kept = {c: b["kept_by_split"] for c, b in meta["near_duplicates"]["by_corpus"].items()}
    assert {c: k["train"] for c, k in kept.items()} == ae.RECIPE["train_lists"]
    assert {c: k["validation"] for c, k in kept.items()} == ae.RECIPE["validation_lists"]


# ---------------------------------------------------------------- the code (written after the declarations commit)


def _torch():
    return pytest.importorskip("torch")


def test_the_scale_rule_on_toy_lists():
    teacher = [[3.0, 1.0, 0.0], [2.0, 2.5]]
    cosine = [[0.9, 0.7, 0.6], [0.8, 0.85]]
    t = np.array([2.0, 3.0, 1.0, -0.5])  # every i < j margin of both lists, pooled
    c = np.array([0.2, 0.3, 0.1, -0.05])
    fit = et.scale_from_lists(teacher, cosine)
    assert fit["pairs"] == 4 and fit["lists"] == 2
    assert fit["s"] == pytest.approx(np.std(t, ddof=0) / np.std(c, ddof=0), rel=1e-12)
    assert fit["temperature"] == pytest.approx(1 / fit["s"])
    scaled = et.scale_from_lists(teacher, [[x / 7.0 for x in row] for row in teacher])  # cosines that are the teacher / 7
    assert scaled["s"] == pytest.approx(7.0, rel=1e-12)
    assert list(et.pair_margins([1.0, 4.0, 2.0])) == [-3.0, -1.0, 2.0]
    with pytest.raises(ValueError, match="no spread"):
        et.scale_from_lists([[1.0, 2.0]], [[0.5, 0.5]])


def test_the_mixed_sampler_resumes_mid_epoch_exactly():
    from app.eval.distill_train import MixedSampler

    sizes = {"fixture": 5, "scifact": 9, "fiqa": 12}
    full = MixedSampler(sizes, seed=3)
    first = [full.draw() for _ in range(17)]  # past a reshuffle of the smallest corpus
    saved = et.sampler_state(full)
    rest = [full.draw() for _ in range(40)]
    again = MixedSampler(sizes, seed=99)  # another seed: everything comes from the state
    et.load_sampler_state(again, saved)
    assert [again.draw() for _ in range(40)] == rest
    fresh = MixedSampler(sizes, seed=3)
    assert [fresh.draw() for _ in range(17)] == first  # the added state methods did not change the draw sequence


def _example(i, pos, corpus="c"):
    return {"id": f"{corpus}:{i}", "query": f"q{i}", "pos": pos, "pos_text": f"t{pos}", "neg": [f"n{i}{j}" for j in range(7)],
            "neg_texts": [f"tn{i}{j}" for j in range(7)]}


def test_the_mnrl_sampler_never_puts_a_source_twice_in_a_batch_and_resumes():
    sources = {"a": [f"d{i % 6}" for i in range(30)], "b": [f"e{i % 11}" for i in range(40)]}
    sampler = et.MnrlSampler({"a": 30, "b": 40}, sources, batch=5, seed=0)
    for _ in range(30):
        corpus, idx = sampler.draw()
        assert len(idx) == 5 and len({sources[corpus][i] for i in idx}) == 5
    state = sampler.state()
    rest = [sampler.draw() for _ in range(20)]
    again = et.MnrlSampler({"a": 30, "b": 40}, sources, batch=5, seed=7)
    again.load_state(state)
    assert [again.draw() for _ in range(20)] == rest
    assert sampler.deferred > 0  # collisions were deferred, never dropped
    with pytest.raises(ValueError, match="distinct source documents"):
        et.MnrlSampler({"a": 4}, {"a": ["x", "x", "y", "z"]}, batch=4, seed=0)


def test_the_hard_negatives_are_the_first_seven_of_the_teachers_bottom_half():
    scores = [9.0, 1.0, 5.0, 0.5, 4.0, 3.0, 0.1, 2.0, 7.0, 0.2, 6.0, 0.3, 8.0, 0.4, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    bottom = sorted(np.argsort(scores, kind="stable")[:10])
    assert et.hard_negatives(scores) == [int(i) for i in bottom][:7]
    assert 0 not in et.hard_negatives(scores)  # the teacher's first (the source) is never a negative
    assert len(et.hard_negatives(scores + [-1.0])) == 7  # a 21-candidate list: its 10 lowest


def test_the_mnrl_validation_batches_cover_every_example_once():
    examples = {"fixture": [_example(i, f"d{i % 4}", "fixture") for i in range(37)], "fiqa": [_example(i, f"f{i}", "fiqa") for i in range(70)]}
    batches = et.validation_batches(examples, 32, seed=0)
    seen = [(c, i) for c, idx in batches for i in idx]
    assert sorted(seen) == sorted((c, i) for c, rows in examples.items() for i in range(len(rows)))
    for c, idx in batches:
        assert len(idx) <= 32 and len({examples[c][i]["pos"] for i in idx}) == len(idx)
    assert et.validation_batches(examples, 32, seed=0) == batches  # built once, the same every epoch


def test_an_arms_checkpoint_is_verified_against_the_manifest(tmp_path, monkeypatch):
    directory = tmp_path / "ckpt" / ae.BAR_ARM
    directory.mkdir(parents=True)
    (directory / ae.WEIGHTS_FILE).write_bytes(b"weights")
    manifest = {"arms": {ae.BAR_ARM: {"directory": ae.BAR_ARM, "weights_sha256": hashlib.sha256(b"weights").hexdigest(), "attempt": 1}}}
    found, entry = ae.arm_checkpoint(ae.BAR_ARM, tmp_path / "ckpt", manifest)
    assert found == directory and entry["attempt"] == 1
    (directory / ae.WEIGHTS_FILE).write_bytes(b"other weights")
    with pytest.raises(ae.CheckpointError, match="the manifest records"):
        ae.arm_checkpoint(ae.BAR_ARM, tmp_path / "ckpt", manifest)
    with pytest.raises(ae.CheckpointError, match="is not in"):
        ae.arm_checkpoint(ae.CONTROL_ARM, tmp_path / "ckpt", manifest)
    monkeypatch.delenv(ae.CHECKPOINT_ENV, raising=False)
    with pytest.raises(ae.CheckpointError, match="no checkpoint directory"):
        ae.arm_checkpoint(ae.BAR_ARM, None, manifest)


def test_the_loader_refuses_another_cached_revision(tmp_path, monkeypatch):
    from app.retrieval.embedders import SentenceTransformerEmbedder

    hub = tmp_path / "hub" / "models--BAAI--bge-small-en-v1.5" / "refs"
    hub.mkdir(parents=True)
    (hub / "main").write_text("f" * 40)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    with pytest.raises(RuntimeError, match="the pinned one is 5c38ec7c"):
        SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", revision=ae.INITS["bge-small"]["revision"])
    with pytest.raises(RuntimeError, match="the pinned one is 5c38ec7c"):
        et.load_init("bge-small")


def _engine(tmp_path, **kw):
    from app.eval.beir import Engine, chunks_from_corpus

    return Engine(chunks_from_corpus({"d1": "alpha beta", "d2": "gamma delta", "d3": "beta gamma"}), cache_dir=tmp_path / "cache", **kw)


def test_the_cache_key_of_an_adapted_embedder_is_its_weights_sha256(tmp_path):
    engine = _engine(tmp_path)
    engine.provenance["ft:bge-small-marginmse"] = {"weights_sha256": "ab" * 32, "revision": None}
    assert ("ab" * 32) in engine.cache_path("ft:bge-small-marginmse").name
    engine.provenance["st:BAAI/bge-small-en-v1.5"] = {"revision": "5c38ec7c"}
    assert ".5c38ec7c." in engine.cache_path("st:BAAI/bge-small-en-v1.5").name  # unchanged for a hub model


def test_in_run_matrices_are_computed_and_never_read_from_or_written_to_the_cache(tmp_path):
    engine = _engine(tmp_path, in_run_matrices=True)
    engine.embedder("hash")
    cache = engine.cache_path("hash")
    cache.parent.mkdir(parents=True)
    np.save(cache, np.zeros((3, engine.embedder("hash").dim), dtype=np.float32))  # a stale matrix that must not be used
    before = sorted(p.name for p in cache.parent.iterdir())
    mat = engine.doc_matrix("hash")
    assert np.abs(mat).sum() > 0
    assert engine.provenance["hash"]["doc_embeddings_vs_cache"]["cache_file_found"] is True
    assert engine.provenance["hash"]["doc_embeddings_vs_cache"]["max_abs_difference"] > 0
    assert sorted(p.name for p in cache.parent.iterdir()) == before


def test_reproduces_reads_a_scope_of_a_file_shaped_like_small_rerankers(tmp_path):
    from app.eval.models_2026 import reproduces

    retrieval = {"ndcg@10": 0.7, "recall@10": 0.8}
    (tmp_path / "f.json").write_text(json.dumps({"git_sha": "x", "held_out": {"results": [{"config": "hybrid@bge-small", "retrieval": retrieval}]}}))
    mine = [{"config": "hybrid@bge-small", "retrieval": retrieval}]
    assert reproduces(tmp_path, "f.json", mine, ["hybrid@bge-small"], scope="held_out")["all"] is True
    without = reproduces(tmp_path, "f.json", mine, ["hybrid@bge-small"])
    assert without["rows"]["hybrid@bge-small"] == {"in_committed_file": False, "equal": False} and without["all"] is False
    other = reproduces(tmp_path, "f.json", [{"config": "hybrid@bge-small", "retrieval": {**retrieval, "ndcg@10": 0.71}}], ["hybrid@bge-small"], scope="held_out")
    assert other["all"] is False and other["rows"]["hybrid@bge-small"]["differs"] == {"ndcg@10": [0.7, 0.71]}


def test_the_twin_assertion_fails_at_0901_and_the_splits_are_declared():
    def flag(c):
        return {"set": "scifact", "id": "q", "all": {"cosine": c, "jaccard": 0.1}, "beir": {"cosine": c, "jaccard": 0.1}}

    assert ae.twin_assertion([flag(0.8999), flag(0.9005)])["passed"] is True
    check = ae.twin_assertion([flag(0.8999), flag(0.901)])
    assert check["passed"] is False and check["failing"] == {"scifact": {"all": 0.901, "beir": 0.901}}
    splits = ae.twin_splits({"all": [0.79, 0.80, 0.85, 0.5], "beir": [0.9, 0.1, 0.1, 0.1]})
    assert splits["twin_cosine_0.80"]["all"] == [False, True, True, False] and splits["twin_cosine_0.85"]["all"] == [False, False, True, False]
    assert ae.subgroup_sizes([{**flag(0.81), "set": "fixture", "id": "a"}, {**flag(0.86), "set": "fixture", "id": "b"}],
                             {"a": "held_out", "b": "tune"})["fixture"]["all_held_out"] == {"n": 1, "0.80": 1, "0.85": 0}


def _scores(rng, names, n, metrics=("ndcg@10", "recall@10")):
    return {name: {m: [float(x) for x in rng.random(n)] for m in metrics} for name in names}


def test_a_subgroup_under_ten_queries_gets_no_interval_and_the_fixture_reading_follows_the_rule():
    rng = np.random.default_rng(0)
    names = ["hybrid@bge-small", f"hybrid@{ae.BAR_ARM}"]
    scores = _scores(rng, names, 30)
    for i in range(30):
        scores[names[1]]["ndcg@10"][i] = scores[names[0]]["ndcg@10"][i] + 0.2  # a clear gain everywhere
    flags = [i < 5 for i in range(30)]  # 5 twinned, 25 not
    blocks = ae.subgroup_blocks(scores, [(names[1], names[0])], list(range(30)), {"twin_cosine_0.85": {"all": flags}})
    twinned, clean = blocks
    assert twinned["n"] == 5 and "ndcg@10" not in twinned and "fewer than 10" in twinned["interval"]
    assert clean["n"] == 25 and clean["ndcg@10"]["lo"] > 0
    assert ae.fixture_reading({"subgroups": blocks})["reading"] == "adaptation beyond templated twins"
    small = ae.subgroup_blocks(scores, [(names[1], names[0])], list(range(30)), {"twin_cosine_0.85": {"all": [i < 22 for i in range(30)]}})
    assert ae.fixture_reading({"subgroups": small})["reading"] == "not separable from templated twins"  # 8 not twinned: no interval


def test_the_outcome_words():
    win, tie, loss = {"lo": 0.01, "hi": 0.1}, {"lo": 0.0, "hi": 0.1}, {"lo": -0.1, "hi": -0.01}
    assert ae.outcome([win, win, win]) == "passes"
    assert ae.outcome([win, tie, win]) == "does not pass"  # a stored 0.0 is not above zero
    assert ae.outcome([win, loss, None]) == "does not pass"  # a failing measured leg decides it, even with one missing
    assert ae.outcome([win, None, win]) == "open"
    assert ae.outcome([None, None, None]) == "open"
    assert ae.outcome([win, win, win], trained=False) == "not run"


def _fixture_blob(rng, names, n=24, scores=None):
    """A fixture results file and its sidecar made by the module's own code, as run_fixture makes them."""
    scores = scores or _scores(rng, names, n)
    held, tune = list(range(1, n, 2)), list(range(0, n, 2))
    twins = {s: {"cosine": [float(x) for x in rng.random(n)], "jaccard": [0.1] * n} for s in ae.TRAINING_SETS}
    pairs = ae.comparisons(names, "fixture")
    blob = {"git_sha": "abc1234", "timestamp": "t"}
    for scope, ids in zip(ae.SCOPES, (held, tune, list(range(n)))):
        block = ae._scope_blocks(scores, names, ids, pairs, ("ndcg@10", "recall@10"), {s: t["cosine"] for s, t in twins.items()})
        for row in block["results"]:
            row["retrieval"] = {m: round(sum(scores[row["config"]][m][i] for i in ids) / len(ids), 4) for m in ("ndcg@10", "recall@10")}
        if scope != "held_out":
            block.pop("subgroups")
        blob[scope] = block
    sidecar = {"schema": ae.SIDECAR_SCHEMA, "dataset": "fixture", "git_sha": "abc1234", "timestamp": "t", "metrics": ["ndcg@10", "recall@10"],
               "ids": [f"q{i}" for i in range(n)], "categories": ["c"] * n, "split": {"tune": tune, "held_out": held}, "scores": scores, "twins": twins}
    return blob, sidecar


FIXTURE_ROWS = ["bm25_only", "dense_only@bge-small", "hybrid@bge-small", f"dense_only@{ae.BAR_ARM}", f"hybrid@{ae.BAR_ARM}",
                f"dense_only@{ae.CONTROL_ARM}", f"hybrid@{ae.CONTROL_ARM}", "dense_only@bge-small-marginmse-s1", "hybrid@bge-small-marginmse-s1"]


def test_the_fixture_file_rescores_from_its_sidecar_with_no_model_or_dataset(monkeypatch):
    from app.eval import beir, gold, per_question

    monkeypatch.setattr(beir, "load_beir", lambda *a, **k: pytest.fail("the rescore read a dataset"))
    monkeypatch.setattr(gold, "load_gold", lambda *a, **k: pytest.fail("the rescore read the gold questions"))
    blob, sidecar = _fixture_blob(np.random.default_rng(1), FIXTURE_ROWS)
    assert ("hybrid@bge-small-marginmse-nofixture", "hybrid@bge-small-marginmse") in ae.comparisons(FIXTURE_ROWS, "fixture")
    assert blob["held_out"]["seed_mean"]["n_seeds"] == 2 and blob["held_out"]["subgroups"]
    assert ae.rescore_fixture(blob, sidecar) == []
    assert per_question.differences(blob, sidecar) == []  # routed by its schema name
    blob["held_out"]["subgroups"][0]["delta"] += 0.0001
    assert any("subgroups" in d for d in ae.rescore_fixture(blob, sidecar))


def _beir_blob(rng, n=40):
    from app.eval import beir

    names = ["bm25", "dense@bge-small", "hybrid@bge-small", f"dense@{ae.BAR_ARM}", f"hybrid@{ae.BAR_ARM}"]
    parents = {"dense@bge-small": "bm25", "hybrid@bge-small": "dense@bge-small", f"dense@{ae.BAR_ARM}": "dense@bge-small",
               f"hybrid@{ae.BAR_ARM}": "hybrid@bge-small"}
    metrics = list(beir.PER_QUERY_METRICS)
    ids = [f"q{i}" for i in range(n)]
    scores = _scores(rng, names, n, tuple(metrics))
    rows = [{"config": name, "parent": parents.get(name), "retrieval": {m: round(sum(scores[name][m]) / n, 4) for m in metrics},
             "_per_query": [dict({m: scores[name][m][i] for m in metrics}, id=ids[i]) for i in range(n)]} for name in names]
    pairs = {(r["config"], r["parent"]) for r in rows}
    comps = [p for p in ae.comparisons(names, "scifact") if p not in pairs]
    boot = beir.attach_confidence(rows, n, published=None, comparisons=comps)
    twins = {s: {"cosine": [float(x) for x in rng.random(n)], "jaccard": [0.0] * n} for s in ae.TRAINING_SETS}
    seen = {k: [bool(x > 0.5) for x in rng.random(n)] for k in ae.SEEN_SPLITS}
    blob = {"dataset": "BEIR/scifact", "split": "test", "git_sha": "abc1234", "timestamp": "t", "bootstrap": boot, "results": rows,
            "adapted_embedders": ae.beir_extras(scores, ids, boot, twins, seen, "scifact")}
    sidecar = {"schema": ae.SIDECAR_SCHEMA, "dataset": "BEIR/scifact", "split": "test", "git_sha": "abc1234", "timestamp": "t", "metrics": metrics,
               "ids": ids, "scores": scores, "twins": twins, "seen": seen}
    return blob, sidecar


def test_a_beir_file_rescores_from_its_sidecar_with_no_model_dataset_or_qrels(monkeypatch):
    from app.eval import beir, per_question

    monkeypatch.setattr(beir, "load_beir", lambda *a, **k: pytest.fail("the rescore read a dataset"))
    blob, sidecar = _beir_blob(np.random.default_rng(2))
    splits = {g["split"] for g in blob["adapted_embedders"]["subgroups"]}
    assert splits == {"twin_cosine_0.80", "twin_cosine_0.85", "relevant_is_training_source", "relevant_in_training_list"}
    assert ae.rescore_beir(blob, sidecar) == []
    assert per_question.differences(blob, sidecar) == []
    blob["results"][3]["delta_vs_parent"]["ndcg@10"]["lo"] += 0.0001
    assert any("delta_vs_parent" in d for d in ae.rescore_beir(blob, sidecar))


def _write_files(root, legs, *, trained=(ae.BAR_ARM,)):
    """Results files whose bar-row hybrid delta is each of `legs` (None: no file), and a manifest naming `trained`."""
    (root / "bench" / "distill").mkdir(parents=True, exist_ok=True)
    (root / "bench" / "results").mkdir(parents=True, exist_ok=True)
    (root / ae.MANIFEST_FILE).write_text(json.dumps({"arms": {t: {"weights_sha256": "0" * 64} for t in trained}}))
    for dataset, leg in zip(ae.CORPORA, legs):
        if leg is None:
            continue
        dense, hybrid = ae.row_names(ae.BAR_ARM, dataset)
        bdense, bhybrid = ae.row_names("bge-small", dataset)
        block = {"ndcg@10": {**leg, "delta": (leg["lo"] + leg["hi"]) / 2}}
        ci = {"ndcg@10": {"mean": 0.5, "lo": 0.4, "hi": 0.6}}
        if dataset == "fixture":
            rows = [{"config": bhybrid, "ci": ci}, {"config": hybrid, "ci": ci, "deltas": [{"reference": bhybrid, **block}]}]
            blob = {"git_sha": "abc", "held_out": {"bootstrap": {"n_queries": 68}, "results": rows, "subgroups": []}}
        else:
            rows = [{"config": bhybrid, "ci": ci}, {"config": hybrid, "ci": ci, "delta_vs_parent": {"reference": bhybrid, **block}}]
            blob = {"git_sha": "abc", "bootstrap": {"n_queries": 300}, "results": rows, "adapted_embedders": {}}
        (root / ae.RESULTS[dataset]).write_text(json.dumps(blob))


def test_bar_table_reads_the_outcome_off_the_files(tmp_path):
    win, tie, loss = {"lo": 0.01, "hi": 0.1}, {"lo": -0.01, "hi": 0.1}, {"lo": -0.2, "hi": -0.01}
    cases = [((win, win, win), (ae.BAR_ARM,), "passes", []), ((win, tie, win), (ae.BAR_ARM,), "does not pass", []),
             ((win, win, loss), (ae.BAR_ARM,), "does not pass", ["fiqa"]), ((win, None, win), (ae.BAR_ARM,), "open", []),
             ((None, None, None), (ae.CONTROL_ARM,), "not run", [])]
    for i, (legs, trained, word, losses) in enumerate(cases):
        root = tmp_path / str(i)
        _write_files(root, legs, trained=trained)
        table = ae.bar_table(root)
        assert (table["outcome"], table["losses"]) == (word, losses), (i, table["outcome"])
        assert table["status"][ae.BAR_ARM] == ("trained" if ae.BAR_ARM in trained else "not run")
    assert ae.seeds_line({"seeds": {"n": 1, "k": 1}}) == "one run (seed 0)"
    assert ae.seeds_line({"seeds": {"n": 3, "k": 2}}) == "the bar's form holds for 2 of 3 seeds"


# ---------------------------------------------------------------- the generated docs (README block, evidence rows)


def _docs_root(root, gain=0.0, n=40):
    """A repository root with a fixture file whose bar-row hybrid gains `gain` on every question over hybrid@bge-small,
    a SciFact file (random scores), no FiQA file, and a manifest holding the bar row, the control and seed 1."""
    from app.eval import report  # noqa: F401  (the renderer's module imports cleanly)

    rng = np.random.default_rng(3)
    scores = _scores(rng, FIXTURE_ROWS, n)
    base = scores["hybrid@bge-small"]["ndcg@10"]
    scores[f"hybrid@{ae.BAR_ARM}"]["ndcg@10"] = [min(1.0, b + gain) if gain else x for b, x in zip(base, scores[f"hybrid@{ae.BAR_ARM}"]["ndcg@10"])]
    fixture, _ = _fixture_blob(rng, FIXTURE_ROWS, n, scores=scores)
    fixture.update(device={"device_name": "card"}, reproduces=[])
    beir_blob, _ = _beir_blob(np.random.default_rng(2))
    beir_blob.update(device={"device_name": "card"})
    (root / "bench" / "results").mkdir(parents=True, exist_ok=True)
    (root / "bench" / "distill").mkdir(parents=True, exist_ok=True)
    (root / ae.RESULTS["fixture"]).write_text(json.dumps(fixture))
    (root / ae.RESULTS["scifact"]).write_text(json.dumps(beir_blob))
    entry = {"attempt": 1, "s": 55.9, "lists": {"train": {"fixture": 691, "scifact": 4267, "fiqa": 5129}}, "steps_per_epoch": 1261,
             "epochs": [{"validation_macro": v} for v in (8.1, 7.5, 7.4, 7.39)], "segments": [{"device": "card"}],
             "validation_before_training": {"macro": 11.4}, "chosen_epoch": 4, "chosen_is_last": True, "beats_epoch_0": True, "weights_sha256": "ab" * 32}
    (root / ae.MANIFEST_FILE).write_text(json.dumps({"arms": {t: entry for t in (ae.BAR_ARM, ae.CONTROL_ARM, "bge-small-marginmse-s1")}}))
    return fixture, beir_blob


def test_interval_count_counts_every_paired_interval_of_a_file(tmp_path):
    fixture, beir_blob = _docs_root(tmp_path)
    by_hand = sum(1 for s in ae.SCOPES for r in fixture[s]["results"] for d in r.get("deltas") or [])
    by_hand += sum(1 for g in fixture["held_out"]["subgroups"] if "ndcg@10" in g) + sum(1 for s in ae.SCOPES if fixture[s].get("seed_mean"))
    assert ae.interval_count(fixture) == by_hand > 0
    ext = beir_blob["adapted_embedders"]
    by_hand = sum(1 for r in beir_blob["results"] for k in ("delta_vs_parent", "delta_vs_bm25") if r.get(k)) + len(beir_blob["bootstrap"]["comparisons"])
    by_hand += sum(1 for g in ext["subgroups"] if "ndcg@10" in g) + (1 if ext.get("seed_mean") else 0)
    assert ae.interval_count(beir_blob) == by_hand > 0


def test_the_readme_block_keeps_every_arm_and_reads_the_bar_off_the_files(tmp_path):
    from app.eval.report import adapted_embedders_table

    assert "is missing" in adapted_embedders_table(tmp_path)
    _docs_root(tmp_path)
    table = ae.bar_table(tmp_path)
    text = adapted_embedders_table(tmp_path)
    assert f"**Outcome: {table['outcome']}.**" in text and table["outcome"] in ("does not pass", "open")  # FiQA is not measured: never passes
    for tag in ae.BGE_SMALL_ARMS:  # losers and arms not run stay, in both tables
        rows = [line for line in text.splitlines() if line.startswith(f"| `{tag}` (")]
        assert len(rows) == 2, tag
        if table["status"][tag] != "trained":
            assert all("not run" in r for r in rows)
    hybrid_bar = next(line for line in text.splitlines() if line.startswith(f"| `{ae.BAR_ARM}` ("))
    assert "exploratory" not in hybrid_bar  # the bar row's hybrid legs are the one declared test
    others = [line for line in text.splitlines() if line.startswith("| `bge-small-marginmse-") and "not run" not in line]
    assert others and all("exploratory" in line for line in others)
    n = int(text.split("This block shows ")[1].split(" exploratory intervals")[0])
    assert n >= text.count("exploratory)") > 0
    assert "**What this does not show.**" in text and "Fixture reading:" in text and "The verdict is read on seed 0 alone" in text


def test_evidence_rows_only_where_an_interval_excludes_zero_and_say_what_they_do_not_show(tmp_path):
    from app.eval.evidence import _adapted_embedder_claims

    assert _adapted_embedder_claims(tmp_path) == []
    fixture, _ = _docs_root(tmp_path, gain=0.3)
    table = ae.bar_table(tmp_path)
    claims = _adapted_embedder_claims(tmp_path)
    bar = [c for c in claims if f"`hybrid@{ae.BAR_ARM}`" in c["claim"] and c["claim"].startswith("Fixture")]
    assert len(bar) == 1 and bar[0]["value"].startswith("+")
    n = ae.interval_count(fixture)
    for c in claims:
        lo, hi = (float(x) for x in c["ci"].strip("[]").split(","))
        assert lo > 0 or hi < 0, c["claim"]
        assert f"the bar ({ae.BAR_ARM}, hybrid, three datasets): {table['outcome']}" in c["not_shown"]
        assert "intervals in this file, unadjusted" in c["not_shown"] and "the checkpoints are not committed" in c["not_shown"]
        assert ("secondary read; the served shape is hybrid" in c["not_shown"]) is ("`dense" in c["claim"])
        if c["claim"].startswith("Fixture"):
            assert f"; {n} intervals in this file" in c["not_shown"] and "fixture reading" in c["not_shown"]
        if ae.CONTROL_ARM in c["claim"] and c["claim"].startswith("Fixture"):
            assert "never read this corpus" in c["not_shown"]
        if ae.BAR_ARM + "`" not in c["claim"] and not (ae.CONTROL_ARM in c["claim"] and c["claim"].startswith("Fixture")):
            assert hi < 0, c["claim"]  # any other arm is a row only as a recorded loss

class _Guard:
    """Leakage risk 4: any open of a gold file, a BEIR qrels directory or a queries.jsonl raises."""

    def __init__(self, monkeypatch):
        self.opened = []
        for owner, name in ((builtins, "open"), (pathlib.Path, "open"), (gzip, "open")):
            monkeypatch.setattr(owner, name, self._wrap(getattr(owner, name)))
        from app.eval import beir, distill_queries, gold

        for owner, name in ((beir, "load_beir"), (gold, "load_gold"), (distill_queries, "benchmark_queries")):
            monkeypatch.setattr(owner, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} was called"))

    def _wrap(self, real):
        def guarded(file, *args, **kwargs):
            p = pathlib.PurePath(str(file))
            if "gold" in p.parts and "bench" in p.parts or "qrels" in p.parts or p.name == "queries.jsonl":
                raise AssertionError(f"test data opened: {file}")
            self.opened.append(str(file))
            return real(file, *args, **kwargs)

        return guarded


def _toy_data(tmp_path):
    """A toy BEIR corpus whose queries.jsonl and qrels/ sit beside it (and must not be opened), and toy training files."""
    from app.eval.distill_queries import write_jsonl_gz

    data = tmp_path / "data" / "scifact"
    (data / "qrels").mkdir(parents=True)
    docs = {f"d{i}": f"document {i} about topic {i % 3} with words {' '.join(['w%d' % j for j in range(i)])}" for i in range(8)}
    (data / "corpus.jsonl").write_text("".join(json.dumps({"_id": k, "title": "", "text": v}) + "\n" for k, v in docs.items()))
    (data / "queries.jsonl").write_text(json.dumps({"_id": "t1", "text": "a test query"}) + "\n")
    (data / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nt1\td1\t1\n")
    work = tmp_path / "distill"
    queries, scores = [], []
    for i in range(6):
        cands = [f"d{(i + j) % 8}" for j in range(4)]
        queries.append({"id": f"scifact:d{i}:q1", "corpus": "scifact", "doc_id": f"d{i}", "form": "questions", "query": f"what is topic {i}",
                        "split": "validation" if i == 5 else "train"})
        scores.append({"id": f"scifact:d{i}:q1", "corpus": "scifact", "candidates": cands, "source_rank": 1, "scores": [3.0, 1.0, 0.5, -1.0]})
    qsha = write_jsonl_gz(work / "queries.jsonl.gz", queries)
    ssha = write_jsonl_gz(work / "teacher_scores.jsonl.gz", scores)
    (work / "queries.meta.json").write_text(json.dumps({"files": {"queries.jsonl.gz": {"sha256": qsha}}}))
    (work / "teacher_scores.meta.json").write_text(json.dumps({"file": {"name": "teacher_scores.jsonl.gz", "sha256": ssha}}))
    return tmp_path / "data", work


def test_the_data_path_reads_no_test_data_and_checks_the_files(tmp_path, monkeypatch):
    data, work = _toy_data(tmp_path)
    guard = _Guard(monkeypatch)
    shas = et.verify_data(work / "queries.jsonl.gz", work / "teacher_scores.jsonl.gz")
    _, lists = et.load_lists(ROOT, data, work / "queries.jsonl.gz", work / "teacher_scores.jsonl.gz")
    mine = et.training_set_lists(lists, "beir")
    assert len(mine["train"]["scifact"]) == 5 and len(mine["validation"]["scifact"]) == 1 and set(shas) == {"queries.jsonl.gz", "teacher_scores.jsonl.gz"}
    rows = mine["train"]["scifact"]
    fit = et.scale_from_lists([r["scores"] for r in rows], [[0.9, 0.8, 0.7, 0.5] for _ in rows])
    assert fit["s"] > 0 and any(o.endswith("corpus.jsonl") for o in guard.opened)
    (work / "queries.jsonl.gz").write_bytes(gzip.compress(b"{}\n"))
    with pytest.raises(SystemExit, match="not M43's committed file"):
        et.verify_data(work / "queries.jsonl.gz", work / "teacher_scores.jsonl.gz")


def test_a_manifest_refuses_a_smoke_run_and_changed_weights(tmp_path):
    d = tmp_path / ae.BAR_ARM
    d.mkdir()
    (d / et.WEIGHTS_FILE).write_bytes(b"w")
    record = {"smoke": None, "working_tree_dirty": False, "files": {et.WEIGHTS_FILE: {"sha256": hashlib.sha256(b"w").hexdigest(), "bytes": 1}},
              "sources_sha256": {"declarations": "x"}, "epochs": [], "chosen_epoch": 1}
    (d / et.TRAINING_FILE).write_text(json.dumps(record))
    assert list(et.build_manifest(tmp_path)["arms"]) == [ae.BAR_ARM]
    (d / et.TRAINING_FILE).write_text(json.dumps({**record, "smoke": {"max_steps": 20}}))
    with pytest.raises(SystemExit, match="smoke run"):
        et.build_manifest(tmp_path)
    (d / et.TRAINING_FILE).write_text(json.dumps(record))
    (d / et.WEIGHTS_FILE).write_bytes(b"x")
    with pytest.raises(SystemExit, match="its training recorded"):
        et.build_manifest(tmp_path)


def test_a_declarations_change_changes_their_sha256(monkeypatch):
    before = ae.declarations_sha256()
    monkeypatch.setitem(ae.RECIPE, "learning_rate", 3e-5)
    assert ae.declarations_sha256() != before


# ---------------------------------------------------------------- torch (the GPU environment; skipped on the login node)


def test_the_hand_written_mnrl_equals_the_librarys_and_masks_a_collision():
    torch = _torch()
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    g = torch.Generator().manual_seed(0)
    b, k, dim = 4, 7, 16
    q, cand = torch.randn(b, dim, generator=g), torch.randn(b * (1 + k), dim, generator=g)
    ids = [f"p{i}" for i in range(b)] + [f"n{j}{i}" for j in range(k) for i in range(b)]
    mine, masked = et.mnrl_loss(q, cand, ids, ids[:b])
    lib = MultipleNegativesRankingLoss(None)  # defaults: scale 20, cos_sim, query to document, joint
    theirs = lib.compute_loss_from_embeddings([q, *[cand[i * b:(i + 1) * b] for i in range(1 + k)]], torch.arange(b))
    assert masked == 0 and float(mine) == pytest.approx(float(theirs), abs=1e-6)
    ids_collide = list(ids)
    ids_collide[b + 2] = "p0"  # a hard negative of anchor 2 is anchor 0's positive document
    loss, masked = et.mnrl_loss(q, cand, ids_collide, ids[:b])
    sims = 20 * torch.nn.functional.normalize(q, dim=1) @ torch.nn.functional.normalize(cand, dim=1).T
    sims[0, b + 2] = float("-inf")
    assert masked == 1 and float(loss) == pytest.approx(float(torch.nn.functional.cross_entropy(sims, torch.arange(b))), abs=1e-6)


def test_the_lr_schedule_continues_exactly_after_a_resume():
    torch = _torch()
    from app.eval.distill_train import lr_factor

    def make():
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.AdamW([p], lr=2e-5)
        return p, opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, 40, 0.1))

    p, opt, sched = make()
    lrs = []
    for step in range(40):
        lrs.append(sched.get_last_lr()[0])
        p.grad = torch.ones(2)
        opt.step()
        sched.step()
        if step == 12:  # mid-epoch
            saved = (opt.state_dict(), sched.state_dict())
    p2, opt2, sched2 = make()
    opt2.load_state_dict(saved[0])
    sched2.load_state_dict(saved[1])
    again = []
    for _ in range(13, 40):
        again.append(sched2.get_last_lr()[0])
        p2.grad = torch.ones(2)
        opt2.step()
        sched2.step()
    assert again == lrs[13:]


def _bge_small_or_skip():
    _torch()
    from app.eval.provenance import hf_revision

    if hf_revision(ae.INITS["bge-small"]["model"]) != ae.INITS["bge-small"]["revision"]:
        pytest.skip("bge-small is not cached at its pinned revision")
    return et.load_init("bge-small", device="cpu")


def test_the_training_forward_equals_encode_on_long_texts():
    model = _bge_small_or_skip()
    texts = ["Short query?", "word " * 900, "A medium passage about returns on bonds and their yields. " * 12]
    tokens = et.Tokens(model, texts)
    assert tokens.length(texts[1]) == 512  # cut at the same token as encode()
    feats, direct = tokens.features(texts), model.preprocess(texts)
    assert all(bool((feats[k] == direct[k]).all()) for k in ("input_ids", "attention_mask", "token_type_ids"))
    import torch

    model.eval()
    with torch.no_grad():
        mine = et.embed(model, tokens, texts, block=2).numpy()
    ref = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    assert float(np.max(np.abs(mine - ref))) < 1e-5


def test_one_training_step_reads_no_test_data_and_its_loss_is_marginmse(tmp_path, monkeypatch):
    _bge_small_or_skip()
    data, work = _toy_data(tmp_path)
    _Guard(monkeypatch)
    _, lists = et.load_lists(ROOT, data, work / "queries.jsonl.gz", work / "teacher_scores.jsonl.gz")
    lists = et.training_set_lists(lists, "beir")
    scales = et.compute_scales(ROOT, {**lists, "train": {**lists["train"], "fixture": []}}, device="cpu", keys=[("bge-small", "beir")], log=lambda *_: None)
    s = scales["bge-small/beir"]["s"]
    monkeypatch.setitem(et.ARMS, "toy", {**ae.ARMS[ae.CONTROL_ARM]})
    model, record = et.train("toy", lists, s, device="cpu", max_steps=1, max_epochs=1, time_limit_min=None, log=lambda *_: None)
    assert record["first_step_loss"]["torch"] == pytest.approx(record["first_step_loss"]["numpy"], rel=1e-4)
    assert record["smoke"] == {"max_steps": 1, "max_epochs": 1} and record["steps"] == 1


def test_a_stopped_run_resumes_to_the_same_schedule(tmp_path, monkeypatch):
    _bge_small_or_skip()
    data, work = _toy_data(tmp_path)
    _, lists = et.load_lists(ROOT, data, work / "queries.jsonl.gz", work / "teacher_scores.jsonl.gz")
    lists = et.training_set_lists(lists, "beir")
    monkeypatch.setitem(et.ARMS, "toy", {**ae.ARMS[ae.CONTROL_ARM]})
    kw = dict(device="cpu", max_steps=1, max_epochs=2, time_limit_min=None, log=lambda *_: None)
    _, full = et.train("toy", lists, 5.0, resume_path=tmp_path / "a" / "resume.pt", **kw)
    stopped, none = et.train("toy", lists, 5.0, resume_path=tmp_path / "b" / "resume.pt", stop_at_step=1, **kw)
    assert stopped is None and none is None and (tmp_path / "b" / "resume.pt").exists()
    _, resumed = et.train("toy", lists, 5.0, resume_path=tmp_path / "b" / "resume.pt", **kw)
    assert resumed["schedule"]["lr"] == full["schedule"]["lr"] and resumed["schedule"]["draws"] == full["schedule"]["draws"]
    assert resumed["steps"] == full["steps"] == 2 and [g["stopped"] for g in resumed["segments"]][-1] is None
    assert math.isclose(resumed["epochs"][-1]["validation_macro"], full["epochs"][-1]["validation_macro"], rel_tol=1e-3)


def test_a_saved_checkpoint_reloads_with_the_declared_encoding(tmp_path):
    model = _bge_small_or_skip()
    files = et.save_checkpoint(model, tmp_path / "ckpt")
    assert json.loads((tmp_path / "ckpt" / "sentence_bert_config.json").read_text())["do_lower_case"] is True
    assert set(files) >= {et.WEIGHTS_FILE, "sentence_bert_config.json", "modules.json"} and et.path_leaks(tmp_path / "ckpt") == []
    emb = ae.load_checkpoint_embedder(tmp_path / "ckpt", ae.BAR_ARM, device="cpu")  # raises unless it encodes as ENCODING declares
    texts = ["MiXeD Case Text about EBITDA", "plain text"]
    assert emb.name == "ft:bge-small-marginmse" and float(np.max(np.abs(emb.encode(texts) - model.encode(texts, normalize_embeddings=True)))) < 1e-6
