"""M44: the corpus builder takes a Spec, and with the default Spec it still writes fixture v1 byte for byte.

The search policy trains on corpora the builder writes from other Specs (app/rl/corpora.py); fixture v1 is its test
set and is frozen. Everything here builds into a temporary directory: the committed fixture is only read.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from app.eval.corpus import load_manifest, mtime_mismatches, verify_manifest
from app.eval.gold import load_gold
from app.eval.index_loader import load_chunks
from app.rl.corpora import (
    BARK_POOL,
    ENCODER_POOL,
    normalise_question,
    parse_seeds,
    spec_problems,
    training_set,
    variant_root,
    variant_spec,
    write_variants,
)
from bench.corpus import build as corpus_build
from bench.corpus.build import V1, build_gold
from bench.corpus.registry import ABLATIONS, ARCHIVED, BASELINE_RMSE, RUNS

ROOT = Path(__file__).resolve().parents[2]
GOLD_SHA256 = "06c2421220fa2569c87cfa5fc57902476da06ac28f9ea41a5320623d64e984de"  # tests/eval/test_gold_digest.py
SEEDS = list(range(1, 13))


def test_the_default_spec_rebuilds_fixture_v1_byte_for_byte(tmp_path):
    summary = corpus_build.main(tmp_path)
    committed = load_manifest(ROOT / "bench" / "corpus" / "MANIFEST.json")
    assert summary == {"n_files": committed["n_files"], "n_gold": 136, "sha256": committed["aggregate_sha256"]}
    files = tmp_path / "bench" / "corpus" / "files"
    assert verify_manifest(files, committed) == []  # every file, its size and its digest, and no other file
    assert mtime_mismatches(files, committed) == []
    for rel in ("bench/corpus/MANIFEST.json", "bench/corpus/SNAPSHOT.md", "bench/gold/questions.jsonl"):
        assert (tmp_path / rel).read_bytes() == (ROOT / rel).read_bytes(), rel
    for entry in committed["files"]:  # the bytes, not only the text digest the manifest holds
        assert (files / entry["path"]).read_bytes() == (ROOT / "bench" / "corpus" / "files" / entry["path"]).read_bytes(), entry["path"]
    assert hashlib.sha256((tmp_path / "bench" / "gold" / "questions.jsonl").read_bytes()).hexdigest() == GOLD_SHA256


def test_the_default_spec_is_the_registry_and_calling_the_builder_without_one_is_fixture_v1():
    assert (list(V1.runs), list(V1.archived), list(V1.ablations), V1.baseline_rmse) == (RUNS, ARCHIVED, ABLATIONS, BASELINE_RMSE)
    assert spec_problems(V1) == []
    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    assert [q.to_dict() for q in build_gold()] == [q.to_dict() for q in gold]
    meta: list[dict] = []
    assert [q.to_dict() for q in build_gold(V1, meta)] == [q.to_dict() for q in gold]  # asking for the metadata changes no question
    assert [m["id"] for m in meta] == [q.id for q in gold] and len({m["template"] for m in meta}) == 5 + 56
    assert all(m["template"].split(".")[0] in (q.category, "simple") for m, q in zip(meta, gold))


def test_only_fixture_v1_can_be_written_into_the_repository(tmp_path):
    with pytest.raises(ValueError, match="fixture v1 only"):
        corpus_build.main(ROOT, variant_spec(1))
    with pytest.raises(ValueError, match="outside the repository"):
        write_variants([1], ROOT / "bench" / "variants")
    with pytest.raises(ValueError, match="start at 1"):
        variant_spec(0)
    assert parse_seeds("1-3,7,9-10") == [1, 2, 3, 7, 9, 10]


def test_a_variant_is_a_pure_function_of_its_seed_and_shares_nothing_named_with_fixture_v1():
    v1_ids = {r.run_id for r in RUNS}
    v1_encoders, v1_barks = {r.encoder for r in RUNS}, {r.bark for r in RUNS}
    assert not v1_encoders & {e.key for e in ENCODER_POOL} and not v1_barks & set(BARK_POOL)
    texts = set()
    for seed in SEEDS:
        spec = variant_spec(seed)
        assert spec == variant_spec(seed) and spec_problems(spec) == []
        assert len(spec.runs) == 16 and not {r.run_id for r in spec.runs} & v1_ids and min(r.run_id for r in spec.runs) >= 60
        assert not {r.encoder for r in spec.runs} & v1_encoders and not {r.bark for r in spec.runs} & v1_barks
        assert spec.baseline_rmse != BASELINE_RMSE
        gold = build_gold(spec)
        assert len(gold) == 136 and all(q.answer.strip() and q.relevant_paths for q in gold)
        texts.add(json.dumps([q.to_dict() for q in gold]))
    assert len(texts) == len(SEEDS)  # twelve seeds, twelve different question sets


def test_a_variant_corpus_is_written_outside_the_repository_and_answers_its_own_questions(tmp_path):
    rows = write_variants([3, 4], tmp_path)
    assert [r["seed"] for r in rows] == [3, 4] and rows[0]["corpus_sha256"] != rows[1]["corpus_sha256"]
    fixture = load_manifest(ROOT / "bench" / "corpus" / "MANIFEST.json")
    for row in rows:
        root = variant_root(tmp_path, row["seed"])
        manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
        files = root / "bench" / "corpus" / "files"
        assert verify_manifest(files, manifest) == [] and manifest["aggregate_sha256"] == row["corpus_sha256"] != fixture["aggregate_sha256"]
        assert manifest["n_files"] == fixture["n_files"] == row["n_files"]
        gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
        assert [q.to_dict() for q in gold] == [q.to_dict() for q in build_gold(variant_spec(row["seed"]))]
        on_disk = {e["path"] for e in manifest["files"]}
        assert all(set(q.relevant_paths) <= on_disk for q in gold)  # every relevant path is a file of this corpus
        text = {e["path"]: (files / e["path"]).read_text() for e in manifest["files"]}
        spec = variant_spec(row["seed"])
        for run in spec.runs:  # the values the per-run questions ask for are in that run's config
            config = text[run.config_path]
            assert f"learning_rate: {run.lr}\n" in config and f"bark_type: {run.bark}\n" in config and f"val_rmse: {run.val_rmse}\n" in config
        assert f"src/{spec.encoders[0].key}_encoder.py" in on_disk and spec.encoders[0].model_name in text[f"src/{spec.encoders[0].key}_encoder.py"]
        chunks = load_chunks(files, mtimes="manifest")  # the benchmark's loader reads it as a fixture, with its MANIFEST's mtimes
        assert {c.path for c in chunks} == on_disk
    assert not list((ROOT / "bench").glob("variant*")) and not (ROOT / "bench" / "corpus" / "files" / "configs" / "run_060.yaml").exists()


def test_training_set_drops_fixture_question_texts_and_constant_answers_and_reports_both():
    fixture = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    fixture_text = {normalise_question(q.question) for q in fixture}
    kept, report = training_set(SEEDS)
    assert report["n_questions_built"] == 136 * len(SEEDS)
    assert report["n_kept"] == len(kept) == report["n_questions_built"] - report["n_dropped_same_text_as_a_fixture_question"] - report["n_dropped_constant_answer"]
    assert report["n_dropped_same_text_as_a_fixture_question"] > 0 and report["n_dropped_constant_answer"] > 0
    assert not [q for q in kept if normalise_question(q.question) in fixture_text] and report["n_kept_with_a_fixture_question_text"] == 0
    assert report["variant_run_ids_also_in_fixture_v1"] == []
    # rule (b): what was kept has no constant answer; the fusion sentence and "baseline val RMSE" never reach training
    constant_answers = {"baseline val RMSE", "slurm/", "src/trellis_wires.py", "concatenates per-view features and projects with a 1x1 when fusion is on"}
    assert not [q for q in kept if q.answer in constant_answers]
    assert set(report["dropped_constant_answer_by_template"]) == {"multi_hop.fusion_module_in_headline_run", "semantic.baseline_number",
                                                                  "exact_path.flag_in_slurm_script"}
    # an answer kept in training that is also a fixture answer is a value read from a corpus, never a phrase
    assert report["kept_answers_equal_to_a_fixture_answer"]["longest_in_words"] <= 2
    assert all(len(a) <= 14 for a in report["kept_answers_equal_to_a_fixture_answer"]["answers"])
    assert len({q.key for q in kept}) == len(kept) and {q.corpus for q in kept} == set(SEEDS)
    assert set(report["kept_by_category"]) == {"simple_factual", "conditional", "comparative", "aggregation", "multi_hop", "staleness", "exact_path", "semantic"}
    # the fixture questions no training question shares a template with are named
    untrained = set(report["fixture_questions_whose_template_is_not_trained"])
    assert {"q110", "q131", "q134", "q136"} <= untrained and "q002" not in untrained
    assert training_set(SEEDS)[1] == report  # a pure function of the seeds


def test_the_constant_answer_flag_is_true_exactly_for_answers_that_never_change():
    by_template: dict[str, set[str]] = defaultdict(set)
    flagged: dict[str, bool] = {}
    for spec in [V1] + [variant_spec(seed) for seed in SEEDS]:
        meta: list[dict] = []
        for q, m in zip(build_gold(spec, meta), meta):
            if not m["template"].startswith("simple."):
                by_template[m["template"]].add(q.answer)
            flagged[m["template"]] = m["constant_answer"]
    constant = {t for t, answers in by_template.items() if len(answers) == 1}
    assert constant == {t for t in by_template if flagged[t]}
    assert not any(flagged[t] for t in flagged if t.startswith("simple."))
