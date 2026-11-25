"""M49: the late-interaction rows end to end, with no torch: every `mv:` branch of the BEIR engine, the fixture runner,
the per-query sidecars and the rescore from them, the verdict entries and the plan flags, the committed runs and the
CPU re-derivation, the tripwire and the bar.

The multi-vector model is a stand-in (StubEncoder: one hashed vector per word, the declared settings) and the
cross-encoder references score by token overlap; the bge-small parents use the hash embedder. What is tested is the
wiring the GPU jobs run, so that a crash or a mis-paired row is found here and not in a scoring job.
"""

import copy
import hashlib
import json
import math
import re
import shutil
import string
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.eval import beir, per_question
from app.eval import late_interaction as li
from app.eval.beir import BGE_SMALL
from app.eval.harness import BenchConfig, run_config
from app.eval.gold import load_gold
from app.eval.report import verdict
from app.multimodal.late_interaction import maxsim
from app.retrieval.bm25 import tokenize

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "bench" / "corpus" / "files"


class StubEncoder:
    """A stand-in MultiVectorTextEncoder: one seeded unit vector per lower-cased word (the same word, the same vector),
    the markers and [CLS]/[SEP] as tokens, punctuation dropped on the document side, queries padded with [MASK] to the
    expansion length, documents cut at the document length. Its settings are the declared ones."""

    def __init__(self, tag: str, document_length: int):
        self.tag, self.document_length = tag, document_length
        saved = li.MODELS[tag]["saved"]
        self.dim, self.q_len, self.expand, self.markers = saved["dim"], saved["query_length"], saved["query_expansion"], saved["markers"]

    def _vec(self, word: str) -> np.ndarray:
        v = np.random.RandomState(int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)).standard_normal(self.dim)
        return (v / np.linalg.norm(v)).astype(np.float32)

    @staticmethod
    def _words(text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"\w+|[^\w\s]", text)]

    def encode_queries(self, texts):
        out = []
        for t in texts:
            toks = (["[cls]", self.markers[0].strip()] + self._words(t))[: self.q_len - 1] + ["[sep]"]
            if self.expand:
                toks += ["[mask]"] * (self.q_len - len(toks))
            out.append(np.stack([self._vec(w) for w in toks]))
        return out

    def encode_documents(self, texts):
        out = []
        for t in texts:
            toks = (["[cls]", self.markers[1].strip()] + self._words(t))[: self.document_length - 1] + ["[sep]"]
            out.append(np.stack([self._vec(w) for w in toks if w not in string.punctuation]))
        return out

    def token_counts(self, texts, side):
        return [len(self._words(t)) + 3 for t in texts]

    def similarity(self, queries, documents):
        return np.array([[maxsim(q, d) for d in documents] for q in queries])

    def settings(self):
        return {"dim": self.dim, "query_length": self.q_len, "query_expansion": self.expand, "expansion_attended": False,
                "document_length": self.document_length, "skiplist_words": list(string.punctuation), "skiplist_ids": [1, 2],
                # as the library returns them: the marker as prompt text with a trailing space
                "markers": {side: {"prompt": m if m.endswith(" ") else m + " ", "single_piece": True} for side, m in zip(("query", "document"), self.markers)},
                "text_max_length_override": None}

    def provenance(self):
        return {"model": li.MODELS[self.tag]["model"], "revision": li.MODELS[self.tag]["revision"], "dtype": "stub", "device": "cpu",
                "settings": self.settings()}


class Overlap:
    """A stand-in CrossEncoder (as M43's tests): the share of query tokens in the passage."""

    default_prompt_name, prompts, max_seq_length = None, {}, 512

    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        out = []
        for q, d in pairs:
            qt, dt = set(tokenize(q)), set(tokenize(d))
            x = len(qt & dt) / max(len(qt), 1) - 0.5
            out.append(activation_fn(x) if activation_fn else 1 / (1 + math.exp(-x)))
        return out

    def parameters(self):
        yield type("P", (), {"dtype": "torch.float32"})()


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(li, "load_encoder", lambda tag, L, device=None: StubEncoder(tag, L))
    monkeypatch.setattr(li, "check_fingerprint", lambda encoder, tag: None)  # the card's scores need the real model (J1)
    monkeypatch.setattr(beir, "get_cross_encoder", lambda name, **kw: Overlap())


def _write_dataset(root: Path, name: str = "scifact") -> None:
    d = root / name
    (d / "qrels").mkdir(parents=True)
    docs = {f"d{i}": (f"Topic {i}", f"protein number {i} folds in cells, tissue sample {i} and enzyme {i % 5}") for i in range(30)}
    docs["q3"] = ("Clash", "a document whose id is a query id: protein number 3 in tissue sample 3")  # BEIR's ignore-identical-ids
    with (d / "corpus.jsonl").open("w") as fh:
        for did, (title, text) in docs.items():
            fh.write(json.dumps({"_id": did, "title": title, "text": text}) + "\n")
    with (d / "queries.jsonl").open("w") as fh:
        for i in range(12):
            fh.write(json.dumps({"_id": f"q{i}", "text": f"protein number {i} in tissue sample {i}"}) + "\n")
    (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"q{i}\td{i}\t1\n" for i in range(12)))


def _beir_part(tmp_path: Path, part: str, dataset: str = "scifact"):
    configs = [replace(c, embedder="hash") if c.embedder == BGE_SMALL else c for c in li.beir_configs(part)]
    kwargs = li.beir_run_kwargs(dataset, part, configs, {}, root=ROOT, reproduce=False)
    kwargs["resolve_reranker"] = lambda name: (name, {}, None)
    blob = beir.run(tmp_path, configs, device="cpu", n_boot=200, download=False, dataset=dataset, root=ROOT, **kwargs)
    captures, per_query = blob.pop("_captures", None), blob.pop("_per_query")
    li.finish_beir(blob, per_query, captures, tmp_path / li.RUNS_DIR, dataset)
    return blob, beir.per_query_sidecar(blob, per_query), captures


def test_both_beir_parts_run_every_mv_branch_and_rescore_from_their_sidecars(tmp_path, stubs):
    _write_dataset(tmp_path)
    blob, sidecar, captures = _beir_part(tmp_path, "primary")
    rows = {r["config"]: r for r in blob["results"]}
    assert list(rows) == [r["name"] for r in li.ROWS["beir"]["primary"]]
    assert rows["dense@bge-small"]["delta_vs_parent"] is None and rows[li.PARENT]["delta_vs_parent"] is None  # no bm25 row, no parent
    # the interval count the multiplicity statement declares: 4 bar + 11 others in a primary part
    kinds = [v["kind"] for v in blob["verdicts"]]
    assert len(kinds) == 15 == blob["late_interaction"]["multiplicity"]["this_file"]["ndcg@10_intervals"] and kinds.count("bar") == 4
    assert sorted(tuple(x) for x in ([v["config"], v["reference"]] for v in blob["verdicts"] if v["plan"])) == sorted(
        tuple(f) for f in li.PLAN_FLAGS if f[0].startswith("scifact:"))
    for v in blob["verdicts"]:
        assert v["config"].startswith("scifact:") and v["reference"].startswith("scifact:") and v["scope"] == "test"
        assert v["interval"] == verdict(v)  # the stored word, read off the stored lo and hi
        assert ("bonferroni" in v) == (v["kind"] == "bar")
        if v["kind"] == "bar":
            assert v["bonferroni"]["lo"] <= v["lo"] and v["bonferroni"]["hi"] >= v["hi"]  # wider: 98.75% from the same draws
    # the MaxSim leg's model record: settings asserted, store counts, the batched pass, the equivalence check
    leg = rows["dense@answerai-colbert-small"]["models"]["embedder"]
    assert leg["mv_name"] == "mv:answerai-colbert-small" and leg["settings_read_back"]["document_length"] == 300
    assert leg["store"]["documents"] == 31 and leg["store"]["bytes"]["bge_small_fp32"] == 31 * 384 * 4
    assert leg["batched_pass"]["queries"] == 12 and leg["equivalence"]["within_tolerance"] and leg["load_and_score"]["passed"]
    assert "latency_note" in rows["dense@answerai-colbert-small"]
    reranked = rows["hybrid+colbertv2@bge-small/fp32-doc180/top20"]["models"]["reranker"]
    assert reranked["role"] == "b" and reranked["stored_vs_on_the_fly"]["queries_checked"] == 12
    # the committed runs: pre-filter (the document whose id is q3 is still in q3's MaxSim list), deterministic gzip
    assert set(blob["late_interaction"]["runs"]) - {"_rule"} == {f"{li.RUNS_DIR}/scifact_answerai-colbert-small.run.gz", f"{li.RUNS_DIR}/scifact_colbertv2.run.gz",
                                                                f"{li.RUNS_DIR}/scifact_rerank_answerai-colbert-small.json.gz", f"{li.RUNS_DIR}/scifact_rerank_colbertv2.json.gz"}
    run = {x["qid"]: x for x in li.read_run(tmp_path / li.RUNS_DIR / "scifact_colbertv2.run.gz")}
    assert "q3" in run["q3"]["ids"] and len(run["q3"]["ids"]) == 31 and run["q3"]["scores"] == sorted(run["q3"]["scores"], reverse=True)
    again = li.write_runs(tmp_path / "again", {"x.gz": [{"a": 1}]})
    assert again == li.write_runs(tmp_path / "again", {"x.gz": [{"a": 1}]})  # the same bytes every time
    # every interval and verdict again from the sidecar, and through per_question's routing
    assert li.rescore_beir(blob, sidecar) == [] and per_question.differences(blob, sidecar) == []
    moved = copy.deepcopy(blob)
    moved["verdicts"][0]["interval"] = "above zero" if moved["verdicts"][0]["interval"] != "above zero" else "below zero"
    assert any("verdicts" in p for p in per_question.differences(moved, sidecar))
    # the extra part: the parents again, the sensitivity rows, the stretch rows; 12 intervals, no plan flag
    extra, side_extra, _ = _beir_part(tmp_path, "extra")
    assert [r["config"] for r in extra["results"]] == [r["name"] for r in li.ROWS["beir"]["extra"]]
    assert len(extra["verdicts"]) == 12 and not any(v["plan"] for v in extra["verdicts"])
    assert {v["kind"] for v in extra["verdicts"]} == {"sensitivity", "stretch"} and li.rescore_beir(extra, side_extra) == []
    doc512 = {r["config"]: r for r in extra["results"]}["dense@colbertv2/doc512"]["models"]["embedder"]
    assert doc512["settings_read_back"]["document_length"] == 512 and doc512["store_key"] != leg["store_key"]


def test_the_named_rows_rederive_from_the_committed_lists_on_a_cpu(tmp_path, stubs, monkeypatch):
    _write_dataset(tmp_path)
    blob, _, _ = _beir_part(tmp_path, "primary")
    (tmp_path / "bench" / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / li.results_file("scifact", "primary")).write_text(json.dumps(blob))
    monkeypatch.setattr(beir, "download_dataset", lambda data_dir, dataset: Path(data_dir) / dataset)
    checked, problems = li.rederive_beir(tmp_path, tmp_path, "scifact", 1)
    assert problems == [] and checked == 2 * 2 + 2  # dense@ and hybrid@ of both models, the two primary role-(b) heads
    rows = {r["config"]: r for r in blob["results"]}
    rows["hybrid@colbertv2"]["retrieval"]["ndcg@10"] += 0.01
    (tmp_path / li.results_file("scifact", "primary")).write_text(json.dumps(blob))
    assert any("hybrid@colbertv2 ndcg@10" in p for p in li.rederive_beir(tmp_path, tmp_path, "scifact", 1)[1])


def _leg_factory(name, chunks):
    tag, L = li.parse_mv(name)
    return li.MvLeg(name, chunks, corpus_sha256="fixture", encoder=StubEncoder(tag, L), keep_store=True)


@pytest.fixture(scope="module")
def fixture_runs():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(li, "check_fingerprint", lambda encoder, tag: None)
        load = lambda tag: (Overlap(), {"model": tag, "stand_in": True})  # noqa: E731
        return {part: li.run_fixture(ROOT, part=part, embedder="hash", load_reranker=load, leg_factory=_leg_factory) for part in li.PARTS}


def test_the_fixture_rows_their_parents_and_their_verdicts(fixture_runs):
    blob, sidecar, runs = fixture_runs["primary"]
    names = [r["config"] for r in blob["held_out"]["results"]]
    assert names == [r["name"] for r in li.ROWS["fixture"]["primary"]]
    assert blob["held_out"]["bootstrap"]["n_queries"] == 68 and len(blob["split"]["held_out_ids"]) == 68
    assert len(blob["verdicts"]) == 15 and all(v["scope"] == "held_out" and v["config"].startswith("fixture:") for v in blob["verdicts"])
    assert sum(v["plan"] for v in blob["verdicts"]) == 6
    assert li.rescore_fixture(blob, sidecar) == [] and per_question.differences(blob, sidecar) == []
    assert sidecar["schema"] == li.FIXTURE_SIDECAR_SCHEMA
    assert set(runs) == {"fixture_answerai-colbert-small.run.gz", "fixture_colbertv2.run.gz",
                         "fixture_rerank_answerai-colbert-small.json.gz", "fixture_rerank_colbertv2.json.gz"}
    assert all(len(x["ids"]) == 50 for x in runs["fixture_colbertv2.run.gz"]) and len(runs["fixture_colbertv2.run.gz"]) == 136
    for name, leg in blob["legs"].items():
        assert leg["bm25_equal_to_parent"] and leg["store"]["documents"] == blob["n_chunks"] and leg["stored_vs_on_the_fly"]["queries_checked"] == 136
    assert set(blob["rankings"]) == set(names)
    assert [b["file"] for b in blob["reproduces"]] == ["bench/results/neural.json", "bench/results/rerank_depth.json", "bench/results/small_rerankers.json"]
    assert blob["reproduces"][2]["scope"] == "all_questions" and set(blob["reproduces"][2]["rows"]) == {
        "hybrid+minilm-l12@bge-small/fp32-512/top20", "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"}
    assert all(r.get("in_committed_file") is not False for b in blob["reproduces"] for r in b["rows"].values())  # the scope trap


def test_the_fixture_parents_and_the_maxsim_hybrid_rank_as_the_harness_does(fixture_runs):
    blob, _, _ = fixture_runs["primary"]
    rows = {r["config"]: r for r in blob["all_questions"]["results"]}
    main = {r["config"]: r for r in json.loads((ROOT / "bench" / "results" / "main.json").read_text())["results"]}
    assert rows[li.PARENT]["retrieval"] == main["hybrid"]["retrieval"]  # the hash parent is main.json's `hybrid` row
    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    cfg = BenchConfig(name="x", mode="dense", embedder="hash", enable_rerank=False, enable_router=False, staleness_tier1=False, e2e=False, log_triples=False)
    assert rows["dense_only@bge-small"]["retrieval"] == run_config(cfg, gold, FILES, n_files=blob["n_files"], mtimes="manifest")["retrieval"]


def test_the_fixture_extra_part_and_the_fixture_trip(fixture_runs):
    extra, side, runs = fixture_runs["extra"]
    assert [r["config"] for r in extra["held_out"]["results"]] == [r["name"] for r in li.ROWS["fixture"]["extra"]]
    assert len(extra["verdicts"]) == 12 and li.rescore_fixture(extra, side) == []
    assert set(runs) == {"fixture_rerank_answerai-colbert-small_doc512.json.gz", "fixture_rerank_colbertv2_doc512.json.gz"}
    primary = fixture_runs["primary"][0]
    for tag, c in li.CANDIDATES.items():
        cut = primary["legs"][li.mv_name(tag, c["saved"]["document_length"])]["store"]["cut_at_document_length"]
        same = primary["rankings"][f"dense_only@{tag}"] == extra["rankings"][f"dense_only@{tag}/doc512"]
        if cut == 0:
            assert same  # no chunk cut at the primary length: the doc512 rows must rank identically


def _blob_with(rows: dict[str, float], dataset: str) -> dict:
    return {"results": [{"config": n, "retrieval": {"ndcg@10": v}} for n, v in rows.items()]}


def test_the_tripwire_low_high_explained_and_fixture(tmp_path):
    (tmp_path / "bench" / "results").mkdir(parents=True)
    (tmp_path / "bench/results/beir_scifact.json").write_text(json.dumps({"results": [{"config": "bm25", "retrieval": {"ndcg@10": 0.6650}}]}))

    def write(answerai, colbert, colbert512=None, fixture_differs=False, cut=0):
        sci = _blob_with({"dense@answerai-colbert-small": answerai, "dense@colbertv2": colbert}, "scifact")
        ext = _blob_with({"dense@answerai-colbert-small/doc512": answerai, **({"dense@colbertv2/doc512": colbert512} if colbert512 else {})}, "scifact")
        legs = {li.mv_name(t, c["saved"]["document_length"]): {"store": {"cut_at_document_length": cut}} for t, c in li.CANDIDATES.items()}
        fixture = {"legs": legs, "rankings": {f"{p}@{t}": "x" for t in li.CANDIDATES for p in ("dense_only", "hybrid")}}
        fixture["rankings"].update({f"hybrid+{t}@bge-small/fp32-doc{c['saved']['document_length']}/top20": "x" for t, c in li.CANDIDATES.items()})
        fext = {"rankings": {f"{p}@{t}/doc512": ("y" if fixture_differs else "x") for t in li.CANDIDATES for p in ("dense_only", "hybrid")}}
        fext["rankings"].update({f"hybrid+{t}@bge-small/fp32-doc512/top20": "x" for t in li.CANDIDATES})
        for (d, p), blob in {("scifact", "primary"): sci, ("scifact", "extra"): ext, ("fixture", "primary"): fixture, ("fixture", "extra"): fext}.items():
            (tmp_path / li.results_file(d, p)).write_text(json.dumps(blob))

    write(0.7400, 0.6900)
    assert li.tripwire(tmp_path)["status"] == "clear"
    write(0.7400, 0.6000, colbert512=0.6800)  # colbertv2 below bm25 at 180, clear at 512: truncation-explained
    record = li.tripwire(tmp_path)
    assert record["status"] == "truncation-explained" and record["models"]["colbertv2"]["truncation_explained"]
    write(0.7400, 0.6000)  # no doc512 row: no exception
    assert li.tripwire(tmp_path)["status"] == "trip"
    write(0.8100, 0.6900)  # answerai more than 0.05 above its card: a high trip has no exception
    assert li.tripwire(tmp_path)["models"]["answerai-colbert-small"]["status"] == "trip"
    write(0.7400, 0.6900, fixture_differs=True, cut=0)
    assert li.tripwire(tmp_path)["status"] == "trip"
    write(0.7400, 0.6900, fixture_differs=True, cut=3)  # chunks were cut: a difference is expected, no fixture trip
    assert li.tripwire(tmp_path)["status"] == "clear"


def _primary(dataset: str, words: dict[tuple[str, str], tuple[float, float, float]]) -> dict:
    from app.eval.report import verdict

    verdicts = [{"config": f"{dataset}:{a}", "reference": f"{dataset}:{b}", "delta": d, "lo": lo, "hi": hi, "interval": verdict({"lo": lo, "hi": hi}),
                 "kind": "bar" if b == li.PARENT else "reference", "plan": [f"{dataset}:{a}", f"{dataset}:{b}"] in li.PLAN_FLAGS}
                for (a, b), (d, lo, hi) in words.items()]
    results = [{"config": n, "ci": {"ndcg@10": {"mean": 0.5, "lo": 0.4, "hi": 0.6}}} for n in dict.fromkeys([a for a, _ in words] + [li.PARENT])]
    scope = {"bootstrap": {"n_queries": 10}, "results": results}
    return {"verdicts": verdicts, **({"held_out": scope} if dataset == "fixture" else scope), "git_sha": "x"}


def test_the_bar_pass_not_pass_open_and_the_card_claim(tmp_path):
    (tmp_path / "bench" / "results").mkdir(parents=True)
    a, b = li.bar_row("answerai-colbert-small", "a"), li.bar_row("answerai-colbert-small", "b")
    up, tie, down = (0.05, 0.01, 0.09), (0.01, -0.02, 0.04), (-0.05, -0.09, -0.01)
    for d, word in (("fixture", up), ("scifact", up)):
        (tmp_path / li.results_file(d, "primary")).write_text(json.dumps(_primary(d, {(a, li.PARENT): up, (b, li.PARENT): word, (b, li.MINILM_ROW): down})))
    table = li.bar_table(tmp_path)
    model = table["models"]["answerai-colbert-small"]
    assert model["a"]["status"] == "OPEN" and model["a"]["missing"] == ["fiqa"]  # FiQA has no row yet: open, never a pass
    assert table["models"]["colbertv2"]["a"]["status"] == "not measured"
    assert table["card_claim"]["reading"] == "contradicted on fixture and SciFact"
    assert table["outcome"]["word"] == "measured, not adopted" and any("open for `answerai-colbert-small` in role (a)" in c for c in table["outcome"]["open"])
    (tmp_path / li.results_file("fiqa", "primary")).write_text(json.dumps(_primary("fiqa", {(a, li.PARENT): up, (b, li.PARENT): tie})))
    table = li.bar_table(tmp_path)
    assert table["models"]["answerai-colbert-small"]["a"]["status"] == "PASS" and table["models"]["answerai-colbert-small"]["b"]["status"] == "NOT PASS"
    assert table["models"]["answerai-colbert-small"]["b"]["triple"] == "fixture above zero, SciFact above zero, FiQA covers zero"
    assert table["outcome"]["word"] == "measured: passes" and table["outcome"]["passing"] == ["`answerai-colbert-small` passes in role (a)"]
    assert table["outcome"]["not_measured"] == ["colbertv2"]
    for f in tmp_path.glob("bench/results/*.json"):
        f.unlink()
    assert li.bar_table(tmp_path)["outcome"]["word"] == "labelled a demo"



def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "pack.threads=1", *args], cwd=ROOT, capture_output=True, text=True)


def _needs_the_declarations_commit() -> None:
    """The declarations check reads the declarations commit from this repository's history. A shallow clone without it
    skips, as tests/eval/test_agent_injection_adaptive.py does (CI's eval job fetches the full history); a full clone
    without it fails: the results would name a commit that is not in the history, as a squash merge would leave them."""
    if shutil.which("git") is None or _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip("not a git checkout: the declarations commit cannot be read")
    if _git("cat-file", "-e", f"{li.DECLARATIONS_COMMIT}^{{commit}}").returncode != 0:
        if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
            pytest.skip(f"shallow clone without the declarations commit {li.DECLARATIONS_COMMIT[:7]}: git fetch --unshallow")
        pytest.fail(f"the results name the declarations commit {li.DECLARATIONS_COMMIT}, which is not in this repository's history")


def test_the_bar_command_and_the_declarations_check_on_the_committed_files(capsys):
    """`make late-interaction-bar` on the committed files: the declarations commit is an ancestor of every results
    file's git_sha, BAR, SETTINGS and CANDIDATES at that commit equal each header's, and the command exits 0 with the
    bar of every named model in both roles."""
    _needs_the_declarations_commit()
    files = li.all_files(ROOT)
    assert len(files) == 6 and li.declarations_check(ROOT) == []
    assert li.main(["--bar"]) == 0
    out = capsys.readouterr().out
    assert f"declarations commit: OK ({li.DECLARATIONS_COMMIT})" in out
    for tag in li.CANDIDATES:
        for role in ("a", "b"):
            assert f" {tag:24s} role ({role}) {li.bar_row(tag, role)}: " in out


def test_the_declarations_check_finds_a_changed_header_and_a_commit_outside_the_history(tmp_path, monkeypatch):
    """A copied results file passes as committed and fails, with the named problem, once its header's BAR, SETTINGS or
    CANDIDATES differ from the declarations commit's, or its declarations commit is unknown, not an ancestor of its
    git_sha, or missing. Git reads this repository's history (GIT_DIR), so a problem is never just "not a git
    repository" from the copy's directory."""
    _needs_the_declarations_commit()
    monkeypatch.setenv("GIT_DIR", _git("rev-parse", "--absolute-git-dir").stdout.strip())
    rel = li.results_file("fixture", "primary")
    original = json.loads((ROOT / rel).read_text())
    target = tmp_path / rel
    target.parent.mkdir(parents=True)

    def problems(edit) -> list[str]:
        blob = copy.deepcopy(original)
        edit(blob["late_interaction"])
        target.write_text(json.dumps(blob))
        return li.declarations_check(tmp_path)

    assert problems(lambda h: None) == []
    for name, key in (("BAR", "bar"), ("SETTINGS", "settings"), ("CANDIDATES", "candidates")):
        assert problems(lambda h: h[key].update({"changed": True})) == [f"{rel}: {name} at the declarations commit differs from the header's"]
    unknown = problems(lambda h: h.update({"declarations_commit": "0" * 40}))
    assert len(unknown) == 1 and f"{rel}: the declarations commit {'0' * 12} is not an ancestor of {original['git_sha']}" in unknown[0]
    head = _git("rev-parse", "HEAD").stdout.strip()  # a descendant of the results' commit, never its ancestor
    later = problems(lambda h: h.update({"declarations_commit": head}))
    assert later == [f"{rel}: the declarations commit {head[:12]} is not an ancestor of {original['git_sha']} (not an ancestor)"]
    assert problems(lambda h: h.pop("declarations_commit")) == [f"{rel}: no declarations commit in its header"]


def test_the_bar_command_fails_on_a_declarations_problem(monkeypatch, capsys):
    monkeypatch.setattr(li, "declarations_check", lambda root=None: ["bench/results/x.json: a problem"])
    assert li.main(["--bar"]) == 1
    assert "declarations commit: 1 problems" in capsys.readouterr().out


def test_the_readme_block_states_the_recorded_truncation_and_rests_on_committed_files():
    """The generated block prints, per corpus and model at its length, the documents cut and the queries over the
    query cap as each leg recorded them, names the query cap among what the rows do not show, and cites nothing that
    is not a committed file (no uncommitted diagnostic)."""
    from app.eval import report

    text = report.late_interaction_table(ROOT)
    over = {}
    for d in ("scifact", "fiqa"):
        blob = json.loads((ROOT / li.results_file(d, "primary")).read_text())
        for name, leg in blob["late_interaction"]["legs"].items():
            q, s = leg["queries"], leg["store"]
            assert (f"{s['cut_at_document_length']} of {s['documents']} documents cut · {q['over_query_length']} of {q['n']} queries over "
                    f"{q['query_length']} tokens (longest {q['longest_tokenized']})") in text, (d, name)
            over[d] = f"{q['over_query_length']} of {q['n']}"
    tail = text[text.index("**What this does not show.**"):]
    assert "Full-length queries for the named models' MaxSim legs" in tail
    assert f"BEIR SciFact {over['scifact']} and BEIR FiQA {over['fiqa']} queries are longer and are cut in both roles" in tail
    assert "diagnostic" not in text



def test_the_evidence_rows_are_the_recorded_losses_with_the_declared_union_bound():
    """M43's rule: a row only for a bar comparison below zero (or a model that passes); a stretch row, reported as a
    finding, quotes the declared bound that counts the stretch PASS claims too (MULTIPLICITY['row_4_2'])."""
    from app.eval import evidence

    rows = evidence._late_interaction_claims(ROOT)
    named, stretch = [r for r in rows if "STRETCH" not in r["claim"]], [r for r in rows if "STRETCH" in r["claim"]]
    assert len(named) == 5 and len(stretch) == 2 and all("verdict from the interval: below zero" in r["not_shown"] for r in rows)
    assert all("or at most 20% if the four stretch PASS claims are also reported as findings" in r["not_shown"] for r in stretch)
    assert not any("20%" in r["not_shown"] for r in named)


def test_the_committed_parents_and_references_exist_in_the_files_a_part_compares_with():
    for dataset, pairs in li.LATE_INTERACTION_REPRODUCE.items():
        for rel, names in pairs:
            rows = {r["config"] for r in json.loads((ROOT / rel).read_text())["results"]}
            assert set(names) <= rows, (dataset, rel)
    for rel, scope, names in li.FIXTURE_REPRODUCE:
        blob = json.loads((ROOT / rel).read_text())
        assert set(names) <= {r["config"] for r in (blob[scope]["results"] if scope else blob["results"])}, rel


def test_a_settings_mismatch_stops_the_run_and_a_stretch_model_that_fails_is_skipped(monkeypatch):
    enc = StubEncoder("colbertv2", 180)
    li.check_settings(enc, "colbertv2", 180)
    with pytest.raises(li.SettingsMismatch, match="document_length"):
        li.check_settings(enc, "colbertv2", 512)
    wrong = StubEncoder("colbertv2", 180)
    wrong.expand = False
    with pytest.raises(li.SettingsMismatch, match="query_expansion"):
        li.check_settings(wrong, "colbertv2", 180)

    def load(tag, L, device=None):
        if tag == "gte-moderncolbert":
            raise OSError("not in the cache")
        return StubEncoder(tag, L)

    monkeypatch.setattr(li, "load_encoder", load)
    monkeypatch.setattr(li, "check_fingerprint", lambda encoder, tag: None)
    skip, status = li.stretch_gate("extra")
    assert skip == {"gte-moderncolbert"} and status["gte-moderncolbert"].startswith("not loaded: OSError")
    assert li.stretch_gate("primary") == (set(), {})
    assert li.stretch_gate("extra", "timed out twice") == (set(li.STRETCH), {t: "not run: timed out twice" for t in li.STRETCH})
    assert [r["name"] for r in li.rows("beir", "extra", {"gte-moderncolbert"})] == [r["name"] for r in li.ROWS["beir"]["extra"] if r.get("model") != "gte-moderncolbert"]


def test_the_check_texts_rank_through_maxsim_and_a_model_that_cannot_rank_stops():
    from app.eval.models_2026 import ModelCheckFailed

    assert li.check_late_interaction(StubEncoder("answerai-colbert-small", 300))["passed"]

    class Constant(StubEncoder):
        def encode_documents(self, texts):
            return [np.ones((3, self.dim), dtype=np.float32) for _ in texts]

    with pytest.raises(ModelCheckFailed):
        li.check_late_interaction(Constant("answerai-colbert-small", 300))


def test_the_card_check_reads_the_pinned_readme(tmp_path):
    lines = [*li.PUBLISHED["lines"].values(), li.CARD_CLAIM["card_line"], li.FINGERPRINT["answerai-colbert-small"]["shapes_line"],
             li.FINGERPRINT["answerai-colbert-small"]["scores_line"], f'query = "{li.FINGERPRINT["answerai-colbert-small"]["query"]}"',
             *[f'    "{d}",' for d in li.FINGERPRINT["answerai-colbert-small"]["documents"]]]
    readme = tmp_path / "README.md"
    readme.write_text("\n".join(lines) + "\n")
    out = li.card_checks("answerai-colbert-small", readme)
    assert all(out["found"].values()) and out["sha256_equal"] is False and out["passed"] is False  # not the pinned bytes
    readme.write_text("\n".join(lines[1:]) + "\n")
    assert li.card_checks("answerai-colbert-small", readme)["found"]["published_scifact"] is False


def test_the_beir_main_branch_writes_a_part_its_sidecar_and_its_runs_beside_it(tmp_path, stubs, monkeypatch):
    """`make bench-beir-late-interaction PART=extra SKIP_STRETCH=...` as the scoring job runs it, with the stand-ins."""
    _write_dataset(tmp_path / "data")
    declared = li.beir_configs
    monkeypatch.setattr(li, "beir_configs", lambda part, skip=None: [replace(c, embedder="hash") if c.embedder == BGE_SMALL else c for c in declared(part, skip)])
    monkeypatch.setattr(beir, "download_dataset", lambda data_dir, dataset, **kw: Path(data_dir) / dataset)
    monkeypatch.setattr(li, "reference_resolver", lambda root=None: (lambda name: (name, {}, None)))
    out = tmp_path / "results" / "beir_scifact_late_interaction_extra.json"
    assert beir.main(["--ladder", "late-interaction", "--part", "extra", "--data-dir", str(tmp_path / "data"), "--out", str(out), "--n-boot", "100",
                      "--skip-stretch", "timed out twice"]) == 0
    blob, sidecar = json.loads(out.read_text()), json.loads(per_question.sidecar_path(out).read_text())
    assert not [k for k in blob if k.startswith("_")] and not [k for r in blob["results"] for k in r if k.startswith("_")]
    assert blob["late_interaction"]["stretch_status"] == {t: "not run: timed out twice" for t in li.STRETCH}
    assert [r["config"] for r in blob["results"]] == [r["name"] for r in li.ROWS["beir"]["extra"] if r.get("model") not in li.STRETCH]
    assert blob["late_interaction"]["declarations_commit"] == li.DECLARATIONS_COMMIT and blob["late_interaction"]["part"] == "extra"
    assert per_question.differences(blob, sidecar) == []
    runs = set(p.name for p in (tmp_path / "results" / "late_interaction_runs").iterdir())
    assert runs == {"scifact_rerank_answerai-colbert-small_doc512.json.gz", "scifact_rerank_colbertv2_doc512.json.gz"}
    assert {k for k in blob["late_interaction"]["runs"] if not k.startswith("_")} == {f"{li.RUNS_DIR}/{n}" for n in runs}


def test_the_split_only_scopes_the_scores_and_chooses_nothing(fixture_runs, monkeypatch):
    """Leakage check 3: the tune / held-out split is read for nothing but the scoring scopes. With another split every
    ranking is the same; only the scopes' intervals move."""
    import app.eval.rerank_arms as rerank_arms

    blob, _, _ = fixture_runs["primary"]
    monkeypatch.setattr(rerank_arms, "split_questions", lambda gold: (list(range(1, len(gold), 2)), list(range(0, len(gold), 2))))
    monkeypatch.setattr(li, "check_fingerprint", lambda encoder, tag: None)
    again, _, _ = li.run_fixture(ROOT, part="primary", embedder="hash", load_reranker=lambda tag: (Overlap(), {"model": tag}), leg_factory=_leg_factory)
    assert again["rankings"] == blob["rankings"]
    assert again["split"]["held_out_ids"] != blob["split"]["held_out_ids"]
    assert again["all_questions"]["results"] == blob["all_questions"]["results"]
