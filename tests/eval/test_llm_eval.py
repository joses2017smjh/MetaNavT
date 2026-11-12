"""M4 offline: cited-answer parsing, cited-bytes verification, two-model jury, and the runner with a fake model."""

import json
import re

import pytest

from app.agent.citation_verify import cited_chunks, extract_claims, verify_claims
from app.agent.generate import NOT_IN_SOURCES, build_prompt, cited_answer, parse_citations, source_tag, strip_citations
from app.eval.judge import pairwise_ab_ba
from app.eval.jury import default_jury
from app.retrieval.types import Chunk, RetrievalHit


def _hit(path, text, start=0, rank=1):
    return RetrievalHit(chunk=Chunk(chunk_id=f"{path}::{start}", path=path, text=text, start_byte=start, end_byte=start + len(text.encode())), score=1.0, rank=rank)


HITS = [
    _hit("configs/run_047.yaml", "learning_rate: 3e-4\nencoder: dinov2\nrun_id: 47"),
    _hit("logs/run_047.out", "epoch 12 val_rmse 0.055", rank=2),
]


def test_prompt_shows_tags_and_citations_parse_back():
    prompt = build_prompt("learning rate for run 47?", HITS)
    assert source_tag(HITS[0]) in prompt and NOT_IN_SOURCES in prompt
    text = f"Run 47 uses a learning rate of 3e-4 {source_tag(HITS[0])}. Its RMSE was 0.055 {source_tag(HITS[1])}."
    cits = parse_citations(text)
    assert [(c.path, c.start_byte, c.end_byte) for c in cits] == [(h.chunk.path, 0, h.chunk.end_byte) for h in HITS]
    assert strip_citations(text) == "Run 47 uses a learning rate of 3e-4. Its RMSE was 0.055."


def test_cited_answer_flags_uncited_unknown_and_abstention():
    good = cited_answer(f"3e-4 {source_tag(HITS[0])}", HITS)
    assert good["citations"] and not good["uncited"] and good["cited_paths_in_retrieved"]
    bad = cited_answer("The learning rate is 3e-4.", HITS)
    assert bad["uncited"] and bad["citations"] == []
    unknown = cited_answer("3e-4 [configs/run_099.yaml:0-10]", HITS)
    assert unknown["uncited"] and unknown["unknown_citations"][0]["path"] == "configs/run_099.yaml" and not unknown["cited_paths_in_retrieved"]
    wrong_range = cited_answer("3e-4 [configs/run_047.yaml:5-9]", HITS)
    assert wrong_range["uncited"]  # a byte range that is not a retrieved chunk is not a valid citation
    assert cited_answer(NOT_IN_SOURCES, HITS)["abstained"] and not cited_answer(NOT_IN_SOURCES, HITS)["uncited"]


def test_values_must_be_in_the_cited_bytes():
    chunks = [h.chunk for h in HITS]
    # 0.055 is in the evidence (log) but the answer cites only the config
    claims = extract_claims("Run 47 uses learning rate 3e-4 and reached val_rmse 0.055.")
    loose = verify_claims(claims, chunks)
    assert loose.hallucinated_values == []
    strict = verify_claims(extract_claims("Run 47 uses learning rate 3e-4 and reached val_rmse 0.055."), chunks, citations=[{"path": "configs/run_047.yaml", "start_byte": 0, "end_byte": HITS[0].chunk.end_byte}], cited_only=True)
    assert strict.hallucinated_values == ["0.055"]
    assert cited_chunks(chunks, ["logs/run_047.out"])[0].path == "logs/run_047.out"
    assert cited_chunks(chunks, [("configs/run_047.yaml", 5, 9)])[0].path == "configs/run_047.yaml"
    assert cited_chunks(chunks, [("configs/run_047.yaml", 0, 999)]) == []  # range not covered by the chunk
    # citation tags are not mistaken for values
    assert "0" not in [c.value for c in extract_claims("lr 3e-4 [configs/run_047.yaml:0-40]")]


def test_two_model_jury_keeps_each_members_label():
    answers = {"qwen": "correct", "llama": "wrong"}
    jury = default_jury(models=["qwen", "llama"])
    for name, judge in jury.members[1:]:
        judge.complete = lambda prompt, _n=name: json.dumps({"score": 1.0 if answers[_n] == "correct" else 0.0, "label": answers[_n]})
    assert [n for n, _ in jury.members] == ["heuristic", "qwen", "llama"]
    v = jury.vote("q", "3e-4", ["learning_rate: 3e-4"], gold="3e-4")
    assert v.labels_by_member["qwen"] == "correct" and v.labels_by_member["llama"] == "wrong"


def test_pairwise_position_swap_detects_bias():
    always_first = lambda prompt: "A"  # noqa: E731 - a judge that always prefers the first position
    res = pairwise_ab_ba(always_first, "q", "answer one", "answer two")
    assert res["p_a_ab"] == 1.0 and res["p_a_ba"] == 0.0 and res["p_a"] == 0.5 and res["position_gap"] == 1.0
    consistent = lambda prompt: "B" if prompt.index("answer one") > prompt.index("answer two") else "A"  # noqa: E731
    res = pairwise_ab_ba(consistent, "q", "answer one", "answer two")
    assert res["p_a"] == 1.0 and res["position_gap"] == 0.0


def test_runner_offline_with_a_fake_model(tmp_path):
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    def fake_complete(prompt: str) -> str:
        if "Rules:" in prompt:  # generation prompt: cite the first source tag, echo a value from it
            tag = re.search(r"Source 1 (\[[^\]]+\])\n(.*?)(?:\n\n|$)", prompt, re.S)
            body = tag.group(2)
            value = re.search(r"\b\d+(?:\.\d+)?(?:e-?\d+)?\b", body)
            return f"The value is {value.group(0) if value else 'unknown'} {tag.group(1)}."
        if "Which is better" in prompt:
            return "A"
        return json.dumps({"score": 0.9, "label": "correct"})

    cfg = BenchConfig(name="hybrid+router@hash", mode="hybrid", embedder="hash", enable_rerank=False, enable_router=True, staleness_tier1=True, log_triples=False, e2e=False)
    blob = llm_eval.run(config=cfg, generator_model="fake-gen", judge_models=["fake-a", "fake-b"], limit=6, complete_fn=fake_complete)

    assert blob["n_gold"] == 6 and len(blob["rows"]) == 6
    s = blob["summary"]
    assert s["checks"]["uncited_answers"]["count"] == 0
    assert set(s["kappa"]) == {"fake-a", "fake-b", "jury_majority"}  # gated: the correctness verdicts
    assert set(s["composite"]) == {"heuristic", "fake-a", "fake-b"} and s["composite"]["heuristic"]["heuristic"] is True
    assert s["kappa"]["fake-a"]["heuristic"] is False and s["replies"]["fake-a"]["correctness"] == {"n": 6, "empty": 0, "unparsed": 0}
    assert blob["protocol"]["correctness_prompt"].startswith("Question: {question}\nGold answer: {gold}")
    assert s["pairwise_llm_vs_extractive"]["fake-a"]["p_llm_better_mean"] == 0.5  # 'A' both ways = pure position bias
    assert s["pairwise_llm_vs_extractive"]["fake-a"]["position_gap_mean"] == 1.0
    assert "kappa_gate" in blob and blob["kappa_gate"] == 0.6
    assert "verdict" in s and "Verdict" in llm_eval.markdown_table(blob)
    row = blob["rows"][0]
    assert row["checks"]["cited_in_retrieved"] and row["checks"]["n_citations"] == 1
    assert "heuristic" in row["judges"] and row["generation"]["model"] == "fake-gen"
    assert row["correctness"]["fake-a"]["by"] == "judge" and row["correctness"]["fake-a"]["label"] == "correct"
    assert row["pairwise_llm_vs_extractive"]["fake-a"]["raw"] == ["A", "A"] and row["judges"]["fake-a"]["raw"]["faithfulness"]
    # The stored config has the keys jury.json was recorded with: switches added later stay out while unset.
    from app.eval.harness import LATER_SETTINGS

    assert not set(LATER_SETTINGS) & set(blob["config"]) and blob["config"]["name"] == cfg.name


@pytest.fixture
def fake_ollama():
    """An HTTP server on 127.0.0.1 that answers /api/tags and /api/version as Ollama does; no model behind it."""
    import http.server
    import threading

    listing = {"models": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = {"/api/tags": listing, "/api/version": {"version": "0.0.0-test"}}.get(self.path)
            if body is None:
                self.send_error(404)
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield listing, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


DIGESTS = {"fake-gen": "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67",
           "fake-a": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
           "fake-b": "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"}
HASH_CFG = dict(name="hybrid+router@hash", mode="hybrid", embedder="hash", enable_rerank=False, enable_router=True, staleness_tier1=True, log_triples=False, e2e=False)


def _tag(name, digest):
    return {"name": name, "model": name, "digest": digest, "size": 1000, "details": {"quantization_level": "Q4_K_M", "parameter_size": "1B"}}


def test_the_model_blocks_name_the_digest_ollama_served(fake_ollama):
    """A43: generator and judges are pinned by the manifest digest /api/tags reports, not by the tag alone."""
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    listing, url = fake_ollama
    listing["models"] = [_tag(name, digest) for name, digest in DIGESTS.items()]
    blob = llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="fake-gen", judge_models=["fake-a", "fake-b"], limit=2,
                        complete_fn=lambda prompt: json.dumps({"score": 0.5, "label": "partial"}), base_url=url)
    models = blob["models"]
    assert models["generator"]["revision"] == DIGESTS["fake-gen"] and models["generator"]["backend"] == "ollama"
    assert [(j["model"], j["revision"]) for j in models["judges"]] == [("fake-a", DIGESTS["fake-a"]), ("fake-b", DIGESTS["fake-b"])]
    assert models["generator"]["quantization"] == "Q4_K_M" and blob["ollama"]["version"] == "0.0.0-test"


def test_a_model_run_refuses_a_tag_without_a_digest(fake_ollama):
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    listing, url = fake_ollama
    listing["models"] = [_tag("fake-gen", DIGESTS["fake-gen"]), _tag("fake-a", None)]
    with pytest.raises(SystemExit, match="no digest"):  # before any generation: the fake server has no /api/generate
        llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="fake-gen", judge_models=["fake-a"], limit=1, base_url=url)
    with pytest.raises(SystemExit, match="not available"):
        llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="fake-gen", judge_models=["fake-b"], limit=1, base_url=url)


def test_exact_match_label_treats_equal_numbers_as_equal():
    from app.eval.judge import exact_match_label

    assert exact_match_label("learning_rate: 0.0001 [configs/run_040.yaml:0-249]", "1e-04") == "correct"
    assert exact_match_label("The rate is 3e-4.", "0.0003") == "correct"
    assert exact_match_label("The rate is 3e-4.", "0.0004") == "wrong"
    assert exact_match_label("run 40 used 2 stereo pairs", "2") == "correct"
    assert exact_match_label("bark_type: oak", "oak") == "correct"
    assert exact_match_label("encoder dinov2", "resnet50") == "wrong"
    assert exact_match_label("the encoder is a resnet", "resnet50 encoder") == "partial"


def test_value_match_is_case_insensitive():
    hits = [_hit("configs/run_047.yaml", "learning_rate: 3e-4\nencoder: dinov2\nrun_id: 47")]
    claims = extract_claims("The encoder was DINOv2 for run 47.")
    assert any(c.value == "DINOv2" for c in claims), [c.value for c in claims]
    res = verify_claims(claims, [h.chunk for h in hits], citations=[{"path": "configs/run_047.yaml", "start_byte": 0, "end_byte": hits[0].chunk.end_byte}], cited_only=True)
    assert res.hallucinated_values == []
    res = verify_claims(extract_claims("The encoder was CLIP for run 47."), [h.chunk for h in hits])
    assert res.hallucinated_values == ["CLIP"]


def _row(qid, category, checks, labels, failed=None, correctness=None):
    """`labels`: each judge's composite label; `correctness`: each LLM judge's gated verdict (asked unless the
    abstention rule applies to `checks`)."""
    row = {"id": qid, "category": category, "question": "q", "gold": "g", "retrieved_paths": [], "answer": "a", "generation": {},
           "checks": checks, "judges": {name: {"label": lab, "score": 1.0} for name, lab in labels.items()}, "pairwise_llm_vs_extractive": {},
           "correctness": {name: ({"label": "wrong", "by": "rule"} if checks.get("abstention_only") else {"label": lab, "by": "judge", "parsed": True, "raw": "{}"})
                           for name, lab in (correctness or {}).items()}}
    if failed:
        row["failed"] = failed
    return row


def test_abstention_is_left_out_of_cited_in_retrieved():
    from app.eval.llm_eval import deterministic_checks, summarize
    from app.eval.gold import GoldQuestion

    q = GoldQuestion(id="q1", question="?", category="semantic", answer="x")
    abstain = deterministic_checks(q, NOT_IN_SOURCES, HITS)
    assert abstain["abstained"] and not abstain["uncited"] and abstain["cited_in_retrieved"] is None
    cited = deterministic_checks(q, f"3e-4 {source_tag(HITS[0])}", HITS)
    assert cited["cited_in_retrieved"] is True
    rows = [
        _row("q1", "semantic", abstain, {"heuristic": "wrong", "j": "wrong"}),
        _row("q2", "semantic", cited, {"heuristic": "correct", "j": "correct"}),
        _row("q3", "semantic", {**cited, "cited_in_retrieved": False, "unknown_citations": [{"path": "configs/run_40.yaml"}]}, {"heuristic": "correct", "j": "correct"}),
    ]
    s = summarize(rows, ["j"])
    assert s["checks"]["cited_in_retrieved"] == {"rate": 0.5, "n": 2}  # the abstention is not a citation failure
    assert s["checks"]["abstained"] == {"count": 1, "n": 3}


def test_single_class_reference_closes_the_gate():
    from app.eval import report
    from app.eval.llm_eval import summarize

    def sf(qid, judge_a, judge_b):
        checks = {"uncited": False, "abstained": False, "cited_in_retrieved": True, "n_citations": 1, "unknown_citations": [], "n_values": 1,
                  "values_in_cited_bytes": True, "values_not_in_cited_bytes": [], "values_in_any_evidence": True, "verification_ratio_cited": 1.0,
                  "exact_match_label": "correct", "exact_match": True}
        return _row(qid, "simple_factual", checks, {"heuristic": "wrong", "yes-man": judge_a, "varies": judge_b},
                    correctness={"yes-man": judge_a, "varies": judge_b})

    rows = [sf(f"q{i}", "correct", "correct" if i % 2 else "partial") for i in range(8)]
    s = summarize(rows, ["yes-man", "varies"])
    assert s["reference"]["single_class"] and not s["reference"]["gate_evaluable"]
    assert s["reference"]["label_counts"] == {"correct": 8, "partial": 0, "wrong": 0}
    assert s["kappa"]["yes-man"]["kappa_vs_exact_match_simple_factual"] == 1.0  # agreement by never varying
    assert s["kappa"]["varies"]["kappa_vs_exact_match_simple_factual"] == 0.0
    assert not any(k["readme_ok"] for k in s["kappa"].values())
    assert s["publishable_judges"] == [] and "not evaluable" in s["verdict"]
    blob = {"n_gold": 8, "git_sha": "abc1234", "wall_s": 1.0, "kappa_gate": 0.6, "config": {"name": "cfg"},
            "models": {"generator": {"model": "gen"}, "judges": [{"model": "yes-man"}, {"model": "varies"}], "reranker": {"loaded": True}},
            "rescore": {"rows_git_sha": "abc1234", "checks_git_sha": "def5678"}, "summary": s}
    text = report.jury_table(blob)
    assert "no (not evaluable)" in text and "could not be evaluated" in text and "recomputed by `make bench-jury-rescore` at sha def5678" in text
    assert "Judge scores (published" not in text


def test_rescore_recomputes_checks_from_stored_answers(tmp_path):
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    def fake_complete(prompt: str) -> str:
        if "Rules:" in prompt:
            tag = re.search(r"Source 1 (\[[^\]]+\])\n(.*?)(?:\n\n|$)", prompt, re.S)
            return f"See {tag.group(1)}."
        if "Which is better" in prompt:
            return "TIE"
        return json.dumps({"score": 0.5, "label": "partial"})

    cfg = BenchConfig(name="hybrid+router@hash", mode="hybrid", embedder="hash", enable_rerank=False, enable_router=True, staleness_tier1=True, log_triples=False, e2e=False)
    blob = llm_eval.run(config=cfg, generator_model="fake-gen", judge_models=["fake-a", "fake-b"], limit=4, complete_fn=fake_complete)
    path = tmp_path / "jury.json"
    path.write_text(json.dumps(blob))
    out = llm_eval.rescore(path)
    assert out["summary"]["checks"] == blob["summary"]["checks"]  # same code, same answers: same checks
    assert out["summary"]["kappa"] == blob["summary"]["kappa"]
    assert [r["judges"] for r in out["rows"]] == [r["judges"] for r in blob["rows"]]
    assert out["rescore"]["rows_git_sha"] == blob["git_sha"] and out["rescore"]["rows_command"] == blob["command"]
    assert out["git_sha"] == out["rescore"]["checks_git_sha"]
    published = json.loads(path.read_text())  # a file written before the provenance sanitizer
    published.pop("rescore", None)
    published["command"] = "HF_HOME=/nfs/hpc/share/jdoe42/hf_cache python -m app.eval.llm_eval --out /scratch/jdoe42/jury.json"
    published["device"] = {**published["device"], "hostname": "node-17.cluster.example.edu"}
    path.write_text(json.dumps(published))
    again = llm_eval.rescore(path)
    assert again["rescore"]["rows_command"] == "HF_HOME=<path>/hf_cache python -m app.eval.llm_eval --out <path>/jury.json"
    assert "hostname" not in again["device"] and "node-17" not in json.dumps(again)
    tampered = json.loads(path.read_text())
    tampered["rows"][0]["retrieved_paths"] = ["nope.txt"]
    path.write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match="retrieval differs"):
        llm_eval.rescore(path)


# ---- M29 (D12): a reference with both classes, a gated correctness verdict, the abstention rubric

def _stored(qid):
    """A row of the committed M4 run (bench/results/jury.json, generator and first judge qwen2.5:7b)."""
    from pathlib import Path

    blob = json.loads((Path(__file__).resolve().parents[2] / "bench" / "results" / "jury.json").read_text())
    return next(r for r in blob["rows"] if r["id"] == qid)


def test_the_reference_does_not_find_the_gold_inside_a_longer_number():
    from app.eval.gold import GoldQuestion
    from app.eval.judge import exact_match_label
    from app.eval.llm_eval import deterministic_checks

    row = _stored("q098")  # gold 8; the stored answer counts 5 runs and lists run 48
    answer = strip_citations(row["answer"])
    assert exact_match_label(answer, row["gold"]) == "correct"  # plain containment finds "8" inside "48"
    assert exact_match_label(answer, row["gold"], bounded=True) == "wrong"
    assert exact_match_label("run 42 used 3 stereo pairs", "2", bounded=True) == "wrong"
    assert exact_match_label("val_rmse: 0.0533", "0.053", bounded=True) == "wrong"
    assert exact_match_label("run 40 used 2 stereo pairs.", "2", bounded=True) == "correct"
    assert exact_match_label("learning_rate: 0.0001", "1e-04", bounded=True) == "correct"
    assert exact_match_label("bark_type: oak", "oak", bounded=True) == "correct"
    assert exact_match_label("the dinov2 run used 3 pairs", "2", bounded=True) == "wrong"  # no number inside a word
    assert exact_match_label("the dinov2 run used 3 pairs", "2") == "correct"  # the unbounded rule reads the 2 of dinov2
    assert exact_match_label("encoder: resnet50", "50", bounded=True) == "wrong"
    assert exact_match_label("The val RMSE was 0.0550.", "0.055", bounded=True) == "correct"
    q = GoldQuestion(id="q", question="how many stereo pairs did run 42 use", category="simple_factual", answer="2")
    assert deterministic_checks(q, f"Run 42 used 3 pairs {source_tag(HITS[0])}.", HITS)["exact_match_label"] == "wrong"


def test_the_abstention_rubric_on_the_stored_answers():
    """In the M4 run both judges labelled q109's and q123's NOT IN SOURCES "correct", though both questions have a
    gold answer. The rubric labels such an answer wrong without asking a judge. q088 states a value and also says
    NOT IN SOURCES, so a judge is asked, and its prompt says NOT IN SOURCES gives no answer."""
    from app.eval.gold import GoldQuestion
    from app.eval.judge import correctness_prompt
    from app.eval.llm_eval import abstention_only, deterministic_checks, rule_label

    for qid in ("q109", "q123"):
        row = _stored(qid)
        assert row["answer"] == NOT_IN_SOURCES and row["gold"]
        assert {name: v["label"] for name, v in row["judges"].items() if name != "heuristic"} == {"qwen2.5:7b": "correct", "llama3.1:8b": "correct"}
        q = GoldQuestion(id=qid, question=row["question"], category=row["category"], answer=row["gold"])
        checks = deterministic_checks(q, row["answer"], HITS)
        assert checks["abstention_only"] and rule_label(checks, row["gold"]) == "wrong"
    mixed = _stored("q088")
    answer = strip_citations(mixed["answer"])
    assert NOT_IN_SOURCES in answer and not abstention_only(answer)
    assert rule_label({"abstention_only": abstention_only(answer)}, mixed["gold"]) is None
    assert f"{NOT_IN_SOURCES} gives no answer" in correctness_prompt(mixed["question"], answer, mixed["gold"])
    assert abstention_only(f"{NOT_IN_SOURCES}.") and rule_label({"abstention_only": True}, "") is None  # no gold: a judge decides


def test_the_verdict_parser_counts_what_it_cannot_read():
    from app.eval.judge import parse_verdict

    assert parse_verdict('```json\n{"label": "wrong"}\n```') == ("wrong", True)
    assert parse_verdict("{'label': 'partial'}") == ("partial", True)
    assert parse_verdict("**Correct**") == ("correct", True) and parse_verdict('{"label": "incorrect"}') == ("wrong", True)
    assert parse_verdict("The answer is not correct.") == ("partial", False)  # prose is not read as a label
    assert parse_verdict("") == ("partial", False)


def _sf(qid, reference, composite, verdict, *, abstention=False):
    checks = {"uncited": False, "abstained": abstention, "abstention_only": abstention, "cited_in_retrieved": None if abstention else True,
              "n_citations": 0 if abstention else 1, "unknown_citations": [], "n_values": 0 if abstention else 1, "values_in_cited_bytes": True,
              "values_not_in_cited_bytes": [], "values_in_any_evidence": True, "verification_ratio_cited": 1.0,
              "exact_match_label": reference, "exact_match": reference == "correct"}
    return _row(qid, "simple_factual", checks, {"heuristic": "partial", "j": composite}, correctness={"j": verdict})


def test_the_gate_reads_the_correctness_verdict_and_pairs_labels_per_row():
    """A wrong value copied faithfully from a retrieved chunk scores about 1, 1, 0 on the M4 dimensions, so the
    composite label is partial where the reference says wrong; the correctness verdict can agree with it."""
    from app.eval.llm_eval import summarize

    rows = [_sf(f"q{i}", "correct", "correct", "correct") for i in range(4)]
    rows += [_sf(f"q{i}", "wrong", "partial", "wrong") for i in range(4, 8)]
    rows.append(_sf("q8", "wrong", "correct", None, abstention=True))  # labelled wrong by rule, composite said correct
    for r in rows[2:8]:  # judge k labelled only six of the reference rows
        r["correctness"]["k"] = {"label": r["checks"]["exact_match_label"], "by": "judge", "parsed": True, "raw": "{}"}
    s = summarize(rows, ["j", "k"])
    ref = s["reference"]
    assert ref["n"] == 8 and ref["label_counts"] == {"correct": 4, "partial": 0, "wrong": 4} and ref["gate_evaluable"]
    assert ref["left_out_abstention_only"] == 1 and ref["label_counts_with_abstentions"]["wrong"] == 5
    j = s["kappa"]["j"]
    assert j["kappa_vs_exact_match_simple_factual"] == 1.0 and j["n"] == 8 and j["readme_ok"]
    assert j["confusion_reference_by_judge"]["wrong"] == {"correct": 0, "partial": 0, "wrong": 4}
    assert j["mean_correctness_all"] == round(4 / 9, 4) and j["n_all"] == 9  # the rule-labelled abstention counts as wrong
    assert s["kappa"]["k"]["kappa_vs_exact_match_simple_factual"] == 1.0 and s["kappa"]["k"]["n"] == 6  # paired by row
    assert s["composite"]["j"]["kappa_vs_exact_match_simple_factual"] == 0.3333 and s["composite"]["j"]["abstentions_labelled_correct"] == ["q8"]
    assert s["publishable_judges"] == ["j", "k"] and s["abstention_rule"]["ids"] == ["q8"]
    assert s["kappa"]["j"]["interval"] == {"lo": 1.0, "hi": 1.0}


def test_a_pure_abstention_is_labelled_by_rule_without_a_judge_call():
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    asked = []

    def fake(prompt: str) -> str:
        if "Rules:" in prompt:
            return NOT_IN_SOURCES
        if "Does the answer give the gold answer?" in prompt:
            asked.append(prompt)
            return '{"label": "correct"}'
        if "Which is better" in prompt:
            return "TIE"
        return json.dumps({"score": 1.0, "label": "correct"})  # the M4 prompts, which called abstentions correct

    blob = llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="fake-gen", judge_models=["fake-a"], limit=3, complete_fn=fake)
    s = blob["summary"]
    assert asked == [] and all(r["correctness"]["fake-a"] == {"label": "wrong", "by": "rule"} for r in blob["rows"])
    assert s["abstention_rule"]["n"] == 3 and s["reference"]["n"] == 0 and s["reference"]["left_out_abstention_only"] == 3
    assert not s["reference"]["gate_evaluable"] and s["publishable_judges"] == [] and "empty" in s["verdict"]
    assert s["composite"]["fake-a"]["abstentions_labelled_correct"] == ["q001", "q002", "q003"]
    assert s["kappa"]["fake-a"]["mean_correctness_all"] == 0.0 and s["replies"]["fake-a"]["correctness"]["n"] == 0


def test_empty_and_unreadable_judge_replies_are_counted():
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    def fake(prompt: str) -> str:
        if "Rules:" in prompt:
            tag = re.search(r"Source 1 (\[[^\]]+\])\n(.*?)(?:\n\n|$)", prompt, re.S)
            return f"The value is 7 {tag.group(1)}."
        if "Does the answer give the gold answer?" in prompt:
            return "I think it is fine."
        return ""  # a failed call: ollama_complete returns None, which the judge reads as ""

    blob = llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="fake-gen", judge_models=["fake-a"], limit=3, complete_fn=fake)
    assert blob["summary"]["replies"]["fake-a"] == {"pointwise": {"n": 9, "empty": 9}, "correctness": {"n": 3, "empty": 0, "unparsed": 3},
                                                     "pairwise": {"n": 6, "empty": 6}}
    assert all(r["correctness"]["fake-a"]["label"] == "partial" and not r["correctness"]["fake-a"]["parsed"] for r in blob["rows"])


def test_rescore_refuses_when_the_abstention_rule_would_label_other_rows(tmp_path):
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    def fake(prompt: str) -> str:
        return NOT_IN_SOURCES if "Rules:" in prompt else ("TIE" if "Which is better" in prompt else '{"label": "wrong"}')

    blob = llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="fake-gen", judge_models=["fake-a"], limit=2, complete_fn=fake)
    path = tmp_path / "jury_v2.json"
    path.write_text(json.dumps(blob))
    assert llm_eval.rescore(path)["summary"] == blob["summary"]
    blob["rows"][0]["answer"] = f"{NOT_IN_SOURCES} for run 49."  # still an abstention, no longer a pure one
    path.write_text(json.dumps(blob))
    with pytest.raises(SystemExit, match="abstention rule"):
        llm_eval.rescore(path)


# ---- M29 (D12): the calibration set, the gate's reference when the run has a calibration generator

def test_planted_values_come_from_the_same_field_and_differ_from_the_gold():
    from app.eval.gold import load_gold
    from app.eval.harness import project_root
    from app.eval.judge import exact_match_label
    from app.eval.llm_eval import planted_values, question_template

    gold = load_gold(project_root() / "bench" / "gold" / "questions.jsonl")
    planted = planted_values(gold)
    sf = [q for q in gold if q.category == "simple_factual"]
    assert set(planted) == {q.id for q in sf} and all(planted.values())
    by_template = {}
    for q in sf:
        by_template.setdefault(question_template(q.question), set()).add(q.answer)
    for q in sf:
        assert planted[q.id] in by_template[question_template(q.question)]  # a value the same field takes
        assert exact_match_label(planted[q.id], q.answer, bounded=True) != "correct"  # 2e-04 is not 1e-04, 3 is not 2
    assert planted["q001"] == "2e-04" and planted["q003"] == "pine" and planted["q004"] == "clip"  # run 40: 1e-04, oak, resnet50


def test_planting_replaces_every_standalone_occurrence_in_memory():
    from app.eval.llm_eval import plant

    hits = [_hit("configs/run_040.yaml", "run_id: 40\nlearning_rate: 0.0001\nnum_pairs: 2\nencoder: resnet50"),
            _hit("slurm/run_040.sbatch", "python train.py --num_pairs=2 --encoder resnet50 --model dinov2", rank=2)]
    planted, count = plant(hits, "2", "3")
    assert count == 2 and "num_pairs: 3" in planted[0].chunk.text and "--num_pairs=3" in planted[1].chunk.text
    assert "dinov2" in planted[1].chunk.text and "run_id: 40" in planted[0].chunk.text  # digits inside words and numbers stay
    assert [source_tag(h) for h in planted] == [source_tag(h) for h in hits] and "num_pairs: 2" in hits[0].chunk.text  # tags kept, input untouched
    assert plant(hits, "1e-04", "2e-04")[0][0].chunk.text.count("2e-04") == 1  # the numeric equivalent 0.0001 is replaced
    assert plant(hits, "oak", "pine")[1] == 0


def _echo_and_exact_judge(prompt: str) -> str:
    """A fake model: generation copies every source's text after its first tag; the correctness judge applies the
    bounded exact-match rule (so its kappa is 1); other prompts get fixed replies."""
    from app.eval.judge import exact_match_label

    if "Rules:" in prompt:
        tag = re.search(r"Source 1 (\[[^\]]+\])", prompt).group(1)
        bodies = re.findall(r"Source \d+ \[[^\]]+\]\n(.*?)(?=\n\nSource \d+ |\n\nQuestion: )", prompt, re.S)
        return f"{' '.join(bodies)} {tag}"
    if "Does the answer give the gold answer?" in prompt:
        gold = re.search(r"Gold answer: (.*)\n", prompt).group(1)
        answer = re.search(r"\nAnswer: (.*?)\nDoes the answer", prompt, re.S).group(1)
        return json.dumps({"label": exact_match_label(answer, gold, bounded=True)})
    if "Which is better" in prompt:
        return "TIE"
    return json.dumps({"score": 0.9, "label": "correct"})


def test_the_calibration_set_is_the_reference_and_stays_out_of_the_checks(tmp_path):
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    blob = llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="fake-gen", judge_models=["fake-a", "fake-b"], limit=5,
                        complete_fn=_echo_and_exact_judge, calibration_generator="fake-small")
    cal = blob["calibration"]
    assert cal["generator"]["model"] == "fake-small" and len(cal["rows"]) == 10  # five simple-factual questions, two conditions
    planted = [r for r in cal["rows"] if r["condition"] == "planted"]
    assert all(r["probe"] and r["replacements"] > 0 and r["checks"]["exact_match_label"] != "correct" for r in planted)
    assert all(r["correctness"]["fake-a"]["by"] == "judge" for r in cal["rows"])
    s = blob["summary"]
    ref = s["reference"]
    assert ref["source"] == "calibration" and ref["n"] == 10 and ref["by_condition"]["planted"]["correct"] == 0
    assert ref["gate_evaluable"] and s["kappa"]["fake-a"]["kappa_vs_exact_match_simple_factual"] == 1.0 and s["publishable_judges"] == ["fake-a", "fake-b"]
    assert s["checks"]["exact_match_simple_factual"]["n"] == 5  # the calibration rows are in no deterministic-check rate
    assert s["kappa"]["fake-a"]["n_all"] == 5 and s["kappa"]["fake-a"]["simple_factual_answers"]["n"] == 5
    assert s["replies"]["fake-a"]["calibration_correctness"] == {"n": 10, "empty": 0, "unparsed": 0}
    path = tmp_path / "jury_v2.json"
    path.write_text(json.dumps(blob))
    again = llm_eval.rescore(path)
    assert again["summary"] == s and [r["checks"] for r in again["calibration"]["rows"]] == [r["checks"] for r in cal["rows"]]
    planted[0]["replacements"] += 1
    path.write_text(json.dumps(blob))
    with pytest.raises(SystemExit, match="replacements"):
        llm_eval.rescore(path)


def test_calibration_answers_are_judged_cited_or_not():
    """The verdict sees neither context nor citations, so an uncited calibration answer is still judged; an uncited
    answer of the run itself still fails loud and is not judged."""
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    def uncited(prompt: str) -> str:
        return re.sub(r"\s*\[[^\]]+\]$", "", _echo_and_exact_judge(prompt)) if "Rules:" in prompt else _echo_and_exact_judge(prompt)

    blob = llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="fake-gen", judge_models=["fake-a"], limit=2,
                        complete_fn=uncited, calibration_generator="fake-small")
    assert all(r.get("failed") == "uncited_answer" and not r["correctness"] for r in blob["rows"])
    assert all(r["checks"]["uncited"] and r["correctness"]["fake-a"]["by"] == "judge" for r in blob["calibration"]["rows"])


def test_the_readme_block_renders_the_calibration_gate_and_links_the_earlier_record():
    from app.eval import llm_eval, report
    from app.eval.harness import BenchConfig

    blob = llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="gen:7b", judge_models=["gen:7b", "other:8b"], limit=5,
                        complete_fn=_echo_and_exact_judge, calibration_generator="gen:1b")
    for block in [blob["models"]["generator"], blob["models"]["judges"][0], blob["calibration"]["generator"]]:
        block["family"] = "gen"
    text = report.jury_table(blob)  # a file with a protocol block renders as the M29 table
    assert "`make bench-jury-v2`" in text and "calibration set" in text and "planted value of the same field" in text
    assert "gen:7b is also a judge" in text and "shares a model family (gen) with the judge gen:7b" in text
    assert "| gen:7b | 1.0 | [1.0, 1.0] | 10 | yes | 5 / 0 / 5 |" in text and "Judge scores of the run's answers" in text
    assert "[bench/results/jury.json](bench/results/jury.json)" in text and "`make bench-jury-rescore` refuses" in text
    assert "gen:7b scores answers its own weights generated (self-preference risk)" in text and "labelled all" not in text
    assert "| gen:7b | 1.0 | 5 | 0 |" in text  # the score row says how many labels the abstention rule set
    blob["summary"]["kappa"]["other:8b"]["label_counts"] = {"correct": 10, "partial": 0, "wrong": 0}  # a judge with one verdict
    assert "other:8b labelled all 10 calibration answers correct" in report.jury_table(blob)
    blob["summary"]["publishable_judges"] = []
    blob["summary"]["kappa"]["gen:7b"]["readme_ok"] = False
    assert "Judge scores of the run's answers" not in report.jury_table(blob)


def test_the_evidence_rows_publish_a_judge_score_only_after_the_gate():
    from app.eval import evidence, llm_eval
    from app.eval.harness import BenchConfig

    blob = llm_eval.run(config=BenchConfig(**HASH_CFG), generator_model="gen:7b", judge_models=["gen:7b", "other:8b"], limit=5,
                        complete_fn=_echo_and_exact_judge, calibration_generator="gen:1b")
    rows = evidence._jury_v2_claims(blob, "bench/results/jury_v2.json")
    assert all(r["command"] == "make bench-jury-v2" and r["source"].startswith("bench/results/jury_v2.json @ ") and r["not_shown"] for r in rows)
    scores = [r for r in rows if "mean correctness" in r["claim"]]
    assert [r["claim"].split(" (")[0] for r in scores] == ["Judge gen:7b", "Judge other:8b"] and "self-preference" in scores[0]["not_shown"]
    assert scores[0]["not_shown"].startswith("0 of the 5 labels come from the abstention rule")
    kappas = [r for r in rows if "Cohen's kappa of its correctness verdicts" in r["claim"]]
    assert [r["ci"] for r in kappas] == ["[1.0, 1.0]", "[1.0, 1.0]"] and all("planted" in r["not_shown"] for r in kappas)
    blob["summary"]["publishable_judges"] = []
    assert not [r for r in evidence._jury_v2_claims(blob, "bench/results/jury_v2.json") if "mean correctness" in r["claim"]]


def test_claim_values_come_out_in_the_same_order_in_every_process():
    """The value checks list the values an answer states; deduplicating them through a set made that order follow
    the process's hash seed, so a rescore of the same answers wrote the lists of four rows in another order."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = ("from app.agent.citation_verify import extract_claims; "
            "print([c.value for c in extract_claims('Runs 42, 47 and 53 used 3 pairs with dinov2 and clip, val 0.0568 and 0.0521, lr 3e-4.')])")
    root = Path(__file__).resolve().parents[2]
    outs = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": str(root)}).stdout for seed in range(1, 7)}
    assert len(outs) == 1, outs


def test_the_committed_answer_evaluation_is_evaluable_pinned_and_unedited():
    """D12 and A43 on bench/results/jury_v2.json: a reference with both classes, each judge's kappa and the gate's
    verdict from it, no null model revision, and a summary that is what summarize() computes from the stored rows."""
    from pathlib import Path

    from app.eval.llm_eval import summarize

    blob = json.loads((Path(__file__).resolve().parents[2] / "bench" / "results" / "jury_v2.json").read_text())
    models = blob["models"]
    pinned = [models["generator"], *models["judges"], blob["calibration"]["generator"], models["embedder"], models["reranker"]]
    assert all(m["revision"] for m in pinned) and models["reranker"]["loaded"]
    judges = [j["model"] for j in models["judges"]]
    s = blob["summary"]
    ref = s["reference"]
    assert ref["source"] == "calibration" and ref["gate_evaluable"] and not ref["single_class"]
    assert ref["label_counts"]["correct"] > 0 and ref["label_counts"]["partial"] + ref["label_counts"]["wrong"] > 0
    for name in judges:
        k = s["kappa"][name]
        assert k["n"] == ref["n"] and k["readme_ok"] == (k["kappa_vs_exact_match_simple_factual"] >= blob["kappa_gate"])
    assert s["publishable_judges"] == [name for name in judges if s["kappa"][name]["readme_ok"]]
    assert summarize(blob["rows"], judges, blob["calibration"]["rows"]) == s
    planted = [r for r in blob["calibration"]["rows"] if r["condition"] == "planted"]
    assert planted and all(r["probe"] == bool(r["replacements"]) and r["planted_value"] != r["gold"] for r in planted)
