"""M42: the judge-only entry point (app/eval/judge_stored.py): one judge, gated on the committed calibration answers,
over stored answers; offline, with fake judges and the committed bench/results/jury_v2.json."""

import http.server
import json
import re
import threading
from pathlib import Path

import pytest

from app.agent.generate import strip_citations
from app.eval import judge_stored
from app.eval.judge import correctness_prompt, exact_match_label

ROOT = Path(__file__).resolve().parents[2]
JURY_V2 = ROOT / "bench" / "results" / "jury_v2.json"
FIXTURE = ROOT / "tests" / "eval" / "fixtures" / "stored_answers.jsonl"
BLOCK = {"role": "judge", "model": "fake-judge", "backend": "callable", "revision": "0" * 64}


def exact_judge(prompt: str) -> str:
    """A judge that applies the bounded exact-match rule to the correctness prompt (kappa 1 on the calibration set)
    and prefers the first answer of a pair."""
    if "Which is better" in prompt:
        return "A"
    gold = re.search(r"Gold answer: (.*)\n", prompt).group(1)
    answer = re.search(r"\nAnswer: (.*?)\nDoes the answer", prompt, re.S).group(1)
    return json.dumps({"label": exact_match_label(answer, gold, bounded=True)})


def yes_judge(prompt: str) -> str:
    return "TIE" if "Which is better" in prompt else '{"label": "correct"}'


def _run(tmp_path, complete, answers=(FIXTURE,), name="out.json", judge="fake-judge", **kwargs):
    kwargs.setdefault("judge_block", {**BLOCK, "model": judge})
    return judge_stored.run(judge_model=judge, answers=list(answers), out=tmp_path / name, complete=complete, **kwargs)


def test_the_calibration_items_are_the_committed_ones():
    """The gate's reference is read from the committed jury_v2.json, never regenerated: 160 answers, both classes,
    every reference label equal to the stored one, and one digest for the items a judge is shown."""
    cal = judge_stored.load_calibration(ROOT)
    blob = json.loads(JURY_V2.read_text())
    assert cal["file"] == "bench/results/jury_v2.json" and cal["n"] == 160 == blob["summary"]["reference"]["n"]
    assert cal["generator"]["model"] == "qwen2.5:1.5b" and cal["generator"]["revision"] and cal["rows_git_sha"] == blob["rescore"]["rows_git_sha"]
    labels = [item["reference"] for item in cal["items"]]
    assert {lab: labels.count(lab) for lab in ("correct", "partial", "wrong")} == blob["summary"]["reference"]["label_counts"]
    assert not any(item["abstention_only"] for item in cal["items"]) and all("[" not in item["answer"] or "]" in item["answer"] for item in cal["items"])
    # the digest of the items as a judge sees them; it changes only if jury_v2.json's calibration answers do
    assert cal["items_sha256"] == "4ecf9a02bbc988f0ef5ef879b73dc97c2e0a469424811312aff7dc58277609e6"


def _replay(judge: str):
    """A judge that returns what `judge` replied in the committed run, prompt by prompt."""
    blob = json.loads(JURY_V2.read_text())
    replies: dict[str, str] = {}
    for row in blob["calibration"]["rows"] + blob["rows"]:
        verdict = (row.get("correctness") or {}).get(judge) or {}
        if verdict.get("by") == "judge":
            prompt = correctness_prompt(row["question"], strip_citations(row["answer"]), row["gold"])
            assert replies.setdefault(prompt, verdict["raw"]) == verdict["raw"]
    return blob, lambda prompt: replies[prompt]


@pytest.mark.parametrize("judge", ["qwen2.5:7b", "llama3.1:8b"])
def test_replaying_a_committed_judge_gives_the_committed_gate_and_score(tmp_path, judge):
    """The judge-only path is the committed protocol: fed the replies a judge of jury_v2.json gave, it computes that
    file's kappa, interval, confusion and mean correctness, from the calibration answers and the answers stored there."""
    blob, complete = _replay(judge)
    out = _run(tmp_path, complete, answers=[JURY_V2], judge=judge)
    committed = blob["summary"]["kappa"][judge]
    gate = out["gate"]
    assert (gate["kappa"], gate["interval"], gate["n"]) == (committed["kappa_vs_exact_match_simple_factual"], committed["interval"], committed["n"])
    assert gate["label_counts"] == committed["label_counts"] and gate["confusion_reference_by_judge"] == committed["confusion_reference_by_judge"]
    assert gate["complete"] and gate["passes"] == committed["readme_ok"] == (judge == "qwen2.5:7b")
    assert gate["reference"]["by_condition"] == blob["summary"]["reference"]["by_condition"]
    s = out["answer_sets"][0]["summary"]
    assert (s["mean_correctness"]["mean"], s["mean_correctness"]["n"]) == (committed["mean_correctness_all"], committed["n_all"])
    assert s["n"] == 136 and s["failed"] == {"uncited_answer": 2, "no_answer": 0} and s["abstention_only"]["ids"] == blob["summary"]["abstention_rule"]["ids"]
    own = committed["simple_factual_answers"]
    assert (s["simple_factual_answers"]["agree"], s["simple_factual_answers"]["n"]) == (own["agree"], own["n"])
    assert s["exact_match_simple_factual"] == blob["summary"]["checks"]["exact_match_simple_factual"]
    assert s["replies"]["correctness"] == {"n": 127, "empty": 0, "errors": 0, "unparsed": 0}
    assert s["self_judged"] == (judge == "qwen2.5:7b") and out["answer_sets"][0]["generator"]["model"] == "qwen2.5:7b"  # its own answers
    assert s["publishable"] == gate["passes"] and s["pairs_not_collected"] and s["pairwise_llm_vs_extractive"] is None  # jury_v2 stores no extractive answer
    assert s["cited_in_retrieved"] == blob["summary"]["checks"]["cited_in_retrieved"]  # the file's own check, not recomputed from paths
    agreement = s["agreement_with_stored_judges"]  # this judge against the verdicts the file stores: itself, and the other judge
    other = next(name for name in ("qwen2.5:7b", "llama3.1:8b") if name != judge)
    assert agreement[judge] == {"kappa": 1.0, "agree": 127, "n": 127, "confusion_stored_by_this_judge": agreement[judge]["confusion_stored_by_this_judge"]}
    assert (agreement[other]["kappa"], agreement[other]["n"]) == (blob["summary"]["judge_agreement"]["kappa"], blob["summary"]["judge_agreement"]["n"])
    assert out["status"] == "complete" and judge_stored.check(tmp_path / "out.json") == []


def test_a_validated_judge_scores_the_stored_answers_and_a_failed_one_is_not_published(tmp_path):
    """The addendum's command on the fixture file: the gate and the labels in one run. A judge that passes has its
    scores marked publishable; one that calls everything correct fails the gate and nothing of it is."""
    out = _run(tmp_path, exact_judge)
    assert out["status"] == "complete" and out["gate"]["kappa"] == 1.0 and out["gate"]["passes"] and out["gate"]["n"] == 160
    answers = out["answer_sets"][0]
    assert answers["file"] == "tests/eval/fixtures/stored_answers.jsonl" and answers["n"] == 8 and len(answers["sha256"]) == 64
    rows = {r["id"]: r for r in answers["rows"]}
    assert {qid: r["correctness"]["label"] for qid, r in rows.items() if "correctness" in r} == {
        "q001": "correct", "q002": "wrong", "q003": "correct", "q004": "wrong", "q005": "wrong", "q081": "partial"}
    assert rows["q005"]["correctness"] == {"label": "wrong", "by": "rule"}  # NOT IN SOURCES and nothing else: no call
    assert rows["q006"]["failed"] == "uncited_answer" and rows["q090"]["failed"] == "no_answer" and "correctness" not in rows["q006"]
    assert rows["q001"]["correctness"]["by"] == "judge" and rows["q001"]["correctness"]["raw"] == '{"label": "correct"}'
    s = answers["summary"]
    assert s["n_judged"] == 6 and s["failed"] == {"uncited_answer": 1, "no_answer": 1} and s["abstention_only"]["count"] == 1
    assert s["mean_correctness"]["mean"] == round(2.5 / 6, 4) and s["mean_correctness"]["n"] == 6
    assert s["mean_correctness"]["interval"]["lo"] < s["mean_correctness"]["mean"] < s["mean_correctness"]["interval"]["hi"]
    assert s["label_counts"] == {"correct": 2, "partial": 1, "wrong": 3} and s["exact_match_simple_factual"] == {"rate": 0.4, "n": 5}
    assert s["cited_in_corpus"] == {"rate": 0.8, "n": 5} and s["cited_in_retrieved"] == {"rate": 0.5, "n": 2}
    assert s["simple_factual_answers"]["agree"] == 4 == s["simple_factual_answers"]["n"]  # the rule-labelled abstention is not the judge's
    assert s["pairwise_llm_vs_extractive"] == {"p_llm_better_mean": 0.5, "position_gap_mean": 1.0, "position_flips": 2, "n": 2, "complete": True}
    assert rows["q001"]["pairwise"]["raw"] == ["A", "A"] and "pairwise" not in rows["q003"]
    assert s["publishable"] and not s["self_judged"] and "may be published" in out["verdict"]
    assert out["protocol"]["not_collected"]["composite"] and out["calibration"]["items_sha256"] == judge_stored.load_calibration(ROOT)["items_sha256"]
    assert "| fake-judge | 1.0 | [1.0, 1.0] | 160 |" in judge_stored.markdown_table(out)

    failed = _run(tmp_path, yes_judge, name="yes.json")
    assert failed["gate"]["kappa"] == 0.0 and failed["gate"]["complete"] and not failed["gate"]["passes"]
    assert failed["gate"]["label_counts"] == {"correct": 160, "partial": 0, "wrong": 0}
    s = failed["answer_sets"][0]["summary"]
    assert not s["publishable"] and s["mean_correctness"]["mean"] == round(5 / 6, 4)  # in the file, not for the README
    assert "did not reach kappa >= 0.6" in failed["verdict"] and "off the README" in failed["verdict"]


def test_a_judge_that_stops_answering_interrupts_the_run_and_resume_finishes_it(tmp_path):
    """Failed calls are not verdicts: five in a row stop the run with what it has, the gate is then not complete and
    nothing is publishable; --resume asks the failed and the missing items and keeps the rest."""
    calls = []

    def dies_after_40(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) > 40:
            raise TimeoutError("timed out")
        return exact_judge(prompt)

    out = _run(tmp_path, dies_after_40)
    assert out["status"] == "interrupted" and "5 judge calls failed in a row" in out["interrupted"] and len(calls) == 45
    gate = out["gate"]
    assert not gate["complete"] and not gate["passes"] and gate["n"] == 44 and gate["replies"]["errors"] == 4  # the fifth failure stops the run
    assert "not fully judged" in out["verdict"] and not out["answer_sets"][0]["summary"]["publishable"]
    assert json.loads((tmp_path / "out.json").read_text())["status"] == "interrupted"

    asked = []

    def working(prompt: str) -> str:
        asked.append(prompt)
        return exact_judge(prompt)

    done = _run(tmp_path, working, resume=True)
    assert done["status"] == "complete" and "interrupted" not in done and done["gate"]["passes"] and done["gate"]["replies"]["errors"] == 0
    assert len(asked) == (160 - 40) + 5 + 2 * 2  # the calibration items without a verdict, the fixture's five judged answers, two pairs
    assert not set(asked) & set(calls[:40])  # nothing that had a verdict was asked again
    assert [s["resumed"] for s in done["segments"]] == [False, True] and done["segments"][1]["calls"] == len(asked)
    assert judge_stored.check(tmp_path / "out.json") == []


def test_resume_refuses_another_judge_and_changed_answers(tmp_path):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(FIXTURE.read_text())
    _run(tmp_path, exact_judge, answers=[answers], limit=3)
    with pytest.raises(SystemExit, match="--limit 3"):
        _run(tmp_path, exact_judge, answers=[answers], resume=True)
    with pytest.raises(SystemExit, match="another judge"):  # other weights under the same tag
        _run(tmp_path, exact_judge, answers=[answers], resume=True, limit=3, judge_block={**BLOCK, "revision": "1" * 64})
    answers.write_text(answers.read_text().replace("0.0001", "0.0002"))
    with pytest.raises(SystemExit, match="changed since"):
        _run(tmp_path, exact_judge, answers=[answers], resume=True, limit=3)


def test_a_limited_run_is_never_publishable(tmp_path):
    out = _run(tmp_path, exact_judge, limit=3)
    assert out["status"] == "partial (limit)" and out["gate"]["n"] == 3 and not out["gate"]["complete"] and not out["gate"]["passes"]
    assert len(out["calibration"]["rows"]) == 160 and sum(1 for r in out["calibration"]["rows"] if "verdict" in r) == 3
    assert not out["answer_sets"][0]["summary"]["publishable"]


def test_check_finds_an_edited_verdict_and_an_edited_answers_file(tmp_path):
    out_path = tmp_path / "out.json"
    _run(tmp_path, exact_judge)
    assert judge_stored.check(out_path) == [] and judge_stored.main(["--check", str(out_path)]) == 0
    blob = json.loads(out_path.read_text())
    wrong = next(r for r in blob["calibration"]["rows"] if r["verdict"]["label"] == "wrong")
    wrong["verdict"]["label"] = "correct"
    out_path.write_text(json.dumps(blob))
    assert any(p.startswith("gate:") for p in judge_stored.check(out_path)) and judge_stored.main(["--check", str(out_path)]) == 1
    blob = _run(tmp_path, exact_judge)
    blob["answer_sets"][0]["rows"][0]["answer"] = "Run 40 used a learning rate of 0.0002."
    blob["answer_sets"][0]["sha256"] = "0" * 64
    out_path.write_text(json.dumps(blob))
    problems = judge_stored.check(out_path)
    assert any("sha256" in p for p in problems) and any("rows the judge was shown differ" in p for p in problems)


def test_rescore_rereads_the_inputs_and_keeps_every_verdict(tmp_path):
    """A file judged by an earlier version of the module is brought up to date without a model: the inputs are read
    again and must hash as recorded, the verdicts are kept, the header describes the rescore."""
    out_path = tmp_path / "out.json"
    blob = _run(tmp_path, exact_judge, answers=[JURY_V2, FIXTURE])
    older = json.loads(out_path.read_text())
    for row in older["answer_sets"][0]["rows"]:
        row.pop("stored_verdicts", None)  # as a version without the field wrote it
    del older["answer_sets"][0]["summary"]["agreement_with_stored_judges"]
    out_path.write_text(json.dumps(older))
    assert judge_stored.check(out_path)  # differs from what today's code gives
    assert judge_stored.main(["--rescore", str(out_path)]) == 0
    again = json.loads(out_path.read_text())
    assert judge_stored.check(out_path) == [] and again["gate"] == blob["gate"] and again["segments"] == blob["segments"]
    assert [s["summary"] for s in again["answer_sets"]] == [s["summary"] for s in blob["answer_sets"]]
    assert [r.get("correctness") for s in again["answer_sets"] for r in s["rows"]] == [r.get("correctness") for s in blob["answer_sets"] for r in s["rows"]]
    assert again["rescore"]["rows_git_sha"] == blob["git_sha"] and again["rescore"]["rows_timestamp"] == blob["timestamp"] and "no model" not in again["rescore"]["what"]
    again["answer_sets"][1]["sha256"] = "0" * 64
    out_path.write_text(json.dumps(again))
    with pytest.raises(SystemExit, match="changed since it was judged"):
        judge_stored.rescore(out_path)
    partial = _run(tmp_path, exact_judge, name="partial.json", limit=2)
    assert partial["status"] == "partial (limit)"
    with pytest.raises(SystemExit, match="finish it with --resume"):
        judge_stored.rescore(tmp_path / "partial.json")


def test_stored_answers_must_be_whole_and_carry_the_gold_sets_question(tmp_path):
    from app.eval.gold import load_gold

    gold = {q.id: q for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")}
    lines = FIXTURE.read_text().splitlines()
    path = tmp_path / "answers.jsonl"
    first = json.loads(lines[0])
    path.write_text(json.dumps({k: v for k, v in first.items() if k != "cited_paths"}) + "\n")
    with pytest.raises(SystemExit, match="needs id, question, gold, answer and cited_paths"):
        judge_stored.load_answers(path, ROOT, gold)
    path.write_text(json.dumps({**first, "gold": "2e-04"}) + "\n")
    with pytest.raises(SystemExit, match="does not carry the question and gold answer"):
        judge_stored.load_answers(path, ROOT, gold)
    path.write_text(lines[0] + "\n" + lines[0] + "\n")
    with pytest.raises(SystemExit, match="more than once"):
        judge_stored.load_answers(path, ROOT, gold)
    other = {"id": "x1", "question": "a question outside the gold set", "gold": "7", "answer": "It is 7.", "cited_paths": ["notes.md"], "category": "simple_factual"}
    path.write_text(json.dumps(other) + "\n")
    row = judge_stored.load_answers(path, ROOT, gold)["rows"][0]
    assert row["category"] == "simple_factual" and row["exact_match_label"] == "correct" and row["cited_in_corpus"] is False


def test_a_generate_only_run_is_judged_with_its_pairs(tmp_path):
    """The M42 flow: llm_eval writes answers with no judge, judge_stored labels them later; the stored extractive
    answer is the pair's other side, and an uncited answer stays out, as in the one-process run."""
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    def generator(prompt: str) -> str:
        tag = re.search(r"Source 1 (\[[^\]]+\])\n(.*?)(?:\n\n|$)", prompt, re.S)
        if "run 41" in prompt.split("Question:")[-1]:
            return "The value is in the config."  # no citation: fails loud
        return f"{tag.group(2)} {tag.group(1)}"

    cfg = BenchConfig(name="hybrid+router@hash", mode="hybrid", embedder="hash", enable_rerank=False, enable_router=True, staleness_tier1=True, log_triples=False, e2e=False)
    blob = llm_eval.run(config=cfg, generator_model="fake-gen", judge_models=[], limit=7, complete_fn=generator)
    answers = tmp_path / "jury_v3_answers_fake-gen.json"
    answers.write_text(json.dumps(blob))
    out = _run(tmp_path, exact_judge, answers=[answers])
    judged = out["answer_sets"][0]
    assert judged["generator"]["model"] == "fake-gen" and judged["rows_git_sha"] == blob["git_sha"] and judged["format"].startswith("answer-evaluation file")
    s = judged["summary"]
    uncited = [r["id"] for r in blob["rows"] if r.get("failed")]
    assert uncited == ["q006", "q007"] and s["failed"]["uncited_answer"] == 2 and s["n_judged"] == 5 == s["mean_correctness"]["n"]
    assert s["pairwise_llm_vs_extractive"]["n"] == 5 and s["pairwise_llm_vs_extractive"]["complete"] and s["pairs_not_collected"] is None
    assert all(r["extractive"] == b["extractive"] for r, b in zip(judged["rows"], blob["rows"]))
    assert [r["correctness"]["label"] == r["exact_match_label"] for r in judged["rows"] if "correctness" in r] == [True] * 5
    assert s["exact_match_simple_factual"]["rate"] == blob["summary"]["checks"]["exact_match_simple_factual"]["rate"]


# ---- the Ollama side: a model that cannot load is its own result ------------------------------------------------

@pytest.fixture
def fake_server(monkeypatch):
    """An HTTP server that answers as Ollama does (/api/tags, /api/version, /api/ps, /api/generate); `state` sets what
    it lists, what it reports as loaded, and how it replies."""
    state = {"tags": [], "ps": [], "reply": lambda body: {"response": '{"label": "correct"}', "eval_count": 7, "thinking": "it matches"}, "posts": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            body = {"/api/tags": {"models": state["tags"]}, "/api/version": {"version": "0.0.0-test"}, "/api/ps": {"models": state["ps"]}}.get(self.path)
            self._send(body if body is not None else {}, 200 if body is not None else 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
            state["posts"].append(body)
            if "prompt" not in body:  # a load request
                self._send({"model": body["model"], "done": True, "done_reason": "load"})
                return
            reply = state["reply"](body)
            self._send(reply if reply is not None else {"error": "model runner has unexpectedly stopped"}, 200 if reply is not None else 500)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield state
    server.shutdown()
    server.server_close()


DIGEST = "714d86958b6e" + "a" * 52


def _tag(name, digest=DIGEST):
    return {"name": name, "model": name, "digest": digest, "size": 65_000, "details": {"family": "gpt-oss", "quantization_level": "MXFP4", "parameter_size": "116.8B"}}


def _loaded(name, vram=70_000):
    return {"name": name, "model": name, "digest": DIGEST, "size": 70_000, "size_vram": vram, "context_length": 8192}


def test_a_model_the_server_cannot_serve_is_recorded_as_could_not_load(tmp_path, fake_server):
    """Not listed, not wholly on the GPU, or silent on the warm-up prompt: the file says could not load, no item is
    judged, there is no gate block, and the command exits 3. A failed judge has a kappa; this has none."""
    out = tmp_path / "judge.json"
    args = ["--judge", "big:120b", "--answers", str(FIXTURE), "--out", str(out)]
    assert judge_stored.main(args) == 3
    blob = json.loads(out.read_text())
    assert blob["status"] == "could not load" and "does not list big:120b" in blob["could_not_load"] and blob["gate"] is None
    assert blob["calibration"]["rows"] == [] and blob["answer_sets"] == [] and "not a gate result" in blob["verdict"]
    assert judge_stored.check(out) == []

    fake_server["tags"] = [_tag("big:120b")]
    fake_server["ps"] = [_loaded("big:120b", vram=40_000)]  # part of it on the CPU
    assert judge_stored.main(args) == 3
    blob = json.loads(out.read_text())
    assert "not wholly on the GPU" in blob["could_not_load"] and fake_server["posts"] == [{"model": "big:120b", "stream": False, "keep_alive": "30m"}]

    fake_server["ps"] = [_loaded("big:120b")]
    fake_server["reply"] = lambda body: None  # loaded, but every generation fails
    assert judge_stored.main(args) == 3
    assert "warm-up call failed" in json.loads(out.read_text())["could_not_load"]

    fake_server["tags"] = [_tag("big:120b", digest=None)]
    with pytest.raises(SystemExit, match="no digest"):
        judge_stored.main(args)


def test_the_command_gates_and_labels_through_ollama_with_the_committed_request(tmp_path, fake_server, capsys):
    fake_server["tags"] = [_tag("big:120b")]
    fake_server["ps"] = [_loaded("big:120b")]
    fake_server["reply"] = lambda body: {"response": exact_judge(body["prompt"]) if "Gold answer" in body["prompt"] or "Which is better" in body["prompt"] else '{"label": "correct"}',
                                         "eval_count": 40, "thinking": "compare the two"}
    out = tmp_path / "judge.json"
    assert judge_stored.main(["--judge", "big:120b", "--answers", str(FIXTURE), "--out", str(out)]) == 0
    blob = json.loads(out.read_text())
    judge = blob["judge"]
    assert (judge["model"], judge["revision"], judge["server_version"], judge["backend"]) == ("big:120b", DIGEST, "0.0.0-test", "ollama")
    assert judge["loaded"]["context_length"] == 8192 and judge["loaded"]["size_vram_bytes"] == judge["loaded"]["size_bytes"] == 70_000
    assert judge["warmup"]["reply"] == '{"label": "correct"}' and judge["warmup"]["prompt"] == judge_stored.WARMUP_PROMPT
    assert judge["request"]["think"].startswith("not sent") and blob["gate"]["passes"] and blob["status"] == "complete"
    generate = [b for b in fake_server["posts"] if "prompt" in b]
    assert all(b == {"model": "big:120b", "prompt": b["prompt"], "stream": False, "keep_alive": "30m", "options": {"temperature": 0.0, "seed": 0}} for b in generate)
    assert len(generate) == 1 + 160 + 5 + 4  # the warm-up, the calibration items, the fixture's judged answers, two pairs
    verdict = blob["calibration"]["rows"][0]["verdict"]
    assert verdict["eval_count"] == 40 and verdict["thinking_chars"] == len("compare the two") and verdict["seconds"] >= 0
    assert blob["segments"] == [{k: blob["segments"][0][k] for k in ("git_head", "working_tree_dirty", "timestamp", "command", "device_name", "resumed", "calls", "wall_s")}]
    assert "Verdict: kappa >= 0.6 for big:120b" in capsys.readouterr().out


def test_the_make_targets_name_the_module():
    makefile = (ROOT / "Makefile").read_text()
    for target in ("bench-judge-stored:", "bench-jury-v3-answers:", "bench-jury-v3-judge:", "bench-jury-v3-check:"):
        assert f"\n{target}" in makefile, target
    assert "-m app.eval.judge_stored --judge $(JUDGE)" in makefile and "--judges none" in makefile
