"""M43: what a reranker would cost the demo's API. scripts/rerank_timing_payload.py is the measuring program,
scripts/measure_rerank_cpu.py runs it on whatever CPU is at hand (indicative), and scripts/time_reranker_on_host.py
sends it to the live host through SSM (the serving bar). Checked without torch, Docker or AWS: no call leaves this
process, and the host's script is only generated and parsed.
"""

import ast
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))  # the scripts import each other by name, as when run from a checkout
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


payload = _load("rerank_timing_payload")
host = _load("time_reranker_on_host")
cpu = _load("measure_rerank_cpu")
api_memory = _load("measure_api_memory")

RESULT = {"max_length": 512, "threads": 2, "torch_threads": 2, "torch": "2.5.1+cpu", "cpu_count": 2, "reranker_load_seconds": 1.5,
          "memory_mib": {"after_import_main": {"VmRSS": 400, "VmHWM": 400, "RssAnon": 280, "RssFile": 120},
                         "after_embedder": {"VmRSS": 780, "VmHWM": 850, "RssAnon": 540, "RssFile": 240},
                         "after_reranker_load": {"VmRSS": 880, "VmHWM": 880, "RssAnon": 600, "RssFile": 280}},
          "reranker": {"revision": "abc", "precision": "fp32", "max_length": 512, "device": "cpu", "dtype": "torch.float32"},
          "shapes": [{"depth": 20, "p50_ms": 2100.0, "p95_ms": 2400.5, "min_ms": 2000.0, "max_ms": 2500.0, "n": 30, "warm_up_ms": 2600.0,
                      "memory_mib": {"VmRSS": 990, "VmHWM": 1010, "RssAnon": 700, "RssFile": 290}, "cgroup_peak_mib": 1015}]}


# ---- the measuring program


def test_the_payload_needs_only_the_standard_library_before_it_meets_the_app():
    tree = ast.parse((ROOT / "scripts" / "rerank_timing_payload.py").read_text())
    top = {name.name for node in tree.body if isinstance(node, ast.Import) for name in node.names}
    assert top == {"json", "os", "random", "sys", "time"}  # the image on the host is the deployed one, whatever commit
    assert not [node for node in tree.body if isinstance(node, ast.ImportFrom)]


def test_the_payload_sets_the_single_hosts_settings_as_the_memory_record_did():
    mine = {**payload.SETTINGS}
    theirs = {k: v for k, v in api_memory.SETTINGS.items() if k not in ("RERANKER_MODEL", "RERANKER_MAX_LENGTH")}
    assert mine == theirs  # the reranker's own two come from TIMING_MODEL and TIMING_MAX_LENGTH
    assert (payload.QUERY, payload.PASSAGE) == (api_memory.QUERY, api_memory.PASSAGE)


def test_the_timed_texts_are_seeded_distinct_and_longer_than_the_cap():
    a, b = payload.texts(3, 20), payload.texts(3, 20)
    assert a == b and len(a) == 20
    assert len({passage for _, passage in a}) == 20 and len({query for query, _ in a}) == 1
    assert payload.texts(4, 20)[0] != a[0]  # each timed call is another query with other passages
    assert all(len(passage.split()) == payload.PASSAGE_WORDS >= 600 for _, passage in a)  # past 512 tokens: cut at the cap


def test_percentiles_are_numpys():
    samples = [5.0, 1.0, 9.5, 3.25, 7.0, 2.0, 8.0]
    for q in (50, 95, 0, 100):
        assert payload.percentile(samples, q) == pytest.approx(float(np.percentile(samples, q)))
    assert payload.summary_ms(samples) == {"p50_ms": 5.0, "p95_ms": 9.1, "min_ms": 1.0, "max_ms": 9.5, "n": 7}
    assert payload.percentile([4.0], 95) == 4.0


# ---- the host's script (never run here: generated, parsed, read)


def _script(**changes):
    args = dict(model="cross-encoder/ms-marco-MiniLM-L6-v2", model_dir=None, depths=[20], max_length=512, repeats=30, min_repeats=5,
                budget_seconds=240, memory=None, stop_api=True)
    return host.remote_script(**{**args, **changes})


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash to parse the script")
def test_the_hosts_script_parses_and_carries_the_payload():
    for script in (_script(), _script(stop_api=False), _script(model=None, model_dir="/opt/metanavit/student", memory="1536m", depths=[20, 5, 10])):
        assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0
    script = _script()
    encoded = script.split("<<'PAYLOAD_B64'\n", 1)[1].split("\nPAYLOAD_B64\n", 1)[0]
    assert base64.b64decode(encoded).decode() == (ROOT / "scripts" / "rerank_timing_payload.py").read_text()
    with pytest.raises(SyntaxError):  # a payload that does not parse is caught here, not on the host
        _script(payload="def broken(:\n")


def test_the_one_off_container_is_the_deployed_image_under_the_apis_limits():
    script = _script()
    assert 'IMAGE=$(setting METANAVIT_IMAGE "$HOME_DIR/image.env")' in script  # the image the stack runs, by digest: never a tag
    assert 'MEMORY=$(setting API_MEMORY "$HOME_DIR/app.env")' in script  # the api container's own limit unless --memory says otherwise
    assert '--cpus 2 --memory "$MEMORY" --memory-swap "$MEMORY"' in script  # no swap: a model that does not fit is killed, not slowed
    assert "-v metanavit_models:/root/.cache/huggingface" in script  # the stack's model volume
    assert '"$IMAGE" python - <' in script and "TIMING_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2" in script
    assert "TIMING_DEPTHS=20" in script and "TIMING_MAX_LENGTH=512" in script and "TIMING_THREADS=2" in script  # the served shape
    assert "TIMING_EXIT 137" in script
    assert "latest" not in script and "docker pull" not in script and "terraform" not in script
    assert "MEMORY=1536m" in _script(memory="1536m")
    mounted = _script(model=None, model_dir="/opt/metanavit/student")
    assert "-v /opt/metanavit/student:/models/candidate:ro" in mounted and "TIMING_MODEL=/models/candidate" in mounted
    assert "TIMING_DEPTHS=5,10,20" in _script(depths=[20, 5, 10])  # ascending: the memory peak never falls


def test_stopping_the_api_always_starts_the_stack_again():
    script = _script(stop_api=True)
    assert 'if [ "1" = "1" ]; then' in script and "docker stop $API" in script
    assert "trap restore EXIT" in script and '"$HOME_DIR/deploy.sh" --start' in script  # also when the measurement fails
    assert script.index("trap restore EXIT") < script.index("docker stop $API")
    assert 'flock --wait 600 9' in script and "flock --unlock 9" in script  # never during a deploy; released before deploy.sh --start
    beside = _script(stop_api=False)
    assert 'if [ "0" = "1" ]; then' in beside and "an upper bound" in beside


def test_only_plain_values_reach_the_hosts_shell():
    for bad in ("x; rm -rf /", "$(reboot)", "a/b c", "no-slash", "a/b/c", "`id`/x"):
        with pytest.raises(host.DeployError, match="Hugging Face id"):
            _script(model=bad)
    for bad in ("relative/path", "/opt/$(id)", "/opt/a b", "/opt/x;y"):
        with pytest.raises(host.DeployError, match="absolute path"):
            _script(model=None, model_dir=bad)
    with pytest.raises(host.DeployError, match="exactly one"):
        _script(model_dir="/opt/x")
    with pytest.raises(host.DeployError, match="exactly one"):
        _script(model=None)
    with pytest.raises(host.DeployError, match="1024m"):
        _script(memory="1g; reboot")
    with pytest.raises(host.DeployError):
        _script(depths=[0])
    with pytest.raises(host.DeployError):
        _script(max_length=8)


def test_run_commands_carry_the_script_base64_encoded():
    script = _script()
    first, second = host.commands(script)
    assert base64.b64decode(first.split()[1]).decode() == script and first.endswith("> /run/metanavit-rerank-timing.sh")
    assert second == "bash /run/metanavit-rerank-timing.sh"


class FakeAws:
    """An aws CLI that answers like EC2 and Systems Manager, and remembers what it was asked."""

    def __init__(self, output, status="Success"):
        self.calls, self.output, self.status = [], output, status

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("ec2", "describe-instances"):
            return {"Reservations": [{"Instances": [{"InstanceId": "i-0123456789abcdef0"}]}]}
        if args[:2] == ("ssm", "send-command"):
            return {"Command": {"CommandId": "cmd-1"}}
        if args[:2] == ("ssm", "list-command-invocations"):
            return {"CommandInvocations": [{"CommandId": "cmd-1"}]}
        if args[:2] == ("ssm", "get-command-invocation"):
            return {"Status": self.status, "StandardOutputContent": self.output, "StandardErrorContent": ""}
        raise AssertionError(args)


def test_the_owner_must_choose_how_to_run_beside_a_live_demo(monkeypatch, capsys):
    monkeypatch.setenv("SINGLE_HOST_NAME", "metanavit-single-host")
    aws = FakeAws("RESULT " + json.dumps(RESULT))
    assert host.main(["--model", "cross-encoder/ms-marco-MiniLM-L6-v2"], aws=aws) == 1
    assert aws.calls == [] and "choose --stop-api" in capsys.readouterr().err  # nothing was sent
    with pytest.raises(SystemExit):
        host.main(["--model", "a/b", "--stop-api", "--beside-api"], aws=aws)
    monkeypatch.delenv("SINGLE_HOST_NAME")
    assert host.main(["--model", "a/b", "--stop-api"], aws=aws) == 1 and aws.calls == []
    assert host.main(["--model", "a/b", "--print-remote-script"], aws=aws) == 0 and aws.calls == []  # printing calls nothing


def test_a_run_sends_one_shell_script_command_and_reports_the_serving_numbers(monkeypatch, capsys):
    monkeypatch.setenv("SINGLE_HOST_NAME", "metanavit-single-host")
    aws = FakeAws("image repo@sha256:abc\nRESULT " + json.dumps(RESULT) + "\n")
    assert host.main(["--model", "cross-encoder/ms-marco-MiniLM-L6-v2", "--stop-api"], aws=aws) == 0
    sent = [c for c in aws.calls if c[:2] == ("ssm", "send-command")]
    assert len(sent) == 1 and sent[0][sent[0].index("--document-name") + 1] == "AWS-RunShellScript"  # the document the deploy role may run
    assert sent[0][sent[0].index("--instance-ids") + 1] == "i-0123456789abcdef0"
    parameters = json.loads(sent[0][sent[0].index("--parameters") + 1])
    assert len(parameters["commands"]) == 2 and parameters["executionTimeout"] == ["3000"]
    out = capsys.readouterr().out
    assert "depth  20: p50 2100.0 ms, p95 2400.5 ms per rerank call over 30 calls; process peak 1010 MiB, container peak 1015 MiB" in out
    assert "memory before the reranker: 780 MiB resident (peak 850 MiB); after loading it: 880 MiB" in out


def test_a_killed_container_is_a_failed_measurement_not_a_number(monkeypatch, capsys):
    monkeypatch.setenv("SINGLE_HOST_NAME", "metanavit-single-host")
    aws = FakeAws("TIMING_EXIT 137: the container was killed\n", status="Failed")
    assert host.main(["--model", "BAAI/bge-reranker-v2-m3", "--stop-api"], aws=aws) == 1
    assert "does not fit the memory limit" in capsys.readouterr().err
    assert host.parse_result("no result here") is None


# ---- the indicative measurement on another CPU


def test_the_cpu_record_is_labelled_indicative_and_names_a_student_by_its_sha256(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt" / "student-minilm-l6"
    ckpt.mkdir(parents=True)
    runs = []

    def run(env):
        runs.append((env["TIMING_MODEL"], env["TIMING_MAX_LENGTH"], env["TIMING_DEPTHS"], env["TIMING_THREADS"], env["CUDA_VISIBLE_DEVICES"]))
        return subprocess.CompletedProcess([], 0, stdout="noise\nRESULT " + json.dumps({**RESULT, "max_length": int(env["TIMING_MAX_LENGTH"])}) + "\n", stderr="")

    from app.eval import small_rerankers as sr

    monkeypatch.setattr(cpu, "run_payload", run)
    monkeypatch.setattr(sr, "reranker_source", lambda tag, d=None: (str(ckpt), {"model": tag, "weights_sha256": "f" * 64, "student_of": "cross-encoder/ms-marco-MiniLM-L6-v2"}))
    out = tmp_path / "cpu.json"
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert cpu.main(["--models", "student-minilm-l6", "--out", str(out), "--checkpoint-dir", str(tmp_path / "ckpt")]) == 0
    assert runs == [(str(ckpt), "256", "5,10,20", "2", ""), (str(ckpt), "512", "5,10,20", "2", "")]  # a fresh process per max_length
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before  # CUDA is hidden from the measuring process only, not from this one
    assert cpu.PAYLOAD == "scripts/rerank_timing_payload.py"  # the host's own measuring program
    text = out.read_text()
    record = json.loads(text)
    assert "indicative only" in record["label"] and "scripts/time_reranker_on_host.py" in record["label"]
    assert str(tmp_path) not in text  # a student is its sha256 and its init, never a path
    rows = record["results"]
    assert [(r["tag"], r["max_length"]) for r in rows] == [("student-minilm-l6", 256), ("student-minilm-l6", 512)]
    assert rows[0]["weights_sha256"] == "f" * 64 and rows[0]["config"] == "hybrid+student-minilm-l6@bge-small/fp32-512/top20"
    assert record["method"]["depths"] == [5, 10, 20] and record["method"]["max_lengths"] == [256, 512] and record["device"]["threads"] == 2
    assert set(record["source_sha256"]) >= {"scripts/measure_rerank_cpu.py", "scripts/rerank_timing_payload.py"}


def test_a_failed_child_stops_the_cpu_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "run_payload", lambda env: subprocess.CompletedProcess([], 1, stdout="", stderr="Killed"))
    from app.eval import small_rerankers as sr

    monkeypatch.setattr(sr, "reranker_source", lambda tag, d=None: ("some/model", {"model": "some/model"}))
    monkeypatch.setattr(sr, "check_revision", lambda tag, revision: None)
    with pytest.raises(SystemExit, match="failed"):
        cpu.main(["--models", "minilm-l6", "--out", str(tmp_path / "cpu.json")])
    assert not (tmp_path / "cpu.json").exists()
