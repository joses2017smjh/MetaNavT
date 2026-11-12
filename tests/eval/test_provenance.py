import json
import re
import sys
from pathlib import Path

import pytest

import app.eval.provenance as prov
from app.eval.provenance import (REPO_ROOT, command_line, device_info, hf_revision, model_provenance, ollama_model_provenance,
                                 sanitize_command, sanitize_path, scrub, scrub_names)


def test_hf_revision_reads_refs_main_from_the_cache(tmp_path, monkeypatch):
    slug = tmp_path / "models--BAAI--bge-small-en-v1.5"
    (slug / "refs").mkdir(parents=True)
    (slug / "refs" / "main").write_text("5c38ec7c405ec4b44bffa4de0e3f7d5d7b3d1c7f\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert hf_revision("BAAI/bge-small-en-v1.5") == "5c38ec7c405ec4b44bffa4de0e3f7d5d7b3d1c7f"
    assert hf_revision("BAAI/not-cached") is None


def test_model_provenance_labels_non_neural_backends():
    assert model_provenance("hash", "embedder") == {"role": "embedder", "model": "hash", "revision": None, "neural": False}
    p = model_provenance("st:BAAI/bge-small-en-v1.5", "embedder")
    assert p["model"] == "BAAI/bge-small-en-v1.5" and p["neural"] is True and "revision" in p


def test_an_ollama_model_is_pinned_by_the_digest_its_tag_resolved_to():
    """A43: an Ollama tag names weights that can change under it; the manifest digest /api/tags reports pins them."""
    tags = {"qwen2.5:7b": {"name": "qwen2.5:7b", "digest": "845dbda0ea48", "size": 4683087332,
                           "details": {"family": "qwen2", "quantization_level": "Q4_K_M", "parameter_size": "7.6B"}},
            "llama3:latest": {"name": "llama3:latest", "digest": "365c0bd3c000"}}
    block = ollama_model_provenance("qwen2.5:7b", "judge", tags)
    assert block == {"role": "judge", "model": "qwen2.5:7b", "revision": "845dbda0ea48", "neural": True, "backend": "ollama",
                     "family": "qwen2", "quantization": "Q4_K_M", "parameter_size": "7.6B", "size_bytes": 4683087332}
    assert ollama_model_provenance("llama3", "judge", tags)["revision"] == "365c0bd3c000"  # Ollama reads a bare name as :latest
    assert ollama_model_provenance("qwen2.5:0.5b", "generator", tags)["revision"] is None  # not on the server: no pin


def test_device_info_works_without_cuda():
    info = device_info("cpu")
    assert info["device"] == "cpu" and "python" in info and "torch" in info


# ---- the sanitizer: no absolute path, user name or host name in a recorded command or machine description

USER, HOST = "jdoe42", "node-17.cluster.example.edu"


@pytest.fixture
def identity(monkeypatch):
    """Run as user jdoe42 on host node-17.cluster.example.edu."""
    for key in ("USER", "LOGNAME"):
        monkeypatch.setenv(key, USER)
    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.setenv("HOME", f"/home/{USER}")
    monkeypatch.setattr(prov.platform, "node", lambda: HOST)
    monkeypatch.setattr(prov.socket, "gethostname", lambda: HOST)


ARGV = [
    str(REPO_ROOT / "app" / "eval" / "harness.py"), "--frontier",
    "--out", str(REPO_ROOT / "bench" / "results" / "frontier.json"),
    f"--trec-dir=/scratch/{USER}/trec_fiqa", "--url", "http://node-17:8000", "--note", f"{USER}-run",
]
ENV = {"BGE_ALLOW_DOWNLOAD": "1", "HF_HOME": f"/nfs/hpc/share/{USER}/hf_cache", "CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin"}
EXPECTED = ("BGE_ALLOW_DOWNLOAD=1 HF_HOME=<path>/hf_cache CUDA_VISIBLE_DEVICES=0 python -m app.eval.harness --frontier "
            "--out bench/results/frontier.json --trec-dir=<path>/trec_fiqa --url http://<host>:8000 --note <user>-run")


def _clean(text: str) -> bool:
    return USER not in text and "node-17" not in text and not re.search(r"(^|[\s=])[/~]", text)


def test_sanitize_command_keeps_module_and_arguments_without_paths_user_or_host(identity):
    cmd = sanitize_command(ARGV, ENV)
    assert cmd == EXPECTED and _clean(cmd)


def test_command_line_is_sanitized(identity, monkeypatch):
    monkeypatch.setattr(sys, "argv", ARGV)
    for key in ("BGE_ALLOW_DOWNLOAD", "HF_HOME", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(key, ENV[key])
    assert command_line() == EXPECTED


def test_sanitize_path_is_repository_relative_or_a_placeholder(identity, tmp_path):
    assert sanitize_path(str(REPO_ROOT / "bench" / "results" / "main.json")) == "bench/results/main.json"
    assert sanitize_path(str(REPO_ROOT)) == "."
    link = tmp_path / "checkout"
    link.symlink_to(REPO_ROOT)  # the same repository reached through a symlinked prefix
    assert sanitize_path(str(link / "bench" / "results" / "main.json")) == "bench/results/main.json"
    assert sanitize_path(f"/nfs/{USER}/models/bge") == "<path>/bge"
    assert sanitize_path("~/cache/x.json") == "<path>/x.json"
    assert sanitize_path(f"/data/{USER}") == "<path>/<user>"
    assert sanitize_path("bench/results/main.json") == "bench/results/main.json"  # relative: left as written


def test_scrub_reaches_nested_values_and_leaves_urls_alone(identity):
    health = {"indexing": {"data_dir": f"/data/{USER}/corpus", "n": 3}, "dsn": f"postgresql://{USER}@127.0.0.1:5432/db",
              "url": "http://127.0.0.1:8000/api/retrieve/", "log": [f"wrote /tmp/{USER}/x.log on {HOST}", None, 1.5]}
    assert scrub(health) == {"indexing": {"data_dir": "<path>/corpus", "n": 3}, "dsn": "postgresql://<user>@127.0.0.1:5432/db",
                             "url": "http://127.0.0.1:8000/api/retrieve/", "log": ["wrote <path>/x.log on <host>", None, 1.5]}


def test_scrub_names_replaces_user_and_host_names_and_keeps_paths_as_written(identity):
    # for text that must stay as a model saw or wrote it, such as tool results and arguments
    value = {"arguments": {"path": f"/data/{USER}/x.yaml"}, "result": ["#!/bin/bash", f"ran on {HOST}", 2]}
    assert scrub_names(value) == {"arguments": {"path": "/data/<user>/x.yaml"}, "result": ["#!/bin/bash", "ran on <host>", 2]}


def test_device_info_records_no_host_name(identity):
    info = device_info("cpu")
    assert "hostname" not in info and "node-17" not in json.dumps(info)


def test_generic_account_names_are_not_replaced(monkeypatch):
    monkeypatch.setenv("USER", "runner")
    monkeypatch.setenv("LOGNAME", "runner")
    monkeypatch.setenv("HOME", "/home/runner")
    assert sanitize_command(["app/eval/harness.py", "--runner-notes", "runner"], {}) == "python -m app.eval.harness --runner-notes runner"
