"""M48: scripts/measure_embed_cpu.py, the indicative CPU cost of an adapted embedder against the bge-small it was
trained from. The claim is the structural identity (by function: architecture keys, modules, pooling, encoding, token
ids, parameter names and shapes); the timing only checks it on one machine, read by a declared band with a noise check."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu = _load("measure_embed_cpu")
BAND = (0.90, 1.10)


def test_the_band_is_the_declared_one():
    from app.eval import adapted_embedders as ae

    assert ae.CPU_BAND == BAND


def test_a_noisy_node_is_inconclusive_whatever_the_arm_shows():
    inside, outside = {"lo": 0.95, "hi": 1.05}, {"lo": 0.85, "hi": 1.02}
    assert cpu.timing_word(inside, inside, BAND) == "unchanged"
    assert cpu.timing_word(inside, outside, BAND) == "inconclusive (noisy node)"
    assert cpu.timing_word(outside, outside, BAND) == "inconclusive (noisy node)"
    assert cpu.timing_word(outside, inside, BAND) == "not shown unchanged"
    assert cpu.timing_word({"lo": 0.90, "hi": 1.10}, inside, BAND) == "unchanged"  # the band's ends are inside it


def test_the_ratio_interval_is_paired_over_the_questions():
    base = np.linspace(10.0, 30.0, 136)
    same = cpu.ratio_ci(base, base)
    assert same == {"ratio": 1.0, "lo": 1.0, "hi": 1.0}  # a paired interval of a ratio with itself has no width
    slower = cpu.ratio_ci(base * 1.2, base)
    assert slower["ratio"] == 1.2 and slower["lo"] == pytest.approx(1.2) and slower["hi"] == pytest.approx(1.2)
    rng = np.random.default_rng(0)
    noisy = cpu.ratio_ci(base * rng.uniform(0.8, 1.2, 136), base)
    assert noisy["lo"] < noisy["ratio"] < noisy["hi"]
    assert noisy == cpu.ratio_ci(base * np.random.default_rng(0).uniform(0.8, 1.2, 136), base)  # seed 0, 2000 resamples: reproducible


class _Param:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def numel(self):
        return int(np.prod(self.shape))


class _Config:
    def __init__(self, **kw):
        self.kw = kw

    def to_dict(self):
        return dict(self.kw)


class _Module:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get_config_dict(self):
        return dict(self.config)


class Transformer(_Module):
    pass


class Pooling(_Module):
    pass


class Normalize(_Module):
    pass


class _Model:
    """Just enough of a SentenceTransformer for identity(): modules, config, tokenizer output, parameters."""

    def __init__(self, *, layers=12, lower=True, vocab_shift=0, pooling="cls", modules=("Transformer", "Pooling", "Normalize")):
        cfg = {k: 1 for k in cpu.ARCH_KEYS} | {"num_hidden_layers": layers, "model_type": "bert"}
        kinds = {"Transformer": Transformer, "Pooling": Pooling, "Normalize": Normalize}
        self.mods = [kinds[m](auto_model=_Module(config=_Config(**cfg)), do_lower_case=lower, config={"pooling_mode": pooling}) for m in modules]
        self.max_seq_length, self.shift = 512, vocab_shift
        self.params = [(f"layer.{i}.weight", _Param((4, 4))) for i in range(layers)]

    def __getitem__(self, i):
        return self.mods[i]

    def __iter__(self):
        return iter(self.mods)

    def preprocess(self, texts):
        ids = np.asarray([[len(t) + self.shift, 1] for t in texts])
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}

    def named_parameters(self):
        return iter(self.params)

    def parameters(self):
        return iter(p for _, p in self.params)


def test_structural_identity_is_by_function_and_names_what_differs():
    texts = [f"text {i}" for i in range(300)]
    same = cpu.identity(_Model(), _Model(), texts)
    assert same["identical"] and same["token_ids"] == {"texts": 300, "equal": 300}
    assert same["parameters"] == {"tensors": 12, "count": 12 * 16}
    for other, failed in ((_Model(layers=6), "architecture_equal"), (_Model(lower=False), "encoding_equal"),
                          (_Model(vocab_shift=1), "token_ids_equal"), (_Model(pooling="mean"), "pooling_equal"),
                          (_Model(modules=("Transformer", "Pooling")), "modules_equal")):
        got = cpu.identity(_Model(), other, texts)
        assert not got["identical"] and not got[failed], failed


class _Emb:
    def __init__(self, name):
        self._model, self.name = name, name


def test_the_cpu_record_is_indicative_names_an_arm_by_sha256_and_reads_the_band(tmp_path, monkeypatch):
    from app.eval import adapted_embedders as ae

    arm = ae.BAR_ARM
    med = {"A": [10.0] * 136, "arm": [10.2] * 136, "A2": [10.1] * 136}
    children = []

    def child(args, extra):
        children.append(extra)
        if extra[1] == "timing":
            return {"model": extra[3], "medians_ms": med, "rounds": args.rounds, "threads": args.threads}
        return {"model": extra[3], "start": {"VmRSS": 100.0}, "loaded": {"VmRSS": 300.0}, "after_encoding": {"VmRSS": 320.0, "VmHWM": 330.0}}

    loads = []

    def load(name, checkpoint_dir):
        loads.append((name, checkpoint_dir))
        record = {"weights_sha256": "a" * 64, "model": arm} if name.startswith("ft:") else {"pinned_revision": ae.INITS["bge-small"]["revision"]}
        return _Emb(name), record

    monkeypatch.setattr(cpu, "_child", child)
    monkeypatch.setattr(cpu, "_load", load)
    monkeypatch.setattr(cpu, "_set_threads", lambda n: n)
    monkeypatch.setattr(cpu, "_texts", lambda data_dir: (["chunk"] * 71, ["query"] * 1084, ["question"] * 136))
    monkeypatch.setattr(cpu, "identity", lambda a, b, texts: {"identical": a == "st:BAAI/bge-small-en-v1.5" and b == f"ft:{arm}", "token_ids": {"texts": len(texts)}})
    out = tmp_path / "cpu.json"
    ckpt = tmp_path / "secret-checkpoints"
    assert cpu.main(["--arms", arm, "--checkpoint-dir", str(ckpt), "--out", str(out), "--rounds", "5"]) == 0
    text = out.read_text()
    record = json.loads(text)
    assert str(tmp_path) not in text  # an arm is its sha256, never a path
    assert "indicative only" in record["label"] and record["band"] == list(BAND) and record["device"]["threads"] == 2
    row = record["results"][0]
    assert row["arm"] == arm and row["weights_sha256"] == "a" * 64 and row["structural_identity"]["identical"]
    assert row["structural_identity"]["token_ids"]["texts"] == 71 + 1084  # the declared text set: the fixture chunks and every test query
    t = row["timing"]
    assert t["arm_over_base"]["ratio"] == 1.02 and t["base_again_over_base"]["ratio"] == 1.01 and t["reading"] == "unchanged"
    assert t["n_questions"] == 136 and t["rounds"] == 5 and t["threads"] == 2
    assert [e[1] for e in children] == ["timing", "memory", "memory"]  # each memory figure from its own fresh process
    assert set(record["memory_mib"]) == {"st:BAAI/bge-small-en-v1.5", f"ft:{arm}"}
    assert record["not_shown"] and any("host" in s for s in record["not_shown"])


def test_the_make_target_passes_the_beir_directory_for_the_token_id_texts():
    make = (ROOT / "Makefile").read_text()
    line = next(x for x in make.splitlines() if "scripts/measure_embed_cpu.py" in x and "$(PYTHON)" in x)
    assert "--data-dir $(BEIR_DATA_DIR)" in line and "--threads 2" in line and "CUDA_VISIBLE_DEVICES=" in line
