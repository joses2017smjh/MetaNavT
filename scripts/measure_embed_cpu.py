"""What an adapted embedder costs on a CPU, against the bge-small it was trained from (M48). INDICATIVE.

    python scripts/measure_embed_cpu.py --checkpoint-dir <dir> [--data-dir <BEIR dir>] [--threads 2] [--rounds 5]
                                        [--out bench/results/adapted_embedders_cpu.json]

The claim is structural, and is checked first for every trained bge-small arm against BAAI/bge-small-en-v1.5 at its
pinned revision. By function, not file bytes (transformers 5.17 rewrites the saved tokenizer and config files): the
architecture keys of config.json; the module types, the Pooling config, max_seq_length and do_lower_case; the token ids
both produce on the declared text set (the 71 fixture chunks and the 1,084 test queries, text only); the parameter
names, shapes and count. Equal everywhere: the same computation per query, so the same cost.

The timing only checks that claim on one machine. In one fresh process, CUDA hidden, `--threads` threads: fixture v1's
136 questions x `--rounds` rounds, each question embedded by base (A), the arm, and base again (A'), the order rotated
each round; per question the median of its rounds; then the arm/base ratio and the A'/A ratio of the summed medians,
each with a paired percentile bootstrap 95% interval over the questions (2000 resamples, seed 0). "unchanged" when the
arm/base interval lies inside adapted_embedders.CPU_BAND; "inconclusive (noisy node)" when A'/A's does not, whatever
the arm's. Peak memory: each model alone in a fresh process (load, embed the 136 questions and the 71 chunks), VmHWM
and VmRSS from /proc/self/status, MiB.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/measure_embed_cpu.py"
ARCH_KEYS = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "max_position_embeddings",
             "vocab_size", "hidden_act", "layer_norm_eps", "type_vocab_size", "position_embedding_type")
INDICATIVE = "indicative only: one HPC CPU, not the demo's host; the claim is the structural identity, the timing only checks it"


def memory_mib() -> dict[str, float]:
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
            out[key] = round(int(value.split()[0]) / 1024.0, 1)
    return out


def ratio_ci(num, den, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """sum(num) / sum(den) with a paired percentile bootstrap 95% interval over the items (one index matrix)."""
    import numpy as np

    from app.eval.stats import resample_indices

    a, b = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    idx = resample_indices(len(a), n_boot=n_boot, seed=seed)
    ratios = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {"ratio": round(float(a.sum() / b.sum()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def timing_word(arm: dict, noise: dict, band: tuple[float, float]) -> str:
    """The declared reading: noise first (A'/A outside the band), then the arm's interval against the band."""
    inside = lambda block: band[0] <= block["lo"] and block["hi"] <= band[1]  # noqa: E731
    if not inside(noise):
        return "inconclusive (noisy node)"
    return "unchanged" if inside(arm) else "not shown unchanged"


def identity(base, arm, texts: list[str]) -> dict:
    """The structural identity of two loaded SentenceTransformers, by function (see the module docstring)."""
    cfg_a, cfg_b = base[0].auto_model.config.to_dict(), arm[0].auto_model.config.to_dict()
    arch = {k: [cfg_a.get(k), cfg_b.get(k)] for k in ARCH_KEYS}
    modules = [[type(m).__name__ for m in base], [type(m).__name__ for m in arm]]
    pooling = [base[1].get_config_dict(), arm[1].get_config_dict()]
    enc = [[base.max_seq_length, bool(base[0].do_lower_case)], [arm.max_seq_length, bool(arm[0].do_lower_case)]]
    ids_equal = 0
    for start in range(0, len(texts), 256):
        part = texts[start: start + 256]
        fa, fb = base.preprocess(part), arm.preprocess(part)
        same = fa["input_ids"].shape == fb["input_ids"].shape and bool((fa["input_ids"] == fb["input_ids"]).all()) \
            and bool((fa["attention_mask"] == fb["attention_mask"]).all())
        ids_equal += len(part) if same else 0
    params = [[(n, list(p.shape)) for n, p in base.named_parameters()], [(n, list(p.shape)) for n, p in arm.named_parameters()]]
    out = {"architecture_keys": {k: v[0] for k, v in arch.items()}, "architecture_equal": all(v[0] == v[1] for v in arch.values()),
           "modules": modules[0], "modules_equal": modules[0] == modules[1], "pooling": pooling[0], "pooling_equal": pooling[0] == pooling[1],
           "max_seq_length_and_do_lower_case": enc[0], "encoding_equal": enc[0] == enc[1],
           "token_ids": {"texts": len(texts), "equal": ids_equal}, "token_ids_equal": ids_equal == len(texts),
           "parameters": {"tensors": len(params[0]), "count": int(sum(p.numel() for p in base.parameters()))},
           "parameters_equal": params[0] == params[1] and sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in arm.parameters())}
    out["identical"] = all(out[k] for k in ("architecture_equal", "modules_equal", "pooling_equal", "encoding_equal", "token_ids_equal", "parameters_equal"))
    return out


def _texts(data_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """(the fixture's 71 chunks, the 1,084 test queries' text, fixture v1's 136 questions)."""
    from app.eval.distill_queries import benchmark_queries, fixture_documents
    from app.eval.gold import load_gold

    chunks = [t for _, t in fixture_documents(ROOT)]
    tests = [t["text"] for t in benchmark_queries(ROOT, data_dir)]
    questions = [q.question for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")]
    return chunks, tests, questions


def _set_threads(n: int) -> int:
    import torch

    torch.set_num_threads(n)
    return torch.get_num_threads()


def _load(name: str, checkpoint_dir: str | None):
    from app.eval import adapted_embedders as ae

    emb, record = ae.resolve_embedder(name, checkpoint_dir, ROOT)
    return emb, record


def worker(args) -> int:
    """One fresh process: `timing` (base, arm, base again per question) or `memory` (one model alone)."""
    threads = _set_threads(args.threads)
    from app.eval.gold import load_gold

    questions = [q.question for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")]
    if args.worker == "memory":
        before = memory_mib()
        emb, _ = _load(args.model, args.checkpoint_dir)
        loaded = memory_mib()
        from app.eval.distill_queries import fixture_documents

        emb.encode(questions)
        emb.encode([t for _, t in fixture_documents(ROOT)])
        print("RESULT " + json.dumps({"model": args.model, "start": before, "loaded": loaded, "after_encoding": memory_mib()}), flush=True)
        return 0
    base, _ = _load("st:BAAI/bge-small-en-v1.5", None)
    arm, _ = _load(args.model, args.checkpoint_dir)
    for q in questions[:3]:  # warm-up
        base.encode([q])
        arm.encode([q])
    order = [("A", base), ("arm", arm), ("A2", base)]
    times: dict[str, list[list[float]]] = {k: [[] for _ in questions] for k, _ in order}
    for r in range(args.rounds):
        rotated = order[r % 3:] + order[: r % 3]
        for i, q in enumerate(questions):
            for key, model in rotated:
                t0 = time.perf_counter()
                model.encode([q])
                times[key][i].append((time.perf_counter() - t0) * 1000.0)
    import numpy as np

    med = {k: [float(np.median(v)) for v in rows] for k, rows in times.items()}
    print("RESULT " + json.dumps({"model": args.model, "medians_ms": med, "rounds": args.rounds, "threads": threads}), flush=True)
    return 0


def _child(args, extra: list[str]) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
           "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, str(ROOT / SCRIPT), "--threads", str(args.threads), "--rounds", str(args.rounds), *extra]
    if args.checkpoint_dir:
        cmd += ["--checkpoint-dir", str(args.checkpoint_dir)]
    done = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if done.returncode != 0 or not lines:
        from app.eval.provenance import scrub

        raise SystemExit(f"{extra} failed:\n{scrub(done.stderr[-3000:])}")
    return json.loads(lines[-1][len("RESULT "):])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "adapted_embedders_cpu.json")
    p.add_argument("--checkpoint-dir", default=None, help="where the arms' checkpoints are (default: $ADAPTED_EMBEDDER_CHECKPOINTS)")
    p.add_argument("--data-dir", type=Path, default=ROOT / "bench" / "external", help="the BEIR datasets (the test queries' text, for the token-id check)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads (the host has 2 vCPUs)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--arms", default=None, help="comma-separated arms (default: every trained bge-small arm)")
    p.add_argument("--worker", choices=["timing", "memory"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if args.worker:
        return worker(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from app.eval import adapted_embedders as ae
    from app.eval.provenance import command_line, device_info, git_sha, git_state, run_provenance, scrub

    started = git_state(ROOT)
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else [t for t in ae.trained_arms(ROOT) if t in ae.BGE_SMALL_ARMS]
    chunks, tests, questions = _texts(args.data_dir)
    _set_threads(args.threads)
    base, base_record = _load("st:BAAI/bge-small-en-v1.5", None)
    rows = []
    for tag in arms:
        emb, record = _load(ae.embedder_name(tag), args.checkpoint_dir)
        ident = identity(base._model, emb._model, chunks + tests)
        timing = _child(args, ["--worker", "timing", "--model", ae.embedder_name(tag)])
        med = timing["medians_ms"]
        arm_ratio, noise = ratio_ci(med["arm"], med["A"]), ratio_ci(med["A2"], med["A"])
        import numpy as np

        rows.append({"arm": tag, "weights_sha256": record.get("weights_sha256"), "structural_identity": ident,
                     "timing": {"p50_ms": {k: round(float(np.median(v)), 3) for k, v in med.items()}, "arm_over_base": arm_ratio,
                                "base_again_over_base": noise, "reading": timing_word(arm_ratio, noise, ae.CPU_BAND), "n_questions": len(med["A"]),
                                "rounds": timing["rounds"], "threads": timing["threads"]},
                     "per_question_medians_ms": med})
        print(tag, "identical" if ident["identical"] else "NOT IDENTICAL", rows[-1]["timing"]["reading"], arm_ratio, noise, flush=True)
    memory = {name: _child(args, ["--worker", "memory", "--model", name]) for name in ["st:BAAI/bge-small-en-v1.5", *[ae.embedder_name(t) for t in arms]]}
    blob = {"git_sha": git_sha(ROOT), **run_provenance(root=ROOT, started=started), "command": command_line(),
            "device": {**device_info("cpu"), "threads": args.threads}, "label": INDICATIVE, "base": {"model": "BAAI/bge-small-en-v1.5", **base_record},
            "band": list(ae.CPU_BAND),
            "method": {"identity": "config.json architecture keys, module types, Pooling config, max_seq_length and do_lower_case, token ids on the 71 fixture chunks and the 1,084 test queries (text only), parameter names, shapes and count",
                       "timing": f"one fresh process, CUDA hidden, {args.threads} threads: fixture v1's {len(questions)} questions x {args.rounds} rounds, base (A), arm, base again (A') per question, the order rotated each round; per-question medians; ratio of summed medians with a paired percentile bootstrap 95% interval (2000, seed 0)",
                       "reading": "unchanged when the arm/base interval lies inside the band; inconclusive (noisy node) when the A'/A interval does not",
                       "memory": "each model alone in a fresh process: load, embed the 136 questions and the 71 chunks; VmHWM and VmRSS, MiB"},
            "results": rows, "memory_mib": memory,
            "not_shown": ["the demo host's latency: another CPU; the host serves bge-small with LlamaIndex's query instruction",
                          "quality: the GPU rows in adapted_embedders.json and the BEIR files"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scrub(blob), indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
