"""Score the search policy's arms on fixture v1 (M44, `make search-policy-eval`): greedy decoding, one model.

Arms, all with the same model, prompt and environment (app/rl/episode.py), on the same questions:

  policy-untrained   the base model under the prompt: it decides what to search and when to answer;
  single-shot-rag    one search with the question itself, then the base model answers (a cap of one search);
  policy-<run>       the base model with a trained LoRA adapter (--adapter name=directory, repeatable).

--split held_out is the test: the `test` half of app.eval.rerank_arms.split_questions, the half rrf_sweep holds
out. --split tune is the other half, which checkpoint selection already used; it is never the reported result.

Each arm is written as one file of episodes with their scores under --out-dir; app/rl/results.py turns those files
and the training runs into bench/results/search_policy.json and its sidecars. Needs a GPU and the `rl` extra.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import load_gold
from app.eval.provenance import command_line, device_info, git_sha, git_state, hf_revision, run_provenance, sha256_file
from app.eval.rerank_arms import split_questions
from app.rl.episode import INFO_CHUNKS, MAX_NEW_TOKENS, MAX_SEARCHES, corpus_env, greedy_episodes, prompt_sha256
from app.rl.train import EMBEDDER, MODEL

ROOT = Path(__file__).resolve().parents[2]
UNTRAINED, SINGLE_SHOT = "policy-untrained", "single-shot-rag"
BASE_ARMS = {UNTRAINED: False, SINGLE_SHOT: True}  # arm -> forced first search


def adapter_files(path: Path) -> dict[str, str]:
    """sha256 of every file of a saved adapter that defines it (weights and config)."""
    return {name: sha256_file(Path(path) / name) for name in ("adapter_config.json", "adapter_model.safetensors")}


def library_versions() -> dict:
    import peft
    import transformers

    return {"transformers": transformers.__version__, "peft": peft.__version__}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", default="held_out", choices=["held_out", "tune"])
    p.add_argument("--arms", default=f"{UNTRAINED},{SINGLE_SHOT}", help="base-model arms to run; empty for none")
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=DIR", help="a trained adapter: the arm is policy-NAME")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--embedder", default=EMBEDDER)
    p.add_argument("--max-searches", type=int, default=MAX_SEARCHES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"])
    args = p.parse_args(argv)

    import torch

    from app.retrieval.embedders import SentenceTransformerEmbedder
    from app.rl.policy import HFPolicy

    started = git_state(ROOT)
    files_root = ROOT / "bench" / "corpus" / "files"
    manifest = load_manifest(ROOT / "bench" / "corpus" / "MANIFEST.json")
    problems = verify_manifest(files_root, manifest)
    if problems:
        raise RuntimeError("corpus drift:\n" + "\n".join(problems))
    gold_path = ROOT / "bench" / "gold" / "questions.jsonl"
    gold = load_gold(gold_path)
    tune_ids, test_ids = split_questions(gold)
    questions = [gold[i] for i in (test_ids if args.split == "held_out" else tune_ids)]
    embedder = SentenceTransformerEmbedder(args.embedder)
    env = corpus_env(files_root, embedder)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def header(policy) -> dict:
        return {
            "git_sha": git_sha(ROOT), **run_provenance(root=ROOT, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command_line(), "device": device_info(args.device), "libraries": library_versions(),
            "gpu": torch.cuda.get_device_name(0) if policy.device.type == "cuda" else None, "precision": policy.precision,
            "model": {"name": args.model, "revision": hf_revision(args.model)},
            "embedder": {"name": args.embedder, "revision": hf_revision(args.embedder), **embedder.provenance()},
            "corpus_sha256": manifest["aggregate_sha256"], "gold_sha256": sha256_file(gold_path), "n_files": manifest["n_files"],
            "n_chunks": len(env.index.chunks), "split": args.split, "question_ids": [q.id for q in questions],
            "decoding": "greedy (argmax; no temperature, top-k, top-p or repetition penalty)",
            "prompt_sha256": prompt_sha256(args.max_searches), "max_searches": args.max_searches, "max_new_tokens_per_piece": MAX_NEW_TOKENS,
            "information_chunks_per_search": INFO_CHUNKS,
        }

    def run(policy, arm: str, forced: bool, extra: dict) -> None:
        t0 = time.time()
        rows = greedy_episodes(policy, questions, env, forced_first_search=forced, max_searches=args.max_searches,
                               encode_queries=embedder.encode_queries)
        blob = {"arm": arm, **extra, **header(policy), "max_searches": 1 if forced else args.max_searches,
                "prompt_sha256": prompt_sha256(1 if forced else args.max_searches), "seconds": round(time.time() - t0, 1), "rows": rows}
        name = arm.replace("/", "_") + ".json"
        (args.out_dir / name).write_text(json.dumps(blob, indent=1) + "\n")
        scores = [rows[q.id]["score"] for q in questions]
        n = len(scores)
        print(f"{arm} [{args.split}, n={n}]: em {sum(s['em'] for s in scores) / n:.4f} em_bounded {sum(s['em_bounded'] for s in scores) / n:.4f} "
              f"reward {sum(s['reward'] for s in scores) / n:+.4f} searches {sum(s['n_search'] for s in scores) / n:.2f} "
              f"uncited_or_empty {sum(s['uncited_or_empty'] for s in scores) / n:.3f} ({blob['seconds']} s) -> {name}", flush=True)

    base_arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in base_arms if a not in BASE_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; the base-model arms are {sorted(BASE_ARMS)}")
    if base_arms:
        policy = HFPolicy(args.model, device=args.device, precision=args.precision)
        for arm in base_arms:
            run(policy, arm, BASE_ARMS[arm], {"kind": "single_shot" if BASE_ARMS[arm] else "untrained", "adapter": None})
        del policy
        torch.cuda.empty_cache()
    for spec in args.adapter:
        name, _, directory = spec.rpartition("=")  # grpo/seed=0=runs/grpo-seed0/best
        directory = Path(directory)
        policy = HFPolicy(args.model, device=args.device, adapter=directory, precision=args.precision)
        step = json.loads((directory / "step.json").read_text()) if (directory / "step.json").exists() else None
        run(policy, f"policy-{name}", False, {"kind": "trained", "adapter": {"name": name, "sha256": adapter_files(directory), "selected": step}})
        del policy
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
