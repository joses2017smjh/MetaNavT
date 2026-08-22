"""Materialize the frozen corpus files + gold JSONL from registry.py."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.corpus.registry import (
    ABLATIONS,
    ARCHIVED,
    BASELINE_RMSE,
    RUNS,
    Run,
    current_runs,
    dinov2_fusion_on,
    lowest_rmse_run,
)
from app.eval.gold import GoldQuestion, write_gold
from app.eval.hashing import content_hash

FUSION_SRC = '''"""Stereo fusion module.

When fusion is on, per-view DINOv2 (or backbone) features are concatenated
and passed through a 1x1 projection before the depth head. When fusion is off,
only the reference view is used.

This is the module referenced by configs as `fusion: true|false`.
"""


def project(features, fusion_on: bool):
    if not fusion_on:
        return features[0]
    import torch
    return torch.cat(features, dim=1)
'''

DINOV2_SRC = '''"""DINOv2 encoder wrapper used by runs 45-49, 52, 53, 55."""

MODEL_NAME = "dinov2_vitb14"

def load_encoder(freeze_backbone: bool = True):
    """Load DINOv2 ViT-B/14. Learning rate is set in the run config, not here."""
    return MODEL_NAME
'''

TRELLIS_SRC = '''"""Blender trellis-wire helper.

Renders used the trellis wire overlay when `render.trellis: true` in the run config.
Find the Blender render with the trellis wires by searching this flag.
"""

TRELLIS_DEFAULT = True
'''

README = """# Frozen research corpus (demo)

Snapshot of a heterogeneous experiment tree: YAML configs, Slurm scripts,
`.out` logs, ablation CSVs, paper drafts, Python modules, checkpoint metadata.

Do not edit files here by hand. Rebuild with `python -m bench.corpus.build`.
Gold answers are generated from `bench/corpus/registry.py`.
"""

PAPER_V1 = """# Depth fusion draft v1 (superseded)

We find that stereo fusion does **not** help. The fusion-off DINOv2 run (run 45)
matches fusion-on within noise, so we recommend disabling fusion.

Val RMSE of the headline DINOv2 fusion run was previously reported as 0.0601
(archived config). This draft is stale.
"""

PAPER_V2 = """# Depth fusion draft v2 (current)

Stereo fusion **does** help. The current DINOv2 + fusion + 3-pair run (run 47)
reaches val RMSE 0.0521, beating the 0.0550 baseline. Bark type for that run
is birch. Learning rate is 3.0e-4.

This draft supersedes draft v1. Prefer this file for current claims.
"""


def _yaml_for(run: Run) -> str:
    return f"""# run {run.run_id} config {'(CURRENT)' if run.current else '(SUPERSEDED)'}
run_id: {run.run_id}
encoder: {run.encoder}
fusion: {str(run.fusion).lower()}
num_pairs: {run.num_pairs}
bark_type: {run.bark}
learning_rate: {run.lr}
val_rmse: {run.val_rmse}
seed: {run.seed}
baseline_rmse: {BASELINE_RMSE}
render:
  trellis: {'true' if run.run_id in (46, 47, 48) else 'false'}
checkpoint: checkpoints/run_{run.run_id:03d}.ckpt.meta.json
notes: {run.notes or 'none'}
"""


def _log_for(run: Run) -> str:
    return f"""Slurm job for run {run.run_id}
encoder={run.encoder} fusion={run.fusion} num_pairs={run.num_pairs}
bark_type={run.bark} learning_rate={run.lr}
[epoch 10] val_rmse={run.val_rmse}
finished status={'SUCCESS' if run.current else 'ARCHIVED'}
config={run.config_path}
"""


def _sbatch_for(run: Run) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=run_{run.run_id:03d}
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
python train.py --config configs/run_{run.run_id:03d}.yaml --num_pairs={run.num_pairs} --encoder {run.encoder}
"""


def _ckpt_for(run: Run) -> str:
    return json.dumps(
        {
            "run_id": run.run_id,
            "val_rmse": run.val_rmse,
            "encoder": run.encoder,
            "path": f"checkpoints/run_{run.run_id:03d}.pt",
            "produced_by": f"slurm/run_{run.run_id:03d}.sbatch",
        },
        indent=2,
    )


def write_files(root: Path) -> list[Path]:
    root = Path(root)
    written: list[Path] = []

    def dump(rel: str, content: str, mtime: float | None = None) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        written.append(path)

    dump("README.md", README, mtime=1_700_000_000)
    dump("src/fusion.py", FUSION_SRC, mtime=1_700_000_100)
    dump("src/dinov2_encoder.py", DINOV2_SRC, mtime=1_700_000_100)
    dump("src/trellis_wires.py", TRELLIS_SRC, mtime=1_700_000_100)

    # current configs get later mtimes than archives
    for run in ARCHIVED:
        dump(run.config_path, _yaml_for(run), mtime=1_699_000_000 + run.run_id)
    for run in RUNS:
        dump(run.config_path, _yaml_for(run), mtime=1_710_000_000 + run.run_id)
        dump(run.log_path, _log_for(run), mtime=1_710_100_000 + run.run_id)
        dump(run.slurm_path, _sbatch_for(run), mtime=1_710_000_000 + run.run_id)
        if run.run_id in (46, 47, 48):
            dump(run.ckpt_path, _ckpt_for(run), mtime=1_710_200_000 + run.run_id)

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["ablation", "encoder", "fusion", "val_rmse", "run_id"]
    )
    writer.writeheader()
    for row in ABLATIONS:
        writer.writerow(
            {
                "ablation": row.name,
                "encoder": row.encoder,
                "fusion": row.fusion,
                "val_rmse": row.val_rmse,
                "run_id": row.run_id,
            }
        )
    dump("results/ablation.csv", buf.getvalue(), mtime=1_710_300_000)
    dump("paper/draft_v1.md", PAPER_V1, mtime=1_699_500_000)
    dump("paper/draft_v2.md", PAPER_V2, mtime=1_711_000_000)
    return written


def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}" if lr < 1e-3 or lr in (1e-4, 3e-4, 2e-4, 5e-4, 1e-5, 1e-3) else str(lr)


def build_gold() -> list[GoldQuestion]:
    qs: list[GoldQuestion] = []
    n = 0

    def add(category: str, question: str, answer: str, paths: list[str], notes: str = "") -> None:
        nonlocal n
        n += 1
        qs.append(
            GoldQuestion(
                id=f"q{n:03d}",
                question=question,
                category=category,
                answer=answer,
                relevant_paths=paths,
                notes=notes,
            )
        )

    # --- simple factual (per-run LR, pairs, bark, encoder, rmse) ---
    for run in RUNS:
        cfg = run.config_path
        add(
            "simple_factual",
            f"what learning rate did run {run.run_id} use",
            _fmt_lr(run.lr),
            [cfg],
        )
        add(
            "simple_factual",
            f"how many stereo pairs did run {run.run_id} use",
            str(run.num_pairs),
            [cfg, run.slurm_path],
        )
        add(
            "simple_factual",
            f"what bark type did run {run.run_id} use",
            run.bark,
            [cfg],
        )
        add(
            "simple_factual",
            f"what encoder did run {run.run_id} use",
            run.encoder,
            [cfg],
        )
        add(
            "simple_factual",
            f"what was val RMSE for run {run.run_id}",
            str(run.val_rmse),
            [cfg, run.log_path],
        )

    # --- conditional ---
    dino_fusion = dinov2_fusion_on()
    lrs = sorted({_fmt_lr(r.lr) for r in dino_fusion})
    add(
        "conditional",
        "what LR did the DINOv2 runs use with fusion on",
        ", ".join(lrs),
        [r.config_path for r in dino_fusion],
    )
    add(
        "conditional",
        "which DINOv2 fusion runs beat the 0.0550 baseline",
        ", ".join(str(r.run_id) for r in dino_fusion if r.beats_baseline),
        [r.config_path for r in dino_fusion if r.beats_baseline],
    )
    three_pair_dino = [r for r in RUNS if r.encoder == "dinov2" and r.num_pairs == 3]
    add(
        "conditional",
        "what bark types appear in DINOv2 runs that used 3 stereo pairs",
        ", ".join(sorted({r.bark for r in three_pair_dino})),
        [r.config_path for r in three_pair_dino],
    )
    fusion_off_resnet = [r for r in RUNS if r.encoder == "resnet50" and not r.fusion]
    add(
        "conditional",
        "what learning rate did the ResNet50 runs use with fusion off",
        ", ".join(sorted({_fmt_lr(r.lr) for r in fusion_off_resnet})),
        [r.config_path for r in fusion_off_resnet],
    )
    clip_fusion = [r for r in RUNS if r.encoder == "clip" and r.fusion]
    add(
        "conditional",
        "what val RMSE did CLIP runs with fusion on report",
        ", ".join(str(r.val_rmse) for r in clip_fusion),
        [r.config_path for r in clip_fusion],
    )
    birch_fusion = [r for r in RUNS if r.bark == "birch" and r.fusion]
    add(
        "conditional",
        "which fusion-on birch runs exist",
        ", ".join(str(r.run_id) for r in birch_fusion),
        [r.config_path for r in birch_fusion],
    )
    add(
        "conditional",
        "what encoder is used when fusion is on and num_pairs is 1",
        ", ".join(sorted({r.encoder for r in RUNS if r.fusion and r.num_pairs == 1})),
        [r.config_path for r in RUNS if r.fusion and r.num_pairs == 1],
    )
    pine_three = [r for r in RUNS if r.bark == "pine" and r.num_pairs == 3]
    add(
        "conditional",
        "what learning rates did pine bark runs with 3 pairs use",
        ", ".join(sorted({_fmt_lr(r.lr) for r in pine_three})),
        [r.config_path for r in pine_three],
    )

    # --- comparative ---
    best = lowest_rmse_run()
    add(
        "comparative",
        "which ablation had the lowest val RMSE",
        "dino_3pair_fusion",
        ["results/ablation.csv", best.config_path],
    )
    worst = max(RUNS, key=lambda r: r.val_rmse)
    add(
        "comparative",
        "which run had the highest val RMSE",
        str(worst.run_id),
        [worst.config_path, "results/ablation.csv"],
    )
    add(
        "comparative",
        "which run had the lowest val RMSE",
        str(best.run_id),
        [best.config_path, best.log_path],
    )
    add(
        "comparative",
        "did fusion-on DINOv2 beat fusion-off DINOv2",
        "yes: 0.0521 vs 0.0568 for the headline pair (47 vs 45)",
        ["configs/run_047.yaml", "configs/run_045.yaml", "results/ablation.csv"],
    )
    add(
        "comparative",
        "which encoder has the best current val RMSE",
        best.encoder,
        [best.config_path],
    )
    add(
        "comparative",
        "compare val RMSE of run 40 and run 47",
        "0.0612 vs 0.0521",
        ["configs/run_040.yaml", "configs/run_047.yaml"],
    )
    add(
        "comparative",
        "which ablation is worse, fusion_off_resnet or fusion_on_resnet",
        "fusion_off_resnet",
        ["results/ablation.csv"],
    )
    add(
        "comparative",
        "is run 52 better than the 0.0550 baseline",
        "yes",
        ["configs/run_052.yaml"],
    )

    # --- aggregation ---
    n_three = sum(1 for r in RUNS if r.num_pairs == 3)
    add(
        "aggregation",
        "how many runs used 3 stereo pairs",
        str(n_three),
        [r.config_path for r in RUNS if r.num_pairs == 3],
    )
    n_dino = sum(1 for r in RUNS if r.encoder == "dinov2")
    add(
        "aggregation",
        "how many runs used DINOv2",
        str(n_dino),
        [r.config_path for r in RUNS if r.encoder == "dinov2"],
    )
    n_fusion = sum(1 for r in RUNS if r.fusion)
    add(
        "aggregation",
        "how many current runs have fusion on",
        str(n_fusion),
        [r.config_path for r in RUNS if r.fusion],
    )
    n_beat = sum(1 for r in RUNS if r.beats_baseline)
    add(
        "aggregation",
        "how many runs beat the 0.0550 baseline",
        str(n_beat),
        [r.config_path for r in RUNS if r.beats_baseline],
    )
    add(
        "aggregation",
        "how many unique bark types appear in current runs",
        str(len({r.bark for r in RUNS})),
        [r.config_path for r in RUNS],
    )
    add(
        "aggregation",
        "what is the average num_pairs across current runs",
        f"{sum(r.num_pairs for r in RUNS) / len(RUNS):.2f}",
        [r.config_path for r in RUNS],
    )
    add(
        "aggregation",
        "how many CLIP runs are there",
        str(sum(1 for r in RUNS if r.encoder == "clip")),
        [r.config_path for r in RUNS if r.encoder == "clip"],
    )
    add(
        "aggregation",
        "how many ablations are listed",
        str(len(ABLATIONS)),
        ["results/ablation.csv"],
    )
    add(
        "aggregation",
        "how many current runs used oak bark",
        str(sum(1 for r in RUNS if r.bark == "oak")),
        [r.config_path for r in RUNS if r.bark == "oak"],
    )
    add(
        "aggregation",
        "how many runs used a learning rate of 3e-4",
        str(sum(1 for r in RUNS if abs(r.lr - 3e-4) < 1e-12)),
        [r.config_path for r in RUNS if abs(r.lr - 3e-4) < 1e-12],
    )

    # --- multi-hop ---
    add(
        "multi_hop",
        "which bark type was used in the run that beat the 0.0550 baseline with the lowest RMSE, and where is its config",
        f"{best.bark}; {best.config_path}",
        [best.config_path, best.log_path],
    )
    add(
        "multi_hop",
        "which slurm script launched the lowest-RMSE run and what encoder did it use",
        f"{best.slurm_path}; {best.encoder}",
        [best.slurm_path, best.config_path],
    )
    add(
        "multi_hop",
        "what checkpoint was produced by the best current run",
        best.ckpt_path,
        [best.config_path, best.ckpt_path],
    )
    add(
        "multi_hop",
        "what does the fusion module do in the run that used 3 pairs and birch bark with DINOv2",
        "concatenates per-view features and projects with a 1x1 when fusion is on",
        ["src/fusion.py", "configs/run_047.yaml"],
    )
    add(
        "multi_hop",
        "which paper draft reports the current RMSE of run 47 and what is that RMSE",
        "paper/draft_v2.md; 0.0521",
        ["paper/draft_v2.md", "configs/run_047.yaml"],
    )
    r46 = next(r for r in RUNS if r.run_id == 46)
    add(
        "multi_hop",
        "the run that first beat the baseline with 2 pairs: bark type and config path",
        f"{r46.bark}; {r46.config_path}",
        [r46.config_path],
    )
    add(
        "multi_hop",
        "where is the ablation table that lists fusion_off_dino and what RMSE does it show",
        "results/ablation.csv; 0.0568",
        ["results/ablation.csv"],
    )
    add(
        "multi_hop",
        "which source file documents trellis wires and which runs enabled them",
        "src/trellis_wires.py; 46, 47, 48",
        ["src/trellis_wires.py", "configs/run_046.yaml", "configs/run_047.yaml", "configs/run_048.yaml"],
    )

    # --- staleness ---
    add(
        "staleness",
        "what's the current learning rate for the DINOv2 run 47",
        "3e-4",
        ["configs/run_047.yaml"],
        notes="archive has 1e-5; gold target is current chunk only",
    )
    add(
        "staleness",
        "what is the current val RMSE for run 47",
        "0.0521",
        ["configs/run_047.yaml", "logs/run_047.out"],
    )
    add(
        "staleness",
        "does the current paper draft say fusion helps",
        "yes",
        ["paper/draft_v2.md"],
    )
    add(
        "staleness",
        "what is the current learning rate for run 46",
        "1e-4",
        ["configs/run_046.yaml"],
    )
    add(
        "staleness",
        "is fusion currently on for run 48",
        "yes",
        ["configs/run_048.yaml"],
    )
    add(
        "staleness",
        "what learning rate should I use if I resume the DINOv2 3-pair birch run",
        "3e-4",
        ["configs/run_047.yaml"],
    )
    add(
        "staleness",
        "what does the current paper report as headline val RMSE",
        "0.0521",
        ["paper/draft_v2.md"],
    )
    add(
        "staleness",
        "current encoder and LR for run 47",
        "dinov2; 3e-4",
        ["configs/run_047.yaml"],
    )

    # --- exact path ---
    add(
        "exact_path",
        "open configs/run_047.yaml",
        "file configs/run_047.yaml",
        ["configs/run_047.yaml"],
    )
    add(
        "exact_path",
        "find slurm/run_040.sbatch",
        "file slurm/run_040.sbatch",
        ["slurm/run_040.sbatch"],
    )
    add(
        "exact_path",
        "show me results/ablation.csv",
        "file results/ablation.csv",
        ["results/ablation.csv"],
    )
    add(
        "exact_path",
        "read src/fusion.py",
        "file src/fusion.py",
        ["src/fusion.py"],
    )
    add(
        "exact_path",
        "locate paper/draft_v2.md",
        "file paper/draft_v2.md",
        ["paper/draft_v2.md"],
    )
    add(
        "exact_path",
        "config_run_047.yaml",
        "configs/run_047.yaml",
        ["configs/run_047.yaml"],
    )
    add(
        "exact_path",
        "logs/run_055.out",
        "file logs/run_055.out",
        ["logs/run_055.out"],
    )
    add(
        "exact_path",
        "--num_pairs=3 in slurm/run_047.sbatch",
        "3",
        ["slurm/run_047.sbatch"],
    )

    # --- semantic ---
    add(
        "semantic",
        "what does the fusion module do",
        "concatenates per-view features and applies a 1x1 projection when fusion is on",
        ["src/fusion.py"],
    )
    add(
        "semantic",
        "how is the DINOv2 encoder loaded",
        "load_encoder wraps dinov2_vitb14",
        ["src/dinov2_encoder.py"],
    )
    add(
        "semantic",
        "find the Blender render helper with the trellis wires",
        "src/trellis_wires.py",
        ["src/trellis_wires.py"],
    )
    add(
        "semantic",
        "what is the 0.0550 number in this corpus",
        "baseline val RMSE",
        ["configs/run_047.yaml", "paper/draft_v2.md"],
    )
    add(
        "semantic",
        "where are slurm job scripts for the depth runs",
        "slurm/",
        [r.slurm_path for r in RUNS[:4]],
    )
    add(
        "semantic",
        "explain the difference between fusion on and fusion off",
        "fusion on concatenates per-view features; fusion off uses the reference view only",
        ["src/fusion.py"],
    )

    return qs


def write_manifest(root: Path, files: list[Path], out: Path) -> dict:
    entries = []
    h = hashlib.sha256()
    for path in sorted(files, key=lambda p: str(p)):
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest = content_hash(path.read_text(encoding="utf-8"))
        st = path.stat()
        entries.append(
            {
                "path": rel,
                "sha256": digest,
                "bytes": st.st_size,
                "mtime": int(st.st_mtime),
            }
        )
        h.update(f"{rel}:{digest}".encode())
    manifest = {
        "root": "bench/corpus/files",
        "n_files": len(entries),
        "aggregate_sha256": h.hexdigest(),
        "files": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(project_root: Path | None = None) -> dict:
    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    files_root = project_root / "bench" / "corpus" / "files"
    gold_path = project_root / "bench" / "gold" / "questions.jsonl"
    manifest_path = project_root / "bench" / "corpus" / "MANIFEST.json"
    written = write_files(files_root)
    gold = build_gold()
    write_gold(gold_path, gold)
    manifest = write_manifest(files_root, written, manifest_path)
    snapshot = project_root / "bench" / "corpus" / "SNAPSHOT.md"
    snapshot.write_text(
        f"""# Corpus snapshot

- files: {manifest['n_files']}
- aggregate sha256: `{manifest['aggregate_sha256']}`
- gold questions: {len(gold)}
- baseline RMSE: {BASELINE_RMSE}
- freeze rule: do not edit `bench/corpus/files` by hand; rebuild from registry.py

This is the stand-in for a personal capstone experiment tree. Swap the root
in `bench/corpus/MANIFEST.json` when you freeze the real directory, then
hand-correct `bench/gold/questions.jsonl`.
"""
    )
    return {"n_files": manifest["n_files"], "n_gold": len(gold), "sha256": manifest["aggregate_sha256"]}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
