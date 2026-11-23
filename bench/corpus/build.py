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

from dataclasses import dataclass

from bench.corpus.registry import (
    ABLATIONS,
    ARCHIVED,
    BASELINE_RMSE,
    RUNS,
    AblationRow,
    Run,
    current_runs,
)
from app.eval.gold import GoldQuestion, write_gold
from app.eval.hashing import content_hash


@dataclass(frozen=True)
class Encoder:
    key: str  # the value a config holds: dinov2
    display: str  # the name questions and prose use: DINOv2
    short: str  # the name ablation rows use: dino
    model_name: str  # MODEL_NAME of its wrapper module: dinov2_vitb14
    arch: str  # ViT-B/14


@dataclass(frozen=True)
class Spec:
    """Everything write_files and build_gold are built from (M44). The default, V1, is fixture v1's registry, and
    with it both write fixture v1 byte for byte (tests/eval/test_corpus_spec.py rebuilds it into a temporary
    directory and compares every file, the MANIFEST and the gold file). Any other spec is a training corpus for the
    search policy (app/rl/corpora.py) and is never written into this tree.

    What a spec must hold for every question to have an answer: the eight ablation rows of ablation_names();
    the lowest-RMSE run uses encoders[0] with fusion on, three pairs and barks[0], is one of trellis_runs and is the
    only run with that encoder, pair count and bark; archived holds older versions of three current runs, the
    lowest-RMSE run's first."""

    runs: tuple[Run, ...]
    archived: tuple[Run, ...]
    ablations: tuple[AblationRow, ...]
    baseline_rmse: float
    encoders: tuple[Encoder, Encoder, Encoder]  # the three the questions name, the headline one first
    barks: tuple[str, str, str]  # the three bark types the questions name
    trellis_runs: tuple[int, ...]  # the runs rendered with trellis wires; they have checkpoint metadata


V1 = Spec(
    runs=tuple(RUNS),
    archived=tuple(ARCHIVED),
    ablations=tuple(ABLATIONS),
    baseline_rmse=BASELINE_RMSE,
    encoders=(
        Encoder("dinov2", "DINOv2", "dino", "dinov2_vitb14", "ViT-B/14"),
        Encoder("resnet50", "ResNet50", "resnet", "resnet50", "ResNet-50"),
        Encoder("clip", "CLIP", "clip", "clip_vitb16", "ViT-B/16"),
    ),
    barks=("birch", "pine", "oak"),
    trellis_runs=(46, 47, 48),
)


def ablation_names(spec: Spec) -> tuple[str, ...]:
    """The ablation rows a spec must list, in the order of results/ablation.csv."""
    a, b, c = (enc.short for enc in spec.encoders)
    return (f"fusion_off_{b}", f"fusion_on_{b}", f"fusion_off_{a}", f"fusion_on_{a}", f"{a}_3pair_fusion", f"{c}_fusion",
            f"{b}_3pair", f"{a}_1pair")

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

def _id_ranges(ids) -> str:
    """45-49, 52, 53, 55: three or more consecutive ids as a range."""
    ids = sorted(ids)
    out: list[str] = []
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out += [f"{ids[i]}-{ids[j]}"] if j - i >= 2 else [str(x) for x in ids[i : j + 1]]
        i = j + 1
    return ", ".join(out)


def _short_lr(lr: float) -> str:
    """3e-4: the form the staleness questions and the archive notes use (_fmt_lr gives 3e-04)."""
    mantissa, exponent = f"{lr:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _paper_lr(lr: float) -> str:
    """3.0e-4: the form the paper draft uses."""
    mantissa, exponent = f"{lr:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _row(spec: Spec, name: str) -> AblationRow:
    for row in spec.ablations:
        if row.name == name:
            return row
    raise ValueError(f"the spec has no ablation row {name!r} (see Spec)")


def _best(spec: Spec) -> Run:
    return min(spec.runs, key=lambda r: r.val_rmse)


def _encoder_src(spec: Spec) -> str:
    enc = spec.encoders[0]
    ids = _id_ranges(r.run_id for r in spec.runs if r.encoder == enc.key)
    return (
        f'"""{enc.display} encoder wrapper used by runs {ids}."""\n'
        "\n"
        f'MODEL_NAME = "{enc.model_name}"\n'
        "\n"
        "def load_encoder(freeze_backbone: bool = True):\n"
        f'    """Load {enc.display} {enc.arch}. Learning rate is set in the run config, not here."""\n'
        "    return MODEL_NAME\n"
    )


def _paper_v1(spec: Spec) -> str:
    enc = spec.encoders[0]
    off = _row(spec, f"fusion_off_{enc.short}")
    return f"""# Depth fusion draft v1 (superseded)

We find that stereo fusion does **not** help. The fusion-off {enc.display} run (run {off.run_id})
matches fusion-on within noise, so we recommend disabling fusion.

Val RMSE of the headline {enc.display} fusion run was previously reported as {spec.archived[1].val_rmse}
(archived config). This draft is stale.
"""


def _paper_v2(spec: Spec) -> str:
    enc, best = spec.encoders[0], _best(spec)
    return f"""# Depth fusion draft v2 (current)

Stereo fusion **does** help. The current {enc.display} + fusion + {best.num_pairs}-pair run (run {best.run_id})
reaches val RMSE {best.val_rmse}, beating the {spec.baseline_rmse:.4f} baseline. Bark type for that run
is {best.bark}. Learning rate is {_paper_lr(best.lr)}.

This draft supersedes draft v1. Prefer this file for current claims.
"""


def _yaml_for(run: Run, spec: Spec = V1) -> str:
    return f"""# run {run.run_id} config {'(CURRENT)' if run.current else '(SUPERSEDED)'}
run_id: {run.run_id}
encoder: {run.encoder}
fusion: {str(run.fusion).lower()}
num_pairs: {run.num_pairs}
bark_type: {run.bark}
learning_rate: {run.lr}
val_rmse: {run.val_rmse}
seed: {run.seed}
baseline_rmse: {spec.baseline_rmse}
render:
  trellis: {'true' if run.run_id in spec.trellis_runs else 'false'}
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


def write_files(root: Path, spec: Spec = V1) -> list[Path]:
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
    dump(f"src/{spec.encoders[0].key}_encoder.py", _encoder_src(spec), mtime=1_700_000_100)
    dump("src/trellis_wires.py", TRELLIS_SRC, mtime=1_700_000_100)

    # current configs get later mtimes than archives
    for run in spec.archived:
        dump(run.config_path, _yaml_for(run, spec), mtime=1_699_000_000 + run.run_id)
    for run in spec.runs:
        dump(run.config_path, _yaml_for(run, spec), mtime=1_710_000_000 + run.run_id)
        dump(run.log_path, _log_for(run), mtime=1_710_100_000 + run.run_id)
        dump(run.slurm_path, _sbatch_for(run), mtime=1_710_000_000 + run.run_id)
        if run.run_id in spec.trellis_runs:
            dump(run.ckpt_path, _ckpt_for(run), mtime=1_710_200_000 + run.run_id)

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["ablation", "encoder", "fusion", "val_rmse", "run_id"]
    )
    writer.writeheader()
    for row in spec.ablations:
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
    dump("paper/draft_v1.md", _paper_v1(spec), mtime=1_699_500_000)
    dump("paper/draft_v2.md", _paper_v2(spec), mtime=1_711_000_000)
    return written


def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}" if lr < 1e-3 or lr in (1e-4, 3e-4, 2e-4, 5e-4, 1e-5, 1e-3) else str(lr)


def build_gold(spec: Spec = V1, meta: list[dict] | None = None) -> list[GoldQuestion]:
    """The gold questions of a spec, in fixture v1's order: every answer and every relevant path is read from the
    spec. `meta`, when given, receives one entry per question: its id, the name of its template, and whether its
    answer is a constant of this builder (the same string for every spec: a sentence about src/fusion.py, a fixed
    path, the number of ablation rows) instead of a value of the spec."""
    qs: list[GoldQuestion] = []
    n = 0
    runs = list(spec.runs)
    by_id = {r.run_id: r for r in runs}
    enc_a, enc_b, enc_c = spec.encoders
    bark_a, bark_b, bark_c = spec.barks
    display = {enc.key: enc.display for enc in spec.encoders}
    baseline = f"{spec.baseline_rmse:.4f}"

    def beats(run: Run) -> bool:
        return run.val_rmse < spec.baseline_rmse

    def add(category: str, question: str, answer: str, paths: list[str], notes: str = "", *, template: str,
            constant: bool = False) -> None:
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
        if meta is not None:
            meta.append({"id": f"q{n:03d}", "template": template, "constant_answer": constant})

    # --- simple factual (per-run LR, pairs, bark, encoder, rmse) ---
    for run in runs:
        cfg = run.config_path
        add(
            "simple_factual",
            f"what learning rate did run {run.run_id} use",
            _fmt_lr(run.lr),
            [cfg],
            template="simple.lr",
        )
        add(
            "simple_factual",
            f"how many stereo pairs did run {run.run_id} use",
            str(run.num_pairs),
            [cfg, run.slurm_path],
            template="simple.pairs",
        )
        add(
            "simple_factual",
            f"what bark type did run {run.run_id} use",
            run.bark,
            [cfg],
            template="simple.bark",
        )
        add(
            "simple_factual",
            f"what encoder did run {run.run_id} use",
            run.encoder,
            [cfg],
            template="simple.encoder",
        )
        add(
            "simple_factual",
            f"what was val RMSE for run {run.run_id}",
            str(run.val_rmse),
            [cfg, run.log_path],
            template="simple.rmse",
        )

    # --- conditional ---
    a_fusion = [r for r in runs if r.encoder == enc_a.key and r.fusion]
    lrs = sorted({_fmt_lr(r.lr) for r in a_fusion})
    add(
        "conditional",
        f"what LR did the {enc_a.display} runs use with fusion on",
        ", ".join(lrs),
        [r.config_path for r in a_fusion],
        template="conditional.lr_headline_encoder_fusion_on",
    )
    add(
        "conditional",
        f"which {enc_a.display} fusion runs beat the {baseline} baseline",
        ", ".join(str(r.run_id) for r in a_fusion if beats(r)),
        [r.config_path for r in a_fusion if beats(r)],
        template="conditional.headline_encoder_fusion_beats_baseline",
    )
    three_pair_a = [r for r in runs if r.encoder == enc_a.key and r.num_pairs == 3]
    add(
        "conditional",
        f"what bark types appear in {enc_a.display} runs that used 3 stereo pairs",
        ", ".join(sorted({r.bark for r in three_pair_a})),
        [r.config_path for r in three_pair_a],
        template="conditional.barks_headline_encoder_3_pairs",
    )
    fusion_off_b = [r for r in runs if r.encoder == enc_b.key and not r.fusion]
    add(
        "conditional",
        f"what learning rate did the {enc_b.display} runs use with fusion off",
        ", ".join(sorted({_fmt_lr(r.lr) for r in fusion_off_b})),
        [r.config_path for r in fusion_off_b],
        template="conditional.lr_second_encoder_fusion_off",
    )
    c_fusion = [r for r in runs if r.encoder == enc_c.key and r.fusion]
    add(
        "conditional",
        f"what val RMSE did {enc_c.display} runs with fusion on report",
        ", ".join(str(r.val_rmse) for r in c_fusion),
        [r.config_path for r in c_fusion],
        template="conditional.rmse_third_encoder_fusion_on",
    )
    bark_a_fusion = [r for r in runs if r.bark == bark_a and r.fusion]
    add(
        "conditional",
        f"which fusion-on {bark_a} runs exist",
        ", ".join(str(r.run_id) for r in bark_a_fusion),
        [r.config_path for r in bark_a_fusion],
        template="conditional.fusion_on_bark_runs",
    )
    add(
        "conditional",
        "what encoder is used when fusion is on and num_pairs is 1",
        ", ".join(sorted({r.encoder for r in runs if r.fusion and r.num_pairs == 1})),
        [r.config_path for r in runs if r.fusion and r.num_pairs == 1],
        template="conditional.encoder_fusion_on_1_pair",
    )
    bark_b_three = [r for r in runs if r.bark == bark_b and r.num_pairs == 3]
    add(
        "conditional",
        f"what learning rates did {bark_b} bark runs with 3 pairs use",
        ", ".join(sorted({_fmt_lr(r.lr) for r in bark_b_three})),
        [r.config_path for r in bark_b_three],
        template="conditional.lr_bark_3_pairs",
    )

    # --- comparative ---
    best = _best(spec)
    lowest_row = min(spec.ablations, key=lambda row: row.val_rmse)
    add(
        "comparative",
        "which ablation had the lowest val RMSE",
        lowest_row.name,
        ["results/ablation.csv", by_id[lowest_row.run_id].config_path],
        template="comparative.lowest_ablation",
    )
    worst = max(runs, key=lambda r: r.val_rmse)
    add(
        "comparative",
        "which run had the highest val RMSE",
        str(worst.run_id),
        [worst.config_path, "results/ablation.csv"],
        template="comparative.highest_rmse_run",
    )
    add(
        "comparative",
        "which run had the lowest val RMSE",
        str(best.run_id),
        [best.config_path, best.log_path],
        template="comparative.lowest_rmse_run",
    )
    # the headline pair: the three-pair fusion row of the headline encoder against its fusion-off row
    on = by_id[_row(spec, f"{enc_a.short}_3pair_fusion").run_id]
    off = by_id[_row(spec, f"fusion_off_{enc_a.short}").run_id]
    add(
        "comparative",
        f"did fusion-on {enc_a.display} beat fusion-off {enc_a.display}",
        f"{'yes' if on.val_rmse < off.val_rmse else 'no'}: {on.val_rmse} vs {off.val_rmse} for the headline pair ({on.run_id} vs {off.run_id})",
        [on.config_path, off.config_path, "results/ablation.csv"],
        template="comparative.fusion_on_vs_off_headline_encoder",
    )
    add(
        "comparative",
        "which encoder has the best current val RMSE",
        best.encoder,
        [best.config_path],
        template="comparative.best_encoder",
    )
    first = runs[0]
    add(
        "comparative",
        f"compare val RMSE of run {first.run_id} and run {best.run_id}",
        f"{first.val_rmse} vs {best.val_rmse}",
        [first.config_path, best.config_path],
        template="comparative.compare_two_runs",
    )
    off_b, on_b = _row(spec, f"fusion_off_{enc_b.short}"), _row(spec, f"fusion_on_{enc_b.short}")
    add(
        "comparative",
        f"which ablation is worse, {off_b.name} or {on_b.name}",
        max((off_b, on_b), key=lambda row: row.val_rmse).name,
        ["results/ablation.csv"],
        template="comparative.worse_ablation",
    )
    one_pair = by_id[_row(spec, f"{enc_a.short}_1pair").run_id]
    add(
        "comparative",
        f"is run {one_pair.run_id} better than the {baseline} baseline",
        "yes" if beats(one_pair) else "no",
        [one_pair.config_path],
        template="comparative.run_vs_baseline",
    )

    # --- aggregation ---
    n_three = sum(1 for r in runs if r.num_pairs == 3)
    add(
        "aggregation",
        "how many runs used 3 stereo pairs",
        str(n_three),
        [r.config_path for r in runs if r.num_pairs == 3],
        template="aggregation.runs_3_pairs",
    )
    n_a = sum(1 for r in runs if r.encoder == enc_a.key)
    add(
        "aggregation",
        f"how many runs used {enc_a.display}",
        str(n_a),
        [r.config_path for r in runs if r.encoder == enc_a.key],
        template="aggregation.runs_headline_encoder",
    )
    n_fusion = sum(1 for r in runs if r.fusion)
    add(
        "aggregation",
        "how many current runs have fusion on",
        str(n_fusion),
        [r.config_path for r in runs if r.fusion],
        template="aggregation.runs_fusion_on",
    )
    n_beat = sum(1 for r in runs if beats(r))
    add(
        "aggregation",
        f"how many runs beat the {baseline} baseline",
        str(n_beat),
        [r.config_path for r in runs if beats(r)],
        template="aggregation.runs_beat_baseline",
    )
    add(
        "aggregation",
        "how many unique bark types appear in current runs",
        str(len({r.bark for r in runs})),
        [r.config_path for r in runs],
        template="aggregation.unique_barks",
    )
    add(
        "aggregation",
        "what is the average num_pairs across current runs",
        f"{sum(r.num_pairs for r in runs) / len(runs):.2f}",
        [r.config_path for r in runs],
        template="aggregation.average_pairs",
    )
    add(
        "aggregation",
        f"how many {enc_c.display} runs are there",
        str(sum(1 for r in runs if r.encoder == enc_c.key)),
        [r.config_path for r in runs if r.encoder == enc_c.key],
        template="aggregation.runs_third_encoder",
    )
    add(
        "aggregation",
        "how many ablations are listed",
        str(len(spec.ablations)),
        ["results/ablation.csv"],
        template="aggregation.ablations_listed",
        constant=True,
    )
    add(
        "aggregation",
        f"how many current runs used {bark_c} bark",
        str(sum(1 for r in runs if r.bark == bark_c)),
        [r.config_path for r in runs if r.bark == bark_c],
        template="aggregation.runs_bark",
    )
    add(
        "aggregation",
        f"how many runs used a learning rate of {_short_lr(best.lr)}",
        str(sum(1 for r in runs if abs(r.lr - best.lr) < 1e-12)),
        [r.config_path for r in runs if abs(r.lr - best.lr) < 1e-12],
        template="aggregation.runs_learning_rate",
    )

    # --- multi-hop ---
    add(
        "multi_hop",
        f"which bark type was used in the run that beat the {baseline} baseline with the lowest RMSE, and where is its config",
        f"{best.bark}; {best.config_path}",
        [best.config_path, best.log_path],
        template="multi_hop.best_run_bark_and_config",
    )
    add(
        "multi_hop",
        "which slurm script launched the lowest-RMSE run and what encoder did it use",
        f"{best.slurm_path}; {best.encoder}",
        [best.slurm_path, best.config_path],
        template="multi_hop.best_run_slurm_and_encoder",
    )
    add(
        "multi_hop",
        "what checkpoint was produced by the best current run",
        best.ckpt_path,
        [best.config_path, best.ckpt_path],
        template="multi_hop.best_run_checkpoint",
    )
    # the one run of the headline encoder with three pairs and the first bark
    headline = next(r for r in runs if r.encoder == enc_a.key and r.num_pairs == 3 and r.bark == bark_a)
    add(
        "multi_hop",
        f"what does the fusion module do in the run that used 3 pairs and {bark_a} bark with {enc_a.display}",
        "concatenates per-view features and projects with a 1x1 when fusion is on",
        ["src/fusion.py", headline.config_path],
        template="multi_hop.fusion_module_in_headline_run",
        constant=True,
    )
    add(
        "multi_hop",
        f"which paper draft reports the current RMSE of run {best.run_id} and what is that RMSE",
        f"paper/draft_v2.md; {best.val_rmse}",
        ["paper/draft_v2.md", best.config_path],
        template="multi_hop.paper_draft_and_rmse",
    )
    two_pair = next(r for r in runs if r.num_pairs == 2 and beats(r))
    add(
        "multi_hop",
        "the run that first beat the baseline with 2 pairs: bark type and config path",
        f"{two_pair.bark}; {two_pair.config_path}",
        [two_pair.config_path],
        template="multi_hop.first_2_pair_run_beating_baseline",
    )
    off_a = _row(spec, f"fusion_off_{enc_a.short}")
    add(
        "multi_hop",
        f"where is the ablation table that lists {off_a.name} and what RMSE does it show",
        f"results/ablation.csv; {off_a.val_rmse}",
        ["results/ablation.csv"],
        template="multi_hop.ablation_table_and_rmse",
    )
    trellis = [by_id[i] for i in spec.trellis_runs]
    add(
        "multi_hop",
        "which source file documents trellis wires and which runs enabled them",
        "src/trellis_wires.py; " + ", ".join(str(r.run_id) for r in trellis),
        ["src/trellis_wires.py", *[r.config_path for r in trellis]],
        template="multi_hop.trellis_source_and_runs",
    )

    # --- staleness: the current versions of the archived runs ---
    cur = [by_id[r.run_id] for r in spec.archived]
    add(
        "staleness",
        f"what's the current learning rate for the {display.get(cur[0].encoder, cur[0].encoder)} run {cur[0].run_id}",
        _short_lr(cur[0].lr),
        [cur[0].config_path],
        notes=f"archive has {_short_lr(spec.archived[0].lr)}; gold target is current chunk only",
        template="staleness.current_lr_named_encoder",
    )
    add(
        "staleness",
        f"what is the current val RMSE for run {cur[0].run_id}",
        str(cur[0].val_rmse),
        [cur[0].config_path, cur[0].log_path],
        template="staleness.current_rmse",
    )
    add(
        "staleness",
        "does the current paper draft say fusion helps",
        "yes",
        ["paper/draft_v2.md"],
        template="staleness.paper_says_fusion_helps",
        constant=True,
    )
    add(
        "staleness",
        f"what is the current learning rate for run {cur[1].run_id}",
        _short_lr(cur[1].lr),
        [cur[1].config_path],
        template="staleness.current_lr",
    )
    add(
        "staleness",
        f"is fusion currently on for run {cur[2].run_id}",
        "yes" if cur[2].fusion else "no",
        [cur[2].config_path],
        template="staleness.fusion_currently_on",
    )
    add(
        "staleness",
        f"what learning rate should I use if I resume the {enc_a.display} 3-pair {bark_a} run",
        _short_lr(headline.lr),
        [headline.config_path],
        template="staleness.resume_headline_run_lr",
    )
    add(
        "staleness",
        "what does the current paper report as headline val RMSE",
        str(best.val_rmse),
        ["paper/draft_v2.md"],
        template="staleness.paper_headline_rmse",
    )
    add(
        "staleness",
        f"current encoder and LR for run {cur[0].run_id}",
        f"{cur[0].encoder}; {_short_lr(cur[0].lr)}",
        [cur[0].config_path],
        template="staleness.current_encoder_and_lr",
    )

    # --- exact path ---
    add(
        "exact_path",
        f"open {best.config_path}",
        f"file {best.config_path}",
        [best.config_path],
        template="exact_path.open_config",
    )
    add(
        "exact_path",
        f"find {first.slurm_path}",
        f"file {first.slurm_path}",
        [first.slurm_path],
        template="exact_path.find_slurm",
    )
    add(
        "exact_path",
        "show me results/ablation.csv",
        "file results/ablation.csv",
        ["results/ablation.csv"],
        template="exact_path.show_ablation_csv",
        constant=True,
    )
    add(
        "exact_path",
        "read src/fusion.py",
        "file src/fusion.py",
        ["src/fusion.py"],
        template="exact_path.read_fusion_src",
        constant=True,
    )
    add(
        "exact_path",
        "locate paper/draft_v2.md",
        "file paper/draft_v2.md",
        ["paper/draft_v2.md"],
        template="exact_path.locate_paper_draft",
        constant=True,
    )
    add(
        "exact_path",
        f"config_run_{best.run_id:03d}.yaml",
        best.config_path,
        [best.config_path],
        template="exact_path.config_file_name",
    )
    last = runs[-1]
    add(
        "exact_path",
        last.log_path,
        f"file {last.log_path}",
        [last.log_path],
        template="exact_path.log_path",
    )
    add(
        "exact_path",
        f"--num_pairs={best.num_pairs} in {best.slurm_path}",
        str(best.num_pairs),
        [best.slurm_path],
        template="exact_path.flag_in_slurm_script",
        constant=True,  # the lowest-RMSE run has three pairs in every spec (see Spec)
    )

    # --- semantic ---
    add(
        "semantic",
        "what does the fusion module do",
        "concatenates per-view features and applies a 1x1 projection when fusion is on",
        ["src/fusion.py"],
        template="semantic.fusion_module",
        constant=True,
    )
    add(
        "semantic",
        f"how is the {enc_a.display} encoder loaded",
        f"load_encoder wraps {enc_a.model_name}",
        [f"src/{enc_a.key}_encoder.py"],
        template="semantic.encoder_loaded",
    )
    add(
        "semantic",
        "find the Blender render helper with the trellis wires",
        "src/trellis_wires.py",
        ["src/trellis_wires.py"],
        template="semantic.trellis_helper",
        constant=True,
    )
    add(
        "semantic",
        f"what is the {baseline} number in this corpus",
        "baseline val RMSE",
        [best.config_path, "paper/draft_v2.md"],
        template="semantic.baseline_number",
        constant=True,
    )
    add(
        "semantic",
        "where are slurm job scripts for the depth runs",
        "slurm/",
        [r.slurm_path for r in runs[:4]],
        template="semantic.slurm_scripts",
        constant=True,
    )
    add(
        "semantic",
        "explain the difference between fusion on and fusion off",
        "fusion on concatenates per-view features; fusion off uses the reference view only",
        ["src/fusion.py"],
        template="semantic.fusion_on_vs_off",
        constant=True,
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


def main(project_root: Path | None = None, spec: Spec = V1) -> dict:
    """Write a spec's corpus, MANIFEST, gold questions and snapshot note under `project_root` (default: this
    repository, which only ever holds fixture v1: another spec needs a project_root of its own)."""
    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    if spec != V1 and project_root.resolve() == _ROOT.resolve():
        raise ValueError("this repository holds fixture v1 only: give a spec other than V1 its own project_root")
    files_root = project_root / "bench" / "corpus" / "files"
    gold_path = project_root / "bench" / "gold" / "questions.jsonl"
    manifest_path = project_root / "bench" / "corpus" / "MANIFEST.json"
    written = write_files(files_root, spec)
    gold = build_gold(spec)
    write_gold(gold_path, gold)
    manifest = write_manifest(files_root, written, manifest_path)
    snapshot = project_root / "bench" / "corpus" / "SNAPSHOT.md"
    snapshot.write_text(
        f"""# Corpus snapshot

- files: {manifest['n_files']}
- aggregate sha256: `{manifest['aggregate_sha256']}`
- gold questions: {len(gold)}
- baseline RMSE: {spec.baseline_rmse}
- freeze rule: do not edit `bench/corpus/files` by hand; rebuild from registry.py

This is the stand-in for a personal capstone experiment tree. Swap the root
in `bench/corpus/MANIFEST.json` when you freeze the real directory, then
hand-correct `bench/gold/questions.jsonl`.
"""
    )
    return {"n_files": manifest["n_files"], "n_gold": len(gold), "sha256": manifest["aggregate_sha256"]}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
