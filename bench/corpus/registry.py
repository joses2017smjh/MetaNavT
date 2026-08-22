"""Source of truth for the frozen demo corpus.

A stand-in for the capstone experiment tree. Gold questions are generated from
this registry so answers and retrieval targets cannot drift from the files.
When you freeze a real directory, replace this with a snapshot + hand-corrected gold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BASELINE_RMSE = 0.0550


@dataclass(frozen=True)
class Run:
    run_id: int
    encoder: str
    fusion: bool
    num_pairs: int
    bark: str
    lr: float
    val_rmse: float
    seed: int = 0
    current: bool = True
    version: int = 2
    notes: str = ""

    @property
    def beats_baseline(self) -> bool:
        return self.val_rmse < BASELINE_RMSE

    @property
    def config_path(self) -> str:
        if self.current:
            return f"configs/run_{self.run_id:03d}.yaml"
        return f"configs/archive/run_{self.run_id:03d}_v{self.version}.yaml"

    @property
    def log_path(self) -> str:
        return f"logs/run_{self.run_id:03d}.out"

    @property
    def slurm_path(self) -> str:
        return f"slurm/run_{self.run_id:03d}.sbatch"

    @property
    def ckpt_path(self) -> str:
        return f"checkpoints/run_{self.run_id:03d}.ckpt.meta.json"


# Current (live) runs
RUNS: list[Run] = [
    Run(40, "resnet50", False, 2, "oak", 1e-4, 0.0612, seed=7),
    Run(41, "resnet50", True, 2, "oak", 1e-4, 0.0588, seed=7),
    Run(42, "resnet50", True, 3, "pine", 3e-4, 0.0571, seed=11),
    Run(43, "clip", False, 1, "maple", 1e-4, 0.0720, seed=3),
    Run(44, "clip", True, 3, "maple", 2e-4, 0.0595, seed=3),
    Run(45, "dinov2", False, 2, "birch", 1e-4, 0.0568, seed=13),
    Run(46, "dinov2", True, 2, "birch", 1e-4, 0.0542, seed=13, notes="beats baseline"),
    Run(47, "dinov2", True, 3, "birch", 3e-4, 0.0521, seed=13, notes="best current"),
    Run(48, "dinov2", True, 3, "oak", 3e-4, 0.0533, seed=17),
    Run(49, "dinov2", False, 3, "pine", 5e-4, 0.0559, seed=17),
    Run(50, "resnet50", False, 3, "oak", 1e-3, 0.0640, seed=19),
    Run(51, "clip", False, 2, "pine", 1e-4, 0.0681, seed=23),
    Run(52, "dinov2", True, 1, "birch", 3e-4, 0.0548, seed=29),
    Run(53, "dinov2", True, 3, "maple", 2e-4, 0.0530, seed=31),
    Run(54, "resnet50", True, 3, "birch", 3e-4, 0.0564, seed=37),
    Run(55, "dinov2", False, 3, "oak", 3e-4, 0.0555, seed=41),
]

# Superseded versions (same run id, older mtime, different values)
ARCHIVED: list[Run] = [
    Run(
        47,
        "dinov2",
        True,
        3,
        "birch",
        1e-5,
        0.0630,
        seed=13,
        current=False,
        version=1,
        notes="superseded learning rate 1e-5",
    ),
    Run(
        46,
        "dinov2",
        True,
        2,
        "birch",
        5e-4,
        0.0601,
        seed=13,
        current=False,
        version=1,
        notes="superseded lr",
    ),
    Run(
        48,
        "dinov2",
        False,
        3,
        "oak",
        3e-4,
        0.0580,
        seed=17,
        current=False,
        version=1,
        notes="superseded fusion off",
    ),
]


@dataclass(frozen=True)
class AblationRow:
    name: str
    encoder: str
    fusion: bool
    val_rmse: float
    run_id: int


ABLATIONS: list[AblationRow] = [
    AblationRow("fusion_off_resnet", "resnet50", False, 0.0612, 40),
    AblationRow("fusion_on_resnet", "resnet50", True, 0.0588, 41),
    AblationRow("fusion_off_dino", "dinov2", False, 0.0568, 45),
    AblationRow("fusion_on_dino", "dinov2", True, 0.0542, 46),
    AblationRow("dino_3pair_fusion", "dinov2", True, 0.0521, 47),
    AblationRow("clip_fusion", "clip", True, 0.0595, 44),
    AblationRow("resnet_3pair", "resnet50", True, 0.0571, 42),
    AblationRow("dino_1pair", "dinov2", True, 0.0548, 52),
]


def current_runs() -> list[Run]:
    return list(RUNS)


def run_by_id(run_id: int, current_only: bool = True) -> Run:
    src = RUNS if current_only else RUNS + ARCHIVED
    for run in src:
        if run.run_id == run_id and (run.current if current_only else True):
            if current_only and run.current:
                return run
            if not current_only:
                return run
    for run in RUNS:
        if run.run_id == run_id:
            return run
    raise KeyError(run_id)


def lowest_rmse_run() -> Run:
    return min(RUNS, key=lambda r: r.val_rmse)


def dinov2_fusion_on() -> list[Run]:
    return [r for r in RUNS if r.encoder == "dinov2" and r.fusion]
