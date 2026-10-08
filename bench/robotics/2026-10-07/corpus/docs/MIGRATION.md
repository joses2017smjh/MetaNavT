# Getting this project off the cluster

Written on 2026-09-05, against losing HPC access. What follows is what was
migrated, what was deliberately left behind, and how to restore.

## What the project weighs

| | size | fate |
|---|---|---|
| `external/.../logs` — 222 runs, 10,584 checkpoints | 36 GB | **partly migrated** — see tiers |
| venvs (`venv`, `venv-isaac60`, `venv60`, …) | 84 GB | left — rebuilt by `slurm/01_uv_sync.sbatch` |
| `container/` apptainer image | 26 GB | left — rebuilt by `slurm/00_build_container.sbatch` |
| Slurm job logs | 2.2 GB | **migrated** (compresses to a fraction; it is text) |
| the repo itself | 270 MB `.git` | already on GitHub |

The 110 GB of venvs and container is reproducible from two `sbatch` files and is
not worth moving. Everything else was tiered by what retraining could not
recreate.

## The tiers

**Tier 0 — in git, permanently.** Code, docs, `SLURM_JOBS.md`, all 25 clips.
Already pushed; nothing to do.

**Tier 1 — in git, added by this migration.** `runs/` holds the numbers:

| file | what it is |
|---|---|
| `runs/manifest.csv` | one row per run: iterations, checkpoint count, and the final / tail-mean / max of **every** scalar that run logged |
| `runs/params/<run>/*.yaml` | the env and agent config that produced each run |

`runs/scalars.csv.gz` — every `(run, tag, step, value)` triple, the full curves —
is a **release asset** rather than a git file: it lands around 200 MB and
GitHub rejects any file over 100 MB. `manifest.csv` is the summary of it and is
small enough to keep in the repo, which is why the summary is the thing that
gets committed and the raw curves are the thing that gets downloaded.

This is the tier that matters. `extract_curves.py` pulls a curated tag list for
plotting; `archive_runs.py` pulls everything, because the CSV has to answer
questions nobody has asked yet once the event files are gone. Every published
number in [FINDINGS](FINDINGS.md) is re-derivable from `manifest.csv` alone.

**Tier 2 — GitHub Release `hpc-archive-2026-09-05`.** Too big for git, small
enough for release assets (2 GB each):

| asset | contents |
|---|---|
| `final-checkpoints.tar.zst` | the highest-iteration checkpoint from each of 212 runs |
| `events-and-params.tar.zst` | all 212 raw TensorBoard event files, plus every `params/*.yaml` |
| `slurm-logs.tar.zst` | every job's console output |
| `scalars.csv.gz` | every `(run, tag, step, value)` triple across all 222 runs |

One final checkpoint per run is 0.67 GB and is what "can I ever replay this
policy again" depends on. Raw event files are kept as well as the CSV because a
CSV is a reading of them, and a reading can be wrong.

**Tier 3 — deliberately left behind.** The other ~10,370 intermediate
checkpoints, 31 GB. Recreating one means retraining its run, which is 12–20 GPU
hours; keeping all of them means moving 31 GB to answer a question nobody here
has asked in two months of work. If mid-training weights ever matter, the run is
re-trainable from `runs/params/`.

## Restoring

```bash
git clone --recurse-submodules https://github.com/joses2017smjh/bhl-robustness-ladder.git
cd bhl-robustness-ladder
gh release download hpc-archive-2026-09-05 -D /tmp/archive
zstd -d -c /tmp/archive/final-checkpoints.tar.zst | tar -xf - -C .
zstd -d -c /tmp/archive/events-and-params.tar.zst | tar -xf - -C .
```

Both tarballs store paths relative to the repo root, so they unpack back into
`external/Berkeley-Humanoid-Lite/logs/rsl_rl/...` where the tooling expects them.

To read the numbers without unpacking anything:

```python
import pandas as pd
m = pd.read_csv("runs/manifest.csv")
m[m.run.str.contains("ice")][["run", "iterations", "Curriculum/terrain_levels|tail5"]]
```

## With more storage

The tiering above assumed release assets. With a 2 TB Drive the dropped tier
fits easily — 36 GB of full training logs, 36 GB for the other project, 2.2 GB
of job logs, about 75 GB against 2 TB. [DRIVE_SETUP.md](DRIVE_SETUP.md) has the
headless OAuth flow and `scripts/to_drive.sh` does the transfer.

The venvs and container stay excluded even then. They are reproducible, and
they are pinned to this cluster's driver — restoring them elsewhere produces
something that does not run.

## What this does not save

The environment. Isaac Sim 5.1 / 6.0, the driver pairing, and the cluster's
quirks are not portable, and `docs/ISAAC_RENDER.md` exists partly because that
knowledge was expensive. Rebuilding elsewhere means re-running `00` and `01` on
whatever GPU you land on, and expecting a different set of version skews.
