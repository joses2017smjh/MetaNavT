# Cloth folding recovery, 19 September 2026

Actual folding lives in the adjacent `lehome-fold-repro` project, using two
SO-101 arms with grippers. This repository's Berkeley Humanoid Lite task sorts
garments by sweeping; the stock humanoid has no gripper joints. Neither a sort
nor an idealized cloth picker is evidence of humanoid folding.

## Measured status, 20 September 2026

The weekend jobs did not disappear: Slurm accounting shows that both adaptation
training jobs completed, and all 12 class-evaluation array tasks terminated:
**5 completed, 2 failed, and 5 reached their six-hour time limit**. The strict
camera smoke `21359521` also completed; its 12-step episodes only verify the
pipeline. It is not a folding benchmark.

The table below reports saved, completed episodes, not missing episodes counted
as failures. Every complete class evaluation requested 24 episodes: two per
garment, comprising 20 Seen and 4 Unseen episodes. All used evaluation seed 100,
a 600-step cap, real GPU PhysX cloth and the strict Storm-camera entrypoint.
Files are under `results/weekend-20260919/` with names
`fold-{baseline,adapt0,adapt1}-{class}-s100.json`.

| Policy | Class | Slurm task | Final state | Scorer successes / saved episodes |
| --- | --- | --- | --- | --- |
| Historical raster-adapted baseline | `pant_short` | `21359573_0` | Completed | **8/24** |
| Historical raster-adapted baseline | `top_short` | `21359573_1` | Failed; incomplete | 0/6 |
| Historical raster-adapted baseline | `top_long` | `21359573_2` | Timed out; incomplete | 0/4 |
| Historical raster-adapted baseline | `pant_long` | `21359573_3` | Timed out; incomplete | 0/2 |
| New adaptation seed 0 | `pant_short` | `21359574_0` | Timed out; incomplete | 0/14 |
| New adaptation seed 0 | `top_short` | `21359574_1` | Failed; incomplete | 0/6 |
| New adaptation seed 0 | `top_long` | `21359574_2` | Completed | 0/24 |
| New adaptation seed 0 | `pant_long` | `21359574_3` | Completed | 0/24 |
| New adaptation seed 1 | `pant_short` | `21359575_0` | Completed | **3/24** |
| New adaptation seed 1 | `top_short` | `21359575_1` | Timed out; incomplete | 0/2 |
| New adaptation seed 1 | `top_long` | `21359575_2` | Timed out; incomplete | 0/2 |
| New adaptation seed 1 | `pant_long` | `21359575_3` | Completed | 0/24 |

Short-pants baseline success was 8/20 Seen and 0/4 Unseen (33.3% overall).
Adaptation seed 1 achieved 3/20 Seen and 0/4 Unseen (12.5% overall).
**The new adaptation has not demonstrated an improvement**; four-class reliable
folding and unseen-garment generalization remain unestablished. These are small
samples, not evidence of a statistically established performance difference.

Every garment recorded in these 12 result files has finite robot/cloth state
and a positive fresh-camera count at least as large as its recorded step count.
The logs contain no swallowed `render failed` exceptions. The completed
short-pants baseline and adaptation-seed-1 runs recorded 25,057 and 27,727
rendered observations respectively. These checks address the old stale-frame
failure, but do not turn incomplete arrays into completed benchmarks.

The reported success is the **unmodified official checker's ever-triggered
success**, not a separately measured final settled-fold verdict. The official
loop latches the first success, executes 50 additional settling steps, and does
not recheck success at the end. Do not describe these counts as independently
validated stable terminal folds.

### Why the incomplete jobs stopped

Both failed short-top tasks stop on `Top_Short_Seen_3` inside the official
geometric scorer: `success_checker_chanllege.py` indexes particle 11,029 in a
mesh with only 10,869 particles. This is a scorer/mesh index mismatch, not a
measured failed policy episode; the first three garments were saved normally.

All five timeouts have their final progress message during garment switching,
immediately after `Old garment object deleted`, followed by roughly five hours
without new progress before Slurm terminated them. Examples include baseline
long pants switching to `Pant_Long_Seen_1` at 00:28:38 and timing out at
06:19:53 on September 20, and adaptation seed 0 switching to
`Pant_Short_Seen_7` at 16:27:54 and timing out at 21:26:38 on September 19.
This localizes the stall to the switch/configuration/physics-cleanup path;
the precise blocking operation is not yet established. Merely extending the
wall-time limit is not a demonstrated fix.

### Media available, and the blocked new-media campaign

The sibling campaign `lehome-fold-repro/campaigns/20260919-media` requested
16 new single-garment policy videos. Its gate `21360435` failed after 36 seconds,
before any physics step or rendered observation. The robot USD resolved to the
nonexistent `lehome-fold-repro/Assets/robots/lerobot/so101_follower_good.usd`.
The actual 23,251,377-byte asset exists under `lehome-data/Assets/robots/lerobot/`.
The copied official constant derives `ASSETS_ROOT` from the enclosing Git root;
the rollout's `--assets` argument is used later by the observer and does not
override that robot configuration. The saved state is `infrastructure_error`,
not a failed fold.

Consequently `21360436_[1-15%1]` is pending with
`DependencyNeverSatisfied` on `afterok:21360435`, and `21360437` waits for that
array. There are **zero completed new-media episodes and zero new campaign
GIFs**, not 16 policy failures. No jobs were canceled or resubmitted during this
September 20 audit.

> **Update, 2026-09-23 (scheduler state re-checked with `sacct`).** The array
> `21360436_[1-15%1]` and the report `21360437` are **CANCELLED by 19646** at
> 2026-09-20 16:58:44, never having started; they are not pending and cannot be
> revived. The repair gate `21367715` completed at 10:35:15 and rendered one
> seed-0 short-top episode (checker ever and terminal both false: rendering
> passed, the fold failed). The later folding work lives in the linked
> repository: the horizon pilot (`campaigns/20260921-horizon-pilot`) **ran** and
> was negative (H50 4/8, H10 0/8, H5 0/8 settled); closed-loop AWR training v2
> (`campaigns/20260922-closed-loop-training-v2`) **closed negative** — no
> candidate improved development H10 over the matched baseline (best 1/8 vs
> 2/8), so the **untouched baseline checkpoint is retained** and the adaptation
> seeds 0 and 1 above are archival; recovery-supervision v3
> (`campaigns/20260923-recovery-supervision-v3`) is running under its own
> driver. BHL does not submit folding jobs while that driver is live. The weekend class evaluators saved per-garment camera PNGs,
not full rollout videos; their measured successes cannot be illustrated with
new success GIFs unless they are replayed in a separately identified run.

Existing historical single-garment **policy** examples are available under the
sibling's `results/` directory:

- `rollout_Pant_Short_Seen_0_repro_ep509_policy_success.gif` — 27,541,422 bytes.
- `rollout_Pant_Short_Seen_0_repro_ep503_policy_failure.gif` — 26,059,358 bytes.

Both GIFs decode as 1920 × 480, 134-frame camera triptychs. Their adjacent
`rollout_Pant_Short_Seen_0_repro_ep509.json` and `...ep503.json` identify
`mode: policy`, `replay_episode: null`, 400 steps, Storm RGB, and the historical
`bc_smolvla_raster_ft_full` checkpoint, with success true and false respectively.
The episode numbers identify the **initial-pose source**, not action replay.
These are useful qualitative examples, but they are not new adaptation results
and their older metadata lacks the new finite-state/fresh-frame audit fields.

Eight additional historical GIFs in that campaign's `reference_replays/`
directory provide success and failure examples for each garment class. They
are explicitly **demonstration-action replays, not learned-policy rollouts**:
short tops episodes 2/1, long tops 251/250, short pants 501/502, and long pants
750/751 (success/failure respectively). Keep that distinction visible in any
portfolio caption; the existence of a replay success does not establish
learned-policy competence on that garment class.

## What failed, and what already works

The latest sibling ledger reports **6/24 folds** in job `21214241`, but a fresh
audit of its full log found **23,250 swallowed rendering failures** after
garment switches, all from the same USD layer-identifier collision described
below. Its geometric checker recorded those folds with real physics, but the
policy received stale images. That number is **not a valid correctly observed
closed-loop baseline**. A new strict baseline must establish that every camera
continues to observe the actual garment. Independent single-garment runs in
the sibling report occasional short-pants folds, which motivates retaining
this checkpoint, but reliable four-class folding remains unestablished.

The diagnosed problems were specific:

- `--device cpu` reached the simulation configuration and silently froze the
  PhysX particle cloth. GPU simulation is necessary even when policy inference
  could run on CPU.
- Launching Isaac 5.1 with `--enable_cameras` crashed the RTX delegate before a
  camera shim could be installed. Existing Storm RGB cameras avoid that path.
- The Storm observer originally kept the first garment's mesh when physics
  switched to another garment; the sibling's `retarget()` repair now follows it.
- A controlled same-state comparison established a renderer domain gap:
  the converged SmolVLA had action skill +0.976 on training images and −0.038
  on Storm images. Raster-image adaptation made it manipulate and sometimes fold.
- The old raster fine-tuning script divided individual frames randomly between
  train and validation. Adjacent frames from the same demonstration therefore
  appeared on both sides. It also used failed replays, although the original
  demonstration's future actions need not be appropriate after replay drift.

The last two data issues are addressed by `scripts/cloth/finetune_fold.py`.
The audit found 91 archives: **47 successful adaptation training episodes,
8 successful validation episodes, and 36 failed replays excluded**. Complete
garment identities Seen_8 and Seen_9 are held out under the initial default
split (the production jobs use the stratified split described below). The
original 30,000-step pretrained policy did see released Seen garments; these
are **adaptation holdouts**, not entirely unseen pretraining examples. Official
Unseen garments remain the stronger generalization test.

The success-filtered validation set contains only long tops (3 episodes) and
short pants (5): none of the captured short-top or long-pants replays on
Seen_8/9 succeeded. Their validation loss is therefore unavailable under this
split. Training contains 11 short-top, 14 long-top, 19 short-pants and only
3 long-pants episodes; new full-episode evaluations must expose this imbalance.

The production option `--validation-mode stratified-success` repairs that
coverage issue by holding out the highest-numbered successful garment in each
class: short top Seen_7, long top Seen_9, short pants Seen_9 and long pants
Seen_5. It still yields 47 training and 8 validation episodes, now with all
four classes present on each side (training 8/16/21/2; validation 3/1/3/1).
It refuses a dataset with fewer than two successful garment identities in any
class. `--class-balanced` samples classes uniformly, then episodes and frames.
Success-filtered validation measures prediction on successful replays, not an
unbiased population fold rate; the official evaluator supplies that rate.

Successful replay filtering reduces bad supervision but does not eliminate all
trajectory drift. This is behavior cloning, not online DAgger or reinforcement
learning. Validation loss is not a fold-success metric.

## Implemented experiment

`finetune_fold.py` trains the existing SmolVLA visual pathway and action expert
from the converged checkpoint. It retains original RGB pixels, stores one
capture at a time into a disposable memory-map cache, and never concatenates
the roughly 30 GB training set in RAM. Checkpoints include optimizer and RNG
state every 100 updates; each checkpoint is published atomically. Fixed
validation flow noise makes successive loss measurements comparable.

Run under the existing training interpreter inside `bhl.sif`:

```bash
python scripts/cloth/finetune_fold.py \
  --policy-path /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/outputs/train/bc_smolvla_seed0/checkpoints/030000/pretrained_model \
  --captures '/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/storm_capture100/*.npz' \
  --out RESULTS/cloth/adapt_s0 --cache NODE_SCRATCH/fold-cache \
  --steps 1500 --save-every 100 --batch-size 4 --unfreeze vision+action --seed 0 \
  --validation-mode stratified-success --class-balanced
```

Replace `RESULTS` and `NODE_SCRATCH` with concrete campaign and scratch paths.
`--audit-only` checks metadata without a GPU. The GPU health gate adds
`--smoke --steps 2 --save-every 1 --val-per-episode 1 --batch-size 2` and uses
a different output directory. A smoke checkpoint cannot resume full-data
training because its split hash differs. Float32 is used. The first DGX2 GPU
smoke (`21359477`) established that the installed cu128 PyTorch wheel excludes
V100/SM70 kernels: changing dtype cannot repair that. H100/H200 or supported
RTX GPUs are required with the existing training environment. A one-element
CUDA kernel now checks this before capture preparation or model loading.

The subsequent [isolated cu126 recovery](V100_TORCH_RECOVERY.md) passed real
V100 kernels and SmolVLA updates without modifying shared environments. Both
production seeds completed 1,500 updates (H100 `21359522`, V100 `21359530`).
The September 20 matched-evaluation results above supersede the original
queued status; training completion and lower validation loss alone did not
establish an improved learned fold-success rate.

`eval_fold.py` invokes the official loop and geometric scorer, captures the
metrics that loop returns, records finite robot/particle state and particle
motion, and requires all expected garments and episodes before declaring
completion. It writes progress after each garment, exposes Seen/Unseen
results separately, and avoids copying helper modules into shared source.

```bash
python scripts/cloth/eval_fold.py \
  --lehome-repo /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro \
  --assets /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/Assets \
  --dataset-root /nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/Datasets/example/four_types_merged \
  --policy-path RESULTS/cloth/adapt_s0/best.json \
  --out RESULTS/cloth/adapted_pant_short_s0.json \
  --garment-type pant_short --episodes 2 --max-steps 600 --seed 0
```

Use the existing `bc_smolvla_raster_ft_full` checkpoint for the matched baseline.
Run both with the same class, seed and step budget. There are 12 garments per
class, so `--episodes 2` means 24 episodes, not two total. A 12-step smoke only
tests the pipeline and must not be reported as folding performance. Storm
requires a graphics-capable allocation (A40/RTX8000 in the known working path).
Its depth channel is synthetic; this policy uses the three RGB cameras and
joint state, not that depth.

The first new evaluator smoke (`21359478`) exposed another real bug in the
sibling observer: after a garment switch, old camera/attribute handles retained
the USD layer, and recreating `obs_stage.usda` failed. Its shim caught those
exceptions and kept returning stale images. That run's metrics are void. The
new entrypoint now gives each garment a distinct layer path, clears old USD
handles, checks all three RGB outputs and propagates rendering errors.
The same error was then confirmed in the full historical `lh-n1-21214241.out`
log, overturning the ledger's interpretation of its 25% headline. Consequently,
new reports must require both geometric physics validity and fresh camera
observations; numeric success predicates alone did not catch this failure.

## MuJoCo diagnostic and the humanoid boundary

`scripts/cloth/fold_mujoco.py` adds a small towel with 2D flex physics, self
collision and two idealized Cartesian picker attachments. It lifts the near
edge over the crease, releases **both** attachments, and lets the towel settle.
The gate compares paired material-point alignment, the preserved transverse
width, half-size footprint, table support, finite state and solver warnings
against an untouched-towel negative control. It randomizes size, mass and
friction. It is a controller/physics diagnostic; it contains no humanoid or
gripper model. MuJoCo's native CPU solver does not benefit from reserving a GPU.

```bash
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python \
  scripts/cloth/fold_mujoco.py --episodes 4 --resolution 7 \
  --out RESULTS/cloth/mujoco_towel_gate.json
```

The BHL Newton failures remain separate: two-way coupling produced NaN at
first hand contact; larger constraint buffers did not help. One-way coupling
allowed pinned-base shirt and sock sorting, but jackets and free-base Newton
still became nonfinite, including the 200 Hz hold probe. Repeating those jobs
without a distinct solver hypothesis is not useful. They are not repaired by
the SO-101 folding adaptation.

New measured diagnostic result: Slurm job `21359475` completed four randomized
MuJoCo folds and four matched untouched controls. All 4 folds passed after
release; all 4 controls failed. Mirror-point RMSE was 0.58–0.69 mm, with finite
states and zero solver warnings throughout. This validates the explicitly
idealized picker physics gate only. The H100 SmolVLA smoke `21359481` wrote
both requested update checkpoints; two updates do not establish learned
folding performance.

## Recent primary research and how it affects this work

- [FolDeX, 9 September 2026](https://arxiv.org/abs/2609.10243) emphasizes
  long-horizon real-robot garment manipulation. It motivates reporting stage
  reliability and held-out garments; simulated success is not physical transfer.
- [Learning to Fold, 2026](https://arxiv.org/abs/2606.27163) and its
  [released implementation](https://github.com/IliaLarchenko/lehome_solution)
  demonstrate the LeHome competition approach with π0.5, asynchronous learning
  and corrective data. The immediately runnable local baseline is SmolVLA;
  reproducing that larger method needs a separate environment, not replacing
  the working LeRobot dependencies. Its README states over 500 GB for the full
  data/checkpoint setup.
- [LeHome, 2026](https://arxiv.org/abs/2604.22363) provides the household
  deformable task context; the [official challenge repository](https://github.com/lehome-official/lehome-challenge)
  supplies the actual garment assets, cameras and geometric evaluation here.
- [Dynamic robotic cloth folding with Koopman MPC, May 2026](https://arxiv.org/abs/2605.18373)
  studies model-based folding trajectories. It supports retaining a transparent
  trajectory/physics baseline alongside learned visuomotor policies, without
  claiming the simple picker controller implements that paper's MPC.
- [MuJoCo 3.3.5 XML reference](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html#body-flexcomp)
  documents the installed solver's `flexcomp`, elastic bending, contacts and
  equality constraints used in the diagnostic.

Research was checked on 19 September 2026. These are relevant recent sources,
not a claim that every 2026 folding publication was exhaustively reviewed.
