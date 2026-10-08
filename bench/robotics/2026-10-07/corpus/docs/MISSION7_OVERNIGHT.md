# Mission7 bounded overnight diagnosis — September 20, 2026

**Subsequent diagnostic phase completed:** command-response, reward, C4,
six controlled PPO cells and paired route/contact studies are reported in
[Mission7 Approach diagnosis](MISSION7_APPROACH_DEBUG.md). That follow-up
found partial learning signals and a pressure-plate contact failure mechanism;
no stable-learning or sensor-comparison gate passed. Historical overnight
results below remain unchanged.

## Final audit — September 21, 2026

All **14 Slurm tasks COMPLETED with exit 0:0**, with no tasks left queued. **Eleven studies executed; three sensor studies skipped training** because the learning prerequisite failed. Total elapsed allocated CPU time was **16.22 CPU-hours**, versus the 56 CPU-hour cap; no GPUs were allocated. This measures allocated CPU time, not actual CPU utilization.

Nine training runs each finished 200 updates /12,800 decisions with finite scalar logs and final checkpoints. **Every run ended at 0/8 validation success and failed its diagnostic learning gate. All recorded training episodes had zero terminal successes.** C4 briefly reached 2/8 at update 100, then returned to 0/8 at updates 150 and 200; that transient result never passed the 3/8 gate.

| Study | Slurm task | Outcome | Validation successes at updates 0 /50 /100 /150 /200 | Training successes /episodes |
|---|---|---|---|---|
| A1 | `21370064_0` | Learning gate FAILED | 0 / 0 / 0 / 0 / 0 (each /8) | 0/183 |
| A2 | `21370064_1` | Learning gate FAILED | 0 / 0 / 0 / 0 / 0 (each /8) | 0/84 |
| A3 | `21370064_2` | Learning gate FAILED | 0 / 0 / 0 / 0 / 0 (each /8) | 0/106 |
| A4 | `21370064_3` | Learning gate FAILED | 0 / 0 / 0 / 0 / 0 (each /8) | 0/100 |
| A5 | `21370064_4` | Learning gate FAILED | 0 / 0 / 0 / 0 / 0 (each /8) | 0/112 |
| B1 | `21370064_5` | Learning gate FAILED | 0 / 0 / 0 / 0 / 0 (each /8) | 0/198 |
| C2 | `21370064_6` | Learning gate FAILED | 0 / 0 / 0 / 0 / 0 (each /8) | 0/201 |
| C3 | `21370064_7` | Learning gate FAILED | 0 / 0 / 0 / 0 / 0 (each /8) | 0/186 |
| C4 | `21370064_8` | Learning gate FAILED | 0 / 0 / 2 / 0 / 0 (each /8) | 0/216 |
| S1 | `21370066_9` | SKIPPED_NOT_LEARNABLE | — | — |
| S2 | `21370066_10` | SKIPPED_NOT_LEARNABLE | — | — |
| S3 | `21370066_11` | SKIPPED_NOT_LEARNABLE | — | — |
| D1 | `21370065_12` | COMPLETED_DIAGNOSTIC | — | — |
| D2 | `21370065_13` | COMPLETED_DIAGNOSTIC | — | — |

Source snapshot hashes, all fourteen config hashes, expected artifacts, update counts and scheduler receipts were cross-checked. Final machine-readable evidence: `results/mission7-overnight-20260920/final-audit-20260921.json`; raw accounting: `scheduler-final-20260921.psv`. `matrix.json` now distinguishes scheduler completion from each study outcome. Original checkpoints, source snapshots and raw result files remain unchanged.

### What the evidence establishes

**The first observed learning failure is Approach.** Neither privileged goal input (B1), increased exploration (C2), conservative optimization (C3), nor the easier start (C4) produced retained success at this seed and budget. Higher-stage runs do not identify a later curriculum boundary because the first stage has not been mastered. In training, A4 activated one button but crossed no gates; A5 acquired twice but crossed no gates. These isolated events are not learned interaction competence.

**Sparse positive reward and ineffective movement discovery are leading hypotheses, not a proven single cause.** D1 found random, untrained and original trained policies each had 0/16 approach successes; the scripted oracle had 12/16. The deterministic untrained/trained policies received only the time penalty (−0.18 per episode). Random-policy absolute reward contributions were 73.4% failure penalties, 17.6% contact penalties and 9.0% time cost, with zero success reward. The audit reward exactly matched the production reward. Finite PPO updates do not rule out optimization or observation-design problems. Adaptive learning rates reached roughly 1e-5 in most runs; fixed 1e-4 C3 also failed, so learning rate alone is not an established explanation.

**Later mechanics are feasible on some layouts, but unreliable.** Full-route privileged Doors: **4/16 success**, seven falls, three timeouts, two excessive-contact failures. Full-route Transport: **4/16 success**, eight falls, three timeouts, one excessive-contact failure; acquired in 13/16. Localized door components scored **6/8** and **7/8**; localized transport **7/8**, acquisition 8/8. The local fixtures use different starting conditions from full routes. Carrying remains a kinematic abstraction and placement a floor drop zone, not physical grasping or shelf manipulation.

**Sensors are not globally constant or disconnected.** All sampled actor tensors were finite; LiDAR validity was 99.87%, paired-depth validity 80.46%, with correct binary flags and zero-filled invalid values. Both encoders varied; mean fusion weights were 0.474/0.526. LiDAR differed at all eight junction/goal geometry probes; paired depth differed at seven, while one probe was effectively unchanged. Paired-eye correlation was 0.894. These checks establish signal flow, not goal identifiability, learned usefulness or sensor advantage. S1–S3 did not train and must not be presented as a completed four-sensor comparison.

**The layouts differ structurally across splits.** All 352 topology hashes were unique. Mean route lengths train/validation/test were 24.02/23.82/24.10 m; mean turns 8.37/8.75/8.03. Minimum validation/test edge symmetric differences from training were 12/14. Minimum nominal approach spawn-to-post clearance was 0.328 m; that point-clearance measure does not certify full-body safety. Held-out layouts received static geometry analysis only, with no test rollouts or test-performance-based selection.

### Next steps — proposed, not submitted

1. **Calibrate the frozen gait first.** Run bounded training-layout command probes across forward/backward/lateral directions, the existing speed range, and command durations. Measure actual displacement, stopping distance, falls and contacts. The earlier 0.28-versus-0.40 m/s lateral probe makes this a concrete question; one layout does not define a universal movement threshold. Choose a stable command interface from measured responses.
2. **Isolate learning signal from action persistence in a small privileged Approach experiment.** After calibration, compare a 2×2 design: original sparse reward versus explicitly labeled goal-progress shaping, and existing 0.2 s independent commands versus temporally coherent exploration using the calibrated interface. Keep simulated interaction time and evaluation layouts matched; account explicitly for changed decision frequency if action holding changes it. Goal-based shaping belongs only to this diagnostic, not an unlabeled change to the deployable benchmark. Log goal approach/dwell, actual velocity, successful training episodes, KL/clip fraction and actor updates. If even this positive control fails, inspect PPO integration before spending more on larger tasks.
3. **Establish repeatable deployable Approach learning before releasing sensor comparisons.** Use a successful privileged result to guide curriculum or a clearly labeled imitation warm start, then require the sensor-only Both actor to pass the original 16-layout learning gate and repeat on additional training seeds. Do not cherry-pick C4 update 100 as a learning pass. Only then run matched sensor arms with identical budgets, rewards and training initialization rules.
4. **Improve door/transport reliability before expanding the curriculum.** Inspect the saved fall/contact traces, test safer turns, braking and plate approach, then repeat the same full-route training-layout feasibility cases. Localized success is useful evidence but the current 25% full-route completion is too fragile to treat as a solved controller. Keep the full 60-training/108-evaluation campaign gated.

No new jobs were submitted in this status/documentation update. Existing results remain the baseline for any subsequent changes.

## Original campaign design and submission history

The original Both/seed-0 pilot finished all 400 updates but failed its learning
gate: **0/16 validation successes**, including the final checkpoint. Slurm calls
job `21369796` FAILED (exit 2); that is the explicit scientific gate, not a
nonfinite training crash. The checkpoint and both diagnostic jobs are preserved.
`21369990` and `21369991` completed successfully as diagnostics.

The approach feasibility study found 22/32 successes with the privileged
0.28 m/s controller, versus 0/32 with 0.12 m/s and 0/32 standing still. The
checkpoint study found normal 0/16, LiDAR removed 0/16, depth removed 2/16,
both removed 0/16. These results do not establish sensor benefit. The latter
checkpoint never passed its learning gate.

## Submission matrix

Fourteen additional tasks in three arrays; eleven have no dependency, three
wait for C4. A1 also serves as C1 (current PPO); C4 also serves as the matched
Both sensor arm. This avoids two redundant runs. Every training cell starts
from scratch, stays at its designated stage, and uses seed 0.

| Index | Study | Sensor | Fixed stage | Training / diagnostic budget | Dependency |
|---:|---|---|---|---|---|
| 0 | A1 / C1: stage isolation and current PPO | Both | Approach | 200 ×64 decisions | none |
| 1 | A2: stage isolation | Both | Branches | 200 ×64 | none |
| 2 | A3: stage isolation | Both | Navigation | 200 ×64 | none |
| 3 | A4: stage isolation | Both | Doors | 200 ×64 | none |
| 4 | A5: stage isolation | Both | Transport | 200 ×64 | none |
| 5 | B1: privileged goal PPO | Both + explicit goal vector | Approach | 200 ×64 | none |
| 6 | C2: increased exploration | Both | Approach | 200 ×64 | none |
| 7 | C3: conservative optimizer | Both | Approach | 200 ×64 | none |
| 8 | C4 / sensor Both: easier start | Both | Approach, 0.40 m start | 200 ×64 | none |
| 9 | S1: matched sensor pilot | Blind | Approach, 0.40 m start | 200 ×64 | C4 completion and learning evidence |
| 10 | S2: matched sensor pilot | LiDAR | Approach, 0.40 m start | 200 ×64 | C4 completion and learning evidence |
| 11 | S3: matched sensor pilot | Paired ray depth | Approach, 0.40 m start | 200 ×64 | C4 completion and learning evidence |
| 12 | D1: rewards, observations, layouts | Four masks / Both controls | Approach / static geometry | 64 reward episodes; geometry and feature probes | none |
| 13 | D2: privileged interaction feasibility | Both, oracle control | Doors and transport | 32 full-route +24 localized episodes | none |

Each task requests **2 CPUs, 12 GB, zero GPUs, two hours**: a maximum of
**56 allocated CPU-hours**. Training array `0-8%3` permits three concurrent
tasks; audits `12-13%2` permit two; conditional sensors `9-11%3` permit three.
The independent arrays can start immediately subject to Slurm scheduling.
The sensor array uses `afterok:<training-array>_8`, with invalid dependencies
cancelled automatically. It also checks C4's JSON result before training.

All training cells receive 12,800 high-level decisions, evaluation on the same
eight validation layouts at updates 0/50/100/150/200, and checkpoint saves every
25 updates. Stage time limits remain 18/50/180/180/180 simulated seconds.
Diagnostics cannot promote the production curriculum. Completion records
successful execution even when task success is zero.

C4 must achieve at least 3/8 final validation successes and improve by at least
1/8 over its untrained baseline. Otherwise S1–S3 write
`SKIPPED_NOT_LEARNABLE` without training. This is a small diagnostic gate,
not the original 16-layout production gate or proof of sensor advantage.
The easier spawn is outside the 0.36 m goal radius but needs little travel;
even a positive comparison remains an approach pilot, not maze navigation.

## Configuration and interpretation

Inspected baseline: learning rate 3e-4, adaptive schedule, entropy coefficient
0.01, initial Gaussian std 0.5, rollout 64, four minibatches, four PPO epochs,
gamma 0.995, lambda 0.95, no observation normalizer, reward multiplier 1.
Existing inputs use fixed physical scaling and explicit validity masks.

C2 changes entropy to 0.02 and initial std to 0.8. C3 changes learning rate
to 1e-4 with a fixed schedule. C4 changes only approach distance from 0.65 m
to 0.40 m. All other parameters remain matched. The four-cell sweep is
A1/C1, C2, C3, C4. It targets Approach because that is already the first
observed learning failure. Higher-stage runs from scratch identify obvious
signal failures; zero success at this small budget does not prove a stage
unlearnable or measure transfer from a mastered previous stage.

B1 exposes a separate, labeled three-value privileged observation containing
body-frame goal offset and distance, to actor and critic. It is only an
Approach diagnostic. Deployable policies retain their original observations.

Training logs contain reward trajectory, episode length, falls, collision
intervals, button/gate/acquisition/placement counts, PPO losses and entropy,
action std/magnitude, command speed, and learning rate. Validation JSON and
`success-trajectory.json` retain the stage-specific outcomes.

D1 compares random, deterministic untrained, privileged goal-directed, and
the original update-400 checkpoint on 16 training layouts each. Instrumented
reward terms are checked against the unchanged production reward on every
decision. JSON includes mean/std/min/max, nonzero frequencies, absolute
contribution fractions and signed fractions of net return; Markdown presents
the readable table. Approach's only positive reward is terminal success.
Buttons, gates and pickup rewards are structurally inactive there.

Observation audits use paired physical trajectories with modality masks,
actual ray captures at junctions versus goals, validity/finite statistics,
temporal changes, near-constant features, paired-eye correlation, separate
encoder output statistics, and fusion weights. Paired depth is idealized ray
depth, not RGB stereo. These feature checks cannot establish sensor utility.

The layout audit reports all 352 layouts' route lengths, branch decisions,
turns, dead ends, corridor clear widths, switch/door counts, object-to-drop
distance, open-gate centerline path lengths, approach spawn clearance and
cross-split edge differences. **Test layouts receive static topology checks
only: no test rollouts, reward results or hyperparameter selection.** The
centerline distance excludes physical detours to press switches.

D2 keeps full-route Doors and Transport reports separate: eight training
layouts ×two reset seeds each. Its route controller uses the allowed 0.40 m/s
speed cap. A short preflight on layout 1 showed that the 0.28 m/s lateral
command barely moved, whereas 0.40 m/s reached the next cell. This is evidence
of command sensitivity, not a full-route feasibility pass.

D2 also runs eight layouts for each localized component: first door, second
door, and acquisition/carry/release/drop-zone dwell. These reset near the
interaction, then use ordinary physical steps and intentional actions;
there are no in-episode teleports or forced gate-success flags. Transport
places the initial parcel within pickup reach near the final corridor.
Component success requires acquisition, at least 0.4 m carried, release and
one second settled in-zone. Its dwell is sampled every 0.2 s, compared with
the production scorer's 0.04 s. Component success is reported separately from
full-mission success and can identify mechanics failures even when navigation
cannot reach them. Carrying remains a kinematic attachment abstraction;
the task measures a floor drop zone, not true grasping or shelf manipulation.

## Artifacts and reproduction

Root: `results/mission7-overnight-20260920/`. Each named study has its own
directory (`A1` … `D2`). `matrix.json` contains config hashes, commit, task IDs,
dependencies and expected artifacts. `submissions.jsonl` records exact sbatch
arguments, source hashes and submission timestamps. Frozen source is under
`source/` and checked against SHA256 at each task's startup. Qualification
records tests, smoke results and input hashes in `qualification.json`.

The submission helper defaults to dry run and refuses duplicate campaigns.
Source qualification must match before it submits:

```bash
PY="${PYTHON:-python3}"
PYTHONPATH="$PWD/src" "$PY" scripts/submit_mission7_overnight.py \
  --campaign results/mission7-overnight-20260920
# The authorized submission adds --submit after qualification.
```

Slurm IDs and live-state observations are appended below after submission and
to [SLURM_JOBS.md](../SLURM_JOBS.md). Status vocabulary: VALIDATED means local
execution checks passed; SUBMITTED means a scheduler receipt exists; RUNNING
means observed in Slurm; COMPLETED means the diagnostic executed; FAILED
requires its exit/error reason. Learning success must always be stated
separately. The full 60-training/108-evaluation campaign remains unsubmitted.

## Submitted receipts and startup observation

All **14 tasks were submitted** in three arrays on September 20 at 17:57 PDT.
The source passed 160 tests, 15 final targeted checks and finite execution
smokes. Slurm confirmed two CPUs and 12 GB per running task, with no GPU TRES.

| Study | Slurm task | Observed startup state | Dependency |
|---|---|---|---|
| A1 / C1 | `21370064_0` | RUNNING, cn-b01 | none |
| A2 | `21370064_1` | RUNNING, cn-b02 | none |
| A3 | `21370064_2` | RUNNING, cn-b02 | none |
| A4 | `21370064_3` | SUBMITTED / PENDING, array throttle | none |
| A5 | `21370064_4` | SUBMITTED / PENDING, array throttle | none |
| B1 | `21370064_5` | SUBMITTED / PENDING, array throttle | none |
| C2 | `21370064_6` | SUBMITTED / PENDING, array throttle | none |
| C3 | `21370064_7` | SUBMITTED / PENDING, array throttle | none |
| C4 | `21370064_8` | SUBMITTED / PENDING, array throttle | none |
| S1 | `21370066_9` | SUBMITTED / PENDING, dependency | `afterok:21370064_8` and C4 JSON learning gate |
| S2 | `21370066_10` | SUBMITTED / PENDING, dependency | same |
| S3 | `21370066_11` | SUBMITTED / PENDING, dependency | same |
| D1 | `21370065_12` | RUNNING, cn-b03 | none |
| D2 | `21370065_13` | RUNNING, cn-b03 | none |

These are dated observations; the scheduler and result JSON provide current
status. The jobs execute their frozen snapshot through sbatch and survive the
desktop ending. Unrelated desktop and prune jobs were not modified.

Early output audit, while the arrays were still running: A1 completed initial
validation and wrote its first finite PPO update. D1 completed the static
layout audit: all 352 topology hashes are unique; the closest validation/test
topologies differ from training by 12/14 edges in symmetric difference.
D2's localized component reports completed: first door **6/8**, second door
**7/8**, and acquisition/carry/release/dwell **7/8**. Door failures were three
falls; transport acquired in 8/8 and had one excessive-contact failure.
These are privileged localized trials; full-route D2 rollouts and all learned
results remain pending. They are not full-mission or learned-policy successes.

```bash
# All original and new Mission7 jobs, in one query:
squeue -r -j 21369796,21369990,21369991,21370064,21370065,21370066 \
  -o '%.18i %.20j %.10T %.12M %.35R'
sacct -X -j 21369796,21369990,21369991,21370064,21370065,21370066 \
  --format=JobID%20,State,ExitCode,Elapsed,AllocTRES%55

# Per-study outcomes, including any conditional skips:
python - <<'PY'
import json
from pathlib import Path
root = Path('results/mission7-overnight-20260920')
for row in json.loads((root/'matrix.json').read_text())['rows']:
    p = root/row['name']/'result.json'
    r = json.loads(p.read_text()) if p.exists() else {}
    print(row['name'], row['slurm_id'], r.get('status', 'No final result yet'),
          'learning_gate=', r.get('learning_gate_passed', 'n/a'))
PY

tail -n 5 results/mission7-overnight-20260920/A1/learning.jsonl
cat results/mission7-overnight-20260920/D1/rewards.md
cat results/mission7-overnight-20260920/D1/layouts.md
```
