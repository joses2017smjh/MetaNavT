# Mission7 Approach diagnosis — September 21, 2026

This phase separates command execution, reward/exploration and PPO stability.
It does not release the sensor comparison or full Mission7 campaign. All
debugging uses training/validation layouts; no held-out test layouts are read
by these new studies. Previous checkpoints, raw results and frozen sources
are preserved. Results: `results/mission7-approach-debug-20260921/`.

## Current decision — September 21

The measured recovery controller passed the fixed feasibility gate: **54/64
successes (84.4%), zero falls**, using the unchanged 0.65 m Approach start,
command limits, geometry and scorer. The simpler translation controller
scored 46/64; bearing/turn/forward scored 5/32. One 0.30 m/s forward pulse
lasting 0.40 s on a stalled positive-lateral start resolved all eight
translation timeouts. The remaining ten failures are excessive goal-post
contacts, not falls. Recovery median initial/minimum/final distance is
**0.653/0.165/0.183 m**; median fraction closed is 74.7%.

This is feasibility on these validation layouts after a targeted controller
revision, not an independent final generalization estimate. Both reset seeds
use the same 32 layouts. The gate remained ≥75% success and ≤10% falls.
`feasibility-gate.json` records source/result hashes and the prerequisite jobs.

All six privileged PPO controls completed. Final success counts were
**P1–P6: 0, 0, 0, 2, 0, 0 /16**. P6 reached 5/16 at update 150 before
regressing to 0/16. Dense reward produced partial progress and direction-specific
successes, but **no cell met the unchanged stable-learning gate**. Sensor-only
Both, the four-sensor comparison and the full campaign remain gated.

All 18 retained scheduler tasks completed, with one invalid cross-architecture
replay cancelled and replaced by a verified same-node retry. The campaign
used **11.96 allocated CPU-hours and zero GPUs**. Full-route modifications
improved success modestly but failed the majority target. Exact paired replay
identified pressure-plate contact as a cause of all ten selected Doors falls.
The result tables and next steps below are final for this bounded phase.

## Measured command response

The frozen gait's deployment configuration uses 40 ms policy updates and
0.5 ms physics. Mission7 makes a new decision every five gait updates
(200 ms). Its command limits are vx ±0.40 m/s, vy ±0.35 m/s and yaw ±0.40 rad/s.
The original gait training configuration permits vx ±1, vy ±0.5, yaw ±1.5;
training-range probes outside the Mission7 cap are labeled separately.

The first gait array tests 16 commands ×seven durations ×three reset seeds
= **336 open-floor trials**. Durations are 0.08/0.20/0.40/0.52/0.80/1.00/2.00 s,
all exact multiples of 40 ms (0.52 s is the nearest upper multiple to 0.5 s).
Every trial includes 1.2 s settling and 3 s zero-command braking. Obstacles
are disabled only in this identification fixture, not in the benchmark.
Actual velocity, displacement, heading, drift, braking distance/path/time,
fall/recovery and the full 40 ms trace are retained.

Mean displacement at a **two-second hold**, over three trials:

| Command | Along displacement | Interpretation |
|---|---:|---|
| Forward 0.10 m/s | 0.009 m | Almost stationary |
| Forward 0.20 m/s | 0.019 m | Almost stationary |
| Forward 0.30 m/s | 0.534 m | Sustained walking; mean brake displacement 0.011 m |
| Forward 0.40 m/s | 0.770 m | Sustained walking; mean brake displacement 0.102 m |
| Lateral +0.20 m/s | 0.016 m | Almost stationary |
| Lateral −0.20 m/s | 0.200 m | Directional asymmetry |
| Lateral +0.35 m/s | 0.443 m | Sustained motion |
| Lateral −0.35 m/s | 0.554 m | Sustained motion; larger braking drift |

Pure yaw ±0.35 rad/s produced only about +0.036/−0.077 rad in two seconds,
far below the commanded ±0.70 rad. A turn-then-forward controller cannot
assume that the requested heading rate will be executed. No falls occurred
in these 336 short open-floor trials; this does not establish collision-rich
maze reliability or recovery after a fall. There are only three trials per
command/duration cell. Full mean/std tables and figures:

- [Gait table](../results/mission7-approach-debug-20260921/gait-response.md)
- [Gait figure](../results/mission7-approach-debug-20260921/gait-response.png)
- [Exportable PDF](../results/mission7-approach-debug-20260921/gait-response.pdf)

`gait-summary.json` distinguishes first threshold crossing from sustained
latency: three consecutive 40 ms samples above half the commanded speed
(at least 0.05 m/s). Short transient sways are not evidence of sustained
command execution. The 48-trial follow-up also completed with zero falls. Two-second backward
−0.30/−0.40 m/s commands moved 0.629/0.803 m. Even ±1.2 rad/s yaw commands
produced only +0.148/−0.250 rad, so increasing the yaw cap alone is not a
supported solution. These yaw probes lie inside the original gait training
range but outside the current Mission7 cap.

For a 0.30 m/s forward command, mean displacement by hold duration was:

| Hold | 0.08 s | 0.20 s (current) | 0.40 s | 0.52 s | 0.80 s | 1.00 s | 2.00 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Displacement | 0.021 m | 0.026 m | 0.073 m | 0.106 m | 0.195 m | 0.219 m | 0.534 m |

The 0.80 s and 2.00 s trials met the sustained-response criterion after
0.24–0.28 s in all three resets. No 0.20 s pulse met it. This measures latency
from a settled start, not a fixed delay for every command during walking.
Braking is also phase dependent: the 0.30 m/s, 1 s pulse drifted 0.064 m after
release versus 0.011 m after the 2 s pulse. All open-floor trials ended upright;
recovery from a fall remains unmeasured because none fell.

## Success and reward audit

The unchanged Approach scorer requires:

- Base center strictly within **0.36 m** of the goal.
- Tilt below **0.78 rad** and planar base speed below **0.20 m/s**.
- **One continuous second** satisfying those conditions, checked every 40 ms.
- Timeout at **18 s**; an earlier fall or more than five contact intervals
  terminates with failure. Wall, gate and goal-post contacts count.

Entering briefly resets dwell on exit. Falling before completion is failure,
even after entering the target. Once a full valid dwell completes, the episode
ends; the scorer makes no claim about hypothetical falls after termination.
Deterministic tests cover inside/upright, outside, fallen, brief crossing,
excess speed and excessive contact. Instrumentation was also checked against
the original environment for identical observations, physics and sparse reward.

| Reward term | Original | Dense diagnostic | Approximate Approach episode bound |
|---|---|---|---|
| Time | −0.01 ×seconds | Same | −0.18 (−0.184 at long-hold timeout) |
| Wall/gate/post contact | −0.2 per 40 ms contact interval | Same | −1.2 before six-interval termination |
| Fall/excessive-contact failure | −5 once | Same | −5 |
| Terminal success | +10 once | Same | +10 |
| Goal-distance progress | Absent | **10 ×(previous distance −new distance)** | Net positive ≤10 ×initial distance, approximately +6.5 for the 0.65 m start |
| Upright/survival, velocity, heading, action penalty | Absent | Absent | 0 |
| Buttons/gates/acquisition | Inactive in Approach | Same | 0 |

Only one privileged shaping term is introduced. Its increments telescope,
so moving away and returning cannot repeatedly earn net progress reward.
There is no large upright bonus hiding the goal objective. Dense reward is a
diagnostic, not deployable sensor-driven evidence. Negative progress is not
clipped; measured contributions will be reported alongside the bound.

The offline audit covers all **1,386 recorded training episodes**. It reports
per-term mean/median/std, absolute and signed contribution fractions, and
correlations with terminal goal distance and sampled partial goal entry.
Distance progress and survival rewards were absent, not merely small.
Previous logs retained episode term totals rather than per-decision terms;
exact nonzero frequencies are reported where recoverable and otherwise
bounded. The roughly one-second old trajectories can miss brief goal entry.
Constant-variable correlations are undefined and reported as null.

- [Existing reward audit](../results/mission7-approach-debug-20260921/offline-audit/existing-reward-audit.md)
- [Machine-readable audit](../results/mission7-approach-debug-20260921/offline-audit/existing-reward-audit.json)

Standing still is not rewarded with survival credit: its expected sparse
return is approximately −0.18 when it times out. It can nevertheless dominate
unsuccessful movement that incurs collisions/falls. The matched comparison
tests standstill, random, scripted translation, untrained PPO and C4 update
100, on validation layouts 0–15 at both 0.65 and 0.40 m starts. Results are
kept separate by start distance. All 160 matched controller episodes completed:

| Controller | 0.65 m success | Mean sparse return | 0.40 m success | Mean sparse return |
|---|---:|---:|---:|---:|
| Standstill | 0/16 | −0.1800 | 2/16 | +1.0906 |
| Random | 0/16 | −2.8542 | 0/16 | −2.1179 |
| Scripted translation | 11/16 | +5.9694 | 13/16 | +8.0607 |
| Untrained PPO | 0/16 | −0.1800 | 1/16 | +0.4553 |
| Preserved C4 update 100 | 0/16 | −0.9391 | 3/16 | +0.9676 |

All cells had zero falls. Standing still is a local safe alternative to
failed exploration, not the global return optimum. At the easier 0.40 m
start, settling alone can satisfy the scorer on some layouts: C4's small
success count therefore does not establish learned navigation. C4 even has
lower mean sparse return than standstill there because of contacts. The exact
matched success sets were standstill layouts {5,13}, untrained {13}, and C4
{2,5,13}; C4 adds just one completion over settling on this 16-layout subset.
[Matched return plot](../results/mission7-approach-debug-20260921/rewards/controller-returns.png).

## C4 and PPO dynamics

Exact C4 checkpoints 75/100/125 were copied under `preserved-c4/` and checked
against SHA256. Each is evaluated on the same 32 fixed validation layouts,
with the original 0.40 m spawn and unchanged policy observations. The original
files remain untouched. Every new diagnostic records initial/minimum/final
distance, fraction closed and time of closest approach at 40 ms resolution.

C4 entropy and action std did not collapse around update 100: mean entropy
was 3.693 at updates 76–100, 3.703 at 101–125 and 3.708 at 126–150; action std
stayed approximately 0.499–0.520. There was no curriculum transition. Value
loss varied, but these logs alone do not establish a catastrophic update.
**KL, clip fraction, explained variance and gradient norms were not recorded**
in the old runs, so no retrospective causal claim is made from missing data.

The expanded checkpoint evaluation completed:

| C4 update | Success /32 | First eight layouts | Median final distance | Median closest distance | Falls | Excessive-contact episodes |
|---|---:|---:|---:|---:|---:|---:|
| 75 | 5/32 | 1/8 | 0.410 m | 0.383 m | 0 | 0 |
| 100 | 6/32 | 2/8 | 0.416 m | 0.389 m | 0 | 5 |
| 125 | 3/32 | 0/8 | 0.407 m | 0.380 m | 0 | 0 |

All successful episodes across these checkpoints have the goal in the world
−x direction: 5/7, 6/7 and 3/7 respectively, with **0/25 in the other three
directions** at every checkpoint. This is a direction-specific outcome, not
general goal-seeking competence. At update 100 median fraction of initial
distance closed was only 2.7%. The repeated first-eight result is real under
the fixed reset protocol, but it was a fragile near-boundary signal. The
matched standstill comparison tests whether settling/body bias explains it;
entropy collapse and curriculum transition are not supported explanations.

- [C4 optimization curves](../results/mission7-approach-debug-20260921/offline-audit/c4-dynamics.png)
- [C4 action std](../results/mission7-approach-debug-20260921/offline-audit/c4-action-std.png)
- [All-run PPO audit](../results/mission7-approach-debug-20260921/offline-audit/existing-ppo-audit.json)

The new PPO runner records analytic Gaussian KL and clip fraction on the
rollout after each complete update, explained variance before/after, actor
parameter change, gradient norms before/after clipping, losses, entropy,
action std and learning rate. Post-update KL/clip statistics are not averages
over the optimizer's individual minibatches. Validation includes success,
distance distributions, progress and falls at updates 0/25/50/100/150/200.

One metadata issue in the already-frozen C4 evaluator is documented in
`c4-metric-schema-note.json`: its `hold_s` field reports action duration rather
than dwell. This does not affect actions, observations, scoring or distance
metrics. Later source versions preserve scoring `hold_s` and use
`action_hold_s` separately; the frozen job and raw results were not rewritten.

## Decision gates and scope

1. Evaluate the simple bearing/turn/forward baseline on 32 validation layouts
   and planar translation within unchanged command bounds on 64 episodes
   (32 layouts ×two reset seeds).
2. Require at least 75% translation success and at most 10% falls before PPO
   release. Also require complete gait and reward audits. If this fails,
   work on the physical interface first; no sensor comparison is released.
3. Compare four small seed-0 privileged Approach training cells: sparse
   baseline; dense progress; dense progress with a measured longer hold;
   dense progress with conservative PPO updates. Exact hold and configuration
   hashes are recorded in `training-matrix.json` before submission.
4. A within-seed learning signal requires ≥5/16 successes in both final
   validation checkpoints, ≥2/16 improvement and successful training episodes.
   This still requires replication, then sensor-only Both learning, before
   any Blind/LiDAR/depth/Both comparison.

The full-route diagnostic keeps the task/geometry/scorer unchanged and compares
the original script with slower measured command limits plus a 0.4 s pause at
cell transitions and interactions. It uses 16 paired validation layouts for
each controller and each of Doors/Transport (64 episodes). High-rate traces
classify falls by nearby contacts, acceleration/braking/turning and interaction
phase. These are temporal associations, not guaranteed causal explanations.
Older full-route failures receive a separate coarse trace audit; exact contact
histories cannot be recovered from their one-second samples.

All long studies run through Slurm with 2 CPUs, 12 GB and zero GPUs, with a
two-hour per-task cap. Receipts include git commit, exact command, hashes,
dependencies and immutable per-mode source snapshots in `submissions.jsonl`.
The [Slurm ledger](../SLURM_JOBS.md) is updated at every submission. Actual
job states and outcome tables are appended as the studies finish.

## Released Approach PPO matrix

The initial four cells use seed 0, fixed Approach, the same train layouts and 16 fixed
validation layouts, and Both sensor features plus the privileged goal vector.
They have 200 updates and equal **nominal** 2,560 simulated seconds of training.
Actual seconds are logged because termination can interrupt a held action.
Validation is at updates 0/25/50/100/150/200; checkpoints are saved every 25.

| Array task | Cell | Reward | Hold | Horizon | PPO |
|---|---|---|---:|---:|---|
| 21384285_0 | P1 | Original sparse | 0.20 s | 64 | LR 3e−4 adaptive, 4 epochs |
| 21384285_1 | P2 | +10 ×distance progress | 0.20 s | 64 | Same as P1 |
| 21384285_2 | P3 | Same dense progress | 0.80 s | 16 | Same LR/epochs; gamma/lambda exponentiated ×4 |
| 21384285_3 | P4 | Same dense progress | 0.20 s | 64 | LR 1e−4 fixed, 2 epochs |

The 0.80 s hold comes from measured sustained response versus the small
0.20 s displacement. P3 also spans four times as much physical time in its
four-frame history and uses fewer decisions per update. It tests a temporal
interface configuration, so its result cannot isolate action persistence
from every associated sampling effect. The existing environment checks timeout
at decision boundaries: a 0.80 s hold can end at 18.40 s versus 18.00 s for
the original hold, another small confound reported rather than hidden.
Entropy coefficient is 0.01 and
initial action standard deviation is 0.5 throughout. Exact hashed configs
are in `training-matrix.json`.

Outputs are `P1/` through `P4/` under the campaign directory, including
`learning.jsonl`, `episodes.jsonl`, `validation-NNN.json`, `model_N.pt`,
`learning-curve.json` and final `result.json`. Training has no live scheduler
dependency because all prerequisite jobs already completed; the runtime
feasibility/reward guard and frozen release evidence enforce the logical
dependency. No dependent sensor jobs were submitted.

## Full-route follow-up

Job **21384354** applies the already-validated recovery pulse to the slower
route controller on the same 16 validation layouts for Doors and Transport
(32 episodes). It requires 1.2 s of positive-lateral commands with less than
0.05 m displacement and allows one 0.30 m/s, 0.40 s forward pulse per waypoint
and interaction phase. Scoring, geometry, timeout and command bounds remain
unchanged. This isolates a specific observed stall mechanism from the original
and slower baselines in job 21384077. Outputs: `fullroute_recovery/`.

Qualification: the earlier full suite passed 169 tests; after the two bounded
recovery revisions, all 11 diagnostic tests passed, including physical/reward
instrumentation equivalence, unchanged PPO updates/RNG under telemetry and
bounded recovery behavior. Short route and PPO runner smokes also passed.

<!-- mission7-debug-status:start -->
## Submitted diagnostic jobs

Snapshot: 2026-09-21T22:34:26.052020+00:00. All tasks: 2 CPUs, 12 GB, zero GPUs, two-hour cap.
13 submissions /19 tasks; 11.96 allocated CPU-hours so far.

| Job | Study | Scheduler state | Scheduler dependency | Output under campaign |
|---|---|---|---|---|
| 21383807_0 | gait | COMPLETED | none | `gait-0/` |
| 21383807_1 | gait | COMPLETED | none | `gait-1/` |
| 21383807_2 | gait | COMPLETED | none | `gait-2/` |
| 21383808 | c4 | COMPLETED | none | `c4/` |
| 21384040 | gait_extra | COMPLETED | none | `gait_extra/` |
| 21384041 | approach | COMPLETED | afterok:21383807 | `approach/` |
| 21384042 | rewards | COMPLETED | afterok:21383807 | `rewards/` |
| 21384077 | fullroute | COMPLETED | afterok:21384040 | `fullroute/` |
| 21384200 | approach_recovery | COMPLETED | none | `approach_recovery/` |
| 21384285_0 | train | COMPLETED | none | `P1/` |
| 21384285_1 | train | COMPLETED | none | `P2/` |
| 21384285_2 | train | COMPLETED | none | `P3/` |
| 21384285_3 | train | COMPLETED | none | `P4/` |
| 21384354 | fullroute_recovery | COMPLETED | none | `fullroute_recovery/` |
| 21384572_0 | train_exploration | COMPLETED | none | `P5/` |
| 21384572_1 | train_exploration | COMPLETED | none | `P6/` |
| 21384641 | gait_endurance | COMPLETED | none | `gait_endurance/` |
| 21384691 | fall_replay | CANCELLED by 19646 | none | `fall_replay/` |
| 21384742 | fall_replay_retry | COMPLETED | none | `replay-on-original-node/fall_replay/` |

PPO has completed logical prerequisites in `feasibility-gate.json` and a runtime guard; no expired scheduler dependency was attached. No sensor jobs were submitted.

## Current controlled-study outcomes

Read from result files at 2026-09-21T22:34:26.052020+00:00. Incomplete cells remain pending.

| Cell | Completed update | Initial success /16 | Latest success /16 | Median final distance | Training successes | Stable within seed |
|---|---:|---:|---:|---:|---:|---|
| P1 | 200 | 0 | 0 | 0.662 m | 0 | False |
| P2 | 200 | 0 | 0 | 0.634 m | 0 | False |
| P3 | 200 | 0 | 0 | 0.654 m | 9 | False |
| P4 | 200 | 0 | 2 | 0.643 m | 0 | False |
| P5 | 200 | 0 | 0 | 0.507 m | 0 | False |
| P6 | 200 | 0 | 0 | 0.548 m | 1 | False |

| Route controller | Stage | Episodes available | Success | Falls | Complete |
|---|---|---:|---:|---:|---|
| legacy | doors | 16 | 2 | 10 | True |
| legacy | transport | 16 | 1 | 10 | True |
| measured | doors | 16 | 4 | 3 | True |
| measured | transport | 16 | 1 | 7 | True |
| recovery | doors | 16 | 5 | 7 | True |
| recovery | transport | 16 | 3 | 12 | True |
<!-- mission7-debug-status:end -->

## Geometry and interpretation limits

The ten remaining Approach contact failures all occur when the goal lies in
world −x. The visual goal posts are fixed at goal x+0.52 m and y±0.35 m,
with 0.045 m capsule radius, leaving 0.61 m between their surfaces. This
creates direction-dependent approach clearance for the full humanoid.
Successful episodes also exist from that direction; these results do not
prove that a layout is impossible. Geometry was left unchanged. Resolving
this marker/clearance issue requires a separately versioned geometry audit,
not silently deleting colliders from this campaign.

The high-rate fall table samples named wall/gate/post contacts every 40 ms.
It cannot exclude a brief contact between samples or prove a cause; raised
button-plate contacts are not among those named collision traces. Phase and
command-change associations therefore remain diagnostic classifications,
with an explicit “other” category, not a causal reconstruction.

## Reproducing this report

The follow-up contact-safe and privileged-control work is tracked separately in
[`MISSION7_APPROACH_FOLLOWUP.md`](MISSION7_APPROACH_FOLLOWUP.md), including its
pose-matched contact probe, route evaluation, four-direction matrix, and gate
status. The sensor comparison remains closed there until both control gates
pass.

Use the existing CPU environment and repository-local output directory:

```bash
PYTHONPATH="$PWD/src:$PWD/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR="$PWD/results/mission7-approach-debug-20260921/mpl-cache" \
python3 scripts/mission7_debug_report.py \
  --campaign results/mission7-approach-debug-20260921
python3 scripts/mission7_debug_status.py \
  --campaign results/mission7-approach-debug-20260921 --accounting --docs
```

These commands analyze recorded artifacts and refresh status; they do not
submit experiments. The submit helper rejects duplicate mode submissions.
All three preserved C4 checkpoints still match both their manifest and their
original files. All 296 files across the thirteen submitted immutable source
snapshots passed SHA256 verification; the gait assets, original/preserved C4
checkpoints, and six executed configuration hashes were also verified.

Additional artifacts: [controller paths](../results/mission7-approach-debug-20260921/approach/controller-paths.png),
[fall classifications](../results/mission7-approach-debug-20260921/fall-events.md),
[fall heatmap](../results/mission7-approach-debug-20260921/fall-heatmap.png),
[new PPO diagnostics](../results/mission7-approach-debug-20260921/new-ppo-dynamics.png),
[new reward decomposition](../results/mission7-approach-debug-20260921/new-reward-audit.md).
Figures also have standalone PDFs where applicable. Raw traces retain exact
initial/minimum/final distance, fraction closed and time of closest approach.

## Final two exploration controls

After the first completed update-50 evaluations, all P1/P2/P3 deterministic
commands remained below the isolated-command ranges that produced sustained
walking; P4's completed update-25 evaluation showed the same pattern. This
motivated a single exploration change, consuming the remaining two slots in
the requested **maximum six-cell PPO study**. The first four configs and
running sources were not modified.

| Array task | Cell | Matched reference | Only configured change |
|---|---|---|---|
| 21384572_0 | P5 | P2, dense /0.20 s hold | Initial Gaussian action std 0.5 →0.8 |
| 21384572_1 | P6 | P3, dense /0.80 s hold | Initial Gaussian action std 0.5 →0.8 |

A 0.30 m/s forward command requires raw action `atanh(0.30/0.40)=0.973`.
For a zero-mean Gaussian, the probability of exceeding that magnitude in one
sample rises from **5.17% to 22.39%**. This calibrates exploration support;
it is not a success probability or the probability of sustained walking.
Entropy coefficient, observations, reward, seed, layout sequence, validation
set and optimization settings match each reference. Configs and this rationale
were frozen before submission in `training-matrix-exploration.json`.

The long-hold std-0.8 smoke produced finite losses/telemetry and verified
learned std near 0.8; all 11 diagnostic tests passed after the runner change.
P5/P6 have the same completed feasibility/reward prerequisites and runtime
guard, with no live scheduler dependency. Outputs are `P5/` and `P6/`.
No further training cells or sensor studies are released in this phase.

## Sustained-command follow-up

Job **21384641** tests eight straight/mixed commands for 5 and 10 seconds over
three reset seeds (**48 trials**) in the same open-floor fixture. Commands
are forward/backward 0.30 m/s, lateral +0.35/−0.30 m/s, each lateral command
with vx +0.04 m/s, and each lateral command with yaw +0.10 rad/s. Every command
is inside the unchanged Mission7 caps. Each trial retains 1.2 s settling and
3 s zero-command recovery observation.

This was triggered by 0/384 falls in the short gait trials versus frequent
falls on full routes. It tests duration and mixed-command stability without
route transitions, walls or button plates; it does not change or retrain the
gait. Sources are frozen separately; output is `gait_endurance/`, with a
postprocessed table in `gait-endurance.md`. The campaign now has 11 Slurm
submissions, including arrays; only six tasks perform PPO training.

## Counterfactual replay of route falls

Job **21384691** replays the ten original Doors failures at their recorded 40 ms
command cadence. It recreates each reset's RNG sequence and recorded gate
opening times. Three paired variants retain all geometry, disable only
pressure-plate collisions, or disable all obstacle collisions. These are
**30 replay trials**, ending at each original fall time; they do not count as
new route completions. The unchanged replay must match the recorded xy,
yaw and tilt throughout to within 1e−6 before intervention results are
interpreted. A short three-variant replay smoke passed exact baseline matching.

This tests the contrast between frequent route falls and the stable isolated
command trials. Benchmark files and ongoing PPO tasks remain unchanged.
Output: `fall_replay/`. The final campaign is bounded at **12 Slurm submissions
including arrays, 18 tasks, and six PPO training cells**. No further studies
are released in this phase.

The initial replay 21384691 failed its required trajectory-identity check
and was cancelled with partial files preserved. It ran on cn-b01
(Ivy Bridge/AVX), while the original route ran on cn-c22 (Haswell/AVX2).
Tiny early differences grew during the long recorded command sequence.
Retry **21384742** requests the exact original node cn-c22 and stops
before interventions if full-trajectory matching fails. Output:
`replay-on-original-node/fall_replay/`. This is one corrective retry of the
same diagnostic, not another scientific condition: **12 planned submissions
plus one retry, 19 scheduler tasks total**. See `replay-retry-note.json`.

## Completed P1–P4 evidence

| Cell | Final success /16 | Training successes | Median final distance | Mean fraction closed at closest approach |
|---|---:|---:|---:|---:|
| P1 sparse | 0 | 0 | 0.662 m | 4.2% |
| P2 dense | 0 | 0 | 0.634 m | 28.8% |
| P3 dense /0.80 s hold | 0 | 9 | 0.654 m | 7.7% |
| P4 dense /conservative update | 2 | 0 | 0.643 m | 14.3% |

All earlier P4 checkpoints were 0/16. Its final successes were layouts 12
and 14, both with the goal toward world −y; they closed 49.8% and 61.9%
of initial distance and completed the actual one-second dwell. This is a
weak final-checkpoint signal, not stable or repeatable Approach learning.
P3's nine terminal training successes also did not transfer to deterministic
validation. None of P1–P4 met the predeclared learning gate.

P2 entered the goal circle in five final validation episodes: four subsequently
terminated for excessive contact, while layout 15 timed out after spending
3.28 s inside the circle but achieving at most **0.68 s continuous valid dwell**.
P4's two goal entries both completed a valid dwell. The report independently
reconstructs dwell from 40 ms distance/tilt/velocity traces and checks it against
the recorded success for every completed PPO validation episode.
See `validation-arrival-audit.json` and `validation-command-audit.md`.

Median post-update telemetry over updates 151–200:

| Cell | Analytic KL | Clip fraction | Explained variance after update | Gradient norm before clipping | Actual LR |
|---|---:|---:|---:|---:|---:|
| P1 | 0.0304 | 40.6% | 0.0034 | 6.85 | 1e−5 |
| P2 | 0.0297 | 41.4% | 0.0031 | 9.78 | 1e−5 |
| P3 | 0.0434 | 56.2% | 0.0405 | 21.34 | 1e−5 |
| P4 | 0.0112 | 17.2% | 0.0107 | 12.26 | 1e−4 |

Action std stayed roughly 0.49–0.52, with finite model parameters and losses.
There is no entropy/variance collapse. Adaptive cells often reach the LR
floor yet still have substantial post-update clipping; critics explain little
return variation. P4's fewer epochs/fixed LR reduce measured KL and clipping,
but one seed and two late successes do not establish the optimizer as the
sole cause. Reward sparsity, command dead zones, arrival control and weak
optimization all remain relevant. No training extension is released.

CPU provenance matters: P1–P4 ran on Haswell/AVX2; P5/P6 were allocated
Ivy Bridge/AVX. The configured comparison is matched, but hardware is an
additional uncontrolled factor. The failed cross-architecture replay shows
that tiny numeric differences can grow in long physical trajectories.
Do not attribute an outcome difference solely to exploration std. Any next
replication should pin the same CPU class/node; `slurm-accounting.psv` and
`node-features.psv` retain this provenance.

## Final physical findings

All **432 open-floor trials** completed without a fall: 336 original response
trials, 48 backward/yaw probes and 48 endurance trials. Thus the tested static
commands alone, including 5/10 s mixed commands, do not reproduce route falls.
This does not prove that arbitrary command transitions or contacts are safe.
The [endurance table](../results/mission7-approach-debug-20260921/gait-endurance.md)
retains every tested command, duration, recovery and final-upright outcome.

The paired full-route validation results are:

| Controller | Doors success | Doors falls | Transport success | Transport falls |
|---|---:|---:|---:|---:|
| Original | 2/16 | 10/16 | 1/16 | 10/16 |
| Slower commands + pauses | 4/16 | 3/16 | 1/16 | 7/16 |
| Same + measured stall recovery | 5/16 | 7/16 | 3/16 | 12/16 |

These are new validation layouts; the earlier 4/16 results used training
layouts and are not the matched reference for this table. Neither modification
achieves majority route completion. Recovery resolves some starts but increases
falls relative to the slower controller. Resolving stalls also exposes more
episodes to later contacts, so the count does not isolate the pulse itself
as the immediate cause. Across these 96 route episodes there
were 49 falls; the time/command/phase table retains every event.

The **same-node counterfactual replay** is decisive for the ten selected
original Doors falls: all ten unchanged trajectories match exactly (maximum
recorded pose difference **0.0**); all ten fall. Disabling only pressure-plate
collisions prevents **all ten falls** through their original termination time.
Disabling all obstacle collisions also gives **0/10 falls**. This supports a
causal contribution from pressure-plate contact under these recorded commands,
beyond the earlier phase-only fall classifications. It does not establish
that every other fall has the same cause, or that the altered scene completes
a valid mission: gate-opening times were replayed, and intervention trials
stop at the original fall time. No benchmark geometry was changed.

The existing plates protrude **0.03 m above the floor**. The next physical
work should isolate safe plate entry/exit and gait response to that contact,
or validate a separately versioned geometry correction while preserving real
button activation and the sensor cue. Removing plate collisions is a causal
probe, not a valid solution to the mission. Topological layout checks and
successful local interactions do not prove full-body clearance; no impossible
layout is established by these diagnostics.

[Paired replay table](../results/mission7-approach-debug-20260921/fall-replay.md)
— raw verified replay: `replay-on-original-node/fall_replay/`.

## Final exploration results and decision

| Cell | Final success /16 | Best checkpoint | Training successes | Median final distance | Mean fraction closed | Actual training seconds |
|---|---:|---|---:|---:|---:|---:|
| P5 dense /0.20 s /std 0.8 | 0 | 0/16 throughout | 0 | 0.507 m | 36.5% | 2,544.28 |
| P6 dense /0.80 s /std 0.8 | 0 | 5/16 at update 150 | 1 | 0.548 m | 18.1% | 2,440.96 |

P6's five update-150 successes were all world −y goals: 5/6 in that direction,
0/10 in the other directions. All five disappeared at update 200. This is
partial, unstable directional learning, not repeatable target-conditioned
Approach competence. P5 improved distance but never completed validation.
The fixed six-cell budget is exhausted; no run was extended to chase success.

P6 entropy stayed around 6.01 and action std around 0.78–0.83 around the
regression. Median post-update clip fraction remained **62.5%**; maximum
post-peak analytic KL was **0.168 at update 161**. Median explained variance
was only 0.047 at updates 151–175 and 0.027 at 176–200. These observations
support noisy/large updates and weak value fitting as concerns; they do not
prove which individual update caused the loss. There was no curriculum
transition or entropy collapse. Copies of P6 checkpoints 150/175/200 and
SHA256 hashes are preserved under `preserved-p6/`; the detailed windows are
in `p6-regression-audit.json`.

The evidence now answers the diagnostic questions:

- **A — Physical Approach:** yes on the fixed validation set with an explicit
  calibrated controller and bounded stall recovery (54/64, no falls). Current
  commands have dead zones and weak yaw; goal posts create asymmetric clearance.
- **B — Standstill:** no survival bonus or global standstill optimum. Sparse
  penalties make it preferable to unsuccessful random motion locally.
- **C — Dense PPO:** measurable distance improvement and transient directional
  successes, but no stable or repeatable Approach learning in the six cells.
- **D — C4:** a fragile near-boundary, world −x result; standstill reproduces
  part of it. Expanded evaluation confirms limited behavior. Missing original
  KL/clip/gradient metrics prevent an exact causal attribution of its regression.
- **E — Falls/overshoot:** zero falls in 432 isolated trials; braking and
  direction asymmetry are measured. Plate contact caused the ten selected
  full-route failures in a validated intervention replay.
- **F — Full routes:** improved to at best 5/16 Doors and 3/16 Transport;
  majority completion remains unproven. Contact handling must improve before
  treating later full missions as reliable.

Next work should preserve this benchmark version and isolate two issues:
(1) safe pressure-plate entry/exit with physical activation and sensor cues
intact; (2) privileged goal-conditioned control across **all four directions**,
including braking/dwell, with a calibrated command interface and controlled
optimization. Use the same CPU class for paired physical comparisons. Any
promising Approach policy then needs independent seed replication and stable
validation, followed by **sensor-only Both** learning. Only after those gates
may Blind/LiDAR/paired-depth/Both resume. These next studies are not submitted.

The historical training audit also found exactly **three prior PPO falls**,
all in A2. Their recorded commands, timing and explicitly coarse phase
associations are preserved in
[the historical fall table](../results/mission7-approach-debug-20260921/offline-audit/previous-training-falls.md).
The older D2 route failures remain in `offline-audit/previous-route-failures.json`.
New PPO training/validation had no falls; its failures were contacts/timeouts.

[Final learning curves](../results/mission7-approach-debug-20260921/approach-learning-curves.png)
([PDF](../results/mission7-approach-debug-20260921/approach-learning-curves.pdf));
[final machine-readable summary](../results/mission7-approach-debug-20260921/final-summary.json).
