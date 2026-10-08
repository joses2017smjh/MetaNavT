# Navigation and coordination evidence — 20 September 2026

This dated audit records completed navigation/cooperation results, not a claim
that every weekend track succeeded. Folding has its own
[status and evidence](CLOTH_FOLDING_WEEKEND.md). The earlier failures remain
in [the job ledger](../SLURM_JOBS.md) and [maze history](MAZE_RIG.md).

## Isaac corridor recovery: completed

Slurm accounting reports **36/36 completed, exit code `0:0`** across Approach
`21359486`, Corridor `21359499`, and Full `21359510`. All 36 JSON gates passed,
all logs contain `MAZE_PROBE_PASS` and `MAZE_STAGE_PASS`, and all referenced
checkpoints exist. The arrays are absent from the live queue because they
finished. The last element ended `2026-09-20T01:34:07` in Slurm's reported
clock. No jobs were submitted, cancelled or requeued during this audit.

Each stage evaluates the first episode in **32 environments per checkpoint**,
avoiding a score inflated by repeatedly finishing easy episodes. Training
seeds 0/1/2 use evaluation seeds 100/101/102; every first episode completed.
Success requires a goal arrival held for eight control steps while upright
and moving slowly, rather than merely avoiding a fall. The progression gate
was 30%; every measured cell exceeded 90%.

| Arm | Approach seed 0 / 1 / 2 | Corridor seed 0 / 1 / 2 | Full seed 0 / 1 / 2 |
|---|---|---|---|
| Blind | 32 / 32 / 32 | 32 / 32 / 30 | 32 / 29 / 32 |
| Lidar | 31 / 32 / 32 | 30 / 31 / 32 | 32 / 32 / 32 |
| Paired ray depth (`Stereo`) | 31 / 32 / 32 | 32 / 30 / 32 | 32 / 32 / 30 |
| Both | 30 / 32 / 32 | 30 / 32 / 32 | 32 / 32 / 32 |
| Stage total, denominator 384 | 380 (98.958%) | 375 (97.656%) | 379 (98.698%) |

All entries in the arm rows are successful first episodes **out of 32**.
Full-stage aggregates and direct evidence:

| Arm | Aggregate | Seed 0 JSON | Seed 1 JSON | Seed 2 JSON |
|---|---:|---|---|---|
| Blind | 93/96 (96.875%) | [32/32](../results/weekend-20260919/maze-eval-Full-Blind-s0.json) | [29/32](../results/weekend-20260919/maze-eval-Full-Blind-s1.json) | [32/32](../results/weekend-20260919/maze-eval-Full-Blind-s2.json) |
| Lidar | 96/96 (100%) | [32/32](../results/weekend-20260919/maze-eval-Full-Lidar-s0.json) | [32/32](../results/weekend-20260919/maze-eval-Full-Lidar-s1.json) | [32/32](../results/weekend-20260919/maze-eval-Full-Lidar-s2.json) |
| Paired ray depth | 94/96 (97.917%) | [32/32](../results/weekend-20260919/maze-eval-Full-Stereo-s0.json) | [32/32](../results/weekend-20260919/maze-eval-Full-Stereo-s1.json) | [30/32](../results/weekend-20260919/maze-eval-Full-Stereo-s2.json) |
| Both | 96/96 (100%) | [32/32](../results/weekend-20260919/maze-eval-Full-Both-s0.json) | [32/32](../results/weekend-20260919/maze-eval-Full-Both-s1.json) | [32/32](../results/weekend-20260919/maze-eval-Full-Both-s2.json) |

[The generated summary](../results/weekend-20260919/SUMMARY.md) links every
earlier-stage JSON as well. Later training stages loaded the exact preceding
passed checkpoint; final checkpoint names are `model_499.pt`,
`model_1998.pt`, and `model_5997.pt`. All completed Full episodes, including
repeats after resets, yielded 1860/1875 successes; that is not the first-episode denominator
used above and must not replace it in comparisons.

### What these results establish

The **12-DoF biped** learned the repaired arrival/dwell objective in Isaac Lab
on a known approximately 3 m corridor route. The correction bundle addresses
reward timestep scaling, double-counted dwell, success penalties and braking;
this sweep does not isolate each change's causal contribution. Old zero-success
checkpoints are not relabelled as successful.

The planner supplies oracle waypoint commands. Every terrain tile has the
same geometry, and evaluation explicitly sets `enable_corruption=False`.
These are therefore **not** held-out-layout, sensor-only navigation, noise
robustness, real IMU deployment, or stereo/SSD recognition results. Camera
observations are independent raycast depth images. Blind also succeeds, so
the sweep alone does not establish that adding sensors improves navigation.

## Separate MuJoCo feasibility demonstrations

These use the **22-DoF humanoid**, with the already trained
`2026-08-18_20-57-50_arms-dr1.0-s0/exported/policy.onnx` gait. They are not
cross-simulator tests of the new 12-DoF recovery checkpoints. A supervisor
uses exact map/pose; optional braking consumes actual simulated lidar,
paired idealized depth and IMU measurements. The supervisor is not newly
trained MARL, SLAM, a visual-language policy or an object-carrying controller.

| Experiment | Nominal result | Matched negative controls | Evidence |
|---|---|---|---|
| Two-turn inspection, seeds 0–2 | 3/3; two ordered dwells and exit; 17.24–17.44 s; approximately 6.9 m; zero wall contacts | Wrong branch 0/3; complete exteroceptive outage 0/3, stopped translation and timed out | [Gate JSON](../results/weekend-20260919/inspection-maze-gate.json) |
| Two-person airlock, seeds 0–4 | 5/5 coordinated completions, zero robot/wall contacts | No-wait 0/5; withheld teammate 0/5 | [Pair JSON](../results/weekend-20260919/team2.json) |
| Three-person airlock, seeds 0–4 | 5/5 coordinated completions, zero robot/wall contacts | No-wait 0/5; withheld teammate 0/5 | [Trio JSON](../results/weekend-20260919/team3.json) |
| Sensor-reactive pair, seeds 0–2 | 3/3 coordinated completions, zero robot/wall contacts | No-wait 0/3; withheld teammate 0/3 | [Sensor JSON](../results/weekend-20260919/team2-sensors.json) |

Contact checks run at every physics substep. These small fixed-layout samples
demonstrate feasibility and meaningful task controls, not population-level
reliability. Inspection stations are proximity dwells, not recognized objects
or hand-operated switches. The robot keeps its eastward torso heading and
sidesteps during the northbound leg; changes in travel direction do not imply
a tested torso-heading turn skill.

[Inspection video](../results/weekend-20260919/inspection-maze.mp4) and
[three-person video](../results/weekend-20260919/team3-airlock.mp4) are simulator
rollouts. Their video-only JSONs mark `controls_complete=False`; this does not
mean their nominal mission failed. The multi-seed gate JSONs above, not a
selected video, supply the comparison evidence.

## Appropriate public claim

Implemented and evaluated a curriculum-based repair for humanoid goal-reaching,
with checkpoint-gated HPC training across four sensor configurations and three
seeds; separately demonstrated shared-world multi-humanoid synchronization and
sensor-reactive inspection using a frozen learned gait.

RGB stereo matching and SSD input adapters are separate implemented components
([stereo contract](STEREO_INPUT.md), [IMU contract](IMU_INPUT.md)). They have
not yet replaced privileged route/target information in these policies. That
integration and held-out-layout evaluation remain the next capability test.
