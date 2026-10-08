# Full-humanoid inspection maze

This adds a physical maze beyond the existing straight-corridor recovery task:
two changes in travel direction, two ordered inspection stations, a wrong
branch ending in a dead end, and a final exit. It uses the existing 22-DoF
Isaac-trained BHL locomotion policy in MuJoCo, with actual simulated LiDAR,
IMU, and paired idealized ray depth feeding a reactive brake.

The first seed-0 physical smoke **passed in 17.36 s**, with two completed
inspections, five completed waypoints, no falls, and zero wall-contact
intervals. Its complete sensor-outage control failed to reach any waypoint
within 60 s and stayed upright without contact. Evidence is in
[`results/inspection_maze_probe.json`](../results/inspection_maze_probe.json).
This is feasibility evidence on one seed, not a measured generalization rate.

The subsequent CPU Slurm evaluation **21359677 passed all three nominal
seeds** (17.24, 17.44, 17.28 s), with no contacts or falls. Complete-outage and
wrong-branch controls each completed 0/3 missions; the latter entered the
dead end after station A. Evidence:
[`inspection-maze-gate.json`](../results/weekend-20260919/inspection-maze-gate.json).
These are small fixed-layout tests, not broad maze generalization. The separate
[video](../results/weekend-20260919/inspection-maze.mp4) shows the original
17.36 s rollout; small cross-machine numerical differences are expected.

## Layout and capability boundary

Each open cell is 1.6 m square; 8 cm thick walls leave approximately 1.52 m of
clear corridor width. The boundary walls surround the union of six open cells.
The route is 6.4 m along its planned centerline. There are no tall steps or
unsupported grasping requirements.

```text
                    turn east ─── inspection B ─── EXIT
                        │
                        │
START ─ inspection A ─ junction
                        │
                        │
                     DEAD END
```

Inspection A is at `(1.0, 0.0)`. The two turn waypoints are `(1.6, 0.0)` and
`(1.6, 1.6)`. Inspection B is at `(3.2, 1.6)`, followed by the exit at
`(4.8, 1.6)`. The dead-end branch extends south to `(1.6, -1.6)`.

Inspection means standing within 0.23 m of an assigned point for 0.6 s while
upright. It is not object classification or a button press. Only the next
ordered waypoint can advance the mission; reaching the exit directly cannot
succeed. Entering the dead-end branch, leaving the maze, falling, or wall
contact fails the episode. Contact is checked at every 0.5 ms physics step;
the logged contact count measures 40 ms control intervals containing contact.

The supervisor knows the map and exact robot position. The learned gait takes
body-frame velocities; the torso holds its initial east-facing heading, using
sidestepping for the northbound leg. Thus the path makes two right-angle turns
without claiming an untested heading-turning skill. The LiDAR is 360 degrees
and can protect lateral travel; paired forward depth is used for forward
travel. Sensors can slow or stop commanded motion, but they do not discover
the correct branch or localize the robot. Those are separate future policy
requirements. This is a useful motor/perception feasibility reference, not a
new learned maze policy, SLAM system, or reproduction of a research paper.

Sensor definitions and their tests are shared with
[the cooperative airlock benchmark](TEAM_AIRLOCK.md#optional-actual-sensor-inputs-and-a-reactive-brake).
LiDAR and paired depth are body-attached MuJoCo rays; gyro, acceleration, and
attitude come from actual MJCF sensors and use the `sensor_io` adapter. Paired
ray depth is idealized geometry, not stereo correspondence. SSD is explicitly
unconfigured because RGB imagery and detector weights are not connected here.
The USB AHRS is not assumed calibrated by this simulation.

## Evaluation and HPC invocation

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python scripts/bench/inspection_maze.py \
  --deploy external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml \
  --upstream external/Berkeley-Humanoid-Lite \
  --cache-dir /tmp/bhl-inspection-maze-eval \
  --seeds 3 --seconds 60 \
  --sensor-modes reactive reactive_dropout \
  --output results/inspection_maze_eval.json --gate
```

The default dropout probability is **1.0**, giving a complete-outage negative
control with a 150 ms freshness limit. Set `--dropout-probability 0.35` in a
separate output to test intermittent loss; this is an uncalibrated stress-test
parameter, not a device specification. `--sensor-modes record reactive` gives
a comparison between recording and actually consuming the same sensor types.

`--routes wrong_branch --sensor-modes reactive` deliberately chooses the
southbound branch after inspection A and checks the dead-end failure rule.
The seed-0 physical control completed A and then failed with
`dead_end_entered` at 7.00 s, without contact; see
[`results/inspection_maze_deadend_probe.json`](../results/inspection_maze_deadend_probe.json).
`--routes ordered wrong_branch` evaluates both routes. The gate requires at
least 80% successful ordered reactive episodes and a lower success rate in
the dropout control; missing control arms are labelled incomplete. Three
seeds require all three nominal episodes to pass this threshold and still do
not establish population reliability.

The evaluator uses CPU physics and one-thread ONNX inference. A 2–4 core,
8 GB CPU allocation with a 1 h limit is ample for the three-seed nominal/outage
comparison. The repository's generic weekend CPU wrapper can pass the command
arguments unchanged. Use unique cache/output paths for concurrent jobs.
Optional `--video results/clips/inspection_maze.mp4` records the first ordered
reactive seed and requires EGL or OSMesa; graphics need not occupy a DGX GPU.

Every completed episode is flushed to JSON with station times, completed
waypoints, failure reason, path length, contact intervals, sensor packet and
braking statistics, and per-second pose/command/sensor traces. The initial
nominal smoke travelled 6.93 m, received 174 range packets, and had 27 control
steps whose translation was reduced by observed ranges. The outage control
received zero of 600 attempted packets, visited zero stations, and stopped
translation in all 1,500 control steps while retaining normal low-level IMU
balance feedback.

The five mission tests cover station ordering, continuous/idempotent dwell,
completion, dead-end/fall failures, and at least 0.70 m centerline-to-wall
clearance. Physical gait feasibility is established by simulator runs, not
those logic tests.
