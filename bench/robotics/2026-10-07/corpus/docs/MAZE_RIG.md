# B5 — maze navigation with a lidar and a stereo pair

## The parts

| | modelled as | why |
|---|---|---|
| **RPLIDAR C1** | `LidarPatternCfg`, 500 rays, 0.72°, 12 m, 10 Hz | ray-cast, so it costs what the depth rung measured — 1.6% of throughput — and works on both stacks |
| **MMlove global-shutter stereo** | two `RayCasterCameraCfg`, 60 mm baseline, 64×64 → pooled 16×16 | the baseline is the part that makes stereo depth possible |
| **7″ screen** | **not modelled** | it is an output device. It does not enter the observation, and its absence is deliberate rather than forgotten |

A note on "global shutter": simulation has no rolling shutter, so it cannot show
the advantage. What it can show is the baseline geometry. The honest statement
is that a rolling-shutter part would be *worse on hardware* in a way this sim
cannot reproduce — this robot's own clips show the base oscillating several
centimetres a step, which is exactly the motion that skews a rolling frame.

## Built on the half that works

On `BipedBumpyEnvCfg`, the locomotion rung — not the v2 manipulation tasks.
Locomotion is where this project has real effects (push, terrain, arms). Every
manipulation task here has scored zero. A new sensor capability belongs on the
half that walks.

## Four arms, and why each hazard is chosen

| arm | obs width |
|---|---|
| blind | 45 |
| lidar | 81 (45 + 36 sector minima) |
| stereo | 557 (45 + 2 × 256) |
| both | 593 |

Each hazard is visible to exactly one sensor, which is what makes the
comparison mean anything:

* **floor obstacles at 0.10 m** — under the 0.34 m lidar plane, inside the
  cameras' down-pitched cone. Stereo's hazard.
* **corners and walls** — at lidar height, outside a 64-pixel forward cone
  until the turn is already made. Lidar's.
* **arrow plates** — geometric, not coloured, so they read as depth structure
  to stereo and as one more wall to lidar. Stereo's, and the reason the maze
  cannot be solved by wall-following alone.

Deliberately geometric rather than textured: 5.1's RTX renderer segfaults on
this cluster, and a marker only a colour camera could read would make the task
unrunnable there.

The blind arm is the control. Without it a lidar number is a fact about the
maze, not about lidar — the mistake G-B2 made when it measured its own
iteration budget and called it a terrain verdict.

## Button pressing: in, as a whole-body bump

The maze robot is the 12-DoF biped. There is no hand to press a plate, so
success is **base proximity** to the wall plate (0.40 m, held 8 steps) and is
labelled as a bump, not a dexterous press. `MazeWaypointCommand` heading-tracks
the tile-relative path (junction at \(1.5, 0\) then the plate at \(3.0, 0\))
at 0.40 m/s; progress rewards closing on the plate; a dead-end term ends an
episode that leaves the 0.90 m corridor or walks −x. Spawn is inside the
corridor, facing +x. Terrain-level promotion is off.

## Rubik's cube: out, and this is not a scheduling problem

This robot has **one revolute DoF per hand**, a single finger closing against
the palm. Solving a Rubik's cube requires holding the cube in one hand while
rotating a face with the other, independently actuated fingers, and force
control fine enough not to crush or drop it.

This project already established the weaker claim and it settles the stronger
one: *"A humanoid with no fingers cannot pick up a sock. Not with a better
policy, not with more iterations."* A cube is rigid, so it is easier to grip
than fabric — but face rotation needs in-hand manipulation, which is
categorically further from one DoF than picking is.

Adding it would mean either a different robot or a scripted animation labelled
as a result. Both are worse than saying so.

## Status

`Velocity-BHL-Maze-{Blind,Lidar,Stereo,Both}-v0` are registered. Geometry
smoke `21353199` **PASS** 4/4: `MazeWaypointCommand` on every arm, spawn
`|y|_max` 0.12–0.14 m inside the corridor, lidar wall 0.442 / 0.454 m.
Hazards are fused into each generated `/World/ground` tile. Old B5 PPO rows
are terrain-perception results and are not relabelled.

**The stereo pose was wrong, and a range check could not show it** (found
2026-09-13). The pair looked 20° *up*, upside down: B5 runs on Isaac Lab 3.0,
which reads a camera offset as `(x, y, z, w)`, and `STEREO_ROT` was written
`(w, x, y, z)`. A 0.61–6.00 m range and 100% finite values are what an
upward camera over bumpy ground returns too. `native_quat` fixes the pose, and
`scripts/bench/stereo_pitch_probe.py` checks the pose itself: pitch, up axis,
and the fraction of pixels that hit terrain (14.8% as trained, 77.5% fixed).

**Navigation training, 2026-09-17–18.** Run names `mazenav-*` (not `maze-*`).
Train smoke `21353395` (3 iterations, eplen 26.6–28.1) then PPO `21353396`–`398`
(6,000 iterations, four sensor arms, n=3). Terrain curriculum is off; the score
is button/dead-end, not `terrain_levels`.

**Result: they walk to timeout.** Last-50 from the event files
(`results/mazenav_last50.csv`):

| arm | eplen | track | time_out | button_reached |
|---|---:|---:|---:|---:|
| blind | 483.8 | 0.606 | 0.948 | **0.000** |
| lidar | 482.3 | 0.596 | 0.940 | **0.000** |
| stereo | 483.2 | 0.597 | 0.949 | **0.000** |
| both | 485.4 | 0.580 | 0.954 | **0.000** |

`button_reached` is 0.000 in every seed, including the max over training.
`progress_to_button` never exceeds 0.0005. Sensors do not separate.
`Metrics/success_rate` ~0.99 is surviving without falling, not a bump on the
plate. Seed-0 camera-sensor clips: `docs/gifs/isaac/mazenav_seed0.gif`
(`21355466`, 200 frames × 4, robot and walls in shot). Pooling arms wait on a
navigation score.

## Recovery campaign, 2026-09-19

The legacy IDs above and their uncommitted configuration fixes remain intact.
New IDs are `Velocity-BHL-MazeRecovery-{Approach,Corridor,Full}-{Blind,Lidar,Stereo,Both}-v0`.
At submission these were a repair experiment, not yet a successful navigation
result; the September 20 results below now establish the repaired corridor
baseline. Three concrete
MDP problems explain why more of the identical PPO sweep was a poor next test:

* `progress_to_button` returned a displacement per step, and Isaac's reward
  manager multiplied it by `step_dt` again. The recovery term returns m/s, so
  the integrated return measures distance made toward the target.
* The legacy button callback increments its hold counter when both termination
  and reward managers call it. Recovery caches one sample per physical step,
  requires eight samples, and clears state on individual environment resets.
  Dwell also requires gravity z below −0.90, planar speed at most 0.20 m/s,
  and roll/pitch angular speed at most 0.60 rad/s. A fall or fast crossing
  clears the dwell, including on its final sample.
* A success was also a termination: inherited weight −10 could cancel the
  +10 button reward. Recovery excludes success from the fall penalty and gives
  a one-off +12 return. The command slows down and stops inside the target.

These are verified logic defects; their individual contribution to zero success
has not yet been established by simulator ablations. In particular, the
in-corridor sign plates and 10 cm floor obstacles can still obstruct a gait.
The current route is a straight corridor with obstacles, not a branch-choice
maze. A perfect result here is a prerequisite, not the requested final mission.

| Stage | Spawn x relative to the terrain origin | Purpose |
|---|---|---|
| Approach | 2.35–2.45 m | learn final approach and eight-step dwell |
| Corridor | 1.65–1.75 m | cross/avoid the final 10 cm obstacle and stop |
| Full | −0.20–0.20 m | traverse the original 3 m route and its obstacles |

Every stage retains the same observations/actions within its sensor arm, so the
preceding checkpoint can initialize the next stage. Resume only within that
arm. Recovery stereo uses a 4×4 image per eye (32 features), and the combined
arm adds 36 lidar sectors. The old 512-feature stereo checkpoints are not
shape-compatible with the new stereo arms. Blind/lidar retain their shapes.

Proprioception already includes simulated angular velocity and gravity
direction, corresponding to gyro and attitude-derived IMU channels. Recovery
exposes Gaussian noise in the usual Hydra fields
`env.observations.policy.base_ang_vel.noise.std` (0.05 rad/s) and
`env.observations.policy.projected_gravity.noise.std` (0.02, unitless).
These are **uncalibrated stress-test assumptions**, not specifications for the
user's USB 10-axis AHRS. The exact module, rate, axis convention, covariance,
bias and timestamp behavior must come from that device's documentation/recording.
Magnetometer heading is not a reliable absolute indoor-position measurement.

`scripts/bench/maze_recovery_probe.py` runs a finite integration smoke or a
checkpoint rollout and writes the actual button success, timeouts, dead ends,
minimum goal distance, and displacement. Its promotion test uses the first
episode in every environment, avoiding inflation by repeatedly completing easy
episodes while hard environments are still running. It requires every first
episode to terminate and the configured success fraction to pass. An
integration smoke with zero actions is never reported as a policy evaluation.

Inside the existing v60 container/venv on a supported GPU:

```bash
"$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
  --task Velocity-BHL-MazeRecovery-Approach-Both-v0 --num-envs 16 --steps 600 \
  --output "$REPO/results/weekend/maze-smoke.json"

"$PY" "$REPO/scripts/train.py" --headless --enable_cameras \
  --task Velocity-BHL-MazeRecovery-Approach-Both-v0 \
  --num_envs 1024 --max_iterations 500 --seed 0 --run_name maze-recovery-approach-both-s0

"$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
  --task Velocity-BHL-MazeRecovery-Approach-Both-v0 --num-envs 32 --steps 600 \
  --checkpoint /absolute/path/to/model_499.pt --minimum-success 0.30 \
  --output "$REPO/results/weekend/maze-approach-eval.json"
```

Require both JSON `passed: true` and the `MAZE_PROBE_PASS` sentinel. Isaac's
shutdown has previously swallowed Python errors. A three-iteration train
smoke remains a separate requirement before the pilot; episode length alone
cannot certify a task. Advance only after the evaluation passes. The initial
30% gate detects nonzero learning; it is not a portfolio-quality final target.

The weekend driver runs this evaluation after every training stage and exits
nonzero if it fails. Approach pilots intentionally train from scratch for all
four arms. Later stages require the previous stage name as the final argument,
for example `weekend_maze.sh train Corridor Both 0 1024 1500 Approach`, then
`weekend_maze.sh train Full Both 0 1024 4000 Corridor`. A resume is allowed only
from the exact checkpoint named by that arm/seed's passed policy-evaluation
JSON. An integration smoke or a merely existing checkpoint cannot promote it.

## Recovery results verified — 2026-09-20

All 36 stage jobs completed with exit code `0:0`: Approach `21359486`,
Corridor `21359499`, and Full `21359510`, each with four arms and three seeds.
Every stage passed its checkpoint evaluation and emitted both required PASS
sentinels; all checkpoint files exist. No training or promotion gate failed.
The arrays have finished and are no longer in the live queue.

Full-stage **first-episode** successes are below. Each training seed is evaluated
in 32 environments, at evaluation seed `100 + training_seed`; all 32 first
episodes completed in every evaluation.

| Arm | Seed 0 | Seed 1 | Seed 2 | Total |
|---|---:|---:|---:|---:|
| Blind | 32/32 | 29/32 | 32/32 | 93/96 (96.875%) |
| Lidar | 32/32 | 32/32 | 32/32 | 96/96 (100%) |
| Paired ray depth (`Stereo`) | 32/32 | 32/32 | 30/32 | 94/96 (97.917%) |
| Both | 32/32 | 32/32 | 32/32 | 96/96 (100%) |

Combined Full success is **379/384 (98.70%)**. Earlier stage totals are
Approach 380/384 and Corridor 375/384. These are measured successes for the
new task definition, not a retroactive correction of the old zero-success
checkpoints. [All stage/seed counts and JSON evidence](WEEKEND_RESULTS_2026-09-20.md).

The known route and layout are unchanged across these evaluations, and the
probe explicitly disables observation corruption. This establishes the
**12-DoF biped's oracle-guided corridor arrival/dwell baseline**, not held-out
maze generalization, RGB stereo/SSD navigation, calibrated physical IMU input,
or robustness to observation noise. Blind's strong result also prevents a
sensor-benefit claim from this sweep alone.

The separate [MuJoCo inspection mission](INSPECTION_MAZE.md) completes a longer
two-turn route using the **22-DoF humanoid and frozen August 18 gait**. Its
videos and team-airlock results are not cross-simulator evaluations of the new
12-DoF recovery checkpoints. They demonstrate separately measured feasibility
with oracle map/pose supervision and, when enabled, simulated sensor braking.

## Research and the final mission specification

Primary sources checked 2026-09-19:

* [PASSAGE, submitted 2026-09-16](https://arxiv.org/abs/2609.18732) uses a
  perception-conditioned planner and whole-body tracker; its onboard system
  combines 3D lidar mapping, 6.25 Hz planning and 50 Hz tracking. Its 100 hours
  of human motion data and full-body embodiment are major differences from BHL.
* [TANGO, submitted 2026-09-08](https://arxiv.org/abs/2609.09158) learns
  language-conditioned whole-body traversal from synthesized feasible
  trajectories and RGB observations. Its G1 controller is not a BHL checkpoint.
* [HumanoidVLN, submitted 2026-08-13](https://arxiv.org/abs/2608.12860) evaluates
  navigation with an Isaac Sim hierarchy of navigation/path tracking and RL
  locomotion, across multiple humanoid embodiments. The paper says its code and
  benchmark will be released upon acceptance; do not assume they are available.
* [HEAD, CoRL 2025](https://arxiv.org/abs/2508.03068) separates a visual
  high-level planner from a whole-body controller that tracks hands and eyes.
* [Puppeteer, ICLR 2025](https://www.nicklashansen.com/rlpuppeteer/) studies
  hierarchical world models for visually controlled whole-body motion.

The practical inference for this robot is a slower navigation/task policy above
an independently tested walking controller, followed by a sensor-only student
and cross-simulator evaluation. This recovery campaign implements the initial
oracle-route baseline only. It does not reproduce those papers' models.

The intended moderate-difficulty mission is a 6–10 m route with two 90° turns,
one dead-end distractor, 0.9–1.2 m corridors, three low obstacles no higher than
the independently measured stepping limit, and two ordered inspection stations
before returning to a marked exit. At each station the robot must identify the
requested target from SSD boxes/classes/scores, approach using valid depth/range,
and dwell upright for 0.5 seconds. Later a reachable torso switch can use physical
contact sensing; proximity alone remains an arrival/inspection event. No hand
manipulation is assigned to the 12-DoF biped.

Mission completion must require both correctly ordered station events and the
exit, with no fall. Use held-out layouts and seeds; report task success, wrong
station visits, wall contacts, path efficiency and time. Increase difficulty
only while retaining measurable success, then test IMU bias, lidar dropout,
depth invalid pixels and delayed detections separately.

The current pair provides **two independent ray-cast depth images**. It does
not run stereo matching, and no SSD model currently supplies the maze policy.
The planned perception stage needs synchronized rendered left/right RGB,
calibrated intrinsics/extrinsics, stereo disparity/depth with validity, an
actual SSD checkpoint with declared classes, and timestamped detection inputs.
Ground-truth boxes may train or score a detector but must be labelled oracle
and excluded from the deployed policy. The waypoint command likewise exposes
the planned route; success under it cannot establish autonomous perception.
