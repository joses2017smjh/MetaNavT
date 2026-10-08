# Randomized mazes, mapped by the robot's own lidar

**Ask (2026-09-24):** mazes that change every run so a clip cannot be a
memorised route, the sensor panels of the three-panel clips, a view of how the
robot mapped the maze along its trajectory, and a robot that turns and then
walks forward instead of sliding sideways.

## What runs

| Piece | What it is | Learned / scripted / oracle |
|---|---|---|
| Maze | recursive-backtracker maze on an n×m grid of 1.4 m cells (1.32 m clear corridors, 1.1 m walls) plus 1–2 random extra openings so routes have loops; **a new maze per seed** (`bhl_robust/eval/random_maze.py`) | generated; unknown to the planner |
| Gait | the biped `dr-default-s0` PPO checkpoint, frozen, driven by body-frame (vx, 0, wz) commands | **learned** |
| Sensing | the `team_sensors` rig on the biped's mounts (lidar +0.34 m, stereo pair +0.30 m, 20° down): 108-ray 12 m lidar and an 8×8 paired ray depth at 10 Hz, plus the existing speed brake | actual simulated sensors |
| Map | log-odds occupancy grid, 0.10 m cells, integrated from the raw lidar returns | built online |
| Localization | the simulator's true pose | **oracle** (says so on every frame) |
| Planner | A* on the inflated map (0.30 m), **unknown = free**, replanned every 0.4 s; greedy line-of-sight simplification to corner waypoints; the last waypoint is the exact goal coordinate | scripted; goal coordinate is oracle |
| Motion | turn in place toward the next waypoint (|heading error| > 0.40 rad, 0.6 rad/s), walk forward once within 0.15 rad (0.30 m/s, yaw correction ≤ 0.4 rad/s); **vy is always 0** | scripted |

Success: goal disc reached within the time limit with no fall. *Clean*: also
no wall contact at any physics substep. The table over seeds is the evidence;
the clip is one seed at the median clean completion time, never the fastest.

## Why the biped, and why earlier clips walk sideways

Measured on the login node (flat world, 6 s of a constant command after 1 s of
standing; yaw change and displacement of the base):

| Gait | (0, 0, 0.6 rad/s) | (0, 0, 1.0) | (0.2, 0, 0.8) | (0.35, 0, 0) |
|---|---|---|---|---|
| humanoid `arms-dr1.0-s0` (22 DoF, the Mission 7 / inspection gait) | **0.2°**, 0.06 m | 4.2° | 1.1° | −18° drift over 2.0 m |
| biped `dr-default-s0` (12 DoF) | **239°**, 0.07 m | 393° | 318°, 0.14 m | 7° over 2.1 m |
| biped `push-adaptive-s0` | 6° | 7° | 199°, 0.57 m | −54° over 1.9 m |
| biped `terrain-bumpy-s0` | 7° | 9° | 311°, 0.19 m | 13° over 2.1 m |

The humanoid checkpoint ignores the yaw-rate command. That is why every
Mission 7 and inspection clip translates holonomically with a fixed facing:
the route controllers never had a turn available to them. `dr-default-s0`
tracks a yaw-rate command at about 1.2× the commanded rate and walks
straight, so this mission uses it. The biped has no arms and no vision arms
trained on it.

## Results

Job `21412035` (CPU MuJoCo, 9:52 for all 24 episodes), one maze per seed, seeds 0–11, initial heading +x:

| configuration | seeds | reached goal | clean (no wall contact) | falls | completion, median (range) | route cells, median (range) | path walked, median | turns, median (range) | median-time seed |
|---|---|---|---|---|---|---|---|---|---|
| base-5x5 | 12 | **12/12** | **12/12** | 0 | 37.2 s (28.9–70.2) | 9 (9–19) | 12.4 m | 5 (1–22) | seed 5 |
| hard-6x6 | 12 | **12/12** | **12/12** | 0 | 73.1 s (56.8–116.4) | 17 (11–31) | 22.6 m | 15 (9–38) | seed 1 |

Every episode reached its goal with no fall and no wall contact at any physics substep. The mapped fraction of the
grid at the goal was 0.43–0.78: the planner never needed the whole maze, only what the 12 m lidar had returned along
the way. "Turns" counts walk→turn transitions, i.e. every time the next waypoint needed a turn in place.

Predeclared rule applied: the harder configuration scored ≥ 10/12 clean, so the clip is its median-time seed
(seed 1: 73.1 s, 24.8 m walked for a 21-cell route, 23 turns, 172 replans), rendered by job `21412068` →
`docs/gifs/random-maze-explore.gif`. Evidence: `results/maze-explore-20260924/{base-5x5,hard-6x6}/seed*.json`
(per-step traces) and `summary.json`.

## Reproduce

```bash
# multi-seed table, CPU MuJoCo, no render
python scripts/bench/maze_explore.py --upstream external/Berkeley-Humanoid-Lite --cache-dir /tmp/mz \
    --seeds 12 --n 6 --m 6 --extra-openings 1 --time-limit 180 --out-dir results/maze-explore-20260924/hard-6x6
# three-panel clip of one seed (needs EGL: MUJOCO_GL=egl on a GPU node)
python scripts/bench/maze_explore.py ... --seeds 1 --seed-start K --render --gif docs/gifs/random-maze-explore.gif
```

## Robustness (job `21434853`, same 12 maze seeds per configuration)

| Perturbation | 5×5 reached / clean | 6×6 reached / clean | Falls | Verdict (predeclared) |
|---|---|---|---|---|
| none (published table) | 12 / 12 | 12 / 12 | 0 | reference |
| 35 % of lidar/depth packets dropped | 12 / 12 | 12 / 11 | 0 | **PASS** (≥ 10/12 reached, 0 falls) |
| 70 % dropped | 12 / 12 | 9 / 7, 3 time-outs | 0 | reported, not gated |
| random initial heading | 12 / 12 | 12 / 12 | 0 | **PASS** |
| speed brake off | 12 / 12 | 12 / **8** | 0 | **brake is load-bearing** on 6×6 (clean ≤ 9/12) |

Dropout slows the robot rather than stopping it (median 47 s vs 37 s on 5×5, 85 s vs 73 s on 6×6 at 35 %): the brake stops translation while packets are stale, and the map keeps what earlier scans saw. With the brake off the planner still reaches every goal, but four of twelve 6×6 runs brush a wall, so the "clean" part of 24/24 belongs to the reactive layer, not to the map.

## Sensor-fusion stress (job `21435083`, hard 6×6, seeds 0–11)

Errors feed the map, the planner and the controller; the judge always uses the true pose.

| Stress | Levels (reached with no fall / 12) | Predeclared verdict |
|---|---|---|
| constant heading error | 0°: 12, 1°: 11, 3°: 10, 10°: 9 | **tolerance 3°** |
| IMU delivery delay (Mahony 200 Hz) | 0 ms: 11, 20: 11, 30: 11, 40: 1 (11 falls), 60: 0 (12 falls) | **budget 30 ms** |
| constant position bias | 0.05 m: 3, 0.15 m: 2, 0.30 m: 3 | FAIL — mostly *arrived, not judged* (9, 7, 8) |
| position noise 0.10 m | 8 | FAIL |

On a map the robot builds itself, a heading error is much more dangerous than on a known map (3° here against 20° in SF-02) because every scan is painted rotated. A constant position bias barely hurts exploration: the map and the robot stay consistent, so the robot finds the goal on its own map and stops exactly where its biased estimate says the goal is. The failure is goal localisation (the goal has to live in the map frame, or the bias has to be estimated), not navigation. The IMU budget of the biped gait, 30 ms, matches the 22-DoF gait's.

## A learned policy on the same interface (NavGym)

`bhl_robust/navgym/env.py` is a Gymnasium environment whose action is exactly
the gait's command (vx, wz; no vy) and whose observation is what the physics
runner can reproduce: the 36 lidar sectors, a 3×24×24 egocentric crop of a
0.2 m log-odds map built from the same 108 rays, and goal distance/bearing.
The proxy dynamics are a unicycle with the gait's measured yaw-rate gain, a
first-order lag, walking drift and command latency, randomized per episode.
`scripts/bench/navgym_train.py` trains it with Stable-Baselines3 PPO on CPU
under a maze-size curriculum, evaluates on mazes with seeds ≥ 10 000 (never
trained on) and exports the deterministic actor to ONNX;
`scripts/bench/maze_explore.py --policy actor.onnx` then drives the physics
robot with it. Jobs `21412154` / `21412155`; the predeclared transfer test is in
`SLURM_JOBS.md`.

Results so far (all in `SLURM_JOBS.md`): v1 NEGATIVE; v2 missed its held-out bar
by one clause; v3 (action-std cap) NEGATIVE 0/3 on fresh mazes. **Exploratory
transfer** (`21442350`): the best v2 actor drove the physics biped to **10/12**
never-seen 6×6 goals with 0 falls, against A*'s 12/12
(`docs/gifs/navgym-learned-maze.gif`). A matched-seed v2 control and a v4 that
prices stalling are queued.

## Sensor panels (2026-09-29)

`maze_explore.py --imu-panel --rig-panels` adds display-only panels to a clip: a
simulated IM10A-like IMU (Hiwonder datasheet noise, 100 Hz; gyro and accelerometer
traces, 6-axis attitude filter against truth, magnetometer and barometer shown but
not used), the stereo rig's RGB and 160×120 rendered depth, and left-eye optical
flow. They read the simulator only; the hero episode's trace is identical with and
without them. The README hero, `docs/gifs/random-maze-explore-sensors.gif`, is that
episode; the noise model switches to measured values once the real IM10A is recorded (SF-05).

## The 22-DoF humanoid on the same mission (2026-09-28)

`maze_explore.py --variant humanoid` swaps in the full-body humanoid with the
only qualified turning gait, `arms-turn-turnboth-s0` (one checkpoint: turn
10/10, walk 3/3, push 7/60; it marches in place at zero command and turns
while stepping). Same lidar map, planner, turn-then-walk controller and judge;
settings frozen on pilot mazes ≥ 100 (cruise 0.30 m/s, turn 0.6 rad/s,
inflation 0.50 m for its 0.31 m footprint). Scored once on hard 6×6 maze
seeds 12–23, never run before (job `21463686`, predeclared PASS ≥ 10/12 with
0 falls):

| Robot (learned gait) | Reached | Falls | Clean | Median time |
|---|---|---|---|---|
| 22-DoF humanoid, TurnBoth-s0 | **12 / 12** | 0 | 12 / 12 | 70.5 s |
| 12-DoF biped, dr-default-s0 (reference) | 12 / 12 | 0 | 12 / 12 | 58.2 s |

**PASS.** Labels: LEARNED gait; SCRIPTED planner and controller; ORACLE pose
and goal. The humanoid's lidar sits at 0.66 m, and when the body pitches a few
rays pass over the 1.10 m walls (0.67 % on a pilot seed); not scored.
`results/maze-humanoid-20260928/verdict.json`.



## NavGym v5 and fresh confirmation (2026-10-06)

The final NavGym v5 actors trained with visitation memory, a wider coarse map, a yaw-change cost and a deployment range brake passed their predeclared gym gate on all three training seeds. In the subsequent MuJoCo transfer, the final actors reached **11/12, 12/12 and 12/12** goals with **zero falls**; clean (no physics-step wall contact) counts were **11/12 for each actor**. This is **35/36 actor-layout rollouts on 12 shared held-out layouts**, not 36 independent layouts. The matched A* reference reached 12/12. The gait and navigator are learned; pose and goal coordinate are oracle inputs; the reactive speed brake is scripted. The older published NavGym GIF remains an exploratory v2 example. Evidence: [`verdict.json`](../results/navgym-v5-transfer-20261002/verdict.json) and [final gym gate](../results/navgym-v5-20261002/verdict_v5.json).

The **2026-10-06 confirmatory campaign** freezes all three final actors and the same gait for 48 new 6×6 layouts (72000–72047), each under nominal sensing and 35% lidar/depth packet dropout. A* runs the identical layouts and sensing conditions. The project targets, declared before scored execution, require each actor to reach at least 40/48 nominal goals and 36/48 dropout goals with zero falls; wall contact and completion times are reported separately. Failures and missing files remain in the denominator. Array **21601709**, after locomotion array **21601701**; final evidence verdict **21601710**. [Frozen artifacts](../campaigns/20261006-confirmatory/); live outputs remain under `Computer_Vision/project_results_upgrade/humanoid_confirmatory`. These jobs are **pending evidence**, not a measured improvement.


<!-- bhl-confirmation-results-2026-10-06 source_sha256=d74574dbc07e93b48655168268d22b84feb481122b232bf220c31b7e227b7b5e -->
### Recorded confirmation outcomes — campaign 2026-10-06
Append-only evidence update at `2026-10-06T12:56:38.990567+00:00`. This dated record supersedes the earlier queued-status paragraph for this campaign; historical experiment results above remain their original records.
**Locomotion: PASS. Navigation: NEGATIVE.** Recorded episode files: locomotion **3600/3,600 planned**; navigation **384/384 planned**. File counts alone do not certify valid episodes.
| Randomization rung (3 training seeds) | Flat falls / reported episodes | Push04 falls / reported episodes | Flat mean displacement |
|---|---|---|---|
| dr-off | 77/360 | 341/360 | 1.408 m |
| dr-s0.5 | 4/360 | 268/360 | 1.849 m |
| dr-default | 0/360 | 13/360 | 1.984 m |
| dr-s1.5 | 0/360 | 0/360 | 1.767 m |
| dr-aggressive | 60/360 | 60/360 | 0.129 m |

Flat denominators are 360 per rung when complete (3 policies × 6 commands × 20 shared reset seeds), with another 360 disturbed episodes per rung. The flat replication rule requires zero falls for each default policy and fewer falls than its matched unrandomized policy. Push04 is descriptive. Full per-training-seed/command tracking, displacement and paired outcomes remain in `dr_verdict.json`; surviving-step errors alone cannot rank failed policies. When the verdict is INCOMPLETE, reported partial counts do not establish a validated complete experiment.

| Navigation actor | Condition | Goals / reported episodes | Clean | Falls | Complete evidence |
|---|---|---|---|---|---|
| armV5-s8 | nominal | 44/48 | 44/48 | 1/48 | True |
| armV5-s8 | drop35 | 43/48 | 43/48 | 0/48 | True |
| armV5-s9 | nominal | 46/48 | 44/48 | 1/48 | True |
| armV5-s9 | drop35 | 44/48 | 44/48 | 2/48 | True |
| armV5-s10 | nominal | 44/48 | 44/48 | 2/48 | True |
| armV5-s10 | drop35 | 46/48 | 45/48 | 0/48 | True |
| astar | nominal | 44/48 | 39/48 | 0/48 | True |
| astar | drop35 | 45/48 | 42/48 | 0/48 | True |

Each complete navigation cell contains 48 shared held-out 6×6 layouts. Each final learned actor must meet both ≥40/48 nominal and ≥36/48 drop35 goals with zero falls; all actors and the matched A* control are reported. Clean means no wall contact at any physics step. An incomplete cell cannot establish validated success, regardless of its reported partial goal count.

Scope: simulation only. Isaac-trained frozen 12-DoF PPO gait; navigation uses a learned PPO actor with oracle pose/goal and a scripted reactive speed brake; A* is the reference. Shared layouts, commands and reset seeds are clustered observations, not independent trained policies. Neither PASS nor a zero observed fall count certifies hardware deployment.

Immutable JSON sources: `/nfs/hpc/share/sanchej7/Computer_Vision/project_results_upgrade/humanoid_confirmatory/dr_verdict.json` and `/nfs/hpc/share/sanchej7/Computer_Vision/project_results_upgrade/humanoid_confirmatory/verdict.json`. Locomotion JSON SHA256: `7bd2629ae71b8bddd61e1b4c9d88c85d1abd5116f145d00f3783755bd82fcc04`; navigation JSON SHA256: `7b9a1901f1febd48e6ff4849ecbd26a2ca6c15f36326da5be378d2020b9b8393`. Combined source SHA256: `d74574dbc07e93b48655168268d22b84feb481122b232bf220c31b7e227b7b5e`. Executable protocol and publisher are archived in `campaigns/20261006-confirmatory`.
