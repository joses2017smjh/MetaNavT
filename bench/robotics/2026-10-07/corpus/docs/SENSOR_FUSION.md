# Sensor fusion for the maze and the other sensor-dependent tasks

LiDAR + stereo camera + ROS 2 ten-axis IMU, for the Berkeley Humanoid Lite.
Written 2026-09-23 as a **methodology and plan**. Nothing in this document is a
hardware result: every sensor in this repository is still simulated, no state
estimator runs anywhere, and every policy consumes raw channels. Section 1 is
what exists (measured from the code), Sections 2–5 are what to build and in
which order, Section 6 is the budgeted experiment plan with predeclared gates,
and Section 7 is the literature the recommendations rest on.

Companion contracts: [physical IMU input](IMU_INPUT.md),
[calibrated stereo input](STEREO_INPUT.md), [maze rig](MAZE_RIG.md),
[inspection maze](INSPECTION_MAZE.md), [team airlock sensors](TEAM_AIRLOCK.md).

## 1. What exists today

| Channel | Simulated source (Isaac training) | Simulated source (MuJoCo evaluation) | How the policy sees it | Hardware counterpart | Fusion today |
|---|---|---|---|---|---|
| LiDAR | `RayCasterCfg`, one plane at **0.34 m** on the 12-DoF base, 0.72° (500 rays), 12 m, `ray_alignment="base"`, 10 Hz nominal (`sensors_rig.py`) | 108 body-attached rays at 0.72 m on the 22-DoF torso, 12 m, 10 Hz (`team_sensors.py`) | **36 sector minima** scaled by range, Gaussian noise std 0.0025 | RPLIDAR C1 (single plane, named in `maze_env_cfg.py`); no driver, no recording | none: one observation term |
| Stereo | two `RayCasterCameraCfg` eyes, **60 mm** baseline, 0.30 m height, **20° down**, 64×64 `distance_to_image_plane`, 6 m, update period 0 | paired idealized ray depth, 8×8, 60 mm baseline, 20° down, 6 m | pooled to 16×16 (depth rung) or **4×4 per eye = 32 features** (recovery arms), noise std 0.0033 | none identified; `scripts/perception/stereo_depth.py` runs OpenCV SGBM on calibrated rectified pairs (`bhl-rectified-stereo-v1`) | none; **ray depth is not stereo matching** |
| IMU | `base_ang_vel` (noise 0.05 rad/s) and `projected_gravity` (noise 0.02) from simulator ground truth | MJCF site quaternion, gyro and accelerometer through `sensor_io.imu_features` → `[gravity(3), gyro(3), specific force(3), valid]` | inside the 75-obs proprioception; **zero latency, no bias, no filter in the loop** | probable **Hiwonder IM10A** (gyro, accel, mag, baro; factory 10 Hz, ≤200 Hz on three outputs; ENU attitude). The upstream low-level driver instead expects a **BNO085** (w,x,y,z quaternion, gyro, gravity). Two different parts; neither recorded | none; mag and baro unconsumed |
| Localization | simulator pose + known map: the maze route teacher (`RecoveryWaypointCommand`) is an **oracle** | oracle pose and map (`localization: oracle_simulator_pose_and_map`) | as a body-frame velocity command | nothing | none |
| Timing | Isaac: every sensor at every policy step | `SensorTiming(max_age_s=0.15, stereo_skew_s=0.035)`, exteroception every 0.1 s, command-level dropout modes | — | unmeasured | freshness check only |

Two facts shape everything below:

1. **The policies have no memory.** Every maze arm is a feedforward MLP on the
   current observation; `history_length` is 0 for the deployed gait. The only
   recurrent precedent in the repo is the terrain rung's `scan-student` (an LSTM
   student distilled from a privileged height-scan teacher, REPORT §4: level
   1.570 against the blind 1.443/1.446/1.417 and the teacher 1.413/1.288). That
   precedent is the pattern to reuse for fusion.
2. **High-dimensional exteroception already failed once.** At pool 4 the stereo
   term was 512 of 557 observations and dragged `both` below `lidar`
   (`maze_env_cfg.py`, pooling sweep). Fusion here is as much about *what not to
   feed the policy* as about estimation.

## 2. What fusion must deliver, per task

| Task | Needs | Which sensor carries it | Gap today |
|---|---|---|---|
| Maze recovery / inspection route | attitude (roll, pitch) for the gait | IMU gyro + accel → projected gravity | policy uses ground truth; filter never in the loop |
| | heading (yaw) for the waypoint command | gyro integration (drifts), LiDAR scan matching against the walls (does not); magnetometer only if the indoor field is clean | oracle |
| | 2-D position in the maze frame for the route teacher | 2-D LiDAR localization against the known map with IMU + leg odometry as motion model | oracle |
| | walls and junctions | LiDAR plane at 0.34 m | fine |
| | 0.10 m floor obstacles | stereo cone (below the LiDAR plane) | fine in sim; unmeasured for real stereo |
| | outage handling | freshness of every channel | command brake exists (`reactive_dropout`) |
| Team airlock | the above + teammate range/bearing | LiDAR (legs at plane height) + stereo | oracle teammate state |
| Terrain / depth locomotion | attitude + forward ground geometry | IMU + stereo depth (replaces ray depth) | ray depth, zero latency |
| Ice | attitude + friction cue | IMU; slip from leg kinematics vs IMU velocity | not exposed to ice in training yet (open probe) |
| Coop lift, cube-to-shelf | object state | privileged in sim | not a sensor-fusion problem yet |

## 3. Recommended architecture

Layered, so each layer can be validated alone before it feeds the next. The
first three layers are mature engineering; the last two are where the research
is.

**Layer 0 — time base and calibration.** One clock domain (ROS 2 time). The
USB IMU cannot be hardware-triggered, so stamp at driver receive and *measure*
the offset and jitter against the LiDAR and camera stamps (`ros2 topic hz`,
`ros2 topic delay`, and a mechanical event such as a tap seen by all three).
Camera–IMU extrinsics and time offset with Kalibr; LiDAR–IMU extrinsics with a
LiDAR–IMU calibration tool; stereo intrinsics/rectification per the existing
contract. Extend `SensorTiming` with a per-channel measured latency so the
evaluation harness can replay the measured skews, not a guess of 35 ms.

**Layer 1 — attitude (≥100 Hz).** A Mahony or Madgwick complementary filter on
gyro + accelerometer, publishing orientation and bias-corrected angular
velocity. Magnetometer *off* by default indoors; enable only after the field
magnitude and inclination pass a stationarity check in the maze, otherwise yaw
gets pulled by steel. Barometer ignored: its resolution is coarser than the
robot is tall. This layer's outputs are exactly the sim's `projected_gravity`
and `base_ang_vel`, so the locomotion policy needs no retraining if the filter
error is inside what the policy tolerates. SF-03 measured that tolerance: the
frozen gait walks on an estimated attitude with gravity RMSE up to ≈0.05 but
**falls once the IMU sample it acts on is more than about 30 ms old** (SF-03b: 3/3 up
to 30 ms at a 200 Hz filter, 1/3 at 40 ms, 0/3 at 60 ms). The requirement is
therefore on latency, not accuracy: the attitude must reach the policy within
≈30 ms end-to-end, sensor to observation. The 12-DoF maze policies, trained with a
zero-latency IMU, roughly halve their success with one 20 ms step of IMU delay
(pooled 0.969 → 0.477) and fall to 0.023 with two. Fine-tuning the same policy
for 2,000 iterations with a randomized 0–1 step delay restores 32/32 at one step
and 30/32 at two (SF-04 delay-only arm): the latency requirement can be bought
back in training, cheaply, rather than only met in hardware. The IM10A's factory 10 Hz output (100 ms period) cannot
meet that; it must be configured to ≥ 100 Hz over a fast enough serial link,
or the gait retrained with delay randomization (Isaac Lab `DelayBuffer`).
Filter choice (Mahony vs Madgwick, gain 0.3 vs 1.0) made no difference to
either outcome, consistent with Caruso et al. 2021: tuning and timing dominate
filter choice.

**Layer 2 — base state for legged odometry.** A contact-aided kinematic–inertial
filter (invariant EKF form) fusing IMU with leg kinematics and foot contacts to
give base velocity and a drift-limited position. The deployed gait does not
observe base linear velocity, so this layer is not required for balance; it is
required as the *motion model* for Layer 3 and for slip detection on ice.

**Layer 3 — 2-D pose in the maze frame.** Scan-matching localization of the
single-plane LiDAR against the known maze map (Nav2 AMCL or a scan-to-map ICP),
with Layer 2 odometry as the prediction step. This replaces the oracle pose that
the route teacher consumes. Expect centimetre-level position and 1–3° yaw in a
walled maze; SF-02 measures how much error the route teacher tolerates before
that is a requirement rather than a hope. A full 3-D LiDAR-inertial SLAM stack
is not needed for a known 1.6 m-cell maze; it becomes relevant for unknown
spaces and 3-D LiDAR hardware.

**Layer 4 — hazard layer.** Keep LiDAR and stereo as *separate* channels for the
policy — sector minima from LiDAR, a small pooled depth footprint from stereo —
rather than merging them into one costmap the policy has to decode. The failure
modes differ (LiDAR misses low obstacles, stereo fails on texture-less walls and
at corners), and the pooling sweep showed that dimensionality, not information,
was what hurt. Merge them only at the *command* layer (brake/slow), where the
harness already does it.

**Layer 5 — policy-side fusion.** Replace the feedforward `both` arm with a
recurrent belief encoder trained with **modality dropout** and **measured
latency**, distilled from a teacher that sees privileged pose and map (the
`scan-student` pattern, one level up). The student then sees only what Layers
1–4 can deliver on hardware. Training-time randomization must cover the
measured IMU bias/noise, LiDAR range noise and dropout, stereo invalid-pixel
patterns, and per-channel delays; otherwise the sim result is about the
simulator's clean sensors.

## 4. Simulation-side changes so the sim matches the hardware

_Status 2026-09-23 (evening): items 1–2 are implemented for the maze in the
`BothRobust` arm (`src/bhl_robust/tasks/maze_robust.py`: per-episode LiDAR or
stereo dropout p = 0.2, gyro/gravity bias std 0.02, IMU delay 0–1 policy step,
clean asymmetric critic), item 3 is implemented as evaluation-time error
injection in both harnesses (`inspection_maze.py --pose-*`,
`maze_recovery_probe.py --settings`), item 4 is not started._

1. **Per-channel delay and rate** for Isaac observation terms (a ring buffer per
   term: LiDAR at 10 Hz with 1–2 policy steps of delay, stereo at its measured
   frame rate and latency, IMU at policy rate). The maze arms currently get
   every ray at zero latency.
2. **IMU model**: gyro bias random walk, accelerometer noise and a Layer 1 filter
   *in the loop*, so `projected_gravity` comes from the estimated attitude, not
   the simulator quaternion.
3. **Localization error injection** for the route teacher: pose bias + noise +
   yaw drift with a configurable model, so "how good must Layer 3 be" is a
   measured number.
4. **A rendered-stereo validation slice**: render RGB stereo in Isaac, run the
   existing SGBM script, and compare its depth with the ray-cast depth the
   policy was trained on (invalid fraction, error vs range). Not for training;
   for knowing what the policy will actually receive.

## 5. What to adopt first

0. **Measured 2026-09-24:** the cheapest policy-side remedies all work on the
   maze — a 2,000-iteration fine-tune with a randomized 0–1 step IMU delay restores
   the memoryless policy to 32/32 and 30/32 under one and two steps of delay, and
   distilling it into a recurrent student on degraded observations keeps ≥ 27/32
   under every tested outage and delay; and a 2,000-iteration MLP fine-tune with
   all three degradations randomized per episode scores 29–32/32 on every row.
   Randomize latency, bias and outages when fine-tuning the working policy; do
   not train a recurrent policy from scratch. (Fixed known route; see the SF-04 caveat.)

1. ~~Layer 1 attitude filter in the MuJoCo loop (SF-03)~~ **done**: the gait
   tolerates the estimate; it does not tolerate ≥ 40 ms of IMU latency. The
   first hardware task is therefore an IMU path with < 40 ms end-to-end delay
   (IM10A at ≥ 100 Hz, or a BNO085-class part with on-board fusion as the
   upstream low-level driver already expects), measured with a tap test.
2. IMU noise/latency sensitivity on the trained maze checkpoints (SF-01):
   evaluation only, sets the hardware requirement.
3. Localization-error injection (SF-02): sets the Layer 3 requirement before any
   localization stack is chosen.
4. Hardware acceptance recordings (SF-05, no HPC): identify the IMU actually
   mounted, record stationary/rotation captures, Allan variance, stereo
   calibration, LiDAR–IMU extrinsics.
5. Only then the one justified training run: recurrent `both` with modality
   dropout and measured delays (SF-04).

## 6. Experiment plan (predeclared)

| ID | Question | Method | Compute | Gate / predeclared reading |
|---|---|---|---|---|
| SF-01 | How much IMU noise and delay do the trained Full-stage maze checkpoints tolerate? | **DONE** (clean rerun `21404944`, fixed probe; `results/repo-gpu-20260923/maze-sf01/summary.json`) | 1.2 GPU-h | Pooled over four Full s0 checkpoints: baseline 0.969; gyro 0.10/0.20 → 0.977; gravity 0.05 → 0.984, 0.10 → 0.891; **IMU delay 20 ms → 0.477, 40 ms → 0.023, 80 ms → 0.000**. Noise tolerated; one control period of IMU latency halves success, two end it (single-arm delay-1 numbers carry ±5-env run-to-run spread) |
| SF-02 | How accurate must maze localization be? | **DONE both sides** (MuJoCo `21402726`; Isaac `21404944`) | done | MuJoCo route controller: bias ≤ 0.15 m 3/3, 0.20 m 1/3, noise 0.10 m 3/3, heading ≤ 20° 3/3, **drift 0.01 m/s 0/3**. Isaac teacher: bias 0.05 m → 0.961, 0.10 m → 0.930, **0.20 m → 0.773**, heading 3°/10° → 0.969. Requirement: bounded, map-anchored error ≤ 0.10–0.15 m and a few degrees; no drift |
| SF-03 | Does the frozen 22-DoF gait tolerate an estimated attitude? | **DONE** (`21402531`, `results/sensor-fusion-20260923/sf03_summary.json`): Mahony/Madgwick at the 25 Hz policy rate on corrupted MuJoCo gyro + accelerometer, 0.4 s stationary alignment, inspection maze, seeds 0–2 | 48 CPU episodes spent | **Noise is tolerated, latency is not.** L1 (gyro 0.02 rad/s, accel 0.3 m/s², bias 0.02) 3/3 with both filters, gravity RMSE 0.034–0.041; accel noise 1.0 m/s² alone 3/3; gyro noise + bias 0.05 alone 3/3; **one policy step (40 ms) of IMU delay alone 0/3**, and every arm with ≥ 40 ms delay falls whatever the filter or gain. **SF-03b (`21402725`, filter at 200 Hz): 0–30 ms 3/3, 40 ms 1/3, 60 ms 0/3 — the latency budget is ≈30 ms end-to-end** |
| SF-04 | Does a recurrent, modality-dropout `both` policy degrade gracefully? | **DONE.** From scratch: NEGATIVE. From the `Both` checkpoint: delay-only, dropout-only, all-ingredients fine-tunes and the distillation all succeed; bias-only unmeasured (the probe has no bias setting) | ≈16 GPU-h | All-ingredients MLP fine-tune **32 / 32 / 29 / 32 / 32 / 30** (baseline, delay 1, delay 2, LiDAR off, stereo off, both off) vs control 30 / 27 / 0 / 18 / 0 / 0; delay-only 31/32/30/24/0/0; dropout-only 30/22/1/30/32/31; bias-only 31/27/0/32/0/0 (bias tolerance measured separately 2026-09-26, `21434971`: BothBias 31/31/32/32 and the Both control 30/31/31/31 at constant gyro+gravity bias 0/0.02/0.05/0.10 — bias not a threat at this level); student (LSTM) 30/30/27/31/30/30. Each ingredient repairs the rows it randomizes; the recurrent-from-scratch recipe was the failure. Caveat: fixed route, Blind already 32/32 |
| SF-05 | What is the hardware actually? | Recordings per `IMU_INPUT.md` checklist; Allan variance; stereo calibration JSON; LiDAR–IMU extrinsics; a taped maze route with AprilTag or tape ground truth for evo ATE/RPE | none (bench) | Produces the noise/latency numbers SF-01/SF-04 must randomize over |

None of these is funded by this document; each is a backlog row (SF-01…SF-05
in [`REPO_TASKS.md`](REPO_TASKS.md)) under the campaign's compute envelope.

## 7. Literature and open-source practice

The full survey, with a verification flag on every reference, is
[`SENSOR_FUSION_SURVEY.md`](SENSOR_FUSION_SURVEY.md). The recommendations above
rest on these, by layer:

- **Attitude (Layer 1).** Madgwick 2010 (x-io report) and the maintained
  *Fusion* library; Mahony, Hamel, Pflimlin 2008 (IEEE TAC); Valenti et al. 2015
  (the decoupled inclination/heading filter behind ROS 2
  `imu_complementary_filter`, `use_mag=false` by default); Laidig & Seel 2023
  *VQF* (best accuracy, magnetic-disturbance rejection, MIT library); Caruso et
  al. 2021 (tuning dominates filter choice); BROAD benchmark for orientation
  metrics; Kalibr IMU noise model + `allan_variance_ros` for the four sigmas that
  should also parameterize the simulator's `NoiseModelWithAdditiveBias`.
  Barometer: differential altimetry is a floor detector (0.29 m RMSE), not a
  height sensor for a 0.7 m robot.
- **Legged base state (Layer 2).** Bloesch et al. RSS 2012 (kinematic–inertial
  EKF); Hartley et al. IJRR 2020 (contact-aided invariant EKF, validated on a
  biped); Pronto (ROS 2 Humble, multi-sensor); Nobili et al. RSS 2017
  (low-rate LiDAR/stereo corrections into a high-rate proprioceptive filter);
  Cerberus (VINS-Fusion + leg factors, cheap sensing); 2026 humanoid work on
  contact reliability without foot sensors: FOCUS, CoCo-InEKF, Lin et al.
  learned contact; "Four Simple Proprioceptive Estimators" (GTSAM, ROS 2).
- **LiDAR odometry and localisation (Layer 3).** 3-D only: FAST-LIO2, Point-LIO
  (handles IMU saturation from foot impacts), GLIM (native ROS 2, GPU), KISS-ICP
  (LiDAR-only baseline), LIKO and KILVO (bipedal/humanoid LiDAR–inertial–
  kinematic odometry with public datasets); GrandTour benchmark (63 methods on
  legged robots). 2-D: `slam_toolbox`, `nav2_amcl`, Cartographer; humanoid sway
  needs the scan tilt-compensated by the attitude estimate, or a 3-D LiDAR
  sliced with `pointcloud_to_laserscan` in a gravity-aligned frame to
  synthesise the 36-sector observation.
- **Stereo (Layer 4).** SGBM (Hirschmüller) fails on texture-less maze walls;
  learned matchers RAFT-Stereo, IGEV, FoundationStereo / Fast-FoundationStereo
  (TensorRT on Orin); Isaac ROS ESS is the only stereo network with published
  Orin latency (66.5 fps / 17 ms at 576p on AGX Orin, confidence output).
  Depth Anything is monocular and not a substitute. Map-level fusion: nvblox
  ESDF or `elevation_mapping_cupy`, from which both LiDAR sectors and pooled
  depth can be re-derived consistently.
- **Policy-side fusion (Layer 5).** Miki et al. 2022 Science Robotics
  (recurrent belief encoder with a learned attention gate; noise curriculum
  with whole-modality failure at 60/30/10 %; ablations show GRU > MLP and
  gated > ungated); Lee et al. 2020 and RMA (privileged distillation); Pinto
  et al. (asymmetric actor-critic); Liu et al. CoRL 2017 (sensor dropout);
  Agarwal et al. CoRL 2022 (measured 100 ± 20 ms camera period and 10 ± 10 ms
  latency modelled in training); Isaac Lab `DelayBuffer`, per-sensor
  `update_period`, `NoiseModelWithAdditiveBiasCfg`; DPL 2026 (realistic depth
  synthesis for humanoids). Closest system-level analogue: Compton et al. 2026,
  a LiDAR navigation policy over a **frozen** locomotion policy on a Unitree G1.
- **Timing and calibration (Layer 0).** Kalibr (camera–IMU spatial + temporal),
  LI-Init (targetless LiDAR–IMU), `direct_visual_lidar_calibration`
  (LiDAR–camera), Qin & Shen 2018 / OpenVINS (online time offset), VersaVIS
  and GrandTour (hardware triggering, PTP), ROS 2 `message_filters`
  ApproximateTime for scan/depth tuples with the IMU latched at the policy tick.
- **Evaluation.** evo and rpg_trajectory_evaluation (ATE/RPE with PosYaw
  alignment), OpenVINS NEES consistency, BROAD orientation metrics; datasets
  LIKO, KILVO, GrandTour, Hilti; validate estimator-level failures in the
  closed loop, not only command-level dropout.
