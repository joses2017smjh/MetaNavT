# Two- and three-humanoid coordination: inspection and airlock

The runnable baseline is a team navigation task in **one MuJoCo world**, using
the repository's existing **22-DoF Isaac-trained humanoid locomotion policies**.
The supervisor currently reads exact base poses and the map. It is an oracle
feasibility reference for future sensor-based coordination, not an end-to-end
learned perception policy, a new MARL result, or a cooperative lifting result.

Each member must reach its assigned inspection station. All assigned stations
must be occupied simultaneously by upright robots for 0.6 s to latch a physical
door open. Members then take turns merging into the shared doorway; the next
member is released after its predecessor reaches its exit station. Success
requires every member to traverse the doorway and hold its assigned exit for
0.6 s, without falling or contacting another robot or a wall. Station triggers
are proximity events, not hand-operated buttons. The floor is flat and the
1.5 m doorway has ample clearance for BHL's modest joint authority.

This combines navigation, waiting, role synchronization, ordered passage, and
team rendezvous while using an already trained motor skill. Roles have
staggered starts. Two controls expose the cooperation requirement: `no_wait`
leaves an inspection station immediately on individual arrival; `withhold_last`
keeps the last robot at its start, retaining normal balance control. Removing
one role cannot be compensated by simply surviving to timeout.

## Why not restart the old lifting jobs?

The existing job ledger records all six long plank runs stuck at the 0.04 m
curriculum floor with zero task success. Its stand-off correction removed
initial ejection, but the agents learned to stand near the plank. Earlier
cooperative results also depended on incorrect spawn quaternions and floor
placement. The old MuJoCo `coop_replay` explicitly models independent pairs,
plus a solo control for odd crew sizes: its three-robot clips do not establish
three-member cooperation on one object.

Those are reasons to separate two prerequisites: demonstrate a real shared
team objective using existing locomotion, and validate grasp/contact/stance
before investing in carrying. The new task addresses the first prerequisite.
No changes to existing lifting training or user-edited spawn fixes are needed.

## Research checked on 2026-09-19

| Primary source | Relevant result | What transfers to this project |
| --- | --- | --- |
| [Rhythm, RSS 2026; revised September 10](https://arxiv.org/abs/2603.02856) | Dual G1 interactions use interaction-aware retargeting and coupled-dynamics rewards, with real robot demonstrations. | Physical interaction requires feasible embodiment-specific references and measured contact; its G1 policy is not a BHL checkpoint. |
| [TeamHOI, CVPR 2026](https://arxiv.org/abs/2603.07988) | One policy covers two to eight simulated humanoids with teammate tokens, a masked motion prior, and formation rewards. | Variable team size needs explicit teammate representation and one shared task; merely cloning independent pairs is insufficient. |
| [SynAgent, revised August 2, 2026](https://arxiv.org/abs/2604.18557) | Transfers solo interaction skills into cooperative imitation and trajectory-conditioned control. | Reuse validated solo skills before adapting cooperative control; interaction-preserving retargeting is additional work. |
| [Marope, June 2026](https://arxiv.org/abs/2606.08064) | Hierarchical MARL couples decentralized manipulation with centralized timing in multi-humanoid rope skipping. | Separate motor execution from team scheduling and test synchronization explicitly; rope skipping itself exceeds the current BHL capability evidence. |

The new baseline adopts the hierarchy and explicit coordination evaluation as
engineering choices. It does not claim to reproduce those papers. A subsequent
learned supervisor should consume local LiDAR/stereo features and communicated
role state, while a privileged training critic can retain global state. First
compare that supervisor against the oracle reference at matched seeds and
delays, then introduce communications loss. A physical carrying extension
additionally needs a stable loaded stance, hand collision geometry, an actual
grasp model, and sustained shared-payload support in both simulators.

## Running and interpreting the benchmark

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python scripts/bench/team_airlock.py \
  --deploy external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml \
  --upstream external/Berkeley-Humanoid-Lite \
  --cache-dir /tmp/bhl-team-airlock-pair \
  --crew 2 --seeds 5 --seconds 60 \
  --modes coordinated no_wait withhold_last \
  --output results/team_airlock_pair.json --gate
```

Repeat with `--crew 3` and distinct cache/output paths. The script deliberately
rejects biped and non-75-observation checkpoints. It preserves the trained PD
gains, torque limits, 40 ms policy period, and 0.5 ms MuJoCo physics period.
ONNX inference uses one CPU thread per shared session. No GPU is required for
the benchmark. The optional `--video results/clips/team_pair.mp4` renders the
first coordinated seed and needs EGL or OSMesa.

An HPC allocation of 2–4 CPU cores and 8 GB is sufficient; use a 1–2 h limit
for five seeds × three controls per crew, allowing for network asset reads.
Submission must respect the cluster's current account/QoS: older `share`
scripts in this repository recorded `QOSMinGRES` on GPU-eligible queues.
Independent crew sizes or checkpoint seeds can run concurrently, each with a
unique cache path. Heavy GPU training is not useful for this fixed-policy
baseline.

The JSON is flushed after every episode and contains per-second position,
command, phase, and tilt traces, door release/completion times, station visits,
members crossing, failure reasons, minimum separation, and contact intervals.
Contacts are checked at **every physics substep**. The gate requires at least
80% clean coordinated success, lower no-wait success, and zero withheld-role
success. An incomplete control set is reported separately. Passing on one
seed is a smoke test, not statistical evidence of 80% population reliability.

Initial seed-0 feasibility probes on September 19 completed the pair task in
22.96 s with the DR1.0-s0 checkpoint and 23.2 s with terrain-s0, with no sampled
falls or contacts. Those initial probes preceded the stronger physics-substep
contact monitor; the subsequent gated runs are the authoritative validation.
A 0.28 m/s velocity command left one member standing, while 0.40 m/s initiated
walking reliably in these probes. The latter is the default; retest instead
of assuming any slower command is automatically easier for a learned gait.

The stricter monitor subsequently confirmed pair success at 22.96 s, no-wait
failure with robot/wall collisions and no door release, and withheld-role
failure with only one station visited. The three-humanoid coordinated smoke
also completed in 30.68 s, with three stations and crossings, no falls or
contacts, and 0.448 m minimum base separation. See
`results/team_airlock_pair_gate.json` and `results/team_airlock_trio_probe.json`.
These are single-seed smoke results; the queued matched-seed evaluations are
needed to measure reliability.

## Optional actual sensor inputs and a reactive brake

`--sensor-mode off` is the original default. Its commands and mission rules are
unchanged. Three opt-in modes provide a separate, clearly labelled experiment:

| Mode | Inputs | Effect on commanded motion |
| --- | --- | --- |
| `record` | Actual simulated rays and IMU measurements | Records inputs; retains the oracle supervisor's commands. |
| `reactive` | Same sensor packets | Brakes translation for nearby geometry or excessive measured IMU tilt/angular speed. |
| `reactive_dropout` | Same acquisition with seeded packet loss | Retains the last received packet for up to 150 ms, then stops translation until fresh input returns. |

The full humanoid carries a body-attached 360-degree scanner at 0.72 m above
its base-frame origin: 108 rays become 36 sector minima, with a 12 m cutoff.
The pair of 8×8 ray cameras has a 60 mm baseline, 60.5-degree vertical field
of view, and 20-degree downward pitch at a 0.70 m mounting height. Full body
roll and pitch affect these rays. Visual meshes and the sensing robot itself
are excluded, while collision geometry belonging to other robots and the world
remains visible. Exteroceptive samples nominally arrive at 10 Hz, quantized to
the 40 ms control clock; actual sample timestamps are recorded.

The depth channels are **idealized paired ray depth**, not RGB images or
stereo-disparity estimates. They cannot demonstrate stereo correspondence,
texture sensitivity, or detection accuracy. IMU values come from the MJCF's
actual site quaternion, gyro, and accelerometer, converted through
`bhl_robust.sensor_io.imu_features` into timestamped body gravity, rad/s, and
specific force in m/s². These mounts and dropout rates are declared simulation
choices, not calibration for the user's USB module. No detector is configured:
every sensor trace states `ssd_status=not_configured_no_rgb_or_weights`.

The brake consumes directional LiDAR ranges and forward central depth pixels;
the nearest selected range scales translation from normal speed at 0.90 m to
zero at 0.42 m. It uses no simulator geom labels to identify obstacles. The
mission planner still uses exact map/base position, so this is **sensor-reactive
motion with oracle localization**, not sensor-only navigation or SLAM. Normal
proprioceptive IMU feedback to the learned low-level policy remains active even
during exteroceptive packet loss.

Append `--sensor-mode reactive` to the earlier command, and use a distinct
output path. The direct safety control is `--modes no_wait`: compare contact
intervals and progress against the original no-wait baseline at identical
seeds and horizons. To test a complete exteroceptive outage, use
`--sensor-mode reactive_dropout --sensor-dropout-probability 1`. The default
drop probability is 0.35, an explicitly uncalibrated stress test. Mission
success gates remain unchanged; collision avoidance alone does not count as
successful cooperation.

Sensor-enabled JSON includes raw feature arrays, timestamps/freshness masks,
the original and filtered commands, brake factors, and aggregate packet,
range-braking, IMU-braking, and stale-stop counts. The basic transforms,
directional brake, stale-input behavior, and actual MuJoCo ray distance/own-body
mask restoration are covered by `tests/test_team_sensors.py`.

The seed-0 matched 30 s no-wait check measured an actual input effect:

| Input use | Robot-contact intervals | Wall-contact intervals | Team success |
| --- | --- | --- | --- |
| Record only | 536 | 580 | No |
| Reactive brake | 0 | 0 | No |

An interval is a 40 ms control step with contact in at least one physics
substep, not a count of distinct collisions. Both arms received 600 actual
exteroceptive packets. The reactive arm changed commands in 1,271 of 1,500
robot-steps. It avoided collision but correctly still failed the missing
synchronization objective. With coordination restored, the reactive pair
completed in 22.96 s with zero contacts. A separate 8 s complete-outage smoke
received zero of 160 attempted packets, visited zero stations, and stopped
translation in all 400 robot-steps while retaining balance.

These are single-seed checks, not robustness claims. Evidence is in
`results/team_airlock_sensor_record_probe.json`,
`results/team_airlock_sensor_reactive_probe.json`, and
`results/team_airlock_sensor_dropout_probe.json`.
