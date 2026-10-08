# Recording the physical IMU and measuring its noise (probable Hiwonder IM10A)

Written 2026-10-01. **No physical recording exists yet.** `src/bhl_robust/eval/imu_sim.py`
(`IM10A_DATASHEET`) still holds the vendor's datasheet upper bounds, and
`src/bhl_robust/fusion/attitude.py` (`ImuNoise`) still holds stress-test settings.
This page is the procedure that replaces them with measured values:

1. record three ROS 2 bags of the unit you will mount, plus a settings note (sections 1-3);
2. copy them to the HPC (section 4);
3. run [`scripts/sensors/imu_allan.py`](../scripts/sensors/imu_allan.py) (section 5), which
   writes `imu_measured.json`;
4. plug the JSON into `imu_sim` and `ImuNoise` (section 7).

The conventions and unit conversions are in section 6. They are checked against synthetic
data by `tests/test_imu_allan.py`.

The board identification is the one in [IMU_INPUT.md](IMU_INPUT.md): *probably* a Hiwonder
IM10A, which the vendor's PC software treats as a JY901B. Confirm the label first. The
analysis does not depend on that identification.

Keep robot actuation **disabled** for every capture (IMU_INPUT.md checklist item 3). Sections
1-3 run on the PC that has ROS 2. Only the analysis runs on the HPC, on a login node, without
ROS and without Slurm.

**Sources.** Vendor references were re-read on 2026-10-01:

- the Hiwonder IMU user manual,
  <https://docs.hiwonder.com/projects/IMU-Module/en/latest/docs/1.User_Manual.html>,
  sections 1.1.5, 1.1.6, 1.1.7, 1.5.1, 1.6.1 and 1.7.1-1.7.3;
- the Hiwonder ROS tutorial,
  <https://docs.hiwonder.com/projects/IMU-Module/en/latest/docs/3.IMU_Application_Instructions-ROS_Application.html>,
  sections 3.1.2 and 3.2.1-3.2.3.

Section numbers below refer to these two pages. Anything not taken from them is marked as
this repo's choice or as **unverified**.

**Placeholders.**

| Placeholder | Meaning |
|---|---|
| `<IMU_TOPIC>` | The IMU topic. The vendor ROS 2 example uses `/imu/data_raw` (tutorial §3.2.3). |
| `<MAG_TOPIC>` | The magnetometer topic. The vendor ROS 1 example uses `/wit/mag`. Whether the ROS 2 driver publishes one is **unverified**. |
| `<BARO_TOPIC>` | The barometer topic, if any. |
| `<REF_IMU_TOPIC>` | A second IMU, used as the reference in the tap test. |
| `<hpc-login-host>` | The host you normally `ssh` to. |
| `<date>` | The recording date. |

## 1. Configure the module and write the settings down (vendor PC software)

Use the vendor PC program **`MiniIMU.exe`** with model **JY901B** selected (manual §1.5.1).
Its configuration window is described in §1.6.1. Change only the settings below. Then
screenshot every configuration tab into `settings/`.

| Setting | Set to | Source and reason |
|---|---|---|
| Communication: output rate | **200 Hz**. Use 100 Hz if 200 Hz is not delivered (see the `ros2 topic hz` check in section 2). | §1.6.1: 0.2-200 Hz, factory default 10 Hz. At 10 Hz a sample is up to 100 ms old. The gait's measured budget is about 30 ms end to end ([SENSOR_FUSION.md](SENSOR_FUSION.md), SF-03b). |
| Content | **Three outputs**. The manual's own 200 Hz example is *Acceleration, Angular Velocity and Angle*. | §1.6.1: "If a 200Hz data return rate is required, only three parameters can be selected". IMU_INPUT.md: too much content, or too low a baud, makes the module lower its rate. **Which packets the ROS 2 driver parses is unverified.** Neither the tutorial nor this repo says. Search the driver source for the packet types it handles before you swap Angle for Quaternion; a driver may build its orientation from the Angle packet. |
| Communication: baud | **115200** or higher | §1.6.1: one packet is 11 bytes, and for high rates the manual suggests a baud "usually around 115200". Three packets at 200 Hz need 3 × 11 B × 10 bit × 200 Hz = 66 kbit/s, which is 57 % of 115200. |
| Range: Bandwidth | **The value the robot will use**. The default is 20 Hz. | §1.1.6 and §1.6.1: 5-256 Hz, default 20 Hz. This setting changes both the per-sample noise and the filter delay. Keep it identical in all captures and on the robot. |
| Algorithm | six-axis or nine-axis. **Record which one.** | §1.6.1: nine-axis adds the magnetometer to the attitude. SENSOR_FUSION.md (Layer 1) keeps the magnetometer off indoors by default. |
| Gyro auto-calibration | **Off for capture (a)**: Z-axis Stillness Threshold **0** and Gyroscopic Stillness Threshold **0.001**. Afterwards restore the robot's setting (the defaults are 0.3 and 0.000), and write down which state each capture used. | §1.7.3. While the board is still, the firmware takes the 1 s average rate as the new bias. That erases the bias drift that capture (a) is meant to measure. |

Do the accelerometer level calibration (§1.7.1) and the magnetic calibration (§1.7.2) before
recording, if the robot will use them. Do not change them between captures. Keep the board at
least 20 cm from magnets, steel and electronics (§1.1.5).

**ROS 2 driver.** The vendor example keeps the baud rate in
`~/ros_ws/src/wit_ros2_imu/launch/rviz_and_imu.launch.py` and must be rebuilt after that file
is edited (tutorial §3.2.2). A plain `colcon build` in a workspace you own is enough. The
tutorial's `sudo colcon build` is not needed.

The following are **unverified** for the vendor driver, which this repo has not audited
(IMU_INPUT.md):

- whether it writes configuration to the device at startup (IMU_INPUT.md checklist item 1). A
  driver that sends rate or baud commands silently undoes the table above. Read its source
  first, for example `grep -rn "write(" <driver_src>`;
- what it stamps messages with;
- whether it publishes magnetometer or pressure topics.

Record the driver version or git hash.

Do not run the tutorial's `chmod 777` (§3.1.2) or udev (§3.2.1) steps without reading them. On
Ubuntu, membership in the `dialout` group is the usual non-root way to open `/dev/ttyUSB*`.
None of this belongs on the HPC (IMU_INPUT.md).

Write `settings/settings.txt` with:

- the unit's label (and a photo);
- the firmware version, if shown;
- the table above as actually set;
- the driver name, hash and launch command;
- the PC, OS and ROS distribution;
- the date and the room temperature.

## 2. Confirm the device and the topic (read-only)

```bash
lsusb | grep -i 10c4                 # Silicon Labs CP210x USB-serial bridge (commonly 10c4:ea60)
ls -l /dev/serial/by-id/             # stable device name -> ../../ttyUSBn
ros2 launch wit_ros2_imu rviz_and_imu.launch.py   # vendor example (tutorial §3.2.2), or your driver
ros2 topic list -t                   # find <IMU_TOPIC> [sensor_msgs/msg/Imu]; magnetometer/pressure if any
ros2 topic info -v <IMU_TOPIC>       # exactly one publisher; note its QoS (reliability)
ros2 topic hz -w 2000 <IMU_TOPIC>    # mean ~= configured rate; note min/max/std dev
ros2 topic delay <IMU_TOPIC>         # header-stamp age on arrival: small and positive
ros2 topic echo --once <IMU_TOPIC>
```

Check these in the `echo` output and copy them into `settings.txt`:

- `header.frame_id` is not empty and names the sensor frame. `header.stamp` is current wall
  time and increasing.
- `orientation` has a norm of about 1. A value of `orientation_covariance[0] == -1` means no
  orientation is provided (ROS convention). All-zero covariances mean "unknown", not "perfect".
- Lying flat with the label up, `linear_acceleration.z` reads about **+9.8**. That is specific
  force in m/s² (REP-145). A reading of about 1.0 means the driver publishes g units. A reading
  of -9.8 means an inverted sign.
- At rest, `angular_velocity` is small (|ω| < 0.02 rad/s). Turning the board counter-clockwise
  seen from above gives `angular_velocity.z` > 0, in rad/s rather than deg/s.
- `ros2 topic hz` shows the configured rate. About half the rate means the module lowered its
  output rate (§1.6.1). Raise the baud or drop content.

## 3. Record

Use the MCAP storage if your distribution has it:

- it is the default from ROS 2 Iron onwards;
- on Humble, install `ros-humble-rosbag2-storage-mcap`;
- on Foxy (the vendor tutorial's distribution) its availability is unverified. Drop `-s mcap`
  and you get sqlite3 `.db3` files instead.

The analysis reads both formats.

Keep the laptop on mains power with sleep disabled, and leave other USB devices idle. If a
bag stays empty because the publisher is best-effort, see `ros2 bag record --help` for the QoS
override option of your distribution (`--qos-profile-overrides-path`).

### (a) Stationary, at least 2 h (Allan variance)

Lay the board flat, label up, on a heavy rigid surface: a concrete floor or a massive bench.
Keep it away from fans, vents, foot traffic and motors, and at least 20 cm from steel and
magnets (§1.1.5). Tape the cable so it cannot tug. Power the board on, start the bag at once,
and touch nothing until it ends:

```bash
mkdir -p ~/imu_rec && cd ~/imu_rec
timeout -s INT 8700 ros2 bag record -s mcap -o imu_static_$(date +%Y%m%d_%H%M) \
    <IMU_TOPIC> [<MAG_TOPIC>] [<BARO_TOPIC>]          # 2 h 25 min; SIGINT closes the bag cleanly
```

The analysis drops the first 15 min (`--trim-start-s 900`) as warm-up. That is this repo's
choice, not a vendor figure. The manual gives only a 1000 ms startup time (§1.1.7) and the
temperature coefficients: ±0.005-0.015 (°/s)/°C for the gyro and ±0.15 mg/°C for the
accelerometer (§1.1.6). Self-heating after power-on is the usual reason to discard the start.
The trimmed minutes stay visible in the drift plot.

**Why 2 h, and what a 30 min minimum loses.** An Allan point at averaging time τ, from a
record of length T, has a relative error of about 1/√(2(T/τ - 1)). That is 24 % at τ = T/10.
Each term therefore needs T ≈ 10τ, and the fits stop at τ = T/5.

- **White noise N** (read at τ = 1 s) is fixed within about 0.5 % by minutes of data.
- **Bias instability B** sits at the minimum of the curve. For MEMS parts the minimum
  typically lies between τ ≈ 10 s and 1000 s.
- **Rate random walk K** is the rising +½ slope beyond that minimum.

The synthetic results in section 6 show the difference:

| Quantity (16 seeds, 200 Hz) | 2 h record | 30 min record |
|---|---|---|
| min-method B / analytic minimum (16th / 50th / 84th percentile) | 0.96 / 0.99 / 1.04 | 0.80 / 0.91 / 1.01 |
| least-squares B / true B | ±2 % | ±3 % |
| records in which K was visible | 13 of 16 (median 1.00, 16-84 % 0.78-1.13) | 3 of 16 |

At **30 min, the minimum to accept**, N is still accurate, but B reads about 10 % low on
average, and B is flagged as an upper bound whenever the curve minimum sits at the edge of the
usable τ range. K is usually lost. The JSON then carries only an upper bound for
`gyro_bias_walk`, and any K it does report is marked LOW CONFIDENCE. Temperature drift and
firmware bias steps are also harder to see.

### (b) Slow rotations about each labelled axis

This capture checks signs, frames and the quaternion convention, and gives the six-position
accelerometer bias.

Rest the board against a square block or a box corner on the table for every hold, not in
your hand. Each move is a slow rotation of about 3 s about the board's printed axis. Use the
right-hand rule: thumb along +axis, fingers curl in the positive direction. A 10 s hold follows
each move. The first and last holds are 30 s. "Z-up" means flat with the label up.

```bash
ros2 bag record -s mcap -o imu_rot_$(date +%Y%m%d_%H%M) <IMU_TOPIC>
```

| # | Move (then hold) | Face pointing up | Expected |
|---|---|---|---|
| H1 | start flat, 30 s | +Z | a_z ≈ +g |
| M1 | +90° about X (the +Y edge rises) | +Y | a_y ≈ +g; ω_x > 0 during the move |
| M2 | -90° about X | +Z | |
| M3 | -90° about X | -Y | a_y ≈ -g |
| M4 | +90° about X | +Z | |
| M5 | +90° about Y (the -X edge rises) | -X | a_x ≈ -g; ω_y > 0 |
| M6 | -90° about Y | +Z | |
| M7 | -90° about Y | +X | a_x ≈ +g |
| M8 | +90° about Y | +Z | |
| M9 | +90° about Z (counter-clockwise seen from above) | +Z | ω_z > 0; gravity unchanged |
| M10 | -90° about Z | +Z | |
| M11 | +180° about X (a flip, about 5 s) | -Z | a_z ≈ -g |
| M12 | -180° about X, then hold 30 s | +Z | |

The `rotations` subcommand checks this exact sequence. Its defaults are `--expect-faces` and
`--expect-moves`. It then does three things:

- It integrates the gyro between holds and compares each rotation with the change in gravity
  seen by the accelerometer. All 48 signed axis permutations × 3 unit scalings are tried, which
  catches axis swaps, sign flips and deg/s published as rad/s.
- It compares the published quaternion with the gravity direction and with the gyro. Four
  readings are tried: xyzw or w-first order, each as body→world or world→body.
- It computes the six-position accelerometer bias and scale error per axis.

A single lying pose cannot separate accelerometer bias from gravity, which is why this capture
is needed.

### (c) Tap test: latency against another sensor

Fix the IMU and a reference sensor rigidly to the same plate, or tape the IMU to the camera
body. Both must be stamped in the same clock: the same PC, or clocks synchronised with chrony
or PTP. Give about 15 sharp taps with a pen or screwdriver handle, about 2 s apart, on the
plate between the sensors. Nothing else should move.

```bash
ros2 bag record -s mcap -o imu_tap_$(date +%Y%m%d_%H%M) <IMU_TOPIC> <REF_IMU_TOPIC>
```

The `tap` subcommand returns the **offset** of this IMU's stamps relative to the reference for
the same physical tap. A positive offset means this IMU is later. It is an absolute stamp
latency only after adding the reference's own latency, which must be known from elsewhere and
is passed as `--ref-latency-ms`. Only that sum fills `imu_noise.delay_steps`; without it,
`delay_steps` stays 0 and the JSON keeps the relative offset. On the synthetic test the method
recovers a 23 ms offset within 2 ms (two synthetic runs: 0.4 and 0.7 ms error).

The end-to-end figure to compare with the ≈30 ms budget is the stamp latency plus the
consumer's processing time. If the only reference is a camera image stream, note the stamp of
the first moving frame of each tap and pass them as a one-column CSV with `--ref-events`. Every
bag also reports the IMU's receive-minus-stamp time. That is transport delay only: it says
nothing about how old the sample was when it was stamped.

### (d) Optional: magnetometer and barometer

At 200 Hz the three allowed outputs leave no room for the magnetic field or air pressure. If
the driver publishes them:

1. reconfigure the module to a lower output rate with those outputs added, and recheck the baud
   with the 11-byte-per-packet rule;
2. record 15 min stationary with capture (a)'s command and topics.

If the driver does not publish them, `mag_resolution_mgauss` and `baro_noise_pa` stay at the
datasheet values, labelled as such in the JSON's `provenance`.

When you are done, **restore the robot's gyro auto-calibration setting**.

## 4. Check the bags and copy them to the HPC

```bash
ros2 bag info ~/imu_rec/imu_static_*/  # message count ~= rate x duration (200 Hz x 8700 s ~ 1.74 M)
du -sh ~/imu_rec/*                     # (a) at 200 Hz is ~0.7 GB (370-390 bytes per Imu message)
```

Copy whole bag directories, `metadata.yaml` included, plus `settings/`. The destination below
is outside the repo. **Check the quota first.** On 2026-10-01, project 30762, which holds
`/nfs/hpc/share/sanchej7`, used 1.529 TB. That is over its 1.5 TB soft limit (hard limit
2 TB), with a grace period of about 3 weeks 6 days left.

```bash
ssh sanchej7@<hpc-login-host> 'lfs quota -h -p 30762 /nfs/hpc/share; mkdir -p /nfs/hpc/share/sanchej7/Humanoid_Lite/imu_recordings/<date>'
rsync -avP ~/imu_rec/ sanchej7@<hpc-login-host>:/nfs/hpc/share/sanchej7/Humanoid_Lite/imu_recordings/<date>/
```

Run `mkdir -p` first: rsync creates only the last directory level of the destination, unless
you use rsync 3.2.3 or later with `--mkpath`.

## 5. Analyse on the HPC (login node, no ROS, no Slurm)

The script needs only numpy. matplotlib is optional and is used for the plot. Reading bags
needs the pure-Python `rosbags` package, which the shared venv did not have on 2026-10-01. Use
one of these:

- `pip install rosbags` into the environment you analyse with;
- a separate small venv that holds `numpy`, `matplotlib` and `rosbags`, since the script needs
  nothing from the repo;
- `imu_allan.py export-csv <bag> --output x.csv` on any machine that has rosbags, then copy the
  CSV. Every subcommand accepts CSV input.

```bash
R=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
REC=/nfs/hpc/share/sanchej7/Humanoid_Lite/imu_recordings/<date>
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python   # or the python of the venv that has rosbags
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # login-node limits

$PY $R/scripts/sensors/imu_allan.py rotations  $REC/imu_rot_*    --imu-topic <IMU_TOPIC> --out-dir $REC/out/rot
$PY $R/scripts/sensors/imu_allan.py tap        $REC/imu_tap_*    --imu-topic <IMU_TOPIC> --ref-topic <REF_IMU_TOPIC> \
    [--ref-latency-ms <reference latency>] --out-dir $REC/out/tap
$PY $R/scripts/sensors/imu_allan.py stationary $REC/imu_static_* --imu-topic <IMU_TOPIC> --nominal-rate 200 \
    --trim-start-s 900 --trim-end-s 60 --latitude-deg <lat> --height-m <m> \
    --rotations-report $REC/out/rot/rotations_report.json --tap-report $REC/out/tap/tap_report.json \
    --out-dir $REC/out/static
```

Pass `--latitude-deg` and `--height-m` (for example 44.56 and 70 for Corvallis) so the gravity
check uses WGS84 normal gravity, or pass `--local-g`. Expect 1-2 min for the 2 h capture on a login node (measured
here: about 20 µs per bag message to read, about 5 s per channel for the Allan curve).
`python imu_allan.py --doc` prints the conventions.

| Output | Contents |
|---|---|
| `imu_measured.json` | The drop-in values (section 7). |
| `imu_allan_report.json` | Everything else, including the Allan curves. |
| `imu_allan.png` | Allan deviation per axis with the fitted lines, plus the drift plot. |
| `rotations_report.json` | The rotations capture's checks and six-position values. |
| `tap_report.json` | The tap offsets. |

Read every `WARNING:` line. Each one flags a known trap:

- the firmware lowering the output rate;
- duplicate samples;
- correlated samples behind the device low-pass filter;
- output steps (0.061 °/s and 0.0005 g per LSB, §1.1.6) comparable to the noise;
- gyro auto-calibration left on (a bias step, or a near-zero mean bias);
- the Allan minimum at the edge of the usable τ range;
- g units or deg/s instead of SI;
- a gravity sign or axis other than REP-145's;
- a quaternion convention other than xyzw body→world.

## 6. Conventions and unit conversions

These are used by every number in the outputs. They are also printed by `imu_allan.py --doc`
and checked by `tests/test_imu_allan.py`.

| Quantity | Definition | Units |
|---|---|---|
| Allan deviation | IEEE Std 952-1997 overlapping estimator on the rate samples, τ = m/f_s | rate units |
| **White noise N** (gyro: angle random walk; accel: velocity random walk) | The slope -½ line σ(τ) = N/√τ, **read at τ = 1 s**. Its level is the -½ term of a least-squares fit of the IEEE 952 model σ²(τ) = N²/τ + (0.6643 B)² + K²τ/3 + R²τ²/2 over the white region and beyond. | (rad/s)/√Hz = rad/√s; (m/s²)/√Hz. ARW in °/√h = N in (°/s)/√Hz × 60. |
| **Per-sample std at output rate f** | **σ_d = N·√f** | rad/s, m/s² |
| **Bias instability B** | **B = σ_min / 0.6643**, where 0.6643 = √(2 ln 2/π) is the flicker floor. σ_min is the minimum of the lightly smoothed curve within τ ≤ T/5. Flagged as an upper bound at the range edge. The least-squares flat term is reported as `fit.B`. | rad/s (also °/h), m/s² (also mg) |
| **Rate random walk K** | The +½ line σ(τ) = K·√(τ/3), so **σ(3 s) = K on that line**. Reported only when at least 3 points beyond the minimum rise above it by more than 2 relative errors. Its level comes from the least-squares fit, with the graphical read kept as `K_graph`. Otherwise `K_upper` is an approximate one-sided 95 % bound (chi-square, random-walk equivalent degrees of freedom, NIST SP 1065). | (rad/s)/√s = rad/s²/√Hz |
| Gyro bias | The mean of the trimmed stationary record. It includes Earth rotation (≤ 0.0042 °/s) and is one sample of the turn-on bias. | rad/s |
| Accelerometer bias | From the six-position capture (b). From one pose, only the along-gravity part, \|mean\| - g, is observable; the JSON then labels it PARTIAL. | m/s², mg |

**Why σ_d = N·√f.** Independent samples of standard deviation σ_d at rate f have
AVAR(τ) = σ_d²/(f·τ) exactly. The mean of m = f·τ samples has variance σ_d²/m, and the Allan
variance of white noise equals that variance. Hence N² = σ_d²/f.

In PSD terms, N² is the two-sided white level. The rms in a one-sided band B_w is
N·√(2·B_w), which equals N·√f at the Nyquist band. **"N·√(f/2)" understates the per-sample
noise by √2.** It is right only when N denotes the one-sided density.

The vendor quotes noise "at 100 Hz bandwidth" (§1.1.6). For an ideal 100 Hz band that is
N·√200. It is reported only for comparison, as `*_datasheet_equiv_100hz_bw`.

The same two conventions are the Kalibr IMU noise model's
(<https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model>):

- noise density = Allan deviation at τ = 1 s, with discrete std N/√Δt;
- random walk = the fitted +½ line at τ = 3 s, with discrete step K·√Δt.

`imu_measured.json` carries them as `kalibr_imu`. Kalibr advises inflating static values for
low-cost IMUs.

**Constants.**

| Conversion | Value |
|---|---|
| 1° | π/180 rad |
| °/h | °/s × 3600 |
| 1 mg | 9.80665 × 10⁻³ m/s² (standard gravity, as `imu_sim.G`) |
| 1 T | 10⁷ mGauss = 10⁶ µT |
| Pressure | Pa |

**Correlated samples.** Behind the device low-pass filter (Bandwidth below Nyquist, default
20 Hz), the measured per-sample std is smaller than N·√f. At 20 Hz with a 200 Hz output it was
about 0.48 × N·√f in the synthetic test. The JSON's noise keys are the white-equivalent
N·√f, which reproduces the angle random walk in a simulator that adds independent noise per
sample. The raw per-sample std is in the report as `sample_std`.

In that configuration N itself reads about 2 % low: 20 seeds gave a median of 0.981 and a range
of 0.959-1.003. Before the white-region start was tightened in this script it read 5 % low.

**Output steps.** The output steps (0.061 °/s and 0.0005 g) add q²/12 of white variance per
sample. When the noise is below 2 steps, the white-noise figure is quantization-limited and the
script warns.

**Synthetic validation (true vs recovered).** Function level, one seed per row; produced
2026-10-01. Each channel is white (σ_d = N·√f drawn directly), plus flicker (floor 0.6643 B),
plus an ImuNoise-style rate random walk.

- Gyro: N 0.0035 (°/s)/√Hz, B 10 °/h, K 0.000261 (°/s)/√s.
- Accelerometer: N 0.070 mg/√Hz, B 0.05 mg, K 0.00407 mg/√s.

How to read the columns:

- "B_min / truth" divides by the analytic minimum of the generated process / 0.6643, because
  white noise and RRW lift the minimum above 0.6643 B.
- "n/v" means K was not visible; the number in brackets is then K_upper / K.

| f | T | sensor | N rec / true | N·√f / σ_d | N·√(f/2) / σ_d | B_min / truth | B_fit / B | K rec / true |
|---|---|---|---|---|---|---|---|---|
| 100 Hz | 2 h | gyro | 0.9996 | 0.9996 | 0.707 | 0.966 | 1.004 | 0.91 |
| 100 Hz | 2 h | accel | 0.9991 | 0.9991 | 0.706 | 0.905 | 0.991 | 0.74 |
| 100 Hz | 30 min | gyro | 1.0007 | 1.0007 | 0.708 | 0.815 | 0.991 | n/v (bound 2.11) |
| 100 Hz | 30 min | accel | 0.9949 | 0.9949 | 0.704 | 0.809 | 0.991 | n/v (bound 1.74) |
| 200 Hz | 2 h | gyro | 1.0006 | 1.0006 | 0.708 | 1.036 | 0.982 | 1.09 |
| 200 Hz | 2 h | accel | 0.9992 | 0.9992 | 0.707 | 1.032 | 0.985 | 1.17 |
| 200 Hz | 30 min | gyro | 1.0009 | 1.0009 | 0.708 | 0.930 | 1.001 | n/v (bound 1.67) |
| 200 Hz | 30 min | accel | 0.9970 | 0.9970 | 0.705 | 1.008 | 0.983 | 1.56 (LOW CONFIDENCE) |

A white-only control on the same draws gave N·√f within 0.1 % of the drawn sample std at every
row, while N·√(f/2) was 0.70-0.71 of it.

The repo's own `ImuNoise` class makes a round trip: 100 Hz, 10 min, 20 seeds. `gyro_std` and
`accel_std` came back within 0.7 %, and `gyro_bias_walk` came back at 0.90-1.09 of its value.

The 16-seed figures quoted in section 3(a) come from a separate Monte Carlo at 200 Hz.

## 7. Plugging `imu_measured.json` into the repo

The script changes nothing in the repo. When a real recording exists, the numbers go in the
places below.

### 7.1 `src/bhl_robust/eval/imu_sim.py`: `IM10A_DATASHEET` and `SimIM10A`

`imu_measured.json` carries the **same eight keys** as `IM10A_DATASHEET`:
`gyro_noise_dps`, `gyro_bias_dps`, `accel_noise_mg`, `accel_bias_mg`, `mag_resolution_mgauss`,
`baro_noise_pa`, `max_rate_hz` and `source`. Each key has a `provenance` entry that says
*measured*, *PARTIAL* or *datasheet fallback*.

- **Do not overwrite `IM10A_DATASHEET` in place.**
  `tests/test_imu_rig_panels.py::test_datasheet_numbers_are_the_published_upper_bounds` pins
  it, and it stays the comparison baseline. Pass the measured values through the existing
  `SimIM10A(..., params=...)` argument instead.
- **Rescale the two noise keys to the panel's rate.** `SimIM10A` adds independent
  N(0, σ) noise per output sample at its own rate. That rate is `--imu-panel-rate`, default
  100 Hz; use the rate it actually runs at, `1/(every·physics_dt)`. The JSON's
  `gyro_noise_dps` and `accel_noise_mg` are per-sample σ at the *measured* rate (key
  `rate_hz`). Recompute them from `noise_density`:

  ```python
  import json, math
  from pathlib import Path
  from bhl_robust.eval.imu_sim import IM10A_DATASHEET, SimIM10A

  def im10a_measured(path, rate_hz):
      m = json.loads(Path(path).read_text())
      p = {k: m[k] for k in IM10A_DATASHEET}            # the eight drop-in keys only
      p["gyro_noise_dps"] = m["noise_density"]["gyro_dps_rthz"] * math.sqrt(rate_hz)
      p["accel_noise_mg"] = m["noise_density"]["accel_mg_rthz"] * math.sqrt(rate_hz)
      return p

  imu = SimIM10A(model, slot, seed, physics_dt, rate_hz=r, params=im10a_measured(path, r))
  ```

  Pass only the eight keys: `SimIM10A.meta()` copies every param into the clip's JSON sidecar.
- `max_rate_hz` stays the device capability, 200. `SimIM10A.__init__` raises an error if
  `rate_hz > max_rate_hz`, so writing the delivered rate there (for example 199.6) would break
  `--imu-panel-rate 200`. The delivered rate is in `rate_hz`.
- `gyro_bias_dps` and `accel_bias_mg` keep the simulator's meaning: the magnitude of a fixed
  bias in a random direction.
  - The gyro value is one power cycle's turn-on bias. Any residual motion inflates it. If the
    vendor auto-calibration was on, it is deflated, and the script warns.
  - `accel_bias_mg` is the six-position |bias| only when `--rotations-report` was merged.
    Otherwise it is the along-gravity component, labelled *PARTIAL*.
- Then update the `imu_sim` module docstring ("DATASHEET UPPER BOUNDS") and `source` to cite
  the recording: bag name, date, rate, Bandwidth and auto-calibration state.

### 7.2 `src/bhl_robust/fusion/attitude.py`: `ImuNoise`

`imu_noise` holds exactly the constructor fields (all but `seed`), per call at
`imu_noise_rate_hz` (default 200 Hz, `--sim-rate`). `imu_noise_by_rate` repeats them at 25,
50, 100 and 200 Hz:

```python
from bhl_robust.fusion.attitude import ImuNoise
m = json.loads(Path(path).read_text())
noise = ImuNoise(**m["imu_noise_by_rate"]["200"], seed=seed)      # ImuNoise called at 200 Hz
```

| `ImuNoise` field | JSON source | Unit and conversion |
|---|---|---|
| `gyro_std` | max-axis N_gyro · √(call rate) | rad/s per call. The factor is √rate because ImuNoise draws independent noise per call. |
| `accel_std` | max-axis N_accel · √(call rate) | m/s² per call |
| `gyro_bias` | \|mean gyro\| of the stationary record | rad/s, as a magnitude; ImuNoise picks the direction |
| `gyro_bias_walk` | the rate random walk K | rad/s/√s. No rate conversion: ImuNoise adds N(0,1)·K·√dt per call, a Wiener process whose Allan deviation is K·√(τ/3). |
| `delay_steps` | round(`imu_latency_ms` · call rate / 1000) | Integer calls. Filled only when a tap report with `--ref-latency-ms` was merged; otherwise 0. |

- **`gyro_bias_walk` can be an upper bound.** When K was not visible, `gyro_bias_walk` is the
  K_upper bound, and `gyro_bias_walk_kind` says so. A bound in a stress model is a
  conservative choice. Override it with 0, or with the least-squares value from the report,
  if you prefer.
- **Rate conventions differ between the two bench scripts.**
  - `scripts/bench/maze_explore.py` runs the sensor-rate filter by default
    (`--imu-rate-hz 200`). Its flags are `--imu-gyro-std`, `--imu-accel-std`,
    `--imu-gyro-bias` and `--imu-gyro-bias-walk`, taken from `imu_noise_by_rate["200"]`.
    Set `--imu-delay-ms` to the stamp latency plus processing.
  - `scripts/bench/inspection_maze.py` defaults to `--imu-rate-hz 0`, which is the 25 Hz
    policy rate. There, use `imu_noise_by_rate["25"]` and `--imu-delay-steps`.
  - The 25 Hz σ assumes the 200 Hz stream is *averaged* down. If the deployment *decimates*
    (keeps every 8th sample), use the 200 Hz σ.
- **What `ImuNoise` cannot hold.** It has no field for bias instability, accelerometer bias,
  per-axis differences or the magnetometer. Those stay in the JSON (`*_bias_instability_*`,
  `per_axis`, `kalibr_imu`). They are available for a richer model, such as Isaac Lab's
  `NoiseModelWithAdditiveBias` (SENSOR_FUSION.md §7), or for training randomization ranges
  (SENSOR_FUSION.md §4 item 2, SF-04).

## 8. What this covers of the IMU_INPUT.md acceptance checklist

| IMU_INPUT.md item | Covered by | Not covered |
|---|---|---|
| 1. Identity, serial settings, output fields, algorithm mode, mounting transform, driver version; inspect the driver for startup writes | Section 1: `settings/` screenshots, `settings.txt`, driver hash, the source check | The **mounting transform**: the captures use the loose module, so repeat (b) on the robot for it |
| 2. Read-only `ros2 topic list -t`, `echo --once`, `hz`; frame_id, timestamps, finite values, quaternion availability, actual rate | Section 2. The report repeats these from the bag: `frame_ids`, covariance, `orientation_unavailable_frac`, timing, stamp age. | - |
| 3. Stationary capture plus rotations about each labelled axis; gravity sign, gyro signs, orientation direction; bias, noise, dropout and delay from recordings; raw data and settings kept; actuation off | Captures (a), (b) and (c), with the `stationary`, `rotations` and `tap` reports | The ENU yaw reference (board Y = north in nine-axis mode, manual §1.3) is not checked. Stationary yaw drift is reported. |
| 4. Align IMU, camera and LiDAR clocks; replay through validity checks; reject stale or future samples | Partly: the tap offset against a reference, and the receive-minus-stamp distribution | Replay through the `sensor_io.imu_features` freshness checks; camera/LiDAR clock alignment ([STEREO_INPUT.md](STEREO_INPUT.md)) |

SF-05 ([SENSOR_FUSION.md](SENSOR_FUSION.md) §6) needs more than this. This page delivers the
IMU part: identity and settings, the stationary and rotation captures, Allan variance, and
latency when a reference exists. Stereo calibration, LiDAR-IMU extrinsics and the taped
ground-truth route are separate tasks.

## 9. Caveats that need the physical device

- **Identity.** The board, its revision and firmware, and which vendor protocol it speaks are
  still a *probable* identification (IMU_INPUT.md).
- **Driver behaviour is unverified.** That covers the packet types the ROS 2 driver parses (and
  so which three outputs to enable at 200 Hz), any configuration writes at startup, its
  timestamps, and whether it publishes magnetometer and pressure topics in ROS 2.
- **Delivered rate.** Whether 200 Hz is actually delivered at the chosen baud and content.
- **MCAP on Foxy.** Whether the MCAP storage plugin is available on the robot PC's ROS 2
  distribution.
- **Absolute latency** needs a reference sensor whose own latency is known. Without one, only
  a relative offset is measured and `delay_steps` stays 0.
- **Mounting.** The mounting transform (sensor to `base_link`) and the yaw reference need the
  module mounted on the robot.
- **Scope of one recording.** It covers one unit, one temperature, one power cycle and one
  Bandwidth setting. The turn-on bias changes between power cycles, and the temperature
  coefficients are not measured.
- **Short records.** From a 2 h record, rate random walk scatters by tens of percent
  (synthetic: 16-84 % 0.78-1.13). Bias instability is overstated when the white and random-walk
  lines overlap the floor; `fit.B` is then the less biased figure.
- **HPC quota.** As of 2026-10-01 the share is over its soft quota, with about 3.8 weeks of
  grace left (section 4).
