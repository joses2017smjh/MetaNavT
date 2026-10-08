# Physical IMU input contract

Reviewed 2026-09-19. The user's product title matches Hiwonder's **IM10A**,
SKU **21090105**. This is a probable identification, not confirmation of the
physical board, revision, firmware, or configuration. Confirm its label before
selecting a driver or protocol. The advertised ten axes are three gyroscope,
three accelerometer, three magnetometer, and one barometer channel; they do
not establish drift-free position. The product describes positioning combined
with GNSS, not standalone indoor localization.
[Manufacturer product page](https://www.hiwonder.com/products/imu-module).

## Verified manufacturer constraints

- USB Type-C is a serial connection. Default baud is 9600; supported settings
  span 4800–921600. Host and device must agree.
- Factory output is **10 Hz**, configurable over 0.2–200 Hz. The manual limits
  200 Hz to three selected outputs and warns that excessive output content or
  insufficient baud lowers the delivered rate. Measure actual message cadence.
- Nine-axis attitude uses an ENU reference; board Y points north at zero yaw.
  The vendor's car visualization uses board Y forward. That does not establish
  the robot mounting transform.
- Calibration: place the module flat for the vendor accelerometer/reference
  procedure; follow its spherical magnetic calibration around all three axes
  away from magnetic interference. Gyroscope auto-calibration is enabled by
  default. Record the resulting settings, not assumed factory values.

These are vendor procedures and limits, not changes performed in this repo.
[IMU user manual, sections 1.1, 1.3, 1.6 and 1.7](https://docs.hiwonder.com/projects/IMU-Module/en/latest/docs/1.User_Manual.html).

The vendor ROS1 example publishes `/imu/data` and `/wit/mag`; its ROS2 example
shows `/imu/data_raw` and was documented for Foxy. This is not evidence that
every driver version publishes a fused orientation or supports every ROS2
distribution. The linked program archive was not audited; its exact quaternion
ordering, unit conversion, timestamps, and `frame_id` remain acceptance checks.
Do not copy its privileged permission/USB-binding commands onto the HPC.
[Manufacturer ROS tutorial](https://docs.hiwonder.com/projects/IMU-Module/en/latest/docs/3.IMU_Application_Instructions-ROS_Application.html).

## Repo boundary: units and frames

`src/bhl_robust/sensor_io.py::imu_features` is a NumPy adapter, not a ROS node or
device driver. The caller must provide the following consistent inputs:

| Input | Required interpretation |
|---|---|
| `orientation_xyzw` | Quaternion mapping the **robot body frame into ENU world**; not a vendor Euler-angle tuple |
| `angular_velocity_rad_s` | Three body-frame angular velocities, radians/second |
| `specific_force_m_s2` | Three body-frame specific-force components, meters/second squared; not gravity-subtracted acceleration |
| `stamp_s`, `now_s` | Original measurement time and consumption time in one clock domain |
| `orientation_available` | False when orientation is missing, including ROS orientation covariance element zero equal to `-1` |

ROS defines acceleration in m/s² and angular velocity in rad/s. A covariance
array of zeros means unknown covariance, not a perfect measurement; a first
element of `-1` marks an unavailable estimate.
[Official sensor_msgs/Imu definition](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/Imu.msg).

REP-145 describes sensor-frame measurements and specific force: a stationary
sensor with Z upward reports approximately **+g** on Z. Its message `frame_id`
names the sensor frame, not automatically `base_link`. An attitude estimate
relates that sensor frame to its world reference.
[ROS REP-145](https://github.com/ros-infrastructure/rep/blob/master/rep-0145.rst).
The robot body convention is X forward, Y left, Z up; camera optical frames
instead use Z forward, X right, Y down.
[ROS REP-103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst).

Therefore apply a **measured** sensor-to-body rotation consistently to gyro,
specific force, and orientation before calling this adapter. Do not assume an
identity mount or convert an already-SI ROS message a second time. Preserve
sensor-frame data separately for auditing. Rotation alone does not compensate
for acceleration differences between an offset sensor and the body origin.

The adapter emits `[body_gravity_unit(3), gyro(3), specific_force(3), valid]`.
This ten-float feature vector is unrelated to the hardware's ten-axis marketing
count. Magnetometer and barometer are not currently consumed. Invalid/stale
samples emit zeros with `valid=0`, not a valid stationary measurement. The
150 ms default freshness threshold is an application setting, not measured
hardware latency.

## Acceptance checklist before a physical-policy claim

1. Record board/firmware identity, serial settings, enabled output fields,
   algorithm mode, mounting transform and driver version/hash. Inspect the
   driver before running it; some drivers issue configuration writes on startup.
2. Check the existing publisher read-only using `ros2 topic list -t`,
   `ros2 topic echo /imu/data_raw --once`, and `ros2 topic hz /imu/data_raw`
   (substitute the actual topic). Verify `frame_id`, timestamps, finite values,
   quaternion availability and actual publication rate. Topic names alone are
   not a validity guarantee.
3. Record a stationary sample and controlled rotations about each labeled
   axis. Verify gravity sign, gyro signs, and orientation direction. Estimate
   bias, noise, dropout and delay from recordings; preserve raw data and settings.
   Keep robot actuation disabled during this sensor-only check.
4. Align IMU and camera/lidar clocks, replay recorded data through validity
   checks, and reject stale/future-dated samples. Receiving a packet later must
   not replace its measurement timestamp. See [stereo input](STEREO_INPUT.md)
   for image timestamps, calibration and SSD depth provenance.

The current maze training still uses **simulated** body angular velocity and
projected gravity. Its configured gyro noise of 0.05 rad/s and gravity-vector
noise of 0.02 are stress-test assumptions, not IM10A calibration. No physical
IMU subscription, complete multi-sensor state estimator, or hardware policy
deployment has been demonstrated here.
