# Waiter program interfaces (phases 2-4)

Frozen 2026-10-05 before any phase 2-4 code exists, so the scene/expert, the VLA pipeline and the RL fine-tune are
built against one contract. Changes are additive only (new optional fields), recorded in `SLURM_JOBS.md`.
Program and phase 1: `docs/WAITER_PROGRAM.md`.

## Robot and low-level controller (phase 1, exists)

- 24-DoF Waiter model: MuJoCo `prepare_mjcf(upstream, cache, "waiter")` / `build_multi(..., variant="waiter")`;
  joint order (deploy order) = arms 0-9 (left: shoulder pitch, roll, yaw, elbow pitch, roll; then right), legs 10-21,
  grippers 22 (left), 23 (right). Grippers: 0 rad open, 1.20 rad closed, 2 Nm.
- Low level = the phase 1 WBC, `gait_clock.make_controller(stamped deploy.yaml)` -> `WaiterWbcController`, run at
  25 Hz (`policy_dt` 0.04 s): `ctrl.set_upper_body(q12)` (absolute rad: left arm 5, right arm 5, grippers 2), the
  base velocity command goes in `runner.observe(i, (vx, vy, wz))`, `runner.step([ctrl.update(obs)])`.
  Until a WBC seed is selected (`results/waiter-20261005/wbc/selection.json`), develop with a FIXED BASE (the robot's
  base welded to the world) so arm and grasp mechanics do not wait on locomotion.
- Kinematics (MuJoCo FK, arms hanging at 0; base origin at ground level when standing): shoulders ~0.76 m high,
  hands ~0.49 m, fingertips ~0.38 m; LEFT shoulder pitch < 0 swings the arm forward, LEFT elbow pitch > 0 brings the
  forearm forward; the RIGHT arm mirrors the left by negation (every joint-limit pair is negated). Finger: a
  0.02 x 0.05 x 0.07 m box hinged at the hand tip (hand frame z = -0.11 m), closing toward the palm; the palm has a
  convex-hull collider (the hand mesh).

## High-level action, state and observation (phases 2-4)

- **High-level step: 0.08 s** (12.5 Hz) = 2 WBC steps.
- **Action** `a` (15, float32): `[vx, vy, wz]` body-frame base velocity command (m/s, m/s, rad/s; clipped to
  |vx| <= 0.4, |vy| <= 0.3, |wz| <= 0.6), then the 10 arm targets (absolute rad, joint-limit clipped), then the 2
  gripper closures in [0, 1] (x 1.20 rad).
- **State** `s` (30, float32): the 24 joint positions (absolute rad, deploy order), base projected gravity (3),
  base angular velocity (3).
- **Image**: `head_rgb`, 224 x 224 RGB uint8, from a SIMULATED head camera on the torso top looking forward-down
  (labelled as simulated: the hardware has no head camera).
- **Instruction**: one string from templates, e.g. "pick up the red tumbler, put it on the tray and bring the tray
  to the left table". Template and colour/side sets are declared by phase 2; phase 3 holds out some templates.

## Environment API (built by phase 2, consumed by 3 and 4)

`bhl_robust.eval.waiter_env.WaiterEnv(deploy, upstream, cache_dir, render=True, fixed_base=False)`:
- `reset(seed, instruction_id=None) -> obs` with `obs = {"image": (224,224,3) uint8, "state": (30,) f32,
  "instruction": str}`; the layout (object poses, colours, target table) is a pure function of `seed`.
- `step(a) -> (obs, reward, done, info)`: advances 0.08 s; `reward` = 1.0 on the step success is reached, else 0.0;
  `done` on success, robot fall (tilt >= 0.78 rad), or the 60 s time limit; `info` = {"success", "fell", "stage",
  "tumbler_on_tray", "tray_on_target", "t"}.
- `oracle() -> dict` of object and robot poses (the scripted expert only; never an input to a VLA).
- Success (frozen definition, the gate of every phase): the tumbler upright (tilt < 20 deg) resting on the tray,
  the tray resting on the target table inside its target zone, both grippers open, the robot upright; held 1.0 s.

## Data format (phase 2 writes, phase 3 reads)

One `.npz` per episode: `image` (T,224,224,3) uint8, `state` (T,30) f32, `action` (T,15) f32 (the action applied
after observing step t), `instruction` str, `seed` int, `success` bool, `stage` (T,) int8. Plus `manifest.json`
listing episodes with their seeds, success and sha256.

## Seed ranges (frozen; nothing outside its range, ever)

| range | use |
|---|---|
| 1000-1999 | development / exploration (any phase) |
| 2000-2099 | phase 2 scored expert gate |
| 3000-3999 | phase 2 demonstrations (phase 3 training data) |
| 4000-4099 | phase 3 scored evaluation (held-out layouts; plus held-out instruction templates) |
| 5000-5999 | phase 4 RL training layouts |
| 6000-6099 | phase 4 scored evaluation (never used before it) |
