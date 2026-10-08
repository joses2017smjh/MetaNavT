# Waiter program: grippers, a whole-body controller and a VLA

Started 2026-10-05 on the user's request ("start retraining with the grippers in mind ... picking up a tumbler, they
have big handles, and putting it [on a] tray, carrying [like a] waiter ... do all 4 phases").

**The task.** The 24-DoF Berkeley Humanoid Lite (the hardware's two 1-DoF grippers restored, `docs/GRIPPER.md`)
stands at a table, picks a tumbler up by its large handle with one gripper, sets it on a tray, grips the tray's
two handles, lifts it, carries it like a waiter to a second table (walking and turning), and sets it down. Every
limb works: legs walk, turn and brace; arms reach, lift and hold level; grippers hook the handles; the torso
balances the load. A language instruction picks which tumbler and which table, so the vision-language-action
model (phase 3) has something to ground.

**Why handles.** The finger closes against the palm, so a handle is held by geometry (form closure), not by
squeeze friction. Squeezing a cube between welded hands rolled it in every cooperative-lift run (Stand3/4/5).

## Phases

| Phase | What | Gate (frozen before that phase's scored run) |
|---|---|---|
| 1 | **Whole-body controller (WBC)**: an RL leg policy that walks to a velocity command while the arms and grippers follow commanded targets, with payloads and forces on the hands | below, frozen 2026-10-05 |
| 2 | **Expert demonstrations**: a scripted expert with oracle object poses (arm IK, phase 1's legs) performs the task in MuJoCo; record head-camera RGB, proprioception, instruction and actions | expert success on held-out layouts, frozen before the scored run |
| 3 | **Supervised fine-tune** of SmolVLA: image + instruction + proprioception -> chunks of (base velocity, arm targets, gripper targets), executed by phase 1's WBC | success on held-out layouts and held-out instructions, frozen before evaluation |
| 4 | **RL fine-tune** of the VLA: DSRL first (a small SAC agent steers the frozen flow head's initial noise), full flow-policy RL (piRL-style PPO) only if DSRL plateaus | improvement over phase 3 on never-used seeds, frozen before evaluation |

Labels throughout: LEARNED (phase 1 legs, phase 3/4 VLA), SCRIPTED (phase 2 expert), ORACLE (expert's object
poses; scoring), MODIFIED ASSET (grippers, palm colliders, arm actuator limits).

## Phase 1, frozen 2026-10-05 (before any WBC code ran)

**Asset (MODIFIED, declared).** The 24-DoF gripper URDF (`scripts/add_gripper.py`) plus a collision shape on each
palm: the hand link's own visual mesh as a convex hull, so a handle can be held between finger and palm (the
gripper asset had a finger collider only). Arm actuators: effort limit **8 Nm** (was the upstream training cap of
4 Nm; the motors peak at about 25-32 Nm and the upstream MJCF allows 20 Nm) and stiffness **20** (was 10; damping
2 unchanged), so a 0.5 kg load at the hand sags about 0.1 rad instead of 0.3. Legs unchanged (6 Nm, 20 / 2).
Grippers: effort **2.0 Nm** (the August assumption; the real servo is not documented, so this is replaced by a
measured value when the user provides one), stiffness 20, damping 1, 0 rad open to 1.2 rad closed.

**Control split.** The policy acts on the 12 leg joints only (upstream scale 0.25 around the default pose). The 10
arm joints and 2 grippers are PD-driven to the **upper-body command** (12 targets), which a teleoperator, the
phase 2 expert or the phase 3 VLA supplies at deploy time. This is the decoupled loco-manipulation split (the
lower body learns to stay up and walk under whatever the upper body does). The MARL limb split (`docs/FINDINGS.md`
12) is not used: skrl never matched rsl-rl's PPO (`21330392`, `21338294`).

**Training distribution of the upper-body command.** A new goal every 1.0-3.0 s, reached by linear interpolation
over 0.4-1.2 s. Goal kinds: the default pose (25%); each arm joint uniform in the middle 70% of its range (50%);
the same with the right arm mirroring the left (25%, symmetric carries). Each gripper goal is open or closed with
probability 0.5.

**Hand loads.** At startup each hand link gets an added mass uniform in [0, 0.8] kg (independently per hand and
env). Every 1-3 s each hand gets an external force with each component uniform in [-6, 6] N (zero force with
probability 0.3).

**Everything else is R1 (clock-s2's recipe, `Velocity-BHL-Arms-TurnGaitClock-v0`)**: its velocity command mix,
gait clock (period 0.8 s), `feet_gait` and swing-height rewards, fixed +/-0.5 m/s body pushes every 5-9 s, PPO
settings. Changes needed by the split only: torque, acceleration and joint-limit penalties over the leg joints;
the arm joint-deviation penalties removed (the policy does not move the arms). Observations (actor 83): velocity
command 3, base angular velocity 3, projected gravity 3, joint positions 24 and velocities 24 (relative to
default), last leg actions 12, the current upper-body command 12 (arm targets relative to default, grippers 0-1),
gait clock 2. Critic: the same plus base linear velocity (86).

**Runs.** Task `Velocity-BHL-Waiter-WBC-v0`, from scratch, seeds 0, 1, 2, 6000 iterations, the task's env count.

**Gates (MuJoCo, the 24-DoF model with palm colliders; arms and grippers PD-driven as in training).**
- **Q1 locomotion**: `turn_test.py --protocol v2` PASS and `cpu_turn_qualify` QUALIFIED (turn 9/10, walk drift
  2/3, push <= 9/60), protocols unchanged, arms commanded to the default pose, grippers open.
- **Q2 upper body under load**: 12 episodes = 4 arm trajectories x reset seeds 0-2, with 0.5 kg added to each
  hand: 3 s standing while the trajectory runs, then 5 s walking at 0.25 m/s while it continues. Trajectories
  (generated by `scripts/waiter/q2_trajectories.py`, committed before any WBC checkpoint exists): T1 both arms to a
  tray-carry pose (forearms forward and level) and hold; T2 right arm reaches forward-down, closes the gripper,
  returns, opens; T3 both arms raise and lower three times; T4 a smooth random motion from the training
  distribution with a fixed seed. **PASS iff 0 falls in 12 and heading drift <= 15 deg on >= 10 of the 12 walking
  segments.**
- **Selection**: the seed passing Q1 and Q2 with the lowest push-fall rate (tie: lowest seed). None passes ->
  phase 1 NEGATIVE, and phase 2 falls back to the frozen gait clock-s2 with scripted arm targets (labelled so).
- Reported, not gated: push falls with 0.5 kg per hand in the carry pose; hand-target tracking error.
