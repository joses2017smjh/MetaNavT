# Three tasks that can actually be completed

The original lift tasks had no completion criterion. A policy could earn every
reward term and the episode would still end on a timeout, so "success" was never
a state the environment recognised — only a number that went up. Worse, the
payload sat on the floor, inside the reach of a robot that had already fallen
over, and nothing in the reward or the terminations referred to base height.
Both cube arms learned to drop ~41 cm in the first 0.2 s and hold a brace.

These three replace it. Each has a terminal success state, and each puts its
payload where reaching it requires standing up.

## The one number the redesign turns on

Measured by forward kinematics over the shoulder and elbow ranges, feet planted
(`slurm/inner/_reach.sh`, constants in `tasks/reach_band.py`):

| condition | hands can reach |
|---|---|
| standing, arms only | 0.404 – 0.610 m |
| collapsed 41 cm, arms only | −0.006 – **0.200 m** |

`GRASP_Z = 0.30 m` for all three payloads:

* **10 cm below** the standing arms-only floor, so the knees have to bend — the
  arms cannot get there on their own.
* **10 cm above** the collapsed ceiling, so falling over *loses* the payload
  rather than winning it.

That is a structural fix rather than a tuned one. It does not require a height
penalty large enough to outweigh a 15.0-weight lift bonus; it makes the collapse
stop paying at all.

---

## 1 · Cube to a shelf that barely fits it

| | |
|---|---|
| payload | 0.28 m cube, on a 0.16 m plinth, centre at 0.30 m |
| target | shelf slot **0.34 × 0.34 m** — 6 cm of total clearance on a 28 cm cube |
| deck height | 0.38 m, so the carried cube centre ends at 0.52 m (inside the 0.61 m standing ceiling) |
| carry | shelf 1.2 m from the spawn, so it has to be moved, not just raised |
| success | cube centre inside the slot volume, speed < 0.05 m/s, held 0.5 s |

The tight slot is the point: a 6 cm margin cannot be hit by throwing the cube at
the shelf, so the terminal condition rewards placement rather than displacement.

## 2 · Ball carried to a spot and thrown into a net

| | |
|---|---|
| payload | r = 0.18 m ball, on a 0.12 m tee, centre at 0.30 m |
| release zone | 0.6 m radius marker, 1.5 m from the spawn |
| net | 0.7 × 0.7 m mouth, rim at 0.60 m, 2.0 m beyond the release zone |
| success | ball centre inside the net volume with downward velocity |

Staged, because the throw is worthless before the carry works: reach → lift →
both robots inside the release zone → release with horizontal velocity toward
the net → net entry. This is the only one of the three with a ballistic phase,
and the only one where the pair must agree on a moment as well as a position.

## 3 · Plank leaned against a wall

| | |
|---|---|
| payload | 1.5 × 0.4 × 0.08 m plank, on two supports at 0.26 m, centre 0.30 m |
| target | wall 1.0 m away |
| success | lower end within 0.30 m of the wall base, upper end touching the wall above 0.5 m, plank 50–80° from horizontal, stable 0.5 s |

The plank is 1.5 m and one robot's hands span 0.355 m, so neither robot can
bracket it alone — this is the one task where the two-robot requirement is
geometric rather than a matter of load.

---

## Vision: three conditions per task, all nine on one stack

Every task runs blind, with depth, and with RGB. Nine cells, **all of them on
`BHL_STACK=v60`.**

| condition | sensor |
|---|---|
| blind | privileged object pose |
| depth | `TiledCameraCfg`, `distance_to_image_plane`, RTX |
| RGB | `TiledCameraCfg`, `rgb`, RTX |

Rendering is solved and it is Isaac Sim doing it. The 6.0 probe is unambiguous —
rgb 256×256, mean 222.2, **std 24.4 across 1,902 unique colours**, with depth
correct to the unit in the same frame. Isaac Sim 5.1's RTX renderer segfaults on
this cluster inside `omni.usd.create_hydra_engine`; 6.0 does not.

What had never run was RGB *training*, and not for a rendering reason. Job
`21036909` died after four minutes on both arms with `ModuleNotFoundError: No
module named 'rsl_rl'` — the v60 venv was built with Isaac Sim and Isaac Lab and
no RL library. Those two failures were filed against RGB; they were a missing
package. `rsl-rl-lib==3.0.1` is installed there now and imports.

**Running all nine cells on v60 is what makes the vision comparison mean
anything.** An earlier plan put blind and depth on v51 and only RGB on v60,
which would have made the headline comparison — does colour beat depth beat
nothing — a comparison across two Isaac Sim major versions. Same stack for all
three arms, and the question is about the sensor instead of the simulator. The
v51 numbers stay where they are, as the record of the old tasks; nothing needs
quarantining because nothing is mixed.

## Gates, before any of it trains

The lesson from G-B2 is that a gate without a control measures its own budget.
Each task gets a *kinematic* gate first, which needs no training at all:

* **G-T1** — can a standing robot reach `GRASP_Z` with a squat rather than a
  collapse? **PASS**: 96 of 625 swept postures put both hands within 3 cm of
  0.30 m, the best of them descending **15.5 cm** from standing. That is knee
  flexion, and it is well short of the 41 cm the current policies drop.
* **G-T3** — is the collapsed posture excluded? **PASS**: of the postures
  reachable from a base 41 cm down, **zero** get to 0.30 m.
* **G-T2** — is the success volume enterable? Not yet run; needs the scene
  furniture the three tasks add.

Two of the three passing means the height is right before anything trains on
it. Both took seconds, because they are forward kinematics and not policies.

The gate was wrong twice before it was right, in ways worth recording. It first
bent the knees with the root pinned in place, which lifts the feet off the floor
instead of squatting, and so reported 0.30 m unreachable with a closest approach
of 0.407 m — the arms-only figure, which is what you get when the legs are not
really moving. It also compared the base body origin against an absolute
threshold, in a frame whose origin sits 0.137 m *below* the feet, so no posture
could ever have passed. Feet are re-planted per posture now, and the criterion
is descent from standing.

No tier trains until its gate passes.

---

# B4 · Limb agents

## The two design calls, and why they went the way they did

**Partition: four agents, one per limb.** The robot's 22 DoF split cleanly as
left arm 5, right arm 5, left leg 6, right leg 6 — `ARM_JOINTS` and
`LEG_JOINTS` are already ordered that way, so the partition is a slice rather
than a remapping. Four is also the literal reading of "an agent for each limb",
and it is the interesting version of the question: a 2-way upper/lower split
mostly reproduces the existing controller with a seam through it.

A 2-agent arms/legs split runs as the ablation, because this project already
knows the lift lives entirely in the arms while the legs do the standing, and
that is exactly the credit-assignment boundary a 2-way split would test.

**Algorithm: MAPPO, with IPPO as the ablation.** The single-agent baseline here
is already an *asymmetric* actor-critic — the critic sees object twist and both
robots, the actor sees proprioception. MAPPO is the direct generalisation of
that: centralised critic, decentralised actors. Choosing IPPO instead would
change two things at once, the agent count and the critic's information, and no
row in the table would isolate either. IPPO then answers the narrower question
of what the centralised critic is worth.

**Ablation the work order requires:** `joint_deviation_arms` must be off in any
22-DoF limb-agent run. It penalises arm deviation from the default pose, which
is the same term §5 found fighting a squat-and-pinch — with a separate agent
owning each arm, it would be a per-agent penalty for doing the task.

## G-B4, before any of the 24 rows

* Does a `DirectMARLEnv` with the four-limb partition construct, reset and step,
  and does each agent receive its own action slice of the right width?
* Do the four action slices reassemble into exactly the 22-DoF vector the
  single-agent controller applies? If they do not, MAPPO is not being compared
  against PPO — it is being compared against a different robot.

---

# The handled tote: making the lift physically possible

Three sections of this project have now measured a pair of robots forming a good
pinch and never lifting. The redesign above fixes *where* the payload sits. This
fixes *what it is*, and it comes from finally sizing the task against the
actuators instead of against intuition.

## What the arms can actually do

Static holding torque at the limiting arm joint — shoulder roll, moment arm
0.257 m in the pinch pose, 4 Nm limit:

| | mass, hanging |
|---|---|
| one arm | 1.59 kg |
| one robot, two arms | 3.17 kg |
| **a pair, four arms** | **6.35 kg** |

The cube is **0.5 kg**. It is twelve times inside budget and has never been
lifted once. **Mass was never the constraint**, which means every intervention
aimed at making the lift easier by making the object lighter or the reward
larger was aimed at the wrong thing.

## What the constraint actually is

With no fingers, a side pinch holds by friction, and friction needs inward
normal force. That force's reaction pushes each robot *outward*, away from its
own base of support. The harder a policy grips, the more it destabilises itself.
That is the trade every arm in §5 was losing, and it explains the one behaviour
that kept appearing: a braced collapse onto the floor is a stable way to
generate squeeze, because the floor absorbs the reaction the legs cannot.

**A handle deletes the trade.** The load hangs off the limb, gravity does the
retaining, and the torque budget goes to holding instead of to fighting the
robot's own grip reaction.

## The design

| | value | why |
|---|---|---|
| mass | **4.0 kg** | 1.26× one robot's ceiling, 0.63× a pair's |
| handle bar | r = 15 mm, 200 mm long | inside one robot's 355 mm hand span |
| standoff | **75 mm** | clearance under the bar for a ~50 mm hand to hook |
| body | 0.30 × 0.26 × 0.24 m | narrow in y so a robot's hands straddle it |
| grasp height | 0.30 m | the measured squat band, unchanged |

**Mass is what enforces cooperation now, and that is the part worth keeping.**
One robot at 3.17 kg physically cannot lift 4 kg however good its policy; a pair
has a 1.6× margin. Compare the ball, whose cooperation was supposed to be
enforced by geometry and was not — a 0.36 m ball fits inside one robot's 0.355 m
hand span, so nothing ever stopped a single robot solving it. A physical ceiling
is a guarantee; a span argument was an assumption, and it was wrong.

## If this also fails

Then the conclusion is a hardware one and should be written as such: 4 Nm
shoulders with no fingers cannot do non-prehensile cooperative manipulation, and
the fix is a gripper or a stronger shoulder rather than more iterations. That is
a legitimate result — it is the same shape as §4's "the plateau is torque, not
sensing" — but it should be reached by having removed every solvable obstacle
first, which is what this tote is for.
