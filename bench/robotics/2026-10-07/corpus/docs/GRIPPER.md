# The robot has grippers. The simulation asset does not model them.

## What the hardware has

Upstream's own driver,
`berkeley_humanoid_lite_lowlevel/robot/bimanual.py`, commands **two grippers,
one per hand**, over serial:

```python
# 0.2: open
# 0.85: closed
# 0.9: tightly closed
gripper_left_raw_value  = 0.2 + self.gripper_left_target * 0.6
gripper_right_raw_value = 0.2 + self.gripper_right_target * 0.6
data = struct.pack("<ffb", gripper_left_raw_value, gripper_right_raw_value, 0x0C)
```

Targets are normalised to **[0, 1]** and mapped to a raw **[0.2, 0.8]**, with
0.2 open and 0.85 closed. `scripts/teleop/run_teleop.py` places them at
`robot_actions[10]` and `[11]`, so the real bimanual action vector is **ten arm
joints plus two grippers = twelve**.

## What the simulation has

Ten. Both hand joints in the URDF are welded:

```xml
<joint name="arm_left_hand_l" type="fixed"> ... </joint>
<joint name="arm_hand_r"      type="fixed"> ... </joint>
```

`arm_*_hand_link` is a rigid 74 × 69 × 136 mm block bolted to the elbow roll.
There is no gripper degree of freedom anywhere in the asset, so `JOINTS` is 22
and not 24, and **no policy in this project has ever been able to close a hand.**

## Why that matters more than it sounds

Every manipulation result here was produced by a robot that physically cannot
grasp. The cooperative lift was non-prehensile not because non-prehensile
manipulation was the research question, but because the asset offered no
alternative and nobody checked. Section 5's whole arc — a pinch that forms and
never lifts, nine interventions that move nothing, a policy that discovers a
braced collapse because squeezing destabilises it — is the behaviour of a
machine trying to hold things between two fixed blocks.

The grasp the hardware actually performs is different in kind: lay the open hand
over the object, close, and the closed fingers form a **form-closure manifold**
that retains it. Retention comes from geometry, not from friction, so it does
not need the inward normal force whose reaction was destabilising the stance.
That is the same fix the handled tote was reaching for, except the robot already
has it.

**I asserted "no fingers" as a fact about the robot several times, including in
conclusions about hardware limits.** It was true of the asset and false of the
machine, and the difference inverts the conclusion: the manipulation ceiling
measured so far is a property of the model, not of the robot.

## What has to change

1. **Asset** — add one revolute DoF per hand to the URDF and MJCF, plus finger
   geometry that can close on an object. Convention follows the driver: action
   in [0, 1], 0 open and 1 closed.
2. **Joint list** — `JOINTS` becomes 24, actions 24 per robot, and every
   observation width that quotes 22 or 44 moves with it.
3. **Re-run** — every manipulation result becomes a measurement of the old
   asset. The terrain and locomotion rungs are untouched, because they never
   used the arms for contact.

That last point is the expensive one and it should be stated plainly rather than
absorbed quietly: §5, the three v2 tasks, the tote and the cloth design were all
specified against a robot with welded hands.
