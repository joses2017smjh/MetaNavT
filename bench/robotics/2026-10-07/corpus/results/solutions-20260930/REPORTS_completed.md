# Completed investigation reports (2026-09-30), verbatim summaries for the follow-up workflow

All scratch paths in these reports that start with
`/scratch/sanchej7/tmp/claude-19646/-nfs-hpc-share-sanchej7-Computer-Vision/cc4dca4e-228f-469e-9df3-2f9b1f7d605a/scratchpad/solutions/<topic>/`
now live at `/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/<topic>/` (the /scratch copy was node-local).
Repo: `/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder` ($R).

---------------------------------------------------------------------------------------------------
## A. Mission 7 replay gate stuck at 9/10 (investigator report)

What the gate is: `$R/scripts/mission7_plate_stage.py` (`run()`, `_episode()`). 10 episodes = the 10 plate-caused falls of
the legacy measured route controller on validation Doors layouts 0,1,4,5,7,9,12,13,14,15
(`$R/results/mission7-replay-smoke-20260921/fullroute/legacy-doors.json`). Same reset stream, pinned to cn-c22 for bitwise
reproducibility; each 40 ms tick replays the recorded body-frame command unless `PlateStage.command()` takes over within
0.78 m of an unopened correct plate (or the wrong plate when the correct one is > 1.0 m away); doors open at the
recorded activation times; each episode runs to the original fall time; fall = tilt >= 0.78 rad; PASS iff 10/10 upright.
On 10/10, `$R/slurm/repo20260923/m7_replay_gate_followup.sbatch` releases a 16 Doors + 16 Transport route gate.
Plate: 3 cm high, r 0.24 m, 0.42 m off the corridor centreline; wrong (square) plate mirrored.

Variants: G-ref 21397732 (pre-point 0.30 m, settle 0.40 s, fixed 1.2 s crossing) PASS 10/10; V2 21401689 7/10;
v2-align 21405537 9/10; v3-settle190 21405539 6/10; v4-settle080 21405541 9/10; v3-settle190-align 21405543 7/10;
v4-settle080-align 21405545 7/10.

Failures behind 9/10: v2-align layout 7 and v4-settle080 layout 0 -- after the 'kick' the gait walks ~0.2 m and stops
with one foot on the 3 cm plate edge (along ~ -0.29 m) while still commanded sideways and/or yawing, then topples
over 0.5-0.7 s (falls 1.9-2.4 s after the kick). Pooled over 35 crossings in 5 variants: sideways entries (> 0.8 rad off
the door direction) 21, 7 fell in-crossing, 2 cleared; forward entries 14, 0 fell in-crossing (one-sided 95% UB 0.19)
but 7/14 stalled at the edge and 3 fell after hand-back. Which layout fails flips between variants (p(10/10) per
variant ~0.09-0.30) -> selecting a variant by its 10/10 would be selection, not evidence.
Post-handback falls (5/12) are an artefact of replay: recorded commands are applied 82-90 deg off their intended
direction after an align stage changes heading.

Gate validity: (i) the passing G-ref never crossed a plate (crossings moved 0.02-0.22 m, ended at along -0.14..-0.33;
layouts 5 and 15 stood stalled ~27 s); (ii) coverage: 7 crossings, all door 0, six +y and one +x; none -y/-x/door-1;
layouts 9, 12, 13 end 2.2-3.0 s after takeover; (iii) post-handback replay is not the route controller;
(iv) 'crossings completed: 7' counts 4 s time-outs (real clears 4/7 v2-align, 1/7 v4-settle080);
(v) route gate has two readings: code `episodes >= 16` (`$R/scripts/mission7_plate_safe.py:158`) vs docs 16/16
successes (`$R/docs/MISSION7_TASKS.md:32,48`); (vi) episode budget inconsistent (STATUS/REPO_TASKS 101 remain vs
SLURM_JOBS.md:2737 1 remains + 50 charged on 09-24). Login-node re-run of v2-align layout 7 matched cn-c22 bitwise
(1,112 samples; fall at 18.28 s).

Solutions ranked: (1) hand the plate crossing to TurnBoth-s0 (turn in place to the door direction, cross straight
forward, turn back before hand-back; swap `env.controller.policy` to a CpuPolicy in `PlateStage._start`, revert at
hand-back; plumb `--stage-gait` through mission7_plate_stage.py, mission7_route_handoff_probe.py and the
m7_replay_gate_followup.sbatch flag translation), conditional on a NEW plate bench (train layouts 0-31, both doors,
64 staged crossings at entry headings 0/+90/-90/180 deg; PASS >= 62/64 clear to along +0.35 m within 10 s, 0 falls,
>= 15/16 per heading), then the unchanged exact replay gate once; (2) fallback: turn first then cross straight with the
current gait + stall watchdog; (3) repair the gate (post-handback driven by the route controller, >= 15 s after
hand-back, every takeover must clear the plate, -y/-x bench) -- a stated gate change needing sign-off;
(4) learned local crossing (fine-tune on 3 cm discs) as a last resort.

---------------------------------------------------------------------------------------------------
## B. Cooperative lift / carry / cube placement (investigator report)

Limits: 6 Nm (legs) / 4 Nm (arms) are upstream Isaac training caps copied into deploy.yaml, not hardware ratings
(6512 / 5010 motors, 15:1 cycloidal, 20 A -> ~25/32 Nm peak; paper tested 20 Nm; upstream MJCF +-20 Nm; firmware/
deploy run legs at 4-6 Nm, arms 1-2 Nm). Arm statics fit the 4 Nm cap (worst: scripted squeeze keyframe shoulder roll
3.30 Nm = 83%). Friction hold needs only ~2 N/side; MuJoCo shows 7-9 N per pad. Isaac hands have NO collision
geometry (only upper-arm r 0.04 and forearm r 0.03 cylinders touch the cube); the replay `hand_force` is 0 every step.
MuJoCo pads sit >= 26 deg off the cube face (edge/corner contacts), default contact has no torsional friction -> the
cube pivots ~1.1 rad about the inter-contact line beyond the pads' own 0.66 rad rotation.
THE "CROUCH NEEDS 28-30 Nm" CLAIM IS WRONG: in `$R/results/repo-gpu-20260923/spawn_pose/..._pd_hold__pinch.json`
28-30 Nm appears only at step 0 (foot force 0, reset transient); steps 1-7 with feet loaded: 2.1-5.9 Nm; static
estimate 4.8 Nm at the knee (~80% of 6 Nm).
Gait never trained with a payload (arms-dr1.0-s0: +2 kg torso, +-2 N / +-2 Nm push, arm-deviation penalties); holding
an arm out at the squeeze pose + cube + squeeze reaction = 2-4x the disturbance the gait saw.

Stand3 replay (21470828): s0 "carries over the deck but doesn't set down" is really a rolled cube cocked on the deck
edge (centre 0.613-0.632 m, rolled 15-36 deg, lowest corner 0.422 m between plinth 0.41 and deck 0.43); s1 falls short
(bases move only 0.026 m; 19/96 falls). BOTH FARM THE LIFT REWARD BY ROLLING THE CUBE: 74-75% of steps pay
`lifting_object`; on paying steps the cube is tilted a median 40/43 deg; the lift curriculum stalled at
0.0555/0.0583 m ~= 0.14*(sqrt2-1) = 0.058 m, the rise from rolling a 0.28 m cube onto its edge. In Isaac the cube rolls
about the pinch axis in 79%/64% of episodes.

MuJoCo mechanism check (exploration seeds 100-104 only; overlap tuning seeds -> mechanism, not a result):
lift-and-hold 0/3 published -> 3/3 with torsional pad friction (condim 4, 0.04 m) + elliptic cone impratio 10;
lift-hold-place 0/5 -> 5/5 with both (straight reverse lowering).

Solutions: S1 fix the grasp contact (opt-in CarryParams settings + a physical flush-pad wrist variant) and re-score
lift-hold / lift-place on never-used seeds (20-29 hold, 30-39 place), unchanged clauses + cube tilt <= 0.35 rad
(~15 CPU-min); S2 Stand4 with hand colliders (`assets/cloth/berkeley_humanoid_lite_hand_colliders.usda`) and a
roll-proof lift reward (lowest corner above support, tilt <= 15 deg; seat requires tilt <= 8 deg and release) after S1
(~24 GPU-h); S3 a tray/handle payload held by geometry (forearm tray 0.9-1.35 Nm); S4 retrain the gait with hand
loads for the carry (~15 GPU-h). Outputs: coop/static_limits.json, com_shift.json, stand3_*.json,
mechanism_summary.json, grip_mechanism_*.json, place_mechanism_*.json.

---------------------------------------------------------------------------------------------------
## C. Turning gait works on 1/12 seeds (investigator report)

Root cause: PPO optimizes the NOISY closed loop; the gates (MuJoCo turn_test, Isaac probe) are noise-free.
F1: four non-turners, reset seeds 0-1, +-0.6 rad/s after a 3 s settle: noise-free 0/16 reach 150 deg (10-18 deg,
0 lift-offs); with each checkpoint's own training action noise 16/16 turn 164-256 deg (14-28 lift-offs, no falls);
1/4 of the noise suffices for TurnBoth-s1. F2: two kinds of non-turner (stuck-at-rest-but-able-to-step: TurnBoth-s1,
TurnHip-s0; steps-only-under-noise: TurnCmd-s0, arms-dr1.0-s0). F3: training OBSERVATION noise alone -> 10/11
non-turning arm seeds pass v2 (`turning/v2_obsnoise_x1.0.json`); the Isaac probe ran with enable_corruption=False, so
its "not a sim2sim gap" reading was noise-free too. F4: action std rises to 1.75-1.95 at iterations 2000-3000; legs
0.88-1.37 at 6000; leg torque saturates 25-33% of the time under noise. F5: the turner is invisible in TensorBoard.
F6: every checkpoint that marches at zero command (7/7) passes v2; none of the other 15 does. F7: pure turns are
0.27-1.27% of training commands; feet_air_time pays only in single stance. F8: push fall rate tracks training fall rate
(rho -0.98), leg noise (-0.93), action rate (+0.97) -- one lineage, not independent. F9: deploying WITH noise turns but
fails the push gate (TurnBoth-s0 13/60, TurnBoth-s1 18/60 vs 7/60 and 9/60 noise-free).

Recipes: R1 gait clock (sin/cos of a 0.6 s phase, 75 -> 77 obs) + a contact-schedule reward `feet_gait` paid at EVERY
command (unitree_rl_lab form) + foot clearance, feet_air_time weight 0, fixed +-0.5 m/s pushes, default PPO noise;
needs a clock-aware RlController in turn_test.py, run_eval.py, turn_diagnose.py and its own launcher; R2 the same
reward with the clock only in the critic (zero deploy cost); R3 low-noise TurnRest + pushes (low confidence).
Gate (unchanged): seed PASS = noise-free turn_test v2 PASS; arm PASS >= 2/3; then cpu_turn_qualify per seed
(v2x >= 9/10, walk drift <= 15 deg on >= 2/3, push <= 9/60); usable iff >= 2/3 qualify. ~12 GPU-h per arm.
Outputs: turning/mj_probe*.py, exp1-exp6 logs, v2_obsnoise_x1.0.json, push_*_obs*.csv, batch6.log, summ_*.txt,
tb_scalars.pkl, push_corr.json, cmd_dist.py.

---------------------------------------------------------------------------------------------------
## D. NavGym v4 transfer (coordinator's own analysis)

Gym replay of the two gate-passing final actors on the SAME 12 transfer mazes (maze seeds 50000-50011; 5 randomized-
dynamics draws + 1 nominal-dynamics heading-0 run per maze; `navgym/gym_replay.json`, script `navgym/gym_replay.py`):
- armV4-s6: physics failures 50001, 50002, 50009 are the SAME mazes it fails in the gym (50001 0/5 + nominal t/o;
  50002 2/5 + nominal t/o; 50009 0/5 + nominal t/o) -> ~no transfer gap for s6; its misses are the policy's own.
- armV4-s5: gym fails only 50009 (nominal) yet physics failed 50000, 50002, 50009, 50010 -> a real gap of ~3 mazes.
- Maze 50009 (shortest route, detour 1.23) defeats both in every condition. Plot `navgym/traj_50009.png`: both drive
  greedily along the bottom corridor into the dead-end pocket nearest the goal; s6 shuttles start<->dead end (no memory
  of the visited dead end; its 3x24x24 egocentric crop never shows the route up).
- The actors turn bang-bang: |wz| saturated 94-99% of steps in physics, flip sign ~15-18 times/s in the gym; A*
  never saturates. Physics failures are loops (revisit fraction 0.48-0.83) or slow dithering, 0 falls, 0 wall contacts.
- Gym "collisions" become physics "time-outs" (the speed brake stops the robot at the wall).
Implication: the dominant failure is reactive local navigation without global route memory; the gym-dynamics fix
helps s5 only. Candidate fixes: (N1) hybrid -- A* on the lidar map supplies a sub-goal 1.5-3 m ahead to the learned
local policy (no retraining needed to test; the learned actor is faster than A* turn-then-walk where it succeeds);
(N2) add a 'visited' channel (breadcrumb layer) to the ego map and/or a recurrent policy; (N3) action smoothness
(yaw-rate penalty / bounded squashed policy) + the identified physics dynamics for the s5-type gap.
The ledger line "the gym-to-physics gap costs 3-4 goals in 12" is wrong for s6 and must be corrected.
