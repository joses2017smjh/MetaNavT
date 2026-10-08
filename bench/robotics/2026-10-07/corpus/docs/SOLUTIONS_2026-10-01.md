# Solutions plan for the five open problems (2026-10-01)

Inputs: the investigation reports A–D (`solutions-20260930/REPORTS_completed.md`), the biped sysid and IM10A-kit
investigations, and an independent read-only verification of A–D (`solutions-20260930/verify-{coop,turning,mission7,navgym}/`).
`solutions-20260930/` is archived in the repo at [`results/solutions-20260930/`](../results/solutions-20260930/), small files
only; the original is `/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/`.
No job was submitted and no scored seed set was run for this plan. Every step below gets its rule written into the
launcher header before it runs. No gate is weakened, and labels say learned / scripted / oracle.

Evidence tags: **[V]** confirmed by the independent re-check; **[V~]** partly confirmed, stated here in its corrected
scope; **[I]** investigator report, not independently verified; **[U]** unverified.

## Summary

| Problem | Root cause (confidence) | Recommended next step | Cost |
|---|---|---|---|
| 1. NavGym v4 physics transfer: 8/12 and 9/12 (need ≥ 10/12 each) | s6 and maze 50009: reactive driving with no route or visitation memory; s6's three physics misses are its gym misses (high). s5's extra 2–3 goals: a deployment-pipeline loss. The gym reproduces it with the full pipeline (identified dynamics, brake, 10 Hz packets integrated one step late), and capture-pose integration removes it (medium in the gym, low in physics, [I]) | **N0 done, NEGATIVE** (`21501509`: s5 15 → 17/24, net +2, needed +4; s6 23 → 23; 0 falls). Next: N1 (A* sub-goal hybrid, labelled learned + scripted) or N2 (retrain with a visitation channel) | N1 ≈ 1–2 CPU-h; N2 3 seeds × 30 M steps |
| 2. Turning recipe (1/12 seeds) | PPO learned to step only in the noisy training loop: the four non-turners probed turn with their own training action noise (16/16 runs), while the gates and the robot run the noise-free mean; nothing pays stepping at zero command (medium-high in MuJoCo; Isaac with noise [U]) | One launcher, two arms × 3 seeds: R1 (gait clock in the actor) and R2 (clock in the critic only), with a contact-schedule reward at every command and pushes from the start; gate unchanged | harness change + ≈ 24 GPU-h |
| 3. Mission 7 replay gate (best 9/10 among the cross-clear variants, the ones that keep crossing until the plate is cleared; four fixed-1.2 s-crossing variants reach 10/10) | Falls: a standstill-plus-kick entry onto the 3 cm round-plate edge, sideways, because the shipped gait cannot turn in place (medium; confounded by layout). Gate: a replay 10/10 cannot certify a crossing (high) | **M1 done in code** (route gate counts successes; real crossing clears). Next: the user reconciles the episode budget, then a bench budget line (M2: 106 episodes) | M1 0 episodes; M2 106 episodes |
| 4. Cooperative lift and placement | Learned: `object_is_lifted` pays centre height only, so Stand3 earns it by rolling a supported cube (high). Scripted: edge/corner pad contacts let the cube roll ≈ 90° in the hands (medium) | Flush-pad wrist mechanism probe in MuJoCo on exploration seeds, lowering target frozen first | ≈ 15 CPU-min + a geometry edit |
| 5. IM10A recording (SF-05) | Not a failure: needs the physical device; the analysis kit is committed with this plan | User records per `docs/IMU_RECORDING.md` (≥ 2 h stationary at 200 Hz, rotations, tap test) | bench time; ≈ 0.7 GB per 2 h bag into an over-quota project |

## 1. Learned navigation transfer (NavGym v4 → physics biped)

**What the logs show**
- `21484212`: NEGATIVE by its predeclared rule. armV4-s5 scored 8/12 and armV4-s6 9/12 on never-used maze seeds 50000–50011, with 0 falls and 0 wall contacts; every miss was a time-out (one s5 miss classed stuck). A* scored 12/12, median 67.0 s (`results/navgym-v4-transfer-20260930/verdict.json`).
- Gym replay on the same 12 layouts (layout hashes equal to the physics JSONs [V], `verify-navgym/layout_check.json`; `navgym/gym_replay.json`):
  - s6 fails exactly its physics misses (50001, 50002, 50009) at nominal dynamics, so its transfer gap is 0 [V].
  - s5 fails only 50009 at nominal dynamics, against 4 misses in physics: a gap of 3 against nominal, ≈ 2 against its randomized draws (2.0 expected, 4 observed) [V~].
  - Only the nominal comparison is time-matched (180 s), and it is n = 1 per maze. The randomized draws ran route-scaled limits of 121–180 s (121 s on 50009) against physics' 180 s [V]. A shorter limit can only add time-outs, so the ≈ 2 is not time-matched and, if anything, understates the gap.
  - Two of s5's extra misses were already marginal in the gym (50000 took 111 s and 50002 153 s of 180 s). 50010 is the clean gap case: 45 s in the gym, a physics time-out with the brake on for 3468 of 4500 steps [V].
- Maze 50009 (shortest route, 12.2 m) defeats both actors in gym and physics; A* solves it in 41.0 s.
  - s6 shuttles three round trips to the dead-end pocket (5,1). In physics s5 instead dithers for 87 s in the x = 2 column [V~].
  - The ego crop does show the route's first leg. What the policy lacks is visitation history, and the route's turn lies 2.8 m away, round a corner and past the 2.4 m crop edge [V].
  - The goal input is straight-line only, and the trap is nearer the goal in a straight line (5.60 m vs 5.77 m) [V].
- Yaw is bang-bang in both worlds: at the 1 rad/s limit on 95–98 % of gym steps and 90–99 % of sampled physics steps. Final policy std is 9.4 and 13.7, since `--ppo v2` has no std ceiling. This is a policy property, not a transfer artefact [V~].
- Sysid of the biped gait [I] (`solutions-20260930/sysid/data/`):
  - Physics yaw responds within one 40 ms step. Fitted in the gym's own form, tau is 0.06–0.08 s with latency 0, against the gym's 0.15–0.45 s and 0–2 steps.
  - The identified dynamics alone do not significantly change s5 in the gym: 47/72 vs 50/72 (p = 0.51). Predeclared check P1 failed on block 9048 (17 vs 20), so a small residual effect is not ruled out.
  - Dynamics + brake + 10 Hz lidar packets integrated at the next step's pose + heading 0 + settle reproduce the gap: 33/72 (p = 0.0005); on the non-exploratory blocks alone 24/48 vs 34/48 (p = 0.013).
  - Integrating each packet at its capture pose removes the gap: 48/72 (vs the full pipeline p = 0.0015). s6 is insensitive: 44/48 with the full pipeline, 42/48 with the fix.
  - The plan author re-read the pooled counts from `data/gym_reeval_final.json`. The physics evidence for the fix is one episode on training maze 9001: stuck → goal in 50.1 s.
  - The code asymmetry is real. The humanoid branch integrates `grid` from the capture pose (`scripts/bench/maze_explore.py:789-795`, pose recorded at `:837-840`), but the learned ego map is updated at the loop-top pose (`:798-799`).

**Root cause.**
- s6 and 50009: reactive local navigation with no route or visitation memory, steered by a straight-line goal (high confidence).
- s5's extra 2–3 goals [I]: integrating the map one step late, under fast physics yaw and bang-bang commands, smears the ego map, and the brake then stalls the robot (medium in the gym, low in physics).

**Solutions (ranked)**

| # | Exact change | Predeclarable rule | Cost | Risk |
|---|---|---|---|---|
| N0 | In `maze_explore.py`'s biped `--policy` branch, update `learned["emap"]` from the pose recorded when the packet was captured, as the humanoid branch does for `grid`. Put it behind an opt-in flag so the default path and `21484212` reproduce byte for byte. Labels: LEARNED gait + LEARNED actor, ORACLE pose and goal | Screen, both actors, with vs without the fix, on training-range mazes 9120–9143 (unused). PASS iff s5 nets ≥ +4 with ≤ 1 lost, s6 nets ≥ −1, 0 falls and wall-contact steps not increased. McNemar is reported, not gated; the gym estimate is +5/24, so +4 is at the edge. Then the unchanged rule (≥ 10/12 for each actor, 0 falls) on a fresh 12-maze set declared first (60000–60011, unused as of today) | ≈ 5 lines; 96 episodes of ≤ 3 min | Recovers s5's residual only: s6's misses reproduce in the gym, so N0 alone is unlikely to take s6 to 10/12. Screening mazes may have been seen in training (gym seeds < 10000) |
| N1 | Hybrid: A* on the lidar map gives the actor a sub-goal 1.5–3 m along the route as its goal input; no retraining. Labels: LEARNED local actor + SCRIPTED route planner + ORACLE pose | Tune on training-range mazes only. Judge on maze seeds 62000–62011 (declared now; unused as of 2026-10-01): ≥ 10/12 for each actor, 0 falls | small code; ≈ 1–2 CPU-h | The actor trained with a fixed goal, so a moving near goal is out of distribution (it may slow near the carrot). A hybrid does not meet the "learned navigation" goal as written. The learned actor is faster than A* in only 12 of 17 matched successes [V] |
| N2 | Retrain (NavGym v5) with a visitation (breadcrumb) channel and/or a recurrent actor; for 50009-type look-alike columns, also a wider or coarser map channel (or rely on N1's sub-goal). Optional extras: the deployment flags already in `SysidNavEnv` [I] (brake, 10 Hz packet hold, capture-pose integration) and yaw smoothness (hold yaw 3 steps, or penalise yaw changes) | V4_RULE unchanged on never-used maze seeds 61000–61047, then the physics rule (≥ 10/12 for each actor, 0 falls) on maze seeds 63000–63011 (both declared now; unused as of 2026-10-01) | 3 seeds × 30 M steps (≈ 9 h × 16 CPUs each) | Not seed-proof (v4 s7 missed every clause). A new observation makes this a new arm, never compared with v4's gate. A visitation channel addresses s6's shuttle but not 50009's look-alike columns: the decisive turn (4,2)→(5,2) is 2.8 m out, past the 2.4 m crop edge, so from the corridor the route column and the x = 3 dead-end column look alike. That needs the wider or coarser channel, or N1's sub-goal [V] |

Not recommended:
- A relaxed brake: in the gym it trades time-outs for collisions (0 → 6) [I].
- A yaw low-pass at deployment: mixed gym evidence (+4 and +2 for s5), and physics maze 9001 stayed stuck [I].

Not planned as a separate step: sysid change 5, the identified ranges in the gym's `Dynamics` (yaw tau 0.04–0.45 s with latency 0–2, turn-in-place and forward dead zones, stride wobble). The sysid report ranks it fifth and expects little transfer gain, because the identified dynamics alone do not significantly change s5 (47/72 vs 50/72). It is a fidelity change: it can join N2 as an opt-in flag, with the report's own test (0.48 s heading RMSE ≤ 0.065 rad on the 4 saved physics replays) [I].

**Recommended next step:** N0's screen. It fixes a deploy-side correctness bug whatever comes after, isolates one factor and needs no training. N1 and N2 follow for the s6 and 50009-type misses. Never run 40000–40047 or 50000–50011.

**Result, same day (`21501509`, rule predeclared in the ledger): N0's screen is NEGATIVE.** armV4-s5 went 15 → 17/24 (won 4, lost 2, net +2; McNemar p 0.69; the rule needed net ≥ +4 with ≤ 1 lost), armV4-s6 23 → 23/24, 0 falls, wall contacts 14 → 11 and 0 → 0. The fresh set 60000–60011 was not run and stays unused. The fix is correct and harmless but recovers far less than the gym re-creation predicted (+5/24), so the next step is N1 or N2.

## 2. Turning recipe

**What the logs show**
- 1/12 seeds qualifies: TurnBoth-s0, through v2x (`21442352`: turns 10/10 on reset seeds 10–14, walk 3/3, push 7/60) [V].
  - Under the noise-free turn_test v2 (reset seeds 0–2), 0/12 seeds pass; TurnBoth-s0 scores 5/6 [V].
  - Fine-tunes trade turning for push robustness: TurnRest push falls 0.48/0.52/0.45 (`21443270`/`271`/`279`), CONT 0.32/0.42/0.32. The PUSH arm keeps push falls ≤ 0.10 but loses the turn (`21443377`).
- MuJoCo probe of four non-turners, ±0.6 rad/s after a 3 s settle [V] (`turning/exp1_noise.log`; re-run in `verify-turning/rerun_results.jsonl`):
  - Noise-free, 0/16 runs reach 150° (10–18°, 0 lift-offs).
  - With each checkpoint's own training action std, 16/16 turn 164–256° with no falls. At zero command with noise they turn −7.5 to +4.2°.
  - The probe's actor is exact: within 1.9e-5 of the ONNX export.
  - Caveat: the 16 noisy runs share two RNG streams. A third, independent stream gave 238°.
- Training observation noise alone (no action noise; the noise config matches all 13 runs' env.yaml) lets 10 of the 11 non-turning arm seeds pass a v2-style replay [V] (`turning/v2_obsnoise_x1.0.json`). This is a diagnostic, not turn_test, and 10/11 is a floor.
- Isaac probe `21442353` ran noise-free [V]: `enable_corruption = False` (`scripts/bench/turn_diagnose.py:359`), the deterministic TorchScript actor, and the stock task, not the arms' training tasks.
  - So "not a sim2sim gap" holds only for the noise-free loop, and "< 8/32 = not learned" does not follow.
  - Isaac behaviour with noise on: [U].
- Deploying with observation noise turns but fails the push gate: TurnBoth-s0 13/60 and TurnBoth-s1 18/60, vs 7/60 and 9/60 noise-free [V~].
  - Those runs used the qualify gate's push seeds 0–9, so TurnBoth-s1 has no unspent push seeds left.
- "Marching at zero command predicts turning" is post-hoc and stays inside one lineage. All 7 marchers turn, but only 4/7 pass v2, and there are counterexamples [V~]. Do not use it as a criterion.
- `feet_air_time` pays only in single stance and only when ‖cmd[:2]‖ > 0.1 [V], so a pure turn pays for no stepping. Pure turns are 0.27–1.27 % of training commands [I].

**Root cause.** PPO optimises the stochastic closed loop: exploration and observation noise supply the stepping a pure-yaw turn needs. The mean policy sits at a standing fixed point, and the gates and the robot run that mean policy without noise.

**Solutions (ranked)** (labels: LEARNED gait; MuJoCo gates)

| # | Exact change | Predeclarable rule | Cost | Risk |
|---|---|---|---|---|
| R1 | Gait clock in the actor: sin/cos of the phase, 75 → 77 obs. Contact-schedule reward `feet_gait`, XNOR(stance, contact), paid at every command including zero, plus foot clearance. `feet_air_time` weight 0. Fixed ±0.5 m/s pushes from iteration 0; default PPO noise. Harness: a clock-aware RlController in `turn_test.py`, `run_eval.py` and `turn_diagnose.py`, with the 75-obs path byte-identical under test and the phase reset at `runner.reset` in step with Isaac's `episode_length_buf` | Unchanged: v5's joint rule (SLURM_JOBS.md:3055). PASS iff ≥ 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's unchanged rule (v2x ≥ 9/10 on reset seeds 10–14, walk ≤ 15° on ≥ 2/3, push ≤ 9/60); else FAIL; INCOMPLETE if any JSON is missing. A seed that fails v2 does not count, even if it qualifies through v2x | harness + ≈ 12 GPU-h | Period and weight are unverified. The investigator's 0.6 s and 1.0 are its own choices; the only documented precedent is 0.8 s and 0.5 (unitree_rl_lab G1). Freeze them in the header from that precedent or from a pilot on unscored seeds. Paying at zero command follows unitree_rl_gym, not unitree_rl_lab [V]. Pushes from iteration 0 may cost the turn: the v5 PUSH arm (`21443377`, TurnRest + adaptive pushes) kept push falls ≤ 0.10 but turned 0/10 on fresh reset seeds on all three seeds, and 22-DoF from-scratch push training showed no detectable benefit at four seeds (disclosed at SLURM_JOBS.md:3055) |
| R2 | Same reward, but the clock goes to the critic only; no deploy or harness change | same (v5's joint rule) | ≈ 12 GPU-h | No Unitree repo has tested a clockless actor that marches at zero command [V]. Same push risk as R1 |
| R3 | Lower exploration noise (clamp `runner.alg.policy.std`, which matches the rsl-rl 3.x checkpoints [V]) + TurnRest commands + pushes | same (v5's joint rule) | ≈ 12 GPU-h | Low confidence (report C's own rating): with less noise the mean policy must learn to step by itself, which may simply not happen. Without the std clamp, TurnRest + adaptive pushes is the v5 PUSH arm (a fine-tune of TurnBoth-s0), which lost the turn on every seed |

Not a gate change, but a caveat: the per-seed screen would have rejected TurnBoth-s0 (v2 5/6) and CONT-s0/-s1 (walk drift), so expect a strict screen.

**Recommended next step:** one launcher with R1 and R2 as two arms × 3 seeds, its rule in the header. Before submission, land the clock-aware RlController with two checks: a byte-identity test on the 75-obs path, and a smoke test showing the phase resets with the episode.

## 3. Mission 7 replay gate

**What the logs show** (all [V] unless tagged)
- Four fixed-1.2 s-crossing variants pass the exact ten-fall replay 10/10 (`results/mission7-campaign-20260923/replay-gate-matrix-summary.json`; G-ref's own `results/mission7-approach-followup-20260922/plate-stage-cn-c22-guarded/result.json`):
  - G-ref `21397732`, wait-open `21400863`, lateral 0.35 `21400897`, and lateral 0.35 + wait-open `21400953`.
  - Wait-open is identical to G-ref crossing for crossing. The two lateral-0.35 gates started 7 crossings to G-ref's 5, and whether they cleared a plate is [U]: only their `result.json` was kept.
  - Lateral 0.25 `21400805` scored 8/10. The cross-clear variants, which keep crossing until the base is 0.35 m past the plate centre or 4 s pass, score 6–9/10 (`21405537`–`21405545`), and V2 `21401689` 7/10.
- G-ref's five staged crossings never cleared a plate: they moved +0.015 to +0.215 m and ended 0.187–0.328 m short of the plate centre.
  - After hand-back, the replayed commands carried the robot across the round plate in layout 7. In layouts 0, 1 and 4 it stood on the disc without falling.
  - Door 0 opened before every G-ref crossing, so the gate never needed the stage to press a plate (`verify-mission7/verify_crossings.out`, `poststage_check.out`).
- Of the 35 crossings in the cross-clear gates, 15 were on the wrong square plate.
  - All 7 in-crossing falls were sideways entries onto the round plate in layouts 0, 4 and 7.
  - The 14 forward entries (0 falls) are 5 repeats of layout 1 plus 9 square-plate crossings, so they are not 14 independent trials.
- The two 9/10 falls:
  - `21405537` layout 7 entered 1.02 rad off the door direction. It stalled at along −0.29 m with a foot on the plate edge, under a mostly forward command, and fell 2.40 s after the kick.
  - `21405541` layout 0 got a pure sideways command, 1.46–1.82 rad off, with an uncommanded yaw of −0.99 rad/s, and fell 1.92 s after the kick.
- Gate plumbing:
  - `crossings_completed` (`slurm/repo20260923/m7_replay_gate_followup.sbatch:90`) counts time-outs and falls as completed crossings.
  - `scripts/mission7_plate_safe.py:158-159` sets `doors_16` and `transport_16` to episodes ≥ 16, so `21398514` reports both true at 1/16 and 0/16 successes.
  - The route-handoff probe has no pass criterion.
- Budget: the docs say 101 of 512 episodes remain; the ledger says 1 remained, and 54 more were then charged (SLURM_JOBS.md:2737, 2742, 2842, 2883).
- TurnBoth-s0's deploy.yaml differs from the shipped gait's only in the policy path and the unused `command_velocity`, so swapping the policy is functionally equivalent. `prev_actions` would carry the old gait's last action across the swap.

**Root cause.**
- The falls come from a standstill-plus-kick entry onto the plate edge with a sideways or yawing command. The stage enters sideways because the shipped gait cannot turn in place (medium: two failures traced and 7/7 in-crossing falls sideways, but sideways entry and layout are confounded).
- The replay cannot certify a crossing mechanism (high).
- The "commands applied 82–90° off" explanation of the post-hand-back falls is refuted: it fits 1 of 5.

**Solutions (ranked)**

| # | Exact change | Predeclarable rule | Cost | Risk |
|---|---|---|---|---|
| M1 | No episodes. Code the route gate as successes (`doors_16 = successes ≥ 16`, likewise transport), matching MISSION7_TASKS.md:32 and :48, and give the handoff probe the same criterion. Count `crossings_completed` as real clears (along ≥ +0.35 m before hand-back, upright). Reconcile the budget in the ledger | Unit tests on saved outputs: `21398514` must read FAIL, and `21405537` / `21405541` must read 4/7 and 1/7 clears | code only | Makes the coded gate stricter, matching the docs |
| M2 | Plate bench with TurnBoth-s0 as the stage gait (LEARNED gait, SCRIPTED stage). Swap `env.controller.policy` in `PlateStage._start` and reset `prev_actions` at the swap and at hand-back. The stage turns to the door direction, crosses straight forward and turns back. Plumb `--stage-gait` through `mission7_plate_stage.py`, `mission7_route_handoff_probe.py` and the follow-up flag translation | Bench: 64 crossings, one per training layout L = 0–31 and door d = 0/1, in a 4 × 2 × 2 grid of 16 cells × 4 crossings. Entry heading 0°, +90°, −90°, 180° for L mod 4 = 0, 1, 2, 3; round (correct) or square (wrong) plate for ⌊L/4⌋ mod 2 = 0 or 1; entry from standstill or from continuous walking for d XOR (⌊L/8⌋ mod 2) = 0 or 1. Each cell then holds 2 door-0 and 2 door-1 crossings, and each heading 16 (2 plates × 2 entry modes × 4). Plate activation is measured separately, not gated. PASS iff ≥ 62/64 clear to along +0.35 m within 10 s, 0 falls, and ≥ 15/16 per heading; together these allow at most 2 misses and at most 1 in any cell. Then the unchanged exact replay once, then the route gate as coded by M1 | A new budget line for the user to authorize: 64 bench + 10 replay + 32 route-gate episodes = 106 episodes, ≈ 1–4 CPU-h on 2-CPU jobs, no GPU. Estimate from sacct: 16-layout Doors handoff runs `21400801`–`804` took 0.18–0.28 CPU-h each, replay gates 0.13–0.33 CPU-h, and `21398514` (10 replay + 32 route episodes) 1.26 CPU-h; the bench-episode cost is unmeasured | Only one qualified checkpoint exists; it marches in place, and its falls on plates are unmeasured |
| M3 | Fallback: turn while stepping with the shipped gait (`--align-yaw` is only a moving yaw term) + a stall watchdog | M2's bench grid and rule | Its own request of the same size (106 episodes, ≈ 1–4 CPU-h), made only if M2 fails its bench | A sideways component remains while turning |

Repairing the gate itself (route controller after hand-back, every takeover must clear the plate) would be a declared gate change whose net direction is not established. "Every takeover must clear the plate" is stricter. But driving the post-hand-back phase with the route controller removes the replayed phase in which 5 of the 12 falls in the five cross-clear gates (`21405537`–`21405545`) happened, and only 1 of those 5 fits the misapplied-command artefact. It requires the user's sign-off, and the post-hand-back falls should be measured first, not discounted.

Not planned: report A's option (4), a learned local crossing (fine-tune on 3 cm discs), which report A itself ranks as a last resort. It needs a training budget and its own gate, and is reconsidered only if M2 and M3 both fail the bench.

**Recommended next step:** M1, with no episodes. Propose no Mission 7 episode until the user reconciles the budget and authorizes a bench line (M2: 106 episodes): under the ledger's reading the line is already ≈ 53 episodes over.

**Done, same day (no episodes):** M1 is in code (`scripts/mission7_gates.py`, used by `mission7_plate_safe.py`, the route-handoff probe and the replay follow-up). 21398514 now reads FAIL as the docs say, and 21405537 / 21405541 read 4 and 1 real clears of 7. Two independent reviewers checked it; `tests/test_mission7_gates.py` holds 51 tests. The budget is still the user's to reconcile.

## 4. Cooperative lift and placement

**What the logs show**
- The "pinch crouch needs 28–30 Nm" figure (`21402699`) is the step-0 snapshot, taken after `env.reset` and before any `env.step`, with 0 N on every foot [V].
  - With the feet loaded (steps 2–5) the knees need 4.15–6.35 Nm. That matches the 4.8 Nm static estimate, ≈ 80 % of the 6 Nm cap (`verify-coop/spawn_torque_table.json`, `coop/static_limits.json`).
  - Both the crouch and the standing pose topple under pure PD hold, so neither infeasibility nor feasibility is shown [V].
- Stand3 replay `21470828` [V]:
  - On steps with the cube centre above 0.61 m, the lowest corner sits within 1 cm of the plinth or deck top on 97.4 % (s0) and 79.7 % (s1) of steps.
  - A cube within 8° of flat is above 0.61 m on 2 of 44,990 steps (s0) and 0 of 40,510 (s1).
  - `object_is_lifted` (`coop_lift_mdp.py:249-264`) tests centre height only, so the income comes from re-orienting a supported cube.
  - The curriculum sat at its configured 0.06 m ceiling (`stand_mdp.py:501`), not at a roll height.
  - The replay traces are untracked (`.gitignore:27`).
- Isaac hands: hand force is 0 on every step, and 127 s1 steps have a hand inside the cube at zero force [V]. The hand-collider asset was not inspected [U].
- MuJoCo scripted harness, exploration seeds 100–104:
  - Lift-hold goes 0/3 → 3/3 only with both pad torsional friction (condim 4, 0.04 m) and elliptic cone + impratio 10. In all three passes the cube is still held rolled 0.63–0.93 rad [V].
  - Lift-place 5/5 also needed reverse lowering, which was found after three targets were tried on the same seeds. With the published lowering, base scores 1/5 (that cube placed on its side) and the contact variant 0/5 [V~].
  - The pads meet the cube ≥ 26° off-face, so 0.04 m torsional friction assumes a flush pad that does not exist [V~].
- Actuator limits:
  - 6 Nm (legs) / 4 Nm (arms) are upstream training caps, but upstream's deploy runtime writes them into motor firmware (`real_humanoid.cpp:53, :515`). Any fix that needs more torque is sim-only until those limits are raised [V~].
  - The ideal current-limited output is 27.6 / 35.3 Nm [V~].
  - "Paper tested 20 Nm", the cycloidal gearbox and the motor mapping: [U].

**Root cause.**
- Learned: a hackable lift reward, centre height with no orientation or support test (high).
- Scripted: edge/corner pad contacts let the cube roll ≈ 90° in the hands, so opening a hand on its top face drags it off (medium; exploration seeds only).
- CubeToShelf crouch: the blocker was misattributed. What is missing is a balance-controller test, and none is defined: no controller, poses, pass rule or cost. It is not planned here.

**Solutions (ranked)**

| # | Exact change | Predeclarable rule | Cost | Risk |
|---|---|---|---|---|
| C1 | Flush-pad wrist variant in the scripted MuJoCo harness: pad faces parallel to the cube faces at the squeeze pose, as opt-in CarryParams. Declare the elliptic cone as a harness change (it also changes foot contact). Freeze the lowering target before any scored seed. Labels: LEARNED gait (frozen) + SCRIPTED arms + ORACLE cube pose (scoring only) | Probe on exploration seeds 120–124 (100–119 were used for tuning; check before use): proceed iff cube tilt ≤ 0.35 rad throughout lift-hold on ≥ 4/5, with the robot-fall clauses unchanged. Then score on never-used seeds 20–29 (hold) and 30–39 (place): unchanged LIFT_HOLD / LIFT_PLACE clauses plus tilt ≤ 0.35 rad, ≥ 8/10 per crew or pair | ≈ 15 CPU-min + geometry | A modified end-effector, not the stock robot. The contact change alone leaves the cube rolled, so failing the tilt clause is the expected outcome unless the pad stops the roll |
| C2 | Stand4, a new task never compared with Stand3. Roll-proof lift reward: pay only when the lowest corner is ≥ 2 cm above every support and tilt ≤ 15°; seating requires tilt ≤ 8° and release. Confirm the hand colliders first (`assets/cloth/berkeley_humanoid_lite_hand_colliders.usda`). Log `minimal_height`, pinch distance and the upright gate. First run with the tilt fix (`2976f36`) | Stand3's structure (kill rule at model_1000, completeness, success ≥ 0.10 on ≥ 1 of 2 seeds), with release and tilt checked in success. **This overrides a predeclared funding rule and needs the user's sign-off.** SLURM_JOBS.md:3080 declared "A Stand4 is justified only if both checkpoints give the same R2, R4 or R5", and :3086 recorded "no Stand4 on this evidence" (s0 R4, s1 R2). C2 rests instead on the confirmed farming finding | ≈ 24 GPU-h | Lift income is expected to vanish; the curriculum may stall at stage 1, as the crews did |

Not planned: C3, a payload held by geometry (forearm tray, 0.9–1.35 Nm per arm [I]) or a gait retrained with hand loads (≈ 15 GPU-h [I]). Both change the task, neither has a predeclarable rule, and the tray has no cost estimate. Either variant needs its own rule and budget before any run.

**Recommended next step:** C1's mechanism probe. It decides cheaply whether the scripted route has a physical fix. CubeToShelf's crouch stays closed: re-opening it needs a balance-controller test that is not yet defined (controller, poses, pass rule, cost), so nothing is planned for it. Stop citing "exceeds 6 Nm" as the blocker.

## 5. IM10A recording (SF-05)

**What exists**
- Three new repo files, committed with this plan: `scripts/sensors/imu_allan.py`, `tests/test_imu_allan.py` and `docs/IMU_RECORDING.md`.
- `tests/test_imu_allan.py`: 18 passed, 1 skipped (the bag reader; `rosbags` is not in the shared venv). The plan author re-ran it on 2026-10-01 (21.1 s), and so did the coordinator before the commit (18.9 s) [V].
- Synthetic recovery [I]:
  - White noise N comes back within 0.6 % of the true value.
  - At 200 Hz over 16 seeds, bias instability (minimum method) reads 0.96 / 0.99 / 1.04 × true (16th / 50th / 84th percentile) at 2 h, against 0.80 / 0.91 / 1.01 at 30 min.
  - In the same runs, rate random walk is visible in 13/16 records at 2 h but only 3/16 at 30 min.
  - N still reads ≈ 2 % low behind the module's default 20 Hz bandwidth.

**Open, needs the device** [U]:
- The board identity (probably an IM10A).
- The vendor ROS 2 driver: which packets it parses, whether it writes settings at startup, its time stamps, and its magnetometer and pressure topics.
- Whether 200 Hz is actually delivered, and whether MCAP storage exists on Foxy.
- Absolute latency, which needs a reference sensor of known latency; without one the delay setting stays 0.
- The mounting transform and yaw reference. Each recording covers one unit, one temperature and one power cycle.

**Disk** [V]: project 30762 holds 1.529 TB against a 1.5 TB soft limit (2 TB hard), with 3 weeks 5 days of grace (`lfs quota -p 30762`, 2026-10-01). STATUS.md's ≈ 60 GB reclaim estimate for `results/weekend-20260919/fold-adapt-s{0,1}` is stale: those folders now hold 4.3 GB each (checked 2026-10-01). What to prune is the user's decision.

| # | Exact change | Predeclarable rule | Cost | Risk |
|---|---|---|---|---|
| I1 | Record three bags per `docs/IMU_RECORDING.md`: ≥ 2 h stationary at 200 Hz, slow rotations per labelled axis, a tap test. Run `imu_allan.py` on a login node | Done when `imu_measured.json` (gyro and accel N, B, K) comes from a ≥ 2 h record in which K is measured on all three axes of both sensors (`gyro_bias_walk_kind` and `accel_bias_walk_kind` read "measured (max over axes)", not an upper bound), is committed, and `IM10A_DATASHEET` / `ImuNoise` read it. The script's LOW CONFIDENCE flag fires only below 1 h, so it says nothing about a 2 h record. If K is only a bound on any axis, I1 is not done: record a longer stationary bag rather than commit a bound as measured | bench time; ≈ 0.7 GB per 2 h bag | Driver behaviour is unverified; one unit, one temperature. A 2 h record can still miss K: it was not visible in 3 of 16 synthetic 2 h records [I] |
| I2 | If 200 Hz is not delivered (the `ros2 topic hz` check): 2 h at 100 Hz (synthetic recovery there: N 1.00, fit B 0.99–1.00, K 0.74–0.91 × true [I]). Missing MCAP is not a trigger: drop `-s mcap` and record sqlite3 `.db3` (`docs/IMU_RECORDING.md:129-130`); the bag reader passes on both formats [I] | same | same | K reads less reliably |
| I3 | Latency: tap test against a second IMU of known latency | Report absolute latency only with a known reference; otherwise a relative offset, with the sim delay kept at 0 | bench | The 30 ms budget stays a sim-only number until then |

**Recommended next step:** the user runs I1; the kit is committed with this plan. Planned compute is CPU-light (login node).

## Corrections to earlier claims

Ledger and doc claims this investigation overturned; the last column gives the verification verdict on the correction.

| Earlier claim (where) | Corrected reading | Verification |
|---|---|---|
| The pinch crouch "demands 28–30 Nm … infeasible by a factor of five"; COOP-17..20 "BLOCKED — spawn posture exceeds the 6 Nm actuator limit" (`21402699`; SLURM_JOBS.md:2799, 2907; STATUS.md:15; REPO_TASKS.md:94–97) | A step-0 reset transient: 0 N on the feet, and the figure equals 20 Nm/rad × the reset knee angle. Loaded knees need 4.15–6.35 Nm, the static estimate is 4.8 Nm (≈ 80 %). Infeasibility is not shown, because both poses topple under pure PD hold | confirmed |
| "Applied torque pinned at 6 Nm from step 0"; "standing pose 0.2–0.7 Nm at step 2" (`21402699`) | At step 1, 0/8 robot-envs are saturated. The standing step-2 value is a free-fall reading | confirmed (both statements wrong) |
| "The gym-to-physics gap (gait response lag, the speed brake, 10 Hz lidar) costs 3–4 goals in 12" (SLURM_JOBS.md:3096; ROADMAP.md:25 "close the gym-to-physics gap") | On hash-identical layouts s6's gap is 0 and s5's is at most 3 (≈ 2 against its randomized draws, which are not time-matched). The comparison with v2's 10/12 was not re-checked | confirmed; v2 comparison [U] |
| Isaac probe `21442353`: "< 8/32 = not learned … not a sim2sim gap" (SLURM_JOBS.md:3027; STATUS.md:13; REPO_TASKS.md:72) | The probe ran noise-free (corruption off, deterministic actor, stock task). "Not a sim2sim gap" holds for that loop, but "not learned" does not follow: in MuJoCo four non-turners, three of them among the probe's, turn with their own training action noise | confirmed |
| Exact replay `21397732` PASS 10/10 read as a working crossing (MISSION7_TASKS.md:47) | Its staged crossings never cleared a plate, and door 0 was open before each one. Wait-open `21400863` is identical crossing for crossing. Lateral 0.35 `21400897` and `21400953` also pass 10/10 but started 7 crossings to G-ref's 5; whether they cleared a plate is unverified | partly: confirmed for `21397732` and `21400863`; the lateral-0.35 gates [U] |
| "Crossings completed 7" in the replay summaries (`m7_replay_gate_followup.sbatch:90`) | It counts 4.04 s time-outs and fallen crossings. Real clears: 4/7 (`21405537`) and 1/7 (`21405541`) | confirmed |
| Route gate `doors_16` / `transport_16` true (`mission7_plate_safe.py:158-159`; `21398514`) | Coded as episodes ≥ 16; the documented gate is 16/16 successes (MISSION7_TASKS.md:32, 48) | confirmed |
| Mission 7 "101 of 512 episodes remain" (STATUS.md:23; REPO_TASKS.md:22; MISSION7_TASKS.md:37, and :75 "0 / 512") | The ledger says 1 remained after `21401987` and 54 more were charged on 09-24 (SLURM_JOBS.md:2737, 2742, 2842, 2883) | confirmed (unresolved) |

Investigation claims (reports A–D) after verification:

| Claim (report) | Status |
|---|---|
| A: "G-ref never crossed a plate" | partly: true of the staged crossings; after hand-back, layout 7 crossed the round plate (max tilt 0.16) and layouts 0, 1 and 4 stood on the disc |
| A: post-hand-back falls are a replay artefact (commands 82–90° off after an align stage) | **refuted**: fits 1 of 5 falls; there is no align stage |
| A: 0/14 falls on forward entries, 95 % upper bound 0.19 | partly: the counts reproduce, but 15/35 crossings were on the wrong plate and the forward set is layout-1 repeats plus square plates, so the bound does not hold |
| A: p(10/10) per variant 0.09–0.30 | **unverified** (no derivation in the files) |
| A: login-node re-run "bitwise"; TurnBoth-s0 deploy.yaml "differs only in the policy path" | partly: the physics is identical but the JSON differs in the last bits; deploy.yaml also differs in the unused `command_velocity` |
| B: "28–30 Nm appears only at step 0" | partly: it also recurs after the robots fall (pinch step 24: 29.59 Nm at tilt 1.82 rad); among upright, loaded steps it appears only at step 0 |
| B: "steps 1–7 with feet loaded: 2.1–5.9 Nm" | partly: that range is robot_a env 0 only; over all 8 robot-envs, steps 1–7 span 2.10–6.83 Nm |
| B: the lift curriculum stalled at 0.058 m = the roll height | **refuted**: it is capped at 0.06 m (`STAND3_LIFT_MAX`) |
| B: 74–75 % of steps pay `lifting_object` | partly: 66.5 / 74.0 % at the 0.06 level, 85.7 / 81.5 % at 0.04; this is the height clause only |
| B: the cube rolls about the pinch axis in 79 / 64 % of episodes | partly: the up-face changes about any horizontal axis; the tilt axis is the pinch axis in 93 % (s0) / 43 % (s1) |
| B: the s0 cube is "cocked on the deck edge" | partly: its lowest corner is over the plinth 47.7 %, over the gap 42.9 % and over the deck 9.4 % of held steps |
| B: MuJoCo lift-hold 0/3 → 3/3, lift-place 0/5 → 5/5 | partly: each needs both contact changes; the cube stays rolled 0.63–0.93 rad; place also needed a lowering target chosen on the same seeds |
| B: 6 / 4 Nm "not hardware ratings"; 25 / 32 Nm peak | partly: firmware enforces them through upstream's runtime; ideal output is 27.6 / 35.3 Nm |
| B: "paper tested 20 Nm", cycloidal gearbox, 6512/5010 motors | **unverified** |
| B: Isaac hands have no collision geometry | partly: zero hand force confirmed; asset **unverified** |
| C: F6 "all 7 marchers pass v2, none of the other 15" | partly: 4/7 pass; the association is post-hoc, inside one lineage, with counterexamples |
| C: the numpy actor is "within ~3 %" of ONNX | corrected: it is exact (≤ 1.9e-5); the 3 % came from a yaw-origin mismatch |
| C: R1 is the unitree_rl_lab recipe | partly: same functional form, but unitree_rl_lab gates it at ‖cmd‖ > 0.1 and ships its clock commented out; the 0.6 s period is **unverified** |
| C: F9 "deploying with noise" | partly: observation noise only, and it used push seeds 0–9 |
| D: "the crop never shows the route up" on 50009 | **refuted**: it shows (4,1) on every pass; the route's turn is past the crop edge |
| D: both actors drive into the 50009 pocket; yaw-rate magnitude saturated 94–99 % in physics, 15–18 flips/s | partly: s5 in physics never passes x = 3.07 m; saturation is 90–99 %; the flip rate fits s5 only, and the physics flip rate is **unverified** |

## What this plan could not support

- **Sysid as a whole is [I]:** only the pooled counts and the code asymmetry were re-checked. The physics evidence for the capture-pose fix is one episode on a training maze, and splitting the effect between dynamics, brake and late integration rests on the exploratory block 9000.
- **Turning [U]:** Isaac behaviour with noise on; the fraction of noise realisations that produce a turn; the F2 two-kind split (reset seed 0 only); the R1 period (0.6 s) and weight (1.0).
- **NavGym [U]:** the physics yaw-flip rate (traces are sampled at 0.2 s) and the v2 comparison at SLURM_JOBS.md:3096.
- **Mission 7 [U]:** the p(10/10) range, and whether the lateral 0.35 gates (`21400897`, `21400953`) cleared a plate (only `result.json` was kept).
- **Coop:** [U] "paper tested 20 Nm", the gearbox type, the motor mapping, the hand-collider asset, and gearbox safety above 6 / 4 Nm; [I] the tray torques and the gait-retrain cost.
- **IM10A:** everything under "Open, needs the device".
