# Mission 7 task checklist

Persistent, evidence-linked status for the Mission 7 campaign. Updated after
each meaningful batch. Status vocabulary: **DONE** (evidence linked),
**ACTIVE**, **BLOCKED** (names the blocker), **SUPERSEDED** (names what
replaced it). Route failures are never dropped from a denominator.

## Current status

- **Last updated:** 2026-09-23, second campaign closeout.
- **Source checkpoint:** see `git log`; every job's source hashes are in its
  `submission.json`.
- **Achieved milestone:** a **validated, gate-compatible improvement** —
  candidate 1 (unchanged guarded stage, early handoff, `prev_actions_reset`
  with a 3 s stall trigger). It fires only on genuine stalls (pauses stay
  tick-identical), and on every genuine stall seen (5 across development
  and confirmation) it restored forward progress, with route success in 4
  and no regression anywhere. Predeclared confirmatory on never-used
  validation 16–31: Doors 1/16 → 1/16, **Transport 2/16 → 4/16**, improved
  2 / regressed 0, falls unchanged — **underpowered (2 discordant pairs,
  p = 0.5), not null**, exactly as predeclared.
- **Baseline vs candidate:** development (validation 0–15): Doors 3/16 →
  4/16, Transport 1/16 → 1/16; confirmation (16–31): Doors 1/16 → 1/16,
  Transport 2/16 → 4/16.
- **Mechanisms established (traces, not conjecture):** stall = gait feedback
  fixed point; in-stage/post-exit failures = the raised plate (settle →
  fixed point on the plate edge; sideways crossing when unaligned; gait
  cannot turn in place; route walks back to the pre-door waypoint after
  hand-back); world −x = a 0.546–0.605 m walking envelope against a 0.61 m gap (marginal; the 0.632 m sphere-bound claim is withdrawn). Doors/1
  completes end-to-end with the composed crossing fix (77.7 s, rendered),
  but that composition fails the exact replay 7–9/10 and is not promoted.
- **Open tasks and exact blockers:** (a) route gate ≥16/16 — blocked by the
  raised-plate crossing: needs a crossing that keeps the gait walking from a
  standstill without a lateral component *and* holds the ten-fall replay at
  10/10; (b) privileged Approach — **passed** 62/64 with fine centring (`21401966`); (c) sensor-release and four-sensor
  studies — closed until (a) and (b) pass.
- **Budget:** 411 / 512 episodes; **101 unspent by decision** as written here. **Unresolved 2026-10-01:** the ledger records 1 remaining after `21401987` and 54 more charged on 09-24 (≈ 53 over); no Mission 7 episode until the user reconciles the line. Retained
  artifacts 7.7 GB of 10 GB after trimming raw traces of withdrawn arms and
  failed gates (compact results and receipts kept); disk ≥ 130 GB free.
- **Validation:** 176 tests; every job preflighted and hash-frozen; replay
  gate unchanged for the frozen candidate.

## Gates (unchanged unless evidence says otherwise)

| Gate | Status | Evidence |
|---|---|---|
| Pressure-plate exact replay, 10/10 upright | **PASS** (2026-10-01: its staged crossings never cleared a plate and door 0 was already open, so the replay cannot certify a crossing) | `21397732` |
| Doors ≥16 and Transport ≥16 (documented route protocol) | **FAIL** 1/16, 0/16 (2026-10-01: the code's `doors_16`/`transport_16` counted episodes, not successes, and read true here; fixed in code the same day by `scripts/mission7_gates.py`, under which 21398514 reads false/false) | `21398514` |
| Privileged Approach ≥60/64, zero falls, ≥14/16 per direction | **PASS** 62/64, 0 falls, −x 14/16 (`center_bias`) | `21401966` |
| M2 plate bench (TurnBoth-s0 stage gait) ≥62/64, 0 falls, ≥15/16 per heading | **FAIL** 8/64, 8 falls (26/64 lost in the bench's run-up; turns stall at 0.4 rad/s); replay and route gate not run | `21507962` |
| M3 on bench v2 (no run-up; shipped gait turns while stepping) ≥39/41, 0 falls, ≥11/10/9/7 per heading | **FAIL** 16/41, 5 falls (wall contact in 30/41; 180° 0/8); replay and route gate not run | `21514946` |
| clock-s2 as the stage gait on bench v2 (controller swap at takeover, M2's turnboth law) ≥39/41, 0 falls, ≥11/10/9/7 per heading | **FAIL** 18/41, 1 fall (180° 0/8: 6 of 8 clear only after 10.8–17.3 s; 13 of 15 other misses never reach the clear position); replay and route gate not run | `21517668` |
| Sensor-only Both readiness | **CLOSED** — route gate still fails | — |
| Four-sensor comparison | **CLOSED** | — |

## Campaign

| # | Task | Status | Evidence / blocker |
|---|---|---|---|
| 1 | Commit + push guarded probe, sbatch preflight, ignore shapes, CLAUDE.md | DONE | `1a526c7` = origin |
| 2 | Chain instrumentation: cmd→prev_actions→raw→targets→ctrl→joints→feet→base | DONE (pending review) | `--chain-trace`; local Doors/1 baseline classified `frozen_targets`; 99 s/episode, 31 MB trace |
| 3 | `prev_actions_reset` intervention with delivery verification | DONE (pending review) | local Doors/1: APPLIED, delivered-zero verified, route success 82.7 s |
| 4 | Campaign A: 16+16 early-handoff episodes, chain traced, mechanism per episode | DONE | 32 episodes; during-stage 15 (wall), after-stage 12 (falls), stall exposures 5 (`frozen_targets` 4); `campaign-a-summary.json` |
| 5 | Campaign B: baseline vs `prev_actions_reset` on the 5 exposed episodes, fingerprint-matched | DONE | 5/5 applied, fingerprints matched; 2/5 genuine stalls, both un-stuck (stall-anchored); 3/5 transient pauses; Doors/1 → success, Doors/3 regressed; not promoted. Review corrections applied |
| 6 | In-stage crossing workstream | CLOSED for this budget — mechanism validated, no gate-passing composition | standstill → fixed point on the plate edge (kick restarts once); sideways crossing when unaligned, gait cannot turn in place; route walks back to the pre-door waypoint after hand-back. Doors/1 full-route success with V2 + centre ramp + rejoin (77.7 s); replay 7/10 (V2), 8/10 (V3), ≤9/10 (V4). Exact unmet prerequisite: a crossing that keeps the gait walking from a standstill without a lateral component, verified 10/10 on the replay set |
| 7 | Freeze one recovery under the predeclared rule | DONE — candidate 1 frozen | unchanged guarded stage + early handoff + `prev_actions_reset` with a 3 s stall trigger; gate-compatible by construction; +1/−0 in development, +2/−0 on fresh layouts |
| 8 | Confirmatory eval on validation 16–31 (predeclared) | DONE | paired: Doors 1/16 → 1/16, Transport 2/16 → 4/16; improved 2 (T24, T31), regressed 0, discordant 2, p = 0.5 → **underpowered, not null**; no new falls; `candidate1-confirmatory-summary.json` |
| 9 | Exact replay regression for every PlateStage change | DONE — all variants FAIL | wait-open PASS 10/10; V2 7/10 (`21401689`); submitted 2026-09-24: **V2+align 9/10, V3 6/10, V4 9/10, V3+align 7/10, V4+align 7/10** (`21405537–545`), no 10/10, route gate never released; candidate 1 leaves PlateStage unchanged |
| 10 | World −x / privileged Approach workstream | **DONE — GATE PASSED, held-out replicated** | matrix **62/64, 0 falls, 16/16/16/14** (`21401966`, test split, where `center_bias` was also tuned); predeclared held-out −x replication **21/23 vs baseline 12/23** (+10/−1, p = 0.012; `21402017–018`) |
| 11 | Sensor-only and four-sensor studies | BLOCKED | gates closed |
| — | `forward_pulse` rejoin fix | SUPERSEDED | no-op by construction; `21399503` reinterpreted |
| — | "Post-stage route rejoin" framing | SUPERSEDED | stall is free-space, route-state valid (`21400561`) |

## Envelope (authorized 2026-09-23)

≤512 new episodes total · ≤8 concurrent Slurm jobs · ≤10 GB new retained
artifacts · keep ≥100 GB free on `/nfs/hpc/share` · `--constraint=haswell&el8`
by default, `--node cn-c22` only for bitwise replay. Episodes used: see **Budget** above (inconsistent with the ledger; unresolved 2026-10-01).

## Job ledger for this campaign

See `SLURM_JOBS.md` (append-only) for every receipt. Campaign-specific rows
are added here as they complete.

| Job | Purpose | Status |
|---|---|---|
| 21400730 | Approach −x, `recovery` baseline (16 layouts) | DONE 8/16, 0 falls — reproduces 21386145 |
| 21400731 | Approach −x, `guard` (16 layouts) | DONE 0/16 — rejected |
| 21400732 | Approach −x, `pulse` detour (16 layouts) | DONE 0/16 — rejected |
| 21400745 | Approach −x, `recovery030` 0.30 m/s onset | DONE 7/16 |
| 21400746 | Approach −x, `settle14` phase shift | DONE 8/16 |
| 21400747 | Approach −x, `settle16` phase shift | DONE 9/16 — success set moves with phase |
| 21400752–54, 21400780 | Campaign A: 16 Doors + 16 Transport, chain traced | DONE Doors 3/16, Transport 1/16; `21400755` OOM, fixed, 13–15 rerun |
| 21400801–804 | In-stage arms F1, F1+F2, F1+F3, F1+F2+F3 on 16 Doors | DONE 0, 3, 0, 3 /16; falls 5→1 with F2 |
| 21400805 | Replay gate with F1 0.25 m | DONE **8/10 FAIL** — approach trips on the plate; F1 rejected |
| 21400806–807 | Campaign B: `prev_actions_reset` on 5 exposed | DONE 5/5 APPLIED; +1 −1; not promoted alone |
| 21400825–827 | In-stage arms F2, F3, F2+F3 (plate-centre lateral) | DONE 1, 1, 1 /16 — worse; wait-open standstill harms |
| 21400863 | Replay gate with `--wait-open=2.0` | DONE **10/10 PASS** |
| 21400895–896 | A1 activate-only; A2 A1 + lateral 0.35 | DONE 2/16, 0/16 — A2 clears walls, crossings find the door closed |
| 21400897 | Replay gate at lateral 0.35 | DONE **10/10 PASS** |
| 21400952 | F1+F2+F3 at 0.35 m (last predeclared arm) | DONE 1/16, falls 5→1 |
| 21400953 | Combined replay gate 0.35 m + wait-open 2.0 | DONE **10/10 PASS** |
| 21400961 | Interaction: reset + F1F2F3@0.35 on 4 exposed Doors | DONE 0/4 — withdrawn with the misplaced-target arms |
| 21401685–686 | Reset with `--stall-min-s 3.0`, unchanged stage, 5 exposures | DONE — 3 NOT_EXPOSED tick-identical (Doors/3 preserved); Doors/6 → success; Doors/1 → door 2 then fall; 1/4 → 2/4 |
| 21401689 | Replay gate V2 stage (pre 0.45, clear 0.35, kick) | DONE **7/10 FAIL** — sideways crossing when yaw is 90° off |
| 21401728–729 | Candidate 1 confirmatory, validation 16–31 (32) | DONE Doors 1/16, Transport 4/16, 3 exposures |
| 21401788–789 | Matched baseline reruns for the 3 exposed pairs | DONE — T24/T31 stall to timeout without the reset |
| 21401734 | Media render (GPU): Doors/1 stall vs crossing fix | DONE — MP4s + hashed sidecars + GIF, gallery updated |

## Confirmatory protocol (predeclared 2026-09-23, before any in-stage or Campaign B result was read)

- **Candidate selection** (development only): validation layouts 0–15. The
  frozen candidate is the in-stage arm with the highest Doors route success on
  `21400801–804`, combined with `prev_actions_reset` only if Campaign B shows
  it APPLIED with ≥1 matched improvement and 0 matched regressions among the
  exposed episodes. Ties break toward fewer factors.
- **Regression gate:** the frozen candidate's PlateStage change must keep the
  exact ten-fall replay at 10/10 upright (`21400805` for F1); F2/F3 act in the
  live route controller and are outside the scripted replay by construction.
- **Evaluation set:** validation layouts **16–31**, never used by any run so
  far, both stages: 32 candidate episodes and 32 matched baseline episodes
  (unchanged controller, same early handoff), 64 total, run under the hardware
  constraint with the same source snapshot.
- **Primary metric:** route success (paired by stage/layout). Secondary: falls,
  in-stage collisions, post-exit falls, stall exposures. Denominator is every
  episode; nothing is excluded.
- **Precision:** with n = 32 per arm, a paired exact test on discordant pairs
  is reported with a 95 % Clopper–Pearson interval on each arm's rate; only a
  difference whose interval excludes zero is called an improvement. If the
  discordant-pair count is under 6 the result is reported as underpowered, not
  as null.
- **Stopping rule:** one evaluation. No tuning against layouts 16–31 and no
  second look; the test split (seeds 20000+) is untouched and reserved for its
  documented final use.
