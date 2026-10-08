# Weekend robotics campaign — 19 September 2026

**September 21 completed Approach diagnosis:** the privileged controller
reached **54/64 successes with zero falls**, but none of six PPO cells passed
the stable-learning gate. P4 ended at 2/16; P6 briefly reached 5/16 then fell
to 0/16. Exact paired replays implicated pressure-plate contact in ten route
falls. All 18 retained Slurm tasks completed; one invalid replay was cancelled
and replaced. **11.96 allocated CPU-hours, zero GPUs.** Sensor and full-campaign
gates remain closed. [Results, plots, jobs and next steps](MISSION7_APPROACH_DEBUG.md).

This campaign repairs the prior experiments, uses measured progress gates,
and adds a real shared-world task for two and three Berkeley Humanoid Lites.
Scheduler receipts and source hashes are in
`results/weekend-20260919/submissions.jsonl`; live job identities and failures
are maintained in [SLURM_JOBS.md](../SLURM_JOBS.md).

**September 21 — separate Mission7 follow-up:** all 14 overnight tasks finished
with exit 0; nine training studies and two audits executed, while three sensor
pilots skipped because their learning prerequisite failed. All nine policies
ended at 0/8 validation success, including privileged-goal PPO. Scripted
full-route Doors and Transport each succeeded 4/16. These new procedural
22-DoF results must not be conflated with the earlier fixed-maze recovery
results below. [Final Mission7 audit and next steps](MISSION7_OVERNIGHT.md).

| Track | Running implementation | Promotion / evidence | Capability boundary |
| --- | --- | --- | --- |
| Learned maze recovery | 12-DoF biped in Isaac Lab; Approach → Corridor → Full; blind, lidar, paired depth, both; three seeds | Sept 20: all 36 stage jobs completed and passed; Full first-episode successes 379/384 (98.70%) | Known route, fixed layout, observation noise disabled for evaluation; not an RGB/SSD navigation policy |
| Two-turn inspection maze | Full 22-DoF humanoid, 6.4 m planned route, two ordered inspection dwells, dead-end distractor and exit in MuJoCo | 3/3 nominal completions in 17.24–17.44 s, zero contacts/falls; 0/3 complete-outage and 0/3 wrong-branch completions | Frozen learned gait with oracle map/pose and actual sensor braking; inspection is proximity dwell, not visual recognition |
| Folding | SmolVLA RGB + joint-state policy, dual SO-101 grippers and GPU particle cloth; two adaptation seeds | Both trainings complete; evaluations: 5 complete, 2 failed, 5 timed out. Short pants: baseline 8/24 vs adapted seed 1 3/24 | No demonstrated adaptation improvement; official ever-triggered scorer, not independent terminal-fold validation; not humanoid folding |
| Cooperative mission | Two/three 22-DoF humanoids share inspection stations, a latched airlock and ordered exits in MuJoCo; frozen Isaac-trained locomotion | Five matched seeds per crew; coordinated, no-wait and withheld-teammate controls; contacts checked every physics substep | This establishes team navigation, not carrying or a newly trained MARL policy |
| Sensor extension | Body-attached MuJoCo lidar, paired idealized depths, gyro/accelerometer/attitude; timestamped input affects collision braking | Sensor-reactive and dropout runs, same mission; packets and changed commands recorded | Global route/pose remains privileged; ray depth is not stereo matching; SSD is not silently replaced with oracle boxes |
| Real-image perception component | CPU calibrated StereoSGBM matching plus actual SSDLite predictions, metric depth and validity/provenance checks | Known-disparity and command-line roundtrip tests; required calibration and original capture timestamps | Separate from trained navigation; real-device accuracy and station detector still unmeasured |
| Cloth physics diagnostic | MuJoCo deformable towel with two idealized picker attachments | 4/4 released folds, 0/4 untouched controls, all finite with zero solver warnings | No humanoid or gripper is present in this diagnostic |

## Findings that changed the jobs

The twelve previous maze runs reached timeout without reaching their goal.
The recovery tasks correct an extra timestep factor on progress reward, a
dwell counter updated by both reward and termination callbacks, the success
termination penalty, and missing braking. Success now requires consecutive
upright, slow samples. Existing task IDs and historical checkpoints retain
their original meaning. [Maze details and research](MAZE_RIG.md).

The initial September 19 seed-0 Approach gate recorded **32/32 blind, 31/32
lidar, 31/32 paired depth, 30/32 both** and released matching Corridor jobs.
The **September 20 audit** confirms all 36 stage jobs completed with exit code
`0:0`, all 36 task gates passed, and all checkpoint files exist. The arrays are
absent from the live queue because they finished, not because they were lost.

Full-stage first-episode success, with 32 environments per training seed:

| Arm | Seed 0 | Seed 1 | Seed 2 | Total |
|---|---:|---:|---:|---:|
| Blind | 32/32 | 29/32 | 32/32 | 93/96 (96.875%) |
| Lidar | 32/32 | 32/32 | 32/32 | 96/96 (100%) |
| Paired ray depth (`Stereo`) | 32/32 | 32/32 | 30/32 | 94/96 (97.917%) |
| Both | 32/32 | 32/32 | 32/32 | 96/96 (100%) |

This establishes reliable oracle-guided arrivals on the fixed corridor, not
sensor-only maze solving or a sensor advantage. Evaluation disables observation
corruption; these scores do not measure IMU-noise robustness. The Isaac runs
train the **12-DoF biped**. The inspection/team videos below instead use the
**22-DoF humanoid with its frozen August 18 gait**, not these newly trained
maze checkpoints. [Dated results, controls and artifact links](WEEKEND_RESULTS_2026-09-20.md).

Cloth sorting and cloth folding are different projects here. Stock BHL has no
gripper joints, and its Newton free-base cloth scene becomes nonfinite even
without manipulation. The folding work instead uses the existing nearby
`lehome-fold-repro` assets and pretrained SmolVLA, without editing that sibling.
Frame-level train/validation leakage and failed-replay action supervision are
removed from the new adaptation. A deeper audit found **23,250 swallowed
camera errors** in historical folding job `21214241`; its recorded 6/24 fold
events are not a validated closed-loop visual-policy baseline. New evaluation
uses separate per-garment USD layers and fails on rendering errors or missing
fresh frames. [Folding details](CLOTH_FOLDING_WEEKEND.md).

Both 1,500-update folding adaptations completed: job `21359522` on H100 and
`21359530` on V100. The strict camera gate passed; all 12 class-evaluation
tasks have now ended: **5 completed, 2 failed, 5 timed out**. Completed short-pants
evaluations score baseline **8/24** and adaptation seed 1 **3/24**, both 0/4
Unseen. New adaptation has not demonstrated an improvement. Short-top scorer
indices mismatch one mesh; other jobs stalled during garment switches.
[Full table, validity and blocked-media diagnosis](CLOTH_FOLDING_WEEKEND.md#measured-status-20-september-2026).

The cooperation benchmark has **5/5 clean coordinated completions for each
crew size**. Both negative controls score **0/5 for each crew size**. These are
small fixed-layout evaluations, not a population reliability guarantee. The
two robots typically finish in about 23 s and the three in about 31 s. Each
robot must inspect its own station; all must wait together before the door
unlocks, take turns crossing, and rendezvous at separate exit stations.
[Task and research](TEAM_AIRLOCK.md).

The expanded [inspection maze](INSPECTION_MAZE.md) establishes a feasible
moderate-difficulty layout using the existing full-body gait: two changes of
travel direction, two ordered stops, a dead end, and an exit. Its first clean
completion received 174 range packets; those measurements reduced commanded
translation on 27 control steps. The torso keeps its initial heading and
sidesteps on the northbound leg. It does not claim an untested heading-turn
skill. [Rendered maze](../results/weekend-20260919/inspection-maze.mp4) and
[three-robot mission](../results/weekend-20260919/team3-airlock.mp4) are actual
simulator rollouts, not illustrations.

## Sensor and detector boundary

The supplied product title exactly matches the vendor listing for
[Hiwonder IM10A](https://www.hiwonder.com/products/imu-module); this is a probable
identification, pending the device label/link. Noise levels in the new maze
are explicitly simulation stress assumptions. They
are not a calibration of that device. `sensor_io.py` uses ROS xyzw attitude,
angular rates in rad/s and specific force in m/s². It rejects stale, missing
and numerically invalid data. Vendor axis convention, gravity convention,
sample rate, mounting transform and covariance need the actual datasheet and
stationary/moving captures. Magnetometer and barometer channels advertised
for IM10A are not silently added to the policy's IMU vector.
[Verified vendor limits and physical-input checklist](IMU_INPUT.md): the
factory stream is 10 Hz, not the advertised maximum 200 Hz; actual driver
orientation, frame and cadence must be checked before use.

`scripts/perception/ssd_input.py` runs an actual SSDLite model and packs its
boxes, classes and scores into fixed input slots, including registered depth
and validity masks. It can use an existing checkpoint; `--weights coco`
explicitly downloads official torchvision weights. Standard COCO classes do
not include a custom maze button. It does not create fake detections if a
model sees nothing. Depth is optical-axis depth, not Euclidean slant range.
This adapter is implemented but is not yet connected to the trained maze
policy; detector accuracy and hardware calibration remain unmeasured.
An actual COCO-weight inference on the rendered maze frame completed in
0.149 s on CPU and produced zero accepted detections
(`ssd-render-smoke.json`). That verifies execution, not station recognition;
custom simulator stations are outside the pretrained categories.

`scripts/perception/stereo_depth.py` computes actual image-pair correspondence,
with left/right consistency and texture rejection, optical-axis metric depth,
and explicit invalid pixels. It requires measured/exported calibration and
original synchronized capture timestamps. The SSD adapter verifies image
identity and depth provenance. [Stereo usage and limits](STEREO_INPUT.md).

The next perception-policy stage replaces privileged station knowledge with
verified SSD station classes and replaces ray depth with estimated stereo
depth. It must retain ordered task completion, collision/fall checks and
timing measures. Current robots receive no unsupported hand-manipulation
requirement. New terrain should be introduced only below measured stepping
limits. The straight-corridor recovery has now passed. A learned expanded-maze
campaign still needs its own feasibility and held-out-layout gates; the
existing expanded-maze evaluation is a CPU reference with oracle planning.

The sensor interfaces follow the [ROS IMU conventions](https://www.ros.org/reps/rep-0145.html)
and [torchvision SSD output contract](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html).

## Research choices

Primary sources were checked through September 19, including
[PASSAGE (September 16)](https://arxiv.org/abs/2609.18732),
[TANGO (September 8)](https://arxiv.org/abs/2609.09158),
[HumanoidVLN (August 13)](https://arxiv.org/abs/2608.12860),
[FolDeX (September 9)](https://arxiv.org/abs/2609.10243),
[Learning to Fold / LeHome 2026](https://arxiv.org/abs/2606.27163),
[Rhythm (revised September 10)](https://arxiv.org/abs/2603.02856), and
[Marope](https://arxiv.org/abs/2606.08064).

The implemented choices are research-informed: validated motor skills below
task planning, explicit synchronization for teams, whole-garment holdouts,
and physics-scored closed-loop folding. These papers' G1 controllers, human
motion datasets and released-task results are not claimed as reproduced on
BHL. In particular, large-model or motion-retargeting training is not a
substitute for fixing unreachable geometry and incorrect reward metrics.

## Hardware and queue discipline

H100/H200 handle folding adaptation and compatible headless policy training.
Rendering stays on the known working RTX8000/A40 route. DGX2 V100s need a
separate CUDA 12.6 Torch binary: the existing CUDA 12.8 binary excludes SM70.
An isolated overlay in the campaign directory preserves the existing shared
venvs; it passes real CUDA convolution, optimizer, attention, save/load and
two-step SmolVLA checks before releasing its full training job.
[V100 details](V100_TORCH_RECOVERY.md).

CPU MuJoCo checks reserve no GPUs. Each maze array permits four concurrent
cells; each folding-evaluation array permits one. The cluster's per-user GPU
limit can still hold jobs despite idle GPUs elsewhere. Existing user jobs,
interactive allocations and held historical arrays are not cancelled or
released. The two idle GPUs already allocated to the current desktop can
perform short verification and rendering without requesting another GPU.

The report artifact
[`SUMMARY.md`](../results/weekend-20260919/SUMMARY.md) has been generated; the
September 20 audit read its 15:23 UTC snapshot. Job `21359631` was configured
to refresh it after the final training and folding-evaluation arrays finish,
including failures. Artifact availability alone does not establish that every
evaluation succeeded or that the report job itself has exited. Refresh at any time
with `python scripts/summarize_weekend.py --slurm`. Local validation after the
stereo/inspection additions: **145 tests passed**.

Each full maze cell depends on its matching curriculum cell via `aftercorr`;
failed task gates prevent longer runs. Folding evaluations depend on both
the strict camera smoke and their training checkpoint. Slurm success alone
is insufficient: maze scripts require a JSON verdict and PASS sentinel;
folding writes complete official episode records with physics/camera checks.

Submission helper defaults to a dry run. For example:

```bash
python scripts/submit_weekend_job.py --name my-check --cpu -- \
  scripts/cloth/fold_mujoco.py --episodes 4 --out results/my-check.json
```

Add `--submit` only for an intended new run. Existing receipts should be
checked first to avoid duplicate campaigns. No old results are overwritten.
