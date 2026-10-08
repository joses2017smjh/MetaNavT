# Getting Isaac Lab to render a policy to video

Six attempts over three days produced nothing; the seventh produced two mp4s.
The renderer was never the problem. Every failure was in the policy-loading path
of `play.py`, before the rollout that draws a frame — and the job reported
`COMPLETED` for most of them.

This is the checklist that got it working, in the order the failures appear.

## 0. Assume the exit code is lying

Isaac's `train.py` and `play.py` exit **0 on failure**: Hydra catches the
exception and `simulation_app.close()` hard-exits before Python can set a
status. Four dead runs reported `COMPLETED` here.

Never gate on the exit code. Gate on the artefact:

```bash
marker=$(mktemp)
"$PY" scripts/train_play.py --task "$TASK" --video --video_length 300 ... || true
n=$(find "$LOGROOT" -name '*.mp4' -newer "$marker" | wc -l)
[ "$n" -gt 0 ] || { echo "FAILED: no video written" >&2; exit 1; }
```

Count files **newer than a marker**, not total files. `RecordVideo` writes to
`<run>/videos/play/rl-video-step-0.mp4` and *overwrites* it, so a re-render
leaves a total count unchanged — a guard that counted totals here reported "no
video was written" about a 78 MB file with a fresh mtime, which is worse than no
guard because it sends you debugging a success.

## 1. Run from the directory the log root is relative to

`play.py` builds its log root as the **relative** path `logs/rsl_rl/<experiment>`.
If training ran from a different working directory, playback looks in the wrong
place and raises `FileNotFoundError` — reported as success, per §0.

```bash
cd "$UPSTREAM"     # wherever training ran from, not wherever the repo is
```

## 2. Migrate the agent config to the installed rsl-rl

`RslRlMLPModelCfg.to_dict()` still emits `stochastic`, `init_noise_std`,
`state_dependent_std` and `noise_std_type`. rsl-rl ≥ 5.0 splats that dict into
`MLPModel.__init__`, which accepts none of them:

```
TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'
```

Isaac Lab ships the migration and its own scripts call it. `play.py` may not:

```python
try:
    from importlib.metadata import version as _pkg_version
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
except ImportError:
    pass          # older stack, nothing to migrate
```

Guard it, so an older rsl-rl where the helper does not exist is untouched.

**This one is worth checking first if training works and playback does not** —
`train.py` had the migration and `play.py` did not, so every checkpoint trained
fine and not one could be replayed.

## 3. The actor got renamed

rsl-rl 3.x exposes it as `alg.policy`, 5.x as `alg.actor`:

```python
actor = getattr(ppo_runner.alg, "policy", None) or ppo_runner.alg.actor
```

## 4. `get_observations()` changed arity

3.x returned `(obs, extras)`; 5.x returns the `TensorDict` alone. **Unpacking the
5.x return into two names does not raise** — a TensorDict with `policy` and
`critic` groups unpacks into those two *key strings*, so the policy is silently
handed the string `"policy"` and fails somewhere unrelated.

```python
_obs = env.get_observations()
obs = _obs[0] if isinstance(_obs, tuple) else _obs
```

## 5. Make every export best-effort

Between the checkpoint load and the rollout, `play.py` exports ONNX/JIT and a
deployment YAML. Both are **sim2real artefacts, not preconditions for replay**,
and both assume the task they were written for — a single articulation named
`robot`, a `joint_pos` action term, a velocity command:

```
AttributeError: 'CoopLiftSceneCfg' object has no attribute 'robot'
ValueError: Policy does not have an actor/student module.
```

Wrap both in `try/except`, print a warning, continue. A task that cannot produce
a deployment config still deserves to be recorded.

Wrap the debug banner too. `print_dict` pretty-prints callables via
`callable_to_string`, which reads the lambda's source line and does
`.split("lambda")[1]` — so a line number that resolves elsewhere raises
`IndexError` before a frame exists. A formatting helper should not decide
whether a video gets made.

## 6. Frame the camera on one environment

The default viewer holds every environment in shot. On a 4-env replay that is
four robots as specks on a dark grid floor — correct output, useless as a clip.

```python
env_cfg.viewer.origin_type = "env"     # anchor on one env's origin
env_cfg.viewer.env_index  = 0
env_cfg.viewer.eye        = (1.7, 1.7, 1.05)   # metres from that robot
env_cfg.viewer.lookat     = (0.0, 0.0, 0.35)
```

**Use colons, not commas, if these come from environment variables.** Both
`sbatch --export` and Apptainer's `--env` split their arguments on commas, so
`"2.4,2.4,1.5"` arrives as three broken assignments.

## 7. Flags and cluster notes

```bash
"$PY" scripts/train_play.py \
    --task "$TASK" --num_envs 4 --headless --enable_cameras \
    --video --video_length 300 --load_run "$RUN_NAME"
```

`--enable_cameras` is required for `--video`; without it the render product is
never created. On a headless node you will see
`Failed to create NGX context` and `Could not get NGX parameters block` — those
are DLSS warnings and are **not** fatal. The signal that rendering is actually
working is:

```
[INFO]: RTX streaming completed in 0.04 s.
```

Isaac Sim **5.1's RTX renderer segfaults** against driver 595/610; 6.0 works, and
no node constraint helps. If you are pinned to 5.1, render elsewhere.

## 8. The output is huge and noisy

1280×720 × 16 s came out at 77–88 MB, and the path-traced floor grain destroys
GIF compression — a naive conversion was 25 MB. Crop to the subject and denoise:

```bash
V="crop=620:350:300:70,hqdn3d=16:12:24:18,fps=6,scale=440:-1:flags=lanczos"
ffmpeg -i in.mp4 -vf "$V,palettegen=max_colors=48:stats_mode=diff" pal.png
ffmpeg -i in.mp4 -i pal.png \
  -lavfi "$V[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle" out.gif
```

That gets 25 MB to 5.8 MB. Keep the mp4 out of git.

## 9. Check the first frame before believing it

A render that runs is not a render that shows anything. Pull frames and look:

```bash
ffmpeg -ss 5 -i out.mp4 -frames:v 1 frame.png
```

Doing this here is what surfaced the spawn bug: the robots were lying in the
floor, which no log line said and no metric caught — `fallen` read 0.0000
throughout. See [FINDINGS](FINDINGS.md).

## 10. No robot in the clip: the viewport draws a stale pose

Symptom: the clip shows the scene — terrain, props, even the command arrows —
and no robot, or a robot that never moves from its spawn. Every maze clip in
`21247917` looked like this.

Measured, not guessed (`21299608`, `scripts/bench/render_probe.py`): all of the
robot's meshes load and compute `inherited` visibility, so it is not the asset.
It is *which pose gets drawn*. With `use_fabric=True`, physics writes
articulation poses to Fabric and USD keeps the spawn pose — the maze robot's
USD base sat at (0, 0, 0) while physics had it at (−27.9, −76.4). The viewport
render product, which is what `RecordVideo` records through `env.render()`,
draws the body at that stale pose. On a flat task that is roughly where the
robot is, which is why the TaskV2 clips always showed one. On a terrain task
the robot is reset onto a terrain origin tens of metres from its env's grid
origin, and a camera aimed at it films empty ground. `--disable_fabric` did not
fix it in the probe either: the body was drawn back at the grid origin.

**A camera sensor draws the body where physics has it**, fabric on. So record
through one:

```bash
BHL_CAMERA_CLIP=1 BHL_CLIP_DIR=/path/to/frames \
BHL_VIEW_EYE="1.8:1.8:1.1" BHL_VIEW_LOOKAT="0:0:0.3" \
"$PY" scripts/train_play.py --task "$TASK" --num_envs 1 --headless \
    --enable_cameras --video_length 400 --load_run "$RUN"
ffmpeg -framerate 25 -i /path/to/frames/frame_%04d.png ...     # then section 8
```

`train_play.py` mounts one `CameraCfg` per env, aims it at that env's first
articulation root every step (eye and look-at are world-frame offsets from the
root), and writes PNGs — no `--video`, no `RecordVideo`. Gate on the frame
count, per section 0. `slurm/inner/maze_video.sh` is the worked example.

## 12. Why every Isaac clip is speckled, measured (2026-09-24)

The speckle is not the scene and not the GIF quantisation. On a frame of the
stock maze clip the spatial grain (mean |f − 3×3 box blur of f|, 0–255) is
**36.1**, against **0.4** for a MuJoCo frame; consecutive frames of a static
camera differ by 47 on the same scale.

Cause, from the experience files this stack runs: `isaaclab.python.headless.rendering.kit`
sets `rtx.directLighting.sampledLighting.enabled = true` with
`samplesPerPixel = 1` and leaves the clean-up to DLSS
(`rtx.post.dlss.execMode = 0`; `balanced.kit` would add the DL denoiser). Both
of those are NGX features, and NGX fails to initialise on these nodes
(`Failed to create NGX context`, section 7). The raw 1-spp samples are what
reaches the render product.

The readback from inside a recording job (`21408633`) names the mode: this
stack's default is **`/rtx/rendermode = RealTimePathTracing`** — a stochastic
real-time path tracer at 1 spp whose clean-up is DLSS Ray Reconstruction, an
NGX feature. Without NGX the raw samples are the image. That is why toggling
the sampled-lighting terms (still reported `true` afterwards; the kit preset
wins over `RenderCfg.carb_settings`) and adding TAA changed nothing.

What did not fix it: turning the stochastic terms off (sampled direct
lighting, AO, indirect diffuse, reflections) via `RenderCfg.carb_settings`
measured **34.05** (job `21408622`, `maze_record.py --render-quality clean`),
so the noise is not those terms alone. The profiles that replace the missing
temporal pass — `taa` (TAA instead of DLSS) and `pathtrace` (16 spp through the
OptiX denoiser, which does not need NGX) — are measured against the same bar
(≤ 12) by `21408633` / `21408634`; the ledger carries the outcome. Outcome: **`pathtrace` meets the bar — spatial grain 0.32, temporal 0.00**
(`21408634`); `taa` measured 34.16 and `rtl` 34.17 — the request for
`RaytracedLighting` is not honoured on this stack (the readback still says
`RealTimePathTracing`), so the classic real-time mode is not available as a
fix here. The working recipe, applied both through `RenderCfg.carb_settings`
and directly into carb right after `AppLauncher`:

```
/rtx/rendermode = PathTracing
/rtx/pathtracing/spp = 16, /rtx/pathtracing/totalSpp = 16, /rtx/pathtracing/clampSpp = 16
/rtx/pathtracing/optixDenoiser/enabled = true, /rtx/pathtracing/optixDenoiser/blendFactor = 0
```

Cost on a DGX H100 node: a 161-step episode with a 1280×720 overhead camera
plus the two ray-cast eyes rendered and composed in about seven minutes of
job time. Section 8's nlmeans pass is no longer needed for new clips; it stays
documented for the ones already published.

