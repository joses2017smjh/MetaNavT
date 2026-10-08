# Reproduce the evidence

This is an HPC experiment repository, not a self-contained pretrained robot
package. An example command is not a clean-machine installation test.

## Public artifacts and prerequisites

The repository includes task overlays, reward repairs, sensor adapters, CPU
tests, Slurm entrypoints, per-evaluation JSON and selected recordings. Policy
weights, shared Python environments, Isaac installations, full garment assets
and raw frame caches are not vendored. Logical `artifact://` identifiers in results identify external checkpoints and
assets, not public download URLs. See [publication details](PUBLIC_EVIDENCE.md).

For CPU setup commands and dependency versions, see the [README](../README.md#quickstart-cpu-checks)
and [requirements-test.txt](../requirements-test.txt). A fresh dependency
installation remains unverified. A subsequent independent check of the final
publication passed **146 tests in 17.84 seconds**; see the
[verification note](PUBLICATION_CHECK_2026-09-20.md).

## Validation performed on September 20

```bash
PYTHONPATH=src python -m pytest -q tests
python scripts/summarize_weekend.py
python scripts/build_portfolio_media.py
```

Initialize nested upstream assets with `git submodule update --init --recursive`
before running the full suite: two tests read the pinned upstream URDF and
actuator configuration directly. Without those files, the other 144 tests pass
but those two fail with missing-file errors.

The curated publication passed **146 tests in 18.33 seconds** using the existing
Python 3.11 environment. This includes the new receipt-free reporting regression.
The two upstream fixtures were read-only links to files verified byte-for-byte
against nested assets commit `fc90fedd008b1e56a22e3c5221548d6b24f49707`, pinned
by BHL commit `984741a3623c93b0583ccfdc479f1f8b1c4d900e`. The original 145-test
audit preceded that additional regression. No simulation jobs were run for this
publication check.

Public artifact reporting requires no scheduler access. Scheduler refresh
requires local submission receipts and Slurm access. Media conversion requires ffmpeg with drawtext and fontconfig;
it refuses to overwrite its three GIFs and SHA-256 sidecars. Inspect the
committed artifacts without rebuilding. The wrong-route video is an
intentional negative control, not a failed training job or renderer crash.

| Layer | Requirements | Evidence |
|---|---|---|
| Logic/tests | Python, pytest, NumPy, PyTorch, MuJoCo and upstream imports used by tests | `tests/` |
| New maze PPO | Isaac Sim 6 / Isaac Lab 3 beta, compatible CUDA GPU, upstream robot | `slurm/inner/weekend_maze.sh`; 36 evaluated stages |
| Shared-world MuJoCo | MuJoCo, ONNX Runtime, OmegaConf, upstream controller and exported 22-DoF gait | `scripts/bench/inspection_maze.py`, `team_airlock.py` |
| Folding | Adjacent LeHome checkout/assets, SmolVLA environment, GPU PhysX, working Storm cameras | `docs/CLOTH_FOLDING_WEEKEND.md` |
| Hardware perception | Rectified calibrated cameras, original timestamps, detector weights, measured IMU mounting/units | `docs/STEREO_INPUT.md`, `docs/IMU_INPUT.md` |

For another cluster, adapt account/partitions and paths in `slurm/_env.sh`
and the batch scripts, provide compatible dependencies/assets, then pass an
integration gate before a sweep. Existing `00`, `01`, `02` entrypoints
describe the original v51 setup, not installation of the parallel v60 stack.

The new corridor score is an Isaac first-episode result with oracle waypoints,
fixed geometry and observation noise disabled. MuJoCo inspection/cooperation
clips use a different, older gait; they do not demonstrate transfer of the
new maze checkpoints. Fold rates require completed class evaluations and
valid cameras/particles. Partial files are progress, not zero-filled scores.
