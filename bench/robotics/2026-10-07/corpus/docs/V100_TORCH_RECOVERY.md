# DGX2 V100 training compatibility, 2026-09-19

Job `21359477` reached the real LeRobot fine-tuning path on
`Tesla V100-SXM3-32GB` and failed on a CUDA cast with `no kernel image is
available for execution on the device`. The existing training environment's
`torch==2.7.0+cu128` reports the compiled architectures `sm_75 sm_80 sm_86
sm_90 sm_100 sm_120 compute_120`. V100 requires `sm_70`.

This is a binary compatibility failure, not a cloth-policy failure. Setting
`TORCH_CUDA_ARCH_LIST` when running a prebuilt wheel cannot add missing kernels.
The [tagged PyTorch 2.7 build script](https://github.com/pytorch/pytorch/blob/v2.7.0/.ci/manywheel/build_cuda.sh#L49)
explicitly includes Volta in its CUDA 12.6 build and removes it from CUDA 12.8.
The [official installation archive](https://pytorch.org/get-started/previous-versions/#v270)
publishes the matching Torch 2.7.0 / torchvision 0.22.0 CUDA 12.6 wheels.

The user's `depth-env` is not a drop-in substitute: its locally queried
Torch 2.10.0+cu128 binary includes `sm_70`, but it uses Python 3.10 and is outside
LeRobot 0.4.3's declared `torch>=2.2.1,<2.8.0` constraint. The training
environment uses Python 3.11. It should remain unchanged.

`scripts/bench/torch_cu126.py install` builds a separate overlay at
`results/weekend-20260919/deps/torch-cu126`. It installs only the matching
Torch/torchvision pair and the exact NVIDIA/Triton dependencies declared by
that Torch wheel. General Python/LeRobot dependencies remain those of the
existing training environment. Downloads use the official PyTorch index.
Installation writes to a fresh staging directory and refuses to replace an
existing target; a failed staging directory is retained for diagnosis.

Use the existing container launcher with `stack=train`:

```bash
# CPU allocation is sufficient for installation and compiled-architecture check.
slurm/inner/weekend_torch126.sh install

# Must run in a V100 GPU allocation before submitting long DGX2 training jobs.
slurm/inner/weekend_torch126.sh smoke

# Run a two-step real fine-tuning/save smoke with the isolated overlay.
slurm/inner/weekend_torch126.sh run slurm/inner/weekend_fold.sh \
  train 0 2 smoke-v100-cu126 --smoke --val-per-episode 1
```

The wrapper prepends the overlay to `PYTHONPATH` only for that process. CUDA
smoke checks the selected Torch path, versions, compiled architectures, real
convolution forward/backward, AdamW, attention, checkpoint save/load, and
SmolVLA import, then prints `TORCH_CU126_CUDA_PASS`. This is an infrastructure
gate; the real fine-tuning and simulator evaluation remain separate requirements.
CPU import alone is not evidence that kernels executed on V100.

This overlay is for offline policy training, not Isaac Sim rendering. V100's
CUDA training compatibility does not establish renderer compatibility.
