#!/bin/bash
#SBATCH --job-name=metanavit-robotics-http
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint=h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT/bench/results/revamp_20261007/robotics-http-%j.log
set -euo pipefail
TASK_REPO=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT
TASK_PY=/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311gpu/bin/python
export PYTHONPATH=/nfs/stak/users/sanchej7/hpc-share/mnt_work/resume-m2m3-deps:$TASK_REPO:/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311app/lib/python3.11/site-packages
export HF_HOME=/nfs/stak/users/sanchej7/hpc-share/mnt_work/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BGE_ALLOW_DOWNLOAD=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/tmp
cd "$TASK_REPO"
"$TASK_PY" scripts/revamp_http_service.py --container /nfs/stak/users/sanchej7/hpc-share/containers/pgvector-pg16-resume.sif \
    --data bench/robotics/2026-10-07/corpus --queries bench/robotics/2026-10-07/questions.jsonl --device cuda --precision fp16 --requests 16 \
    --out-dir bench/results/revamp_20261007/http_robotics_gpu_fp16_512
