#!/bin/bash
#SBATCH --job-name=metanavit-http-cpu
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT/bench/results/revamp_20261007/http-cpu-%j.log
set -euo pipefail
TASK_REPO=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT
TASK_PY=/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311gpu/bin/python
export PYTHONPATH=/nfs/stak/users/sanchej7/hpc-share/mnt_work/resume-m2m3-deps:$TASK_REPO:/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311app/lib/python3.11/site-packages
export HF_HOME=/nfs/stak/users/sanchej7/hpc-share/mnt_work/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BGE_ALLOW_DOWNLOAD=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/tmp
cd "$TASK_REPO"
"$TASK_PY" scripts/revamp_http_service.py --container /nfs/stak/users/sanchej7/hpc-share/containers/pgvector-pg16-resume.sif \
    --data bench/corpus/files --queries bench/gold/questions.jsonl --device cpu --precision fp32 --requests 16 \
    --out-dir bench/results/revamp_20261007/http_cpu_fp32_512_paired16
