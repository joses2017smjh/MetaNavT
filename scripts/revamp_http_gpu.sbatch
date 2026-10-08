#!/bin/bash
#SBATCH --job-name=metanavit-http-gpu
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint=h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT/bench/results/revamp_20261007/http-gpu-%j.log
set -euo pipefail
TASK_REPO=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT
TASK_PY=/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311gpu/bin/python
export PYTHONPATH=/nfs/stak/users/sanchej7/hpc-share/mnt_work/resume-m2m3-deps:$TASK_REPO:/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311app/lib/python3.11/site-packages
export HF_HOME=/nfs/stak/users/sanchej7/hpc-share/mnt_work/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BGE_ALLOW_DOWNLOAD=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/tmp
cd "$TASK_REPO"
# Complete paired HTTP workload: same fixture, questions, fp32, 512-token cap,
# depth20 and16requests/level as CPU. Device/host differ, recorded explicitly.
"$TASK_PY" scripts/revamp_http_service.py --container /nfs/stak/users/sanchej7/hpc-share/containers/pgvector-pg16-resume.sif \
    --data bench/corpus/files --queries bench/gold/questions.jsonl --device cuda --precision fp32 --requests 16 \
    --out-dir bench/results/revamp_20261007/http_gpu_fp32_512_paired16
# Isolate truncation while retaining fp32, candidate depth and every test query.
"$TASK_PY" -m app.eval.beir --dataset nfcorpus --device cuda --only hybrid+bge-rerank@bge-small,hybrid+bge-rerank@bge-small/fp32-512/top20 \
    --n-boot 1000 --require-trec-eval --trec-dir bench/results/revamp_20261007/trec_nfcorpus_cap \
    --out bench/results/revamp_20261007/beir_nfcorpus_cap_gpu.json
# Additional real-corpus scale measurement. The questions are generated scalar
# lookups; this is service throughput, not human answer-quality evaluation.
"$TASK_PY" scripts/revamp_http_service.py --container /nfs/stak/users/sanchej7/hpc-share/containers/pgvector-pg16-resume.sif \
    --data bench/robotics/2026-10-07/corpus --queries bench/robotics/2026-10-07/questions.jsonl --device cuda --precision fp16 --requests 16 \
    --out-dir bench/results/revamp_20261007/http_robotics_gpu_fp16_512
