#!/bin/bash
#SBATCH --job-name=metanavit-neural
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint=a40|rtx8000|l40s|h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --output=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT/bench/results/revamp_20261007/neural-gpu-%j.log
set -euo pipefail
TASK_REPO=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT
TASK_PY=/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311gpu/bin/python
export PYTHONPATH=/nfs/stak/users/sanchej7/hpc-share/mnt_work/resume-m2m3-deps:$TASK_REPO:/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311app/lib/python3.11/site-packages
export HF_HOME=/nfs/stak/users/sanchej7/hpc-share/mnt_work/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BGE_ALLOW_DOWNLOAD=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/tmp
cd "$TASK_REPO"
# Identical 512-token cap: the fp32-to-fp16 comparison changes precision only.
"$TASK_PY" -m app.eval.beir --dataset nfcorpus --device cuda --only bm25,dense@bge-small,hybrid@bge-small,hybrid+bge-rerank@bge-small/fp32-512/top20,hybrid+bge-rerank@bge-small/fp16-512/top20 \
    --n-boot 1000 --require-trec-eval --trec-dir bench/results/revamp_20261007/trec_nfcorpus_gpu \
    --out bench/results/revamp_20261007/beir_nfcorpus_gpu.json
"$TASK_PY" -m app.eval.beir --dataset fiqa --device cuda --only bm25,dense@bge-small,hybrid@bge-small \
    --n-boot 1000 --require-trec-eval --trec-dir bench/results/revamp_20261007/trec_fiqa_gpu \
    --out bench/results/revamp_20261007/beir_fiqa_gpu.json
"$TASK_PY" scripts/revamp_http_service.py --container /nfs/stak/users/sanchej7/hpc-share/containers/pgvector-pg16-resume.sif \
    --data bench/corpus/files --queries bench/gold/questions.jsonl --device cuda --precision fp16 --requests 32 \
    --out-dir bench/results/revamp_20261007/http_gpu_fp16_512
