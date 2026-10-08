#!/bin/bash
#SBATCH --job-name=metanavit-beir-cpu
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT/bench/results/revamp_20261007/beir-cpu-%j.log
set -euo pipefail
TASK_REPO=/nfs/stak/users/sanchej7/hpc-share/MetaNavT/MetaNaviT
TASK_PY=/nfs/stak/users/sanchej7/hpc-share/mnt_work/venvs/venv311app/bin/python
export PYTHONPATH=/nfs/stak/users/sanchej7/hpc-share/mnt_work/resume-m2m3-deps:$TASK_REPO
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/tmp
cd "$TASK_REPO"
for dataset in scifact nfcorpus fiqa; do
    "$TASK_PY" -m app.eval.beir --dataset "$dataset" --device cpu --only bm25 --n-boot 1000 \
        --require-trec-eval --trec-dir "bench/results/revamp_20261007/trec_${dataset}_cpu" \
        --out "bench/results/revamp_20261007/beir_${dataset}_cpu.json"
done
