#!/bin/bash
#SBATCH --job-name=run_052
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
python train.py --config configs/run_052.yaml --num_pairs=1 --encoder dinov2
