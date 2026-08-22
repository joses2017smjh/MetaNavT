#!/bin/bash
#SBATCH --job-name=run_048
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
python train.py --config configs/run_048.yaml --num_pairs=3 --encoder dinov2
