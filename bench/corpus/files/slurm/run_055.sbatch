#!/bin/bash
#SBATCH --job-name=run_055
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
python train.py --config configs/run_055.yaml --num_pairs=3 --encoder dinov2
