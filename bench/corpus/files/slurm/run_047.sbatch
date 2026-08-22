#!/bin/bash
#SBATCH --job-name=run_047
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
python train.py --config configs/run_047.yaml --num_pairs=3 --encoder dinov2
