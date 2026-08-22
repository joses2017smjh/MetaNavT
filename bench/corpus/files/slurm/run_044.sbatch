#!/bin/bash
#SBATCH --job-name=run_044
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
python train.py --config configs/run_044.yaml --num_pairs=3 --encoder clip
