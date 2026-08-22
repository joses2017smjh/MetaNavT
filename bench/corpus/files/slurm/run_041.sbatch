#!/bin/bash
#SBATCH --job-name=run_041
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
python train.py --config configs/run_041.yaml --num_pairs=2 --encoder resnet50
