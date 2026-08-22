#!/bin/bash
#SBATCH --job-name=run_050
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
python train.py --config configs/run_050.yaml --num_pairs=3 --encoder resnet50
