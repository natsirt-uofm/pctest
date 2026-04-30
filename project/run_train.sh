#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=bigTiger
#SBATCH --job-name=mlp_train
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate malware

cd ~/project
python src/train_mlp.py
