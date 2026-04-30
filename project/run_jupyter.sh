#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=bigTiger
#SBATCH --job-name=jupyter

source ~/.bashrc
conda activate malware

echo "*** Starting Jupyter on: "$(hostname)
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
