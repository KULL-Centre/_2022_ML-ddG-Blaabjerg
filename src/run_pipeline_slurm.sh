#!/bin/bash
#SBATCH --job-name=rasp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G

python3 -u run_pipeline.py
