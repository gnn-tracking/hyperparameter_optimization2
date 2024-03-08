#!/usr/bin/env bash

#SBATCH --job-name=pixel-gc
#SBATCH --output=slurm_logs/pixel-gc-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python3 run_gc.py fit $@
