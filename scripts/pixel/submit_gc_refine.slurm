#!/usr/bin/env bash

#SBATCH --job-name=pixel-gc-refine
#SBATCH --output=slurm_logs/pixel-gc-refine-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu
#SBATCH --constraint=gpu80

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun \
  python3 run_gc_refine.py fit\
  --model configs/gc_refined/model.yml \
  --data configs/data.yml \
  --config configs/config.yml \
  $@
