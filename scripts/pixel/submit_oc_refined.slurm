#!/usr/bin/env bash

#SBATCH --job-name=pixel-oc-refine
#SBATCH --output=slurm_logs/pixel-oc-refine-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=25G
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu
#SBATCH --constraint=gpu80

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun \
  python3 run_oc.py fit\
  --model configs/oc_refined/model.yml \
  --trainer configs/oc/train.yml \
  --data configs/oc/data.yml \
  $@
