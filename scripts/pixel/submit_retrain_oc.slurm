#!/usr/bin/env bash

#SBATCH --job-name=pixel-oc
#SBATCH --output=slurm_logs/pixel-oc-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=25G
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu
#SBATCH --signal=SIGUSR1@90
#SBATCH --constraint=gpu80

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python3 retrain_oc.py fit $@
