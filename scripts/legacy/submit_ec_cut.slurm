#!/usr/bin/env bash

#SBATCH --job-name=ec-cut
#SBATCH --output=slurm_logs/ec-cut-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=30G
#SBATCH --mail-type=fail
#SBATCH --mail-user=kl5675@princeton.edu

python3 ec_cut.py $@
