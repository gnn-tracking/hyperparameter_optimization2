#!/usr/bin/env xonsh

for part in range(1, 10):
    for batch in range(0, 4):
        sbatch submit_ec_cut.slurm @(part) @(batch)
