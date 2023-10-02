#!/usr/bin/env bash

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

jobID=$(squeue -u "$USER" -o "%A" -h|fzf)
# shellcheck disable=SC2086
cat slurm_logs/*$jobID*.log
