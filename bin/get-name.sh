#!/usr/bin/env bash

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

jobID=$(squeue -u "$USER" -o "%A" -h|fzf)
# xargs to strip whitespace
# shellcheck disable=SC2086
head -n 1 slurm_logs/*${jobID}*.log|sed 's/─//g'|xargs
