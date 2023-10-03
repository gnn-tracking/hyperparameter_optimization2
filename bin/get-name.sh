#!/usr/bin/env bash

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

if [[ $# -eq 0 ]]; then
  jobID=$(squeue -u "$USER" -o "%A" -h|fzf)
else
  jobID=$1
fi
# xargs to strip whitespace
# shellcheck disable=SC2086
head -n 1 slurm_logs/*${jobID}*.log|sed 's/─//g'|xargs
