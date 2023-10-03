#!/usr/bin/env bash

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

# shellcheck disable=SC2012
# (find instead of ls not necessary here)
name=$(ls lightning_logs|fzf)

# shellcheck disable=SC2086
grep -l "${name}" ./slurm_logs/*.log| tr -d -c 0-9
