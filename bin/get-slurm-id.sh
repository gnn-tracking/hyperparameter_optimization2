#!/usr/bin/env bash

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

# shellcheck disable=SC2086
grep -l "${1}" ./slurm_logs/*.log| tr -d -c 0-9
