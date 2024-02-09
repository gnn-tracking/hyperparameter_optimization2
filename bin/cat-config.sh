#!/usr/bin/env bash

# Cat configuration of run
# No args: select run interactively.

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

# shellcheck disable=SC2012
# (find instead of ls not necessary here)
name=$(ls lightning_logs|fzf)
less "lightning_logs/${name}/config.yaml"
