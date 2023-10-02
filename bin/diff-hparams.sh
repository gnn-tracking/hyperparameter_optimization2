#!/usr/bin/env bash

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

# shellcheck disable=SC2012
# (find instead of ls not necessary here)
name1=$(ls lightning_logs|fzf)
# shellcheck disable=SC2012
name2=$(ls lightning_logs|fzf)
echo "${name1} vs ${name2}"
diff "lightning_logs/${name1}/config.yaml" "lightning_logs/${name2}/config.yaml"
