#!/usr/bin/env bash

# Print CLI args that were used to start a run.
# No args: select run interactively.

# bash strict mode
set -euo pipefail
IFS=$'\n\t'

# shellcheck disable=SC2012
# (find instead of ls not necessary here)
name=$(ls lightning_logs|fzf)

# shellcheck disable=SC2086
less wandb/*${name}*/files/wandb-metadata.json
