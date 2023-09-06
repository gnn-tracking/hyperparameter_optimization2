#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True
$RAISE_SUBPROC_ERROR = True

from pathlib import Path

search_str=$ARG1


glob_str=f"./lightning_logs/*{search_str}*/hparams.yaml"
results=$(find . -path  @(glob_str)).strip()

print(f"{results=}")

if (n_results := len(results.split())) != 1:
    raise ValueError(
      f"Search string {glob_str} bad, found {n_results} results"
    )

wandb_glob_str=f"./wandb/*{search_str}*/files/config.yaml"
wandb_results=$(find . -path @(wandb_glob_str)).strip()

print(f"{wandb_results=}")

if (n_results := len(wandb_results.split())) != 1:
    raise ValueError(
      f"Search string {wandb_glob_str} bad, found {n_results} results"
    )

cat @(results.strip()) >> @(wandb_results)
wandb_base_dir=Path(wandb_results).parent.parent
wandb sync @(wandb_base_dir)
