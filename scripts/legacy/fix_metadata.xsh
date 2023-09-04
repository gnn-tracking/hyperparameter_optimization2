#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True
search_str=$ARG1
results=$(find .  -path  ./lightning_logs/*@(search_str)*/hparams.yaml).strip()

print(results)

if len(results.split()) != 1:
    raise ValueError("Search string bad")

wandb_results=$(find . -path ./wandb/*@(search_str)*/files/config.yaml)

print(wandb_results)

if len(wandb_results.split()) != 1:
    raise ValueError("Search string bad (wandb")

cat @(results.strip()) >> @(wandb_results.strip())
