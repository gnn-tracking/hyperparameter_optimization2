#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True
$RAISE_SUBPROC_ERROR = True

import itertools
import random
import numpy as np

lws = np.linspace(0.35, 0.6, 10)[5:]

configs = lws
random.shuffle(configs)

run = "miniature-ermine-of-chaos"
chkpt = "epoch=28-step=208800.ckpt"
log_dir = p"/home/kl5675/Documents/23/git_sync/hyperparameter_optimization2/scripts/legacy/lightning_logs/"
chkpt_path = log_dir / run / "checkpoints" / chkpt
config_path = log_dir / run / "repeat_config.yaml"

assert log_dir.exists(), log_dir
assert chkpt_path.exists(), chkpt_path
assert config_path.exists(), config_path

exe = "python3"
if len($ARGS) > 1 and $ARG1 == "dry":
    exe = "echo"

for lw in configs:
    lw = round(lw, 2)
    print(f"Running with {lw=}")
    @(exe) run_oc.py fit \
        --config @(config_path) \
        --config configs/config.yml \
        --ckpt_path=@(chkpt_path) \
        --model.init_args.lw_repulsive=@(lw) \
        --model.init_args.optimizer.init_args.lr=0.0002 \
        --trainer.max_epochs=65
