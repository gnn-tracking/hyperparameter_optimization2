#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True
$RAISE_SUBPROC_ERROR = True

import itertools
import random
import numpy as np

lws = np.linspace(0.3, 0.6, 10)

configs = lws
random.shuffle(configs)

run = "miniature-ermine-of-chaos"
chkpt = "epoch=28-step=208800.ckpt"
log_dir = p"/home/kl5675/Documents/23/git_sync/hyperparameter_optimization2/scripts/legacy/lightning_logs/"
chkpt_path = log_dir / "checkpoints" / chkpt
config_path = log_dir / "repeat_config.yaml"

assert log_dir.exists()
assert chkpt_path.exists()
assert config_path.exists()

for lw in configs:
    python3 run_oc.py fit \
        --config @(config_path) \
        --config configs/config.yml \
        --ckpt_path=@(chkpt_path) \
        --model.init_args.lw_repulsive=@(lw) \
        --model.init_args.optimizer.init_args.lr=0.0002 \
        --model.init_args.scheduler=none
