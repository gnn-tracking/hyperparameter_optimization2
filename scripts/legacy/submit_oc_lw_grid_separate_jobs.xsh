#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True
$RAISE_SUBPROC_ERROR = True

import numpy as np
import random


lws = np.linspace(0.3, 0.65, 10)[1:]

configs = lws
random.shuffle(configs)

run = "discerning-unbiased-sponge"
chkpt = "epoch=88-step=640800.ckpt"
log_dir = p"/home/kl5675/Documents/23/git_sync/hyperparameter_optimization2/scripts/legacy/lightning_logs/"
chkpt_path = log_dir / run / "checkpoints" / chkpt
config_path = log_dir / run / "repeat_config.yaml"

assert log_dir.exists(), log_dir
assert chkpt_path.exists(), chkpt_path
assert config_path.exists(), config_path

exe = "sbatch"
if len($ARGS) > 1 and $ARG1 == "dry":
    exe = "echo"

for lw in configs:
    lw = round(lw, 2)
    print(f"Running with {lw=}")
    @(exe) submit_oc.slurm \
        --config @(config_path) \
        --data configs/data_cut_unsectored.yml \
        --ckpt_path=@(chkpt_path) \
        --model.init_args.lw_repulsive=@(lw) \
        --model.init_args.potential_loss.init_args.max_neighbors=400  \
        --data.train.sample_size=200 \
        --trainer.max_epochs=108
