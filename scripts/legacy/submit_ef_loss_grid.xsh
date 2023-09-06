#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True
$RAISE_SUBPROC_ERROR = True

import itertools
import random

gammas = [0, 1, 4]
alphas = [0.3, 0.4, 0.5, 0.6]

configs = list(itertools.product(alphas, gammas))
random.shuffle(configs)

for alpha, gamma in configs:
    sbatch submit_ec.slurm \
        --model configs/ef/model.yml \
        --data configs/data.yml \
        --config configs/config.yml \
        --model.init_args.loss_fct.init_args.alpha=@(alpha) \
        --model.init_args.loss_fct.init_args.gamma=@(gamma) \
        --trainer.max_epochs=20 \
        --ckpt_path=lightning_logs/accurate-lyrical-elephant/checkpoints/epoch=9-step=72000.ckpt