from functools import partial

import torch
from gnn_tracking.metrics.losses import BackgroundLoss, PotentialLoss
from gnn_tracking.postprocessing.dbscanscanner import DBSCANHyperParamScanner
from gnn_tracking.training.tc import TCModule
from gnn_tracking_hpo.orchestrate import maybe_run_distributed
from pytorch_lightning.loggers import WandbLogger
from ray import air, tune
from ray.air import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import LightningConfigBuilder, LightningTrainer
from ray.tune.schedulers import ASHAScheduler

from lstcondensation.loader import default_data_module
from lstcondensation.model import LSGraphTCN

logger = WandbLogger(
    project="lst_oc",
    group="first",
    offline=True,
)
#
# logger = WandbLoggerCallback(
#     project="lst_oc",
#     group="first",
#     offline=False,
# )

maybe_run_distributed()


def n_trials(epoch: int) -> int:
    # if epoch < 10:
    #     return 0
    if epoch % 3 == 0:
        return 1
    else:
        return 0


class TunableModel(TCModule):
    def __init__(self, config):
        model = LSGraphTCN(
            node_indim=config.get("node_indim", 9),
            edge_indim=config.get("edge_indim", 3),
            h_dim=config.get("h_dim", 128),
            e_dim=config.get("e_dim", 128),
            h_outdim=config.get("h_outdim", 12),
            L_hc=config.get("L_hc", 3),
        )
        super().__init__(
            model=model,
            potential_loss=PotentialLoss(
                radius_threshold=1.0,
            ),
            background_loss=BackgroundLoss(),
            lw_repulsive=config.get("lw_repulsive", 1.0),
            lw_background=config.get("lw_background", 0.1),
            optimizer=partial(torch.optim.Adam, lr=config.get("lr", 7.5e-4)),
            cluster_scanner=DBSCANHyperParamScanner(
                n_trials=n_trials, n_jobs=3, min_samples_range=(1, 1)
            ),
        )


config = {
    "node_indim": 9,
    "edge_indim": 3,
    "h_dim": 128,
    "e_dim": 128,
    "h_outdim": 12,
    "L_hc": 3,
    "lw_repulsive": tune.uniform(0.1, 2.0),
    "lw_background": 0.1,
    "lr": 7.5e-4,
}

lightning_config = (
    LightningConfigBuilder()
    .module(cls=TunableModel, config=config)
    .trainer(max_epochs=10, accelerator="gpu", logger=logger)
    .fit_params(datamodule=default_data_module)
    .checkpointing(monitor="trk.double_majority_pt0.9", save_top_k=2, mode="max")
    .build()
)

scaling_config = ScalingConfig(
    num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
)

# Make sure to also define an AIR CheckpointConfig here
# to properly save checkpoints in AIR format.
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="trk.double_majority_pt0.9",
        checkpoint_score_order="max",
    ),
    # callbacks=[logger],
)


# Define a base LightningTrainer without hyper-parameters for Tuner
lightning_trainer = LightningTrainer(
    scaling_config=scaling_config,
    run_config=run_config,
)


scheduler = ASHAScheduler(max_t=3, grace_period=1, reduction_factor=2)

tuner = tune.Tuner(
    lightning_trainer,
    param_space={"lightning_config": lightning_config},
    tune_config=tune.TuneConfig(
        metric="trk.double_majority_pt0.9",
        mode="max",
        num_samples=30,
        scheduler=scheduler,
    ),
    run_config=air.RunConfig(
        name="tune_mnist_asha",
    ),
)
results = tuner.fit()
