from __future__ import annotations

import torch
import wandb
from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

name = random_trial_name()


logger = WandbLogger(
    project="gnn_tracking_ec",
    group="proceedings-legacy",
    offline=True,
    version=name,
    tags=["geometric-gc"],
)

wandb.define_metric(
    "max_mcc_pt0.9",
    step_metric="max_mcc_pt0.9",
    summary="max",
)

tb_logger = TensorBoardLogger(save_dir=".", version=name)


def cli_main():
    torch.set_float32_matmul_precision("medium")

    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa: F841
        datamodule_class=TrackingDataModule,
        trainer_defaults={
            "callbacks": [
                RichProgressBar(leave=True),
                TriggerWandbSyncLightningCallback(),
                PrintValidationMetrics(),
                ExpandWandbConfig(),
                ModelCheckpoint(save_top_k=2, monitor="max_mcc_pt0.9", mode="max"),
            ],
            "logger": [tb_logger, logger],
            "plugins": [SLURMEnvironment()],
            "gradient_clip_val": 0.5,
        },
        seed_everything_default=42,
    )


if __name__ == "__main__":
    cli_main()
