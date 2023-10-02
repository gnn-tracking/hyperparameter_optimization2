from __future__ import annotations

import torch
from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from hpo2.lightning_utils import ContinueTrainingCLI

name = random_trial_name()


wandb_logger = WandbLogger(
    project="gnn_tracking_gc",
    group="full-detector",
    offline=True,
    version=name,
    tags=["full-detector", "continued"],
)


tb_logger = TensorBoardLogger(".", version=name)


def cli_main():
    torch.set_float32_matmul_precision("medium")

    # noinspection PyUnusedLocal
    cli = ContinueTrainingCLI(  # noqa: F841
        datamodule_class=TrackingDataModule,
        trainer_defaults={
            "callbacks": [
                RichProgressBar(leave=True),
                TriggerWandbSyncLightningCallback(),
                PrintValidationMetrics(),
                ExpandWandbConfig(),
                EarlyStopping(monitor="total", mode="min", patience=10),
                ModelCheckpoint(save_top_k=2, monitor="total", mode="min"),
            ],
            "logger": [tb_logger, wandb_logger],
            "plugins": [SLURMEnvironment()],
        },
        compile_model=True,
    )


if __name__ == "__main__":
    cli_main()