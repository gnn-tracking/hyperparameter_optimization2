import torch
import wandb
from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

name = random_trial_name()


logger = WandbLogger(
    project="gnn_tracking",
    group="no-ec-gc-loss",
    offline=True,
    version=name,
    tags=["gc:godlike-auk-of-anger"],
)

# Make sure that wandb init is called
_ = logger.experiment
wandb.define_metric(
    "max_trk.double_majority_pt0.9",
    step_metric="trk.double_majority_pt0.9",
    summary="max",
)

tb_logger = TensorBoardLogger(".", version=name)


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
                # EarlyStopping(monitor="total", mode="min", patience=50),
                ModelCheckpoint(
                    save_top_k=5, monitor="trk.double_majority_pt0.9", mode="max"
                ),
                ModelCheckpoint(
                    every_n_epochs=10,
                ),
                LearningRateMonitor(logging_interval="step", log_momentum=True),
            ],
            "logger": [tb_logger, logger],
            "plugins": [SLURMEnvironment(auto_requeue=False)],
            "max_time": "23:30:00",
        },
    )


if __name__ == "__main__":
    cli_main()
