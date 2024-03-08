import torch
import wandb
from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

name = random_trial_name()


logger = WandbLogger(
    project="gnn_tracking_fd",
    group="gc-loss-legacy-norm",
    offline=True,
    version=name,
    tags=["gc:garrulous-peach-manatee", "no-ec", "gc-loss", "gc-loss-legacy-norm"],
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
                EarlyStopping(
                    monitor="trk.perfect_pt0.9",
                    mode="max",
                    patience=20,
                    verbose=True,
                ),
                ModelCheckpoint(
                    save_top_k=2,
                    monitor="trk.perfect_pt0.9",
                    mode="max",
                    verbose=True,
                ),
                LearningRateMonitor(logging_interval="step", log_momentum=True),
            ],
            "logger": [tb_logger, logger],
            "plugins": [SLURMEnvironment(auto_requeue=False)],
            "max_time": "0:23:30:00",
        },
    )


if __name__ == "__main__":
    cli_main()
