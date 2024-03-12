import torch
from gnn_tracking.training.callbacks import PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger

name = random_trial_name()


tb_logger = TensorBoardLogger(".", version=name)


def cli_main():
    torch.set_float32_matmul_precision("medium")

    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa: F841
        datamodule_class=TrackingDataModule,
        trainer_defaults={
            "callbacks": [
                RichProgressBar(leave=True),
                PrintValidationMetrics(),
                ModelCheckpoint(
                    dirpath="/scratch/gpfs/kl5675/checkpoints/animation",
                    every_n_train_steps=50,
                    save_top_k=-1,
                ),
                LearningRateMonitor(logging_interval="step", log_momentum=True),
            ],
            "logger": [tb_logger],
            "plugins": [SLURMEnvironment(auto_requeue=False)],
            "max_time": "23:30:00",
            "max_epochs": 50,
        },
    )


if __name__ == "__main__":
    cli_main()
