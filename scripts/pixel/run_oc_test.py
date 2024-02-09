"""Like run_oc.py but without any logging to wandb & co."""

import torch
from gnn_tracking.training.callbacks import PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.cli import LightningCLI


def cli_main():
    torch.set_float32_matmul_precision("medium")

    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa: F841
        datamodule_class=TrackingDataModule,
        trainer_defaults={
            "callbacks": [
                RichProgressBar(leave=True),
                PrintValidationMetrics(),
            ],
        },
    )


if __name__ == "__main__":
    cli_main()
