"""Like run_oc.py but without any logging to wandb & co."""

import torch
from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
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
                ExpandWandbConfig(),
                EarlyStopping(
                    monitor="trk.double_majority_pt0.9", mode="max", patience=10
                ),
                ModelCheckpoint(
                    save_top_k=2, monitor="trk.double_majority_pt0.9", mode="max"
                ),
            ],
        },
    )


if __name__ == "__main__":
    cli_main()
