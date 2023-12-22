import torch
from gnn_tracking.training.callbacks import PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from pytorch_lightning.callbacks import RichProgressBar

from hpo2.lightning_utils import TorchCompileCLI


def cli_main():
    torch.set_float32_matmul_precision("medium")

    # noinspection PyUnusedLocal
    cli = TorchCompileCLI(  # noqa: F841
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
