import torch
from gnn_tracking.training.callbacks import PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from hpo2.lightning_utils import ContinueTrainingCLI

name = random_trial_name()


tb_logger = TensorBoardLogger(".", version=name)


def cli_main():
    torch.set_float32_matmul_precision("medium")

    # noinspection PyUnusedLocal
    cli = ContinueTrainingCLI(  # noqa: F841
        datamodule_class=TrackingDataModule,
        trainer_defaults={
            "callbacks": [
                RichProgressBar(leave=True),
                PrintValidationMetrics(),
            ],
            "logger": [tb_logger],
        },
        compile_model=True,
    )


if __name__ == "__main__":
    cli_main()
