from __future__ import annotations

from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

name = random_trial_name()


logger = WandbLogger(
    project="gnn_tracking_gc",
    group="sectorization-embeddings",
    offline=True,
    version=name,
)


tb_logger = TensorBoardLogger(".", version=name)


def cli_main():
    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa: F841
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
            "logger": [tb_logger, logger],
            "plugins": [SLURMEnvironment()],
        },
    )


if __name__ == "__main__":
    cli_main()
