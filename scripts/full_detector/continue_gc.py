from __future__ import annotations

import torch
from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.training.ml import MLModule
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.nomenclature import random_trial_name
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

name = random_trial_name()


wandb_logger = WandbLogger(
    project="gnn_tracking_gc",
    group="full-detector",
    offline=True,
    version=name,
    tags=["full-detector", "continued"],
)


tb_logger = TensorBoardLogger(".", version=name)


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Want weights only, not optimizer states etc.
        parser.add_argument("--ckpt", type=str, required=True)

    def fit(self, model: MLModule, **kwargs):
        ckpt = self.config["fit"]["ckpt"]
        logger.debug(f"Loading state dict from {ckpt}")
        ckpt_loaded = torch.load(ckpt, map_location=model.device)
        model.load_state_dict(ckpt_loaded["state_dict"])
        logger.debug("Compiling model")
        model.model = torch.compile(model.model)
        self.trainer.fit_loop.epoch_progress.load_state_dict(
            ckpt_loaded["loops"]["fit_loop"]["epoch_progress"]
        )
        logger.debug(f"Start epoch set to {self.trainer.current_epoch}")
        self.trainer.fit(model, **kwargs)


def cli_main():
    torch.set_float32_matmul_precision("medium")

    # noinspection PyUnusedLocal
    cli = CLI(  # noqa: F841
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
    )


if __name__ == "__main__":
    cli_main()
