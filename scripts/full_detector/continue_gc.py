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
    tags=["full-detector"],
)


tb_logger = TensorBoardLogger(".", version=name)


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Want weights only, not optimizer states etc.
        parser.add_argument("--ckpt", type=str, required=True)

    @staticmethod
    def _get_epoch_from_ckpt_name(name: str) -> int:
        epoch_str = name.split("-")[0]
        return int(epoch_str.split("=")[1])

    def fit(self, model: MLModule, **kwargs):
        logger.debug("Compiling model")
        model.model = torch.compile(model.model)
        logger.debug(f"Loading state dict from {self.config['ckpt']}")
        state_dict = torch.load(self.config["ckpt"], map_location=model.device)[
            "state_dict"
        ]
        model.load_state_dict(state_dict)
        self.trainer.current_epoch = self._get_epoch_from_ckpt_name(self.config["ckpt"])
        logger.debug(f"Setting start epoch to {self.trainer.current_epoch}")
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
                ModelCheckpoint(save_top_k=2, monitor="total", mode="max"),
            ],
            "logger": [tb_logger, wandb_logger],
            "plugins": [SLURMEnvironment()],
        },
    )


if __name__ == "__main__":
    cli_main()
