from gnn_tracking.utils.loading import TrackingDataModule
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

logger = WandbLogger(
    project="lst_oc",
    group="first",
    offline=True,
)


tb_logger = TensorBoardLogger(".")


def cli_main():
    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa F841
        datamodule_class=TrackingDataModule,
        trainer_defaults=dict(
            callbacks=[RichProgressBar(leave=True)],
            logger=[tb_logger, logger],
        ),
    )


if __name__ == "__main__":
    cli_main()
