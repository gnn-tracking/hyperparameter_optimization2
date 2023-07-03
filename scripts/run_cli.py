import coolname
import wandb
from gnn_tracking.utils.loading import TrackingDataModule
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

name = coolname.generate_slug(3)
print("-" * 80)
print(name)
print("-" * 80)

logger = WandbLogger(
    project="lst_oc",
    group="first",
    offline=True,
    version=name,
)

wandb.define_metric(
    "max_trk.double_majority_pt0.9",
    step_metric="trk.double_majority_pt0.9",
    summary="max",
)

tb_logger = TensorBoardLogger(".", version=name)


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
