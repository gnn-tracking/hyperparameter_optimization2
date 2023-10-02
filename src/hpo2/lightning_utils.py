from __future__ import annotations

import torch
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.log import logger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI


class TorchCompileCLI(LightningCLI):
    """CLI that calls `torch.compile` on the model.

    Adapted from
    https://github.com/Lightning-AI/lightning/issues/17283#issuecomment-1501890603
    """

    def fit(self, model, **kwargs):
        model.model = torch.compile(model.model)
        if model.preproc is not None:
            model.preproc = torch.compile(model.preproc)
        self.trainer.fit(model, **kwargs)


class TorchCompilePreprocCLI(LightningCLI):
    def fit(self, model, **kwargs):
        if model.preproc is not None:
            model.preproc = torch.compile(model.preproc)
        self.trainer.fit(model, **kwargs)


def load_weights_only(
    model: LightningModule,
    trainer: Trainer,
    ckpt: str,
    *,
    compile_model: bool = False,
    continue_epoch: bool = True,
):
    logger.debug(f"Loading state dict from {ckpt}")
    ckpt_loaded = torch.load(ckpt, map_location=model.device)
    model.load_state_dict(ckpt_loaded["state_dict"])
    if compile_model:
        logger.debug("Compiling model")
        model.model = torch.compile(model.model)
    if continue_epoch:
        trainer.fit_loop.epoch_progress.load_state_dict(
            ckpt_loaded["loops"]["fit_loop"]["epoch_progress"]
        )
        logger.debug(f"Start epoch set to {trainer.current_epoch}")


class ContinueTrainingCLI(LightningCLI):
    def __init__(
        self, *, compile_model: bool = False, continue_epoch: bool = True, **kwargs
    ):
        # Attrs need to be set before __init__, because it will already start processing
        self._compile_model = compile_model
        self._continue_epoch = continue_epoch
        super().__init__(**kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Want weights only, not optimizer states etc.
        parser.add_argument("--ckpt", type=str, required=True)

    def fit(self, model: TrackingModule, **kwargs):
        load_weights_only(
            model=model,
            trainer=self.trainer,
            ckpt=self.config["fit"]["ckpt"],
            compile_model=self._compile_model,
            continue_epoch=self._continue_epoch,
        )
        self.trainer.fit(model, **kwargs)
