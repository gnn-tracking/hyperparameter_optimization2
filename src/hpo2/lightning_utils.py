from __future__ import annotations

import torch
from pytorch_lightning.cli import LightningCLI


class TorchCompileCLI(LightningCLI):
    """CLI that calls `torch.compile` on the model.

    Adapted from
    https://github.com/Lightning-AI/lightning/issues/17283#issuecomment-1501890603
    """

    def fit(self, model, **kwargs):
        model.model = torch.compile(model.model)
        self.trainer.fit(model, **kwargs)
