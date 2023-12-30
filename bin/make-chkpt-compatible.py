#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from gnn_tracking.utils.log import logger


def make_compatible(path: Path) -> None:
    logger.debug("Loading checkpoint %s", path)
    ckpt = torch.load(path, map_location="cpu")
    ckpt["state_dict"] = {
        key.replace("._orig_mod", ""): value
        for key, value in ckpt["state_dict"].items()
    }
    logger.info("Saving checkpoint to %s", path)
    new_path = path.parent / (path.stem + ".compat" + path.suffix)
    torch.save(ckpt, new_path)


if __name__ == "__main__":
    make_compatible(Path(sys.argv[1]))
