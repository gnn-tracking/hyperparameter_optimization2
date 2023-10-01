#!/usr/bin/env python3

from pathlib import Path

from gnn_tracking.utils.log import logger

local_bin = Path("~/.local/bin").expanduser()
assert local_bin.is_dir()

scripts = list(Path("bin").iterdir())
logger.info(f"Installing {len(scripts)} scripts")

for script in scripts:
    logger.debug(f"Processing {script=}, {script.stem=}")
    link = local_bin / script.stem
    if link.is_symlink():
        logger.debug(f"Removing {link=}")
        link.unlink()
    logger.debug(f"Linking {link=} to {script=}")
    link.symlink_to(script.resolve())
