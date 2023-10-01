#!/usr/bin/env python3

"""This script installs all scripts from `bin/` to `~/.local/bin/`."""

from pathlib import Path

from gnn_tracking.utils.log import logger


def main():
    local_bin = Path("~/.local/bin").expanduser()
    logger.info(f"Installing scripts to {local_bin=}")
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


if __name__ == "__main__":
    main()
