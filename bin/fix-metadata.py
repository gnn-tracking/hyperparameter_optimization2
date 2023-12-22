#!/usr/bin/env python3
import argparse
import pprint
import subprocess
from pathlib import Path

import yaml
from gnn_tracking.utils.log import logger
from pyfzf.pyfzf import FzfPrompt

"""Fix metadata for wandb.

To run over all files: ``ls lightning_logs|xargs fix-metadata -si``
"""


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--names",
        type=str,
        help="Project names",
        nargs="+",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--ignore-search-errors",
        action="store_true",
        help="Ignore errors from search results",
    )
    parser.add_argument(
        "-s",
        "--skip-sync-if-unchanged",
        action="store_true",
        help="Skip syncing if no changes to wandb config",
    )
    return parser


class InvalidSearchResults(ValueError):
    pass


def find_exactly_one_file(glob_str: str) -> Path:
    """Find exactly one file matching glob_str. Raises InvalidSearchResults if
    there are no or multiple results.
    """
    results = list(Path().glob(glob_str))
    logger.debug(f"{glob_str=} with {results=}")
    if (n_results := len(results)) != 1:
        msg = f"Search string {glob_str} bad, found {n_results} results"
        raise InvalidSearchResults(msg)
    return results[0]


def find_hparams(project_name: str) -> Path:
    """Find exactly one hparams.yaml file for project_name."""
    glob_str = f"./lightning_logs/*{project_name}*/hparams.yaml"
    return find_exactly_one_file(glob_str)


def find_wandb_config(project_name: str) -> Path:
    """Find exactly one wandb config file for project_name."""
    glob_str = f"./wandb/*{project_name}*/files/config.yaml"
    return find_exactly_one_file(glob_str)


def merge_configs(hparams_path: Path, wandb_config_path: Path) -> bool:
    """Merges hparams into wandb config

    Args:
        hparams_path:
        wandb_config_path:

    Returns:
        True if changes were made, false otherwise
    """
    with hparams_path.open() as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    with wandb_config_path.open() as f:
        existing_wandb_config = yaml.load(f, Loader=yaml.SafeLoader)
    new_wandb_config = hparams | existing_wandb_config
    if new_wandb_config == existing_wandb_config:
        logger.info("No changes to wandb config")
        return False
    with wandb_config_path.open("w") as f:
        yaml.dump(new_wandb_config, f)
    logger.info("Merged wandb config")
    logger.debug(pprint.pformat(new_wandb_config, indent=2))
    return True


def sync_wandb(wandb_base_dir: Path) -> None:
    """Call wandb CL utility for sync."""
    logger.info(f"Syncing wandb for {wandb_base_dir=}")
    subprocess.run(["which", "wandb"], check=True)
    subprocess.run(["wandb", "--version"], check=True)
    subprocess.run(["wandb", "sync", str(wandb_base_dir)], check=True)


def main():
    args = cli().parse_args()
    if not args.names:
        fzf = FzfPrompt()
        available_paths = Path("lightning_logs").iterdir()
        args.names = fzf.prompt([path.name for path in available_paths], "--multi")
    for name in args.names:
        logger.info(f"{name=}")
        try:
            hparams_path = find_hparams(name)
            wandb_config_path = find_wandb_config(name)
        except InvalidSearchResults as e:
            if args.ignore_search_errors:
                logger.warning(e)
                continue
            raise e
        changes = merge_configs(hparams_path, wandb_config_path)
        if changes or not args.skip_sync_if_unchanged:
            sync_wandb(wandb_config_path.parent.parent)


if __name__ == "__main__":
    main()
