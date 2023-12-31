#!/usr/bin/env python3
import argparse
from pathlib import Path

import wandb
import wandb.errors
import yaml
from gnn_tracking.utils.log import logger
from pyfzf.pyfzf import FzfPrompt

"""Fix metadata for wandb.
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
        "-p",
        "--project",
        help="Wandb project",
        required=True,
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
    """Find exactly one config.yaml file for project_name."""
    glob_str = f"./lightning_logs/*{project_name}*/config.yaml"
    return find_exactly_one_file(glob_str)


def main():
    api = wandb.Api()
    args = cli().parse_args()
    if not args.names:
        fzf = FzfPrompt()
        available_paths = Path("lightning_logs").iterdir()
        args.names = fzf.prompt([path.name for path in available_paths], "--multi")
    for name in args.names:
        hparams = yaml.safe_load(find_hparams(name).read_text())["model"]["init_args"]
        logger.debug(f"{name=} with {hparams=}")
        try:
            run = api.run(f"gnn_tracking/{args.project}/{name}")
        except (ValueError, wandb.errors.CommError):
            logger.error(f"Run {name} not found")
            continue
        run.config |= hparams
        run.update()


if __name__ == "__main__":
    main()
