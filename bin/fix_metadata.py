#!/usr/bin/env python3
import argparse
import pprint
import subprocess
from pathlib import Path

import yaml
from gnn_tracking.utils.log import logger


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        type=str,
        help="Project names",
        nargs="+",
    )
    return parser


def find_exactly_one_file(glob_str: str) -> Path:
    results = list(Path().glob(glob_str))
    logger.debug(f"{glob_str=} with {results=}")
    if (n_results := len(results)) != 1:
        msg = f"Search string {glob_str} bad, found {n_results} results"
        raise ValueError(msg)
    return results[0]


def find_hparams(project_name: str) -> Path:
    glob_str = f"./lightning_logs/*{project_name}*/hparams.yaml"
    return find_exactly_one_file(glob_str)


def find_wandb_config(project_name: str) -> Path:
    glob_str = f"./wandb/*{project_name}*/files/config.yaml"
    return find_exactly_one_file(glob_str)


def merge_configs(hparams_path: Path, wandb_config_path: Path) -> None:
    with hparams_path.open() as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    with wandb_config_path.open() as f:
        existing_wandb_config = yaml.load(f, Loader=yaml.SafeLoader)
    new_wandb_config = hparams | existing_wandb_config
    if new_wandb_config == existing_wandb_config:
        logger.debug("No changes to wandb config")
        return
    with wandb_config_path.open("w") as f:
        yaml.dump(new_wandb_config, f)
    logger.debug("Merged wandb config:")
    logger.debug(pprint.pformat(new_wandb_config, indent=2))


def sync_wandb(wandb_base_dir: Path) -> None:
    logger.info(f"Syncing wandb for {wandb_base_dir=}")
    subprocess.run(["which", "wandb"], check=True)
    subprocess.run(["wandb", "--version"], check=True)
    subprocess.run(["wandb", "sync", str(wandb_base_dir)], check=True)


def main():
    args = cli().parse_args()
    for name in args.names:
        logger.info(f"{name=}")
        hparams_path = find_hparams(name)
        wandb_config_path = find_wandb_config(name)
        merge_configs(hparams_path, wandb_config_path)
        sync_wandb(wandb_config_path.parent.parent)


if __name__ == "__main__":
    main()
