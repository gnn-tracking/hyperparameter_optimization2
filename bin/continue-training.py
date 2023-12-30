#!/usr/bin/env python3

import copy
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml
from gnn_tracking.utils.log import logger
from pyfzf.pyfzf import FzfPrompt


def select_slurm_script() -> str:
    available_slurm_scripts = list(Path().glob("*.slurm"))
    logger.debug("%d slurm scripts found", len(available_slurm_scripts))
    fzf = FzfPrompt()
    slurm_script = fzf.prompt([path.name for path in available_slurm_scripts])[0]
    assert Path(slurm_script).is_file()
    return slurm_script


def select_run() -> Path:
    available_paths = list(Path("lightning_logs").iterdir())
    logger.debug("%d runs found", len(available_paths))
    fzf = FzfPrompt()
    run = fzf.prompt([path.name for path in available_paths])
    result = Path("lightning_logs", run[0])
    assert result.is_dir(), result
    return result


def select_checkpoint(run: Path) -> Path:
    available_checkpoints = list((run / "checkpoints").glob("*.ckpt"))
    assert len(available_checkpoints) > 0, "No checkpoints found"
    logger.debug("%d checkpoints found", len(available_checkpoints))
    fzf = FzfPrompt()
    checkpoint = fzf.prompt([path.name for path in available_checkpoints])
    result = run / "checkpoints" / checkpoint[0]
    assert result.is_file()
    return result


def is_checkpoint_recent(checkpoint: Path) -> bool:
    last_modified_time = checkpoint.stat().st_mtime
    current_time = time.time()
    time_passed = current_time - last_modified_time
    logger.debug("Time passed since last checkpoint: %d", time_passed)
    return time_passed <= 3600


def persist_checkpoint(checkpoint: Path):
    """Copy checkpoint to "checkpoints_persist" folder in case
    it might still be deleted by lightning
    """
    persist_dir = checkpoint.parent / "checkpoints_persist"
    persist_dir.mkdir(exist_ok=True)
    persist_checkpoint = persist_dir / checkpoint.name
    shutil.copy(checkpoint, persist_checkpoint)
    logger.debug("Copied checkpoint to %s", persist_checkpoint)
    return persist_checkpoint


def create_repeat_config(run: Path) -> Path:
    """Remove bad config items"""
    config_file = run / "config.yaml"
    assert config_file.is_file()
    config = yaml.safe_load(config_file.read_text())
    repeat_config = copy.deepcopy(config)
    del repeat_config["trainer"]["logger"]
    del repeat_config["trainer"]["plugins"]
    yaml.dump(repeat_config, (run / "repeat_config.yaml").open("w"))
    assert (run / "repeat_config.yaml").is_file()
    logger.debug("Created repeat config")
    return run / "repeat_config.yaml"


if __name__ == "__main__":
    slurm_script = select_slurm_script()
    logger.debug("Selected slurm script: %s", slurm_script)
    run = select_run()
    logger.debug("Selected run: %s", run)
    checkpoint = select_checkpoint(run)
    if is_checkpoint_recent(checkpoint):
        logger.debug("Checkpoint is not recent, persisting it")
        checkpoint = persist_checkpoint(checkpoint)
    logger.debug("Selected checkpoint: %s", checkpoint)
    repeat_config = create_repeat_config(run)
    options = input("Any other CL options you want to supply")
    options_tokenized = shlex.split(options)
    cmd = [
        "sbatch",
        slurm_script,
        "--config",
        str(repeat_config.resolve()),
        "--ckpt_path",
        str(checkpoint.resolve()),
        *options_tokenized,
    ]
    logger.debug("Running command: %s", shlex.join(cmd))
    input("Press enter to continue")
    subprocess.run(cmd, check=False)
