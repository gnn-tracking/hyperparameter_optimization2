#!/usr/bin/env python3

"""Print config values for a given key in multiple runs that can be selected
interactively. In fzf, use Tab to select multiple runs and Enter to confirm.
"""
# ruff: noqa: T201

import argparse
from pathlib import Path
from typing import Any

import yaml
from pyfzf.pyfzf import FzfPrompt
from tabulate import tabulate


def find_key(config: dict[str, Any], key: str) -> Any:
    """Find value of key in nested dict."""
    if key in config:
        return config[key]
    for v in config.values():
        if isinstance(v, dict):
            item = find_key(v, key)
            if item is not None:
                return item
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=str, default="lw_repulsive")
    parser.add_argument("-n", "--names", nargs="+", help="Names of runs")
    args = parser.parse_args()

    available_paths = Path("lightning_logs").iterdir()
    names = args.names
    if not names:
        fzf = FzfPrompt()
        names = fzf.prompt([path.name for path in available_paths], "--multi")
    values = []
    for name in names:
        config = yaml.safe_load(Path("lightning_logs", name, "config.yaml").read_text())
        values.append(find_key(config, args.key))
    print(tabulate(zip(names, values), headers=["Name", args.key]))
    print()
    print("To repeat, run the following:")
    print("get-hparam --names", " ".join(names), f"--key {args.key}")
