#!/usr/bin/env python3

"""Currently it's not possible to rename groups in wandb from the wandb
web client. This script does it with the API.
"""

import argparse


def rename_group(
    *, api, user: str, project: str, old_group: str, new_group: str
) -> None:
    runs = api.runs(f"{user}/{project}", filters={"group": old_group})
    for run in runs:
        run.group = new_group
        run.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--old-group", type=str, required=True)
    parser.add_argument("--new-group", type=str, required=True)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    rename_group(api=api, **vars(args))
