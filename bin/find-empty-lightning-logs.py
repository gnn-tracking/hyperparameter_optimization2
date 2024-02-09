#!/usr/bin/env python

"""Find empty lightning logs to delete them."""

# Print statements ok
# ruff: noqa: T201

from pathlib import Path

for dir in Path().iterdir():
    if not dir.is_dir():
        continue
    checkpoints_dir = Path(dir) / "checkpoints"
    if not checkpoints_dir.exists():
        print(dir)
        continue
    checkpoints = list(checkpoints_dir.glob("*.ckpt"))
    if not checkpoints:
        print(dir)
        continue
