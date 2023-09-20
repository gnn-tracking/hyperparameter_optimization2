#!/usr/bin/env python3

from pathlib import Path

local_bin = Path("~/.local/bin").expanduser()
assert local_bin.is_dir()

scripts = Path("bin").glob("*.xsh")
print(f"Installing {len(scripts)} scripts")  # noqa: T201

for script in scripts:
    link = local_bin / script.name
    if link.is_symlink():
        link.unlink()
    script.symlink_to(link)
