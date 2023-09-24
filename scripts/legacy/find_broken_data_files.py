#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm


def test(path: Path):
    time = datetime.fromtimestamp(path.stat().st_mtime)
    if (datetime.now() - time).total_seconds() < 60 * 60:
        # Currently writing these files, so not part of the previously
        # cancelled jobs
        return
    try:
        torch.load(path)
    except:  # noqa: E722
        tqdm.write(str(path))


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1])
    for p in tqdm(list(path.rglob("*.pt"))):
        test(p)
