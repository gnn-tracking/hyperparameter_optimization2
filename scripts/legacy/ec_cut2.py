#!/usr/bin/env python3
import sys
from pathlib import Path

from gnn_tracking.graph_construction.data_transformer import DataTransformer, ECCut
from gnn_tracking.models.edge_classifier import ECFromChkpt

if __name__ == "__main__":
    part = sys.argv[1]
    batch_num = sys.argv[2]

    ec = ECFromChkpt(
        chkpt_path="/home/kl5675/Documents/23/git_sync/hyperparameter_optimization2/scripts/legacy/lightning_logs/dexterous-earthworm-of-satiation/checkpoints_persist/epoch=42-step=309600.ckpt",
        device="cpu",
    )
    ecc = ECCut(
        ec,
        thld=0.03,
    )
    gcc = DataTransformer(
        transform=ecc,
    )
    gcc.process_directories(
        input_dirs=[
            Path(
                f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v7/part_{part}/"
            )
        ],
        output_dirs=[
            Path(
                f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v7_cut/part_{part}/"
            )
        ],
        redo=False,
        n_files=500 * 32,
        start=500 * 32 * int(batch_num),
    )
