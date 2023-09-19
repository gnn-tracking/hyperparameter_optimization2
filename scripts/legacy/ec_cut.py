#!/usr/bin/env python3
import sys
from pathlib import Path

from gnn_tracking.graph_construction.data_transformer import DataTransformer, ECCut
from gnn_tracking.models.edge_classifier import ECFromChkpt

if __name__ == "__main__":
    part = sys.argv[1]
    batch_num = sys.argv[2]

    ec = ECFromChkpt(
        chkpt_path="/home/kl5675/Documents/23/git_sync/hyperparameter_optimization2/scripts/legacy/lightning_logs/adorable-dalmatian-of-rain/checkpoints/epoch=19-step=144000.ckpt",
        device="cpu",
    )
    ecc = ECCut(
        ec,
        thld=0.3,
    )
    gcc = DataTransformer(
        transform=ecc,
    )
    gcc.process_directories(
        input_dirs=[
            Path(
                f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v6/part_{part}/"
            )
        ],
        output_dirs=[
            Path(
                f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v6_cut/part_{part}/"
            )
        ],
        redo=False,
        n_files=300,
        start=300 * int(batch_num),
    )
