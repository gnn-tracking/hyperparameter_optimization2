#!/usr/bin/env python3

"""Update summary metrics in wandb, in case they weren't properly tracked
"""

import argparse
import copy

import numpy as np
import wandb
from gnn_tracking.utils.log import logger
from tqdm import tqdm

summary_metrics_min = []
summary_metrics_max = [
    "max_frac_segment50",
    "frac100_at_max_frac_segment50",
]
for f in [80, 85, 90, 93, 95]:
    summary_metrics_min.append(f"n_edges_frac_segment50_{f}")
    summary_metrics_max.append(f"efficiency_at_segment50_{f}")
    summary_metrics_max.append(f"frac100_at_segment50_{f}")
    summary_metrics_max.append(f"frac75_at_segment50_{f}")
    summary_metrics_max.append(f"purity_at_segment50_{f}")
for pt in [0.5, 0.9, 1.5]:
    summary_metrics_max.append(f"trk.double_majority_pt{pt:.1f}")
    summary_metrics_max.append(f"trk.lhc_pt{pt:.1f}")
    summary_metrics_max.append(f"trk.perfect_pt{pt:.1f}")
summary_metrics_last = [
    "attractive",
    "repulsive",
    "attractive_train",
    "repulsive_train",
]


def update_summary_metrics(run):
    history = run.history()
    for key in summary_metrics_min:
        try:
            current = copy.copy(history[key])
            if isinstance(current, str):
                continue
            current = current.astype("float").to_numpy()
            if np.isfinite(current).any():
                new = np.nanmin(current).item()
                run.summary[key] = new
        except KeyError:
            pass
        except Exception as e:
            logger.error(key, run)
            raise e
    for key in summary_metrics_max:
        try:
            current = copy.copy(history[key])
            if isinstance(current, str):
                continue
            current = current.astype("float").to_numpy()
            if np.isfinite(current).any():
                new = np.nanmax(current).item()
                run.summary[key] = new
        except KeyError:
            pass
        except Exception as e:
            logger.error(key, run)
            raise e
    for key in summary_metrics_last:
        try:
            current = copy.copy(history[key])
            if isinstance(current, str):
                continue
            current = current.astype("float").to_numpy()
            if np.isfinite(current).any():
                new = current[np.isfinite(current)][-1].item()
                run.summary[key] = new
        except KeyError:
            pass
        except Exception as e:
            logger.error(key, run)
            raise e
    run.summary.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", type=str, required=True)
    parser.add_argument("--since", "-s", type=str, required=True)
    args = parser.parse_args()

    api = wandb.Api()

    runs = list(
        api.runs(
            f"gnn_tracking/{args.project}",
            {"$and": [{"created_at": {"$gt": args.since}}]},
        )
    )
    logger.info(f"Found {len(runs)} runs")
    for run in tqdm(runs):
        update_summary_metrics(run)
