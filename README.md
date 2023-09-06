# GNN Tracking Hyperparameter Optimization 2.0

<!-- SPHINX-START -->

[![Actions Status][actions-badge]][actions-link]

This repository hosts submission scripts and framework for hyperparameter
optimization of the models defined in [the main library][main library]. It is a
successor to [the original hyperparameter optimization
repository][original hpo repo] that became obsolete after the main library was
refactored to use [pytorch lightning][pylit].

## ✨ Features

- Configure your model with only [yaml][yaml] files
- Submission files for [SLURM][slurm]
- Upload your results to the [Weights & Biases][wandb] platform

## 🚀 Installation

First, follow the instructions from [the main library][main library] to set up
the mamba environment and install the main library. Then run

```bash
pip3 install --editable '.[dev,testing]'
```

for this library.

For the helper scripts, install [xonsh][]:

```bash
pip3 install xonsh
xonsh link_bin.xsh
```

## 📖 Further reads

- [wandb-osh](https://github.com/klieret/wandb-offline-sync-hook/): package to
  trigger wandb syncs on compute nodes without internet

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/gnn-tracking/hyperparameter_optimization2/workflows/CI/badge.svg
[actions-link]:             https://github.com/gnn-tracking/hyperparameter_optimization2/actions
[main library]:             https://github.com/gnn-tracking/gnn_tracking
[original hpo repo]:        https://github.com/gnn-tracking/hyperparameter_optimization
[pylit]:                    https://lightning.ai/docs/pytorch/stable/
[wandb]:                    https://wandb.ai/
[yaml]:                     https://yaml.org/
[slurm]:                    https://slurm.schedmd.com/documentation.html
[xonsh]: https://xon.sh/

<!-- prettier-ignore-end -->
