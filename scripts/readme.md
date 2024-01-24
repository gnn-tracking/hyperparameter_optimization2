# Submission scripts and more

## Prerequisites

- Set up your wandb API key in your home directory
- Create your own wandb project

## Workflow

### Testing a script/running script interactively

Testing a new submission is best with an interactive node. First grab an
interactive node, for example:

```bash
salloc --nodes=1 --ntasks=1 --time=01:00:00 --gres=gpu:1 --cpus-per-task=3 --mem=150GB --constraint=gpu80
```

Next, you need one of the python scripts `run_*.py`. If you have one that ends
with the `_test.py` suffix, even better (it will have some things like
checkpoints and more disabled). For example, `pixel/run_gc_test.py`.

> [!warning] Currently, all wandb settings are hard-coded in the `.py` files.
> Make sure to select the right project and group there (though we can always
> fix things later).

Then, you need the right config files to set everything up:

1. A data config (usually `configs/data.yml`) -- this has all settings that go
   into the lightning `DataModule`
2. A model config (e.g., `configs/gc/model.yml`) -- this includes everything
   that goes into the `LightningModule`
3. Extra config options, in particular settings for `LightningTrainer`, usually
   `configs/config.yml`

Your final command (from the `pixel` directory) will be

```bash
python3 run_gc_test.py fit --data config/data.yml --model configs/gc/model.yml --config configs/config.yml
```

Note the `fit` command after `run_gc_test.py`.

### Additional settings

You can override config items from the command line, for example

```bash
python3 run_gc_test.py --data config/data.yml --model configs/gc/model.yml --config configs/config.yml \
  --trainer.num_sanity_val_steps=0 --model.init_args.loss_fct.init_args.lw_repulsive=0 \
  --model.init_args.optimizer.init_args.lr=0.0001
```

To continue training, you can specify a `ckpt_path`, for example

```bash
run_oc_test.py fit --config configs/config.yml --data configs/data.yml --model configs/oc/model_gc_loss_from_pretrained_01.yml --ckpt_path lightning_logs/organic-invisible-reindeer/persitent-checkpoints/epoch=197-step=89100.ckpt
```

### Batch submission

Once you have verified that everything runs properly, we can do the same thing
with batch submission, for example:

```bash
sbatch submit_oc.slurm --config configs/config.yml --data configs/data.yml --model configs/oc/model_gc_loss.yml
# or more complicated
sbatch submit_oc.slurm --config lightning_logs/fanatic-righteous-stork/repeat_config.yaml \
  --ckpt_path lightning_logs/fanatic-righteous-stork/checkpoints/epoch=193-step=87300.ckpt \
  --model.init_args.optimizer.init_args.lr=0.00007 --model.init_args.loss_fct.init_args.lw_repulsive=0.05
```

## Checkpoints and wandb

All checkpoints are in your `lightning_logs` directory.
