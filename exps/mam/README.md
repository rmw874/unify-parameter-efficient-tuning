# MAM — SST-2 experiments

Scripts to run the MAM Adapter baseline and two ablations on SST-2 using the repository's `run_glue.py`.

## Setup

This was used with **Linux with an NVIDIA GPU** and **Python 3.9.25**. From the repository root, in your Python environment:

```
python -m pip install 'torch==2.7.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install -r exps/table2_mam/requirements.txt
```

Prepare the pretrained model and SST-2 dataset, and check that the files and environment match the recorded experiments:

```
PYTHONPATH="$PWD/src:$PWD" python exps/table2_mam/verify_setup.py prepare
```

## Run

Run a single experiment:

```
bash exps/table2_mam/run_mam.sh base 42
```

Available conditions: base, scale1, capacity_ffn24.

To skip strict input/environment checks and use standard `roberta-base` loading, prefix a run command with `VERIFY_SETUP=0` <br> (e.g. `VERIFY_SETUP=0 bash exps/table2_mam/run_mam.sh base 42`).

Run all **11 experiments** (one run at a time per GPU):

```
GPU_IDS='0 1' bash exps/table2_mam/run_all.sh plan
GPU_IDS='0 1' bash exps/table2_mam/run_all.sh run
```

Use `GPU_IDS='0'` for a single GPU. Results are saved under `checkpoints/table2_mam/<condition>/seed<seed>/`. Completed runs are skipped, and interrupted runs must be restarted from scratch after moving or deleting their incomplete output directory.
