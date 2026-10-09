#!/usr/bin/env python
"""Progress and results of the Table 2 LoRA runs of one or more sweeps.

    python exps/table2_lora/collect_results.py checkpoints/table2_lora/sst2.r16.a32.wd0.1
    python exps/table2_lora/collect_results.py checkpoints/table2_lora/sst2.*

For every sweep directory (one task and setting, a seed* subdirectory per seed)
this prints, per seed, either the final dev accuracy or, while it is still
training, the current epoch, the best dev accuracy so far and an estimate of
the time left. Once seeds have finished it prints the median and the
mean +- sample std over them (the paper reports the median of five runs) and
the fraction of parameters LoRA tunes. Safe to run at any time.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Lines the training log contains, e.g.
#   [INFO|trainer.py:1185] 2026-10-09 09:40:59,844 >>   Num Epochs = 10
#   {'loss': 0.31, 'learning_rate': 9.1e-05, 'epoch': 2.38}
#   {'eval_loss': 0.21, 'eval_accuracy': 0.9381, ..., 'epoch': 3.0}
TRAIN_START_RE = re.compile(r"\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ >>\s+Num Epochs = (\d+)")
EPOCH_RE = re.compile(r"'epoch': ([0-9.]+)")
EVAL_ACC_RE = re.compile(r"'eval_accuracy': ([0-9.]+)")


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def fmt_duration(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{max(seconds, 0) / 60:.0f} min"


def tuned_fraction(run_dir):
    """Build the model from the saved config (no weights) and count the ef_ parameters."""
    try:
        sys.path.insert(0, REPO_ROOT)
        from transformers import AutoConfig, AutoModelForSequenceClassification

        config = AutoConfig.from_pretrained(os.path.join(run_dir, "config.json"))
        with contextlib.redirect_stdout(io.StringIO()):  # the fork prints a line per initialised module
            model = AutoModelForSequenceClassification.from_config(config)
        total = sum(p.numel() for p in model.parameters())
        tuned = sum(p.numel() for n, p in model.named_parameters() if "ef_" in n)
        return tuned, total - tuned  # the paper counts relative to the pretrained model
    except Exception as exc:  # noqa: BLE001 - best effort, the accuracies matter more
        print(f"  (could not count parameters: {exc})")
        return None


def expected_seeds(sweep_dir):
    """Seeds the launcher started (from pids.txt), plus any seed directory that exists."""
    seeds = set()
    pids = os.path.join(sweep_dir, "pids.txt")
    if os.path.exists(pids):
        for line in open(pids):
            if "seeds:" in line:
                seeds.update(line.split("seeds:", 1)[1].split())
    for run_dir in glob.glob(os.path.join(sweep_dir, "seed*")):
        seeds.add(run_dir.rsplit("seed", 1)[-1])
    return sorted(seeds, key=int)


def progress(run_dir):
    """One-line status of a run that has no results yet."""
    log = os.path.join(run_dir, "log.txt")
    if not os.path.exists(log):
        return "queued (waits for a free GPU slot)"
    with open(log, errors="replace") as fh:
        text = fh.read()
    if "Traceback (most recent call last)" in text:
        return f"FAILED, see {log}"
    idle = time.time() - os.path.getmtime(log)
    stale = f", no log output for {fmt_duration(idle)}" if idle > 600 else ""
    start = TRAIN_START_RE.search(text)
    epochs = EPOCH_RE.findall(text)
    if not start or not epochs:
        return "preparing data and model" + stale
    num_epochs = int(start.group(2))
    epoch = float(epochs[-1])
    accs = [float(a) for a in EVAL_ACC_RE.findall(text)]
    best = f", best dev acc so far {100 * max(accs):.2f}" if accs else ""
    elapsed = time.time() - datetime.strptime(start.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    eta = f", about {fmt_duration(elapsed * (num_epochs - epoch) / epoch)} left" if epoch > 0.02 else ""
    return f"training, epoch {epoch:.2f} of {num_epochs}{best}{eta}{stale}"


def summarise(sweep_dir, count_params):
    print(f"\n== {sweep_dir}")
    rows = []
    for seed in expected_seeds(sweep_dir):
        run_dir = os.path.join(sweep_dir, f"seed{seed}")
        eval_path = os.path.join(run_dir, "eval_results.json")
        if not os.path.exists(eval_path):
            print(f"  seed {seed:>2}: {progress(run_dir)}")
            continue
        row = {"seed": seed, "acc": 100 * load_json(eval_path)["eval_accuracy"], "dir": run_dir}
        mm_path = os.path.join(run_dir, "eval_mm_results.json")
        if os.path.exists(mm_path):
            row["acc_mm"] = 100 * load_json(mm_path)["eval_mm_accuracy"]
        train_path = os.path.join(run_dir, "train_results.json")
        if os.path.exists(train_path):
            row["runtime"] = load_json(train_path).get("train_runtime")
        rows.append(row)
        mm = f", mismatched {row['acc_mm']:.2f}" if "acc_mm" in row else ""
        took = f" (trained in {fmt_duration(row['runtime'])})" if row.get("runtime") else ""
        print(f"  seed {seed:>2}: done, dev accuracy {row['acc']:.2f}{mm}{took}")

    if not rows:
        return
    for key, label in (("acc", "dev accuracy"), ("acc_mm", "dev accuracy, mismatched")):
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        line = f"  {label}: median {statistics.median(vals):.2f}, mean {statistics.mean(vals):.2f}"
        if len(vals) > 1:
            line += f" +- {statistics.stdev(vals):.2f}"
        print(line + f"  (over {len(vals)} finished seeds)")
    if count_params:
        counts = tuned_fraction(rows[0]["dir"])
        if counts:
            tuned, total = counts
            print(f"  tuned parameters: {tuned:,} = {100 * tuned / total:.2f}% of the {total:,} pretrained parameters")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dirs", nargs="+", help="directories that contain one seed* subdirectory per run")
    parser.add_argument("--no-params", action="store_true", help="skip counting the tuned parameters")
    args = parser.parse_args()
    for sweep_dir in args.sweep_dirs:
        summarise(sweep_dir.rstrip("/"), count_params=not args.no_params)


if __name__ == "__main__":
    main()
