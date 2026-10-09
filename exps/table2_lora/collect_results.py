#!/usr/bin/env python
"""Summarise the Table 2 LoRA runs of one or more sweeps.

    python exps/table2_lora/collect_results.py checkpoints/table2_lora/sst2.wd0.1
    python exps/table2_lora/collect_results.py checkpoints/table2_lora/*.wd*

For every sweep directory (one task, one weight decay, a seed* subdirectory per
seed) this prints the dev accuracy of each finished seed, the median and the
mean +- sample std over seeds (the paper reports the median of five runs), the
training time, and the fraction of parameters LoRA tunes.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import statistics
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


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


def fmt_time(seconds):
    return f"{seconds / 3600:.2f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


def summarise(sweep_dir, count_params):
    print(f"\n== {sweep_dir}")
    rows = []
    run_dirs = glob.glob(os.path.join(sweep_dir, "seed*"))
    for run_dir in sorted(run_dirs, key=lambda d: int(d.rsplit("seed", 1)[-1])):
        seed = run_dir.rsplit("seed", 1)[-1]
        eval_path = os.path.join(run_dir, "eval_results.json")
        if not os.path.exists(eval_path):
            print(f"  seed {seed}: not finished (no eval_results.json)")
            continue
        metrics = load_json(eval_path)
        row = {"seed": seed, "acc": 100 * metrics["eval_accuracy"], "dir": run_dir}
        mm_path = os.path.join(run_dir, "eval_mm_results.json")
        if os.path.exists(mm_path):
            row["acc_mm"] = 100 * load_json(mm_path)["eval_mm_accuracy"]
        train_path = os.path.join(run_dir, "train_results.json")
        if os.path.exists(train_path):
            row["runtime"] = load_json(train_path).get("train_runtime")
        rows.append(row)

    if not rows:
        return

    has_mm = any("acc_mm" in r for r in rows)
    header = f"  {'seed':>4}  {'acc':>6}" + (f"  {'acc-mm':>6}" if has_mm else "") + f"  {'train time':>10}"
    print(header)
    for r in rows:
        line = f"  {r['seed']:>4}  {r['acc']:6.2f}"
        if has_mm:
            line += f"  {r.get('acc_mm', float('nan')):6.2f}"
        line += f"  {fmt_time(r['runtime']) if r.get('runtime') else '':>10}"
        print(line)

    for key, label in (("acc", "dev accuracy"), ("acc_mm", "dev-mm accuracy")):
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        line = f"  {label}: median {statistics.median(vals):.2f}, mean {statistics.mean(vals):.2f}"
        if len(vals) > 1:
            line += f" +- {statistics.stdev(vals):.2f}"
        print(line + f"  (n={len(vals)})")

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
