"""Run a one-at-a-time sweep: each run changes ONE setting of an experiment, and is logged
like any other run (results/runs.csv, oof/, preds/).

Usage:
    uv run python src/tune.py configs/sweeps/<name>.yaml            # run (skips variants already logged)
    uv run python src/tune.py configs/sweeps/<name>.yaml --dry-run  # only list the variants

The summary compares every variant with the sweep's reference run, on the same folds, with
the paired per-fold difference and the corrected SE of src/compare.py.

Evidence for report section 6.3 (the LightGBM screening), not part of the final model.
"""
import argparse
import copy
import time
from pathlib import Path

import pandas as pd
import yaml

from compare import RUNS, _run, fold_diff
from config import config_hash, load_config
from cv import run_cv
from folds import folds_hash, load_folds
from runlog import log_run

ROOT = Path(__file__).resolve().parents[1]


def variants(sweep: dict):
    base = load_config(ROOT / sweep["experiment"])
    section = sweep["section"]
    # First the experiment unchanged: a repeat of the reference run. Its gain must be exactly 0;
    # anything else is run-to-run noise that every other line must be read against.
    yield "unchanged (repeat)", copy.deepcopy(base)
    for setting, values in sweep["vary"].items():
        if setting not in base[section]:
            raise KeyError(f"'{setting}' is not defined in {section} of base.yaml")
        for value in values:
            cfg = copy.deepcopy(base)
            cfg[section][setting] = value
            yield f"{setting}={value}", cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sweep = yaml.safe_load(Path(args.sweep).read_text())
    reference = _run(sweep["reference_run"])

    rows = []
    for label, cfg in variants(sweep):
        logged = pd.read_csv(RUNS)
        done = logged[logged.config_hash == config_hash(cfg)]
        if args.dry_run:
            print(f"{label:28} {'already logged as ' + done.run_id.iloc[-1] if len(done) else 'to run'}")
            continue
        if len(done):                                   # same settings already run: reuse, do not repeat
            run_id = done.run_id.iloc[-1]
            print(f"=== {label}: already logged as {run_id}", flush=True)
        else:
            print(f"=== {label}", flush=True)
            t0 = time.perf_counter()
            oof_df, test_df, fold_auc, features = run_cv(cfg)
            run_id = log_run(cfg, f"{args.sweep}#{label}", features, folds_hash(load_folds(cfg)),
                             fold_auc, time.perf_counter() - t0, oof_df, test_df)
        run = _run(run_id)
        d = fold_diff(reference, run)
        rows.append({"variant": label, "run_id": run_id, "mean AUC": run.mean_auc, "gain": d["mean_diff"],
                     "corrected SE": d["corrected_se"],
                     "gain / SE": d["mean_diff"] / d["corrected_se"] if d["corrected_se"] > 0 else float("nan")})

    if rows:
        print(f"\nreference {reference.run_id}  mean AUC {reference.mean_auc:.5f}")
        print(pd.DataFrame(rows).round({"mean AUC": 5, "gain": 5, "corrected SE": 5, "gain / SE": 1}).to_string(index=False))
