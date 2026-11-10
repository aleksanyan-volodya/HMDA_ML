"""95% confidence interval on the pooled out-of-fold AUC of logged runs (TODO 13.4).

Usage:
    uv run python src/bootstrap_ci.py <run_id> [<run_id> ...]

Stratified bootstrap, as in src/compare.py: each resample keeps the observed numbers of accepted and
denied rows. It measures the noise from WHICH ROWS were scored (105,000 of them), not the variation from
retraining the models: the fresh-split checks (reports/tuning/fresh_*.csv) measure that part.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare import BOOT_SEED, ROOT, _run, weighted_auc

N_BOOT = 2000   # TODO 13.4 asks for >= 1000; 2000 halves the Monte-Carlo error of the bounds for ~1 more minute


def bootstrap_ci(run_id: str) -> dict:
    run = _run(run_id)
    oof = pd.read_parquet(ROOT / run.oof_path)
    y, p = oof.accepted.to_numpy(), oof.pred.to_numpy()
    auc = weighted_auc(y, p)
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        w = np.zeros(len(y))                 # how many times each row is drawn, used as weights
        w[pos] = np.bincount(rng.integers(0, len(pos), len(pos)), minlength=len(pos))
        w[neg] = np.bincount(rng.integers(0, len(neg), len(neg)), minlength=len(neg))
        draws[i] = auc(w)
    return {"run": run_id, "experiment": run.experiment, "pooled AUC": roc_auc_score(y, p),
            "mean of folds": run.mean_auc, "CI low": np.quantile(draws, 0.025),
            "CI high": np.quantile(draws, 0.975), "bootstrap SD": draws.std(ddof=1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    table = pd.DataFrame([bootstrap_ci(r) for r in args.runs])
    print(f"stratified bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}; pooled = all 105,000 OOF rows at once")
    print(table.round(5).to_string(index=False))
