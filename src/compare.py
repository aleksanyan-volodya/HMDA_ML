"""Compare two logged runs: is the candidate's AUC gain over the baseline real?

Usage:
    uv run python src/compare.py <baseline_run_id> <candidate_run_id>

Both runs must use the same folds (same folds_hash), so every comparison is paired:
the two models are scored on exactly the same rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from runlog import RUNS   # the run log, defined in one place

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 1000   # resamples: Monte-Carlo error of a 95% percentile bound is small next to the CI width
BOOT_SEED = 0   # arbitrary, fixed so the printed CI is reproducible


def _run(run_id: str) -> pd.Series:
    runs = pd.read_csv(RUNS)
    row = runs[runs.run_id == run_id]
    if len(row) != 1:
        raise SystemExit(f"run_id {run_id!r} not found in {RUNS.relative_to(ROOT)}")
    return row.iloc[0]


def fold_diff(base: pd.Series, cand: pd.Series) -> dict:
    d = np.array(json.loads(cand.fold_auc)) - np.array(json.loads(base.fold_auc))
    k = len(d)
    s = d.std(ddof=1)
    return {
        "mean_diff": d.mean(),
        "fold_diffs": d,
        # Assumes independent folds: too small, because fold models share (k-2)/(k-1) of their rows.
        "naive_se": s / np.sqrt(k),
        # Nadeau & Bengio (2003): variance factor 1/J + n_test/n_train, with J = k and
        # n_test/n_train = 1/(k-1) in k-fold. Derived for repeated random splits, a heuristic here.
        "corrected_se": s * np.sqrt(1 / k + 1 / (k - 1)),
    }


def weighted_auc(y: np.ndarray, p: np.ndarray):
    """AUC(w) for many weight vectors: the ranking of p is computed once, each call is O(n).
    AUC = P(score of a positive > score of a negative), ties counted 1/2 (as roc_auc_score)."""
    group = np.unique(p, return_inverse=True)[1]          # rank of each distinct score
    is_pos = y == 1
    n_groups = group.max() + 1

    def auc(w: np.ndarray) -> float:
        neg_w = np.bincount(group[~is_pos], weights=w[~is_pos], minlength=n_groups)
        neg_below = np.cumsum(neg_w) - neg_w                   # negative weight strictly below each score
        wins = neg_below[group[is_pos]] + 0.5 * neg_w[group[is_pos]]
        return float((w[is_pos] * wins).sum() / (w[is_pos].sum() * neg_w.sum()))

    return auc


def paired_bootstrap(base: pd.Series, cand: pd.Series) -> dict:
    a = pd.read_parquet(ROOT / base.oof_path)
    b = pd.read_parquet(ROOT / cand.oof_path)
    m = a.merge(b[["id", "pred"]], on="id", suffixes=("_base", "_cand"), validate="1:1")
    y = m.accepted.to_numpy()
    p_base, p_cand = m.pred_base.to_numpy(), m.pred_cand.to_numpy()
    auc_base, auc_cand = weighted_auc(y, p_base), weighted_auc(y, p_cand)
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        # Stratified: each resample keeps exactly the observed numbers of accepted and denied rows.
        # Resampling = how many times each row is drawn, used as weights; the same rows for both
        # models (paired).
        w = np.zeros(len(y))
        w[pos] = np.bincount(rng.integers(0, len(pos), len(pos)), minlength=len(pos))
        w[neg] = np.bincount(rng.integers(0, len(neg), len(neg)), minlength=len(neg))
        diffs[i] = auc_cand(w) - auc_base(w)
    return {"pooled_diff": roc_auc_score(y, p_cand) - roc_auc_score(y, p_base),
            "ci_low": np.quantile(diffs, 0.025), "ci_high": np.quantile(diffs, 0.975)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    base, cand = _run(args.baseline), _run(args.candidate)
    if base.folds_hash != cand.folds_hash:
        raise SystemExit("the two runs used different folds: a paired comparison is impossible")

    f = fold_diff(base, cand)
    b = paired_bootstrap(base, cand)
    print(f"baseline  {base.run_id}  mean AUC {base.mean_auc:.5f}  ({base.experiment})")
    print(f"candidate {cand.run_id}  mean AUC {cand.mean_auc:.5f}  ({cand.experiment})")
    print(f"per-fold diffs: {np.round(f['fold_diffs'], 5).tolist()}")
    ratio = f"{f['mean_diff'] / f['corrected_se']:+.1f}" if f["corrected_se"] > 0 else "n/a"
    print(f"mean diff {f['mean_diff']:+.5f}   naive SE {f['naive_se']:.5f}   corrected SE {f['corrected_se']:.5f}"
          f"   -> {ratio} corrected SE")
    print(f"pooled OOF diff {b['pooled_diff']:+.5f}   paired bootstrap 95% CI "
          f"[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]")
    # Decision rule (TODO 5.4): adopt iff the mean per-fold gain >= 2 x the corrected SE
    # (and is positive: with identical runs both sides are 0).
    adopt = f["mean_diff"] > 0 and f["mean_diff"] >= 2 * f["corrected_se"]
    print(f"verdict: {'ADOPT' if adopt else 'REJECT'}  (rule: mean gain >= 2 x corrected SE)")
