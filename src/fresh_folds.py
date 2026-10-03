"""Re-check a selected model on fresh fold splits (TODO 5.5).

A model picked among many runs on the frozen folds is flattered by that choice. Here the
reference and the candidate are scored again on fold splits that were never used for any
choice, and compared fold by fold.

Usage:
    uv run python src/fresh_folds.py <reference_experiment.yaml> <candidate_experiment.yaml>

Nothing is logged in reports/runs.csv (other folds), and data/processed/folds.parquet is untouched.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import load_config
from cv import run_cv
from data import TARGET, load

# Seeds of the fresh splits: arbitrary, only required to differ from fold_seed (42) and from
# each other. 3 repeats x 5 folds = 15 paired comparisons (the "3 x 5" of TODO 5.5).
FRESH_SEEDS = [1, 2, 3]
ROOT = Path(__file__).resolve().parents[1]


def fresh_folds(seed: int, n_folds: int) -> pd.DataFrame:
    train, _ = load()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = pd.DataFrame({"id": train["id"], "fold": -1})
    for k, (_, valid_idx) in enumerate(skf.split(train, train[TARGET])):
        folds.loc[valid_idx, "fold"] = k
    return folds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    ref_cfg, cand_cfg = load_config(args.reference), load_config(args.candidate)
    assert ref_cfg["fold_seed"] not in FRESH_SEEDS, "fresh seeds must differ from the frozen fold seed"

    diffs, rows, per_fold = [], [], []
    for seed in FRESH_SEEDS:
        folds = fresh_folds(seed, ref_cfg["n_folds"])
        print(f"=== fresh split, seed {seed}: reference", flush=True)
        ref_auc = np.array(run_cv(ref_cfg, folds)[2])
        print(f"=== fresh split, seed {seed}: candidate", flush=True)
        cand_auc = np.array(run_cv(cand_cfg, folds)[2])
        d = cand_auc - ref_auc
        diffs.extend(d)
        rows.append({"seed": seed, "reference": ref_auc.mean(), "candidate": cand_auc.mean(),
                     "gain": d.mean(), "folds up": int((d > 0).sum())})
        per_fold += [{"seed": seed, "fold": k, "reference_auc": a, "candidate_auc": b} for k, (a, b) in enumerate(zip(ref_auc, cand_auc))]

    # Saved so the result can be read and cited later (these runs are not in reports/runs.csv).
    out = ROOT / "reports" / "tuning" / f"fresh_{Path(args.candidate).stem}_vs_{Path(args.reference).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_fold).to_csv(out, index=False)
    print(f"per-fold results written to {out.relative_to(ROOT)}")

    d = np.array(diffs)
    k, repeats = ref_cfg["n_folds"], len(FRESH_SEEDS)
    # Nadeau & Bengio corrected variance for r x k-fold: factor 1/(r k) + n_test/n_train, n_test/n_train = 1/(k-1).
    se = d.std(ddof=1) * np.sqrt(1 / (repeats * k) + 1 / (k - 1))
    print("\n" + pd.DataFrame(rows).round(5).to_string(index=False))
    print(f"\nall {len(d)} paired folds: gain {d.mean():+.5f}, corrected SE {se:.5f} -> {d.mean() / se:+.1f} SE, "
          f"{int((d > 0).sum())} of {len(d)} folds up")
    print(f"verdict: {'CONFIRMED' if d.mean() > 0 and d.mean() >= 2 * se else 'NOT CONFIRMED'}"
          "  (same rule: gain > 0 and >= 2 x corrected SE)")
