"""Cross-validation runner: one experiment config in, per-fold AUC + OOF (out of fold) and test predictions out.

Usage:
    uv run python src/cv.py configs/experiments/lgbm_optuna_best.yaml
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import load_config
from data import TARGET, load
from features import build
from folds import folds_hash, load_folds
from models import MODELS
from runlog import log_run
from target_encoding import encode_fold


def reweigh_weights(group: pd.Series, y: pd.Series) -> np.ndarray:
    """Kamiran and Calders (2012): w(g, y) = P(g) P(y) / P(g, y) on the rows given, so group and label
    are independent in the weighted data. The weights average exactly 1."""
    g, y = pd.Series(group.astype(str).to_numpy()), pd.Series(np.asarray(y))
    n = len(y)
    p_g, p_y = g.map(g.value_counts()) / n, y.map(y.value_counts()) / n
    p_gy = y.groupby([g, y]).transform("size") / n
    return (p_g * p_y / p_gy).to_numpy()


def run_cv(cfg: dict, folds: pd.DataFrame | None = None, coefs: list | None = None,
           only_folds: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[float], list[str]]:
    """folds: (id, fold) to use instead of the frozen folds. Only for checks on fresh splits
    (src/fresh_folds.py); such runs are not comparable with results/runs.csv and are not logged.
    coefs: if given, each fold model's coefficient table (linear models only) is appended to it.
    only_folds: score these outer folds only, for a cheap search (tune_optuna.py). The predictions returned
    are then incomplete and test predictions are skipped: such a run must not be logged."""
    if cfg["model"] not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {cfg['model']!r}")
    if cfg["reweigh"] and cfg["model"] != "lgbm":
        raise NotImplementedError("reweigh is implemented for lgbm only")
    train, test = load()
    train, test, features = build(train, test, cfg["features"])
    print(f"{len(features)} features + {len(cfg['target_encode'])} target-encoded", flush=True)
    train = train.merge(load_folds(cfg) if folds is None else folds, on="id", how="left", validate="1:1")

    oof = np.full(len(train), np.nan)
    test_pred = np.zeros(len(test))
    fold_auc = []
    for k in (range(cfg["n_folds"]) if only_folds is None else only_folds):
        is_valid = (train["fold"] == k).to_numpy()
        # Target encodings use labels, so they are computed here, inside the fold.
        fit_part, valid_part, test_k, encoded = encode_fold(train[~is_valid], train[is_valid], test,
                                                            TARGET, cfg["target_encode"])
        cols = features + encoded
        t_fold = time.perf_counter()
        print(f"fold {k}: fitting on {len(fit_part)} rows", flush=True)
        weighted = {}
        if cfg["reweigh"]:   # weights from the fit part only; the column itself need not be a feature
            weighted["weight"] = reweigh_weights(fit_part[cfg["reweigh"]], fit_part[TARGET])
            print(f"  reweigh by {cfg['reweigh']}: weights {weighted['weight'].min():.2f} to "
                  f"{weighted['weight'].max():.2f}", flush=True)
        model = MODELS[cfg["model"]](cfg).fit(fit_part[cols], fit_part[TARGET],
                                               fit_part["fold"].to_numpy(), **weighted)
        oof[is_valid] = model.predict(valid_part[cols])
        fold_auc.append(float(roc_auc_score(valid_part[TARGET], oof[is_valid])))
        extra = (f"  trees {model.n_trees_}" if hasattr(model, "n_trees_")
                 else f"  epochs {model.n_epochs_}" if hasattr(model, "n_epochs_")
                 else f"  C {model.C_}  l1_ratio {model.l1_ratio_}" if hasattr(model, "C_") else "")
        if coefs is not None and hasattr(model, "coef_table"):
            coefs.append(model.coef_table().assign(fold=k))
        print(f"fold {k}  AUC {fold_auc[-1]:.6f}{extra}  ({time.perf_counter() - t_fold:.0f}s)", flush=True)
        # Test prediction = average of the k fold models, càd exactly the models whose
        # scores we measured (a refit on all of train would be an unvalidated model).
        if only_folds is None:
            test_pred += model.predict(test_k[cols]) / cfg["n_folds"]

    assert only_folds is not None or not np.isnan(oof).any(), "some train rows never received an OOF prediction"
    oof_df = train[["id", "fold", TARGET]].assign(pred=oof)
    test_df = test[["id"]].assign(pred=test_pred)
    return oof_df, test_df, fold_auc, cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--no-log", action="store_true", help="debug only: nothing is written")
    args = parser.parse_args()
    cfg = load_config(args.experiment)
    t0 = time.perf_counter()
    coefs = []
    oof_df, test_df, fold_auc, features = run_cv(cfg, coefs=coefs)
    wall = time.perf_counter() - t0
    auc = np.array(fold_auc)
    # SE = std / sqrt(k). Known to understate uncertainty (folds share training rows);
    # the corrected version comes with TODO 5.3.
    print("fold AUC:", np.round(auc, 6).tolist())
    print(f"mean {auc.mean():.6f}  naive SE (lower bound) {auc.std(ddof=1) / np.sqrt(len(auc)):.6f}")
    print(f"oof rows {len(oof_df)}  test rows {len(test_df)}  test pred range "
          f"[{test_df.pred.min():.6f}, {test_df.pred.max():.6f}]")
    # Logging is the default: a score that was seen but not logged cannot be cited later.
    if not args.no_log:
        run_id = log_run(cfg, args.experiment, features, folds_hash(load_folds(cfg)),
                         fold_auc, wall, oof_df, test_df)
        print(f"logged run {run_id}")
        if coefs:   # linear models: coefficients of the 5 fold models, e.g. which columns L1 set to 0
            out = Path(__file__).resolve().parents[1] / "reports" / "coefs" / f"{run_id}.csv"
            out.parent.mkdir(exist_ok=True)
            pd.concat(coefs).to_csv(out, index=False)
            print(f"coefficients written to {out.relative_to(out.parents[2])}")
