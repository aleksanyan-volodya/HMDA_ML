"""Cross-validation runner: one experiment config in, per-fold AUC + OOF (out of fold) and test predictions out.

Usage:
    uv run python src/cv.py configs/experiments/constant.yaml
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import load_config
from data import TARGET, load
from features import build
from folds import folds_hash, load_folds
from runlog import log_run


class ConstantModel:
    """Predicts the training accept rate for every row. All scores are tied, so no row is
    ranked above another and AUC is 0.5 by construction: it checks the pipeline, not a model."""

    def __init__(self, cfg: dict):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "ConstantModel":
        self.rate_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)


class LGBMModel:
    """LightGBM whose tree count is tuned by an inner CV on the fit part only (nested CV).
    Inner folds = the outer folds present in the fit part, so no new split is introduced."""

    def __init__(self, cfg: dict):
        # Single seed for the baseline; seed averaging is TODO 11.2.
        self.params = {**cfg["lgbm_params"], "seed": cfg["model_seeds"][0]}
        self.max_rounds = cfg["lgbm_fit"]["max_rounds"]
        self.patience = cfg["lgbm_fit"]["early_stopping_rounds"]

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "LGBMModel":
        inner = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in np.unique(fold)]
        # lgb.cv averages the AUC curves of the inner folds; the tuned tree count is the
        # peak of that mean curve.
        history = lgb.cv(self.params, lgb.Dataset(X, y), num_boost_round=self.max_rounds,
                         folds=inner, callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        mean_auc = np.asarray(history["valid auc-mean"])
        if len(mean_auc) >= self.max_rounds:  # early stopping never triggered
            raise RuntimeError(f"tree search hit max_rounds={self.max_rounds}: raise the cap")
        self.n_trees_ = int(mean_auc.argmax()) + 1
        self.booster_ = lgb.train(self.params, lgb.Dataset(X, y), num_boost_round=self.n_trees_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster_.predict(X)


MODELS = {"constant": ConstantModel, "lgbm": LGBMModel}


def run_cv(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[float], list[str]]:
    if cfg["model"] not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {cfg['model']!r}")
    train, test = load()
    train, test, features = build(train, test, cfg["features"])
    print(f"{len(features)} features", flush=True)
    train = train.merge(load_folds(cfg), on="id", how="left", validate="1:1")

    oof = np.full(len(train), np.nan)
    test_pred = np.zeros(len(test))
    fold_auc = []
    for k in range(cfg["n_folds"]):
        is_valid = (train["fold"] == k).to_numpy()
        fit_part, valid_part = train[~is_valid], train[is_valid]
        model = MODELS[cfg["model"]](cfg).fit(fit_part[features], fit_part[TARGET],
                                               fit_part["fold"].to_numpy())
        oof[is_valid] = model.predict(valid_part[features])
        fold_auc.append(float(roc_auc_score(valid_part[TARGET], oof[is_valid])))
        extra = f"  trees {model.n_trees_}" if hasattr(model, "n_trees_") else ""
        print(f"fold {k}  AUC {fold_auc[-1]:.6f}{extra}", flush=True)
        # Test prediction = average of the k fold models, càd exactly the models whose
        # scores we measured (a refit on all of train would be an unvalidated model).
        test_pred += model.predict(test[features]) / cfg["n_folds"]

    assert not np.isnan(oof).any(), "some train rows never received an OOF prediction"
    oof_df = train[["id", "fold", TARGET]].assign(pred=oof)
    test_df = test[["id"]].assign(pred=test_pred)
    return oof_df, test_df, fold_auc, features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--no-log", action="store_true", help="debug only: nothing is written")
    args = parser.parse_args()
    cfg = load_config(args.experiment)
    t0 = time.perf_counter()
    oof_df, test_df, fold_auc, features = run_cv(cfg)
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
