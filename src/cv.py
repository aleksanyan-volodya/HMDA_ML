"""Cross-validation runner: one experiment config in, per-fold AUC + OOF (out of fold) and test predictions out.

Usage:
    uv run python src/cv.py configs/experiments/constant.yaml
"""
import argparse
import time
import warnings

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, QuantileTransformer, SplineTransformer

from config import load_config
from data import TARGET, load
from features import build
from folds import folds_hash, load_folds
from runlog import log_run
from target_encoding import encode_fold


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


class LogRegModel:
    """L2 logistic regression. The encoding is fitted on the fit part only:
      categorical -> one-hot (missing = its own level, unseen levels ignored)
      flags (2 values) -> as they are
      other numeric -> missing indicator + median fill -> rank in [0, 1] -> cubic spline with
                       evenly spaced knots on the rank scale (= knots at the quantiles, robust to ties)
      every column then lies in [0, 1], so the penalty treats them alike without rescaling.
    The penalty strength C is chosen by an inner CV on the fit part (inner folds = the outer folds
    present), like LightGBM's tree count."""

    def __init__(self, cfg: dict):
        p = cfg["logreg"]
        self.C_grid, self.n_knots, self.max_iter = p["C_grid"], p["spline_knots"], p["max_iter"]

    def _encoder(self, X: pd.DataFrame) -> ColumnTransformer:
        cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        num = [c for c in X.columns if c not in cat]
        flags = [c for c in num if X[c].nunique(dropna=True) <= 2 and not X[c].isna().any()]
        curves = [c for c in num if c not in flags]
        as_text = FunctionTransformer(lambda d: d.astype("string").fillna("missing").astype(object))
        spline = make_pipeline(SimpleImputer(strategy="median"),
                               QuantileTransformer(n_quantiles=1000, random_state=0),   # 0: fixed subsample
                               SplineTransformer(n_knots=self.n_knots, degree=3, knots="uniform"))
        return ColumnTransformer([
            ("onehot", make_pipeline(as_text, OneHotEncoder(handle_unknown="ignore")), cat),
            ("flags", "passthrough", flags),
            ("missing", MissingIndicator(features="missing-only", error_on_new=False), curves),
            ("spline", spline, curves),
        ], sparse_threshold=1.0)
        # No scaling: every column already lies in [0, 1] (one-hot, flags, spline pieces). Scaling to unit
        # variance would multiply a level present in 0.1% of rows by ~30 and distort the penalty.

    def _fit_lr(self, Z, y, C: float) -> LogisticRegression:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)   # a non-converged fit must not pass silently
            return LogisticRegression(C=C, max_iter=self.max_iter).fit(Z, y)

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "LogRegModel":
        y = np.asarray(y)
        scores = np.zeros((len(np.unique(fold)), len(self.C_grid)))
        for i, f in enumerate(np.unique(fold)):
            inner_fit, inner_val = fold != f, fold == f
            enc = self._encoder(X[inner_fit]).fit(X[inner_fit])          # encoder never sees inner_val
            Z_fit, Z_val = enc.transform(X[inner_fit]), enc.transform(X[inner_val])
            for j, C in enumerate(self.C_grid):
                p = self._fit_lr(Z_fit, y[inner_fit], C).predict_proba(Z_val)[:, 1]
                scores[i, j] = roc_auc_score(y[inner_val], p)
        best = int(scores.mean(axis=0).argmax())
        if best in (0, len(self.C_grid) - 1):
            raise RuntimeError(f"best C is at the edge of the grid ({self.C_grid[best]}): widen C_grid")
        self.C_ = self.C_grid[best]
        self.encoder_ = self._encoder(X).fit(X)
        self.lr_ = self._fit_lr(self.encoder_.transform(X), y, self.C_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.encoder_.transform(X))[:, 1]


class CatBoostModel:
    """CatBoost, same nested scheme as LGBMModel: the tree count is the peak of the mean AUC curve
    of an inner CV on the fit part (inner folds = the outer folds present).
    Categories are given natively: CatBoost encodes them with ordered target statistics, computed
    in a way that never uses a row's own label, which is its main difference from LightGBM."""

    def __init__(self, cfg: dict):
        self.params = {**cfg["catboost_params"], "random_seed": cfg["model_seeds"][0]}
        self.max_rounds = cfg["catboost_fit"]["max_rounds"]
        self.patience = cfg["catboost_fit"]["early_stopping_rounds"]

    @staticmethod
    def _prepare(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # CatBoost needs categories as text with no missing value: missing becomes its own level.
        cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        X = X.copy()
        for c in cat:
            X[c] = X[c].astype("string").fillna("missing").astype(object)
        return X, cat

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "CatBoostModel":
        X, cat = self._prepare(X)
        pool = cb.Pool(X, np.asarray(y), cat_features=cat)
        inner = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in np.unique(fold)]
        history = cb.cv(pool, {**self.params, "iterations": self.max_rounds}, folds=inner,
                        early_stopping_rounds=self.patience, verbose=False, plot=False)
        mean_auc = history["test-AUC-mean"].to_numpy()
        if len(mean_auc) >= self.max_rounds:  # early stopping never triggered
            raise RuntimeError(f"tree search hit max_rounds={self.max_rounds}: raise the cap")
        self.n_trees_ = int(mean_auc.argmax()) + 1
        self.model_ = cb.CatBoost({**self.params, "iterations": self.n_trees_}).fit(pool, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._prepare(X)[0], prediction_type="Probability")[:, 1]


MODELS = {"constant": ConstantModel, "lgbm": LGBMModel, "logreg": LogRegModel, "catboost": CatBoostModel}


def run_cv(cfg: dict, folds: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[float], list[str]]:
    """folds: (id, fold) to use instead of the frozen folds. Only for checks on fresh splits
    (src/fresh_folds.py); such runs are not comparable with reports/runs.csv and are not logged."""
    if cfg["model"] not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {cfg['model']!r}")
    train, test = load()
    train, test, features = build(train, test, cfg["features"])
    print(f"{len(features)} features + {len(cfg['target_encode'])} target-encoded", flush=True)
    train = train.merge(load_folds(cfg) if folds is None else folds, on="id", how="left", validate="1:1")

    oof = np.full(len(train), np.nan)
    test_pred = np.zeros(len(test))
    fold_auc = []
    for k in range(cfg["n_folds"]):
        is_valid = (train["fold"] == k).to_numpy()
        # Target encodings use labels, so they are computed here, inside the fold.
        fit_part, valid_part, test_k, encoded = encode_fold(train[~is_valid], train[is_valid], test,
                                                            TARGET, cfg["target_encode"])
        cols = features + encoded
        model = MODELS[cfg["model"]](cfg).fit(fit_part[cols], fit_part[TARGET],
                                               fit_part["fold"].to_numpy())
        oof[is_valid] = model.predict(valid_part[cols])
        fold_auc.append(float(roc_auc_score(valid_part[TARGET], oof[is_valid])))
        extra = (f"  trees {model.n_trees_}" if hasattr(model, "n_trees_")
                 else f"  C {model.C_}" if hasattr(model, "C_") else "")
        print(f"fold {k}  AUC {fold_auc[-1]:.6f}{extra}", flush=True)
        # Test prediction = average of the k fold models, càd exactly the models whose
        # scores we measured (a refit on all of train would be an unvalidated model).
        test_pred += model.predict(test_k[cols]) / cfg["n_folds"]

    assert not np.isnan(oof).any(), "some train rows never received an OOF prediction"
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
