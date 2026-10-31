"""Cross-validation runner: one experiment config in, per-fold AUC + OOF (out of fold) and test predictions out.

Usage:
    uv run python src/cv.py configs/experiments/constant.yaml
"""
import argparse
import time
import warnings
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
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


# Progress output: boosting models print their inner-CV AUC every PROGRESS_EVERY rounds, the logistic
# regression one line per fitted C, ExtraTrees joblib's tree counter. Output only: the fitted models and
# scores are identical with or without it (checked for CatBoost on toy data, 2026-10-18).
PROGRESS_EVERY = 100


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
                         folds=inner, callbacks=[lgb.early_stopping(self.patience, verbose=False),
                                    lgb.log_evaluation(PROGRESS_EVERY)])
        mean_auc = np.asarray(history["valid auc-mean"])
        if len(mean_auc) >= self.max_rounds:  # early stopping never triggered
            raise RuntimeError(f"tree search hit max_rounds={self.max_rounds}: raise the cap")
        self.n_trees_ = int(mean_auc.argmax()) + 1
        self.booster_ = lgb.train(self.params, lgb.Dataset(X, y), num_boost_round=self.n_trees_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster_.predict(X)


class LogRegModel:
    """Penalised logistic regression (L2 by default; L1 / elastic net with logreg.l1_ratio_grid).
    The encoding is fitted on the fit part only:
      categorical -> one-hot (missing = its own level, unseen levels ignored)
      flags (2 values) -> as they are
      other numeric -> missing indicator + median fill -> rank in [0, 1] -> cubic spline with
                       evenly spaced knots on the rank scale (= knots at the quantiles, robust to ties)
      every column then lies in [0, 1], so the penalty treats them alike without rescaling.
    The penalty strength C and the L1 share l1_ratio are chosen together by an inner CV on the fit
    part (inner folds = the outer folds present), like LightGBM's tree count."""

    def __init__(self, cfg: dict):
        p = cfg["logreg"]
        self.C_grid, self.n_knots, self.max_iter = p["C_grid"], p["spline_knots"], p["max_iter"]
        self.l1_grid = p["l1_ratio_grid"]
        assert self.C_grid == sorted(self.C_grid), "C_grid must be ascending (warm-started path)"

    def _encoder(self, X: pd.DataFrame) -> ColumnTransformer:
        cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        num = [c for c in X.columns if c not in cat]
        flags = [c for c in num if X[c].nunique(dropna=True) <= 2 and not X[c].isna().any()]
        curves = [c for c in num if c not in flags]
        as_text = FunctionTransformer(lambda d: d.astype("string").fillna("missing").astype(object),
                                      feature_names_out="one-to-one")   # names kept for coef_table
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

    def _path(self, Z, y, l1_ratio: float, Cs: list[float]):
        """Yields (C, fitted model) along Cs, ascending. l1_ratio 0 = pure L2 with lbfgs, cold fits
        (unchanged from the L2 runs). l1_ratio > 0 needs saga, which takes 30 s to 2 min per fit
        here, so each fit starts from the previous C's coefficients (warm start along the
        regularisation path, as LogisticRegressionCV does); the optimum reached is the same."""
        lr = LogisticRegression(l1_ratio=l1_ratio, max_iter=self.max_iter, warm_start=l1_ratio > 0,
                                solver="lbfgs" if l1_ratio == 0 else "saga",
                                random_state=0)   # saga shuffles rows: fixed for reproducibility (lbfgs ignores it)
        for C in Cs:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)   # a non-converged fit must not pass silently
                lr.set_params(C=C).fit(Z, y)
            yield C, lr   # outside the with: the error filter must not leak into the caller's code

    def _inner_scores(self, X: pd.DataFrame, y: np.ndarray, fold: np.ndarray, f: int, l1: float) -> np.ndarray:
        """Inner-fold AUC of every C of the grid, for one inner fold f and one l1_ratio."""
        inner_fit, inner_val = fold != f, fold == f
        enc = self._encoder(X[inner_fit]).fit(X[inner_fit])          # encoder never sees inner_val
        Z_fit, Z_val = enc.transform(X[inner_fit]), enc.transform(X[inner_val])
        t0, aucs = time.perf_counter(), []
        for C, lr in self._path(Z_fit, y[inner_fit], l1, self.C_grid):
            aucs.append(roc_auc_score(y[inner_val], lr.predict_proba(Z_val)[:, 1]))
            print(f"    inner fold {f}  l1_ratio {l1}  C {C}: AUC {aucs[-1]:.5f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        return np.array(aucs)

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "LogRegModel":
        y = np.asarray(y)
        # The (inner fold, l1_ratio) fits are independent and each solver uses one core: run them in
        # parallel (at most 8 = the cores of this machine). Same results as a loop.
        tasks = [(f, l1) for f in np.unique(fold) for l1 in self.l1_grid]
        flat = Parallel(n_jobs=min(len(tasks), 8))(delayed(self._inner_scores)(X, y, fold, f, l1) for f, l1 in tasks)
        scores = np.array(flat).reshape(len(np.unique(fold)), len(self.l1_grid), len(self.C_grid))
        r, best = np.unravel_index(scores.mean(axis=0).argmax(), scores.shape[1:])
        # Edge rule on C only: l1_ratio 0 and 1 are the real ends of its range, nothing lies beyond.
        if best in (0, len(self.C_grid) - 1):
            raise RuntimeError(f"best C is at the edge of the grid ({self.C_grid[best]}): widen C_grid")
        self.C_, self.l1_ratio_ = self.C_grid[best], self.l1_grid[r]
        self.encoder_ = self._encoder(X).fit(X)
        *_, (_, self.lr_) = self._path(self.encoder_.transform(X), y, self.l1_ratio_,
                                       self.C_grid[:best + 1])   # path up to the chosen C, keep the last
        return self

    def coef_table(self) -> pd.DataFrame:
        """One row per encoded column: name, source variable, coefficient (0 = removed by L1)."""
        names = [n.split("__", 1)[1].removeprefix("missingindicator_") for n in self.encoder_.get_feature_names_out()]
        inputs = sorted(self.encoder_.feature_names_in_, key=len, reverse=True)   # longest match first
        source = [next(c for c in inputs if n == c or n.startswith(c + "_")) for n in names]
        return pd.DataFrame({"column": self.encoder_.get_feature_names_out(), "variable": source,
                             "coef": self.lr_.coef_[0]})

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
        # logging_level (Silent) left out here only: CatBoost refuses it together with verbose_eval
        cv_params = {k: v for k, v in self.params.items() if k != "logging_level"}
        history = cb.cv(pool, {**cv_params, "iterations": self.max_rounds}, folds=inner,
                        early_stopping_rounds=self.patience, verbose_eval=PROGRESS_EVERY, plot=False)
        mean_auc = history["test-AUC-mean"].to_numpy()
        if len(mean_auc) >= self.max_rounds:  # early stopping never triggered
            raise RuntimeError(f"tree search hit max_rounds={self.max_rounds}: raise the cap")
        self.n_trees_ = int(mean_auc.argmax()) + 1
        self.model_ = cb.CatBoost({**self.params, "iterations": self.n_trees_}).fit(pool, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._prepare(X)[0], prediction_type="Probability")[:, 1]


class XGBModel:
    """XGBoost, same nested scheme as LGBMModel: the tree count is the peak of the mean AUC curve
    of an inner CV on the fit part (inner folds = the outer folds present).
    Categories are given natively (enable_categorical); NaN is handled by the library."""

    def __init__(self, cfg: dict):
        self.params = {**cfg["xgb_params"], "seed": cfg["model_seeds"][0]}
        self.max_rounds = cfg["xgb_fit"]["max_rounds"]
        self.patience = cfg["xgb_fit"]["early_stopping_rounds"]

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "XGBModel":
        data = xgb.DMatrix(X, np.asarray(y), enable_categorical=True)
        inner = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in np.unique(fold)]
        history = xgb.cv(self.params, data, num_boost_round=self.max_rounds, folds=inner,
                         early_stopping_rounds=self.patience, verbose_eval=PROGRESS_EVERY)
        mean_auc = history["test-auc-mean"].to_numpy()
        if len(mean_auc) >= self.max_rounds:  # early stopping never triggered
            raise RuntimeError(f"tree search hit max_rounds={self.max_rounds}: raise the cap")
        self.n_trees_ = int(mean_auc.argmax()) + 1
        self.booster_ = xgb.train(self.params, data, num_boost_round=self.n_trees_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster_.predict(xgb.DMatrix(X, enable_categorical=True))


class ExtraTreesModel:
    """ExtraTrees: an average of fully random trees (random features AND random split thresholds),
    so its errors differ from boosting's. Nothing is searched: more trees only reduces the
    averaging noise, and the other settings are fixed in base.yaml with their reasons.
    Categories are one-hot encoded (every category kept has at most 54 levels; the fixed category
    lists of src/data.py give train and test the same columns). NaN is handled by scikit-learn."""

    def __init__(self, cfg: dict):
        self.params = {**cfg["extratrees_params"], "random_state": cfg["model_seeds"][0], "n_jobs": -1}
        self.rank_numeric = cfg["extratrees_fit"]["rank_numeric"]

    @staticmethod
    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        return pd.get_dummies(X, columns=cat, dummy_na=True, dtype=np.float32).astype(np.float32)

    def _ranks(self, X: pd.DataFrame) -> pd.DataFrame:
        # Numeric columns -> rank in [0, 1] (fitted on the fit part; NaN stays NaN). ExtraTrees draws
        # split thresholds uniformly between a column's min and max: on a skewed column (loan amount
        # up to 375M, median 285k) almost every threshold lands in the empty tail. On ranks they
        # land where the rows are.
        if not self.rank_numeric:
            return X
        X = X.copy()
        X[self.numeric_] = self.quantiles_.transform(X[self.numeric_])
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "ExtraTreesModel":
        if self.rank_numeric:
            self.numeric_ = [c for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype)]
            self.quantiles_ = QuantileTransformer(n_quantiles=1000, random_state=0).fit(X[self.numeric_])
            X = self._ranks(X)
        Z = self._encode(X)
        self.columns_ = Z.columns
        self.model_ = ExtraTreesClassifier(**self.params, verbose=1).fit(Z, np.asarray(y))   # verbose: progress only
        self.model_.set_params(verbose=0)   # the tree counter is for the fit; predictions stay quiet
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._encode(self._ranks(X)).reindex(columns=self.columns_, fill_value=0)
        return self.model_.predict_proba(Z)[:, 1]


def MLPModel(cfg: dict):
    # Imported here: torch is only loaded by the runs that use it.
    from mlp import MLPModel as Model
    return Model(cfg)


MODELS = {"constant": ConstantModel, "lgbm": LGBMModel, "logreg": LogRegModel, "catboost": CatBoostModel,
          "xgb": XGBModel, "extratrees": ExtraTreesModel, "mlp": MLPModel}


def run_cv(cfg: dict, folds: pd.DataFrame | None = None,
           coefs: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[float], list[str]]:
    """folds: (id, fold) to use instead of the frozen folds. Only for checks on fresh splits
    (src/fresh_folds.py); such runs are not comparable with reports/runs.csv and are not logged.
    coefs: if given, each fold model's coefficient table (linear models only) is appended to it."""
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
        t_fold = time.perf_counter()
        print(f"fold {k}: fitting on {len(fit_part)} rows", flush=True)
        model = MODELS[cfg["model"]](cfg).fit(fit_part[cols], fit_part[TARGET],
                                               fit_part["fold"].to_numpy())
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
