"""The models run by src/cv.py, one class per value of "model:" in an experiment config.

Every model has the same interface:
    model = Model(cfg).fit(X, y, fold)    fold: outer fold number of each row of X (the scored fold is not in X)
    model.predict(X)                      predicted probability of acceptance
Settings that need labels (tree count, penalty, number of epochs) are chosen on X only, using the outer
folds present in X as inner folds, so the scored fold is never seen.

Reads nothing and writes nothing. Report: section 6.2 (models); their settings are in configs/base.yaml.
"""
import time
import warnings

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

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray, weight: np.ndarray | None = None) -> "LGBMModel":
        inner = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in np.unique(fold)]
        # lgb.cv averages the AUC curves of the inner folds; the tuned tree count is the
        # peak of that mean curve. With weights, that inner AUC is weighted too (the reweighed
        # objective picks its own tree count); the outer score stays unweighted.
        history = lgb.cv(self.params, lgb.Dataset(X, y, weight=weight), num_boost_round=self.max_rounds,
                         folds=inner, callbacks=[lgb.early_stopping(self.patience, verbose=False),
                                    lgb.log_evaluation(PROGRESS_EVERY)])
        mean_auc = np.asarray(history["valid auc-mean"])
        if len(mean_auc) >= self.max_rounds:  # early stopping never triggered
            raise RuntimeError(f"tree search hit max_rounds={self.max_rounds}: raise the cap")
        self.n_trees_ = int(mean_auc.argmax()) + 1
        self.booster_ = lgb.train(self.params, lgb.Dataset(X, y, weight=weight), num_boost_round=self.n_trees_)
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
        self.single_fit = cfg["catboost_fit"]["single_fit"]

    @staticmethod
    def _prepare(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # CatBoost needs categories as text with no missing value: missing becomes its own level.
        cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        X = X.copy()
        for c in cat:
            X[c] = X[c].astype("string").fillna("missing").astype(object)
        return X, cat

    def _fit_once(self, X: pd.DataFrame, y: np.ndarray, cat: list[str], fold: np.ndarray) -> "CatBoostModel":
        """Search only (tune_optuna.py): one fit on all inner folds but one, the tree count taken on that
        one, no refit. 1 fit instead of 5 and fewer rows, so scores run lower: good enough to rank settings."""
        stop = fold == np.unique(fold)[0]
        params = {k: v for k, v in self.params.items() if k != "logging_level"}   # refused together with verbose
        self.model_ = cb.CatBoost({**params, "iterations": self.max_rounds}).fit(
            cb.Pool(X[~stop], y[~stop], cat_features=cat), eval_set=cb.Pool(X[stop], y[stop], cat_features=cat),
            early_stopping_rounds=self.patience, use_best_model=True, verbose=PROGRESS_EVERY)
        self.n_trees_ = self.model_.tree_count_
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "CatBoostModel":
        X, cat = self._prepare(X)
        if self.single_fit:
            return self._fit_once(X, np.asarray(y), cat, fold)
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
