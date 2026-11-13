"""Smoothed target encoding of a high-cardinality column, shrunk towards a parent level
(reports/feature_plan.md, 4.2): county -> state -> overall.

    parent rate = (accepted in parent + m * overall rate) / (rows in parent + m)
    child rate  = (accepted in child  + m * parent rate)  / (rows in child  + m)

m = number of rows a level needs before its own rate weighs as much as its parent's.
An unseen child gets its parent's rate; an unseen or missing parent gets the overall rate.
A MISSING child stays NaN: "county unknown" is accepted 0.25 in this data, far from any state
average, so filling it would hide a real signal. The model handles the NaN.

The rates are learned from labels, so a row must never be encoded with its own label:
use oof_encode for the rows the rates are learned on, fit + transform for other rows.

Reads and writes nothing (called by src/cv.py in every fold). The final model uses no target
encoding ("target_encode: []"); configs that do: lgbm_county_rate, lgbm_tract_rate, logreg_l1.
Report: section 6.1 (T7) and ablation A4.
"""
import numpy as np
import pandas as pd


class HierarchicalRate:
    def __init__(self, m: float):
        self.m = m

    def fit(self, child: pd.Series, parent: pd.Series, y: pd.Series) -> "HierarchicalRate":
        child, parent = child.astype("string"), parent.astype("string")
        self.prior_ = float(y.mean())
        p = y.groupby(parent).agg(["sum", "size"])                      # NaN parents are left out
        self.parent_rate_ = (p["sum"] + self.m * self.prior_) / (p["size"] + self.m)
        rows = pd.DataFrame({"child": child, "y": y, "parent_rate": self._parent(parent)})
        c = rows.groupby("child").agg(k=("y", "sum"), n=("y", "size"), parent_rate=("parent_rate", "mean"))
        self.child_rate_ = (c.k + self.m * c.parent_rate) / (c.n + self.m)
        return self

    def _parent(self, parent: pd.Series) -> pd.Series:
        return parent.astype("string").map(self.parent_rate_).astype("float64").fillna(self.prior_)

    def transform(self, child: pd.Series, parent: pd.Series) -> np.ndarray:
        own = child.astype("string").map(self.child_rate_).astype("float64")
        rate = own.fillna(self._parent(parent))                  # unseen child -> its parent's rate
        return rate.where(child.notna().to_numpy()).to_numpy()   # missing child -> NaN, not an average


def encode_fold(fit_part: pd.DataFrame, valid_part: pd.DataFrame, test: pd.DataFrame, target: str,
                specs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Add one "<column>_rate" feature per spec ({column, parent, m}) for one CV fold.

    Which rows each encoding is learned from:
      valid_part and test : the whole fit part (never the scored fold's labels)
      fit_part            : out-of-fold inside the fit part (never a row's own label)
    Returns copies; the inputs are not modified.
    """
    fit_part, valid_part, test = fit_part.copy(), valid_part.copy(), test.copy()
    names = []
    for spec in specs:
        col, parent, m = spec["column"], spec["parent"], spec["m"]
        name = f"{col}_rate"
        enc = HierarchicalRate(m).fit(fit_part[col], fit_part[parent], fit_part[target])
        valid_part[name] = enc.transform(valid_part[col], valid_part[parent])
        test[name] = enc.transform(test[col], test[parent])
        fit_part[name] = oof_encode(fit_part[col], fit_part[parent], fit_part[target],
                                    fit_part["fold"].to_numpy(), m)
        names.append(name)
    return fit_part, valid_part, test, names


def oof_encode(child: pd.Series, parent: pd.Series, y: pd.Series, fold: np.ndarray, m: float) -> np.ndarray:
    """Encode each row from the rows of the OTHER folds only, so its own label is never used."""
    out = np.empty(len(child))
    for k in np.unique(fold):
        other, this = fold != k, fold == k
        out[this] = HierarchicalRate(m).fit(child[other], parent[other], y[other]).transform(child[this], parent[this])
    return out
