"""Noise test for the target encoding: a category that is pure noise must not predict the target.

Synthetic data shaped like county_code: many levels, most of them tiny, target rate 0.85.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from target_encoding import HierarchicalRate, oof_encode

N, LEVELS, STATES, FOLDS, M = 20_000, 2_000, 50, 5, 20


def _noise_data():
    rng = np.random.default_rng(0)                                   # fixed: the test must be reproducible
    y = pd.Series(rng.random(N) < 0.85).astype(int)
    # Zipf-like level sizes: a few big levels, a long tail of levels with 1-5 rows, like counties.
    weights = 1 / np.arange(1, LEVELS + 1)
    child = pd.Series(rng.choice(LEVELS, N, p=weights / weights.sum())).astype(str)
    parent = (child.astype(int) % STATES).astype(str)                # each child belongs to one parent
    fold = rng.integers(0, FOLDS, N)
    return child, parent, y, fold


def test_oof_encoding_of_noise_has_no_signal():
    child, parent, y, fold = _noise_data()
    auc = roc_auc_score(y, oof_encode(child, parent, y, fold, M))
    assert abs(auc - 0.5) < 0.02, f"out-of-fold encoding of pure noise scores {auc:.3f}: it leaks"


def test_in_sample_encoding_of_noise_leaks():
    # The mistake the out-of-fold scheme prevents: encoding rows with rates that include their own label.
    # If this did NOT score well above 0.5, the noise test above would be unable to detect a leak.
    child, parent, y, _ = _noise_data()
    auc = roc_auc_score(y, HierarchicalRate(M).fit(child, parent, y).transform(child, parent))
    assert auc > 0.6, f"in-sample encoding scores only {auc:.3f}: the noise test has no power"


def test_missing_and_unseen_levels_fall_back():
    child = pd.Series(["a", "a", "b", None])
    parent = pd.Series(["s", "s", "s", None])
    y = pd.Series([1, 1, 0, 1])
    enc = HierarchicalRate(M).fit(child, parent, y)
    out = enc.transform(pd.Series(["zz", None, "zz"]), pd.Series(["s", "s", "unseen"]))
    assert out[0] == enc.parent_rate_["s"]                           # unseen child -> its parent's rate
    assert np.isnan(out[1])                                          # missing child -> NaN, never an average
    assert out[2] == enc.prior_                                      # unseen child and parent -> overall rate
