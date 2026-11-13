"""Train and test rows go through the same feature path.

The same N_ROWS rows, given once as training rows and once as test rows, must come out of
features.build identical (values, dtypes, column order), for the feature list of every experiment
config. A transform that treats training rows differently from test rows, for instance by reading
the label (which test rows do not have), fails here; checked with two such transforms on purpose.
It does not detect a statistic learned on the training rows: on identical rows it gives identical values.

Reads data/raw/ (the test is skipped without it). Pipeline test.
"""
from pathlib import Path

import pandas as pd
import pytest

from config import load_config
from data import RAW, TARGET, load
from features import build

ROOT = Path(__file__).resolve().parents[1]
N_ROWS = 1000   # the size asked for this test; every transform is row-wise, so more rows add nothing
CONFIGS = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
# One case per distinct feature list (many configs share one), named after the first config that uses it.
FEATURE_LISTS = {}
for path in CONFIGS:
    FEATURE_LISTS.setdefault(tuple(load_config(path)["features"]), path.stem)


@pytest.fixture(scope="module")
def same_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (RAW / "train.csv").exists():
        pytest.skip(f"no data in {RAW}")
    train, _ = load()
    rows = train.head(N_ROWS)
    return rows, rows.drop(columns=TARGET)


@pytest.mark.parametrize("names", list(FEATURE_LISTS), ids=list(FEATURE_LISTS.values()))
def test_train_and_test_rows_get_identical_features(same_rows, names):
    as_train, as_test = same_rows
    train_out, test_out, columns = build(as_train, as_test, list(names))
    pd.testing.assert_frame_equal(train_out[columns], test_out[columns], check_exact=True)
