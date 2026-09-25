"""Turn a logged run's test predictions into a validated Kaggle submission

Usage:
    uv run python src/submit.py <run_id>    # writes submissions/<run_id>.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_submission.csv"


def validate(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise ValueError unless `sub` is uploadable. Checks mirror sample_submission.csv."""
    if list(sub.columns) != ["id", "accepted"]:
        raise ValueError(f"columns must be exactly ['id', 'accepted'], got {list(sub.columns)}")
    if len(sub) != len(sample):
        raise ValueError(f"expected {len(sample)} rows, got {len(sub)}")
    if not np.array_equal(sub["id"].to_numpy(), sample["id"].to_numpy()):
        raise ValueError("ids differ from sample_submission.csv (missing, extra or reordered)")
    if not pd.api.types.is_float_dtype(sub["accepted"]):
        raise ValueError(f"'accepted' must be float, got {sub['accepted'].dtype}")
    if sub["accepted"].isna().any():
        raise ValueError(f"{sub['accepted'].isna().sum()} missing predictions")
    if not sub["accepted"].between(0, 1).all():
        raise ValueError("predictions outside [0, 1]")


def make_submission(run_id: str) -> Path:
    sample = pd.read_csv(SAMPLE)
    preds = pd.read_parquet(ROOT / "preds" / f"{run_id}.parquet")
    sub = sample[["id"]].merge(preds, on="id", how="left", validate="1:1")
    sub = sub.rename(columns={"pred": "accepted"})
    validate(sub, sample)

    out = ROOT / "submissions" / f"{run_id}.csv"
    # Full float precision: rounding would create ties, and ties can only lower AUC.
    sub.to_csv(out, index=False)
    validate(pd.read_csv(out), sample)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(f"wrote {make_submission(args.run_id).relative_to(ROOT)}")
