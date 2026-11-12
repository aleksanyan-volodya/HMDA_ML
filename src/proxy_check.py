"""TODO 12.3, proxy check: how well the columns a fair model keeps predict an applicant's race.

Usage:
    uv run python src/proxy_check.py configs/experiments/ablate_no_demographics.yaml

For each race group with >= MIN_ROWS rows, a LightGBM (the experiment's settings, tree count by nested CV
as in src/cv.py) predicts "this applicant is in the group" against everyone else, on the frozen folds, from
two column sets: all the experiment's columns, then the geography columns only (features.BLOCKS).
AUC 0.5 = the columns say nothing about that group; the closer to 1, the more a model that never sees race
can still act on it. Writes reports/tuning/proxy_check.csv, and the out-of-fold scores to
oof/proxy_race.parquet for the fairness agent.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import load_config
from cv import LGBMModel
from data import load
from features import BLOCKS, build
from folds import load_folds

ROOT = Path(__file__).resolve().parents[1]
MIN_ROWS = 5000   # White, Race Not Available, Black, Asian; the next group (Joint, 2,716) gives ~540 per fold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    experiment = parser.parse_args().experiment
    cfg = load_config(experiment)
    train, test = load()
    train, _, features = build(train, test, cfg["features"])
    if set(features) & set(BLOCKS["demographics"]):
        raise ValueError("the experiment still has demographic columns: this check is for a race-blind list")
    train = train.merge(load_folds(cfg), on="id", how="left", validate="1:1")
    fold, race = train["fold"].to_numpy(), train["derived_race"].astype(str)
    column_sets = {"all": features, "geography": [c for c in features if c in BLOCKS["geography"]]}
    counts = race.value_counts()

    rows, scores = [], []
    for group in counts[counts >= MIN_ROWS].index:
        y = (race == group).to_numpy().astype(int)
        for name, cols in column_sets.items():
            t0 = time.perf_counter()
            pred = np.full(len(train), np.nan)
            for k in range(cfg["n_folds"]):
                valid = fold == k
                model = LGBMModel(cfg).fit(train.loc[~valid, cols], y[~valid], fold[~valid])
                pred[valid] = model.predict(train.loc[valid, cols])
            fold_auc = [roc_auc_score(y[fold == k], pred[fold == k]) for k in range(cfg["n_folds"])]
            rows.append({"group": group, "rows": counts[group], "columns": f"{name} ({len(cols)})",
                         "mean fold AUC": np.mean(fold_auc), "min": min(fold_auc), "max": max(fold_auc)})
            print(f"{group} from {name} ({len(cols)} columns): AUC {np.mean(fold_auc):.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            scores.append(pd.DataFrame({"id": train["id"], "group": group, "columns": name, "pred": pred}))

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "reports" / "tuning" / "proxy_check.csv", index=False)
    (ROOT / "oof").mkdir(exist_ok=True)   # not tracked by git: missing in a fresh clone
    pd.concat(scores).to_parquet(ROOT / "oof" / "proxy_race.parquet", index=False)
    print(f"\ncolumns from {experiment} ({len(features)}, no demographic field); one group vs all others")
    print(table.round(4).to_string(index=False))
