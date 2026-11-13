"""Blend logged runs into a new logged run. No model is retrained: the members' saved
out-of-fold and test predictions are combined with fixed weights.

Usage:
    uv run python src/blend.py configs/blends/<name>.yaml

Blend file:
    members: [<experiment.yaml>, ...]   members, each = the latest logged run of that experiment
    runs: [<run_id>, ...]               or: members given by run id
    method: rank | logit
    weights: [w1, ...]                  optional, default equal
All members must use the same folds. A member given by experiment must have a logged run whose
config is the one that file builds now (so "make cv" on each member comes first).

    rank : weighted mean of each member's percentile rank (rank-based, ignores calibration)
    logit: weighted mean of each member's log-odds, mapped back to a probability. A fixed linear
           combination of log-odds keeps SHAP values additive: the blend's SHAP values are the
           weighted sum of the members' SHAP values.

Reads results/runs.csv and the members' oof/ and preds/ files; logs the blend as a new run.
Pipeline: builds the final model (configs/blends/lgbm_cat_et_xgb.yaml). Report: section 6.4.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from compare import _run
from config import config_hash, load_config
from runlog import RUNS, log_run

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12   # keeps log-odds finite for predictions of exactly 0 or 1


def latest_run(experiment: str) -> str:
    """Run id of the latest logged run of an experiment file, refused if its config differs from
    what the file builds now (the run would not be the model the file describes)."""
    runs = pd.read_csv(RUNS)
    done = runs[runs.experiment.map(lambda e: Path(e).as_posix()) == Path(experiment).as_posix()]
    if done.empty:
        raise SystemExit(f"no logged run of {experiment}: run make cv EXP={experiment} first")
    last = done.iloc[-1]
    if last.config_hash != config_hash(load_config(ROOT / experiment)):
        raise SystemExit(f"latest run of {experiment} ({last.run_id}) used another config: run make cv EXP={experiment}")
    return last.run_id


def combine(preds: list[np.ndarray], weights: np.ndarray, method: str) -> np.ndarray:
    if method == "rank":
        return sum(w * pd.Series(p).rank(pct=True).to_numpy() for w, p in zip(weights, preds))
    if method == "logit":
        z = sum(w * np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))) for w, p in zip(weights, preds))
        return 1 / (1 + np.exp(-z))
    raise ValueError(f"method must be rank or logit, got {method!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("blend")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.blend).read_text())
    t0 = time.perf_counter()
    run_ids = spec["runs"] if "runs" in spec else [latest_run(e) for e in spec["members"]]
    members = [_run(r) for r in run_ids]
    if len({m.folds_hash for m in members}) != 1:
        raise SystemExit("members use different folds: they cannot be blended")
    weights = np.array(spec.get("weights") or [1.0] * len(members), dtype=float)
    weights = weights / weights.sum()          # only the ratios matter for the ranking

    oofs = [pd.read_parquet(ROOT / m.oof_path).sort_values("id").reset_index(drop=True) for m in members]
    tests = [pd.read_parquet(ROOT / m.pred_path).sort_values("id").reset_index(drop=True) for m in members]
    assert all(o.id.equals(oofs[0].id) and o.fold.equals(oofs[0].fold) for o in oofs), "OOF rows differ"
    assert all(t.id.equals(tests[0].id) for t in tests), "test rows differ"

    oof = oofs[0][["id", "fold", "accepted"]].assign(pred=combine([o.pred.to_numpy() for o in oofs], weights, spec["method"]))
    test = tests[0][["id"]].assign(pred=combine([t.pred.to_numpy() for t in tests], weights, spec["method"]))
    fold_auc = [float(roc_auc_score(oof.accepted[oof.fold == k], oof.pred[oof.fold == k])) for k in sorted(oof.fold.unique())]

    cfg = {"model": "blend", "method": spec["method"], "runs": run_ids, "weights": weights.round(6).tolist()}
    run_id = log_run(cfg, args.blend, [f"run:{r}" for r in run_ids], members[0].folds_hash,
                     fold_auc, time.perf_counter() - t0, oof, test)
    print("members:", ", ".join(f"{m.run_id} ({m.mean_auc:.5f})" for m in members))
    print(f"fold AUC: {np.round(fold_auc, 6).tolist()}\nmean {np.mean(fold_auc):.6f}\nlogged run {run_id}")
