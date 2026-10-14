"""Blend logged runs into a new logged run (TODO section 11). No model is retrained: the members'
saved out-of-fold and test predictions are combined.

Usage:
    uv run python src/blend.py configs/blends/<name>.yaml

Blend file:
    runs: [<run_id>, ...]          members, all on the same folds
    method: rank | logit
    weights: [w1, ...]             optional, default equal

    rank : weighted mean of each member's percentile rank (rank-based, ignores calibration)
    logit: weighted mean of each member's log-odds, mapped back to a probability. A fixed linear
           combination of log-odds keeps SHAP values additive: the blend's SHAP values are the
           weighted sum of the members' SHAP values.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from compare import _run
from runlog import log_run

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12   # keeps log-odds finite for predictions of exactly 0 or 1


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
    members = [_run(r) for r in spec["runs"]]
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

    cfg = {"model": "blend", "method": spec["method"], "runs": spec["runs"], "weights": weights.round(6).tolist()}
    run_id = log_run(cfg, args.blend, [f"run:{r}" for r in spec["runs"]], members[0].folds_hash,
                     fold_auc, time.perf_counter() - t0, oof, test)
    print("members:", ", ".join(f"{m.run_id} ({m.mean_auc:.5f})" for m in members))
    print(f"fold AUC: {np.round(fold_auc, 6).tolist()}\nmean {np.mean(fold_auc):.6f}\nlogged run {run_id}")
