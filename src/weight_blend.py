"""Weighted blend (TODO 11.8): weights fitted by non-negative least squares on the members' log-odds.

Usage:
    uv run python src/weight_blend.py configs/blends/<selection>.yaml

Uses the same selection file as src/select_blend.py (base + candidates). Every model is a member;
weights are >= 0, so a weak but different model can get a small weight instead of an equal share,
and correlated members cannot get large cancelling weights fitted to noise.

Honest estimate (nested): for each fold k, the weights are fitted on the other folds only and the
blend is scored on fold k. Then the weights fitted on all folds are written to a blend file.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

from blend import EPS
from compare import _run
from select_blend import passes

ROOT = Path(__file__).resolve().parents[1]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def nnls_weights(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Centring removes the intercept (it cannot change a ranking, and NNLS would force it >= 0).
    w, _ = nnls(Z - Z.mean(axis=0), y - y.mean())
    return w / w.sum() if w.sum() > 0 else np.full(Z.shape[1], 1 / Z.shape[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("selection")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.selection).read_text())
    names = {"base": spec["base"], **spec["candidates"]}
    oofs = {n: pd.read_parquet(ROOT / _run(r).oof_path).sort_values("id").reset_index(drop=True) for n, r in names.items()}
    ref = oofs["base"]
    Z = np.column_stack([logit(o.pred.to_numpy()) for o in oofs.values()])
    y, fold = ref.accepted.to_numpy().astype(float), ref.fold.to_numpy()
    folds = sorted(np.unique(fold))
    base_auc = np.array([roc_auc_score(y[fold == k], Z[fold == k, 0]) for k in folds])

    nested, rows = [], []
    for k in folds:
        w = nnls_weights(Z[fold != k], y[fold != k])
        nested.append(roc_auc_score(y[fold == k], Z[fold == k] @ w))
        rows.append(dict(zip(names, w.round(3)), held_out_fold=k))
    nested = np.array(nested)
    gain = nested - base_auc
    ok, ratio = passes(gain)
    print("weights fitted without each fold (rows) for each model (columns):")
    print(pd.DataFrame(rows).set_index("held_out_fold").to_string())
    print(f"\nnested estimate: mean AUC {nested.mean():.5f} vs base {base_auc.mean():.5f}: gain {gain.mean():+.5f} "
          f"({ratio:+.1f} SE, {int((gain > 0).sum())} of {len(gain)} folds up) -> {'PASSES' if ok else 'does not pass'} the rule")

    w = nnls_weights(Z, y)
    print("\nweights fitted on all folds:", {n: round(float(v), 3) for n, v in zip(names, w)})
    keep = w > 0
    out = ROOT / "configs" / "blends" / f"{Path(args.selection).stem}_weighted.yaml"
    out.write_text(f"# Weighted blend from {args.selection}: NNLS weights on log-odds, fitted on all folds.\n"
                   f"# Nested estimate (weights fitted without the scored fold): {nested.mean():.5f}, gain {gain.mean():+.5f} over the base.\n"
                   + yaml.safe_dump({"runs": [names[n] for n, k in zip(names, keep) if k],
                                     "weights": [round(float(v), 6) for v in w[keep]], "method": "logit"}, sort_keys=False))
    print(f"wrote {out.relative_to(ROOT)}")
