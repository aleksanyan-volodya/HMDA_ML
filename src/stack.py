"""Stacking: a logistic regression on the models' out-of-fold predictions.

Usage:
    uv run python src/stack.py configs/blends/<name>.yaml

Stack file:
    base: <run_id>                 the reference model
    candidates: {name: run_id}     the other inputs of the stack
    reference: [name, ...]         the blend the stack is compared with (equal weights, log-odds)

For each fold k the meta-model is fitted on the other folds' rows and scores fold k, so the quoted AUC
never uses a label of the fold it is scored on. Slightly optimistic all the same: the inputs of those
rows come from models that trained on fold k. Three kinds of input are compared (log-odds, ranks,
probabilities); the coefficients are printed for the log-odds one.

Reads results/runs.csv and oof/; prints the result, writes nothing.
Evidence for report section 6.4, not part of the final model.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from blend import EPS
from compare import _run
from select_blend import fold_aucs, passes

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {"log-odds": lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))),
          "ranks": lambda p: pd.Series(p).rank(pct=True).to_numpy(),
          "probabilities": lambda p: p}


def meta_model():
    # C = 1: a handful of inputs against ~84,000 rows, so the penalty hardly acts; not tuned. Standardised
    # so that the coefficients of the inputs can be compared.
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stack")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.stack).read_text())
    runs = {"base": _run(spec["base"]), **{n: _run(r) for n, r in spec["candidates"].items()}}
    if len({r.folds_hash for r in runs.values()}) != 1:
        raise SystemExit("runs use different folds")
    oofs = {n: pd.read_parquet(ROOT / r.oof_path).sort_values("id").reset_index(drop=True) for n, r in runs.items()}
    assert all(o.id.equals(oofs["base"].id) for o in oofs.values())
    preds = {n: o.pred.to_numpy() for n, o in oofs.items()}
    y, fold = oofs["base"].accepted.to_numpy(), oofs["base"].fold.to_numpy()
    folds = sorted(np.unique(fold))
    reference = fold_aucs(spec["reference"], preds, y, fold, folds, "logit")
    print(f"reference {' + '.join(spec['reference'])}: {reference.mean():.5f}")

    for kind, transform in INPUTS.items():
        Z = np.column_stack([transform(p) for p in preds.values()])
        auc = np.array([roc_auc_score(y[fold == k], meta_model().fit(Z[fold != k], y[fold != k]).predict_proba(Z[fold == k])[:, 1])
                        for k in folds])
        ok, ratio = passes(auc - reference)
        print(f"stack on {kind:13s} {auc.mean():.5f}  gain {(auc - reference).mean():+.5f} ({ratio:+.1f} SE, "
              f"{int((auc > reference).sum())} of {len(folds)} folds up) -> {'PASSES' if ok else 'does not pass'}")
    Z = np.column_stack([INPUTS["log-odds"](p) for p in preds.values()])
    coef = meta_model().fit(Z, y)[-1].coef_[0]
    print("coefficients, log-odds input, all rows:", {n: round(float(c), 3) for n, c in zip(preds, coef)})
