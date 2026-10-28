"""Blend weights by hill climbing (TODO 11.4), from logged runs only.

Usage:
    uv run python src/hill_climb.py configs/blends/<name>.yaml

Blend file:
    base: <run_id>                 the starting model
    candidates: {name: run_id}     models that may be added (the base may be added again)
    reference: [name, ...]         the blend the result is compared with (e.g. [base, catboost_county])
    method: logit | rank
    max_steps: <int>

Greedy forward selection WITH replacement (Caruana et al., 2004): at each step add one more copy of
the model that raises the mean fold AUC the most, stop when no copy raises it. A model added n times
gets weight n / (number of copies), so the weights are chosen to maximise AUC directly (unlike least
squares, which fits probability error).

Weights chosen on the folds they are scored on flatter the blend, so the procedure is also run
NESTED: for each fold k the climb is redone on the other folds only and its blend is scored on k.
Only the nested score is compared with the reference blend, under the project rule.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from compare import _run
from select_blend import fold_aucs, passes

ROOT = Path(__file__).resolve().parents[1]


def climb(start: list[str], names: list[str], preds, y, fold, folds, method, max_steps: int) -> list[str]:
    members = list(start)
    current = fold_aucs(members, preds, y, fold, folds, method).mean()
    for _ in range(max_steps):
        trials = {n: fold_aucs(members + [n], preds, y, fold, folds, method).mean() for n in names}
        best = max(trials, key=trials.get)
        if trials[best] <= current:      # no copy improves the blend: stop
            break
        members.append(best)
        current = trials[best]
    return members


def weights(members: list[str]) -> str:
    return ", ".join(f"{n} {c / len(members):.2f}" for n, c in Counter(members).most_common())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("blend")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.blend).read_text())
    runs = {"base": _run(spec["base"]), **{n: _run(r) for n, r in spec["candidates"].items()}}
    if len({r.folds_hash for r in runs.values()}) != 1:
        raise SystemExit("runs use different folds")
    oofs = {n: pd.read_parquet(ROOT / r.oof_path).sort_values("id").reset_index(drop=True) for n, r in runs.items()}
    assert all(o.id.equals(oofs["base"].id) for o in oofs.values())
    preds = {n: o.pred.to_numpy() for n, o in oofs.items()}
    y, fold = oofs["base"].accepted.to_numpy(), oofs["base"].fold.to_numpy()
    all_folds, names, method = sorted(np.unique(fold)), list(runs), spec["method"]

    full = climb(["base"], names, preds, y, fold, all_folds, method, spec["max_steps"])
    print(f"on all folds ({len(full)} copies): {weights(full)}  "
          f"mean AUC {fold_aucs(full, preds, y, fold, all_folds, method).mean():.5f} (in-sample, optimistic)")

    nested = []
    for k in all_folds:
        m = climb(["base"], names, preds, y, fold, [f for f in all_folds if f != k], method, spec["max_steps"])
        nested.append(fold_aucs(m, preds, y, fold, [k], method)[0])
        print(f"  without fold {k}: {weights(m)}")
    nested = np.array(nested)
    for label, ref in [("reference " + " + ".join(spec["reference"]), spec["reference"]), ("base alone", ["base"])]:
        gain = nested - fold_aucs(ref, preds, y, fold, all_folds, method)
        ok, ratio = passes(gain)
        print(f"nested {nested.mean():.5f} vs {label}: gain {gain.mean():+.5f} ({ratio:+.1f} SE, "
              f"{int((gain > 0).sum())} of {len(gain)} folds up) -> {'PASSES' if ok else 'does not pass'}")
