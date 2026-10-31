"""Choose the final blend by forward selection (TODO 11.4 / 11.10), from logged runs only.

Usage:
    uv run python src/select_blend.py configs/blends/<selection>.yaml

Selection file:
    base: <run_id>                 the starting model
    candidates: {name: run_id}     models that may be added
    method: logit | rank           how members are combined (equal weights)

Forward selection: add the candidate that raises the blend's mean fold AUC the most, keep it only
if the gain passes the project rule (> 0 and >= 2 corrected SE, paired over folds), repeat until
no addition passes.

Choosing members on the same folds flatters the result, so the selection is also run NESTED:
for each fold k, the whole selection is redone on the other folds only, and the blend it picks
is scored on fold k. The mean of those 5 scores is the honest estimate of the procedure.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from blend import combine
from compare import _run

ROOT = Path(__file__).resolve().parents[1]


def fold_aucs(members: list[str], preds: dict, y: np.ndarray, fold: np.ndarray, folds: list[int], method: str) -> np.ndarray:
    p = combine([preds[m] for m in members], np.full(len(members), 1 / len(members)), method)
    return np.array([roc_auc_score(y[fold == k], p[fold == k]) for k in folds])


def passes(gain: np.ndarray) -> tuple[bool, float]:
    k = len(gain)
    se = gain.std(ddof=1) * np.sqrt(1 / k + 1 / (k - 1))   # Nadeau-Bengio corrected SE, as src/compare.py
    return bool(gain.mean() > 0 and gain.mean() >= 2 * se), (gain.mean() / se if se > 0 else np.inf)


def forward(base: str, candidates: list[str], preds, y, fold, folds, method, log=None) -> list[str]:
    members = [base]
    current = fold_aucs(members, preds, y, fold, folds, method)
    while True:
        trials = {c: fold_aucs(members + [c], preds, y, fold, folds, method) for c in candidates if c not in members}
        if not trials:
            return members
        best = max(trials, key=lambda c: trials[c].mean())
        ok, ratio = passes(trials[best] - current)
        if log is not None:
            for c, a in sorted(trials.items(), key=lambda t: -t[1].mean()):
                log.append({"step": len(members), "blend": " + ".join(members), "try adding": c,
                            "mean AUC": a.mean(), "gain": (a - current).mean(), "gain / SE": passes(a - current)[1]})
        if not ok:
            return members
        members, current = members + [best], trials[best]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("selection")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.selection).read_text())
    names = {"base": spec["base"], **spec["candidates"]}
    runs = {n: _run(r) for n, r in names.items()}
    if len({r.folds_hash for r in runs.values()}) != 1:
        raise SystemExit("runs use different folds")
    oofs = {n: pd.read_parquet(ROOT / r.oof_path).sort_values("id").reset_index(drop=True) for n, r in runs.items()}
    ref = oofs["base"]
    assert all(o.id.equals(ref.id) for o in oofs.values())
    preds = {n: o.pred.to_numpy() for n, o in oofs.items()}
    y, fold = ref.accepted.to_numpy(), ref.fold.to_numpy()
    all_folds = sorted(np.unique(fold))
    cands = list(spec["candidates"])

    log = []
    chosen = forward("base", cands, preds, y, fold, all_folds, spec["method"], log)
    print(pd.DataFrame(log).round({"mean AUC": 5, "gain": 5, "gain / SE": 1}).to_string(index=False))
    final = fold_aucs(chosen, preds, y, fold, all_folds, spec["method"])
    base_auc = fold_aucs(["base"], preds, y, fold, all_folds, spec["method"])
    print(f"\nselected: {chosen}\nselected blend, mean fold AUC {final.mean():.5f} (base {base_auc.mean():.5f}, "
          f"gain {final.mean() - base_auc.mean():+.5f})")

    # Nested: redo the selection without fold k, score its choice on fold k.
    nested, picks = [], []
    for k in all_folds:
        inner = [f for f in all_folds if f != k]
        m = forward("base", cands, preds, y, fold, inner, spec["method"])
        picks.append(m)
        nested.append(fold_aucs(m, preds, y, fold, [k], spec["method"])[0])
    nested = np.array(nested)
    gain = nested - base_auc
    print(f"\nnested estimate: mean AUC {nested.mean():.5f}, gain over base {gain.mean():+.5f} "
          f"({passes(gain)[1]:+.1f} SE, {int((gain > 0).sum())} of {len(gain)} folds up)")
    for k, m in zip(all_folds, picks):
        print(f"  without fold {k}: picked {m}")

    out = ROOT / "configs" / "blends" / f"{Path(args.selection).stem}_result.yaml"
    out.write_text(f"# Chosen by {args.selection} (forward selection, equal weights, {spec['method']}).\n"
                   f"# Selected-folds AUC {final.mean():.5f}; nested estimate {nested.mean():.5f}.\n"
                   + yaml.safe_dump({"runs": [names[m] for m in chosen], "method": spec["method"]}, sort_keys=False))
    print(f"\nwrote {out.relative_to(ROOT)} (run it with make blend to log the blend)")
