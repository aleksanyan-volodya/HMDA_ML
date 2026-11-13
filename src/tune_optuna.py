"""Joint search over several settings with Optuna (TPE sampler), scored by the usual CV.

Usage:
    uv run python src/tune_optuna.py configs/sweeps/<name>.yaml

- The first trial is the experiment unchanged, so the search ends at the reference or better.
- Trials are stored in reports/tuning/<study_name>.db: stop with Ctrl-C and relaunch to resume.
- Trials are NOT logged in results/runs.csv. At the end the best settings are written to
  configs/experiments/<study_name>_best.yaml, to be run and compared like any experiment.
- The best of many trials on the same folds is optimistic: it must still pass "make compare"
  and the fresh-folds check before being adopted.
- Optional in the sweep file, for models too slow for a full CV per trial: search_folds (score these
  outer folds only) and trial_overrides (settings applied to the trials only, never to the written
  best experiment). A trial score is then only good for ranking settings: its SE below means nothing,
  and the best settings must go through a full CV.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import optuna
import yaml

from config import load_config
from cv import run_cv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tuning"


def suggest(trial: optuna.Trial, name: str, spec: dict):
    if spec["type"] == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if spec["type"] == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if spec["type"] == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"unknown type {spec['type']!r} for {name}")


def corrected_ratio(base: list[float], new: list[float]) -> tuple[float, float]:
    """Paired mean gain and its Nadeau-Bengio corrected SE (same formula as src/compare.py)."""
    d = np.array(new) - np.array(base)
    k = len(d)
    return d.mean(), d.std(ddof=1) * np.sqrt(1 / k + 1 / (k - 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep")
    args = parser.parse_args()
    sweep = yaml.safe_load(Path(args.sweep).read_text())
    base, section, space = load_config(ROOT / sweep["experiment"]), sweep["section"], sweep["space"]
    unknown = [k for k in space if k not in base[section]]
    if unknown:
        raise KeyError(f"{unknown} not defined in {section} of base.yaml")

    OUT.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=sweep["study_name"], direction="maximize",
                                storage=f"sqlite:///{OUT / (sweep['study_name'] + '.db')}",
                                sampler=optuna.samplers.TPESampler(seed=sweep["seed"]), load_if_exists=True)
    if not study.trials:
        study.enqueue_trial({k: base[section][k] for k in space})   # trial 0 = the experiment unchanged

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base)
        for part, values in sweep.get("trial_overrides", {}).items():
            cfg[part].update(values)
        for name, spec in space.items():
            cfg[section][name] = suggest(trial, name, spec)
        _, _, fold_auc, _ = run_cv(cfg, only_folds=sweep.get("search_folds"))
        trial.set_user_attr("fold_auc", fold_auc)
        return float(np.mean(fold_auc))

    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    # Optional wall-clock cap (hours): the search stops after the trial running at that time,
    # so a night run ends by morning and still writes the best settings below.
    timeout = sweep["timeout_hours"] * 3600 if "timeout_hours" in sweep else None
    study.optimize(objective, n_trials=max(0, sweep["n_trials"] - len(finished)), timeout=timeout)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.to_csv(OUT / f"{sweep['study_name']}.csv", index=False)       # readable record of every trial
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    reference, best = done[0], study.best_trial
    gain, se = corrected_ratio(reference.user_attrs["fold_auc"], best.user_attrs["fold_auc"])
    print(f"\n{len(done)} trials. reference (trial 0): {reference.value:.5f}   best (trial {best.number}): {best.value:.5f}")
    print(f"best vs reference: gain {gain:+.5f}, corrected SE {se:.5f}" + (f" -> {gain / se:+.1f} SE" if se > 0 else ""))
    print("best settings:", best.params)

    experiment = yaml.safe_load((ROOT / sweep["experiment"]).read_text())
    experiment[section] = best.params
    out = ROOT / "configs" / "experiments" / f"{sweep['study_name']}_best.yaml"
    out.write_text(f"# Best of {len(done)} Optuna trials ({args.sweep}), trial {best.number}, CV AUC {best.value:.5f}.\n"
                   "# Selected on the frozen folds, so optimistic: adopt only after make compare + the fresh-folds check.\n"
                   + yaml.safe_dump(experiment, sort_keys=False))
    print(f"wrote {out.relative_to(ROOT)}")
