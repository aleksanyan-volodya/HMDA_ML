"""Add records of every CV run.

One row per run in reports/runs.csv, plus the run's predictions:
    oof/<run_id>.parquet    id, fold, accepted, pred   (105,000 rows)
    preds/<run_id>.parquet  id, pred                   (45,000 rows)
"""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from config import config_hash

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "runs.csv"
CODE_PATHS = ["src", "configs", "pyproject.toml", "uv.lock"]
LIBS = ["numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def log_run(cfg: dict, experiment: str, features: list[str], folds_hash: str,
            fold_auc: list[float], wall_seconds: float,
            oof_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    started = datetime.now(timezone.utc)
    cfg_hash = config_hash(cfg)
    # Sortable by time, and the hash prefix tells at a glance which runs share a config.
    run_id = f"{started:%Y%m%dT%H%M%SZ}_{cfg_hash[:8]}"

    oof_path = ROOT / "oof" / f"{run_id}.parquet"
    pred_path = ROOT / "preds" / f"{run_id}.parquet"
    for folder in (oof_path.parent, pred_path.parent):   # not tracked by git: missing in a fresh clone
        folder.mkdir(exist_ok=True)
    oof_df.to_parquet(oof_path, index=False)
    test_df.to_parquet(pred_path, index=False)

    auc = pd.Series(fold_auc)
    row = {
        "run_id": run_id,
        "utc": started.isoformat(timespec="seconds"),
        "experiment": experiment,
        "commit": _git("rev-parse", "--short", "HEAD"),
        # Untracked files count: a new, uncommitted src file changes what the run computes
        "dirty": bool(_git("status", "--porcelain", "--", *CODE_PATHS)),
        "config_hash": cfg_hash,
        # Full merged config, so the run is reproducible even when it was dirty.
        "config_json": json.dumps(cfg, sort_keys=True),
        "feature_hash": hashlib.sha256("\n".join(features).encode()).hexdigest(),
        "n_features": len(features),
        "folds_hash": folds_hash,
        # Raw per-fold scores: any uncertainty measure can be recomputed from them later
        "fold_auc": json.dumps([round(a, 8) for a in fold_auc]),
        "mean_auc": round(auc.mean(), 8),
        "naive_se": round(auc.std(ddof=1) / len(auc) ** 0.5, 8),
        "wall_seconds": round(wall_seconds, 1),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "pred_path": str(pred_path.relative_to(ROOT)),
        "versions": json.dumps({"python": platform.python_version(), **{m: version(m) for m in LIBS}}),
    }

    new = pd.DataFrame([row])
    if RUNS.exists():
        header = pd.read_csv(RUNS, nrows=0).columns.tolist()
        # A changed schema would silently misalign columns in an appended CSV
        assert header == list(row), f"runs.csv columns changed: {header} vs {list(row)}"
    new.to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
    return run_id
