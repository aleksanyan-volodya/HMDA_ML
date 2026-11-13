"""Frozen CV folds: built once into data/processed/folds.parquet, then loaded everywhere.

The seed and fold count are stored inside the parquet file, and loading refuses a file built with
other settings than configs/base.yaml.

Usage:
    uv run python src/folds.py                       # build (refuses to overwrite a different split)
    uv run python src/folds.py --allow-fold-change   # deliberately rebuild with new settings

Reads data/raw/train.csv; writes data/processed/folds.parquet. Pipeline. Report: section 5.2.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold

from config import load_config
from data import TARGET, load

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "data" / "processed" / "folds.parquet"


def folds_hash(folds: pd.DataFrame) -> str:
    # Hash of the (id, fold) content, not of the file bytes: parquet files embed the
    # writer's library version, so identical splits could otherwise hash differently.
    content = folds.sort_values("id")[["id", "fold"]].to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()


def _stored_settings() -> dict:
    meta = pq.read_schema(FOLDS).metadata
    return {"fold_seed": int(meta[b"fold_seed"]), "n_folds": int(meta[b"n_folds"])}


def build(cfg: dict, allow_change: bool = False) -> pd.DataFrame:
    wanted = {"fold_seed": cfg["fold_seed"], "n_folds": cfg["n_folds"]}
    if FOLDS.exists() and _stored_settings() != wanted and not allow_change:
        sys.exit(f"folds.parquet was built with {_stored_settings()}, config says {wanted}. "
                 "Rerun with --allow-fold-change if this is deliberate.")

    train, _ = load()
    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["fold_seed"])
    folds = pd.DataFrame({"id": train["id"], "fold": -1})
    for k, (_, valid_idx) in enumerate(skf.split(train, train[TARGET])):
        folds.loc[valid_idx, "fold"] = k

    table = pa.Table.from_pandas(folds, preserve_index=False)
    table = table.replace_schema_metadata(
        {**table.schema.metadata, **{k.encode(): str(v).encode() for k, v in wanted.items()}})
    FOLDS.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, FOLDS)
    return folds


def load_folds(cfg: dict) -> pd.DataFrame:
    """The only way the rest of the pipeline gets folds. Aborts on a settings mismatch."""
    wanted = {"fold_seed": cfg["fold_seed"], "n_folds": cfg["n_folds"]}
    if _stored_settings() != wanted:
        sys.exit(f"folds.parquet was built with {_stored_settings()}, config says {wanted}. "
                 "Rebuild deliberately with: python src/folds.py --allow-fold-change")
    return pd.read_parquet(FOLDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-fold-change", action="store_true")
    args = parser.parse_args()
    folds = build(load_config(), allow_change=args.allow_fold_change)
    print(f"wrote {FOLDS.relative_to(ROOT)}  rows={len(folds)}  hash={folds_hash(folds)}")
