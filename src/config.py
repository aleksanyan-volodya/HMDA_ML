"""Experiment configs: configs/base.yaml lists every allowed key with its default, and an
experiment file may only override keys that already exist there (an unknown key is an error, so a
typo cannot become a silent new setting).

Reads configs/base.yaml and the experiment file. Pipeline.
"""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "base.yaml"


def _merge(base: dict, override: dict, path: str = "") -> dict:
    merged = dict(base)
    for key, value in override.items():
        where = f"{path}{key}"
        if key not in base:
            raise KeyError(f"unknown config key '{where}' (not defined in base.yaml)")
        if isinstance(base[key], dict) and isinstance(value, dict):
            merged[key] = _merge(base[key], value, where + ".")
        else:
            merged[key] = value
    return merged


def config_hash(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_config(experiment: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(BASE.read_text()) or {}
    if experiment is not None:
        cfg = _merge(cfg, yaml.safe_load(Path(experiment).read_text()) or {})
    return cfg
