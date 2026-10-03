"""Scan for levels whose outcome is (almost) decided: accept rate < 2% or > 98%.

Usage:
    uv run python src/leakage_scan.py

Covers every level of every categorical column, the missing level of every column,
the special numeric values (<= 0), and the engineered flags of src/features.py.
"""
import pandas as pd

from data import FEATURES, NUMERIC, TARGET, load
from features import TRANSFORMS, build

MIN_ROWS = 50          # with 0 exceptions in 50 rows, the 95% upper bound on the exception rate is ~3/50 = 6%
LOW, HIGH = 0.02, 0.98  # "almost decided": at most 1 exception in 50


def levels(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    for c in columns:
        tr, te = train[c], test[c]
        if c in NUMERIC or pd.api.types.is_numeric_dtype(tr) and tr.nunique() > 20:
            keys = {"missing": (tr.isna(), te.isna()), "<= 0": (tr <= 0, te <= 0)}
        else:
            s_tr, s_te = tr.astype("string").fillna("missing"), te.astype("string").fillna("missing")
            keys = {v: (s_tr == v, s_te == v) for v in s_tr.unique()}
        for level, (m_tr, m_te) in keys.items():
            n = int(m_tr.sum())
            if n >= MIN_ROWS:
                yield {"column": c, "level": level, "train rows": n, "accepted": int(train.loc[m_tr, TARGET].sum()),
                       "accept rate": train.loc[m_tr, TARGET].mean(), "test rows": int(m_te.sum())}


if __name__ == "__main__":
    train, test = load()
    raw = pd.DataFrame(levels(train, test, FEATURES))
    flags = ["exempt_code_count", "entity_applicant", "location_unknown", "has_coapplicant", "income_le_0"]
    tr_f, te_f, _ = build(train, test, [n for n in TRANSFORMS if n in flags])
    engineered = pd.DataFrame(levels(tr_f, te_f, flags))
    scan = pd.concat([raw, engineered])
    hits = scan[(scan["accept rate"] < LOW) | (scan["accept rate"] > HIGH)].sort_values("train rows", ascending=False)
    print(f"{len(scan)} levels with >= {MIN_ROWS} rows scanned, {len(hits)} outside [{LOW}, {HIGH}]")
    print(hits.round(4).to_string(index=False))
