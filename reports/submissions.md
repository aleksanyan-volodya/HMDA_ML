# Submissions

Metric: ROC AUC
Rule: the row is written **before** uploading; "LB", "gap" and "verdict" after.
gap = LB - CV. "SE" is the naive per-fold std/sqtr(k)
Public/private split: public LB = 30% of test (~13,500 rows), final ranking on the other 70% (~31,500). Daily submission limit: 12.

| n | date | file | commit | config | CV | SE | LB | gap | hypothesis | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-26 | `20260925T220117Z_13631089.csv` | e1cf839 (dirty) | `experiments/constant.yaml` | 0.5000 | 0.0000 | 0.5000 | 0.0000 | Pipeline check: It must be 0.5 | Confirmed. Pipeline correct |
| 2 | 2026-09-26 | `20260925T234636Z_4626a824.csv` | d604f55 (dirty) | `experiments/lgbm_baseline.yaml` | 0.9035 | 0.0025 | 0.9060 | +0.0025 | LightGBM baseline. LB in [0.89, 0.915] (CV in public-LB noise ~0.011) confirms CV transfers to test. Outside that range there is a train/test shift or a pipeline bug, to be investigated. | Confirmed: gap well inside public-LB noise, CV transfers to test. |
