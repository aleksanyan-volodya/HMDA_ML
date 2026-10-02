# Submissions

Metric: ROC AUC
Rule: the row is written **before** uploading; "LB", "gap" and "verdict" after.
gap = LB - CV. "SE" is the naive per-fold std/sqtr(k)
Public/private split: public LB = 30% of test (~13,500 rows), final ranking on the other 70% (~31,500). Daily submission limit: 12.

| n | date | file | commit | config | CV | SE | LB | gap | hypothesis | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-26 | `20260925T220117Z_13631089.csv` | e1cf839 (dirty) | `experiments/constant.yaml` | 0.5000 | 0.0000 | 0.5000 | 0.0000 | Pipeline check: It must be 0.5 | Confirmed. Pipeline correct |
| 2 | 2026-09-26 | `20260925T234636Z_4626a824.csv` | d604f55 (dirty) | `experiments/lgbm_baseline.yaml` | 0.9035 | 0.0025 | 0.9060 | +0.0025 | LightGBM baseline. LB in [0.89, 0.915] (CV in public-LB noise ~0.011) confirms CV transfers to test. Outside that range there is a train/test shift or a pipeline bug, to be investigated. | Confirmed: gap well inside public-LB noise, CV transfers to test. |
| 3 | 2026-10-02 | `20261002T141818Z_f7f2dc68.csv` | 10b3725 (dirty) | `experiments/lgbm_no_county_msa.yaml` | 0.9169 | 0.0021 | 0.9185 | +0.0016 | LightGBM with blocks 0 + 2 and without the raw county / MSA ids (+0.0134 CV over submission 2). If CV transfers as before (gap +0.0025), LB is near 0.919; anything in [0.906, 0.928] (CV +- 0.011) is consistent. Below 0.906: the CV gain did not transfer, investigate. | Confirmed: LB +0.0125 over submission 2 for a CV gain of +0.0134. The CV gain transferred almost fully. |
