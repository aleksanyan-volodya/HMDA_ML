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
| 4 | 2026-10-03 | `20261003T084548Z_406e4210.csv` | 500b562 (dirty) | `experiments/lgbm_optuna_best.yaml` | 0.9191 | 0.0021 | 0.9210 | +0.0020 | Tuned LightGBM (Optuna, confirmed on fresh splits: +0.0019 over the untuned model). Submission 3 scored 0.9185 for CV 0.9169. If the gain transfers, LB is near 0.920 to 0.921; anything in [0.908, 0.930] (CV +- 0.011) is consistent. The public LB cannot resolve a +0.002 change (noise ~0.0055), so this upload checks the transfer, it does not prove the gain. | Confirmed: LB +0.0025 over submission 3 for a CV gain of +0.0021; gap +0.0020, in line with the previous +0.0025 and +0.0016. |
| 5 | 2026-10-19 | `20261014T115111Z_918eb62b.csv` | 2a6846d (dirty) | `blends/lgbm_catboost_logit.yaml` | 0.9201 | 0.0022 | 0.9212 | +0.0011 | LightGBM tuned + CatBoost county, 50/50 log-odds blend: +0.0011 CV over submission 4 (2.2 SE, 5 of 5 folds; not yet confirmed on fresh splits, nested selection kept it in 1 of 5 folds). Previous gaps +0.0016 to +0.0025, so LB near 0.922 to 0.923 if it transfers; anything in [0.909, 0.931] (CV +- 0.011) is consistent. The public LB (noise ~0.0055) cannot resolve a +0.001 change: an LB below 0.9210 would not refute the blend, and the final choice stays on CV (11.12). | Consistent: LB 0.92118, +0.0002 over submission 4 for a CV gain of +0.0011, inside the public-LB noise; gap +0.0010, a bit under the previous +0.0016 to +0.0025 but within noise. Neither confirms nor refutes the blend: the fresh-split check (7.25) decides. |
| 6 | 2026-11-09 | `20261031T220901Z_1ff742f2.csv` | c6d4581 (dirty) | `blends/lgbm_cat_et_xgb.yaml` | 0.9205 | 0.0022 | 0.9216 | +0.0010 | 4-model blend (LightGBM 2/6, CatBoost county 2/6, tuned ExtraTrees 1/6, tuned XGBoost 1/6, log-odds): +0.0004 CV over submission 5, confirmed on fresh splits (+0.0007, 7.6 SE, 15 of 15). Previous gaps +0.0010 to +0.0025, so LB near 0.921 to 0.923 if it transfers; anything in [0.909, 0.932] (CV +- 0.011) is consistent. The public LB (noise SD ~0.0038 for one score, ~0.0014 for a paired gap, measured 2026-10-31) cannot resolve +0.0004: an LB below submission 5 (0.92118) would not refute it. This upload checks the transfer of the new members' test predictions; the final choice stays on CV (11.12). | Consistent: LB 0.92158, +0.0004 over submission 5 for a CV gain of +0.0004; gap +0.0010, in line with submission 5's +0.0010. The test predictions of the new members transfer. The equal LB and CV gains are partly luck: the public LB cannot resolve +0.0004. |

## Final submissions (TODO 11.12, chosen on CV, 2026-11-09)

Kaggle ranks each team on the private 70% of test with the 2 submissions it selects. Both are chosen on CV and fresh-split evidence, not on the public LB, which cannot separate them (noise SD ~0.0038 for one score, ~0.0014 for a paired gap).

| | submission | model | CV | fresh splits | public LB |
|---|---|---|---|---|---|
| 1, main | 6, `20261031T220901Z_1ff742f2.csv` | LightGBM 2/6, CatBoost county 2/6, ExtraTrees 1/6, XGBoost 1/6, log-odds | 0.92054 | +0.00070 over the 50/50 blend, 7.6 SE, 15 of 15 folds | 0.92158 |
| 2, safety | 5, `20261014T115111Z_918eb62b.csv` | LightGBM + CatBoost county, 50/50, log-odds | 0.92014 | +0.00083 over LightGBM alone, 3.8 SE, 14 of 15 folds | 0.92118 |

- Main: the best CV model, confirmed on splits never used for any choice.
- Safety: the best model with fewer moving parts, also confirmed on fresh splits. It does not depend on the two models tuned last (ExtraTrees, XGBoost). The two rank test rows very much alike (rank correlation 0.9974), so this is insurance against a fault in submission 6, not a different bet.
- Not chosen: LightGBM alone (below both on CV and on fresh splits), the stack and the other blends (not confirmed on fresh splits).
- To change only if a later model is confirmed on fresh splits under the same rule. Selection on Kaggle is done by hand, by the account owner.
