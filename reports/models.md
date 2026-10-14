# Models

All on the frozen folds, compared with "make compare". The competition model is the best line.

| model | experiment | run | CV AUC | vs tuned LightGBM | public LB |
|---|---|---|---|---|---|
| LightGBM, tuned | configs/experiments/lgbm_optuna_best.yaml | 20261003T084548Z_406e4210 | 0.91905 | reference | 0.92100 |
| CatBoost, untuned, adopted LightGBM features | configs/experiments/catboost.yaml | 20261014T001640Z_6deb776a | 0.91756 | -0.00149 (-1.4 SE, 1 of 5 folds) | not submitted |
| CatBoost, untuned, + raw county and MSA ids | configs/experiments/catboost_county.yaml | 20261014T085450Z_d96c8be4 | 0.91859 | -0.00046 (-0.5 SE, 1 of 5 folds) | not submitted |
| Blend LightGBM + CatBoost county, equal weights, log-odds | configs/blends/lgbm_catboost_logit.yaml | 20261014T115111Z_918eb62b | 0.92014 | +0.00108 (2.2 SE, 5 of 5 folds) | not submitted |
| Blend LightGBM seeds 0-4, equal weights, log-odds | configs/blends/lgbm_seeds5.yaml | 20261014T122347Z_f574a592 | 0.91956 | +0.00050 (1.8 SE, 5 of 5 folds) | not submitted |
| Logistic regression (L2, quantile splines, one-hot, crossings, county rate) | configs/experiments/logreg.yaml | 20261013T154151Z_83e8fb23 | 0.90332 | -0.01573 (-21.4 SE, 5 of 5 folds) | not submitted |

Logistic regression, 2026-10-13:

- Expected 0.89 to 0.91 before the run. The 0.016 gap to LightGBM is what the interactions trees find, beyond the crossings given to the linear model, are worth.
- No use in a blend: out-of-fold rank correlation with LightGBM 0.89, and every logistic weight lowers the pooled out-of-fold AUC (0.91884 alone, 0.91857 at weight 0.1, 0.91442 at 0.5), even with weights chosen on the same predictions.
- Kept as the interpretable model: base of the fairness analysis (readable coefficients for protected attributes and their proxies, leakage ablation).

CatBoost, 2026-10-14 (2 to 3 hours per run):

- The raw county / MSA ids help CatBoost (+0.00103 over the same model without them, 3.6 SE, 5 of 5 folds) while they made LightGBM overfit (-0.009). CatBoost encodes categories with ordered target statistics that never use a row's own label.
- Untuned, it is level with the tuned LightGBM (-0.0005, inside the noise), and its errors differ: out-of-fold rank correlation 0.947.

Blending, 2026-10-14 ("src/blend.py", "make blend"):

- LightGBM + CatBoost county, equal weights: +0.00104 (rank) and +0.00108 (log-odds) over LightGBM alone, both 2.2 SE. Log-odds vs rank: +0.00004 (1.1 SE), no difference. Log-odds is used from now on: a fixed weighted sum of log-odds keeps SHAP values additive.
- Not the final blend: the final one is chosen once, when every candidate is in.

Seeds, 2026-10-14:

- Single-seed CV of the tuned LightGBM: 0.91905 (seed 0), 0.91858, 0.91872, 0.91837, 0.91878 (seeds 1-4); mean 0.9187, std 0.00025. Seed 0 is the best of the five because the Optuna search was run with seed 0 only: the tuned model's CV is flattered by about 0.0003.
- 5-seed average vs seed 0: +0.00050, 1.8 SE, 5 of 5 folds: rejected by the rule (fixed before the run). Seed 0 stays the LightGBM candidate; the average stays in the log for the final blend selection.
