# Models

All on the frozen folds, compared with "make compare". The competition model is the best line.

| model | experiment | run | CV AUC | vs tuned LightGBM | public LB |
|---|---|---|---|---|---|
| LightGBM, tuned | configs/experiments/lgbm_optuna_best.yaml | 20261003T084548Z_406e4210 | 0.91905 | reference | 0.92100 |
| CatBoost, untuned, adopted LightGBM features | configs/experiments/catboost.yaml | 20261014T001640Z_6deb776a | 0.91756 | -0.00149 (-1.4 SE, 1 of 5 folds) | not submitted |
| CatBoost, untuned, + raw county and MSA ids | configs/experiments/catboost_county.yaml | 20261014T085450Z_d96c8be4 | 0.91859 | -0.00046 (-0.5 SE, 1 of 5 folds) | not submitted |
| XGBoost, starting values (max_cat_to_onehot 64 carried over from LightGBM) | configs/experiments/xgb.yaml | 20261014T125236Z_f0fe3121 | 0.91649 | -0.00257 (-4.3 SE, 0 of 5 folds) | not submitted |
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

XGBoost, 2026-10-14: 2.3 minutes. Weaker than LightGBM (-0.0026) but less similar than expected: out-of-fold rank correlation 0.961 with LightGBM, 0.956 with CatBoost. Candidate for the final blend selection.

ExtraTrees, 2026-10-14: 0.89786 on raw numerics, 0.90185 on ranked numerics (+0.0040, 8.1 SE, 5 of 5 folds). ExtraTrees draws split thresholds uniformly on a column's range, which skewed columns (loan amount up to 375M) waste; ranks fix it. A 30k-row sample had shown no gain: the full CV decided. Rank correlation with LightGBM 0.785, the most different model of the pool.

Final blend selection, 2026-10-14 ("src/select_blend.py", "configs/blends/final_selection.yaml"):

- Forward selection, equal weights, log-odds, from the tuned LightGBM over 8 candidates: only CatBoost + county is added (+0.00108, 2.2 SE). Every other model lowers an equal-weight blend (XGBoost -0.0004, ExtraTrees -0.0016, logistic -0.0032; LightGBM seeds +0.0003 at most, under the rule).
- Nested estimate (selection redone without each fold): the selection keeps LightGBM alone in 4 of 5 folds, gain 0.00000. Read as low power, not harm: with 4 folds the rule rarely proves a +0.001 gain, and it never chose a worse blend.
- Weighted blend (non-negative least squares on log-odds, "src/weight_blend.py"), nested: +0.00012 (0.5 SE), does not pass. ExtraTrees and the logistic regression get weight 0 in every fold; CatBoost only ~0.08 (least squares targets probability error, not ranking); the weights jump between LightGBM seeds.
- Candidate final model: LightGBM + CatBoost county, 50/50 log-odds (run 20261014T115111Z_918eb62b, CV 0.92014). To confirm on fresh splits before adoption ("make fresh-blend", about 8 hours).
