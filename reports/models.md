# Models

All on the frozen folds, compared with "make compare". The competition model is the best line.

| model | experiment | run | CV AUC | vs tuned LightGBM | public LB |
|---|---|---|---|---|---|
| LightGBM, tuned | configs/experiments/lgbm_optuna_best.yaml | 20261003T084548Z_406e4210 | 0.91905 | reference | 0.92100 |
| CatBoost, untuned, adopted LightGBM features | configs/experiments/catboost.yaml | 20261014T001640Z_6deb776a | 0.91756 | -0.00149 (-1.4 SE, 1 of 5 folds) | not submitted |
| CatBoost, untuned, + raw county and MSA ids | configs/experiments/catboost_county.yaml | 20261014T085450Z_d96c8be4 | 0.91859 | -0.00046 (-0.5 SE, 1 of 5 folds) | not submitted |
| XGBoost, starting values (max_cat_to_onehot 64 carried over from LightGBM) | configs/experiments/xgb.yaml | 20261014T125236Z_f0fe3121 | 0.91649 | -0.00257 (-4.3 SE, 0 of 5 folds) | not submitted |
| ExtraTrees, Optuna best (max_features 0.35, leaf 5, depth 20) | configs/experiments/extratrees_optuna_best.yaml | 20261019T060754Z_7bc88a0b | 0.91484 | -0.00421 (-5.7 SE, 0 of 5 folds) | not submitted |
| MLP with embeddings, starting values | configs/experiments/mlp.yaml | 20261031T101220Z_4cee5616 | 0.89819 | -0.02086 (-24.0 SE, 0 of 5 folds) | not submitted |
| Blend LightGBM + CatBoost county, equal weights, log-odds | configs/blends/lgbm_catboost_logit.yaml | 20261014T115111Z_918eb62b | 0.92014 | +0.00108 (2.2 SE, 5 of 5 folds); fresh splits +0.00083 (3.8 SE, 14 of 15 folds) | 0.92118 |
| Blend LightGBM seeds 0-4, equal weights, log-odds | configs/blends/lgbm_seeds5.yaml | 20261014T122347Z_f574a592 | 0.91956 | +0.00050 (1.8 SE, 5 of 5 folds) | not submitted |
| Logistic regression (L2, quantile splines, one-hot, crossings, county rate) | configs/experiments/logreg.yaml | 20261013T154151Z_83e8fb23 | 0.90332 | -0.01573 (-21.4 SE, 5 of 5 folds) | not submitted |
| Logistic regression, L1 (lasso), same features | configs/experiments/logreg_l1.yaml | 20261018T172507Z_6c8c7142 | 0.90379 | -0.01526 (-18.9 SE, 5 of 5 folds) | not submitted |
| Logistic regression, elastic net, same features | configs/experiments/logreg_enet.yaml | 20261018T185310Z_11862916 | 0.90362 | -0.01543 (-19.2 SE, 5 of 5 folds) | not submitted |

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
- Candidate final model: LightGBM + CatBoost county, 50/50 log-odds (run 20261014T115111Z_918eb62b, CV 0.92014). Confirmed on fresh splits, see "Fresh-split check of the blend" below.

L1 and elastic net, 2026-10-18 (TODO 7.22):

- Same features and encoding as the L2 model; the penalty strength and the L1 share are chosen by the same inner CV. L1 needs the saga solver (30 s to 5 min per fit against 1 to 5 s for L2), so C is searched on [0.01, 3] only; the timing that justified it is in the two config files.
- L1 vs L2: +0.00047, 2.5 SE, 5 of 5 folds: passes the rule. Chosen C = 0.3 in every fold.
- Elastic net (L1 share 0.25 / 0.5 / 0.75) vs L2: +0.00030, 1.9 SE: rejected. Vs L1: -0.00017, -2.6 SE, 0 of 5 folds up. The inner CV picked the most-L1 mix (0.75) in every fold, so the pure L1 end is where the gain is.
- The gain is tiny and changes nothing for the competition: the three linear models rank rows almost identically (rank correlation 0.996 to 0.999), and their rank correlation with the tuned LightGBM stays at 0.89, where the L2 model already got weight 0 in the blend.
- Removed by L1 (every column of the variable at 0 in all 5 folds): exempt_code_count, applicant_race_observed, negative_amortization, construction_method, entity_applicant, has_coapplicant. In some folds only: the co-applicant ethnicity / race / sex "observed" fields, applicant_ethnicity_observed, manufactured_home_secured_property_type, loan_type. About 270 of 536 encoded columns are 0 in each fold, mostly single spline pieces and category levels.
- A zero means "redundant given the other columns", not "no link with the decision": exempt_code_count repeats is_exempt_record, entity_applicant and has_coapplicant repeat the "no co-applicant" levels of the co-applicant fields, construction_method repeats derived_dwelling_category. Between two near-copies L1 keeps one, and which one can change between folds. The elastic net removes a subset of the same list (it keeps exempt_code_count).
- Coefficients of every fold model: reports/coefs/<run_id>.csv.

Blend selection, second pass, 2026-10-19 ("configs/blends/final_selection_v2.yaml"): tuned ExtraTrees and the L1 logistic regression in place of the older versions.

- Unchanged: LightGBM + CatBoost county is selected again (+0.00108, 2.2 SE); nothing else passes after it.
- The tuned ExtraTrees now helps a little but under the rule: +0.00039 (1.2 SE) next to LightGBM alone, +0.00025 (1.3 SE, 4 of 5 folds) on top of LightGBM + CatBoost. Its 856 out-of-fold predictions of exactly 0 are all real denials, so the log-odds clipping helps it (rank blend: +0.00010).
- Nested estimate: the selection redone without each fold keeps LightGBM alone in 4 of 5 folds, gain 0.00000. The LightGBM + CatBoost blend rests on a 2.2 SE result that the nested selection does not reproduce: the fresh-split check (7.25) decides.

Census tract rate, 2026-10-19 ("configs/experiments/lgbm_tract_rate.yaml", run 20261019T064454Z_561f4e1b):

- The tract is rebuilt from the county + the 6 tract fields (features.py tract_key; 50,587 tracts in train, 76% of test rows in a tract seen in train). Its accept rate, target-encoded inside each fold and shrunk towards the county's (m = 20), is added to the tuned LightGBM.
- Rejected: 0.91841, -0.00064 (-1.4 SE, 1 of 5 folds up, bootstrap CI [-0.00129, -0.00003]). Alone the rate scores AUC 0.60, but most tracts have 1 to 5 training rows: the rate is mostly noise around the county's, and the model fits it (tree counts 120-186 instead of 80-110).
- Not retried with other smoothing values (a value chosen on these folds would flatter the result). Untested: CatBoost with tract_key as a native category (its ordered target statistics are what made the raw county useful to it).

Hill climbing, 2026-10-19 (TODO 11.4, "src/hill_climb.py", "configs/blends/hill_climb.yaml"):

- Greedy forward selection with replacement on the fold AUC (log-odds), from the tuned LightGBM over CatBoost county, XGBoost and the tuned ExtraTrees; the weights maximise AUC directly. Scored nested: the climb is redone without each fold and scored on it.
- Weights found: LightGBM 0.40, CatBoost 0.40, ExtraTrees 0.20, XGBoost 0, the same in 3 of the 5 nested climbs (the other two: 0.50 / 0.33 / 0.17 and 1/3 each). XGBoost is never added.
- Nested AUC 0.92039. Vs the current blend (LightGBM + CatBoost 50/50): +0.00026, 1.4 SE, 4 of 5 folds: does not pass; the 50/50 blend stays. Vs LightGBM alone: +0.00134, 2.9 SE, 5 of 5 folds: passes.
- The second line is new evidence for blending itself: the equal-weight forward selection, gated by the rule at each step, kept LightGBM alone in 4 of 5 nested folds (gain 0); the free climb, scored the same nested way, beats LightGBM alone on every fold.

Fresh-split check of the blend, finished 2026-10-19 (TODO 7.25, "make fresh-blend", "reports/tuning/fresh_blend_lgbm_catboost.log", per-fold results in "reports/tuning/fresh_blend_lgbm_optuna_best_catboost_county_vs_lgbm_optuna_best.csv"):

| split seed | LightGBM | blend 50/50 | gain | folds up |
|---|---|---|---|---|
| 1 | 0.91899 | 0.91992 | +0.00093 | 5 of 5 |
| 2 | 0.91930 | 0.91994 | +0.00064 | 4 of 5 |
| 3 | 0.91914 | 0.92005 | +0.00091 | 5 of 5 |

- All 15 paired folds: +0.00083, corrected SE 0.00022, 3.8 SE, 14 of 15 folds up (worst fold -0.00026): CONFIRMED under the rule. These splits were never used for any choice.
- Adopted: LightGBM + CatBoost county, 50/50 log-odds, is the best model so far. It is the blend of submission 5 (LB 0.92118).
- Fresh gain +0.00083 against +0.00108 on the frozen folds: choosing the blend on those folds flattered it by ~0.00025. The nested selection's gain of 0 was low power, not absence of an effect.
- Scope: only this blend is confirmed. The 0.4 / 0.4 / 0.2 blend with ExtraTrees (11.4) was not on these splits. Any later change to the blend needs its own check.
MLP with embeddings, 2026-10-31 (TODO 7.24, "src/mlp.py", run 20261031T101220Z_4cee5616, 7 minutes on CPU):

- CV 0.89819: below the L1 logistic regression (-0.00560, -5.8 SE) and the tuned ExtraTrees (-0.01665, -13.1 SE). A net that loses to a linear model on the same data is underperforming, not at its limit.
- Overfits early: on fold 0 the inner AUC peaks at epoch 10 (0.89906) while the training loss keeps falling (0.218 to 0.167 by epoch 25). Epochs chosen: 10 to 16 of 60. Training stopped with the learning rate still at ~60% of its maximum, so the low-rate end of the cosine was never reached.
- Different from the trees (rank correlation 0.825 with LightGBM, 0.831 with CatBoost), but too weak to help: added to LightGBM + CatBoost, weight 0.05 gives +0.00004 (0.6 SE), 0.10 gives -0.00006.
