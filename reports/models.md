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
| MLP, final (no raw ids, dropout 0.4, weight decay 0.1, numeric bins, 60 epochs) | configs/experiments/mlp_combo_e60.yaml | 20261031T130048Z_752fd022 | 0.91003 | -0.00903 (-12.8 SE, 0 of 5 folds) | not submitted |
| MLP, 30 epochs, no raw county / MSA ids (best variant) | configs/experiments/mlp_no_ids.yaml | 20261031T110304Z_0de0f840 | 0.90758 | -0.01147 (-9.5 SE, 0 of 5 folds) | not submitted |
| XGBoost, tuned | configs/experiments/xgb_optuna_best.yaml | 20261031T215118Z_05b22e57 | 0.91848 | -0.00057 (-1.0 SE, 2 of 5 folds) | not submitted |
| Blend LightGBM 2/6, CatBoost county 2/6, ExtraTrees 1/6, XGBoost 1/6, log-odds (adopted 2026-11-01) | configs/blends/lgbm_cat_et_xgb.yaml | 20261031T220901Z_1ff742f2 | 0.92054 | +0.00149 vs LightGBM; fresh splits +0.00070 over the 50/50 blend (7.6 SE, 15 of 15) | not submitted |
| CatBoost, Optuna winner (rejected, untuned stays) | configs/experiments/catboost_optuna_best.yaml | 20261101T090757Z_9ef0b007 | 0.91837 | -0.00068 (-0.9 SE, 2 of 5 folds) | not submitted |
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

MLP variants, 2026-10-31 (TODO 7.24). One change each, on the 30-epoch schedule; each reason and prediction was written before the run, in its config file. Reference: mlp_e30, itself compared with the first run.

| variant | run | CV AUC | gain vs reference | gain / corrected SE | folds up | vs L1 | predicted |
|---|---|---|---|---|---|---|---|
| mlp_e30: schedule 60 -> 30 epochs | 20261031T105421Z_bfd111f3 | 0.89917 | +0.00098 | 1.3 (rejected) | 4 | -0.00462 | better: wrong |
| mlp_no_ids: raw county / MSA removed | 20261031T110304Z_0de0f840 | 0.90758 | +0.00841 | 5.7 | 5 | +0.00379 (2.7 SE) | better, least sure: right, and the largest gain |
| mlp_reg: dropout 0.4, weight decay 0.1 | 20261031T111436Z_a38395ae | 0.90415 | +0.00498 | 3.4 | 5 | +0.00036 | better: right |
| mlp_deep: 4 layers of 128 | 20261031T112551Z_97154e36 | 0.90368 | +0.00450 | 5.1 | 5 | -0.00012 | better than the reference: right |
| mlp_small: layers 128, 64 | 20261031T111916Z_8aecb716 | 0.90251 | +0.00333 | 3.4 | 5 | -0.00128 | better: right |
| mlp_numeric: 8 bins per numeric column | 20261031T113420Z_f81dac67 | 0.90021 | +0.00104 | 2.3 | 5 | -0.00358 | the biggest gain: wrong |

- The raw ids behave for the net as for LightGBM (dropping them: +0.0084 here, +0.009 there), not as for CatBoost. The embeddings memorised counties.
- The schedule was not the limit: with the learning rate decayed to 0 the inner AUC stays flat after epoch ~11. The early peak is overfitting, not a too-high rate.
- The numeric bins reach a higher peak much earlier (fold 0, inner AUC at epoch 10: 0.9040 against 0.8988), then overfit faster: loss 0.155 at the end against 0.172. They may be worth more once the net is regularised: the combined run tests it.
- All four changes that limit memorisation pass; regularisation moves the chosen epochs from 10-16 to 25-30.
- Blend, added to LightGBM + CatBoost: see the nested result under "MLP, combined runs" (mlp_no_ids +0.00016 at weight 0.1, 1.5 SE; at weight 0.05 +0.00013 reads 2.7 SE on the same folds, but the weight was chosen on those folds; the others at most +0.00007).

MLP, combined runs, 2026-10-31 (reference: mlp_no_ids, CV 0.90758):

| run | changes | CV AUC | gain vs mlp_no_ids | gain / corrected SE | folds up | vs L1 |
|---|---|---|---|---|---|---|
| mlp_combo, 20261031T115412Z_33a6dc74 | no ids, dropout 0.4, weight decay 0.1, 8 numeric bins | 0.90935 | +0.00177 | 1.0 (rejected) | 4 | +0.00556 (5.1 SE) |
| mlp_combo_deep, 20261031T120130Z_5c758eab | same, layers 128 x 4 | 0.90801 | +0.00042 | 0.2 (rejected) | 3 | +0.00421 (3.7 SE) |

- Highest score of the MLP runs, but mlp_combo does not pass the rule over mlp_no_ids: mlp_no_ids stayed the MLP on record at this step (superseded by the 60-epoch run below).
- The regularisation fixed the overfitting: with mlp_combo the inner AUC now peaks at the end of the schedule (chosen epochs 24 to 30 of 30; fold 0: 0.91334 at epoch 26). The numeric bins needed it: alone they gained +0.00104.
- A stronger net is less useful to the blend: rank correlation with LightGBM 0.870 for mlp_no_ids, 0.926 for mlp_combo.
- Blend test on LightGBM + CatBoost, weight chosen without the scored fold: mlp_no_ids +0.00015 (1.6 SE, 4 of 5 folds): does not pass; mlp_combo gets weight 0 in every fold. Hill climbing with mlp_no_ids among the candidates keeps LightGBM 0.40, CatBoost 0.40, ExtraTrees 0.20 and no MLP.

MLP, last run, 2026-10-31 ("configs/experiments/mlp_combo_e60.yaml", run 20261031T130048Z_752fd022, 18 minutes on CPU):

- mlp_combo with the schedule doubled: CV 0.91003. Vs mlp_combo +0.00067 (1.0 SE, rejected); vs mlp_no_ids +0.00244 (2.4 SE, passes): this run is the MLP on record.
- The budget is no longer the limit: chosen epochs 31 to 47 of 60, and on fold 0 the inner AUC is flat from about epoch 40 (0.9147). Training longer would not help this network.
- Blend on LightGBM + CatBoost, weight chosen without the scored fold (0.025 to 0.05): +0.00001 (0.5 SE), does not pass. Rank correlation with LightGBM 0.910.
- Conclusion (TODO 7.24 closed): after 10 runs a tuned net with embeddings reaches 0.910: clearly above the logistic regressions (+0.0062 over L1), still 0.009 below LightGBM (-12.8 SE), and it adds nothing to the blend. The raw ids and the network's capacity had to be cut back before it beat a linear model. As the net improved it ranked rows more like the trees (rank correlation 0.825 to 0.910), so its value to the blend did not rise with its score.

Expert models, 2026-10-31 (TODO 7.14, 7.15, 7.16, 7.23): the tuned LightGBM restricted to one block of the 55 adopted columns (features.py BLOCKS; configs/experiments/expert_*.yaml), not retuned. About 40 seconds each.

| block | columns | run | CV AUC | rank correlation with LightGBM | blend gain, weight chosen without the scored fold |
|---|---|---|---|---|---|
| finance | 12 | 20261031T132600Z_d6773f67 | 0.86737 | 0.731 | +0.00000 (0.0 SE, 2 of 5 folds) |
| process and loan terms | 18 | 20261031T132717Z_ff0a4ae5 | 0.82946 | 0.697 | +0.00004 (0.9 SE, 4 of 5 folds) |
| geography and census | 8 | 20261031T132634Z_df45cba9 | 0.67314 | 0.403 | -0.00002 (-0.7 SE, 0 of 5 folds) |
| demographics | 17 | 20261031T132746Z_e63bd31a | 0.63546 | 0.364 | not tested: fairness diagnostic, goes to TODO 12.8 ([N12]) |

- As expected, restricting a model decorrelates it strongly, but no block passes: decorrelated models this weak do not lift the blend. Hill climbing with the three blocks among the candidates ignores them (LightGBM 0.40, CatBoost 0.40, ExtraTrees 0.20).
- Alone, finance and process score far higher than geography and demographics. The block scores do not say what each block adds once the others are present: that is the ablation of TODO 13.3.
- Blocks are chosen by meaning, not by these scores. The process block contains the two confirmed process artefacts.

Stacking, 2026-10-31 (TODO 11.5, 11.9, "src/stack.py", "configs/blends/stack.yaml"): a logistic regression on 9 models (all but the demographics expert), fitted on the other folds' rows and scored on each fold.

| input of the meta-model | AUC | gain vs LightGBM + CatBoost 50/50 | gain / corrected SE | folds up |
|---|---|---|---|---|
| log-odds | 0.92038 | +0.00025 | 0.9 | 3 of 5 |
| ranks | 0.91893 | -0.00121 | -6.7 | 0 of 5 |
| probabilities | 0.91880 | -0.00133 | -1.3 | 0 of 5 |

- Does not pass. The 50/50 blend stays: simple averaging, hill climbing and stacking reach the same level (0.9201 to 0.9204), and only the simplest one is confirmed on fresh splits.
- XGBoost and the L1 logistic regression get negative coefficients: with correlated inputs a free meta-model subtracts models to fit noise, the variance risk of note [N14].
- Log-odds is the right input for the stack: ranks lose by -6.7 SE (the reason was not tested). This does not contradict TODO 11.3, where rank and log-odds tied for an equal-weight average of 2 models.

Blending with the tuned XGBoost, 2026-10-31 ("configs/blends/final_selection_v3.yaml", "hill_climb_v2.yaml", "stack_v2.yaml"; rank correlation with LightGBM 0.971, up from 0.961):

| method | AUC | gain vs LightGBM + CatBoost 50/50 | gain / corrected SE | folds up | verdict |
|---|---|---|---|---|---|
| hill climbing, nested | 0.92050 | +0.00036 | 4.0 | 5 of 5 | passes |
| stacking on log-odds, 9 models | 0.92044 | +0.00031 | 1.2 | 4 of 5 | does not pass |
| forward selection, equal weights, gated by the rule | 0.92014 | 0 | | | keeps the 50/50 blend |

- Hill climbing finds LightGBM 2/6, CatBoost 2/6, ExtraTrees 1/6, XGBoost 1/6 in 4 of the 5 nested climbs (run 20261031T220901Z_1ff742f2, 0.92054 on all folds, optimistic). Before XGBoost was tuned it gave +0.00026 (1.4 SE): the tuned model is what lifts it over the rule. On top of the 50/50 at equal weights, ExtraTrees adds +0.00025, XGBoost +0.00009, both together +0.00035: the two gains roughly add up.
- Candidate, not adopted: this is the fourth pass of blend selection on the frozen folds, and the 50/50 blend lost ~0.00025 of its gain on fresh splits. If this one lost as much, ~0.0001 would be left. It needs its own fresh-split check, against the 50/50 blend on the same splits.

Blending with the CatBoost search winner, 2026-11-01 ("configs/blends/final_selection_v4.yaml", "hill_climb_v3.yaml", "stack_v3.yaml"): rejected as a model (-0.00022, -0.6 SE vs the untuned one), and worse in every blend. Rank correlation with LightGBM 0.954 against 0.947: closer to the other models, so less to add.

| blend | AUC | vs candidate 2/6, 2/6, 1/6, 1/6 with the untuned CatBoost (0.92054) |
|---|---|---|
| same weights, winner instead of untuned CatBoost | 0.92025 | -0.00029, -2.1 SE, 0 of 5 folds |
| LightGBM + winner 50/50 | 0.91982 | -0.00072, -4.0 SE, 0 of 5 folds |

The final_selection_v4 result file (LightGBM alone) and hill_climb_v3 are records of this test, not a choice. The candidate stays run 20261031T220901Z_1ff742f2, with the 50/50 blend adopted until it passes on fresh splits.

Fresh-split check of the 4-model blend, finished 2026-11-01 (TODO 7.25, "make fresh-blends SPEC=configs/blends/fresh_candidate.yaml", "reports/tuning/fresh_candidate.log", per-fold results in "reports/tuning/fresh_fresh_candidate.csv", member predictions in oof/fresh/). Candidate LightGBM 2/6, CatBoost county 2/6, ExtraTrees 1/6, XGBoost 1/6 against the 50/50 blend, same 15 folds:

| split seed | 50/50 blend | 4-model blend | gain | folds up |
|---|---|---|---|---|
| 1 | 0.91966 | 0.92038 | +0.00072 | 5 of 5 |
| 2 | 0.91976 | 0.92043 | +0.00066 | 5 of 5 |
| 3 | 0.91971 | 0.92043 | +0.00072 | 5 of 5 |

- All 15 paired folds: +0.00070, corrected SE 0.00009, 7.6 SE, 15 of 15 folds up: CONFIRMED. Adopted: the 4-model blend (run 20261031T220901Z_1ff742f2) replaces the 50/50 blend as the best model.
- The fresh gain (+0.00070) is larger than on the frozen folds (+0.00040 for the logged blend, +0.00036 nested): unlike the 50/50 blend, this one was not flattered by being chosen on the frozen folds.
- It is the tuned ExtraTrees and XGBoost that make the difference: the same LightGBM and the same untuned CatBoost are in both blends.

Public LB gap, 2026-10-31 (written then, lost from this file afterwards, restored 2026-11-09). The leader was given as 0.931 against our 0.92118; corrected 2026-11-09 to 0.92301. Quick checks only, no model refit:

| check | result | reading |
|---|---|---|
| public LB noise, 1,000 draws of 13,500 OOF rows of the 50/50 blend | SD of one score 0.0038; SD of a paired gap between two models 0.0014 (blend vs ExtraTrees) | the scale to read any LB gap against |
| adversarial validation, LightGBM, 5 folds, 200 trees | AUC 0.5015 (55 adopted features), 0.5022 (+ raw county / MSA ids) | test looks like train: no shift to exploit (TODO 2.7) |
| exact twins, all 56 columns | 0.01% of test rows | the label agrees, but too few rows to matter |
| near twins, 6 other keys | 0.28% to 2.76% of test rows | best gain on top of the blend +0.00002 (0.6 SE) |

- Twin value is measured on labelled rows: each row's twin labels come from the other folds only, as a train twin would for a test row, and are added to the blend's log-odds at 5 strengths, keeping the best (so the figure is optimistic). Raw agreement looks strong on near twins (P(accepted) 0.82 if the twin was accepted, 0.28 if denied, on tract + amount + value), yet the blend gains nothing: it already knows what the twin knows. Rows sharing a key are not proven to be the same application.
- The leader's public score is 0.92301, not 0.931 (nor the 0.95 quoted on 2026-10-19). Against our best public score (0.92158, submission 6) the gap is 0.0014: 0.4 SD of one public score and 1.0 SD of a paired gap. It is within the public-LB noise and says nothing about the private 70%. The checks stand (no shift, no duplicates to exploit), but there was no 0.011 gap to explain.
