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
| Blend LightGBM 2/6, CatBoost county 2/6, ExtraTrees 1/6, XGBoost 1/6, log-odds (adopted 2026-11-01) | configs/blends/lgbm_cat_et_xgb.yaml | 20261031T220901Z_1ff742f2 | 0.92054 | +0.00149 vs LightGBM; fresh splits +0.00070 over the 50/50 blend (7.6 SE, 15 of 15) | submission 6, LB 0.92158 |
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

Confidence interval on the final AUC, 2026-11-09 (TODO 13.4, "make ci", "src/bootstrap_ci.py", "reports/tuning/ci_final.log"). Stratified bootstrap of the 105,000 out-of-fold rows, 2,000 resamples, seed 0:

| model | run | pooled OOF AUC | 95% CI | bootstrap SD | mean of fold AUCs |
|---|---|---|---|---|---|
| final: 4-model blend | 20261031T220901Z_1ff742f2 | 0.92043 | [0.91764, 0.92318] | 0.00141 | 0.92054 |
| safety: 50/50 blend | 20261014T115111Z_918eb62b | 0.92000 | [0.91723, 0.92275] | 0.00141 | 0.92014 |
| tuned LightGBM alone | 20261003T084548Z_406e4210 | 0.91884 | [0.91588, 0.92153] | 0.00144 | 0.91905 |

- The interval measures which rows were scored, not retraining: the fresh-split checks measure that part (spread of the 4-model blend over 3 splits: 0.92038 to 0.92043).
- Pooled AUC (all rows ranked at once) is ~0.0001 to 0.0002 below the mean of the fold AUCs: the folds' scores are not on exactly the same scale. Quote one and say which.
- The three intervals overlap, yet the differences between the models are real: two models scored on the same rows share most of the row noise, so the paired gain has a much smaller SD (paired bootstrap in make compare: 4-model blend vs 50/50 [+0.00021, +0.00066]). Overlapping single-model intervals are not a test of the difference.

Leakage ablation, 2026-11-09 (TODO 12.9, configs from the fairness agent: configs/experiments/fair_*.yaml, tuned LightGBM settings, not retuned, same folds). fair_full is identical to the tuned LightGBM, so its run is 20261003T084548Z_406e4210.

| feature list | columns | run | CV AUC | vs full | gain / corrected SE |
|---|---|---|---|---|---|
| full | 55 | 20261003T084548Z_406e4210 | 0.91905 | | |
| clean: without the 2 confirmed leaks (preapproval, initially_payable_to_institution) | 53 | 20261109T170831Z_116f6267 | 0.89305 | -0.02600, bootstrap CI [-0.02793, -0.02421] | -13.6 |
| strict: clean, also without aus-1, LTV, property value, DTI, the census fields, state, location flags | 38 | 20261109T170956Z_cd2c2a46 | 0.82343 | -0.09562, bootstrap CI [-0.09892, -0.09224] | -36.4 |

- Clean vs strict: -0.06962 (-62.8 SE). Reading for the fairness and report agents, not a modelling decision: the strict list removes the main credit variables (LTV, DTI, property value) together with the suspect signals, so its loss is not a measure of leakage alone.
- The two confirmed leaks cover 6% of train rows (4,713 + 1,566) but are worth 0.026 AUC: their outcome is fixed in train (0 denials with preapproval = 1, 0 approvals with initially_payable = 3), so they are the easiest rows to rank.

Fair model cost in AUC, 2026-11-09 (configs from the fairness agent: fair_unaware.yaml, fair_reweigh.yaml; tuned LightGBM settings, not retuned, same folds). Both start from the clean list (confirmed leaks out). Reweighing (Kamiran and Calders 2012, "reweigh" in base.yaml, src/cv.py reweigh_weights): training weights w(g, y) = P(g) P(y) / P(g, y) by derived_race, computed on each fold's fit part (0.28 to 1.81 over the 5 folds, mean 1); derived_race sets the weights and is not a feature; scoring unweighted.

| model | columns | run | CV AUC | vs row above | gain / corrected SE | paired bootstrap 95% CI |
|---|---|---|---|---|---|---|
| clean | 53 | 20261109T170831Z_116f6267 | 0.89305 | | | |
| unaware: clean without the 17 demographic fields | 36 | 20261109T183150Z_8e9bf39c | 0.89060 | -0.00245 | -3.1 | [-0.00358, -0.00152] |
| unaware + reweighing by derived_race | 36 | 20261109T183339Z_9ddab9a2 | 0.88985 | -0.00075 | -2.4 | [-0.00136, +0.00002] |

- AUC is the cost side only. Whether the fair model is fairer is read from its OOF predictions by group (fairness agent, oof/ of both runs).
- Dropping the demographic fields costs 0.0025 (5 of 5 folds down): small, but they carry some signal beyond the other 36 columns. Part of it may come back through proxies (12.3).
- Reweighing costs a further 0.0008 (5 of 5 folds down, CI just touching 0): close to free in AUC. The tree count is picked on the weighted inner AUC, so the reweighed model chooses its own (113 to 158 trees, vs 100 to 152 unweighted).

For the report: T9 ablations, H1-H3 and the fair models, 2026-11-10. All on the tuned LightGBM settings (not retuned), same folds, compared with make compare; base = full list, run 20261003T084548Z_406e4210, CV 0.91905.

T9 rows (prediction = range registered on 2026-10-18):

| # | ablation | predicted | observed | gain / corrected SE | run |
|---|---|---|---|---|---|
| A1 | no demographics (17 fields) | -0.002 [-0.004, 0] | -0.00270, 0 of 5 folds up: inside | -5.2 | 20a010b4 |
| A2 | no process fields (preapproval, submission_of_application, initially_payable_to_institution, aus-1) | -0.015 [-0.025, -0.008] | -0.03384, 0 of 5 up: outside, larger loss | -15.1 | bc281e8f |
| A3 | no census (6 tract columns, MSA median income) | -0.004 [-0.008, -0.002] | -0.00132, 1 of 5 up: outside, smaller loss | -2.0 | 29f2ea4d |
| A4c | no target encoding, L1 logistic regression (vs run 6c8c7142, 0.90379) | -0.002 [-0.005, 0] | -0.00008, CI [-0.00019, +0.00002], 2 of 5 folds up: inside, near 0 | -0.9 | 1dc8e77b |
| A5 | raw typed columns only, county and MSA ids kept | -0.014 [-0.020, -0.010] | -0.01207, 0 of 5 up: inside | -8.8 | fcc2e23f |
| H1-H3 | one block: term_off_grid, ltv_blank_value_present, computed_ltv_above_100 (58 columns) | about +0.0001 to +0.0009 (sum of the T11 ranges) | -0.00016, 2 of 5 up: rejected, not in any blend | -0.2 | ebb1bd30 |

- A2 misses its range because 2 of its 4 fields are the confirmed leaks (A9 alone: -0.026).
- A3: the census columns matter less than predicted; the finance and process columns already carry most of what the tract describes.
- A4c: without the county rate the L1 model has no county information at all and loses nothing measurable (C = 0.3 in every fold, as with the rate): the other columns already carry what the county rate added for a linear model.
- H1-H3: the patterns seen in the worst errors are real in the residuals but give the model nothing new: paired bootstrap CI [-0.00060, +0.00061].

Section 9, fair models: cost on CV and on the public leaderboard (submissions 7-10, not among the 2 final submissions):

| feature list | model | run | CV AUC | public LB |
|---|---|---|---|---|
| Kaggle (leaks kept) | all features (tuned LightGBM, submission 4) | 406e4210 | 0.91905 | 0.9210 |
| Kaggle | race-blind | 20a010b4 | 0.91636 (-0.0027, -5.2 SE) | 0.91869 |
| Kaggle | race-blind + reweighing by derived_race | 6990a174 | 0.91560 (-0.0008 more, -0.9 SE) | 0.91855 |
| clean (leaks out) | all features | 116f6267 | 0.89305 | |
| clean | race-blind | 8e9bf39c | 0.89060 (-0.0025, -3.1 SE) | 0.89408 |
| clean | race-blind + reweighing | 9ddab9a2 | 0.88985 (-0.0008 more, -2.4 SE) | 0.89342 |

- Race-blind costs about 0.0025 on both lists and transfers to unseen rows (LB -0.0023 vs submission 4). Reweighing adds -0.0008 on CV and is within the leaderboard noise. The expensive step is removing the leaks (-0.026), not fairness.

Section 9.3, proxy check (12.3, src/proxy_check.py, reports/tuning/proxy_check.csv): LightGBM predicts one race group vs all others from the 38 race-blind columns of A1; groups with at least 5,000 rows; mean of 5 fold AUCs, fold spread under +-0.01.

| group | rows | from all 38 columns | from geography only (6 tract columns, MSA income, state) |
|---|---|---|---|
| White | 66,107 | 0.740 | 0.695 |
| Race Not Available | 17,872 | 0.672 | 0.587 |
| Black or African American | 8,991 | 0.840 | 0.796 |
| Asian | 8,009 | 0.847 | 0.800 |

- A race-blind model still sees race: the columns it keeps identify Black and Asian applicants at AUC 0.84 to 0.85, mostly through geography (0.80), which includes tract_minority_population_percent. The reading is the fairness agent's.

Reproducibility re-runs of the 4 final members, 2026-11-10 (clean commit d782fce, same feature list and folds; config hashes differ only because base.yaml gained keys since):

| member | logged run | re-run | fold AUCs | predictions |
|---|---|---|---|---|
| LightGBM tuned | 406e4210 | cc720ddf | identical (0.91905) | OOF and test identical |
| ExtraTrees tuned | 7bc88a0b | f308315c | identical (0.91484) | identical to 1e-15 |
| XGBoost tuned | 05b22e57 | 104e3dd7 | identical (0.91848) | identical |
| CatBoost county | d96c8be4 | a8b526e0 | 0.91738 vs 0.91859, -0.0012, 5 of 5 folds lower | Spearman 0.979 with the logged OOF |
| CatBoost county, max_ctr_complexity 4 (2026-11-11) | d96c8be4 | 0810e0b4 | identical (0.91859) | OOF and test identical |

- CatBoost cause, for section 10.2 (found 2026-11-11). On 2026-10-31 five CatBoost settings were written into base.yaml, read with get_all_params on a short fit. CatBoost resolves the default of max_ctr_complexity from the number of trees requested: 1 at 30 trees, 4 at 999 trees or more (checked 2026-11-11 on this data; the other four settings resolve to the written values at 5,000 trees too). Our runs request 5,000 trees. So base.yaml has set max_ctr_complexity 1 (categories never combined) since 2026-10-31, where every earlier run had used 4.
- Test: catboost_county_ctr4.yaml = catboost_county.yaml with max_ctr_complexity 4, nothing else changed; run 20261111T100056Z_0810e0b4, clean commit, 6,938 s. Its 5 fold AUCs equal those of d96c8be4 to 8 decimals, and its 105,000 OOF and 45,000 test predictions are identical. The re-run a8b526e0 (max_ctr_complexity 1) is 0.91738, -0.0012.
- CatBoost runs and the value they used: 6deb776a and d96c8be4 (2026-10-14, nothing written) 4; 9ef0b007 (Optuna best, 2026-11-01) 2; a8b526e0 1; 0810e0b4 4. A4b compares 6deb776a with d96c8be4: both 4.
- Ran with max_ctr_complexity 1: the fresh-split check of 2026-11-01 (configs/blends/fresh_candidate.yaml, CatBoost member from catboost_county.yaml, in both the reference and the candidate blend). The CatBoost Optuna search of 2026-10-31 searched max_ctr_complexity in {1, 2}; its trial 0 (the reference) used 1, and its best settings (2) were compared in the full CV with d96c8be4, which used 4.
- base.yaml keeps the value 1, so every run since 2026-10-31 still reproduces; its comment now states the default. The blend config configs/blends/lgbm_cat_et_xgb.yaml now points at 0810e0b4 for the CatBoost member.
- Final blend rebuilt from the 4 rebuilt-or-logged members (make blend BLEND=configs/blends/lgbm_cat_et_xgb.yaml, run 20261111T104928Z_653a027d, 2026-11-11): fold AUCs identical to 1ff742f2 (mean 0.920542), OOF and test predictions identical; the submission 6 file differs from the rebuilt test predictions by at most 1e-16 (CSV rounding).
- For the report (10.2): all 4 members of the final blend rebuild bit for bit from the code and the lockfile. CatBoost through configs/experiments/catboost_county_ctr4.yaml (the config column of the 10.2 table should name it, not catboost_county.yaml).
