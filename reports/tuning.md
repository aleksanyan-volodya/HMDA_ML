# Tuning

## LightGBM, step 1: which settings matter?

One setting changed at a time around the current values; every run compared with the reference run 20261002T141818Z_f7f2dc68 (CV AUC 0.91695) on the same folds. Sweep file: "configs/sweeps/lgbm_one_at_a_time.yaml". Run on 2026-10-02.

| setting | value | CV AUC | gain | gain / corrected SE | folds up |
|---|---|---|---|---|---|
| unchanged (repeat) | | 0.91695 | 0.00000 | | 0 of 5 |
| num_leaves (reference 63) | 15 | 0.91525 | -0.00169 | -4.0 | 0 |
| | 31 | 0.91631 | -0.00064 | -1.6 | 1 |
| | 127 | 0.91660 | -0.00035 | -1.8 | 1 |
| min_data_in_leaf (reference 100) | 20 | 0.91736 | +0.00041 | +1.0 | 4 |
| | 500 | 0.91480 | -0.00215 | -5.6 | 0 |
| lambda_l2 (reference 1) | 0 | 0.91689 | -0.00006 | -0.2 | 2 |
| | 10 | 0.91681 | -0.00014 | -0.4 | 3 |
| feature_fraction (reference 0.8) | 0.5 | 0.91789 | +0.00094 | +1.8 | 5 |
| | 1.0 | 0.91650 | -0.00045 | -0.9 | 2 |
| cat_smooth (reference 10) | 100 | 0.91697 | +0.00002 | 0.0 | 3 |

Conclusions:

- Runs are deterministic: the repeat of the reference gives exactly the same score on every fold. Two runs can be compared without any run-to-run noise.
- Settings that matter: num_leaves (15 is clearly worse), min_data_in_leaf (500 is clearly worse, 20 slightly better), feature_fraction (0.5 better on all 5 folds, just under the 2 SE rule).
- Settings that do not matter: lambda_l2 and cat_smooth. They are left at their current values.
- No single change passes the adoption rule (gain >= 2 corrected SE), so nothing is adopted from this step alone. The settings that matter go to the joint search (step 2).

## LightGBM, step 1 (second screening): the other settings

Same method and same reference. Sweep file: "configs/sweeps/lgbm_one_at_a_time_2.yaml". Run on 2026-10-02.

| setting | value | CV AUC | gain | gain / corrected SE | folds up |
|---|---|---|---|---|---|
| unchanged (repeat) | | 0.91695 | 0.00000 | | 0 of 5 |
| max_cat_to_onehot (reference 4) | 64 | 0.91771 | +0.00076 | +4.5 | 5 |
| learning_rate (reference 0.05) | 0.02 | 0.91807 | +0.00112 | +2.3 | 5 |
| | 0.1 | 0.91577 | -0.00118 | -2.1 | 0 |
| path_smooth (reference 0) | 10 | 0.91756 | +0.00061 | +2.1 | 5 |
| min_data_per_group (reference 100) | 20 | 0.91703 | +0.00008 | +0.2 | 3 |
| | 500 | 0.91762 | +0.00067 | +1.2 | 4 |
| bagging_fraction (reference 0.8) | 0.5 | 0.91507 | -0.00188 | -3.0 | 0 |
| | 1.0 | 0.91721 | +0.00027 | +0.7 | 4 |
| max_depth (reference: no limit) | 6 | 0.91554 | -0.00141 | -5.3 | 0 |
| | 10 | 0.91650 | -0.00045 | -1.0 | 2 |
| max_bin (reference 255) | 63 | 0.91681 | -0.00014 | -0.3 | 1 |
| | 511 | 0.91654 | -0.00041 | -1.6 | 1 |
| lambda_l1 (reference 0) | 1 | 0.91671 | -0.00023 | -0.7 | 1 |
| extra_trees (reference off) | on | 0.91571 | -0.00124 | -1.6 | 1 |
| cat_l2 (reference 10) | 1 | 0.91670 | -0.00025 | -0.4 | 1 |
| | 100 | 0.91681 | -0.00014 | -0.3 | 2 |

Conclusions:

- Three settings pass the adoption rule alone, each with 5 folds up: max_cat_to_onehot = 64 (categories split one level at a time instead of grouped), learning_rate = 0.02, path_smooth = 10.
- Clearly harmful: max_depth = 6, bagging_fraction = 0.5, learning_rate = 0.1.
- No effect: max_bin, lambda_l1, cat_l2, min_data_per_group = 20.
- Caveat: 27 variants were tested over the two sweeps. At a 2 SE threshold about one false positive is expected by chance, and path_smooth (2.1 SE) is the weakest of the three. A single-setting result is therefore not adopted on its own: the settings that matter go to the joint search, and its winner is re-checked on fresh fold splits.

## LightGBM, step 2: joint search with Optuna

100 trials, TPE sampler (seed 0), learning rate fixed at 0.05, trial 0 = the untuned settings. Search file: "configs/sweeps/lgbm_optuna.yaml"; every trial in "reports/tuning/lgbm_optuna.csv". Run on 2026-10-02/03.

| | CV AUC | gain vs trial 0 | gain / corrected SE | folds up |
|---|---|---|---|---|
| trial 0 (untuned) | 0.91695 | | | |
| best, trial 98 | 0.91905 | +0.00210 | 3.7 | 5 |
| median of the 100 trials | 0.91848 | | | |

The best settings form a consistent region: the 10 best trials are within 0.0003 of each other and agree.

| setting | untuned | 10 best trials | trial 98 |
|---|---|---|---|
| max_cat_to_onehot | 4 | 64 (all 10) | 64 |
| num_leaves | 63 | 98 to 127 | 113 |
| min_data_in_leaf | 100 | 5 to 9 | 5 |
| path_smooth | 0 | 10 to 18 | 14.6 |
| feature_fraction | 0.8 | 0.40 to 0.59 | 0.57 |
| bagging_fraction | 0.8 | 0.94 to 0.99 | 0.95 |

- Small leaves (5 rows) only work together with path_smooth, which pulls a small leaf towards its parent: a joint effect the one-at-a-time screening could not find.
- num_leaves, min_data_in_leaf and bagging_fraction end near the edge of their range. The flat top suggests little is left outside it; not searched further.
- Not adopted yet. The winner is selected among 100 trials on the frozen folds, so it must (1) pass the rule as a logged run, (2) be refitted at learning rate 0.02 ("lgbm_optuna_best_lr002.yaml"), (3) stay ahead of the untuned model on fresh fold splits ("make fresh").

Adoption checks (2026-10-03):

| step | comparison | gain | gain / corrected SE | folds up | verdict |
|---|---|---|---|---|---|
| 1 | winner (run 20261003T084548Z_406e4210) vs untuned | +0.00210 | 3.7 | 5 | adopted on the frozen folds; reproduces trial 98 exactly |
| 2 | winner at learning rate 0.02 (run 20261003T084930Z_49275099) vs at 0.05 | +0.00035 | 0.8 | 3 | rejected: once the other settings are tuned, the lower rate adds nothing reliable |
| 3 | winner vs untuned on 3 fresh splits (seeds 1, 2, 3; 15 paired folds) | +0.00194 | 4.2 | 15 of 15 | confirmed |

Result: the tuned LightGBM is "configs/experiments/lgbm_optuna_best.yaml" (learning rate 0.05), run 20261003T084548Z_406e4210, CV AUC 0.91905.

- +0.00210 over the untuned model on the frozen folds, +0.00194 on fresh splits never used for any choice: selecting the best of 100 trials flattered it by only ~0.0002.
- Per-fold fresh results: "reports/tuning/fresh_lgbm_optuna_best_vs_lgbm_best_features.csv".

## ExtraTrees: joint search with Optuna (TODO 7.19)

Search file "configs/sweeps/extratrees_optuna.yaml", on the ranked-numerics ExtraTrees (run 20261014T135428Z_542eac1f, CV 0.90185). TPE sampler (seed 0), trial 0 = the untuned settings, 300 trees fixed (more trees only reduce averaging noise). Night of 2026-10-18: stopped by the 8 h cap after 47 of 60 trials; every trial in "reports/tuning/extratrees_optuna.csv".

| setting | untuned | best (trial 23) | range searched |
|---|---|---|---|
| max_features | sqrt (17 of 307 columns) | 0.35 (107 columns) | sqrt, 0.1, 0.2, 0.35 |
| min_samples_leaf | 1 | 5 | 1 to 30 |
| bootstrap | false | false | false, true |
| max_depth | none | 20 | none, 30, 20 |

| | CV AUC | gain | gain / corrected SE | folds up |
|---|---|---|---|---|
| untuned (trial 0) | 0.90185 | | | |
| best, logged refit (run 20261019T060754Z_7bc88a0b) | 0.91484 | +0.01299 | 41.0 | 5 of 5 |
| median of the 47 trials | 0.91286 | | | |

- The refit reproduces trial 23 exactly. Adopted as the ExtraTrees candidate: the gain is far beyond the selection effect of ~25 distinct trials.
- The number of columns per split matters most: median AUC 0.907 at 0.1, 0.912 at 0.2, 0.914 at 0.35, still rising at the largest value searched. max_depth 20, the shallowest value searched, beats 30 and no limit. Both are at the edge of their range, so a wider search would probably gain a little more. Bootstrap never reached the top trials.
- About half the trials repeated settings already tried (small, mostly categorical space): ~25 distinct settings were scored.
- Tuning made ExtraTrees stronger but more like the boosted models: rank correlation with LightGBM 0.785 -> 0.910.

## XGBoost: joint search with Optuna

Search file "configs/sweeps/xgb_optuna.yaml", on the XGBoost with starting values (run 20261014T125236Z_f0fe3121, CV 0.91649), ranges taken from what mattered for LightGBM; each range's reason is in the file. TPE sampler (seed 0), trial 0 = the untuned settings, eta fixed at 0.05. Run on 2026-10-31: stopped by the 1.9 h cap after 42 of 60 trials; every trial in "reports/tuning/xgb_optuna.csv".

| setting | untuned | best (trial 21) | 10 best trials | range searched |
|---|---|---|---|---|
| max_depth | 6 | 11 | 10 to 12 | 6 to 12 |
| min_child_weight | 10 | 1.64 | 1.6 to 7.5 | 0.5 to 30 |
| lambda | 1 | 4.76 | 4.8 to 12.3 | 0.5 to 20 |
| colsample_bytree | 0.8 | 0.71 | 0.49 to 0.71 | 0.3 to 1.0 |
| subsample | 0.8 | 0.83 | 0.83 to 0.98 | 0.6 to 1.0 |

| | CV AUC | gain | gain / corrected SE | folds up |
|---|---|---|---|---|
| untuned (trial 0) | 0.91649 | | | |
| best, logged refit (run 20261031T215118Z_05b22e57) | 0.91848 | +0.00199 | 7.1 | 5 of 5 |
| median of the 42 trials | 0.91811 | | | |

- Adopted as the XGBoost: the refit reproduces trial 21, and the gain is close to LightGBM's +0.0021. Tree counts 141 to 173.
- Depth was the main limit, as the LightGBM screening suggested: the two trials at depth 6 peak at 0.91742. The best depth is inside the range and the 10 best trials are within 0.0002, so a wider search is unlikely to add much.
- Level with the untuned CatBoost (-0.00011, -0.3 SE) and with the tuned LightGBM (-0.00057, -1.0 SE): neither gap is significant.

## CatBoost: joint search with Optuna, cheap trials

Search file "configs/sweeps/catboost_optuna.yaml", on the CatBoost with raw county / MSA (run 20261014T085450Z_d96c8be4, CV 0.91859). A full CV was ~1h45, so each trial was 2 single fits: scored on folds 0 and 3, trained on 3 other folds (63,000 rows), tree count taken on the last one, no refit. TPE sampler (seed 0), 80 trials, 3 hours, 2026-10-31/11-01; every trial in "reports/tuning/catboost_optuna.csv".

| setting | untuned | best (trial 61) | 10 best trials |
|---|---|---|---|
| depth | 6 | 9 | 7 to 9 |
| one_hot_max_size | 2 | 64 | 64 |
| max_ctr_complexity | 1 | 2 | 2 |
| l2_leaf_reg | 3 | 24 | 12.7 to 29 |
| rsm | 1 | 0.53 | 0.49 to 0.62 |
| subsample | 0.8 | 0.71 | 0.67 to 0.76 |
| random_strength | 1 | 0.35 | 0.06 to 0.82 |

| | untuned | winner | gain |
|---|---|---|---|
| cheap trial, mean of folds 0 and 3 | 0.91070 | 0.91318 | +0.00248 |
| full CV, folds 0 and 3 | 0.91372 | 0.91400 | +0.00028 |
| full CV, 5 folds (run 20261101T090757Z_9ef0b007) | 0.91859 | 0.91837 | -0.00022, -0.6 SE, 3 of 5 folds up |

Rejected: the untuned CatBoost stays. Blends are worse with the winner too (see "reports/models.md").

- The search's assumption failed: cheap trials do not rank settings as a full CV does. It is not the choice of folds: on the same folds 0 and 3, a gain of +0.0025 in the trials is +0.0003 in the full CV. From the trial to the full CV (84,000 rows, refit) the untuned model gains +0.00302, the winner +0.00082. On fewer rows the search favoured settings that need fewer rows, which is not what the final model needs. Why the untuned model gains more was not tested.
- The winner runs ~6 times faster for the same AUC (20 minutes against ~2 hours, trees 735 to 1,043), but speed is not in the rule.
