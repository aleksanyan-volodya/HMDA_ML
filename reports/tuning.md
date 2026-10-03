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
