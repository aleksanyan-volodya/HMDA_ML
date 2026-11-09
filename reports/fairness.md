# Fairness

## What the target is

The target is a lender's recorded decision, not creditworthiness and not default. "accepted" says what a lender did with an application. It does not say whether the applicant could have repaid, and the data holds no repayment history. A model that scores well here has learned to predict past lending decisions. If some of those decisions treated a group worse than its risk justified, a faithful model carries that on, and a high AUC is no defence: AUC measures agreement with the decision, not its merit.

Three consequences for everything below:

- An "error" of the model means a disagreement with the lender, not a wrong judgement of the applicant.
- A gap between groups is an association. The data has no credit score, no interest rate and no applicant history, so it can't separate a difference in risk from a difference in treatment. We do not conclude discrimination where a gap appears, nor its absence where none does.
- The rows are a selection. We read "accepted" as originated (action_taken = 1) against denied (action_taken = 3): this reading fits the two leak columns, but action_taken itself is not in our files. Applications that were withdrawn or closed as incomplete would then be outside the data, and we can't see who they were.

## Feature lists and the cost of removing the leaks

Three lists are frozen, with the reasons in "reports/leakage.md". All use the tuned LightGBM parameters, unchanged, so that only the columns differ.

| list | config | columns | what it removes | CV AUC | gap to the list above | corrected SE |
|---|---|---|---|---|---|---|
| full | configs/experiments/fair_full.yaml | 55 | nothing | 0.91905 (same config as run 20261003T084548Z_406e4210) | | |
| clean | configs/experiments/fair_clean.yaml | 53 | the 2 confirmed leaks | not run yet | | |
| strict | configs/experiments/fair_strict.yaml | 38 | also aus-1, LTV, property value, DTI, census, state | not run yet | | |

The gap between full and clean is how much of the competition score rests on columns that record the decision. The gap between clean and strict is how much the open verdicts could matter. Strict removes real information along with the suspect signals, so its gap is an upper bound on that, not an estimate of it.

## Group diagnostics of the best model

Model: LightGBM + CatBoost county, 50/50 log-odds (run 20261014T115111Z_918eb62b, pooled out-of-fold AUC 0.920). All numbers are out-of-fold, on the 105,000 training rows, and come from "notebooks/04_fairness.ipynb", which trains nothing.

This is the competition model: it sees the demographic fields and both leaks. The tables describe that model, not a fair one. They are the baseline for the fair model.

Conventions:

- Groups are the official summaries "derived_race", "derived_ethnicity", "derived_sex", plus "applicant_age". "Not Available" and "Joint" are kept: they are large, but they are not demographic groups in the plain sense. "Not Available" mixes applicants who did not answer with applicants who are not persons; a blank age is mostly the latter.
- Groups under 500 rows are shown with their size and not read.
- Intervals are 95% bootstrap intervals over rows inside each group (1000 resamples). They ignore the uncertainty of training the model.
- A gap is measured against the largest group with one declared value: White, Not Hispanic or Latino, Male, age 25-34. It is a base for subtraction, not a norm.
- AUC has no threshold, so any cut-off is a convention. "Same rate" is 0.779, the cut-off that accepts 85% of rows, as lenders did. 0.50 is the plain reading of a probability. Each is the same for all groups.
- TPR = share of accepted rows the model accepts. FPR = share of denied rows the model accepts.

### Calibration and ranking

"pred - obs" is the mean predicted probability minus the share accepted. "AUC without leak rows" leaves out the 6,279 rows with preapproval = 1 or initially_payable_to_institution = 3; the model was still trained with those columns.

| field | group | n | denied | observed | pred - obs | AUC | AUC without leak rows |
|---|---|---|---|---|---|---|---|
| race | White | 66107 | 8721 | 0.868 [0.865, 0.871] | +0.001 [-0.001, 0.002] | 0.921 [0.917, 0.924] | 0.907 [0.903, 0.911] |
| race | Race Not Available | 17872 | 3082 | 0.828 [0.822, 0.833] | +0.002 [-0.001, 0.006] | 0.922 [0.915, 0.928] | 0.908 [0.900, 0.915] |
| race | Black or African American | 8991 | 2349 | 0.739 [0.730, 0.748] | +0.004 [-0.001, 0.010] | 0.909 [0.901, 0.917] | 0.896 [0.886, 0.904] |
| race | Asian | 8009 | 888 | 0.889 [0.882, 0.896] | +0.004 [-0.001, 0.009] | 0.880 [0.865, 0.894] | 0.853 [0.834, 0.870] |
| race | Joint | 2716 | 347 | 0.872 [0.859, 0.884] | -0.000 [-0.008, 0.008] | 0.909 [0.887, 0.932] | 0.893 [0.868, 0.919] |
| race | American Indian or Alaska Native | 810 | 221 | 0.727 [0.696, 0.758] | -0.009 [-0.027, 0.009] | 0.939 [0.917, 0.960] | 0.933 [0.908, 0.956] |
| race | 2 or more minority races | 278 | 87 | not read |  |  |  |
| race | Native Hawaiian or Other Pacific Islander | 196 | 51 | not read |  |  |  |
| race | Free Form Text Only | 21 | 10 | not read |  |  |  |
| ethnicity | Not Hispanic or Latino | 70469 | 9493 | 0.865 [0.863, 0.868] | +0.001 [-0.001, 0.003] | 0.920 [0.916, 0.923] | 0.906 [0.901, 0.910] |
| ethnicity | Ethnicity Not Available | 16232 | 2714 | 0.833 [0.827, 0.838] | +0.001 [-0.002, 0.005] | 0.923 [0.917, 0.930] | 0.909 [0.901, 0.917] |
| ethnicity | Hispanic or Latino | 15102 | 3120 | 0.793 [0.787, 0.800] | +0.003 [-0.001, 0.007] | 0.909 [0.902, 0.916] | 0.897 [0.888, 0.905] |
| ethnicity | Joint | 3170 | 417 | 0.868 [0.856, 0.880] | +0.007 [-0.001, 0.015] | 0.920 [0.903, 0.937] | 0.907 [0.888, 0.926] |
| ethnicity | Free Form Text Only | 27 | 12 | not read |  |  |  |
| sex | Joint | 38731 | 4518 | 0.883 [0.880, 0.887] | +0.001 [-0.001, 0.003] | 0.922 [0.917, 0.927] | 0.909 [0.903, 0.915] |
| sex | Male | 34507 | 5359 | 0.845 [0.841, 0.848] | +0.003 [0.000, 0.005] | 0.905 [0.900, 0.910] | 0.888 [0.882, 0.893] |
| sex | Female | 23462 | 4582 | 0.805 [0.799, 0.809] | +0.000 [-0.003, 0.003] | 0.933 [0.928, 0.937] | 0.922 [0.917, 0.928] |
| sex | Sex Not Available | 8300 | 1297 | 0.844 [0.836, 0.851] | +0.003 [-0.002, 0.008] | 0.912 [0.901, 0.922] | 0.893 [0.880, 0.905] |
| age | 25-34 | 30933 | 4162 | 0.865 [0.862, 0.869] | +0.000 [-0.002, 0.002] | 0.928 [0.923, 0.933] | 0.916 [0.910, 0.922] |
| age | 35-44 | 27055 | 4000 | 0.852 [0.848, 0.857] | +0.002 [-0.001, 0.005] | 0.915 [0.909, 0.921] | 0.899 [0.892, 0.906] |
| age | 45-54 | 17437 | 2897 | 0.834 [0.828, 0.839] | +0.005 [0.001, 0.009] | 0.912 [0.905, 0.919] | 0.896 [0.888, 0.904] |
| age | 55-64 | 11993 | 1903 | 0.841 [0.835, 0.848] | -0.001 [-0.005, 0.004] | 0.920 [0.912, 0.927] | 0.905 [0.895, 0.914] |
| age | <25 | 6763 | 1268 | 0.813 [0.804, 0.821] | +0.002 [-0.004, 0.007] | 0.936 [0.927, 0.944] | 0.927 [0.917, 0.937] |
| age | 65-74 | 5903 | 864 | 0.854 [0.845, 0.862] | +0.000 [-0.005, 0.006] | 0.914 [0.902, 0.925] | 0.900 [0.887, 0.913] |
| age | (blank) | 3088 | 387 | 0.875 [0.863, 0.887] | +0.001 [-0.008, 0.010] | 0.882 [0.859, 0.903] | 0.875 [0.850, 0.898] |
| age | >74 | 1828 | 275 | 0.850 [0.834, 0.865] | +0.001 [-0.010, 0.013] | 0.902 [0.877, 0.924] | 0.891 [0.862, 0.915] |

### Cut-off "same rate" (0.779)

| field | group | n | accepts | TPR | FPR | TPR gap | FPR gap |
|---|---|---|---|---|---|---|---|
| race | White (reference) | 66107 | 0.867 | 0.959 | 0.264 |  |  |
| race | Race Not Available | 17872 | 0.824 | 0.948 | 0.228 | -0.011 [-0.015, -0.007] | -0.035 [-0.053, -0.018] |
| race | Black or African American | 8991 | 0.739 | 0.922 | 0.222 | -0.038 [-0.045, -0.031] | -0.041 [-0.060, -0.021] |
| race | Asian | 8009 | 0.905 | 0.973 | 0.360 | +0.014 [0.010, 0.017] | +0.097 [0.063, 0.129] |
| race | Joint | 2716 | 0.872 | 0.964 | 0.242 | +0.005 [-0.003, 0.012] | -0.021 [-0.066, 0.027] |
| race | American Indian or Alaska Native | 810 | 0.691 | 0.896 | 0.145 | -0.063 [-0.086, -0.039] | -0.119 [-0.165, -0.067] |
| race | 2 or more minority races | 278 | not read |  |  |  |  |
| race | Native Hawaiian or Other Pacific Islander | 196 | not read |  |  |  |  |
| race | Free Form Text Only | 21 | not read |  |  |  |  |
| ethnicity | Not Hispanic or Latino (reference) | 70469 | 0.866 | 0.959 | 0.267 |  |  |
| ethnicity | Ethnicity Not Available | 16232 | 0.826 | 0.947 | 0.225 | -0.013 [-0.016, -0.009] | -0.042 [-0.060, -0.024] |
| ethnicity | Hispanic or Latino | 15102 | 0.796 | 0.943 | 0.232 | -0.016 [-0.021, -0.012] | -0.035 [-0.052, -0.018] |
| ethnicity | Joint | 3170 | 0.872 | 0.962 | 0.278 | +0.003 [-0.005, 0.009] | +0.011 [-0.033, 0.053] |
| ethnicity | Free Form Text Only | 27 | not read |  |  |  |  |
| sex | Joint | 38731 | 0.884 | 0.965 | 0.270 | +0.014 [0.011, 0.017] | -0.009 [-0.026, 0.009] |
| sex | Male (reference) | 34507 | 0.847 | 0.952 | 0.279 |  |  |
| sex | Female | 23462 | 0.801 | 0.945 | 0.207 | -0.007 [-0.011, -0.003] | -0.073 [-0.090, -0.056] |
| sex | Sex Not Available | 8300 | 0.841 | 0.950 | 0.253 | -0.002 [-0.007, 0.003] | -0.026 [-0.052, -0.001] |
| age | 25-34 (reference) | 30933 | 0.867 | 0.965 | 0.237 |  |  |
| age | 35-44 | 27055 | 0.856 | 0.958 | 0.269 | -0.007 [-0.011, -0.004] | +0.032 [0.012, 0.050] |
| age | 45-54 | 17437 | 0.838 | 0.952 | 0.268 | -0.014 [-0.018, -0.009] | +0.030 [0.008, 0.051] |
| age | 55-64 | 11993 | 0.830 | 0.941 | 0.246 | -0.025 [-0.029, -0.020] | +0.009 [-0.015, 0.033] |
| age | <25 | 6763 | 0.811 | 0.954 | 0.191 | -0.011 [-0.017, -0.005] | -0.047 [-0.073, -0.022] |
| age | 65-74 | 5903 | 0.846 | 0.941 | 0.289 | -0.024 [-0.031, -0.017] | +0.052 [0.021, 0.084] |
| age | (blank) | 3088 | 0.867 | 0.947 | 0.310 | -0.019 [-0.027, -0.010] | +0.073 [0.026, 0.123] |
| age | >74 | 1828 | 0.833 | 0.933 | 0.265 | -0.032 [-0.044, -0.020] | +0.028 [-0.027, 0.081] |

### Cut-off 0.50

| field | group | n | accepts | TPR | FPR | TPR gap | FPR gap |
|---|---|---|---|---|---|---|---|
| race | White (reference) | 66107 | 0.903 | 0.986 | 0.355 |  |  |
| race | Race Not Available | 17872 | 0.869 | 0.983 | 0.321 | -0.004 [-0.006, -0.002] | -0.034 [-0.054, -0.015] |
| race | Black or African American | 8991 | 0.794 | 0.972 | 0.293 | -0.015 [-0.019, -0.011] | -0.062 [-0.082, -0.041] |
| race | Asian | 8009 | 0.931 | 0.993 | 0.430 | +0.007 [0.005, 0.009] | +0.075 [0.041, 0.108] |
| race | Joint | 2716 | 0.902 | 0.985 | 0.337 | -0.002 [-0.007, 0.003] | -0.018 [-0.067, 0.032] |
| race | American Indian or Alaska Native | 810 | 0.752 | 0.961 | 0.195 | -0.025 [-0.042, -0.010] | -0.161 [-0.210, -0.107] |
| race | 2 or more minority races | 278 | not read |  |  |  |  |
| race | Native Hawaiian or Other Pacific Islander | 196 | not read |  |  |  |  |
| race | Free Form Text Only | 21 | not read |  |  |  |  |
| ethnicity | Not Hispanic or Latino (reference) | 70469 | 0.902 | 0.987 | 0.356 |  |  |
| ethnicity | Ethnicity Not Available | 16232 | 0.872 | 0.983 | 0.322 | -0.004 [-0.006, -0.002] | -0.034 [-0.052, -0.014] |
| ethnicity | Hispanic or Latino | 15102 | 0.840 | 0.979 | 0.304 | -0.008 [-0.011, -0.005] | -0.051 [-0.070, -0.032] |
| ethnicity | Joint | 3170 | 0.907 | 0.985 | 0.388 | -0.001 [-0.006, 0.003] | +0.033 [-0.013, 0.080] |
| ethnicity | Free Form Text Only | 27 | not read |  |  |  |  |
| sex | Joint | 38731 | 0.915 | 0.988 | 0.364 | +0.003 [0.002, 0.005] | -0.005 [-0.024, 0.014] |
| sex | Male (reference) | 34507 | 0.889 | 0.984 | 0.369 |  |  |
| sex | Female | 23462 | 0.843 | 0.980 | 0.278 | -0.005 [-0.007, -0.002] | -0.091 [-0.109, -0.074] |
| sex | Sex Not Available | 8300 | 0.889 | 0.986 | 0.363 | +0.001 [-0.002, 0.005] | -0.006 [-0.037, 0.021] |
| age | 25-34 (reference) | 30933 | 0.896 | 0.987 | 0.311 |  |  |
| age | 35-44 | 27055 | 0.892 | 0.987 | 0.349 | -0.000 [-0.002, 0.002] | +0.038 [0.019, 0.059] |
| age | 45-54 | 17437 | 0.880 | 0.984 | 0.357 | -0.003 [-0.005, -0.001] | +0.047 [0.025, 0.069] |
| age | 55-64 | 11993 | 0.882 | 0.981 | 0.360 | -0.006 [-0.009, -0.003] | +0.049 [0.024, 0.074] |
| age | <25 | 6763 | 0.845 | 0.979 | 0.267 | -0.009 [-0.013, -0.005] | -0.044 [-0.075, -0.017] |
| age | 65-74 | 5903 | 0.894 | 0.982 | 0.381 | -0.005 [-0.009, -0.002] | +0.070 [0.036, 0.105] |
| age | (blank) | 3088 | 0.924 | 0.988 | 0.475 | +0.001 [-0.003, 0.005] | +0.165 [0.117, 0.216] |
| age | >74 | 1828 | 0.889 | 0.980 | 0.378 | -0.007 [-0.014, -0.000] | +0.068 [0.012, 0.128] |

### Reading

The model reproduces the differences between groups in acceptance rates. In every group read here, the mean prediction is within 0.01 of the observed rate, so it inherits the gaps: observed 0.868 for White and 0.739 for Black or African American applicants, predicted 0.869 and 0.743. At the same-rate cut-off it accepts 0.867 and 0.739, the same 13-point gap as the recorded decisions. The model had these fields as inputs, so this does not show how much of the gap comes through them. Removing them is the next test.

Against each reference group, the same-rate gaps mostly point the same way in TPR and FPR. For Black or African American applicants, TPR is lower by 0.038 [-0.045, -0.031] and FPR by 0.041 [-0.060, -0.021]; for American Indian or Alaska Native by 0.063 and 0.119; for Hispanic or Latino by 0.016 and 0.035. Asian applicants go the other way, +0.014 and +0.097. So in those groups the model accepts both accepted and denied rows less often (more often for Asian applicants), which is what a shift of the whole score looks like, not an error in one direction only. For Female against Male the gap is mostly in FPR: -0.007 and -0.073. At 0.50 the TPRs are all above 0.96, so the gaps sit in the FPR.

Ranking quality differs between groups too. AUC runs from 0.880 for Asian applicants [0.865, 0.894] to 0.939 for American Indian or Alaska Native applicants [0.917, 0.960]; Female is at 0.933 and Male at 0.905, with intervals apart. A lower AUC means the model separates accepted from denied less well inside that group. It does not say why: these groups also differ in the mix of loans and in the number of denials (888 for Asian applicants).

Without the leak rows, AUC falls in every group, by 0.006 to 0.027. It falls most for Asian applicants, from 0.880 to 0.853, so the leaks do not flatter all groups equally. This is a first look only; the model trained without those columns will give the real number.

Limits of these tables: they cover 18 groups against a reference, two gaps each at two cut-offs, so 72 intervals, and about 3 or 4 would exclude 0 by chance alone. Age, sex, race and ethnicity are read one at a time, and the same rows appear in all four, so these are not separate findings. Nothing here compares similar applicants, so a gap is not evidence of unequal treatment, and these numbers do not measure the merit of the decisions.
