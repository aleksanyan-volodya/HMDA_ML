# Fairness

## What the target is

The target is a lender's recorded decision, not creditworthiness and not default. "accepted" says what a lender did with an application. It does not say whether the applicant could have repaid, and the data holds no repayment history. A model that scores well here has learned to predict past lending decisions. If some of those decisions treated a group worse than its risk justified, a faithful model carries that on, and a high AUC is no defence: AUC measures agreement with the decision, not its merit.

Three consequences for everything below:

- An "error" of the model means a disagreement with the lender, not a wrong judgement of the applicant.
- A gap between groups is an association. The data has no credit score, no interest rate and no applicant history, so it can't separate a difference in risk from a difference in treatment. We do not conclude discrimination where a gap appears, nor its absence where none does.
- The rows are a selection. The challenge page (read 2026-11-09) states the scope: rows from the HMDA Modified LAR published by the CFPB / FFIEC, home purchase loans only (loan_purpose = 1), and only applications that were originated or denied (action_taken = 1 or 3); withdrawn, incomplete and other files are excluded. So "accepted" is originated against denied. Applicants who withdrew or never completed their file are outside the data, and we can't see who they were or whether they differ by group. The collection year is not given.

## Feature lists and the cost of removing the leaks

Three lists are frozen, with the reasons in "reports/leakage.md". All use the tuned LightGBM parameters, unchanged, so that only the columns differ.

| list | config | columns | what it removes | CV AUC | gap to the list above | corrected SE |
|---|---|---|---|---|---|---|
| full | configs/experiments/fair_full.yaml | 55 | nothing | 0.91905 (run 20261003T084548Z_406e4210, same config) | | |
| clean | configs/experiments/fair_clean.yaml | 53 | the 2 confirmed leaks | 0.89305 (run 20261109T170831Z_116f6267) | -0.02600 | 0.00191 (13.6 SE) |
| strict | configs/experiments/fair_strict.yaml | 38 | also aus-1, LTV, property value, DTI, census, state | 0.82343 (run 20261109T170956Z_cd2c2a46) | -0.06962 | 0.00111 (62.8 SE) |

The gap between full and clean is how much of the competition score rests on columns that record the decision: 0.026 of AUC, far beyond noise. The gap between clean and strict, 0.070, is what removing the columns under open verdicts costs. Strict removes more than the suspect signals (it also removes the loan-to-value and debt-to-income values themselves, the census columns, the state), so the share of that 0.070 due to the suspect signals is not measured. It does show that the clean model leans heavily on the columns that hold the partial signals, which is why the location-unknown verdict stays the main doubt.

## Group diagnostics of the competition model

Model: the final blend, LightGBM 2/6 + CatBoost county 2/6 + ExtraTrees 1/6 + XGBoost 1/6 on log-odds (run 20261031T220901Z_1ff742f2, pooled out-of-fold AUC 0.920). All numbers are out-of-fold, on the 105,000 training rows, and come from "notebooks/04_fairness.ipynb", which trains nothing.

This is the competition model: it sees the demographic fields and both leaks. The tables describe that model, not a fair one. They are the baseline for the fair model.

Conventions:

- Groups are the official summaries "derived_race", "derived_ethnicity", "derived_sex", plus "applicant_age". "Not Available" and "Joint" are kept: they are large, but they are not demographic groups in the plain sense. "Not Available" mixes applicants who did not answer with applicants who are not persons; a blank age is mostly the latter.
- Groups under 500 rows are shown with their size and not read.
- Intervals are 95% bootstrap intervals over rows inside each group (1000 resamples). They ignore the uncertainty of training the model.
- A gap is measured against the largest group with one declared value: White, Not Hispanic or Latino, Male, age 25-34. It is a base for subtraction, not a norm.
- AUC has no threshold, so any cut-off is a convention. "Same rate" is 0.780, the cut-off that accepts 85% of rows, as lenders did. 0.50 is the plain reading of a probability. Each is the same for all groups.
- TPR = share of accepted rows the model accepts. FPR = share of denied rows the model accepts.

### Calibration and ranking

"pred - obs" is the mean predicted probability minus the share accepted. "AUC without leak rows" leaves out the 6,279 rows with preapproval = 1 or initially_payable_to_institution = 3; the model was still trained with those columns.

| field | group | n | denied | observed | pred - obs | AUC | AUC without leak rows |
|---|---|---|---|---|---|---|---|
| race | White | 66107 | 8721 | 0.868 [0.865, 0.871] | +0.001 [-0.001, 0.002] | 0.921 [0.917, 0.924] | 0.907 [0.903, 0.911] |
| race | Race Not Available | 17872 | 3082 | 0.828 [0.822, 0.833] | +0.003 [-0.001, 0.006] | 0.923 [0.916, 0.929] | 0.909 [0.901, 0.916] |
| race | Black or African American | 8991 | 2349 | 0.739 [0.730, 0.748] | +0.005 [-0.001, 0.011] | 0.909 [0.901, 0.917] | 0.896 [0.886, 0.904] |
| race | Asian | 8009 | 888 | 0.889 [0.882, 0.896] | +0.004 [-0.001, 0.008] | 0.882 [0.867, 0.896] | 0.855 [0.837, 0.872] |
| race | Joint | 2716 | 347 | 0.872 [0.859, 0.884] | -0.001 [-0.009, 0.007] | 0.911 [0.889, 0.933] | 0.895 [0.870, 0.921] |
| race | American Indian or Alaska Native | 810 | 221 | 0.727 [0.696, 0.758] | -0.008 [-0.026, 0.010] | 0.939 [0.917, 0.960] | 0.933 [0.908, 0.956] |
| race | 2 or more minority races | 278 | 87 | not read |  |  |  |
| race | Native Hawaiian or Other Pacific Islander | 196 | 51 | not read |  |  |  |
| race | Free Form Text Only | 21 | 10 | not read |  |  |  |
| ethnicity | Not Hispanic or Latino | 70469 | 9493 | 0.865 [0.863, 0.868] | +0.001 [-0.001, 0.003] | 0.920 [0.916, 0.924] | 0.906 [0.902, 0.910] |
| ethnicity | Ethnicity Not Available | 16232 | 2714 | 0.833 [0.827, 0.838] | +0.001 [-0.002, 0.005] | 0.924 [0.917, 0.931] | 0.910 [0.902, 0.918] |
| ethnicity | Hispanic or Latino | 15102 | 3120 | 0.793 [0.787, 0.800] | +0.004 [-0.001, 0.008] | 0.910 [0.902, 0.917] | 0.897 [0.889, 0.905] |
| ethnicity | Joint | 3170 | 417 | 0.868 [0.856, 0.880] | +0.006 [-0.001, 0.015] | 0.921 [0.903, 0.937] | 0.908 [0.889, 0.926] |
| ethnicity | Free Form Text Only | 27 | 12 | not read |  |  |  |
| sex | Joint | 38731 | 4518 | 0.883 [0.880, 0.887] | +0.001 [-0.001, 0.003] | 0.922 [0.917, 0.927] | 0.909 [0.903, 0.915] |
| sex | Male | 34507 | 5359 | 0.845 [0.841, 0.848] | +0.003 [0.001, 0.006] | 0.905 [0.900, 0.910] | 0.888 [0.882, 0.894] |
| sex | Female | 23462 | 4582 | 0.805 [0.799, 0.809] | +0.000 [-0.002, 0.004] | 0.933 [0.928, 0.938] | 0.922 [0.917, 0.928] |
| sex | Sex Not Available | 8300 | 1297 | 0.844 [0.836, 0.851] | +0.004 [-0.001, 0.009] | 0.913 [0.903, 0.923] | 0.894 [0.882, 0.906] |
| age | 25-34 | 30933 | 4162 | 0.865 [0.862, 0.869] | -0.000 [-0.002, 0.002] | 0.929 [0.923, 0.934] | 0.917 [0.910, 0.922] |
| age | 35-44 | 27055 | 4000 | 0.852 [0.848, 0.857] | +0.002 [-0.000, 0.005] | 0.915 [0.910, 0.921] | 0.899 [0.893, 0.906] |
| age | 45-54 | 17437 | 2897 | 0.834 [0.828, 0.839] | +0.006 [0.002, 0.010] | 0.913 [0.905, 0.919] | 0.897 [0.888, 0.905] |
| age | 55-64 | 11993 | 1903 | 0.841 [0.835, 0.848] | -0.000 [-0.005, 0.004] | 0.920 [0.912, 0.928] | 0.905 [0.896, 0.914] |
| age | <25 | 6763 | 1268 | 0.813 [0.804, 0.821] | +0.002 [-0.004, 0.007] | 0.936 [0.927, 0.945] | 0.928 [0.918, 0.938] |
| age | 65-74 | 5903 | 864 | 0.854 [0.845, 0.862] | -0.000 [-0.006, 0.006] | 0.914 [0.902, 0.925] | 0.901 [0.888, 0.913] |
| age | (blank) | 3088 | 387 | 0.875 [0.863, 0.887] | +0.001 [-0.008, 0.010] | 0.887 [0.864, 0.907] | 0.881 [0.857, 0.902] |
| age | >74 | 1828 | 275 | 0.850 [0.834, 0.865] | -0.000 [-0.011, 0.012] | 0.901 [0.874, 0.923] | 0.889 [0.858, 0.914] |

### Cut-off "same rate" (0.780)

| field | group | n | accepts | TPR | FPR | TPR gap | FPR gap |
|---|---|---|---|---|---|---|---|
| race | White (reference) | 66107 | 0.868 | 0.959 | 0.264 |  |  |
| race | Race Not Available | 17872 | 0.824 | 0.948 | 0.229 | -0.011 [-0.016, -0.008] | -0.035 [-0.052, -0.018] |
| race | Black or African American | 8991 | 0.739 | 0.922 | 0.220 | -0.037 [-0.044, -0.030] | -0.044 [-0.062, -0.025] |
| race | Asian | 8009 | 0.905 | 0.974 | 0.359 | +0.014 [0.010, 0.018] | +0.096 [0.061, 0.127] |
| race | Joint | 2716 | 0.872 | 0.964 | 0.245 | +0.005 [-0.003, 0.012] | -0.019 [-0.064, 0.029] |
| race | American Indian or Alaska Native | 810 | 0.694 | 0.900 | 0.145 | -0.059 [-0.083, -0.036] | -0.119 [-0.164, -0.068] |
| race | 2 or more minority races | 278 | not read |  |  |  |  |
| race | Native Hawaiian or Other Pacific Islander | 196 | not read |  |  |  |  |
| race | Free Form Text Only | 21 | not read |  |  |  |  |
| ethnicity | Not Hispanic or Latino (reference) | 70469 | 0.866 | 0.959 | 0.266 |  |  |
| ethnicity | Ethnicity Not Available | 16232 | 0.825 | 0.946 | 0.225 | -0.013 [-0.017, -0.009] | -0.041 [-0.060, -0.024] |
| ethnicity | Hispanic or Latino | 15102 | 0.797 | 0.944 | 0.235 | -0.016 [-0.020, -0.011] | -0.031 [-0.048, -0.013] |
| ethnicity | Joint | 3170 | 0.872 | 0.963 | 0.271 | +0.003 [-0.004, 0.010] | +0.005 [-0.038, 0.046] |
| ethnicity | Free Form Text Only | 27 | not read |  |  |  |  |
| sex | Joint | 38731 | 0.884 | 0.965 | 0.271 | +0.014 [0.011, 0.017] | -0.009 [-0.026, 0.009] |
| sex | Male (reference) | 34507 | 0.848 | 0.952 | 0.280 |  |  |
| sex | Female | 23462 | 0.800 | 0.945 | 0.205 | -0.007 [-0.011, -0.003] | -0.075 [-0.092, -0.058] |
| sex | Sex Not Available | 8300 | 0.840 | 0.949 | 0.249 | -0.002 [-0.008, 0.003] | -0.031 [-0.058, -0.006] |
| age | 25-34 (reference) | 30933 | 0.868 | 0.966 | 0.237 |  |  |
| age | 35-44 | 27055 | 0.857 | 0.958 | 0.270 | -0.007 [-0.010, -0.004] | +0.033 [0.013, 0.051] |
| age | 45-54 | 17437 | 0.839 | 0.952 | 0.269 | -0.014 [-0.018, -0.010] | +0.032 [0.010, 0.054] |
| age | 55-64 | 11993 | 0.830 | 0.940 | 0.244 | -0.025 [-0.030, -0.021] | +0.007 [-0.016, 0.032] |
| age | <25 | 6763 | 0.809 | 0.952 | 0.192 | -0.014 [-0.020, -0.008] | -0.045 [-0.072, -0.021] |
| age | 65-74 | 5903 | 0.845 | 0.941 | 0.285 | -0.025 [-0.032, -0.018] | +0.048 [0.016, 0.081] |
| age | (blank) | 3088 | 0.866 | 0.946 | 0.307 | -0.019 [-0.028, -0.011] | +0.071 [0.024, 0.120] |
| age | >74 | 1828 | 0.833 | 0.933 | 0.265 | -0.033 [-0.045, -0.020] | +0.029 [-0.026, 0.083] |

### Cut-off 0.50

| field | group | n | accepts | TPR | FPR | TPR gap | FPR gap |
|---|---|---|---|---|---|---|---|
| race | White (reference) | 66107 | 0.903 | 0.986 | 0.355 |  |  |
| race | Race Not Available | 17872 | 0.870 | 0.983 | 0.325 | -0.003 [-0.006, -0.001] | -0.030 [-0.050, -0.012] |
| race | Black or African American | 8991 | 0.796 | 0.973 | 0.296 | -0.014 [-0.018, -0.010] | -0.058 [-0.080, -0.038] |
| race | Asian | 8009 | 0.931 | 0.993 | 0.432 | +0.007 [0.004, 0.009] | +0.078 [0.043, 0.110] |
| race | Joint | 2716 | 0.904 | 0.986 | 0.337 | -0.000 [-0.005, 0.005] | -0.018 [-0.066, 0.031] |
| race | American Indian or Alaska Native | 810 | 0.753 | 0.964 | 0.190 | -0.022 [-0.037, -0.008] | -0.165 [-0.217, -0.111] |
| race | 2 or more minority races | 278 | not read |  |  |  |  |
| race | Native Hawaiian or Other Pacific Islander | 196 | not read |  |  |  |  |
| race | Free Form Text Only | 21 | not read |  |  |  |  |
| ethnicity | Not Hispanic or Latino (reference) | 70469 | 0.902 | 0.987 | 0.356 |  |  |
| ethnicity | Ethnicity Not Available | 16232 | 0.873 | 0.983 | 0.325 | -0.004 [-0.006, -0.001] | -0.030 [-0.049, -0.010] |
| ethnicity | Hispanic or Latino | 15102 | 0.840 | 0.979 | 0.306 | -0.008 [-0.011, -0.005] | -0.050 [-0.069, -0.031] |
| ethnicity | Joint | 3170 | 0.908 | 0.986 | 0.388 | -0.001 [-0.005, 0.004] | +0.033 [-0.014, 0.079] |
| ethnicity | Free Form Text Only | 27 | not read |  |  |  |  |
| sex | Joint | 38731 | 0.916 | 0.988 | 0.365 | +0.004 [0.002, 0.006] | -0.006 [-0.025, 0.014] |
| sex | Male (reference) | 34507 | 0.889 | 0.984 | 0.371 |  |  |
| sex | Female | 23462 | 0.843 | 0.980 | 0.277 | -0.004 [-0.007, -0.002] | -0.094 [-0.112, -0.076] |
| sex | Sex Not Available | 8300 | 0.890 | 0.987 | 0.368 | +0.002 [-0.001, 0.005] | -0.003 [-0.032, 0.025] |
| age | 25-34 (reference) | 30933 | 0.896 | 0.987 | 0.310 |  |  |
| age | 35-44 | 27055 | 0.893 | 0.987 | 0.352 | -0.000 [-0.002, 0.002] | +0.042 [0.023, 0.063] |
| age | 45-54 | 17437 | 0.881 | 0.985 | 0.362 | -0.003 [-0.005, -0.000] | +0.052 [0.031, 0.075] |
| age | 55-64 | 11993 | 0.883 | 0.981 | 0.359 | -0.006 [-0.009, -0.003] | +0.050 [0.024, 0.074] |
| age | <25 | 6763 | 0.845 | 0.979 | 0.267 | -0.009 [-0.013, -0.005] | -0.043 [-0.073, -0.016] |
| age | 65-74 | 5903 | 0.893 | 0.982 | 0.374 | -0.005 [-0.009, -0.001] | +0.064 [0.029, 0.100] |
| age | (blank) | 3088 | 0.926 | 0.990 | 0.483 | +0.002 [-0.002, 0.006] | +0.173 [0.128, 0.225] |
| age | >74 | 1828 | 0.888 | 0.979 | 0.378 | -0.009 [-0.016, -0.002] | +0.068 [0.012, 0.129] |

### Reading

The model reproduces the differences between groups in acceptance rates. In every group read here, the mean prediction is within 0.01 of the observed rate, so it inherits the gaps: observed 0.868 for White and 0.739 for Black or African American applicants, predicted 0.869 and 0.744. At the same-rate cut-off it accepts 0.868 and 0.739, the same 13-point gap as the recorded decisions. The model had these fields as inputs, so this does not show how much of the gap comes through them. The fair models below remove them.

Against each reference group, the same-rate gaps mostly point the same way in TPR and FPR. For Black or African American applicants, TPR is lower by 0.037 [-0.044, -0.030] and FPR by 0.044 [-0.062, -0.025]; for American Indian or Alaska Native by 0.059 and 0.119; for Hispanic or Latino by 0.016 and 0.031. Asian applicants go the other way, +0.014 and +0.096. So in those groups the model accepts both accepted and denied rows less often (more often for Asian applicants). For Female against Male the gap is mostly in FPR: -0.007 and -0.075. At 0.50 the TPRs are all above 0.96, so the gaps sit in the FPR.

Ranking quality differs between groups too. AUC runs from 0.882 for Asian applicants [0.867, 0.896] to 0.939 for American Indian or Alaska Native applicants [0.917, 0.960]; Female is at 0.933 and Male at 0.905, with intervals apart. A lower AUC means the model separates accepted from denied less well inside that group.

Without the leak rows, AUC falls in every group, by 0.006 to 0.027. It falls most for Asian applicants, from 0.882 to 0.855, so the leaks do not flatter all groups equally. The models trained without those columns, below, confirm it: Asian applicants stay the group the model ranks least well.

Limits of these tables: they cover 18 groups against a reference, two gaps each at two cut-offs, so 72 intervals, and about 3 or 4 would exclude 0 by chance alone. Age, sex, race and ethnicity are read one at a time, and the same rows appear in all four, so these are not separate findings. Nothing here compares similar applicants, so a gap is not evidence of unequal treatment, and these numbers do not measure the merit of the decisions.

## Fair models

### What was built

Six LightGBM models with the same tuned parameters, on the same frozen folds, so that one thing changes at a time:

- Two feature lists. The Kaggle list keeps everything, leaks included: it is what the leaderboard scores. The clean list removes the 2 confirmed leaks: it is the one a lender could use before deciding, so the fair model we put forward is built on it.
- Three treatments on each list. With the 17 demographic fields; race-blind, without them (race, ethnicity, sex and age of applicant and co-applicant, and their derived summaries); race-blind with reweighing.
- Reweighing (Kamiran and Calders, 2012): in the training part of each fold, each row gets the weight P(g) P(y) / P(g, y), with g = "derived_race" and y = accepted. In the weighted training data, race and the label are independent: accepted Black applicants weigh more, denied Black applicants less, and the reverse for White applicants (weights 0.3 to 1.6). Race only sets the weights; it is not a feature. Scoring is unweighted, as for every other run.

The fair model we put forward is fair_reweigh: clean list, race-blind, reweighed. Its Kaggle-list twin, fair_full_reweigh, measures what the same two steps cost on the leaderboard.

### Cost

CV AUC changes are paired per-fold differences with the corrected standard error of "make compare". The public leaderboard has a noise SD of about 0.0038 for one score and 0.0014 for a gap between two submissions.

| model | config | columns | CV AUC | vs the step before | public LB (submission) |
|---|---|---|---|---|---|
| final blend (competition model) | blends/lgbm_cat_et_xgb.yaml | | 0.92054 | | 0.92158 (6) |
| Kaggle list, with the demographic fields | fair_full | 55 | 0.91905 | | 0.92100 (4, same config) |
| Kaggle list, race-blind | ablate_no_demographics | 38 | 0.91636 | -0.00270 (-5.2 SE) | 0.91869 (7) |
| Kaggle list, race-blind + reweighing | fair_full_reweigh | 38 | 0.91560 | -0.00075 (-0.9 SE) | 0.91855 (8) |
| clean list, with the demographic fields | fair_clean | 53 | 0.89305 | -0.02600 (-13.6 SE), against fair_full | not submitted |
| clean list, race-blind | fair_unaware | 36 | 0.89060 | -0.00245 (-3.1 SE) | 0.89408 (9) |
| clean list, race-blind + reweighing | fair_reweigh | 36 | 0.88985 | -0.00075 (-2.4 SE) | 0.89342 (10) |

- Making the model race-blind costs 0.0027 of CV AUC on the Kaggle list (5.2 SE) and 0.0025 on the clean list (3.1 SE). On the public leaderboard it costs 0.0023 (submission 4 to 7), about 1.6 paired SD: the same size, so the cost transfers to unseen rows.
- Reweighing on top costs 0.0008 on both lists, not resolved on the Kaggle list (0.9 SE), 2.4 SE on the clean list; on the leaderboard it costs 0.0001 and 0.0007, inside the noise.
- The leaks cost 0.026, seven to eight times more than both fairness steps together. Against the competition blend, the fair Kaggle-list model loses 0.0049 of CV AUC and 0.0030 on the public leaderboard.

### What each step changes, for the three largest pairs

Gaps are other group minus reference group at each model's same-rate cut-off, with 95% bootstrap intervals. The first row of each pair is the recorded decisions: the gap a perfectly faithful model would reproduce.

| pair | model | accept gap | TPR gap | FPR gap |
|---|---|---|---|---|
| Black or African American vs White | recorded decisions | -0.129 [-0.138, -0.120] |  |  |
| Black or African American vs White | final blend | -0.129 [-0.139, -0.119] | -0.037 [-0.044, -0.030] | -0.044 [-0.062, -0.025] |
| Black or African American vs White | Kaggle list, with demographics | -0.129 [-0.139, -0.119] | -0.038 [-0.045, -0.031] | -0.042 [-0.060, -0.023] |
| Black or African American vs White | Kaggle list, race-blind | -0.104 [-0.113, -0.095] | -0.013 [-0.019, -0.007] | -0.017 [-0.037, 0.003] |
| Black or African American vs White | Kaggle list, race-blind + reweighing | -0.098 [-0.107, -0.088] | -0.008 [-0.013, -0.002] | -0.008 [-0.027, 0.012] |
| Black or African American vs White | clean list, with demographics | -0.139 [-0.148, -0.130] | -0.055 [-0.062, -0.047] | -0.065 [-0.084, -0.043] |
| Black or African American vs White | clean list, race-blind | -0.108 [-0.117, -0.099] | -0.024 [-0.031, -0.018] | -0.032 [-0.053, -0.011] |
| Black or African American vs White | clean list, race-blind + reweighing | -0.096 [-0.104, -0.087] | -0.010 [-0.017, -0.005] | -0.023 [-0.044, -0.002] |
| Hispanic or Latino vs Not Hispanic or Latino | recorded decisions | -0.072 [-0.079, -0.065] |  |  |
| Hispanic or Latino vs Not Hispanic or Latino | final blend | -0.069 [-0.075, -0.062] | -0.016 [-0.020, -0.011] | -0.031 [-0.048, -0.013] |
| Hispanic or Latino vs Not Hispanic or Latino | Kaggle list, with demographics | -0.069 [-0.076, -0.063] | -0.015 [-0.020, -0.011] | -0.035 [-0.051, -0.017] |
| Hispanic or Latino vs Not Hispanic or Latino | Kaggle list, race-blind | -0.065 [-0.072, -0.058] | -0.012 [-0.017, -0.008] | -0.026 [-0.044, -0.009] |
| Hispanic or Latino vs Not Hispanic or Latino | Kaggle list, race-blind + reweighing | -0.064 [-0.071, -0.057] | -0.011 [-0.016, -0.007] | -0.025 [-0.042, -0.007] |
| Hispanic or Latino vs Not Hispanic or Latino | clean list, with demographics | -0.071 [-0.078, -0.065] | -0.020 [-0.025, -0.014] | -0.050 [-0.068, -0.031] |
| Hispanic or Latino vs Not Hispanic or Latino | clean list, race-blind | -0.068 [-0.076, -0.062] | -0.017 [-0.022, -0.012] | -0.047 [-0.066, -0.029] |
| Hispanic or Latino vs Not Hispanic or Latino | clean list, race-blind + reweighing | -0.067 [-0.074, -0.060] | -0.016 [-0.022, -0.011] | -0.044 [-0.062, -0.025] |
| Female vs Male | recorded decisions | -0.040 [-0.047, -0.034] |  |  |
| Female vs Male | final blend | -0.047 [-0.053, -0.041] | -0.007 [-0.011, -0.003] | -0.075 [-0.092, -0.058] |
| Female vs Male | Kaggle list, with demographics | -0.048 [-0.054, -0.042] | -0.008 [-0.013, -0.004] | -0.073 [-0.090, -0.056] |
| Female vs Male | Kaggle list, race-blind | -0.048 [-0.054, -0.042] | -0.008 [-0.012, -0.004] | -0.075 [-0.093, -0.059] |
| Female vs Male | Kaggle list, race-blind + reweighing | -0.048 [-0.054, -0.041] | -0.008 [-0.012, -0.004] | -0.074 [-0.092, -0.058] |
| Female vs Male | clean list, with demographics | -0.048 [-0.055, -0.042] | -0.010 [-0.014, -0.005] | -0.082 [-0.100, -0.066] |
| Female vs Male | clean list, race-blind | -0.048 [-0.054, -0.041] | -0.009 [-0.014, -0.005] | -0.082 [-0.101, -0.064] |
| Female vs Male | clean list, race-blind + reweighing | -0.047 [-0.053, -0.040] | -0.008 [-0.013, -0.004] | -0.082 [-0.100, -0.064] |

Step by step: the change of the accept gap, with a paired interval (the bootstrap draws the same rows for every model), and in brackets the share of the gap before the step that the step removes. A positive change on a negative gap means the gap narrowed.

| step | AUC change | Black or African American accept gap | Hispanic or Latino accept gap | Female accept gap |
|---|---|---|---|---|
| Kaggle list, with demographics -> Kaggle list, race-blind | -0.00270 (-5.2 SE) | -0.129 -> -0.104: +0.025 [0.021, 0.028] (19%) | -0.069 -> -0.065: +0.004 [0.002, 0.006] (6%) | -0.048 -> -0.048: -0.000 [-0.002, 0.002] (-0%) |
| Kaggle list, race-blind -> Kaggle list, race-blind + reweighing | -0.00075 (-0.9 SE) | -0.104 -> -0.098: +0.006 [0.004, 0.009] (6%) | -0.065 -> -0.064: +0.001 [-0.001, 0.003] (2%) | -0.048 -> -0.048: +0.000 [-0.001, 0.002] (1%) |
| Kaggle list, with demographics -> clean list, with demographics | -0.02600 (-13.6 SE) | -0.129 -> -0.139: -0.010 [-0.015, -0.006] (-8%) | -0.069 -> -0.071: -0.002 [-0.005, 0.001] (-3%) | -0.048 -> -0.048: -0.000 [-0.003, 0.002] (-1%) |
| clean list, with demographics -> clean list, race-blind | -0.00245 (-3.1 SE) | -0.139 -> -0.108: +0.031 [0.027, 0.035] (23%) | -0.071 -> -0.068: +0.003 [0.000, 0.005] (4%) | -0.048 -> -0.048: +0.001 [-0.002, 0.003] (1%) |
| clean list, race-blind -> clean list, race-blind + reweighing | -0.00075 (-2.4 SE) | -0.108 -> -0.096: +0.012 [0.009, 0.015] (11%) | -0.068 -> -0.067: +0.001 [-0.001, 0.003] (2%) | -0.048 -> -0.047: +0.001 [-0.001, 0.003] (2%) |

### Every group, side by side

Columns: the final blend, then the four fair models by config name (ablate_no_demographics and fair_full_reweigh on the Kaggle list, fair_unaware and fair_reweigh on the clean list). Gaps are at each model's own same-rate cut-off, against the same reference groups as above.

Accept gap:

| field | group | n | final blend | ablate_no_demographics | fair_full_reweigh | fair_unaware | fair_reweigh |
|---|---|---|---|---|---|---|---|
| race | White | 66107 | reference | reference | reference | reference | reference |
| race | Race Not Available | 17872 | -0.044 [-0.050, -0.038] | -0.035 [-0.041, -0.029] | -0.033 [-0.039, -0.027] | -0.035 [-0.041, -0.029] | -0.031 [-0.037, -0.025] |
| race | Black or African American | 8991 | -0.129 [-0.139, -0.119] | -0.104 [-0.113, -0.095] | -0.098 [-0.107, -0.088] | -0.108 [-0.117, -0.099] | -0.096 [-0.104, -0.087] |
| race | Asian | 8009 | +0.038 [0.031, 0.045] | +0.041 [0.034, 0.048] | +0.041 [0.035, 0.048] | +0.048 [0.041, 0.055] | +0.048 [0.041, 0.054] |
| race | Joint | 2716 | +0.005 [-0.008, 0.017] | +0.009 [-0.004, 0.022] | +0.012 [-0.001, 0.024] | +0.011 [-0.003, 0.023] | +0.015 [0.001, 0.028] |
| race | American Indian or Alaska Native | 810 | -0.174 [-0.204, -0.143] | -0.163 [-0.194, -0.132] | -0.165 [-0.196, -0.133] | -0.167 [-0.198, -0.136] | -0.160 [-0.190, -0.128] |
| ethnicity | Not Hispanic or Latino | 70469 | reference | reference | reference | reference | reference |
| ethnicity | Ethnicity Not Available | 16232 | -0.041 [-0.047, -0.034] | -0.035 [-0.041, -0.029] | -0.033 [-0.039, -0.027] | -0.034 [-0.040, -0.028] | -0.032 [-0.038, -0.026] |
| ethnicity | Hispanic or Latino | 15102 | -0.069 [-0.075, -0.062] | -0.065 [-0.072, -0.058] | -0.064 [-0.071, -0.057] | -0.068 [-0.076, -0.062] | -0.067 [-0.074, -0.060] |
| ethnicity | Joint | 3170 | +0.006 [-0.006, 0.017] | +0.009 [-0.003, 0.020] | +0.009 [-0.003, 0.021] | +0.011 [-0.001, 0.022] | +0.010 [-0.002, 0.021] |
| sex | Joint | 38731 | +0.037 [0.032, 0.042] | +0.034 [0.029, 0.039] | +0.032 [0.027, 0.037] | +0.036 [0.031, 0.041] | +0.034 [0.030, 0.039] |
| sex | Male | 34507 | reference | reference | reference | reference | reference |
| sex | Female | 23462 | -0.047 [-0.053, -0.041] | -0.048 [-0.054, -0.042] | -0.048 [-0.054, -0.041] | -0.048 [-0.054, -0.041] | -0.047 [-0.053, -0.040] |
| sex | Sex Not Available | 8300 | -0.008 [-0.016, 0.001] | -0.005 [-0.014, 0.003] | -0.003 [-0.012, 0.006] | -0.005 [-0.013, 0.003] | -0.000 [-0.008, 0.009] |
| age | 25-34 | 30933 | reference | reference | reference | reference | reference |
| age | 35-44 | 27055 | -0.011 [-0.017, -0.005] | -0.006 [-0.012, -0.001] | -0.007 [-0.013, -0.001] | -0.005 [-0.010, 0.000] | -0.004 [-0.010, 0.001] |
| age | 45-54 | 17437 | -0.029 [-0.036, -0.023] | -0.023 [-0.030, -0.017] | -0.023 [-0.029, -0.017] | -0.021 [-0.027, -0.015] | -0.022 [-0.028, -0.016] |
| age | 55-64 | 11993 | -0.038 [-0.046, -0.030] | -0.032 [-0.040, -0.025] | -0.035 [-0.043, -0.027] | -0.034 [-0.041, -0.026] | -0.035 [-0.043, -0.028] |
| age | <25 | 6763 | -0.058 [-0.068, -0.049] | -0.060 [-0.069, -0.050] | -0.061 [-0.071, -0.052] | -0.063 [-0.073, -0.053] | -0.064 [-0.074, -0.055] |
| age | 65-74 | 5903 | -0.023 [-0.033, -0.012] | -0.026 [-0.036, -0.015] | -0.030 [-0.040, -0.020] | -0.032 [-0.043, -0.021] | -0.032 [-0.043, -0.021] |
| age | (blank) | 3088 | -0.001 [-0.013, 0.011] | -0.008 [-0.021, 0.005] | -0.001 [-0.014, 0.011] | -0.021 [-0.033, -0.007] | -0.010 [-0.022, 0.003] |
| age | >74 | 1828 | -0.035 [-0.052, -0.018] | -0.038 [-0.055, -0.021] | -0.043 [-0.060, -0.025] | -0.044 [-0.062, -0.026] | -0.041 [-0.060, -0.023] |

TPR gap:

| field | group | n | final blend | ablate_no_demographics | fair_full_reweigh | fair_unaware | fair_reweigh |
|---|---|---|---|---|---|---|---|
| race | White | 66107 | reference | reference | reference | reference | reference |
| race | Race Not Available | 17872 | -0.011 [-0.016, -0.008] | -0.004 [-0.008, 0.000] | -0.002 [-0.006, 0.001] | -0.007 [-0.011, -0.003] | -0.003 [-0.007, 0.001] |
| race | Black or African American | 8991 | -0.037 [-0.044, -0.030] | -0.013 [-0.019, -0.007] | -0.008 [-0.013, -0.002] | -0.024 [-0.031, -0.018] | -0.010 [-0.017, -0.005] |
| race | Asian | 8009 | +0.014 [0.010, 0.018] | +0.017 [0.013, 0.021] | +0.017 [0.013, 0.021] | +0.022 [0.018, 0.026] | +0.022 [0.018, 0.026] |
| race | Joint | 2716 | +0.005 [-0.003, 0.012] | +0.009 [0.002, 0.017] | +0.012 [0.004, 0.019] | +0.013 [0.005, 0.021] | +0.016 [0.008, 0.024] |
| race | American Indian or Alaska Native | 810 | -0.059 [-0.083, -0.036] | -0.050 [-0.074, -0.028] | -0.051 [-0.075, -0.028] | -0.055 [-0.081, -0.030] | -0.043 [-0.067, -0.020] |
| ethnicity | Not Hispanic or Latino | 70469 | reference | reference | reference | reference | reference |
| ethnicity | Ethnicity Not Available | 16232 | -0.013 [-0.017, -0.009] | -0.008 [-0.012, -0.005] | -0.007 [-0.010, -0.003] | -0.012 [-0.016, -0.007] | -0.009 [-0.013, -0.005] |
| ethnicity | Hispanic or Latino | 15102 | -0.016 [-0.020, -0.011] | -0.012 [-0.017, -0.008] | -0.011 [-0.016, -0.007] | -0.017 [-0.022, -0.012] | -0.016 [-0.022, -0.011] |
| ethnicity | Joint | 3170 | +0.003 [-0.004, 0.010] | +0.005 [-0.002, 0.012] | +0.006 [-0.001, 0.012] | +0.010 [0.002, 0.018] | +0.009 [0.001, 0.016] |
| sex | Joint | 38731 | +0.014 [0.011, 0.017] | +0.011 [0.008, 0.015] | +0.010 [0.007, 0.013] | +0.017 [0.014, 0.020] | +0.016 [0.012, 0.019] |
| sex | Male | 34507 | reference | reference | reference | reference | reference |
| sex | Female | 23462 | -0.007 [-0.011, -0.003] | -0.008 [-0.012, -0.004] | -0.008 [-0.012, -0.004] | -0.009 [-0.014, -0.005] | -0.008 [-0.013, -0.004] |
| sex | Sex Not Available | 8300 | -0.002 [-0.008, 0.003] | -0.001 [-0.006, 0.004] | +0.001 [-0.003, 0.007] | -0.004 [-0.011, 0.002] | +0.001 [-0.005, 0.006] |
| age | 25-34 | 30933 | reference | reference | reference | reference | reference |
| age | 35-44 | 27055 | -0.007 [-0.010, -0.004] | -0.003 [-0.006, 0.001] | -0.003 [-0.007, 0.000] | -0.005 [-0.008, -0.001] | -0.004 [-0.008, 0.000] |
| age | 45-54 | 17437 | -0.014 [-0.018, -0.010] | -0.008 [-0.012, -0.004] | -0.008 [-0.011, -0.004] | -0.009 [-0.013, -0.005] | -0.010 [-0.014, -0.005] |
| age | 55-64 | 11993 | -0.025 [-0.030, -0.021] | -0.022 [-0.027, -0.017] | -0.023 [-0.028, -0.018] | -0.026 [-0.031, -0.020] | -0.027 [-0.033, -0.021] |
| age | <25 | 6763 | -0.014 [-0.020, -0.008] | -0.015 [-0.021, -0.009] | -0.017 [-0.024, -0.011] | -0.021 [-0.027, -0.013] | -0.022 [-0.028, -0.015] |
| age | 65-74 | 5903 | -0.025 [-0.032, -0.018] | -0.029 [-0.037, -0.022] | -0.033 [-0.041, -0.026] | -0.036 [-0.044, -0.028] | -0.036 [-0.044, -0.028] |
| age | (blank) | 3088 | -0.019 [-0.028, -0.011] | -0.025 [-0.034, -0.015] | -0.018 [-0.027, -0.009] | -0.032 [-0.043, -0.022] | -0.020 [-0.030, -0.011] |
| age | >74 | 1828 | -0.033 [-0.045, -0.020] | -0.038 [-0.051, -0.025] | -0.044 [-0.057, -0.030] | -0.042 [-0.056, -0.028] | -0.040 [-0.054, -0.027] |

FPR gap:

| field | group | n | final blend | ablate_no_demographics | fair_full_reweigh | fair_unaware | fair_reweigh |
|---|---|---|---|---|---|---|---|
| race | White | 66107 | reference | reference | reference | reference | reference |
| race | Race Not Available | 17872 | -0.035 [-0.052, -0.018] | -0.021 [-0.039, -0.003] | -0.016 [-0.034, 0.002] | -0.021 [-0.040, -0.003] | -0.017 [-0.035, 0.001] |
| race | Black or African American | 8991 | -0.044 [-0.062, -0.025] | -0.017 [-0.037, 0.003] | -0.008 [-0.027, 0.012] | -0.032 [-0.053, -0.011] | -0.023 [-0.044, -0.002] |
| race | Asian | 8009 | +0.096 [0.061, 0.127] | +0.103 [0.068, 0.135] | +0.102 [0.068, 0.134] | +0.140 [0.107, 0.173] | +0.132 [0.098, 0.166] |
| race | Joint | 2716 | -0.019 [-0.064, 0.029] | -0.020 [-0.063, 0.028] | -0.013 [-0.059, 0.034] | -0.026 [-0.068, 0.024] | -0.017 [-0.060, 0.031] |
| race | American Indian or Alaska Native | 810 | -0.119 [-0.164, -0.068] | -0.105 [-0.150, -0.054] | -0.109 [-0.154, -0.057] | -0.137 [-0.186, -0.079] | -0.144 [-0.194, -0.089] |
| ethnicity | Not Hispanic or Latino | 70469 | reference | reference | reference | reference | reference |
| ethnicity | Ethnicity Not Available | 16232 | -0.041 [-0.060, -0.024] | -0.032 [-0.050, -0.014] | -0.031 [-0.048, -0.012] | -0.026 [-0.043, -0.006] | -0.025 [-0.044, -0.006] |
| ethnicity | Hispanic or Latino | 15102 | -0.031 [-0.048, -0.013] | -0.026 [-0.044, -0.009] | -0.025 [-0.042, -0.007] | -0.047 [-0.066, -0.029] | -0.044 [-0.062, -0.025] |
| ethnicity | Joint | 3170 | +0.005 [-0.038, 0.046] | +0.014 [-0.030, 0.056] | +0.015 [-0.027, 0.057] | +0.003 [-0.043, 0.046] | -0.001 [-0.046, 0.040] |
| sex | Joint | 38731 | -0.009 [-0.026, 0.009] | -0.013 [-0.031, 0.005] | -0.018 [-0.035, 0.001] | -0.021 [-0.039, -0.002] | -0.023 [-0.040, -0.005] |
| sex | Male | 34507 | reference | reference | reference | reference | reference |
| sex | Female | 23462 | -0.075 [-0.092, -0.058] | -0.075 [-0.093, -0.059] | -0.074 [-0.092, -0.058] | -0.082 [-0.101, -0.064] | -0.082 [-0.100, -0.064] |
| sex | Sex Not Available | 8300 | -0.031 [-0.058, -0.006] | -0.027 [-0.052, -0.001] | -0.024 [-0.051, 0.001] | -0.007 [-0.034, 0.022] | -0.004 [-0.033, 0.023] |
| age | 25-34 | 30933 | reference | reference | reference | reference | reference |
| age | 35-44 | 27055 | +0.033 [0.013, 0.051] | +0.039 [0.021, 0.058] | +0.037 [0.019, 0.056] | +0.053 [0.032, 0.073] | +0.053 [0.033, 0.073] |
| age | 45-54 | 17437 | +0.032 [0.010, 0.054] | +0.040 [0.018, 0.061] | +0.038 [0.016, 0.060] | +0.047 [0.024, 0.069] | +0.044 [0.021, 0.066] |
| age | 55-64 | 11993 | +0.007 [-0.016, 0.032] | +0.020 [-0.004, 0.044] | +0.010 [-0.014, 0.034] | +0.027 [0.002, 0.053] | +0.023 [-0.002, 0.049] |
| age | <25 | 6763 | -0.045 [-0.072, -0.021] | -0.046 [-0.073, -0.020] | -0.047 [-0.073, -0.020] | -0.058 [-0.086, -0.031] | -0.060 [-0.088, -0.033] |
| age | 65-74 | 5903 | +0.048 [0.016, 0.081] | +0.052 [0.020, 0.085] | +0.047 [0.015, 0.078] | +0.043 [0.009, 0.077] | +0.042 [0.009, 0.077] |
| age | (blank) | 3088 | +0.071 [0.024, 0.120] | +0.051 [0.007, 0.099] | +0.061 [0.016, 0.110] | +0.011 [-0.033, 0.061] | +0.013 [-0.032, 0.063] |
| age | >74 | 1828 | +0.029 [-0.026, 0.083] | +0.045 [-0.009, 0.100] | +0.041 [-0.012, 0.096] | +0.019 [-0.037, 0.079] | +0.023 [-0.036, 0.083] |

Mean prediction minus observed rate (calibration):

| field | group | n | final blend | ablate_no_demographics | fair_full_reweigh | fair_unaware | fair_reweigh |
|---|---|---|---|---|---|---|---|
| race | White | 66107 | +0.001 [-0.001, 0.002] | -0.005 [-0.007, -0.003] | -0.006 [-0.008, -0.005] | -0.006 [-0.008, -0.004] | -0.007 [-0.009, -0.006] |
| race | Race Not Available | 17872 | +0.003 [-0.001, 0.006] | +0.008 [0.005, 0.012] | +0.010 [0.006, 0.014] | +0.011 [0.008, 0.015] | +0.013 [0.009, 0.016] |
| race | Black or African American | 8991 | +0.005 [-0.001, 0.011] | +0.029 [0.023, 0.035] | +0.035 [0.029, 0.041] | +0.031 [0.025, 0.037] | +0.038 [0.031, 0.044] |
| race | Asian | 8009 | +0.004 [-0.001, 0.008] | +0.006 [0.001, 0.011] | +0.004 [-0.001, 0.009] | +0.007 [0.001, 0.012] | +0.004 [-0.001, 0.010] |
| race | Joint | 2716 | -0.001 [-0.009, 0.007] | +0.002 [-0.006, 0.010] | +0.000 [-0.008, 0.008] | +0.000 [-0.008, 0.009] | -0.000 [-0.008, 0.008] |
| race | American Indian or Alaska Native | 810 | -0.008 [-0.026, 0.010] | +0.000 [-0.018, 0.019] | +0.004 [-0.014, 0.023] | -0.002 [-0.021, 0.019] | +0.002 [-0.017, 0.022] |
| ethnicity | Not Hispanic or Latino | 70469 | +0.001 [-0.001, 0.003] | -0.001 [-0.003, 0.001] | -0.002 [-0.004, -0.000] | -0.002 [-0.004, 0.000] | -0.003 [-0.005, -0.001] |
| ethnicity | Ethnicity Not Available | 16232 | +0.001 [-0.002, 0.005] | +0.005 [0.002, 0.009] | +0.007 [0.003, 0.010] | +0.009 [0.005, 0.013] | +0.010 [0.006, 0.014] |
| ethnicity | Hispanic or Latino | 15102 | +0.004 [-0.001, 0.008] | +0.007 [0.003, 0.011] | +0.010 [0.005, 0.014] | +0.006 [0.001, 0.010] | +0.008 [0.003, 0.012] |
| ethnicity | Joint | 3170 | +0.006 [-0.001, 0.015] | +0.007 [-0.000, 0.016] | +0.006 [-0.002, 0.014] | +0.007 [-0.002, 0.015] | +0.005 [-0.003, 0.013] |
| sex | Joint | 38731 | +0.001 [-0.001, 0.003] | -0.002 [-0.004, 0.000] | -0.004 [-0.006, -0.002] | -0.002 [-0.004, -0.000] | -0.004 [-0.007, -0.002] |
| sex | Male | 34507 | +0.003 [0.001, 0.006] | +0.004 [0.001, 0.006] | +0.004 [0.002, 0.007] | +0.003 [0.001, 0.006] | +0.004 [0.001, 0.006] |
| sex | Female | 23462 | +0.000 [-0.002, 0.004] | +0.001 [-0.002, 0.004] | +0.002 [-0.001, 0.005] | +0.000 [-0.003, 0.004] | +0.002 [-0.001, 0.006] |
| sex | Sex Not Available | 8300 | +0.004 [-0.001, 0.009] | +0.007 [0.001, 0.012] | +0.009 [0.004, 0.014] | +0.013 [0.007, 0.019] | +0.015 [0.009, 0.021] |
| age | 25-34 | 30933 | -0.000 [-0.002, 0.002] | -0.004 [-0.006, -0.001] | -0.004 [-0.006, -0.001] | -0.005 [-0.008, -0.003] | -0.005 [-0.008, -0.002] |
| age | 35-44 | 27055 | +0.002 [-0.000, 0.005] | +0.005 [0.003, 0.008] | +0.005 [0.002, 0.008] | +0.006 [0.003, 0.009] | +0.006 [0.003, 0.009] |
| age | 45-54 | 17437 | +0.006 [0.002, 0.010] | +0.011 [0.007, 0.015] | +0.010 [0.007, 0.014] | +0.012 [0.008, 0.016] | +0.011 [0.007, 0.015] |
| age | 55-64 | 11993 | -0.000 [-0.005, 0.004] | +0.001 [-0.004, 0.005] | -0.000 [-0.005, 0.004] | +0.003 [-0.001, 0.008] | +0.002 [-0.003, 0.007] |
| age | <25 | 6763 | +0.002 [-0.004, 0.007] | -0.002 [-0.008, 0.003] | -0.002 [-0.008, 0.003] | -0.004 [-0.010, 0.002] | -0.003 [-0.010, 0.003] |
| age | 65-74 | 5903 | -0.000 [-0.006, 0.006] | -0.005 [-0.011, 0.001] | -0.008 [-0.013, -0.001] | -0.006 [-0.012, 0.001] | -0.009 [-0.015, -0.002] |
| age | (blank) | 3088 | +0.001 [-0.008, 0.010] | -0.007 [-0.015, 0.003] | +0.000 [-0.009, 0.010] | -0.006 [-0.015, 0.003] | +0.001 [-0.008, 0.010] |
| age | >74 | 1828 | -0.000 [-0.011, 0.012] | -0.008 [-0.019, 0.004] | -0.009 [-0.020, 0.003] | -0.009 [-0.021, 0.003] | -0.011 [-0.023, 0.001] |

AUC inside each group:

| field | group | n | final blend | ablate_no_demographics | fair_full_reweigh | fair_unaware | fair_reweigh |
|---|---|---|---|---|---|---|---|
| race | White | 66107 | 0.921 [0.917, 0.924] | 0.917 [0.913, 0.921] | 0.917 [0.914, 0.921] | 0.891 [0.886, 0.895] | 0.891 [0.886, 0.895] |
| race | Race Not Available | 17872 | 0.923 [0.916, 0.929] | 0.919 [0.912, 0.925] | 0.918 [0.911, 0.925] | 0.890 [0.882, 0.897] | 0.890 [0.883, 0.898] |
| race | Black or African American | 8991 | 0.909 [0.901, 0.917] | 0.908 [0.900, 0.916] | 0.906 [0.897, 0.914] | 0.885 [0.875, 0.893] | 0.885 [0.875, 0.893] |
| race | Asian | 8009 | 0.882 [0.867, 0.896] | 0.880 [0.865, 0.894] | 0.878 [0.863, 0.892] | 0.845 [0.828, 0.860] | 0.844 [0.827, 0.860] |
| race | Joint | 2716 | 0.911 [0.889, 0.933] | 0.906 [0.883, 0.929] | 0.907 [0.885, 0.929] | 0.885 [0.861, 0.908] | 0.885 [0.862, 0.909] |
| race | American Indian or Alaska Native | 810 | 0.939 [0.917, 0.960] | 0.935 [0.911, 0.957] | 0.940 [0.917, 0.961] | 0.930 [0.906, 0.952] | 0.932 [0.907, 0.954] |
| ethnicity | Not Hispanic or Latino | 70469 | 0.920 [0.916, 0.924] | 0.916 [0.912, 0.919] | 0.915 [0.911, 0.918] | 0.890 [0.886, 0.894] | 0.889 [0.885, 0.893] |
| ethnicity | Ethnicity Not Available | 16232 | 0.924 [0.917, 0.931] | 0.921 [0.914, 0.927] | 0.920 [0.914, 0.926] | 0.891 [0.883, 0.898] | 0.891 [0.884, 0.898] |
| ethnicity | Hispanic or Latino | 15102 | 0.910 [0.902, 0.917] | 0.906 [0.899, 0.913] | 0.907 [0.899, 0.914] | 0.881 [0.873, 0.889] | 0.882 [0.873, 0.889] |
| ethnicity | Joint | 3170 | 0.921 [0.903, 0.937] | 0.914 [0.895, 0.931] | 0.918 [0.901, 0.934] | 0.891 [0.871, 0.910] | 0.891 [0.870, 0.909] |
| sex | Joint | 38731 | 0.922 [0.917, 0.927] | 0.916 [0.911, 0.922] | 0.916 [0.911, 0.922] | 0.894 [0.888, 0.900] | 0.893 [0.888, 0.899] |
| sex | Male | 34507 | 0.905 [0.900, 0.910] | 0.901 [0.896, 0.906] | 0.901 [0.896, 0.906] | 0.873 [0.867, 0.879] | 0.872 [0.866, 0.878] |
| sex | Female | 23462 | 0.933 [0.928, 0.938] | 0.930 [0.925, 0.935] | 0.928 [0.923, 0.933] | 0.904 [0.899, 0.910] | 0.904 [0.899, 0.909] |
| sex | Sex Not Available | 8300 | 0.913 [0.903, 0.923] | 0.908 [0.898, 0.919] | 0.907 [0.897, 0.918] | 0.869 [0.857, 0.882] | 0.870 [0.858, 0.882] |
| age | 25-34 | 30933 | 0.929 [0.923, 0.934] | 0.926 [0.920, 0.931] | 0.925 [0.919, 0.930] | 0.901 [0.895, 0.907] | 0.900 [0.894, 0.906] |
| age | 35-44 | 27055 | 0.915 [0.910, 0.921] | 0.911 [0.905, 0.917] | 0.910 [0.904, 0.916] | 0.882 [0.876, 0.889] | 0.882 [0.876, 0.889] |
| age | 45-54 | 17437 | 0.913 [0.905, 0.919] | 0.909 [0.901, 0.915] | 0.909 [0.901, 0.915] | 0.883 [0.874, 0.891] | 0.882 [0.873, 0.889] |
| age | 55-64 | 11993 | 0.920 [0.912, 0.928] | 0.917 [0.909, 0.924] | 0.916 [0.908, 0.923] | 0.887 [0.877, 0.896] | 0.886 [0.876, 0.895] |
| age | <25 | 6763 | 0.936 [0.927, 0.945] | 0.934 [0.924, 0.943] | 0.935 [0.926, 0.944] | 0.913 [0.903, 0.924] | 0.913 [0.903, 0.924] |
| age | 65-74 | 5903 | 0.914 [0.902, 0.925] | 0.910 [0.898, 0.921] | 0.908 [0.895, 0.919] | 0.883 [0.869, 0.896] | 0.880 [0.866, 0.894] |
| age | (blank) | 3088 | 0.887 [0.864, 0.907] | 0.882 [0.858, 0.904] | 0.880 [0.856, 0.901] | 0.876 [0.851, 0.897] | 0.876 [0.853, 0.898] |
| age | >74 | 1828 | 0.901 [0.874, 0.923] | 0.896 [0.870, 0.919] | 0.898 [0.872, 0.921] | 0.870 [0.839, 0.897] | 0.869 [0.839, 0.896] |

### Reading

The two fairness steps narrow the gap between Black and White applicants by about a quarter, for about 0.003 of AUC. The recorded decisions accept Black applicants 12.9 points less often than White applicants, and the competition model reproduces that gap exactly. Race-blind, the gap falls to 10.4 points on the Kaggle list (10.8 on the clean list); reweighed, to 9.8 (9.6). Most of the narrowing comes from removing the fields; reweighing adds 0.6 to 1.2 points.

The narrowing is clearest among applicants the lenders accepted. The TPR gap of Black applicants goes from -0.038 to -0.008 [-0.013, -0.002] on the Kaggle list and from -0.055 to -0.010 [-0.017, -0.005] on the clean list: an accepted Black applicant is now almost as likely as an accepted White applicant to be accepted by the model. The FPR gap of the reweighed Kaggle-list model, -0.008 [-0.027, 0.012], can no longer be told from 0.

It is bought with calibration. The race-blind models predict for Black applicants about 3 points more acceptances than lenders gave (+0.029 on the Kaggle list, +0.031 on the clean list), 3.5 to 4 points when reweighed (+0.035, +0.038), and slightly fewer for White applicants (-0.005 to -0.007). The competition model was within 0.005 for both. This is how the gap narrows: one cut-off applied to scores that are now higher than the recorded rate for Black applicants and lower for White applicants. Whether that is a cost or the point depends on which notion of fairness is chosen; the section on the choice of metric takes it up.

We thought that part of the gap left by the fair model comes through geography: the 8 geography columns (state, the area's median family income, the 6 census tract columns) identify Black applicants with AUC 0.796 on their own (proxy check below). If so, fair_reweigh without those 8 columns (config fair_reweigh_no_geo, 28 columns) should have a narrower accept gap. The rule, fixed before the run: the hypothesis holds if the accept gap narrows with a paired 95% bootstrap interval above 0. Each model is read at its own same-rate cut-off, and the bootstrap draws the same White and Black rows for both (1,000 resamples).

| Black minus White | fair_reweigh | fair_reweigh_no_geo | change [95%] |
|---|---|---|---|
| accept gap | -0.096 | -0.095 | +0.000 [-0.003, 0.003] |
| TPR gap | -0.010 | -0.011 | -0.001 [-0.004, 0.003] |
| FPR gap | -0.023 | -0.022 | +0.002 [-0.006, 0.009] |
| pred - obs, Black | +0.038 | +0.040 | +0.002 [0.001, 0.004] |
| pred - obs, White | -0.007 | -0.009 | -0.002 [-0.002, -0.001] |

The hypothesis is wrong. Without the geography columns the accept gap does not change (+0.0002 before rounding), nor do the TPR and FPR gaps; the calibration gap of Black applicants widens slightly, from +0.038 to +0.040. Removing them costs 0.0039 of CV AUC (0.88985 to 0.88599, 7.5 corrected SE, run 20261111T111505Z_04069927): geography helps predict the decision, but removing it leaves the gap where it was.

Removing the leaks while keeping the demographic fields widens the Black-White gap by 1.0 point [0.6, 1.5] and the TPR gap from -0.038 to -0.055. The leak-free fair model still ends with a gap no wider than the Kaggle-list one (9.6 against 9.8 points).

We thought the widening came from the 6,279 leak rows: there the Kaggle-list model reads the decision from the leak column, so race plays no part in its score, while the clean model must score those rows from the other columns, the demographic fields included. If so, the gap should barely change on the rows without a leak value and widen on the leak rows. The test reads the same out-of-fold predictions on each subset, with each model's same-rate cut-off set on all rows, and a paired bootstrap over rows (1,000 resamples).

| rows | White / Black rows | accept gap, Kaggle list | accept gap, clean list | change [95%] |
|---|---|---|---|---|
| all | 66,107 / 8,991 | -0.129 | -0.139 | -0.010 [-0.015, -0.006] |
| without a leak value | 62,128 / 8,465 | -0.126 | -0.140 | -0.014 [-0.018, -0.010] |
| leak rows only | 3,979 / 526 | -0.179 | -0.125 | +0.054 [0.013, 0.093] |

The hypothesis is wrong. The widening is on the rows without a leak value; on the leak rows the gap narrows. The two groups have the same share of leak rows (6.0% and 5.9%), so the mix of rows does not explain it either.

Other pairs barely move. The Hispanic gap narrows by 0.5 point at most and the Female gap not at all, and their TPR and FPR gaps stay where they were. Neither removing the sex and ethnicity fields nor reweighing on race changes these gaps. The age gaps are also unchanged within their intervals. The Female gap is worth one more remark: at the same-rate cut-off every model accepts women 4.7 to 4.8 points less often than men, against 4.0 points in the recorded decisions, while its mean prediction for women is within 0.002 of the observed rate. A calibrated score can still widen a gap once a single cut-off turns it into yes or no.

Ranking quality falls evenly. In every group, the race-blind Kaggle-list models are at most 0.007 below the competition blend. The clean-list models are 0.02 to 0.04 below it in most groups. The two reweighed models differ only by the 2 leak columns: from fair_full_reweigh to fair_reweigh, AUC falls by 0.021 to 0.037 in every group except American Indian or Alaska Native (0.008) and blank age (0.004). Asian applicants remain the group ranked least well (0.844 for fair_reweigh).

Limits. All of this is on the training rows, out-of-fold; the intervals cover the sampling of rows, not the training of the models, and are not corrected for the many comparisons (where there is no real gap, about 1 interval in 20 still excludes 0). Reweighing was run once, on race only, with no tuning. Nothing here says whether a narrower gap is closer to the right decisions: the target is what lenders did, and the data cannot tell a difference in risk from a difference in treatment.

## Proxy check: what a race-blind model can still see

Removing the 17 demographic fields makes a model blind to race only if the other columns do not carry it. For each race group with at least 5,000 rows, a LightGBM with the competition settings predicts "this applicant is in the group" against everyone else, out-of-fold on the frozen folds. It is fit twice: on the 38 columns of the race-blind Kaggle list, then on its 8 geography columns alone (state, the area's median family income, and six census tract columns, among them the tract's minority share). Code in "src/proxy_check.py", numbers in "reports/tuning/proxy_check.csv".

AUC here means: the chance that a random applicant of the group gets a higher score than a random applicant outside it. 0.5 = the columns say nothing about the group.

| group | rows | AUC from the 38 columns | AUC from the 8 geography columns |
|---|---|---|---|
| White | 66,107 | 0.740 (folds 0.735 to 0.743) | 0.695 (0.689 to 0.699) |
| Race Not Available | 17,872 | 0.672 (0.669 to 0.675) | 0.587 (0.581 to 0.592) |
| Black or African American | 8,991 | 0.840 (0.833 to 0.849) | 0.796 (0.790 to 0.800) |
| Asian | 8,009 | 0.847 (0.840 to 0.851) | 0.800 (0.792 to 0.805) |

To make 0.84 concrete: among the 10% of rows with the highest score from the 38 columns, 39% are Black applicants, against 9% in the whole data, and those 10% of rows hold 45% of all Black applicants. For Asian applicants the figures are 38% against 8%, and 50%.

Reading:

- The race-blind columns still tell a lot about race, most of all for Black and Asian applicants. A model without the demographic fields can still give different scores to these groups, through the columns that stand in for them.
- Geography carries most of it: 8 columns reach 0.796 of the 0.840 for Black applicants, and 0.800 of the 0.847 for Asian applicants. Where the home is says much about who buys it, and the tract's minority share says it by definition. The 30 other columns (loan, income, lender process) add about 0.04.
- "Race Not Available" depends much less on geography (0.587); the other columns lift it to 0.672.

What this does not say. It measures what the columns reveal about race, not how much the competition model relies on that, and not why. Geography also predicts the decision itself: a LightGBM on the geography and census columns alone reaches AUC 0.673 for accepted (run 20261031T132634Z_df45cba9). So the same columns are both a proxy for race and a predictor of the decision, and this check cannot separate the two. The model settings are those tuned for the accept / deny target, not retuned for this task.

## The demographics-only model

A LightGBM trained on the 17 demographic fields alone (race, ethnicity, sex and age of applicant and co-applicant, their derived summaries and the observed flags), with the tuned settings and the frozen folds, predicts the recorded decision with CV AUC 0.63546 (run 20261031T132746Z_e63bd31a, config expert_demographics). For comparison, the geography and census columns alone reach 0.673 and the finance columns alone 0.867. Its out-of-fold rank correlation with the tuned LightGBM is 0.364.

So declared race, ethnicity, sex and age order the recorded decisions clearly better than chance (0.5), with no information on the loan, the income or the property. This is an association with the lenders' decisions, not a measure of their merit, and it does not say how the competition model uses these fields.

Decision: this model does not enter any blend. Two reasons:

- It was never a candidate. It was run as a fairness diagnostic, and the config says so; it was not entered in the blend selection, so its blend gain was not tested. The final blend was chosen without it.
- A component that scores applicants on race, ethnicity, sex and age alone has no place in a model we judge on fairness. The competition model does keep these fields among its 55 columns, to describe what lenders did; a separate component built only on them would add weight on group membership alone, which is the opposite of what the fair models do.

## Which fairness criterion, and why

### Three criteria

With accepted as the positive class and a single cut-off for everyone, there are three common ways to ask whether a model treats two groups alike (in the vocabulary of Barocas, Hardt and Narayanan: independence, separation, sufficiency):

- Equal accept rates (demographic parity): the model accepts the same share of each group.
- Equal error rates (equalized odds): among applicants the lenders accepted, the model accepts the same share in each group (TPR); among applicants the lenders denied, too (FPR). Equal opportunity asks for the TPR only (Hardt, Price and Srebro, 2016).
- Calibration within groups: a given score means the same accept rate in every group, so that a score of 0.9 is accepted 9 times in 10 whoever the applicant is.

### Here they cannot all hold

The recorded decisions accept 86.8% of White applicants and 73.9% of Black applicants. When base rates differ and the model is not perfect, the three criteria conflict. Two facts make it concrete.

First, at any cut-off and for any group, FPR = p / (1 - p) x (1 - PPV) / PPV x TPR, where p is the group's recorded accept rate and PPV the share of the model's acceptances that lenders also accepted (Chouldechova, 2017). It is an identity, and it holds exactly for every model and group below. With our two base rates, equal TPR and equal PPV would force the FPR of Black applicants to 0.43 times that of White applicants. Equal error rates and equal PPV, which is the cut-off form of calibration, cannot hold together (Kleinberg, Mullainathan and Raghavan, 2016, show the same for calibrated scores in general).

Second, the model's accept rate in a group is p x TPR + (1 - p) x FPR. If Black applicants had the same TPR and FPR as White applicants, the model would still accept 8 to 9 points fewer of them (0.129 x (TPR - FPR), with the White rates of the models below). Equal error rates and equal accept rates cannot hold together either.

So a choice has to be made. Any model we build sits somewhere between these criteria, and the tables below show where ours sit.

### Where our models sit

At each model's same-rate cut-off (code in the last part of "notebooks/04_fairness.ipynb"):

| model | group | recorded accept rate | mean prediction | model accepts | TPR | FPR | PPV |
|---|---|---|---|---|---|---|---|
| final blend | White | 0.868 | 0.869 | 0.868 | 0.959 | 0.264 | 0.960 |
| final blend | Black or African American | 0.739 | 0.744 | 0.739 | 0.922 | 0.220 | 0.922 |
| fair_full_reweigh | White | 0.868 | 0.862 | 0.862 | 0.954 | 0.258 | 0.960 |
| fair_full_reweigh | Black or African American | 0.739 | 0.774 | 0.765 | 0.947 | 0.250 | 0.914 |
| fair_reweigh | White | 0.868 | 0.861 | 0.861 | 0.945 | 0.311 | 0.952 |
| fair_reweigh | Black or African American | 0.739 | 0.776 | 0.766 | 0.935 | 0.288 | 0.902 |

Calibration inside score bands (the five quintiles of each model's score over all rows): mean prediction against the recorded accept rate, by group.

| model | score band | White n | White: mean prediction / recorded | Black n | Black: mean prediction / recorded |
|---|---|---|---|---|---|
| final blend | 1: 0.00 to 0.88 | 11479 | 0.415 / 0.402 | 3379 | 0.425 / 0.408 |
| final blend | 2: 0.88 to 0.95 | 10975 | 0.924 / 0.922 | 3671 | 0.921 / 0.923 |
| final blend | 3: 0.95 to 0.97 | 12654 | 0.959 / 0.960 | 1438 | 0.956 / 0.958 |
| final blend | 4: 0.97 to 0.98 | 14079 | 0.974 / 0.977 | 185 | 0.971 / 0.978 |
| final blend | 5: 0.98 to 1.00 | 16920 | 0.986 / 0.990 | 318 | 0.996 / 0.997 |
| fair_reweigh | 1: 0.00 to 0.87 | 12409 | 0.469 / 0.477 | 2680 | 0.388 / 0.331 |
| fair_reweigh | 2: 0.87 to 0.94 | 12890 | 0.911 / 0.919 | 2257 | 0.911 / 0.866 |
| fair_reweigh | 3: 0.94 to 0.96 | 13037 | 0.947 / 0.956 | 1914 | 0.947 / 0.920 |
| fair_reweigh | 4: 0.96 to 0.97 | 13388 | 0.964 / 0.970 | 1372 | 0.964 / 0.947 |
| fair_reweigh | 5: 0.97 to 0.99 | 14383 | 0.978 / 0.985 | 768 | 0.977 / 0.962 |

The competition model is calibrated: in every band, its mean prediction is within 0.017 of the recorded rate, for both groups. A score means the same thing whoever the applicant is. The price is that every other gap stays: it accepts Black applicants 12.9 points less often, with a TPR lower by 0.037, an FPR lower by 0.044 and a PPV lower by 0.038.

The fair model moves toward equal error rates. Its TPR gap is -0.010 and its FPR gap -0.023, and it accepts Black applicants 9.6 points less often, not far above the 8.2 points that fully equal error rates would still leave for this model. The price is calibration:

- For Black applicants its mean prediction is 0.776, against 0.739 in the recorded decisions: +0.038. For White applicants it is 0.861 against 0.868. The race-blind model without reweighing is at +0.031 (Kaggle list +0.029).
- Inside the score bands, the recorded rate of Black applicants is below the model's prediction in every band, by 0.015 to 0.057, while for White applicants it is 0.006 to 0.009 above. At a score between 0.87 and 0.94, lenders accepted 91.9% of the White applicants and 86.6% of the Black applicants the model scored there (binomial standard error about 0.007 for the Black rate). The fair model gives both the same score; the recorded decisions do not treat them the same.
- The PPV gap widens, from -0.038 to -0.051: among the applicants the fair model accepts, a smaller share of Black applicants were accepted by the lenders. This is the identity at work. With the base rates unchanged, narrowing the TPR gap moves the difference into PPV.

### Our choice

For the fair model we take equal opportunity as the main criterion: the TPR gap between groups at the same-rate cut-off, with its interval. We also report the FPR gap and the accept-rate gap, and we report the calibration gap as the price. For the competition model the right criterion is calibration. The two models serve different purposes, so they are judged differently.

Why not calibration for the fair model. The label is what a lender decided, not whether the applicant repaid. A model calibrated to it reproduces each group's recorded accept rate, including whatever part of the 12.9-point gap does not come from risk. Holding the fair model to calibration would hold it to the very decisions under review. Calibration is the right test for the competition model: it is what makes that model's group tables a faithful description of what lenders did. It does not tell us whether what they did was fair.

Why the TPR rather than the FPR. 1 - TPR is the share of applicants the lenders accepted that the model would turn down: the error that costs an applicant a loan. Equal opportunity asks that this error not depend on the group. The FPR concerns applicants the lenders denied whom the model would accept; making it equal is a question of the lender's risk more than of the applicant's harm.

Why not equal accept rates as the main criterion. The data has no credit score, no interest rate and no repayment, so it cannot say how much of the gap between groups is a difference in risk and how much a difference in treatment. Equal accept rates would remove the whole gap, which amounts to assuming that none of it is risk. Equal opportunity does not need that split: it only asks that applicants the lenders accepted be accepted alike.

What this choice does not escape. Our TPR is measured against the lenders' decisions: "accepted by the lender" stands in for "should get the loan". If the lenders' acceptances themselves treated groups unequally, an equal TPR relative to them carries that over. Every criterion that can be computed on this data has the same limit, because all of them are defined against the recorded decision. The TPR also depends on the cut-off: the competition has none, and at 0.50 the gaps differ from those at the same-rate cut-off. Finally, equal opportunity here is measured on race, and the fair model was reweighed on race only; the ethnicity, sex and age gaps were measured but not targeted.

### The "observed" fields

When an applicant applying in person does not give their race, ethnicity or sex, the lender must record them from visual observation or surname. The fields "applicant_race_observed", "applicant_ethnicity_observed" and "applicant_sex_observed" say whether that happened (1 = recorded on that basis, 2 = not, 3 = not applicable). For a group audit this matters: on those rows the group is the lender's perception, not the applicant's answer.

| field | value | rows | accepted |
|---|---|---|---|
| applicant_race_observed | 1 | 1320 | 0.855 |
| applicant_race_observed | 2 | 97421 | 0.849 |
| applicant_race_observed | 3 | 6259 | 0.863 |
| applicant_ethnicity_observed | 1 | 1305 | 0.856 |
| applicant_ethnicity_observed | 2 | 97455 | 0.849 |
| applicant_ethnicity_observed | 3 | 6240 | 0.864 |
| applicant_sex_observed | 1 | 1306 | 0.861 |
| applicant_sex_observed | 2 | 97401 | 0.849 |
| applicant_sex_observed | 3 | 6293 | 0.863 |

Race was recorded by observation for 1,320 rows (1.3%): 122 of the 8,991 Black applicants (1.4%) and 1,091 of the 66,107 White applicants (1.7%). That is too few to change any group table. The accept rate is 0.849 to 0.864 whatever the value. The three fields are among the 17 demographic fields that the race-blind models leave out.

## What deployment would require

A model put to use would score applications before any decision exists. So it would need the clean list by construction, a target closer to what a lender should predict than past approvals (repayment, where it can be observed), and validation on applications later than those it was trained on, which a challenge with no stated year and exchangeable train and test rows cannot provide. It would also need the group checks of this report repeated on the lender's own applicants and over time, and a legal review under fair-lending law (the Equal Credit Opportunity Act and the Fair Housing Act), including whether race may be used at all to set training weights, as reweighing does, and how to give each denied applicant the specific reasons the law requires, which a score alone does not provide.
