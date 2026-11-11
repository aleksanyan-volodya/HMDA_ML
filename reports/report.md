# Predicting mortgage application outcomes from HMDA records

Draft of 2026-11-09. Runs are cited by the last 8 characters of their run_id (for example run 1ff742f2); the full id, config, commit, per-fold AUCs and prediction files of every run are in reports/runs.csv. Tables are T1 to T19, figures F1 to F9; figures are in reports/figs/ and are rebuilt by "make figures" (reports/figs/make_figures.py), which also pastes the generated tables into this file. A line in square brackets marks a part still to write.

## 1. Summary

[draft of 2026-11-11; the reproduction sentence waits for the CatBoost test]

We predict whether a home purchase application in the HMDA records was originated or denied, from 56 columns that include no credit score, scored by ROC AUC (sections 2 and 3). Every choice was made on one frozen 5-fold split, a change was adopted only if its paired gain reached 2 corrected standard errors, and each finalist was re-checked on three fresh splits (section 5). The final model, a log-odds blend of LightGBM, CatBoost, ExtraTrees and XGBoost, reaches a CV AUC of 0.9205 (pooled out-of-fold 0.9204, 95% bootstrap interval [0.9176, 0.9232]) and 0.9216 on the public leaderboard (T1). Two columns are written because of the decision, and removing them costs 0.026 AUC: the competition score overstates what can be predicted when an application arrives (section 4, T9). Its LightGBM member leans most on the debt-to-income ratio (F8). The blend is calibrated in every slice examined, and the three features suggested by its worst errors added nothing (section 8, T11). The competition model reproduces the recorded gap in acceptance between Black and White applicants (12.9 points, T12, T16). A fair model built without the leak columns and the demographic fields, and reweighed by race, narrows it to 9.6 points and the gap in true positive rates from -0.037 to -0.010, at the price of calibration; removing the demographic fields and reweighing cost about 0.003 AUC, and the remaining columns still identify Black applicants at AUC 0.84 (section 9, T15, T13). [pending: one sentence on reproduction, after the CatBoost test: the logged CatBoost member did not rebuild exactly from a clean commit (section 10.2)]

T1. Headline numbers.

| quantity | value | source |
|---|---|---|
| final model | LightGBM 2/6, CatBoost 2/6, ExtraTrees 1/6, XGBoost 1/6, weighted mean of log-odds | run 1ff742f2 |
| CV AUC, mean of 5 folds | 0.92054 | run 1ff742f2 |
| pooled out-of-fold AUC, 95% stratified bootstrap CI (2,000 resamples) | 0.92043 [0.91764, 0.92318] | reports/tuning/ci_final.log |
| gain over the LightGBM + CatBoost 50/50 blend on 3 fresh 5-fold splits | +0.00070, 7.6 corrected SE, 15 of 15 folds | reports/tuning/fresh_fresh_candidate.csv |
| public leaderboard (30% of test) | 0.92158 | submission 6, reports/submissions.md |
| best single model, tuned LightGBM | 0.91905 | run 406e4210 |
| same LightGBM without the 2 columns that record the decision | 0.89305 | run 116f6267 |

## 2. Task, data and framing

### 2.1 Task

Each row is one mortgage application from the Home Mortgage Disclosure Act (HMDA) records; the target "accepted" is 1 if the loan was granted. Train has 105,000 labelled rows, test 45,000, with 56 input columns (T2). The metric is ROC AUC on test; the public leaderboard scores 30% of it and the final ranking uses the other 70% (reports/submissions.md). External data is not allowed, so every model uses these columns only.

T2. Data facts.

<!-- begin t2_data_facts -->
| fact | value |
|---|---|
| train rows, test rows, input columns | 105,000, 45,000, 56 |
| accepted share in train | 0.850 (89,244 accepted, 15,756 denied) |
| train ids / test ids | 100,000 to 599,996 / 600,005 to 999,992, shared ids 0 |
| AUC of the row id alone (no order leak if ~0.5) | 0.4957 |
| train rows with code 1111 (Exempt): in any coded column / in all affected columns | 3,016 / 2,942 (14 columns), accepted 0.914 |
| loan amount and property value on the $10k midpoint grid (v mod 10,000 = 5,000), both files | 1.0000 of non-blank values |
| location unknown (census block all 0): train / test rows, train accept rate | 2,117 / 929, 0.171 |
| derived_msa-md = 99999 (outside any MSA): train rows, accept rate | 15,200, 0.751 |
| county levels in test but not train, test rows affected | 84, 115 (0.26%) |
| exact duplicate rows in train over the 56 inputs | 7 |
<!-- end t2_data_facts -->

### 2.2 What the target is

The target is a lender's recorded decision, not creditworthiness and not default. It says what a lender did with an application, not whether the applicant could have repaid, and the data holds no repayment history. A model that scores well here predicts past lending decisions; if some of those decisions treated a group worse than its risk justified, a faithful model carries that on, and a high AUC is no defence. Three consequences hold for the whole report:

- An error of the model is a disagreement with the lender, not a wrong judgement of the applicant.
- A gap between groups is an association. The data has no credit score, no interest rate and no applicant history, so it cannot separate a difference in risk from a difference in treatment (section 9).
- The rows are a selection. The challenge page (read 2026-11-09, quoted in reports/fairness.md) states the scope: rows of the HMDA Modified LAR published by the CFPB / FFIEC, home purchase loans only, and only applications that were originated or denied; withdrawn, incomplete and other files are excluded. So "accepted" is originated against denied, which also fits the two leak columns of section 4. Applicants who withdrew or never completed their file are outside the data, and we cannot see who they were or whether they differ by group.

## 3. Data: codes, sentinels, missingness

The column meanings come from the HMDA filing guide and Regulation C commentary, archived as extracts in reports/refs/hmda_dictionary.md (the collection year is not stated in the data and stays unknown). Four facts shape everything after:

- There is no credit score. "applicant_credit_score_type" names the scoring model used (Equifax Beacon 5.0, FICO 9, VantageScore 4.0, ...), not the score.
- Code 1111 means "Exempt": the lender was partly exempt from reporting. It appears in all 14 affected columns of the same rows at once (T2), so it is a fact about the row, not 14 separate facts. It is kept as its own level.
- Some values are codes, not quantities. Loan amount and property value are exact $10,000 midpoints (T2); income is in thousands; the reported LTV is the combined LTV over all liens; the census block is 0 on the rows whose location is unknown (T2); "derived_msa-md" = 99999 marks areas outside any metropolitan area and is kept as a level. The cleaning rules, each with its reason, are in "src/features.py" and reports/feature_plan.md (block 0).
- Missingness predicts the outcome. A missing LTV, property value, state or county goes with a much lower accept rate (T3, F1). Trees keep NaN as its own branch; section 4 asks whether these gaps are written because of the decision.

T3. Missingness per column, train and test side by side, with the accept rate when missing and when present.

<!-- begin t3_missingness -->
| column | train blank % | test blank % | gap, points | blank rows (train) | accept if blank | accept if present |
|---|---|---|---|---|---|---|
| co-applicant_age | 55.90 | 55.84 | -0.05 | 58,693 | 0.828 | 0.877 |
| loan_to_value_ratio | 11.31 | 11.29 | -0.02 | 11,874 | 0.538 | 0.890 |
| debt_to_income_ratio | 7.31 | 7.37 | 0.06 | 7,679 | 0.822 | 0.852 |
| property_value | 4.53 | 4.42 | -0.11 | 4,756 | 0.698 | 0.857 |
| income | 4.36 | 4.49 | 0.13 | 4,578 | 0.859 | 0.850 |
| loan_term | 3.15 | 3.10 | -0.05 | 3,307 | 0.872 | 0.849 |
| applicant_age | 2.94 | 3.01 | 0.07 | 3,088 | 0.875 | 0.849 |
| county_code | 1.28 | 1.32 | 0.04 | 1,339 | 0.246 | 0.858 |
| state_code | 0.78 | 0.82 | 0.03 | 824 | 0.100 | 0.856 |
| total_units | 0.42 | 0.38 | -0.04 | 437 | 0.913 | 0.850 |
| conforming_loan_limit | 0.42 | 0.38 | -0.04 | 437 | 0.913 | 0.850 |
| applicant_ethnicity-1 | 0.03 | 0.02 | -0.01 | 27 | 0.556 | 0.850 |
| applicant_race-1 | 0.02 | 0.02 | -0.00 | 21 | 0.524 | 0.850 |
| co-applicant_ethnicity-1 | 0.01 | 0.01 | -0.00 | 9 | 0.667 | 0.850 |
| co-applicant_race-1 | 0.01 | 0.00 | -0.00 | 6 | 0.833 | 0.850 |

41 of the 56 columns have no blank in train. Largest train-test gap: 0.13 points. Blank = empty in the file; codes such as 1111 and census zeros are values here (T2).
<!-- end t3_missingness -->

F1. Accept rate when missing vs present, columns with more than 0.5% missing. [reports/figs/f1_missing_accept.png]

F2. Accept rate by state, with row counts and 95% Wilson intervals; states under 1,000 rows in grey, the 824 rows with a blank state in F1. [reports/figs/f2_state_accept.png]

Train and test look like draws from the same pool: a LightGBM trained to tell them apart reaches AUC 0.5015 (reports/models.md, table "Public LB gap"), and the row id carries no signal (T2).

## 4. Leakage

A column leaks when its value is written because of the decision, not before it. Such a column is still in the test set, so it helps the leaderboard, but it must stay out of a model meant to judge applications. Each verdict rests on the reporting rule for that column (Regulation C official commentary to 12 CFR 1003.4, cited in reports/leakage.md); a scan of all 1,077 levels with at least 50 rows for accept rates below 2% or above 98% found the two leaks and 5 small areas whose rates are chance (reports/leakage.md).

T4. Leakage verdicts and the three feature lists.

| column | what we see (train) | reporting rule | leak? | full | clean | strict |
|---|---|---|---|---|---|---|
| preapproval = 1 | 4,713 rows, all accepted | a denied application is reported as "not requested" | yes | keep | drop | drop |
| initially_payable_to_institution = 3 | 1,566 rows, none accepted | "not applicable" only when the file closed without a decision on it | yes | keep | drop | drop |
| location unknown (census block empty) | accepted 0.17; manufactured home + unknown 1,507 rows, 0.015 | no location until a site is chosen; a closed loan has one | in part | keep | keep | drop |
| LTV, property value or DTI missing | missing LTV accepted 0.54 | "not applicable" when the decision did not rely on the number | in part | keep | keep | drop |
| aus-1 = 6 (no automated system) | accepted 0.61 | describes how the lender works, set before the decision | no | keep | keep | drop |
| submission_of_application | every level accepted 0.83 to 0.91 | | no | keep | keep | keep |

- full (55 columns) is the competition model; clean (53) removes the 2 leaks and is the base of the fair model; strict (38) also removes the partial signals together with the columns they live in (aus-1, LTV, property value, DTI and their ratios, census, state). Strict is a sensitivity check, not a model: it removes real information along with the suspect signal.
- The partial signals stay in clean because they also carry real information: manufactured homes sold without land often have no combined LTV whether approved or not (reports/leakage.md, verdict 5), and a tree cannot drop a missing value without dropping the column for every row.
- Cost of each list for the tuned LightGBM: removing the 2 leaks costs 0.026 AUC, and the strict list a further 0.070 (T9, rows A9 and A10). Location unknown on manufactured homes is the main remaining doubt: it is near-deterministic (22 approvals in 1,507 rows) and stays in clean only because it cannot be separated from the census columns.

## 5. Metric and validation

### 5.1 The metric

AUC is the probability that a random accepted row scores above a random denied one, so it depends only on the order of the scores. Three consequences decide what is not worth doing. A strictly increasing transform of the scores leaves AUC exactly unchanged (T5), so calibration and thresholds cannot raise it. AUC does not depend on the base rate, so the 85/15 imbalance needs no resampling. And a blend may average log-odds, ranks or probabilities: these give different orders, so the choice is measured (section 6.4).

T5. AUC, log loss and Brier score of the final out-of-fold predictions under five strictly increasing transforms.

<!-- begin t5_transforms -->
| scores | AUC | log loss | Brier |
|---|---|---|---|
| as predicted, p | 0.920434 | 0.1949 | 0.0524 |
| p squared | 0.920434 | 0.2168 | 0.0578 |
| square root of p | 0.920434 | 0.2188 | 0.0590 |
| rank / (n + 1) | 0.920434 | 0.6347 | 0.2261 |
| sharpened, sigmoid(2 logit p) | 0.920434 | 0.2620 | 0.0555 |

Out-of-fold predictions of the final blend (run 1ff742f2). Every transform is strictly increasing and keeps scores in (0, 1), so log loss and Brier are defined for all of them.
<!-- end t5_transforms -->

### 5.2 Split

Every run uses the same 5 folds: StratifiedKFold(5, shuffle=True, random_state=42), built once into data/processed/folds.parquet (content hash 3762a22f...) and refused by the code if the seed changes (reports/validation.md). Grouped splits were rejected: grouping duplicates would affect only 7 duplicated rows, and grouping by county would simulate unseen counties when only 115 test rows (0.26%) are in a county absent from train (T2). A grouped split would measure a harder task than the one scored. The shuffled split assumes train and test are exchangeable, which the adversarial check of section 3 supports.

### 5.3 Uncertainty and the decision rule

The five fold models share 3/4 of their training rows, so the naive SE (std of fold AUCs / sqrt(5)) is too small. A change (a feature block, a setting, a model in a blend) is adopted if and only if its paired mean gain over the 5 folds is positive and at least 2 x the Nadeau-Bengio corrected SE, std of the per-fold gains x sqrt(1/k + 1/(k-1)), i.e. 1.5 x the naive SE at k = 5 (reports/validation.md). The rule was fixed before the first comparison and never changed. The formula was derived for repeated random splits, so for k-fold it is a heuristic. A paired stratified bootstrap CI on the pooled out-of-fold predictions is printed with every comparison; it covers test-row noise only.

Because a candidate chosen on the frozen folds is flattered by the choice, each finalist was re-checked on 3 fresh 5-fold splits (seeds 1 to 3, 15 paired folds) never used for any decision. Three choices were confirmed this way: the tuned LightGBM, +0.00194 at 4.2 SE (reports/tuning.md); the LightGBM + CatBoost blend, +0.00083 at 3.8 SE, 14 of 15 folds; the final 4-model blend, +0.00070 over it at 7.6 SE, 15 of 15 (reports/models.md).

### 5.4 Cross-validation against the leaderboard

The public leaderboard was used only to check that CV transfers, never to choose. Over the 5 model submissions the public score was above CV by +0.0010 to +0.0025, and every CV gain moved the leaderboard the same way (F3, reports/submissions.md). On 1,000 draws of 13,500 out-of-fold rows of the final blend, one public score has an SD of 0.0036 (T6). A paired gap between two models is less noisy because both are scored on the same rows: 0.0006 between the final blend and the tuned LightGBM (T6), 0.0014 between the 50/50 blend and ExtraTrees (reports/models.md). The public leaderboard cannot resolve the gains of the last two submissions (+0.0004 each), so the final pair was chosen on CV and fresh splits (reports/submissions.md).

F3. CV against public leaderboard for the 5 model submissions, with the line LB = CV. [reports/figs/f3_cv_lb.png]

T6. Leaderboard noise at 20, 30, 50 and 100% of test, simulated on the final out-of-fold predictions.

<!-- begin t6_lb_noise -->
| share of test | rows | SD of one score | SD of a paired gap (final blend - LightGBM) |
|---|---|---|---|
| 20% | 9,000 | 0.0047 | 0.0007 |
| 30% | 13,500 | 0.0036 | 0.0006 |
| 50% | 22,500 | 0.0026 | 0.0004 |
| 100% | 45,000 | 0.0016 | 0.0003 |

1,000 draws without replacement from the 105,000 out-of-fold rows per share. The public leaderboard is the 30% row; the private ranking uses 70%.
<!-- end t6_lb_noise -->

## 6. Features and models

### 6.1 Features

Each feature block was added to the current set, the LightGBM CV re-run on the frozen folds, and the block kept under the rule of section 5.3; a failed block was deleted, not kept in reserve.

T7. Feature blocks, LightGBM, each against the last adopted run (reports/feature_plan.md, reports/models.md).

| block | run | CV AUC | gain | gain / corrected SE | verdict |
|---|---|---|---|---|---|
| baseline: raw typed columns, starting settings | 4626a824 | 0.90351 | | | reference |
| 0: sentinel codes to NaN or levels, DTI and age as ranks, duplicated columns dropped | ac79d7b0 | 0.90446 | +0.00095 | 3.8 | adopted |
| 1: row-state flags (exempt count, entity applicant, location unknown, co-applicant, income <= 0) | 409b82cd | 0.90437 | -0.00009 | -0.2 | rejected |
| 2: ratios (computed LTV, reported - computed LTV, property / income) | 03f5980a | 0.90790 | +0.00344 | 4.8 | adopted |
| drop the raw county and MSA ids (2,898 and 418 levels) | f7f2dc68 | 0.91695 | +0.00904 | 16.3 | adopted |
| county accept rate, out-of-fold, shrunk to the state (m = 20) | 1114e40c | 0.91723 | +0.00028 | 0.6 | rejected |
| 3: segment flags (manufactured x loan type, DTI above the program limit, ...) | c9a209af | 0.91712 | +0.00018 | 0.8 | rejected |
| census tract accept rate, shrunk to the county (on the tuned model) | 561f4e1b | 0.91841 | -0.00064 | -1.4 | rejected |

The largest gain came from removing columns. We thought the raw county and MSA ids (2,898 and 418 levels) made LightGBM overfit, since its tree search stopped at 60 to 79 trees. Removing them gained +0.00904 (16.3 SE) and the tree search then ran to 121 to 192 trees (reports/feature_plan.md): the hypothesis held. The flag blocks (1 and 3) and the county and tract accept rates did not pass (T7).

### 6.2 Models

T8. Models on the frozen folds; gains are paired against the tuned LightGBM (reports/models.md, reports/tuning.md).

| model | run | CV AUC | vs tuned LightGBM | rank correlation with LightGBM |
|---|---|---|---|---|
| LightGBM, starting settings | f7f2dc68 | 0.91695 | -0.00210 (-3.7 SE) | |
| LightGBM, tuned | 406e4210 | 0.91905 | reference | 1 |
| CatBoost, untuned, with raw county and MSA ids | d96c8be4 | 0.91859 | -0.00046 (-0.5 SE) | 0.947 |
| XGBoost, tuned | 05b22e57 | 0.91848 | -0.00057 (-1.0 SE) | 0.971 |
| ExtraTrees, ranked numerics, tuned | 7bc88a0b | 0.91484 | -0.00421 (-5.7 SE) | 0.910 |
| MLP with embeddings, final | 752fd022 | 0.91003 | -0.00903 (-12.8 SE) | 0.910 |
| logistic regression, L1, splines and crossings | 6c8c7142 | 0.90379 | -0.01526 (-18.9 SE) | 0.89 |
| LightGBM + CatBoost, 50/50 log-odds | 918eb62b | 0.92014 | +0.00108 (2.2 SE) | |
| final blend, 2/6, 2/6, 1/6, 1/6 log-odds | 1ff742f2 | 0.92054 | +0.00149 (3.1 SE) | |

- The boosted models are level with each other: CatBoost untuned and XGBoost tuned are within 1 SE of the tuned LightGBM.
- The raw county id helps CatBoost (+0.00103, 3.6 SE over the same model without it, run 6deb776a), while removing it raised LightGBM by +0.00904 (T7) and the MLP by +0.00841 (reports/models.md).
- The MLP needed the ids removed and strong regularisation before it beat the linear model; it ends 0.009 below LightGBM after 10 runs (reports/models.md).
- LightGBM restricted to one block of columns scores 0.867 (finance), 0.829 (process and loan terms) and 0.673 (geography and census); these models are decorrelated (rank correlation 0.40 to 0.73), and none raised a blend under the rule (best +0.00004, 0.9 SE; reports/models.md, expert models).
- F4 and F5 show the ranking curves and how alike the models rank rows.

F4. ROC curves on out-of-fold predictions: final blend, tuned LightGBM, CatBoost, logistic regression. [reports/figs/f4_roc.png]

F5. Spearman rank correlation between the out-of-fold predictions of the main models. [reports/figs/f5_rank_corr.png]

### 6.3 Tuning

- LightGBM: a one-setting-at-a-time screening of 27 variants found what mattered (leaf size, number of leaves, column sampling, one-hot of categories up to 64 levels), then 100 Optuna trials searched them jointly: +0.00210, 3.7 SE over the starting settings, +0.00194 on fresh splits, so choosing the best of 100 trials flattered it by about 0.0002 (reports/tuning.md). A lower learning rate (0.02) added nothing (+0.00035, 0.8 SE, run 49275099).
- Seed 0, the seed of the search, is the best of 5 seeds (0.91905 against 0.91837 to 0.91878), 0.0003 above their mean (reports/models.md).
- XGBoost: 42 trials, +0.00199, 7.1 SE; best depth 11 against 6 at the start, and the trials at depth 6 peaked at 0.91742. ExtraTrees: 47 trials, +0.01299, 41 SE; more columns per split and a depth limit of 20 (reports/tuning.md).
- CatBoost: a full CV took about 2 hours, so the search scored each trial on 2 cheap fits. Its winner lost on the full CV (-0.00022, run 9ef0b007): the cheap trials did not rank settings as the full CV does (+0.0025 in the trials, +0.0003 in the full CV on the same two folds; reports/tuning.md). The untuned CatBoost stays.

### 6.4 Blending

All blends average log-odds with fixed weights; log-odds and ranks tied for 2 models (+0.00004, 1.1 SE), and log-odds keeps the per-model SHAP values additive (reports/models.md). Weights were chosen four ways, each scored nested (the choice redone without the scored fold):

- forward selection at equal weights, gated by the rule at each step: keeps the 50/50 LightGBM + CatBoost blend;
- non-negative least squares: +0.00012, 0.5 SE, does not pass;
- stacking with a logistic regression on 9 models: +0.00031, 1.2 SE, does not pass, and gives negative weights to correlated models;
- hill climbing on the fold AUC: LightGBM 2/6, CatBoost 2/6, ExtraTrees 1/6, XGBoost 1/6 in 4 of 5 nested climbs, +0.00036, 4.0 SE over the 50/50 blend. Confirmed on fresh splits (+0.00070, 7.6 SE, 15 of 15), so it is the final model, run 1ff742f2.

The MLP, the logistic regressions and the block models never entered a blend: each nested gain stayed under 2 SE (reports/models.md).

## 7. Ablations

Each ablation removes one thing from the tuned LightGBM (run 406e4210) and keeps its settings, same folds, same seed, compared with the rule of section 5.3. The settings were tuned on the full list and not re-tuned for any ablation. The predictions of A1 to A7 were written on 2026-10-18, before any run marked "to run", and are copied here unchanged; parts measured before that date are labelled as such and are not predictions. A8 to A10 were added later without a prediction.

T9. Pre-registered ablations.

| # | ablation | predicted gain vs full | observed | gain / corrected SE | run |
|---|---|---|---|---|---|
| A1 | no demographics: drop the 17 race, ethnicity, sex, "observed" and age fields | -0.002, range [-0.004, 0] | -0.00270, 0 of 5 folds up: inside the range | -5.2 | 20a010b4 |
| A2 | no process fields: drop preapproval, submission_of_application, initially_payable_to_institution, aus-1 | -0.015, range [-0.025, -0.008] | -0.03384, 0 of 5 folds up: outside, a larger loss | -15.1 | bc281e8f |
| A3 | no census: drop the 6 tract columns and the MSA median income | -0.004, range [-0.008, -0.002] | -0.00132, 1 of 5 folds up: outside, a smaller loss | -2.0 | 29f2ea4d |
| A4a | no target encoding, LightGBM | measured before registration | the final LightGBM uses none; adding the county rate gave +0.00028 | 0.6 | 1114e40c |
| A4b | no target encoding, CatBoost (raw county and MSA ids removed) | measured before registration | -0.00103 | -3.6 | 6deb776a vs d96c8be4 |
| A4c | no target encoding, L1 logistic regression (county rate removed) | -0.002, range [-0.005, 0] | -0.00008, 2 of 5 folds up: inside the range | -0.9 | 1dc8e77b vs 6c8c7142 |
| A5 | raw columns only: the 56 typed columns, no cleaning or ratios, county and MSA ids kept | -0.014, range [-0.020, -0.010] | -0.01207, 0 of 5 folds up: inside the range | -8.8 | fcc2e23f |
| A6a | single model vs blend, frozen folds | measured before registration (50/50 blend) | +0.00108 (50/50), +0.00149 (final blend) | 2.2, 3.1 | 918eb62b, 1ff742f2 |
| A6b | 50/50 blend vs LightGBM on 3 fresh splits | +0.0010, range [+0.0005, +0.0015] | +0.00083, 14 of 15 folds | 3.8 | reports/models.md |
| A7 | tuned vs starting settings | measured before registration | +0.00210 (frozen), +0.00194 (fresh) | 3.7, 4.2 | 406e4210 vs f7f2dc68 |
| A8 | full list (55 columns) | reference | 0.91905 | | 406e4210 |
| A9 | clean: the 2 leaks removed | not predicted | -0.02600, 0 of 5 folds up | -13.6 | 116f6267 |
| A10 | strict: also the partial leaks and their columns (vs clean) | not predicted | -0.06962, 0 of 5 folds up | -62.8 | cd2c2a46 |

Of the 6 open predictions, 4 fell inside their range (A1, A4c, A5, A6b) and 2 outside (A2, A3).

- A1: removing the 17 demographic fields costs 0.0027 (bootstrap CI [-0.0035, -0.0017]), inside the predicted range. Shuffling the same fields in the trained model costs more, 0.009 (section 8.2): shuffling and retraining measure different things.
- A2 missed by a wide margin: -0.034 against at most -0.025. The range was set before the leak verdicts, and 2 of the 4 fields turned out to be the confirmed leaks (A9: -0.026). We thought the other two (aus-1, submission_of_application) added the rest. Test: removing them from the clean list costs -0.00784 (-15.4 SE, paired bootstrap CI [-0.0092, -0.0065], run bc281e8f against 116f6267). It held: 0.026 + 0.008 is the 0.034 of A2.
- A3 missed the other way: the census columns cost 0.0013 (CI [-0.0020, -0.0003]), under the predicted 0.002 to 0.008.
- A4c: removing the county rate from the L1 logistic regression changes its CV by -0.00008 (paired bootstrap CI [-0.00019, +0.00002]), inside the predicted range at its upper end; the chosen penalty is the same with and without the rate (C = 0.3 in every fold, reports/models.md).
- A5: cleaning, ratios and dropping the raw county and MSA ids are worth 0.012 together (CI [-0.0133, -0.0105]), inside the range; the same steps added up to 0.013 one after the other with the starting settings (T7).
- A6b fell inside its predicted range.

F6. Ablation gains with error bars of 2 corrected SE. [reports/figs/f6_ablations.png]

## 8. Error analysis and interpretation

### 8.1 Where the final blend ranks well and badly

T10 and F7 cut the out-of-fold predictions of the final blend by 7 slicings fixed before looking: the 15 largest states, loan product, aus-1, income quintile, reported LTV band, exempt record, and the pattern of blanks in 6 key columns. A level under 1,000 rows is listed but not read. AUC inside a slice measures ranking among rows of that slice only, so it can differ from the pooled 0.9204 in either direction. Levels are compared with each other, not with the pooled value.

- Calibration holds everywhere. In the 44 levels read, no residual (accept rate - mean prediction) reaches 2 SE; the largest are Florida -0.008 and New York -0.012, both at 1.95 SE (T10). With 44 levels one or two near 2 SE is what chance gives. The model is not systematically over- or under-confident in any slice read.
- Ranking is weakest on exempt records: 0.773 [0.741, 0.803] against 0.923 [0.920, 0.925] for the others (T10). The blank pattern "LTV + property value + DTI + loan term", which is 99.6% exempt rows, scores 0.764; on those rows DTI, LTV, property value and loan term are all blank.
- The blank pattern "DTI + income" (2,702 rows, 66% of them non-natural-person applicants) is also weak: 0.793 [0.764, 0.822].
- Government loans are ranked less well than conventional ones: FHA and VA first liens 0.850 against 0.937 for conventional first liens, and aus-1 = TOTAL, the FHA scorecard, 0.842 (T10). We thought the DTI rank, the strongest column (F8), separates accepted from denied less well inside FHA and VA, where the tolerated DTI differs (at DTI 50-60%, conventional loans are accepted 0.33 and FHA loans 0.89; reports/feature_plan.md, flag 3.3). Test: AUC of the DTI rank alone, rows with a DTI: 0.680 for conventional first liens, 0.668 for FHA, 0.631 for VA (reports/figs/checks.md). It held for VA and barely for FHA, and the FHA gap stays unexplained.
- Ranking falls with income, from 0.947 in the lowest quintile to 0.860 in the highest, while the denied share falls from 32% to 8% (T10).
- States range from 0.879 (New York) to 0.937 (South Carolina); intervals overlap heavily, and no state stands apart from the others (F7).

The three weak groups (exempt records, entity applicants, government loans) are where the worst-error inspection starts (T11).

T10. Out-of-fold AUC and mean residual by slice, final blend, slices with n >= 1,000.

<!-- begin t10_slices -->
| slice | level | n | denied | accept rate | mean prediction | residual (SE) | AUC [95% CI] |
|---|---|---|---|---|---|---|---|
| state (15 largest) | AZ | 2,692 | 307 | 0.886 | 0.887 | -0.001 (0.006) | 0.887 [0.862, 0.910] |
| state (15 largest) | CA | 7,565 | 927 | 0.877 | 0.879 | -0.002 (0.004) | 0.896 [0.883, 0.909] |
| state (15 largest) | FL | 9,200 | 1,624 | 0.823 | 0.831 | -0.008 (0.004) | 0.892 [0.882, 0.902] |
| state (15 largest) | GA | 3,812 | 666 | 0.825 | 0.830 | -0.004 (0.006) | 0.924 [0.911, 0.937] |
| state (15 largest) | IL | 3,672 | 427 | 0.884 | 0.888 | -0.004 (0.005) | 0.890 [0.870, 0.909] |
| state (15 largest) | IN | 2,463 | 318 | 0.871 | 0.876 | -0.006 (0.007) | 0.920 [0.901, 0.937] |
| state (15 largest) | MI | 3,091 | 464 | 0.850 | 0.844 | +0.006 (0.006) | 0.932 [0.918, 0.946] |
| state (15 largest) | NC | 4,343 | 639 | 0.853 | 0.850 | +0.003 (0.005) | 0.929 [0.916, 0.941] |
| state (15 largest) | NY | 3,422 | 492 | 0.856 | 0.868 | -0.012 (0.006) | 0.879 [0.861, 0.897] |
| state (15 largest) | OH | 3,605 | 404 | 0.888 | 0.890 | -0.002 (0.005) | 0.886 [0.864, 0.907] |
| state (15 largest) | PA | 3,377 | 380 | 0.887 | 0.884 | +0.003 (0.005) | 0.907 [0.888, 0.925] |
| state (15 largest) | SC | 2,467 | 507 | 0.794 | 0.802 | -0.007 (0.008) | 0.937 [0.923, 0.950] |
| state (15 largest) | TN | 2,736 | 430 | 0.843 | 0.849 | -0.006 (0.007) | 0.907 [0.888, 0.926] |
| state (15 largest) | TX | 11,314 | 2,168 | 0.808 | 0.810 | -0.001 (0.004) | 0.934 [0.926, 0.940] |
| state (15 largest) | VA | 2,929 | 303 | 0.897 | 0.893 | +0.004 (0.006) | 0.904 [0.879, 0.926] |
| loan product | Conventional:First Lien | 73,220 | 11,708 | 0.840 | 0.841 | -0.001 (0.001) | 0.937 [0.934, 0.940] |
| loan product | Conventional:Subordinate Lien | 4,295 | 851 | 0.802 | 0.808 | -0.006 (0.006) | 0.895 [0.881, 0.908] |
| loan product | FHA:First Lien | 17,734 | 2,292 | 0.871 | 0.874 | -0.003 (0.003) | 0.850 [0.840, 0.861] |
| loan product | VA:First Lien | 8,690 | 763 | 0.912 | 0.911 | +0.001 (0.003) | 0.850 [0.833, 0.868] |
| aus-1 | 1 DU | 46,972 | 3,786 | 0.919 | 0.921 | -0.002 (0.001) | 0.872 [0.865, 0.879] |
| aus-1 | 1111 exempt | 3,004 | 257 | 0.914 | 0.918 | -0.004 (0.005) | 0.772 [0.739, 0.804] |
| aus-1 | 2 LP / LPA | 20,182 | 1,288 | 0.936 | 0.936 | -0.000 (0.002) | 0.876 [0.864, 0.887] |
| aus-1 | 3 TOTAL | 10,073 | 1,207 | 0.880 | 0.882 | -0.002 (0.003) | 0.842 [0.828, 0.855] |
| aus-1 | 6 no automated system | 22,495 | 8,823 | 0.608 | 0.609 | -0.002 (0.003) | 0.933 [0.929, 0.936] |
| aus-1 | 7 internal system | 1,045 | 195 | 0.813 | 0.834 | -0.021 (0.012) | 0.896 [0.871, 0.919] |
| income quintile | Q1 (-657k to 63k) | 20,096 | 6,528 | 0.675 | 0.680 | -0.004 (0.003) | 0.947 [0.943, 0.950] |
| income quintile | Q2 (63k to 91k) | 20,367 | 3,079 | 0.849 | 0.849 | +0.000 (0.003) | 0.913 [0.907, 0.920] |
| income quintile | Q3 (91k to 126k) | 19,979 | 2,293 | 0.885 | 0.886 | -0.001 (0.002) | 0.905 [0.897, 0.912] |
| income quintile | Q4 (126k to 191k) | 20,011 | 1,674 | 0.916 | 0.917 | -0.001 (0.002) | 0.884 [0.874, 0.893] |
| income quintile | Q5 (191k to 71000k) | 19,969 | 1,535 | 0.923 | 0.925 | -0.002 (0.002) | 0.860 [0.848, 0.871] |
| income quintile | blank | 4,578 | 647 | 0.859 | 0.860 | -0.001 (0.005) | 0.880 [0.862, 0.896] |
| LTV band (reported) | <=80 | 39,599 | 3,594 | 0.909 | 0.910 | -0.000 (0.001) | 0.890 [0.883, 0.896] |
| LTV band (reported) | 80-90 | 11,147 | 1,226 | 0.890 | 0.895 | -0.005 (0.003) | 0.895 [0.884, 0.907] |
| LTV band (reported) | 90-95 | 12,015 | 1,415 | 0.882 | 0.884 | -0.002 (0.003) | 0.927 [0.917, 0.935] |
| LTV band (reported) | 95-100 | 24,654 | 3,032 | 0.877 | 0.879 | -0.002 (0.002) | 0.870 [0.861, 0.878] |
| LTV band (reported) | >100 | 5,711 | 1,006 | 0.824 | 0.829 | -0.005 (0.005) | 0.908 [0.896, 0.920] |
| LTV band (reported) | blank | 11,874 | 5,483 | 0.538 | 0.538 | +0.000 (0.005) | 0.948 [0.944, 0.951] |
| exempt record | exempt (1111) | 3,016 | 259 | 0.914 | 0.917 | -0.003 (0.005) | 0.773 [0.741, 0.803] |
| exempt record | not exempt | 101,984 | 15,497 | 0.848 | 0.850 | -0.002 (0.001) | 0.923 [0.920, 0.925] |
| blank pattern (LTV, property value, DTI, income, term, location) | DTI + income | 2,702 | 281 | 0.896 | 0.899 | -0.003 (0.006) | 0.793 [0.764, 0.822] |
| blank pattern (LTV, property value, DTI, income, term, location) | LTV | 6,267 | 3,174 | 0.494 | 0.485 | +0.009 (0.006) | 0.931 [0.925, 0.937] |
| blank pattern (LTV, property value, DTI, income, term, location) | LTV + location | 1,011 | 1,000 | 0.011 | 0.014 | -0.003 (0.003) | 0.887 [0.803, 0.956] |
| blank pattern (LTV, property value, DTI, income, term, location) | LTV + property value + DTI + loan term | 2,274 | 223 | 0.902 | 0.906 | -0.004 (0.006) | 0.764 [0.729, 0.800] |
| blank pattern (LTV, property value, DTI, income, term, location) | nothing blank | 87,747 | 8,955 | 0.898 | 0.900 | -0.002 (0.001) | 0.889 [0.885, 0.893] |

Final blend, run 1ff742f2, out-of-fold; pooled AUC 0.9204. Residual = accept rate - mean prediction. AUC = ranking inside the slice only, with a 1,000-resample stratified bootstrap interval. Levels under 1,000 rows are not read: aus-1: 2 levels, 1,229 rows; blank pattern (LTV, property value, DTI, income, term, location): 41 levels, 4,999 rows; loan product: 3 levels, 1,061 rows.
<!-- end t10_slices -->

F7. Slice AUC with 95% bootstrap intervals. [reports/figs/f7_slices.png]

### 8.2 What the model uses

Importance needs a model, and only predictions are logged, so the tuned LightGBM (the largest member of the final blend, weight 1/3) was refitted on folds 1 to 4 with its logged settings. The refit reproduces the logged fold-0 AUC exactly (0.912499, run 406e4210), so it is the fold model of that run. It is explained on 10,000 rows of fold 0 drawn with the class balance kept (AUC 0.9106 on those rows; reports/figs/f8_importance.csv). Permutation importance = AUC lost when a column is shuffled (mean of 5 shuffles). SHAP values are LightGBM's exact TreeSHAP contributions in log-odds; they add up to each row's prediction (checked to 1e-6).

- One column dominates: the DTI rank. Shuffling it costs 0.083 AUC, three times the next column (F8). It also has the largest mean |SHAP| (0.64): the highest DTI levels lower the log-odds by 3 to 5 (F9).
- Next come the two leak columns (initially_payable_to_institution 0.029, preapproval 0.022) and aus-1 (0.019). preapproval = 1 adds about +4 to +5 to the log-odds and initially_payable_to_institution = 3 removes 6 to 8.5 (F9).
- Shuffling a whole block at once: finance (12 columns) costs 0.160, process (18) 0.128, demographics (17) 0.009, geography and census (8) 0.009 (F8, right).
- Correlated columns share credit, so single-column numbers understate them. construction_method has the 5th largest mean |SHAP| (0.17) but loses only 0.0015 when shuffled alone. We thought the manufactured-home columns repeat it. Test: every one of the 95,431 site-built rows (construction_method = 1) is "not applicable" or exempt in both manufactured-home columns, and "not applicable" appears on only 16 and 115 other rows (reports/figs/checks.md): it held. LTV, computed LTV and property value each lose 0.006 to 0.007 alone. The block figures are the ones to read for such groups.
- Permutation is not removal. Shuffling the 17 demographic fields costs 0.009 for a model trained with them; a model retrained without them loses 0.0027 (T9, A1).
- SHAP values describe this model, not the world: they say what moved this model's score, not what causes a lender to accept. No sentence of this report reads them causally.

F8. Permutation importance, tuned LightGBM, 10,000 held-out rows: 20 largest single columns (left) and whole blocks (right). [reports/figs/f8_permutation.png]

F9. SHAP values of the 15 columns with the largest mean |SHAP|, same rows; numeric columns coloured by the rank of their value, blanks orange, categories grey. [reports/figs/f9_shap.png]

### 8.3 Worst errors

The 200 worst false positives (denied rows with the highest scores, 0.979 to 0.992) and the 200 worst false negatives (accepted rows with the lowest, 0.009 to 0.185) of the final blend were read by hand (reports/figs/t11_worst_rows.csv). Each group was then compared with the rows of the other class that received the same scores, so that a difference is something the model saw and still could not use (T11a, upper table).

- The worst false positives look like model applications: conventional first liens on site-built homes, owner-occupied, underwritten by DU or LP, moderate DTI, 360-month terms. Nothing in the columns tells them from the accepted rows with the same scores, apart from small shifts: fewer submitted directly to the lender (0.805 against 0.869) and more exempt records (0.040 against 0.017).
- They also hold more single male applicants (0.275 against 0.161, z +4.4) and more applicants over 74 (0.050 against 0.019). These are protected attributes, so they are not candidate features; they go with the lower AUC for Male applicants in T12 (0.905 against 0.933 for Female) to section 9.
- The worst false negatives sit where acceptance is rare: 58% manufactured homes, 80% with no automated underwriting, 45% with DTI above 60% (T11a). The blend is calibrated in those groups (aus-1 = 6: residual -0.002, blank LTV: +0.000, T10).
- Checked and dropped (T11a, last table): the application channel, since every combination of submission and payable with 1,000 rows or more is calibrated within 2 SE; and a county accept rate fitted on the exempt rows only, which lowers their AUC (0.7694 against 0.7729 for the blend alone).

Three candidates survive the residual screen (T11a, middle table). They were chosen after looking at the errors and screened with 9 tests in 3 overlapping regions, so only H1 clears a Bonferroni-corrected threshold (z beyond 2.8). The models were frozen on 2026-11-09, before this analysis, so the three are measured once as a block and reported; they cannot change the final model.

T11a. Worst errors against rows of the other class with the same scores; residual screen of the candidates; candidates dropped before the screen.

<!-- begin t11_worst_errors -->
| rows with | worst FP | accepted, same scores | z FP | worst FN | denied, same scores | z FN |
|---|---|---|---|---|---|---|
| manufactured home | 0.005 | 0.002 | +1.1 | 0.580 | 0.572 | +0.2 |
| location unknown | 0.000 | 0.002 | -0.7 | 0.085 | 0.149 | -2.5 |
| LTV blank | 0.050 | 0.030 | +1.7 | 0.505 | 0.480 | +0.7 |
| no automated underwriting (aus-1 = 6) | 0.010 | 0.018 | -0.9 | 0.800 | 0.717 | +2.6 |
| DTI > 60% | 0.000 | 0.000 | -0.2 | 0.450 | 0.513 | -1.8 |
| conventional loan | 0.920 | 0.919 | +0.1 | 0.950 | 0.847 | +4.0 |
| exempt record (1111) | 0.040 | 0.017 | +2.5 | 0.000 | 0.000 | +0.0 |
| submitted directly to the lender | 0.805 | 0.869 | -2.7 | 0.835 | 0.796 | +1.4 |
| initially payable to the lender | 0.890 | 0.942 | -3.2 | 0.995 | 0.962 | +2.5 |
| single male applicant (derived_sex Male) | 0.275 | 0.161 | +4.4 | 0.330 | 0.309 | +0.6 |
| applicant age > 74 | 0.050 | 0.019 | +3.3 | 0.030 | 0.017 | +1.5 |
| H1 term off the 12-month grid | 0.005 | 0.003 | +0.4 | 0.055 | 0.022 | +3.2 |
| H2 LTV blank, property value present | 0.010 | 0.013 | -0.3 | 0.455 | 0.391 | +1.8 |
| H3 computed LTV > 100 | 0.010 | 0.006 | +0.8 | 0.125 | 0.093 | +1.6 |

Shares of rows. Worst FP: the 200 denied rows with the highest scores (0.979 to 0.992), against the 16,074 accepted rows in that range; worst FN: the 200 accepted rows with the lowest scores (0.009 to 0.185), against the 4,918 denied rows in that range. Final blend, run 1ff742f2. The rows: reports/figs/t11_worst_rows.csv.

Residual (accepted - prediction) with and without each candidate, inside the regions where the worst errors sit:

| candidate | region | rows with | residual with | residual without | difference | z |
|---|---|---|---|---|---|---|
| H1 term off the 12-month grid | low scores: manufactured or no AUS | 1,011 | +0.0299 | -0.0035 | +0.0334 | +3.3 |
| H1 term off the 12-month grid | manufactured homes only | 302 | +0.0723 | -0.0006 | +0.0729 | +3.6 |
| H1 term off the 12-month grid | high scores: prediction > 0.95 | 345 | +0.0039 | +0.0029 | +0.0010 | +0.1 |
| H2 LTV blank, property value present | low scores: manufactured or no AUS | 5,879 | +0.0037 | -0.0040 | +0.0076 | +1.6 |
| H2 LTV blank, property value present | manufactured homes only | 5,482 | +0.0068 | -0.0051 | +0.0118 | +1.9 |
| H2 LTV blank, property value present | high scores: prediction > 0.95 | 1,356 | +0.0124 | +0.0027 | +0.0097 | +2.7 |
| H3 computed LTV > 100 | low scores: manufactured or no AUS | 1,095 | +0.0054 | -0.0025 | +0.0079 | +0.9 |
| H3 computed LTV > 100 | manufactured homes only | 848 | -0.0006 | +0.0019 | -0.0025 | -0.2 |
| H3 computed LTV > 100 | high scores: prediction > 0.95 | 1,166 | +0.0110 | +0.0028 | +0.0082 | +1.9 |

Dropped before the screen:

| candidate | check |
|---|---|
| application channel (submission x payable) | largest absolute z of the residual over 5 combinations with n >= 1,000: 1.9 |
| county accept rate on exempt rows | AUC on the 3,016 exempt rows: blend 0.7729, blend + rate 0.7694, rate alone 0.5393 |
<!-- end t11_worst_errors -->

T11. Wave-2 hypotheses, each a buildable feature, with the prediction written before the run.

| # | feature | definition | evidence (T11a) | why we thought a tree would miss it (tested by the block run) | predicted gain, tuned LightGBM | result |
|---|---|---|---|---|---|---|
| H1 | term_off_grid | loan_term not a multiple of 12 (1,258 train rows: 6, 9, 13, 18 months; 361 to 378 months) | worst FN 0.055 against 0.022 at the same scores; on manufactured homes the residual is +0.073 higher with it (z +3.6) | a multiple of 12 is not a threshold: a tree needs many splits on loan_term to isolate it | +0.0001 to +0.0004, under 2 SE (1.2% of rows) | see the block result below |
| H2 | ltv_blank_value_present | combined LTV blank while property value is reported | residual +0.010 higher with it among scores > 0.95 (z +2.7), +0.008 in the low-score region (z +1.6) | needs a split on each of two NaN patterns in the same branch | +0.0000 to +0.0003, under 2 SE | see the block result below |
| H3 | computed_ltv_above_100 | loan amount > property value | residual +0.008 higher with it among scores > 0.95 (z +1.9) and in the low-score region (z +0.9), but -0.003 on manufactured homes alone (z -0.2) | computed_ltv is in the model, so one split suffices: the weakest of the three | 0 to +0.0002, under 2 SE | see the block result below |

- Block result: the three flags added together to the tuned LightGBM (58 columns, run ebb1bd30) change its CV by -0.00016 against run 406e4210 (-0.2 corrected SE, 2 of 5 folds up, paired bootstrap CI [-0.00060, +0.00061]). Rejected under the rule of section 5.3, and just below the predicted ranges (summed, +0.0001 to +0.0009), though well within noise of their lower end. The patterns of T11a are in the residuals of the blend, but adding them as flags did not raise the CV. The run was made once, after the freeze, and changes nothing in the final model.

## 9. Fairness

Section 2.2 holds the framing: the target is the lender's decision, so everything here describes agreement with past decisions, not their merit. All numbers are out-of-fold on the 105,000 training rows, from notebooks/04_fairness.ipynb, which trains nothing, and reports/fairness.md.

### 9.1 Cost of removing the leaks

Removing the two columns that record the decision costs 0.026 AUC (0.91905 to 0.89305, T9 row A9); removing also the partial signals and the columns they live in costs a further 0.070 (0.82343, row A10). Strict removes real information with the suspect signal, so its gap is an upper bound on what the open verdicts of section 4 could matter, not an estimate of it.

### 9.2 Group diagnostics of the competition model

Model: the final blend (run 1ff742f2). It sees the demographic fields and both leaks, so these numbers describe the competition model, not a fair one; they are the baseline for the fair models of section 9.4.

Conventions: groups are "derived_race", "derived_ethnicity", "derived_sex" and "applicant_age"; groups under 500 rows are not read; intervals are 95% bootstrap over rows (1,000 resamples) and ignore training uncertainty. "Same rate" is the cut-off 0.780, the score quantile that accepts the train accept rate (0.850) of rows, as lenders did; each model gets its own same-rate cut-off. Gaps are against the largest group with one declared value, a base for subtraction, not a norm. TPR = share of accepted rows the model accepts, FPR = share of denied rows it accepts.

T12. Group diagnostics of the final blend, race, ethnicity and sex, groups with n >= 500 (reports/fairness.md; age and the cut-off 0.50 in the same file).

| field | group | n | observed accept rate | pred - obs | AUC | TPR gap, same rate | FPR gap, same rate |
|---|---|---|---|---|---|---|---|
| race | White (reference) | 66,107 | 0.868 | +0.001 | 0.921 [0.917, 0.924] | | |
| race | Race Not Available | 17,872 | 0.828 | +0.003 | 0.923 [0.916, 0.929] | -0.011 [-0.016, -0.008] | -0.035 [-0.052, -0.018] |
| race | Black or African American | 8,991 | 0.739 | +0.005 | 0.909 [0.901, 0.917] | -0.037 [-0.044, -0.030] | -0.044 [-0.062, -0.025] |
| race | Asian | 8,009 | 0.889 | +0.004 | 0.882 [0.867, 0.896] | +0.014 [0.010, 0.018] | +0.096 [0.061, 0.127] |
| race | Joint | 2,716 | 0.872 | -0.001 | 0.911 [0.889, 0.933] | +0.005 [-0.003, 0.012] | -0.019 [-0.064, 0.029] |
| race | American Indian or Alaska Native | 810 | 0.727 | -0.008 | 0.939 [0.917, 0.960] | -0.059 [-0.083, -0.036] | -0.119 [-0.164, -0.068] |
| ethnicity | Not Hispanic or Latino (reference) | 70,469 | 0.865 | +0.001 | 0.920 [0.916, 0.924] | | |
| ethnicity | Ethnicity Not Available | 16,232 | 0.833 | +0.001 | 0.924 [0.917, 0.931] | -0.013 [-0.017, -0.009] | -0.041 [-0.060, -0.024] |
| ethnicity | Hispanic or Latino | 15,102 | 0.793 | +0.004 | 0.910 [0.902, 0.917] | -0.016 [-0.020, -0.011] | -0.031 [-0.048, -0.013] |
| ethnicity | Joint | 3,170 | 0.868 | +0.006 | 0.921 [0.903, 0.937] | +0.003 [-0.004, 0.010] | +0.005 [-0.038, 0.046] |
| sex | Male (reference) | 34,507 | 0.845 | +0.003 | 0.905 [0.900, 0.910] | | |
| sex | Joint | 38,731 | 0.883 | +0.001 | 0.922 [0.917, 0.927] | +0.014 [0.011, 0.017] | -0.009 [-0.026, 0.009] |
| sex | Female | 23,462 | 0.805 | +0.000 | 0.933 [0.928, 0.938] | -0.007 [-0.011, -0.003] | -0.075 [-0.092, -0.058] |
| sex | Sex Not Available | 8,300 | 0.844 | +0.004 | 0.913 [0.903, 0.923] | -0.002 [-0.008, 0.003] | -0.031 [-0.058, -0.006] |

Reading (T12):

- The model reproduces the gaps in acceptance between groups. In every group read, the mean prediction is within 0.01 of the observed rate: 0.869 predicted against 0.868 observed for White applicants, 0.744 against 0.739 for Black or African American applicants. At the same-rate cut-off it accepts 0.868 and 0.739, the same 12.9-point gap as the recorded decisions. The model had the demographic fields as inputs, so this does not show how much of the gap passes through them; the fair models of section 9.4 remove them.
- Against each reference, the TPR and FPR gaps mostly point the same way: lower for Black or African American applicants (-0.037 and -0.044), American Indian or Alaska Native (-0.059 and -0.119) and Hispanic or Latino applicants (-0.016 and -0.031); higher for Asian applicants (+0.014 and +0.096). For Female against Male the gap sits in the FPR (-0.075; TPR -0.007).
- Ranking quality differs: AUC runs from 0.882 (Asian) to 0.939 (American Indian or Alaska Native), and Female (0.933) and Male (0.905) have intervals apart. A lower AUC means the model separates accepted from denied less well inside the group.
- Without the 6,279 leak rows AUC falls in every group, by 0.006 to 0.027, most for Asian applicants (0.882 to 0.855): the leaks do not flatter all groups equally.
- The table holds 72 intervals against a reference; about 3 or 4 would exclude 0 by chance. The four attributes share rows, so these are not separate findings, and nothing here compares similar applicants.

### 9.3 Demographics alone, and proxies

T13. How much of the decision the declared demographics predict (upper table), and how well the other columns predict declared race (lower table).

| model | target | CV AUC | run |
|---|---|---|---|
| LightGBM on the 17 demographic fields only | accepted | 0.63546 | e63bd31a |

| race group (derived_race), one against all others | rows | AUC from the 38 race-blind columns | AUC from geography only (8 columns) |
|---|---|---|---|
| White | 66,107 | 0.740 [0.735, 0.743] | 0.695 [0.689, 0.699] |
| Race Not Available | 17,872 | 0.672 [0.669, 0.675] | 0.587 [0.581, 0.592] |
| Black or African American | 8,991 | 0.840 [0.833, 0.849] | 0.796 [0.790, 0.800] |
| Asian | 8,009 | 0.847 [0.840, 0.851] | 0.800 [0.792, 0.805] |

Proxy check (src/proxy_check.py, reports/tuning/proxy_check.csv): a LightGBM with the competition settings predicts one race group against all others, mean of the 5 fold AUCs on the frozen folds, [lowest, highest] fold in brackets; groups with at least 5,000 rows. The 38 columns are those of ablation A1 (all 55 minus the 17 demographic fields); geography is the 6 tract columns, the MSA median family income and the state. AUC here is the chance that a random applicant of the group scores above a random applicant outside it; 0.5 means the columns say nothing about the group.

- Removing the declared race fields does not remove race from the inputs. The columns left identify Black or African American applicants at AUC 0.840 and Asian applicants at 0.847 (T13). In concrete terms: among the 10% of rows with the highest score from the 38 columns, 39% are Black applicants, against 9% in the whole data, and those rows hold 45% of all Black applicants; for Asian applicants, 38% against 8%, and 50% (reports/fairness.md).
- Most of it comes through geography: the 8 geographic columns alone reach 0.796 and 0.800. They include tract_minority_population_percent, the share of minority residents in the applicant's tract, which describes the area, not the applicant. The 30 other columns add about 0.04.
- "Race Not Available" is the least predictable group and depends least on geography: 0.587 from geography, 0.672 from all 38 columns.
- What this check does not measure: how much the competition model relies on these columns as a stand-in for race. Geography also carries what a lender may legitimately use, such as local prices, so the same columns are both a proxy and a predictor, and the check cannot separate the two.

The demographics-only model was kept out of every blend (reports/models.md). [pending, from reports/fairness.md: the written justification of that decision]

### 9.4 Fair models

Six LightGBM models with the tuned settings, on the frozen folds, so that one thing changes at a time (reports/fairness.md, "Fair models"):

- Two feature lists. The Kaggle list keeps everything, leaks included: it is what the leaderboard scores. The clean list removes the 2 confirmed leaks: it is the one a lender could use before deciding.
- Three treatments on each list: with the 17 demographic fields; race-blind, without them; race-blind with reweighing.
- Reweighing (Kamiran and Calders, 2012): in the training part of each fold, each row gets the weight P(g) P(y) / P(g, y), with g = "derived_race" and y = accepted, so that in the weighted training data race and the label are independent (weights 0.3 to 1.6). Race sets the weights and is not a feature; scoring is unweighted.

The fair model put forward is the clean list, race-blind, reweighed (config fair_reweigh). Its Kaggle-list twin (fair_full_reweigh) measures what the same two steps cost on the leaderboard.

T15. Cost of the fair models. CV changes are paired against the step before, with the corrected SE of section 5.3; public leaderboard scores are submissions 4 and 6 to 10 (reports/submissions.md), none of them among the two final submissions.

| model | config | columns | run | CV AUC | vs the step before | public LB (submission) |
|---|---|---|---|---|---|---|
| final blend (competition model) | blends/lgbm_cat_et_xgb.yaml | | 1ff742f2 | 0.92054 | | 0.92158 (6) |
| Kaggle list, with the demographic fields | fair_full | 55 | 406e4210 | 0.91905 | | 0.92100 (4, same config) |
| Kaggle list, race-blind | ablate_no_demographics | 38 | 20a010b4 | 0.91636 | -0.00270 (-5.2 SE) | 0.91869 (7) |
| Kaggle list, race-blind + reweighing | fair_full_reweigh | 38 | 6990a174 | 0.91560 | -0.00075 (-0.9 SE) | 0.91855 (8) |
| clean list, with the demographic fields | fair_clean | 53 | 116f6267 | 0.89305 | -0.02600 (-13.6 SE), against fair_full | not submitted |
| clean list, race-blind | fair_unaware | 36 | 8e9bf39c | 0.89060 | -0.00245 (-3.1 SE) | 0.89408 (9) |
| clean list, race-blind + reweighing | fair_reweigh | 36 | 9ddab9a2 | 0.88985 | -0.00075 (-2.4 SE) | 0.89342 (10) |

- Race-blind costs 0.0027 of CV AUC on the Kaggle list and 0.0025 on the clean list. On the public leaderboard it costs 0.0023 (submission 4 to 7), about 1.6 times the 0.0014 SD of a paired gap measured between two submitted models (reports/models.md): the same size as on CV, so the cost carries over to unseen rows.
- Reweighing on top costs 0.0008 on both lists: not resolved on the Kaggle list (0.9 SE), 2.4 SE on the clean list; on the leaderboard 0.0001 and 0.0007, inside the noise.
- The leaks cost 0.026, seven to eight times more than both fairness steps together.

T16. Accept, TPR and FPR gaps for the three largest pairs, other group minus reference, at each model's same-rate cut-off, 95% bootstrap intervals. The "recorded decisions" row is the gap a perfectly faithful model would reproduce.

| pair | model | accept gap | TPR gap | FPR gap |
|---|---|---|---|---|
| Black or African American vs White | recorded decisions | -0.129 [-0.138, -0.120] | | |
| Black or African American vs White | final blend | -0.129 [-0.139, -0.119] | -0.037 [-0.044, -0.030] | -0.044 [-0.062, -0.025] |
| Black or African American vs White | Kaggle list, with demographics | -0.129 [-0.139, -0.119] | -0.038 [-0.045, -0.031] | -0.042 [-0.060, -0.023] |
| Black or African American vs White | Kaggle list, race-blind | -0.104 [-0.113, -0.095] | -0.013 [-0.019, -0.007] | -0.017 [-0.037, 0.003] |
| Black or African American vs White | Kaggle list, race-blind + reweighing | -0.098 [-0.107, -0.088] | -0.008 [-0.013, -0.002] | -0.008 [-0.027, 0.012] |
| Black or African American vs White | clean list, with demographics | -0.139 [-0.148, -0.130] | -0.055 [-0.062, -0.047] | -0.065 [-0.084, -0.043] |
| Black or African American vs White | clean list, race-blind | -0.108 [-0.117, -0.099] | -0.024 [-0.031, -0.018] | -0.032 [-0.053, -0.011] |
| Black or African American vs White | clean list, race-blind + reweighing | -0.096 [-0.104, -0.087] | -0.010 [-0.017, -0.005] | -0.023 [-0.044, -0.002] |
| Hispanic or Latino vs Not Hispanic or Latino | recorded decisions | -0.072 [-0.079, -0.065] | | |
| Hispanic or Latino vs Not Hispanic or Latino | final blend | -0.069 [-0.075, -0.062] | -0.016 [-0.020, -0.011] | -0.031 [-0.048, -0.013] |
| Hispanic or Latino vs Not Hispanic or Latino | Kaggle list, with demographics | -0.069 [-0.076, -0.063] | -0.015 [-0.020, -0.011] | -0.035 [-0.051, -0.017] |
| Hispanic or Latino vs Not Hispanic or Latino | Kaggle list, race-blind | -0.065 [-0.072, -0.058] | -0.012 [-0.017, -0.008] | -0.026 [-0.044, -0.009] |
| Hispanic or Latino vs Not Hispanic or Latino | Kaggle list, race-blind + reweighing | -0.064 [-0.071, -0.057] | -0.011 [-0.016, -0.007] | -0.025 [-0.042, -0.007] |
| Hispanic or Latino vs Not Hispanic or Latino | clean list, with demographics | -0.071 [-0.078, -0.065] | -0.020 [-0.025, -0.014] | -0.050 [-0.068, -0.031] |
| Hispanic or Latino vs Not Hispanic or Latino | clean list, race-blind | -0.068 [-0.076, -0.062] | -0.017 [-0.022, -0.012] | -0.047 [-0.066, -0.029] |
| Hispanic or Latino vs Not Hispanic or Latino | clean list, race-blind + reweighing | -0.067 [-0.074, -0.060] | -0.016 [-0.022, -0.011] | -0.044 [-0.062, -0.025] |
| Female vs Male | recorded decisions | -0.040 [-0.047, -0.034] | | |
| Female vs Male | final blend | -0.047 [-0.053, -0.041] | -0.007 [-0.011, -0.003] | -0.075 [-0.092, -0.058] |
| Female vs Male | Kaggle list, with demographics | -0.048 [-0.054, -0.042] | -0.008 [-0.013, -0.004] | -0.073 [-0.090, -0.056] |
| Female vs Male | Kaggle list, race-blind | -0.048 [-0.054, -0.042] | -0.008 [-0.012, -0.004] | -0.075 [-0.093, -0.059] |
| Female vs Male | Kaggle list, race-blind + reweighing | -0.048 [-0.054, -0.041] | -0.008 [-0.012, -0.004] | -0.074 [-0.092, -0.058] |
| Female vs Male | clean list, with demographics | -0.048 [-0.055, -0.042] | -0.010 [-0.014, -0.005] | -0.082 [-0.100, -0.066] |
| Female vs Male | clean list, race-blind | -0.048 [-0.054, -0.041] | -0.009 [-0.014, -0.005] | -0.082 [-0.101, -0.064] |
| Female vs Male | clean list, race-blind + reweighing | -0.047 [-0.053, -0.040] | -0.008 [-0.013, -0.004] | -0.082 [-0.100, -0.064] |

Cost against gain (T15, T16):

- The two fairness steps narrow the gap between Black and White applicants by about a quarter, for about 0.003 AUC. The recorded decisions and the competition model accept Black applicants 12.9 points less often. Race-blind, the gap falls to 10.4 points on the Kaggle list (10.8 on the clean list); reweighed, to 9.8 (9.6). Most of the narrowing comes from removing the fields; reweighing adds 0.6 to 1.2 points (paired intervals in reports/fairness.md).
- The narrowing is clearest among applicants the lenders accepted. The TPR gap of Black applicants goes from -0.038 to -0.008 [-0.013, -0.002] on the Kaggle list and from -0.055 to -0.010 [-0.017, -0.005] on the clean list. The FPR gap of the reweighed Kaggle-list model, -0.008 [-0.027, 0.012], cannot be told from 0.
- The price is calibration. The race-blind models predict, for Black applicants, more acceptances than lenders gave: +0.029 on the Kaggle list and +0.031 on the clean list, and +0.035 and +0.038 reweighed; for White applicants slightly fewer (-0.005 to -0.007). The competition model was within 0.005 for both (T12). With one cut-off for every group, a higher mean score for Black applicants means more of them pass it.
- The other pairs barely move. The Hispanic gap narrows by 0.5 point at most and the Female gap not at all, and their TPR and FPR gaps stay within their intervals. At the same-rate cut-off every model accepts women 4.7 to 4.8 points less often than men, against 4.0 points in the recorded decisions, while its mean prediction for women is within 0.002 of the observed rate: a calibrated score can still widen a gap once one cut-off turns it into yes or no.
- Ranking quality falls evenly across groups. In every group, the race-blind Kaggle-list models are at most 0.007 below the competition blend, and the clean-list models 0.02 to 0.04 below it in most groups (0.038 for Asian applicants). Asian applicants remain the group ranked least well (0.844 for fair_reweigh).

Removing the leaks while keeping the demographic fields widens the Black-White accept gap by 1.0 point [0.6, 1.5], and the TPR gap from -0.038 to -0.055 (T16). We thought the widening came from the 6,279 leak rows: there the Kaggle-list model reads the decision from the leak column, while the clean model has to score those rows from the other columns, the demographic fields included. If so, the gap should barely change on the rows without a leak value and widen on the leak rows. The test reads the same out-of-fold predictions on each subset, with each model's same-rate cut-off set on all rows, and a paired bootstrap over rows (1,000 resamples).

T17. Black-White accept gap, Kaggle list against clean list, both with the demographic fields, by subset of rows (reports/fairness.md).

| rows | White / Black rows | accept gap, Kaggle list | accept gap, clean list | change [95%] |
|---|---|---|---|---|
| all | 66,107 / 8,991 | -0.129 | -0.139 | -0.010 [-0.015, -0.006] |
| without a leak value | 62,128 / 8,465 | -0.126 | -0.140 | -0.014 [-0.018, -0.010] |
| leak rows only | 3,979 / 526 | -0.179 | -0.125 | +0.054 [0.013, 0.093] |

The hypothesis was wrong: the widening is on the rows without a leak value, and on the leak rows the gap narrows. The two groups have the same share of leak rows (6.0% and 5.9%), so the mix of rows does not account for it either. The leak-free fair model still ends with a gap no wider than its Kaggle-list twin (9.6 against 9.8 points, T16).

Limits: all of this is on the training rows, out-of-fold; the intervals cover the sampling of rows, not the training of the models, and are not corrected for the many comparisons (where there is no real gap, about 1 interval in 20 still excludes 0). Reweighing was run once, on race only, with no tuning.

### 9.5 Which fairness criterion, and why

With accepted as the positive class and one cut-off for everyone, there are three common ways to ask whether a model treats two groups alike (independence, separation and sufficiency, in the vocabulary of Barocas, Hardt and Narayanan):

- Equal accept rates (demographic parity): the model accepts the same share of each group.
- Equal error rates (equalized odds): among applicants the lenders accepted, the model accepts the same share in each group (TPR), and among applicants the lenders denied, too (FPR). Equal opportunity asks for the TPR only (Hardt, Price and Srebro, 2016).
- Calibration within groups: a given score means the same accept rate in every group.

Here they cannot all hold. The recorded decisions accept 86.8% of White and 73.9% of Black applicants. At any cut-off and for any group, FPR = p / (1 - p) x (1 - PPV) / PPV x TPR, where p is the group's recorded accept rate and PPV the share of the model's acceptances that lenders also accepted (Chouldechova, 2017). It is an identity, and it holds exactly for every model and group of T18 (checked in notebooks/04_fairness.ipynb). With our two base rates, equal TPR and equal PPV would force the FPR of Black applicants to 0.43 times that of White applicants: equal error rates and equal PPV, the cut-off form of calibration, cannot hold together. And since the model's accept rate in a group is p x TPR + (1 - p) x FPR, equal TPR and FPR would still leave Black applicants accepted 8 to 9 points less often (0.129 x (TPR - FPR) with the White rates of T18): equal error rates and equal accept rates cannot hold together either. A choice has to be made.

T18. Where the models sit, at each model's same-rate cut-off (reports/fairness.md).

| model | group | recorded accept rate | mean prediction | model accepts | TPR | FPR | PPV |
|---|---|---|---|---|---|---|---|
| final blend | White | 0.868 | 0.869 | 0.868 | 0.959 | 0.264 | 0.960 |
| final blend | Black or African American | 0.739 | 0.744 | 0.739 | 0.922 | 0.220 | 0.922 |
| fair_full_reweigh | White | 0.868 | 0.862 | 0.862 | 0.954 | 0.258 | 0.960 |
| fair_full_reweigh | Black or African American | 0.739 | 0.774 | 0.765 | 0.947 | 0.250 | 0.914 |
| fair_reweigh | White | 0.868 | 0.861 | 0.861 | 0.945 | 0.311 | 0.952 |
| fair_reweigh | Black or African American | 0.739 | 0.776 | 0.766 | 0.935 | 0.288 | 0.902 |

T19. Calibration inside score bands (the five quintiles of each model's score over all rows): mean prediction / recorded accept rate, by group (reports/fairness.md).

| model | score band | White n | White | Black n | Black |
|---|---|---|---|---|---|
| final blend | 1: 0.00 to 0.88 | 11,479 | 0.415 / 0.402 | 3,379 | 0.425 / 0.408 |
| final blend | 2: 0.88 to 0.95 | 10,975 | 0.924 / 0.922 | 3,671 | 0.921 / 0.923 |
| final blend | 3: 0.95 to 0.97 | 12,654 | 0.959 / 0.960 | 1,438 | 0.956 / 0.958 |
| final blend | 4: 0.97 to 0.98 | 14,079 | 0.974 / 0.977 | 185 | 0.971 / 0.978 |
| final blend | 5: 0.98 to 1.00 | 16,920 | 0.986 / 0.990 | 318 | 0.996 / 0.997 |
| fair_reweigh | 1: 0.00 to 0.87 | 12,409 | 0.469 / 0.477 | 2,680 | 0.388 / 0.331 |
| fair_reweigh | 2: 0.87 to 0.94 | 12,890 | 0.911 / 0.919 | 2,257 | 0.911 / 0.866 |
| fair_reweigh | 3: 0.94 to 0.96 | 13,037 | 0.947 / 0.956 | 1,914 | 0.947 / 0.920 |
| fair_reweigh | 4: 0.96 to 0.97 | 13,388 | 0.964 / 0.970 | 1,372 | 0.964 / 0.947 |
| fair_reweigh | 5: 0.97 to 0.99 | 14,383 | 0.978 / 0.985 | 768 | 0.977 / 0.962 |

- The competition model is calibrated: in every band its mean prediction is within 0.017 of the recorded rate, for both groups (T19). Every other gap stays: it accepts Black applicants 12.9 points less often, with a TPR lower by 0.037, an FPR lower by 0.044 and a PPV lower by 0.038 (T18).
- The fair model moves toward equal error rates: TPR gap -0.010, FPR gap -0.023, and it accepts Black applicants 9.6 points less often, not far above the 8.2 points that fully equal error rates would still leave for this model (T18). The price is calibration: for Black applicants its mean prediction is 0.776 against 0.739 recorded (+0.038), and inside every score band the recorded rate of Black applicants is below the prediction, by 0.015 to 0.057, while for White applicants it is 0.006 to 0.009 above (T19). At a score between 0.87 and 0.94, lenders accepted 91.9% of the White and 86.6% of the Black applicants the model scored there (binomial SE about 0.007 for the Black rate).
- The PPV gap widens from -0.038 to -0.051. This is the identity at work: with the base rates unchanged, narrowing the TPR gap moves the difference into PPV.

Our choice. For the fair model, equal opportunity is the main criterion: the TPR gap at the same-rate cut-off, with its interval; the FPR gap and the accept gap are reported beside it, and the calibration gap as its price. For the competition model the criterion is calibration. The two models serve different purposes, so they are judged differently. Three reasons:

- Why not calibration for the fair model. The label is what a lender decided, not whether the applicant repaid. A model calibrated to it reproduces each group's recorded accept rate, including whatever part of the 12.9-point gap does not come from risk; holding the fair model to calibration would hold it to the very decisions under review. Calibration is the right test for the competition model because it makes that model's group tables a faithful description of what lenders did; it does not say whether what they did was fair.
- Why the TPR rather than the FPR. 1 - TPR is the share of applicants the lenders accepted that the model would turn down: the error that costs an applicant a loan. Equal opportunity asks that this error not depend on the group. The FPR concerns applicants the lenders denied whom the model would accept, a question of the lender's risk more than of the applicant's harm.
- Why not equal accept rates. The data has no credit score, no interest rate and no repayment, so it cannot say how much of the gap is a difference in risk and how much a difference in treatment. Equal accept rates would remove the whole gap, which amounts to assuming that none of it is risk. Equal opportunity does not need that split.

What the choice does not escape. The TPR is measured against the lenders' decisions: "accepted by the lender" stands in for "should get the loan", so if the lenders' acceptances treated groups unequally, an equal TPR relative to them carries that over. Every criterion that can be computed on this data has the same limit, since all are defined against the recorded decision. The TPR also depends on the cut-off: the competition sets none, and at 0.50 the gaps differ from those at the same-rate cut-off (reports/fairness.md). And equal opportunity is measured on race, and the fair model was reweighed on race only; the ethnicity, sex and age gaps were measured but not targeted.

The "observed" fields. When an applicant applying in person does not give race, ethnicity or sex, the lender records them from visual observation or surname, and "applicant_race_observed", "applicant_ethnicity_observed" and "applicant_sex_observed" say whether that happened. On those rows the group is the lender's perception, not the applicant's answer. Race was recorded by observation for 1,320 rows (1.3%): 122 of the 8,991 Black applicants (1.4%) and 1,091 of the 66,107 White applicants (1.7%), too few to change a group table; the accept rate is 0.849 to 0.864 whatever the value of these fields (reports/fairness.md). They are among the 17 demographic fields the race-blind models leave out.

## 10. Limitations and reproducibility

### 10.1 Limitations

What the data cannot carry:

- No credit score, no interest rate, no applicant history (section 3). The worst false positives cannot be told from accepted rows with the same scores on any column the data has (section 8.3).
- A selected sample: home purchase loans, originated against denied (section 2.2). Nothing here carries over to refinancing, to withdrawn or incomplete files, or to applicants who never applied.
- The collection year is not stated. Code meanings were read from the current filing guide and Regulation C commentary (reports/refs/hmda_dictionary.md); a code whose meaning changed between years would be misread.
- The competition model uses two columns written because of the decision. Its score overstates what can be predicted when an application arrives by 0.026 AUC (T9, row A9), and three partial signals stay even in the clean list (section 4).

What the validation cannot rule out:

- Every choice was made on the same frozen folds. The finalists were re-checked on fresh splits, which measured the flattering for three of them: about 0.0002 for the LightGBM search, 0.00025 for the 50/50 blend, and none for the final blend (section 5.3, reports/models.md); seed 0, the seed of the search, adds about 0.0003 (section 6.3). Earlier choices (feature blocks, the ablations) were not re-checked on fresh splits.
- The corrected SE is a heuristic for k-fold (section 5.3), and the bootstrap intervals of T1 and T10 cover test-row noise only, not the variation from training on other rows.
- Train and test look exchangeable (adversarial AUC 0.5015, section 3), but the private ranking scores 70% of test, a sample with its own noise, between the 50% and 100% rows of T6.
- The ablations keep the settings tuned for the full feature list; none was re-tuned (section 7).

What the interpretation cannot claim:

- Importance and SHAP values describe one member of the blend (the tuned LightGBM, weight 1/3), one fold model, on 10,000 rows (section 8.2). Correlated columns share credit, and the values are attributions for that model, not effects of the columns.
- The worst-error hypotheses were chosen after looking at the errors (section 8.3) and measured once, after the freeze.
- Every group difference in section 9 is an association between declared attributes and recorded decisions. With no credit score, rate or history, the data cannot separate a difference in risk from a difference in treatment: we do not conclude discrimination where a gap appears, nor its absence where none does.
- Every fairness criterion that can be computed here is defined against the recorded decision. An equal TPR relative to the lenders carries over whatever unequal treatment their acceptances contain (section 9.5).
- The fair-model results rest on one reweighing run, on race only, with no tuning; their intervals cover the sampling of rows, not the training of the models, and are not corrected for the many comparisons (section 9.4).

### 10.2 Reproducibility

Environment. uv.lock pins every library, and every Makefile command runs through "uv run --locked", which refuses to run if the lock and the environment disagree. Versions: Python 3.13.12, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, LightGBM 4.7.0 (recorded with each run in reports/runs.csv), CatBoost 1.2.10, XGBoost 3.4.1, PyTorch 2.15.0 CPU (MLP only), Optuna 5.0.0, matplotlib 3.11.2 (read from the locked environment on 2026-11-10; runs.csv does not record these).

Hardware. One laptop: Intel Core i7-10610U, 8 logical CPUs, 30 GB RAM, no GPU. Every model ran on the CPU.

Data and folds. The raw files are not distributed; their SHA256 sums are in configs/data_checksums.txt and "make check-data" verifies them. The 5 folds are built once by "make folds" into data/processed/folds.parquet; the code refuses another fold seed, and every run in the final model carries the same fold hash (3762a22f..., reports/runs.csv).

T14. The runs of the final model (reports/runs.csv).

| member | config | run | commit | uncommitted changes | wall time | CV AUC | re-run from clean commit d782fce |
|---|---|---|---|---|---|---|---|
| LightGBM, tuned | configs/experiments/lgbm_optuna_best.yaml | 406e4210 | 500b562 | yes | 2 min | 0.91905 | cc720ddf: identical fold AUCs and predictions |
| CatBoost, raw county / MSA | configs/experiments/catboost_county.yaml | d96c8be4 | 79f2bb7 | no | 2 h 4 min | 0.91859 | [pending] |
| ExtraTrees, tuned | configs/experiments/extratrees_optuna_best.yaml | 7bc88a0b | 868b301 | yes | 7 min | 0.91484 | f308315c: identical fold AUCs, predictions within 1e-15 |
| XGBoost, tuned | configs/experiments/xgb_optuna_best.yaml | 05b22e57 | c6d4581 | yes | 3 min | 0.91848 | 104e3dd7: identical fold AUCs and predictions |
| final blend, 2/6, 2/6, 1/6, 1/6 log-odds | configs/blends/lgbm_cat_et_xgb.yaml | 1ff742f2 | c6d4581 | yes | under 1 s | 0.92054 | [pending, depends on CatBoost] |

Rebuilding the final model, the submission and the figures (about 2.5 hours, CatBoost being most of it):

    make setup
    make check-data
    make folds
    make cv EXP=configs/experiments/lgbm_optuna_best.yaml
    make cv EXP=configs/experiments/catboost_county.yaml
    make cv EXP=configs/experiments/extratrees_optuna_best.yaml
    make cv EXP=configs/experiments/xgb_optuna_best.yaml
    (put the 4 new run ids in configs/blends/lgbm_cat_et_xgb.yaml)
    make blend BLEND=configs/blends/lgbm_cat_et_xgb.yaml
    make submit RUN_ID=<the blend's run id>
    make figures

Each fold model is trained on the other 4 folds and the test prediction is the mean of the 5 fold models: no model is refitted on all of train, so every test prediction comes from models whose scores were measured (src/cv.py).

What is and is not guaranteed:

- Settings are exact: runs.csv stores the full merged configuration of every run, with its hash.
- Code: three of the four members were logged with uncommitted changes (T14), so the log alone does not pin their code. All four were therefore re-run on 2026-11-10 from clean commit d782fce, with the same configurations and folds (reports/models.md).
- LightGBM, ExtraTrees and XGBoost reproduce: the same AUC on every fold, and the same out-of-fold and test predictions (bit for bit for LightGBM and XGBoost, within 1e-15 for ExtraTrees). Their published scores are rebuilt exactly from commit d782fce and uv.lock.
- CatBoost: [pending: the result of the re-run and of the test of its cause]

### 10.3 What deployment would require

A model put to use would score applications before any decision exists. It would need the clean list by construction; a target closer to what a lender should predict than past approvals (repayment, where it can be observed); and validation on applications later than those it was trained on, which a challenge with no stated year and exchangeable train and test rows cannot provide. It would also need the group checks of section 9 repeated on the lender's own applicants and over time, and a legal review under fair-lending law (the Equal Credit Opportunity Act and the Fair Housing Act), including whether race may be used at all to set training weights, as reweighing does, and how to give each denied applicant the specific reasons the law requires, which a score alone does not provide (reports/fairness.md).
