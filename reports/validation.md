# Validation

## Split choice

We use "StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)" (settings in
"configs/base.yaml"), built in "data/processed/folds.parquet" and loaded for every run. Also k=5 is commun, and as I don't see a reason not to use it, we are using it.
The test set looks like a random draw from the same pool as train, so it's good. 
A shuffled K-fold is ok for this task then. We considered "GroupKFold" in two forms. 
- Grouping by duplicate key would only matter for 7 duplicated rows, which I assume cann not bias a 105,000 lenght dataset. 
- Grouping by county would simulate predicting unseen counties, but only 115 of 45,000 test rows (0.26%) fall in counties absent from train. 
- Grouped validation would measure a harder task than the one scored and would reward the wrong modelling choices. 

This reasoning assumes train and test are exchangeable. 
[Note for Volodya] An adversarial validation must test that assumption, and this section will be revised if it fails.

## Uncertainty about SE

We report $\sigma/\sqrt{k}$ only as a naive lower bound. It assumes independent folds, but fold models share 3/4 (k-2/k-1) of their training data, so it underestimates the true SE [TODO read the complete reason here (Nadeau & Bengio, 2003; no unbiased estimator exists, Bengio & Grandvalet, 2004)]. 

I think about paired per-fold differences, a bootstrap CI. More data (for instance other HMDA years, if allowed...) would shrink the uncertainty but not remove this bias.

Fold content hash (sha256 of the id-sorted `id,fold` CSV):
`3762a22f91732e8423074998d2a647865c70be3cbe007161b2356b24fd57b39c`

## validation method
We will use CV to validate our model and give it to kaggle, and we will try to not overfit watching kaggle score.

## Decision rule

A change (a feature block, a parameter) is adopted if and only if its paired mean gain over the 5 folds is positive and at least 2 × the Nadeau–Bengio corrected SE of that gain. Tool: "make compare BASE=<run_id> NEW=<run_id>" ("src/compare.py").

- Paired: both runs are scored on the same frozen folds, so fold-difficulty effects cancel in the difference. The tool refuses two runs with different folds.
- Corrected SE = std of the per-fold gains × √(1/k + 1/(k−1)), i.e. 1.5 × the naive SE at k = 5. It accounts for the overlap of the training sets. Chosen as the conservative option; for k-fold it is a heuristic (the formula was derived for repeated random splits).
- Also printed on every comparison, so the rule can be revisited on evidence: the naive SE (std / √k, too small) and a paired stratified bootstrap 95% CI on the pooled out-of-fold predictions (1,000 resamples; it measures test-row noise only, not training variability).