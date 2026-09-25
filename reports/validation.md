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

Fold content hash (sha256 of the id-sorted `id,fold` CSV):
`3762a22f91732e8423074998d2a647865c70be3cbe007161b2356b24fd57b39c`
