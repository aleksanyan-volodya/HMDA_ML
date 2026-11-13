# HMDA mortgage outcomes: prediction and fairness

The final code is on branch main. This README is the entry point: what the project does, where
the data goes, and the commands that rebuild the two selected submissions.

Kaggle task: predict whether a home purchase application in the HMDA records was originated
(accepted = 1) or denied, scored by ROC AUC. The project also measures how the model treats
demographic groups and builds a fair model. The write-up is reports/report.md; this file says
how the code is organised and how to rebuild the results.

Final model: a weighted mean of the log-odds of four models (LightGBM 2/6, CatBoost 2/6,
ExtraTrees 1/6, XGBoost 1/6). CV AUC 0.92054 on 5 frozen folds, 0.92158 on the public
leaderboard (report T1).

## Pipeline

    data/raw/train.csv, test.csv
      -> src/data.py           typed columns; train and test get the same category levels
      -> src/features.py       the transforms named in the experiment config, in order (no labels used)
      -> src/folds.py          frozen 5-fold split, data/processed/folds.parquet (seed 42)
      -> src/cv.py             for each fold: target encodings fitted on the fit part, the model fitted
                               with its settings (tree count, penalty, epochs) chosen by an inner CV on
                               the fit part only, out-of-fold prediction on the held-out fold;
                               test prediction = mean of the 5 fold models (no refit on all rows)
      -> src/runlog.py         one row in results/runs.csv, oof/<run_id>.parquet, preds/<run_id>.parquet
      -> src/blend.py          weighted mean of the log-odds of logged runs, logged as a new run
      -> src/submit.py         submissions/<run_id>.csv, checked against data/sample_submission.csv

An experiment is one YAML file in configs/experiments/. configs/base.yaml lists every setting
with its default; an experiment may only override keys that exist there (src/config.py refuses
any other key). A run id is the UTC start time plus the first 8 characters of the hash of the
full merged config, for example 20261031T220901Z_1ff742f2; the report cites runs by those last
8 characters.

## Data

The data is not in the repository. Download it from the competition page,
https://www.kaggle.com/competitions/demandes-de-pret-immobilier-aux-etats-unis (Data tab),
or with the Kaggle command-line tool (not part of this environment) and an API token:

    kaggle competitions download -c demandes-de-pret-immobilier-aux-etats-unis

Unzip it, put train.csv and test.csv in data/raw/ and sample_submission.csv in data/.

To keep the data in another folder, change one line: line 16 of src/data.py,
DATA_DIR = ROOT / "data". That folder must hold raw/train.csv, raw/test.csv and
sample_submission.csv; every script takes the data paths from there, and make folds writes
processed/folds.parquet into it. Then:

    make check-data    # SHA256 of the 3 files against configs/data_checksums.txt
    make folds         # the 5 frozen folds -> data/processed/folds.parquet

The folds are rebuilt from the data and the seed in configs/base.yaml (42): the same split as
every logged run (its hash, 3762a22f..., is the folds_hash column of results/runs.csv).

## Reproducing the two selected submissions (report section 10.2)

| Kaggle submission | model | config | logged run | CV AUC | public LB |
|---|---|---|---|---|---|
| 6 | LightGBM 2/6, CatBoost 2/6, ExtraTrees 1/6, XGBoost 1/6, log-odds | configs/blends/lgbm_cat_et_xgb.yaml | 1ff742f2 | 0.92054 | 0.92158 |
| 5 | LightGBM 1/2, CatBoost 1/2, log-odds | configs/blends/lgbm_catboost_logit.yaml | 918eb62b | 0.92014 | 0.9212 |

Training and prediction: each member is trained by "make cv" (5 fold models; the test prediction
is their mean), then each blend combines its members' predictions and "make submit" writes the
submission file. Both blends share the LightGBM and CatBoost runs. About 2.5 hours on a laptop
CPU (8 logical CPUs, no GPU), CatBoost being most of it.

    make setup
    make check-data
    make folds
    make final        # both submissions: submissions/<run_id>.csv, one file per blend

"make final" runs these steps, which can also be run one by one:

    make cv EXP=configs/experiments/lgbm_optuna_best.yaml
    make cv EXP=configs/experiments/catboost_county_ctr4.yaml
    make cv EXP=configs/experiments/extratrees_optuna_best.yaml
    make cv EXP=configs/experiments/xgb_optuna_best.yaml
    make blend BLEND=configs/blends/lgbm_cat_et_xgb.yaml
    make submit RUN_ID=<run id printed by the blend>      # submission 6
    make blend BLEND=configs/blends/lgbm_catboost_logit.yaml
    make submit RUN_ID=<run id printed by the blend>      # submission 5

A blend file names its members by experiment file; the blend uses the latest logged run of
each, and refuses a run whose config differs from what the file gives now. The blend run logs
the member run ids it used.

The CatBoost config is catboost_county_ctr4.yaml: the same model as the logged CatBoost member
(catboost_county.yaml, run d96c8be4) with max_ctr_complexity 4 written out; its re-run 0810e0b4
gave identical predictions (report section 10.2).

"make figures" rebuilds the report's figures and generated tables. It reads the predictions of
the runs named by id in reports/figs/make_figures.py, so it needs those runs' files in oof/.

Every command runs through "uv run --locked", which stops instead of changing uv.lock.
"make help" lists the other targets: make compare (paired gain of one run over another, with the
decision rule), make ci (bootstrap interval), make fresh / fresh-blend / fresh-blends (re-check on
new fold splits), make tune and make optuna (setting searches), make test and make smoke.

## Files

"Pipeline" = needed to build the final model and its submission. "Evidence" = produces a result
the report quotes, but is not part of the final model.

| file | role |
|---|---|
| src/config.py | pipeline: base.yaml + experiment file -> merged config |
| src/check_data.py | pipeline: SHA256 check of the raw files (section 10.2) |
| src/data.py | pipeline: typed loading of the CSVs; DATA_DIR (line 16) sets the data folder (section 3) |
| src/features.py | pipeline: the named feature transforms and column lists (sections 3, 6.1) |
| src/target_encoding.py | pipeline code path: out-of-fold county and tract accept rates (section 6.1, ablation A4); the final model uses none |
| src/folds.py | pipeline: the frozen folds (section 5.2) |
| src/cv.py | pipeline: the cross-validation loop, logging of the run (section 5) |
| src/models.py | pipeline: the models, one class per "model:" value (section 6.2) |
| src/runlog.py | pipeline: logging of every run |
| src/blend.py | pipeline: fixed-weight blends of logged runs (section 6.4) |
| src/submit.py | pipeline: Kaggle submission file |
| src/compare.py | evidence for section 5.3: paired gain, corrected SE, bootstrap CI and the adopt / reject rule behind every gain in sections 6, 7 and 9 |
| src/bootstrap_ci.py | evidence for T1: interval on the pooled out-of-fold AUC |
| src/fresh_folds.py | evidence for sections 5.3, 6.3 and ablation A6b: re-checks on 3 fresh fold splits |
| src/tune.py | evidence for section 6.3: one-setting-at-a-time screening of LightGBM |
| src/tune_optuna.py | evidence for section 6.3: Optuna searches |
| src/mlp.py | evidence for section 6.2: the MLP (run by src/cv.py with model: mlp) |
| src/select_blend.py, src/weight_blend.py, src/stack.py, src/hill_climb.py | evidence for section 6.4: the four ways of choosing blend weights |
| src/leakage_scan.py | evidence for section 4: levels with an accept rate below 2% or above 98% |
| src/proxy_check.py | evidence for section 9.3: how well the kept columns predict race |
| reports/figs/make_figures.py | evidence: figures F1 to F9 and tables T2, T3, T5, T6, T10, T11, from logged runs |
| notebooks/01_eda.ipynb, 02_eda_multivariate.ipynb | evidence for section 3 and the feature plan of section 6.1 |
| notebooks/03_error_analysis.ipynb | evidence for section 8.3 |
| notebooks/04_fairness.ipynb | evidence for section 9 |
| tests/ | "make test": target-encoding noise test and train / test feature-path test (needs the data); "make smoke": folds and make final on 2,000 rows in a temporary copy (~1.5 min) |
| configs/base.yaml | every setting and its default, with the reason for each value |
| configs/experiments/ | one file per CV experiment the report cites: the 4 final members, the feature steps (T7), models (T8), ablations (T9), fair models (section 9) |
| configs/blends/ | lgbm_cat_et_xgb.yaml (final blend); the other files are the blend checks of section 6.4 and the fresh-split check of T1 |
| configs/sweeps/ | the searches of section 6.3 |

Configs of runs the report does not cite were removed; the full config of every run stays in
results/runs.csv, and the files are on branch volodya.

## Where results live

- reports/report.md: the report. The notes it cites for details: reports/validation.md (folds,
  rule), feature_plan.md, leakage.md, models.md, tuning.md, fairness.md, submissions.md,
  refs/hmda_dictionary.md (code meanings).
- results/runs.csv: one row per logged run, with its full merged config, commit, per-fold AUCs
  and the paths of its predictions. Every run id in the report is a row here. It is the only
  output kept in the repository.

Everything below is generated, not tracked, and rebuilt by a command:

- oof/<run_id>.parquet and preds/<run_id>.parquet: out-of-fold and test predictions of each run
  (make cv, make blend).
- submissions/<run_id>.csv: Kaggle submission files (make submit); the list of submissions and
  their scores is in reports/submissions.md.
- reports/tuning/: trial tables of the searches (make tune, make optuna), fresh-split results
  (make fresh, fresh-blend, fresh-blends), proxy check (src/proxy_check.py).
- reports/coefs/: coefficients of the logistic regression fold models (make cv on a logreg config).
- reports/figs/: figures and generated tables (make figures).
