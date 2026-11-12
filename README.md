# HMDA mortgage outcomes: prediction and fairness

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
      -> src/runlog.py         one row in reports/runs.csv, oof/<run_id>.parquet, preds/<run_id>.parquet
      -> src/blend.py          weighted mean of the log-odds of logged runs, logged as a new run
      -> src/submit.py         submissions/<run_id>.csv, checked against data/sample_submission.csv

An experiment is one YAML file in configs/experiments/. configs/base.yaml lists every setting
with its default; an experiment may only override keys that exist there (src/config.py refuses
any other key). A run id is the UTC start time plus the first 8 characters of the hash of the
full merged config, for example 20261031T220901Z_1ff742f2; the report cites runs by those last
8 characters.

## Rebuilding the final model (report section 10.2)

The raw data is not distributed: put train.csv and test.csv in data/raw/ (data/sample_submission.csv
and the folds are in the repository). "make check-data" verifies the files against
configs/data_checksums.txt. About 2.5 hours on a laptop CPU (8 logical CPUs, no GPU), CatBoost being most of it.

    make setup
    make check-data
    make folds
    make cv EXP=configs/experiments/lgbm_optuna_best.yaml
    make cv EXP=configs/experiments/catboost_county_ctr4.yaml
    make cv EXP=configs/experiments/extratrees_optuna_best.yaml
    make cv EXP=configs/experiments/xgb_optuna_best.yaml
    (put the 4 new run ids, printed at the end of each run, in configs/blends/lgbm_cat_et_xgb.yaml)
    make blend BLEND=configs/blends/lgbm_cat_et_xgb.yaml
    make submit RUN_ID=<the blend's run id>
    make figures

Every command runs through "uv run --locked", which stops instead of changing uv.lock.
"make help" lists the other targets: make compare (paired gain of one run over another, with the
decision rule), make ci (bootstrap interval), make fresh / fresh-blend / fresh-blends (re-check on
new fold splits), make tune and make optuna (setting searches), make test.

## Files

"Pipeline" = needed to build the final model and its submission. "Evidence" = produces a result
the report quotes, but is not part of the final model.

| file | role |
|---|---|
| src/config.py | pipeline: base.yaml + experiment file -> merged config |
| src/check_data.py | pipeline: SHA256 check of the raw files (section 10.2) |
| src/data.py | pipeline: typed loading of the CSVs (section 3) |
| src/features.py | pipeline: the named feature transforms and column lists (sections 3, 6.1) |
| src/target_encoding.py | pipeline: out-of-fold county and tract accept rates (section 6.1, ablation A4) |
| src/folds.py | pipeline: the frozen folds (section 5.2) |
| src/cv.py | pipeline: the models and the cross-validation loop (sections 5, 6.2) |
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
| tests/ | "make test": tests of the pipeline code |
| configs/base.yaml | every setting and its default, with the reason for each value |
| configs/experiments/ | one file per CV experiment the report cites: the 4 final members, the feature steps (T7), models (T8), ablations (T9), fair models (section 9) |
| configs/blends/ | lgbm_cat_et_xgb.yaml (final blend); the other files are the blend checks of section 6.4 and the fresh-split check of T1 |
| configs/sweeps/ | the searches of section 6.3 |

Configs of runs the report does not cite were removed; the full config of every run stays in
reports/runs.csv, and the files are on branch volodya.

## Where results live

- reports/report.md: the report. The notes it cites for details: reports/validation.md (folds,
  rule), feature_plan.md, leakage.md, models.md, tuning.md, fairness.md, submissions.md,
  refs/hmda_dictionary.md (code meanings).
- reports/runs.csv: one row per logged run, with its full merged config, commit, per-fold AUCs
  and the paths of its predictions. Every run id in the report is a row here.
- oof/<run_id>.parquet and preds/<run_id>.parquet: out-of-fold and test predictions of each run
  (not tracked: rebuilt from the config in runs.csv).
- reports/tuning/: trial tables of the searches, fresh-split results, proxy check.
- reports/coefs/: coefficients of the logistic regression fold models.
- reports/figs/: figures and generated tables.
- submissions/: the files submitted to Kaggle, named by run id (list and scores in reports/submissions.md).
