# Entry points for the project. Run `make` (or `make help`) to list them.
# Every command goes through `uv run --locked`: it fails instead of silently
# updating uv.lock, so everyone runs exactly the pinned library versions.

RUN := uv run --locked

.DEFAULT_GOAL := help
.PHONY: help setup check-data folds final cv tune optuna fresh fresh-blend fresh-blends blend compare ci submit test smoke figures

# The run id of the last run logged, i.e. the blend just run (first column of the last line of the log).
LAST_RUN = $$(tail -n 1 results/runs.csv | cut -d, -f1)

# "## " marks a pipeline target, "##e " an evidence target; help lists them in two groups.
help:  ## List available targets
	@echo "Pipeline (data -> folds -> CV -> blend -> submission):"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'
	@echo "Evidence for the report (not part of the final model):"
	@grep -E '^[a-zA-Z_-]+:.*##e ' $(MAKEFILE_LIST) | awk -F':.*##e ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install the exact environment from uv.lock
	uv sync --locked

check-data:  ## Verify data/raw files against configs/data_checksums.txt
	$(RUN) python src/check_data.py

folds:  ## Build the frozen CV folds (refuses to change an existing split)
	$(RUN) python src/folds.py

final:  ## Rebuild both selected submissions (6 and 5): 4 CV runs, 2 blends, 2 files in submissions/ (~2.5 h)
	$(MAKE) cv EXP=configs/experiments/lgbm_optuna_best.yaml
	$(MAKE) cv EXP=configs/experiments/catboost_county_ctr4.yaml
	$(MAKE) cv EXP=configs/experiments/extratrees_optuna_best.yaml
	$(MAKE) cv EXP=configs/experiments/xgb_optuna_best.yaml
	$(MAKE) blend BLEND=configs/blends/lgbm_cat_et_xgb.yaml
	$(MAKE) submit RUN_ID=$(LAST_RUN)
	$(MAKE) blend BLEND=configs/blends/lgbm_catboost_logit.yaml
	$(MAKE) submit RUN_ID=$(LAST_RUN)

cv:  ## Run and log one experiment: make cv EXP=configs/experiments/<name>.yaml
	@test -n "$(EXP)" || { echo "usage: make cv EXP=configs/experiments/<name>.yaml"; exit 1; }
	$(RUN) python src/cv.py $(EXP)

tune:  ##e Run a one-setting-at-a-time sweep: make tune SWEEP=configs/sweeps/<name>.yaml
	@test -n "$(SWEEP)" || { echo "usage: make tune SWEEP=configs/sweeps/<name>.yaml"; exit 1; }
	$(RUN) python src/tune.py $(SWEEP)

optuna:  ##e Joint search with Optuna (resumable): make optuna STUDY=configs/sweeps/<name>.yaml
	@test -n "$(STUDY)" || { echo "usage: make optuna STUDY=configs/sweeps/<name>.yaml"; exit 1; }
	$(RUN) python src/tune_optuna.py $(STUDY)

fresh:  ##e Re-check a candidate on fresh fold splits: make fresh REF=<exp.yaml> CAND=<exp.yaml>
	@test -n "$(REF)" -a -n "$(CAND)" || { echo "usage: make fresh REF=<experiment.yaml> CAND=<experiment.yaml>"; exit 1; }
	$(RUN) python src/fresh_folds.py $(REF) $(CAND)

fresh-blend:  ##e Check a 50/50 blend on fresh splits: make fresh-blend REF=<exp.yaml> WITH=<exp.yaml>
	@test -n "$(REF)" -a -n "$(WITH)" || { echo "usage: make fresh-blend REF=<experiment.yaml> WITH=<experiment.yaml>"; exit 1; }
	$(RUN) python src/fresh_folds.py $(REF) --blend-with $(WITH)

fresh-blends:  ##e Check weighted blends on fresh splits, resumable: make fresh-blends SPEC=configs/blends/<name>.yaml
	@test -n "$(SPEC)" || { echo "usage: make fresh-blends SPEC=configs/blends/<name>.yaml"; exit 1; }
	$(RUN) python src/fresh_folds.py --blends $(SPEC)

blend:  ## Blend logged runs into a new logged run: make blend BLEND=configs/blends/<name>.yaml
	@test -n "$(BLEND)" || { echo "usage: make blend BLEND=configs/blends/<name>.yaml"; exit 1; }
	$(RUN) python src/blend.py $(BLEND)

# NEW defaults to the last run logged in results/runs.csv (first column of the last line).
NEW ?= $(shell tail -n 1 results/runs.csv | cut -d, -f1)

compare:  ##e Paired gain over a reference run: make compare BASE=<run_id> [NEW=<run_id>, default: latest run]
	@test -n "$(BASE)" -a -n "$(NEW)" || { echo "usage: make compare BASE=<run_id> [NEW=<run_id>] (see results/runs.csv)"; exit 1; }
	$(RUN) python src/compare.py $(BASE) $(NEW)

ci:  ##e 95% bootstrap CI on the pooled OOF AUC: make ci RUNS="<run_id> [<run_id> ...]"
	@test -n "$(RUNS)" || { echo "usage: make ci RUNS=\"<run_id> [<run_id> ...]\""; exit 1; }
	$(RUN) python src/bootstrap_ci.py $(RUNS)

submit:  ## Build a validated submission: make submit RUN_ID=<run_id>
	@test -n "$(RUN_ID)" || { echo "usage: make submit RUN_ID=<run_id> (see results/runs.csv)"; exit 1; }
	$(RUN) python src/submit.py $(RUN_ID)

test:  ## Run the test suite (the feature-path test needs data/raw; it is skipped without it)
	$(RUN) pytest -q

smoke:  ## The whole pipeline (folds, make final) on 2,000 rows, in a temporary copy: ~1.5 min
	$(RUN) python tests/smoke.py

# Reads logged runs only; the "shap" step refits one LightGBM fold (~1.5 min). ~3 min in all.
figures:  ##e Rebuild report figures and tables (reports/figs/): make figures [STEPS="slices shap"]
	$(RUN) python reports/figs/make_figures.py $(STEPS)
