# Entry points for the project. Run `make` (or `make help`) to list them.
# Every command goes through `uv run --locked`: it fails instead of silently
# updating uv.lock, so everyone runs exactly the pinned library versions.

RUN := uv run --locked

.DEFAULT_GOAL := help
.PHONY: help setup check-data folds cv tune compare submit test

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install the exact environment from uv.lock
	uv sync --locked

check-data:  ## Verify data/raw files against configs/data_checksums.txt
	$(RUN) python src/check_data.py

folds:  ## Build the frozen CV folds (refuses to change an existing split)
	$(RUN) python src/folds.py

cv:  ## Run and log one experiment: make cv EXP=configs/experiments/<name>.yaml
	@test -n "$(EXP)" || { echo "usage: make cv EXP=configs/experiments/<name>.yaml"; exit 1; }
	$(RUN) python src/cv.py $(EXP)

tune:  ## Run a one-setting-at-a-time sweep: make tune SWEEP=configs/sweeps/<name>.yaml
	@test -n "$(SWEEP)" || { echo "usage: make tune SWEEP=configs/sweeps/<name>.yaml"; exit 1; }
	$(RUN) python src/tune.py $(SWEEP)

# NEW defaults to the last run logged in reports/runs.csv (first column of the last line).
NEW ?= $(shell tail -n 1 reports/runs.csv | cut -d, -f1)

compare:  ## Paired gain over a reference run: make compare BASE=<run_id> [NEW=<run_id>, default: latest run]
	@test -n "$(BASE)" -a -n "$(NEW)" || { echo "usage: make compare BASE=<run_id> [NEW=<run_id>] (see reports/runs.csv)"; exit 1; }
	$(RUN) python src/compare.py $(BASE) $(NEW)

submit:  ## Build a validated submission: make submit RUN_ID=<run_id>
	@test -n "$(RUN_ID)" || { echo "usage: make submit RUN_ID=<run_id> (see reports/runs.csv)"; exit 1; }
	$(RUN) python src/submit.py $(RUN_ID)

test:  ## Run the test suite
	$(RUN) pytest -q
