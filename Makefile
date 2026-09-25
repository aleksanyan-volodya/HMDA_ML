# Entry points for the project. Run `make` (or `make help`) to list them.
# Every command goes through `uv run --locked`: it fails instead of silently
# updating uv.lock, so everyone runs exactly the pinned library versions.

RUN := uv run --locked

.DEFAULT_GOAL := help
.PHONY: help setup check-data test

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install the exact environment from uv.lock
	uv sync --locked

check-data:  ## Verify data/raw files against configs/data_checksums.txt
	$(RUN) python src/check_data.py

test:  ## Run the test suite
	$(RUN) pytest -q
