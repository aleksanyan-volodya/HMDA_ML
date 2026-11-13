"""Smoke test: the whole pipeline on a small sample, in a temporary copy of the repository.

Usage:
    make smoke        (or: uv run --locked python tests/smoke.py)

Copies src/, configs/ and the Makefile to a temporary folder, writes there the first TRAIN_ROWS
training rows and the first TEST_ROWS test rows (with their sample_submission rows), sets the copy
to N_FOLDS folds, and runs, through the copy's own Makefile: make check-data (on the sample's own
checksums), make folds and make final. It then checks that 4 CV runs and 2 blends were logged,
that every fold scored above chance, and that both submission files were written.

Reads data/raw/ and data/sample_submission.csv; writes nothing in this repository.
Pipeline test, not part of the final model.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import islice
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import RAW, SAMPLE  # noqa: E402

TRAIN_ROWS = 2000   # the size asked for the smoke test: every model fits in seconds
TEST_ROWS = 1000
# The fewest folds the nested scheme allows: a fold model chooses its tree count by an inner CV over
# the outer folds in its fit part, so with 2 folds it would have a single inner fold to train on.
N_FOLDS = 3


def head_lines(src: Path, dst: Path, n_rows: int) -> None:
    """Copy the header and the first n_rows lines as they are, so the sample is parsed like the full file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as f, dst.open("w") as out:
        out.writelines(islice(f, n_rows + 1))


def main() -> int:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hmda_smoke_") as tmp:
        copy = Path(tmp)
        for part in ("src", "configs"):
            shutil.copytree(ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "Makefile", copy / "Makefile")
        (copy / "results").mkdir()

        head_lines(RAW / "train.csv", copy / "data" / "raw" / "train.csv", TRAIN_ROWS)
        head_lines(RAW / "test.csv", copy / "data" / "raw" / "test.csv", TEST_ROWS)
        test_ids = pd.read_csv(copy / "data" / "raw" / "test.csv", usecols=["id"]).id
        sample = pd.read_csv(SAMPLE)
        sample[sample.id.isin(test_ids)].to_csv(copy / "data" / "sample_submission.csv", index=False)

        base = copy / "configs" / "base.yaml"
        text = base.read_text()
        assert text.count("\nn_folds: 5\n") == 1, "n_folds line of base.yaml not found"
        base.write_text(text.replace("\nn_folds: 5\n", f"\nn_folds: {N_FOLDS}\n"))

        git = ["git", "-C", tmp, "-c", "user.name=smoke", "-c", "user.email=smoke@localhost"]
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "smoke copy"]):   # runs log the commit
            subprocess.run(git + args, check=True)

        # The copy runs with this repository's locked environment.
        make = ["make", "--no-print-directory", "-C", tmp, f"RUN=uv run --locked --project {ROOT}"]
        print(f"smoke copy in {tmp}: {TRAIN_ROWS} train rows, {TEST_ROWS} test rows, {N_FOLDS} folds", flush=True)
        # The sample's own checksums, so that make check-data runs on the copy as it does on the full data.
        subprocess.run(["uv", "run", "--locked", "--project", str(ROOT), "python", "src/check_data.py", "--write"],
                       cwd=tmp, check=True)
        subprocess.run(make + ["check-data", "folds", "final"], check=True)

        runs = pd.read_csv(copy / "results" / "runs.csv")
        aucs = {r.experiment: json.loads(r.fold_auc) for r in runs.itertuples()}
        subs = sorted((copy / "submissions").glob("*.csv"))
        print("\nlogged runs:")
        for exp, auc in aucs.items():
            print(f"  {exp:48s} fold AUC {[round(a, 3) for a in auc]}")
        assert len(runs) == 6, f"expected 4 CV runs + 2 blends, got {len(runs)}"
        assert all(a > 0.5 for auc in aucs.values() for a in auc), "a fold scored at or below chance"
        assert len(subs) == 2 and all(len(pd.read_csv(s)) == len(test_ids) for s in subs), "submission files missing"
    print(f"\nsmoke test passed in {time.perf_counter() - t0:.0f} s: 6 runs logged, 2 submission files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
