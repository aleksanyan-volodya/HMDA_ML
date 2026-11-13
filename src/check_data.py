"""Check the raw data files against the SHA256 sums in configs/data_checksums.txt.

Usage:
    uv run python src/check_data.py           # verify, exit 1 on any mismatch
    uv run python src/check_data.py --write   # record the reference hashes

Reads data/raw/train.csv, data/raw/test.csv, data/sample_submission.csv.
Pipeline (make check-data). Report: section 10.2.
"""
import argparse
import hashlib
import sys
from pathlib import Path

from data import RAW, SAMPLE

ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = ROOT / "configs" / "data_checksums.txt"
# name in data_checksums.txt -> file read (in the data directory set in src/data.py)
FILES = {"data/raw/train.csv": RAW / "train.csv", "data/raw/test.csv": RAW / "test.csv",
         "data/sample_submission.csv": SAMPLE}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        # Stream in 1 MiB blocks so memory use does not grow with file size.
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="record current hashes as the reference")
    args = parser.parse_args()

    current = {name: sha256(path) for name, path in FILES.items()}

    if args.write:
        # Same "<hash>  <path>" format as sha256sum, so `sha256sum -c` also works (default data directory).
        CHECKSUMS.write_text("".join(f"{h}  {name}\n" for name, h in current.items()))
        print(f"wrote {CHECKSUMS.relative_to(ROOT)}")
        return 0

    expected = dict(reversed(line.split("  ", 1)) for line in CHECKSUMS.read_text().splitlines())
    bad = [name for name in FILES if expected.get(name) != current[name]]
    for name in FILES:
        print(f"{'MISMATCH' if name in bad else 'ok':8}  {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
