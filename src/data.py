"""Loading of the raw CSVs, with an explicit type for every column. Train and test get the
same category levels, so a category code means the same thing on both sides.

Reads data/raw/train.csv and data/raw/test.csv. Pipeline. Report: section 3.

DATA_DIR below is the only line to change to read the data from another folder; every script
takes the data paths from here.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Data directory. It must hold raw/train.csv, raw/test.csv and sample_submission.csv (from Kaggle);
# "make folds" writes processed/folds.parquet into it.
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw"
SAMPLE = DATA_DIR / "sample_submission.csv"
FOLDS = DATA_DIR / "processed" / "folds.parquet"
TARGET = "accepted"

# Real-valued quantities
NUMERIC = [
    "loan_amount", "loan_to_value_ratio", "property_value", "income", "loan_term",
    "total_units", "tract_population", "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]

# Integer codes and labels
CODED = [
    "loan_type", "lien_status", "preapproval", "reverse_mortgage", "open-end_line_of_credit",
    "business_or_commercial_purpose", "negative_amortization", "interest_only_payment",
    "balloon_payment", "other_nonamortizing_features", "construction_method", "occupancy_type",
    "manufactured_home_secured_property_type", "manufactured_home_land_property_interest",
    "applicant_credit_score_type", "co-applicant_credit_score_type",
    "applicant_ethnicity-1", "co-applicant_ethnicity-1",
    "applicant_ethnicity_observed", "co-applicant_ethnicity_observed",
    "applicant_race-1", "co-applicant_race-1", "applicant_race_observed", "co-applicant_race_observed",
    "applicant_sex", "co-applicant_sex", "applicant_sex_observed", "co-applicant_sex_observed",
    "submission_of_application", "initially_payable_to_institution", "aus-1",
]

# Geographic identifiers: labels, with a fixed width.
GEO_WIDTH = {"county_code": 5, "derived_msa-md": 5}

# Text labels. 
# DTI and ages stay text here; src/features.py turns them into ranks (dti_order, age_order).
TEXT = [
    "state_code", "conforming_loan_limit", "derived_loan_product_type", "derived_dwelling_category",
    "derived_ethnicity", "derived_race", "derived_sex", "debt_to_income_ratio",
    "applicant_age", "co-applicant_age",
]

# All features
FEATURES = NUMERIC + CODED + list(GEO_WIDTH) + TEXT


def _as_code(s: pd.Series) -> pd.Series:
    # "4013.0" -> 4013 -> "4013"
    return pd.to_numeric(s).astype("Int64").astype("string")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) with the same feature dtypes and category levels"""
    # Read everything as text first, so pandas don't guess a type.
    train = pd.read_csv(RAW / "train.csv", dtype="string", keep_default_na=False, na_values=[""])
    test = pd.read_csv(RAW / "test.csv", dtype="string", keep_default_na=False, na_values=[""])
    assert set(train.columns) == {"id", TARGET, *FEATURES}, "unexpected train columns"
    assert set(test.columns) == {"id", *FEATURES}, "unexpected test columns"

    for df in (train, test):
        df["id"] = df["id"].astype("int64")
        df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric).astype("float64")
        for c in CODED:
            df[c] = _as_code(df[c])
        for c, width in GEO_WIDTH.items():
            df[c] = _as_code(df[c]).str.zfill(width)
    train[TARGET] = train[TARGET].astype("int64")

    # Category levels = union of train and test.
    # Same levels on both sides => the same integer code means the same thing in both.
    for c in CODED + list(GEO_WIDTH) + TEXT:
        levels = pd.concat([train[c], test[c]]).dropna().unique()
        dtype = pd.CategoricalDtype(sorted(levels))
        train[c] = train[c].astype(dtype)
        test[c] = test[c].astype(dtype)

    assert (train[FEATURES].dtypes == test[FEATURES].dtypes).all(), "train/test dtypes differ"
    return train, test
