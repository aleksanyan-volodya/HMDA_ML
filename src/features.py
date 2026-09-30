"""Feature pipeline: named transforms, applied identically to train and test.

An experiment lists the transforms it uses ("features: [...]" in its config), so a block of
reports/feature_plan.md can be tested as a whole and, if it fails, split into its items with
no code change. Numbers in brackets are the plan's rows.

Every transform here is row-wise and uses no labels, so applying it before the CV split
leaks nothing. (Target encoding, plan block 4, will need fold-aware fitting instead.)
"""
import numpy as np
import pandas as pd

from data import CODED, FEATURES

TRACT = ["tract_population", "tract_minority_population_percent", "tract_to_msa_income_percentage",
         "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
DTI_ORDER = ["<20%", "20%-<30%", "30%-<36%", *[str(i) for i in range(36, 50)], "50%-60%", ">60%"]
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]


def census_zero_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """[0.1] On location-unknown rows the whole census block is 0: a missing code, not a count."""
    unknown = df.tract_population == 0
    assert (df.loc[unknown, TRACT] == 0).all(axis=None), "census block not all-zero on location-unknown rows"
    df.loc[unknown, TRACT] = np.nan
    # MSA median income is 0 only on location-unknown rows (checked on train and test).
    df.loc[df.ffiec_msa_md_median_family_income == 0, "ffiec_msa_md_median_family_income"] = np.nan
    return df


def tract_income_zero_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """[0.1b] A populated tract can't have 0% of its area's income (583 train rows)."""
    df.loc[df.tract_to_msa_income_percentage == 0, "tract_to_msa_income_percentage"] = np.nan
    return df


def housing_age_negative_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """[0.2] A median housing age of -1 year is impossible (5 train rows). 0 is kept: off the
    location-unknown rows it is accepted normally (0.88)."""
    col = "tract_median_age_of_housing_units"
    df.loc[df[col] < 0, col] = np.nan
    return df


def multifamily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """[0.3] total_units and conforming_loan_limit are missing on the same rows, all Multifamily.
    total_units keeps its order with 5 meaning "5 or more" (it only takes 1-4 otherwise);
    the loan limit gets an explicit level instead of NaN."""
    assert (df.total_units.isna() == df.conforming_loan_limit.isna()).all()
    df["total_units"] = df.total_units.fillna(5)
    df["conforming_loan_limit"] = (df.conforming_loan_limit.cat.add_categories("not applicable")
                                   .fillna("not applicable"))
    return df


def _rank(s: pd.Series, order: list[str]) -> pd.Series:
    rank = s.astype("string").map({v: i for i, v in enumerate(order)})
    assert rank.notna().sum() == s.notna().sum(), f"unexpected level in {s.name}"
    return rank.astype("float64")


def dti_order(df: pd.DataFrame) -> pd.DataFrame:
    """[0.5] DTI mixes ranges and exact integers, and as text its levels sort wrongly ("<20%"
    after "50%-60%"). Trees only use order, so the rank (0 = "<20%" ... 18 = ">60%") keeps all
    the information, without inventing midpoints for the open-ended "<20%" and ">60%"."""
    df["dti_order"] = _rank(df.debt_to_income_ratio, DTI_ORDER)
    return df


def age_order(df: pd.DataFrame) -> pd.DataFrame:
    """[0.6] Same for the age bands (0 = "<25" ... 6 = ">74"). Missing stays NaN; telling
    "no co-applicant" from "unknown" is flag 1.4 of block 1."""
    df["applicant_age_order"] = _rank(df.applicant_age, AGE_ORDER)
    df["co-applicant_age_order"] = _rank(df["co-applicant_age"], AGE_ORDER)
    return df


def exempt_code_count(df: pd.DataFrame) -> pd.DataFrame:
    """[1.1] 1111 is one row-level fact (2,942 train rows carry it in all 14 coded columns), not
    14 separate ones. The count per row states it once. The yes/no flag is count >= 1, one split
    for a tree, so only the linear model will get it separately."""
    df["exempt_code_count"] = (df[CODED].astype("string") == "1111").sum(axis=1).astype("int64")
    return df


def entity_applicant(df: pd.DataFrame) -> pd.DataFrame:
    """[1.2] applicant_sex = 4 ("not applicable") = the applicant is not a natural person: 2,910
    rows with no person-level fields. Trees also see it in applicant_sex, but that column is a
    protected attribute and leaves the fair model; this flag stays."""
    df["entity_applicant"] = (df.applicant_sex.astype("string") == "4").astype("int64")
    return df


def location_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """[1.3] The census tract is unknown: tract_population is 0 in the raw data, NaN after
    census_zero_to_nan (block 0), so both are tested and the flag is the same either way."""
    df["location_unknown"] = (df.tract_population.isna() | (df.tract_population == 0)).astype("int64")
    return df


def has_coapplicant(df: pd.DataFrame) -> pd.DataFrame:
    """[1.4] co-applicant_sex = 5 is "no co-applicant", the same ~57.5k rows in every co-applicant
    column. The flag keeps that signal when the fair model drops the demographic columns
    (its own fairness status, as a proxy for joint / married applications, is checked in 12)."""
    df["has_coapplicant"] = (df["co-applicant_sex"].astype("string") != "5").astype("int64")
    return df


def income_le_0(df: pd.DataFrame) -> pd.DataFrame:
    """[1.5] Income 0 is accepted 0.58, negative 0 of 28: a state of the file, not a low income.
    Missing income stays NaN (flag = 0)."""
    df["income_le_0"] = (df.income <= 0).astype("int64")
    return df


# name -> (transform or None, columns it removes from the feature list, columns it adds)
TRANSFORMS = {
    "census_zero_to_nan": (census_zero_to_nan, [], []),
    "tract_income_zero_to_nan": (tract_income_zero_to_nan, [], []),
    "housing_age_negative_to_nan": (housing_age_negative_to_nan, [], []),
    "multifamily_levels": (multifamily_levels, [], []),
    # [0.4] exact functions of loan_type x lien_status and construction x units: duplicates only split importance
    "drop_derived_duplicates": (None, ["derived_loan_product_type", "derived_dwelling_category"], []),
    "dti_order": (dti_order, ["debt_to_income_ratio"], ["dti_order"]),
    "age_order": (age_order, ["applicant_age", "co-applicant_age"], ["applicant_age_order", "co-applicant_age_order"]),
    "exempt_code_count": (exempt_code_count, [], ["exempt_code_count"]),
    "entity_applicant": (entity_applicant, [], ["entity_applicant"]),
    "location_unknown": (location_unknown, [], ["location_unknown"]),
    "has_coapplicant": (has_coapplicant, [], ["has_coapplicant"]),
    "income_le_0": (income_le_0, [], ["income_le_0"]),
}


def build(train: pd.DataFrame, test: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply the named transforms in order; return (train, test, feature list)."""
    unknown = [n for n in names if n not in TRANSFORMS]
    if unknown:
        raise KeyError(f"unknown feature transforms {unknown}; available: {sorted(TRANSFORMS)}")
    train, test = train.copy(), test.copy()
    features = list(FEATURES)
    for name in names:
        fn, removed, added = TRANSFORMS[name]
        if fn is not None:
            train, test = fn(train), fn(test)
        features = [c for c in features if c not in removed] + added
    return train, test, features
