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


def computed_ltv(df: pd.DataFrame) -> pd.DataFrame:
    """[2.1, 2.2] The reported LTV is HMDA's combined LTV: all debt secured by the property / its
    value. This loan alone / property value is a different quantity, in the same unit (%).
    Their gap is the share of the property's value already pledged to other liens.
    NaN when a part is missing (loan_amount and property_value are never <= 0)."""
    df["computed_ltv"] = 100 * df.loan_amount / df.property_value
    df["ltv_gap"] = df.loan_to_value_ratio - df.computed_ltv
    return df


def property_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """[2.3] Price of the property in years of income. income is in $ thousands, hence x 1000.
    NaN when income is missing or <= 0 (a ratio to a non-positive income has no meaning)."""
    income_dollars = (df.income * 1000).where(df.income > 0)
    df["property_to_income"] = df.property_value / income_dollars
    return df


def _is(df: pd.DataFrame, col: str, *codes: str) -> pd.Series:
    return df[col].astype("string").isin(list(codes)).fillna(False).astype(bool)


def manufactured_conventional(df: pd.DataFrame) -> pd.DataFrame:
    """[3.1] Manufactured homes are denied almost only as conventional loans: 0.31 on 8,487 rows,
    against 0.84-0.86 as FHA / VA (notebook 02, 4d)."""
    df["manufactured_conventional"] = (_is(df, "construction_method", "2") & _is(df, "loan_type", "1")).astype("int64")
    return df


def manufactured_location_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """[3.2] Manufactured home with an unknown census tract: 1,516 rows accepted 0.016.
    Neither part explains it alone (0.44 and 0.56). A leakage suspect, see reports/leakage.md."""
    unknown = df.tract_population.isna() | (df.tract_population == 0)
    df["manufactured_location_unknown"] = (_is(df, "construction_method", "2") & unknown).astype("int64")
    return df


def dti_above_program_limit(df: pd.DataFrame) -> pd.DataFrame:
    """[3.3] The DTI a lender tolerates depends on the loan program: at 50-60%, conventional loans
    are accepted 0.33 against 0.89 (FHA) and 0.92 (VA). Limit = 50% for conventional and USDA,
    60% for FHA and VA. These limits were read on the training data (notebook 02, 4d)."""
    high_limit = _is(df, "loan_type", "2", "3")                                   # FHA, VA
    over_60, over_50 = _is(df, "debt_to_income_ratio", ">60%"), _is(df, "debt_to_income_ratio", "50%-60%", ">60%")
    df["dti_above_program_limit"] = np.where(high_limit, over_60, over_50).astype("int64")
    return df


def ltv_at_standard_value(df: pd.DataFrame) -> pd.DataFrame:
    """[3.4] An LTV exactly at a program standard (80, 90, 95, 97, 100) is accepted ~0.05 more than
    its neighbours within 2 points (notebook 02, 3b). 96.5 is left out: no difference there."""
    df["ltv_at_standard_value"] = df.loan_to_value_ratio.isin([80, 90, 95, 97, 100]).astype("int64")
    return df


def error_hypotheses(df: pd.DataFrame) -> pd.DataFrame:
    """[9.2] H1-H3 from the worst errors of the final blend (report 8.3, notebook 03), tested as one
    block for the report only, not for the blend. NaN where an input is missing, so no new missing code.
    H1 term not a multiple of 12 months; H2 combined LTV blank while the value is reported;
    H3 loan amount above property value (computed LTV > 100)."""
    df["term_off_grid"] = (df.loan_term % 12 != 0).astype("float").where(df.loan_term.notna())
    df["ltv_blank_value_present"] = (df.loan_to_value_ratio.isna() & df.property_value.notna()).astype("int64")
    df["computed_ltv_above_100"] = (df.loan_amount > df.property_value).astype("float").where(df.property_value.notna())
    return df


# ---------------------------------------------------------------- linear model only
# A linear model can't build bands or crossings by itself, so they are given explicitly.
# Every band includes "missing" as a level, so missing values keep their own effect.

DTI_BAND = {"<20%": "<20", "20%-<30%": "20-30", "30%-<36%": "30-36", "50%-60%": "50-60", ">60%": ">60",
            **{str(i): "36-49" for i in range(36, 50)}}      # 36-49 merged: flat in notebook 01 (C3)
DTI_LEVELS = ["<20", "20-30", "30-36", "36-49", "50-60", ">60", "missing"]
# Underwriting thresholds: 80 = no mortgage insurance needed, then 90, 95, 100 (notebook 02, 4d).
LTV_EDGES = [0, 80, 90, 95, 100, np.inf]
LTV_LEVELS = ["<=80", "80-90", "90-95", "95-100", ">100", "missing"]
# Doubling bands = equal width on the log scale (notebook 02, 4d).
LOAN_EDGES = [0, 25e3, 50e3, 100e3, 200e3, 400e3, np.inf]
LOAN_LEVELS = ["<=25k", "25-50k", "50-100k", "100-200k", "200-400k", ">400k"]
TERMS = ["360", "300", "240", "180", "120", "other", "missing"]


def _cat(values: pd.Series, levels: list[str]) -> pd.Series:
    """Fixed level list, so train and test always get identical categories."""
    assert set(values.dropna().unique()) <= set(levels), f"unexpected levels {set(values.unique()) - set(levels)}"
    return pd.Series(pd.Categorical(values, categories=levels), index=values.index)


def _band(x: pd.Series, edges: list[float], levels: list[str]) -> pd.Series:
    b = pd.cut(x, edges, labels=levels[: len(edges) - 1], right=True).astype("string")
    return b.fillna("missing") if "missing" in levels else b


def _cross(a: pd.Series, b: pd.Series, a_levels: list[str], b_levels: list[str]) -> pd.Series:
    joined = a.astype("string").fillna("missing") + " x " + b.astype("string").fillna("missing")
    return _cat(joined, [f"{x} x {y}" for x in a_levels for y in b_levels])


def linear_bands(df: pd.DataFrame) -> pd.DataFrame:
    """[0.5, 0.7, LTV bins] Bands a spline can't express: DTI is a set of buckets (with a U-shape),
    loan term lives in exact values (tree 14.5 vs linear <= 7.2 in notebook 02, 3b), and LTV
    has a cliff above 100 at underwriting thresholds."""
    df["dti_band"] = _cat(df.debt_to_income_ratio.astype("string").map(DTI_BAND).fillna("missing"), DTI_LEVELS)
    term = df.loan_term.map(lambda v: "missing" if pd.isna(v) else f"{v:.0f}" if f"{v:.0f}" in TERMS else "other")
    df["loan_term_band"] = _cat(term.astype("string"), TERMS)
    df["ltv_band"] = _cat(_band(df.loan_to_value_ratio, LTV_EDGES, LTV_LEVELS), LTV_LEVELS)
    return df


def linear_crossings(df: pd.DataFrame) -> pd.DataFrame:
    """[linear crossings] The four strongest pairs of notebook 02 (4c, 4d), all >= 1.3 points of
    deviance beyond their separate effects. Needs linear_bands first."""
    lt, cm, ls = (df[c].astype("string") for c in ("loan_type", "construction_method", "lien_status"))
    loan_band = _band(df.loan_amount, LOAN_EDGES, LOAN_LEVELS)
    df["x_dti_loan_type"] = _cross(df.dti_band, lt, DTI_LEVELS, ["1", "2", "3", "4"])
    df["x_dti_ltv"] = _cross(df.dti_band, df.ltv_band, DTI_LEVELS, LTV_LEVELS)
    df["x_construction_loan_type"] = _cross(cm, lt, ["1", "2"], ["1", "2", "3", "4"])
    df["x_lien_loan_amount"] = _cross(ls, loan_band, ["1", "2"], LOAN_LEVELS)
    return df


def tract_key(df: pd.DataFrame) -> pd.DataFrame:
    """Census tract id rebuilt from the county + the 6 tract fields (no tract id in the data): 50,587
    distinct keys in train, 76% of train rows share theirs with another row, 76% of test rows have
    a tract seen in train. Not a feature itself (50k levels): only its target encoding
    (target_encode, parent county) reaches the model. Unknown location (no county, or tract
    population 0 / NaN) -> missing key, so those rows keep their own signal instead of one shared key."""
    unknown = df.county_code.isna() | df.tract_population.isna() | (df.tract_population == 0)
    key = df.county_code.astype("string")
    for c in TRACT:
        key = key + "|" + df[c].astype("string").fillna("na")
    df["tract_key"] = key.mask(unknown.to_numpy())
    return df


def linear_extra(df: pd.DataFrame) -> pd.DataFrame:
    """[1.1 flag, 2.4] For the linear model only: the exempt-record flag (a tree reads it from the
    count) and loan / income (+0.011 AUC for the linear model in notebook 02, 4b; fails for trees)."""
    df["is_exempt_record"] = (df[CODED].astype("string") == "1111").any(axis=1).astype("int64")
    income_dollars = (df.income * 1000).where(df.income > 0)
    df["loan_to_income"] = df.loan_amount / income_dollars
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
    "computed_ltv": (computed_ltv, [], ["computed_ltv", "ltv_gap"]),
    "property_to_income": (property_to_income, [], ["property_to_income"]),
    "manufactured_conventional": (manufactured_conventional, [], ["manufactured_conventional"]),
    "manufactured_location_unknown": (manufactured_location_unknown, [], ["manufactured_location_unknown"]),
    "dti_above_program_limit": (dti_above_program_limit, [], ["dti_above_program_limit"]),
    "ltv_at_standard_value": (ltv_at_standard_value, [], ["ltv_at_standard_value"]),
    "error_hypotheses": (error_hypotheses, [], ["term_off_grid", "ltv_blank_value_present", "computed_ltv_above_100"]),
    # linear model only: bands replace the raw DTI text, the numeric term and the numeric LTV
    "linear_bands": (linear_bands, ["debt_to_income_ratio", "loan_term", "loan_to_value_ratio"],
                     ["dti_band", "loan_term_band", "ltv_band"]),
    "linear_crossings": (linear_crossings, [], ["x_dti_loan_type", "x_dti_ltv", "x_construction_loan_type",
                                                "x_lien_loan_amount"]),
    "linear_extra": (linear_extra, [], ["is_exempt_record", "loan_to_income"]),
    #Remove the two high-cardinality geography ids (2,898 and 418 levels).
    # state_code and the census columns stay, so coarse geography is still available.
    "drop_county_msa_ids": (None, ["county_code", "derived_msa-md"], []),
    # adds the tract_key column for target_encode only, not to the feature list (50k levels)
    "tract_key": (tract_key, [], []),
    # confirmed leaks, each value is reported because of the decision (reports/leakage.md)
    "drop_process_artefacts": (None, ["preapproval", "initially_payable_to_institution"], []),
    # strict list for the 12.9 sensitivity check only (reports/leakage.md); must come LAST in a config
    "drop_partial_leaks": (None, ["aus-1", "loan_to_value_ratio", "property_value", "debt_to_income_ratio",
                                  "dti_order", "computed_ltv", "ltv_gap", "property_to_income",
                                  "tract_population", "tract_minority_population_percent",
                                  "tract_to_msa_income_percentage", "tract_owner_occupied_units",
                                  "tract_one_to_four_family_homes", "tract_median_age_of_housing_units",
                                  "ffiec_msa_md_median_family_income", "state_code", "county_code",
                                  "derived_msa-md", "location_unknown", "manufactured_location_unknown"], []),
}


# Expert models (TODO 7.14-7.16, 7.23, [N11]): the 55 adopted columns split by meaning, each block a
# model of its own. A restricted model is worse alone but its errors differ, which is what a blend needs.
# only_<block> keeps one block: it must come LAST in a config, after every transform that adds columns.
BLOCKS = {
    "finance": ["loan_amount", "income", "property_value", "loan_to_value_ratio", "loan_term", "dti_order",
                "computed_ltv", "ltv_gap", "property_to_income", "conforming_loan_limit",
                "applicant_credit_score_type", "co-applicant_credit_score_type"],
    "geography": ["state_code", "ffiec_msa_md_median_family_income", *TRACT],
    "process": ["loan_type", "lien_status", "preapproval", "reverse_mortgage", "open-end_line_of_credit",
                "business_or_commercial_purpose", "negative_amortization", "interest_only_payment",
                "balloon_payment", "other_nonamortizing_features", "construction_method", "occupancy_type",
                "manufactured_home_secured_property_type", "manufactured_home_land_property_interest",
                "submission_of_application", "initially_payable_to_institution", "aus-1", "total_units"],
    # 7.16: a fairness result first ([N12]), not a blend candidate unless decided in writing in 12.8
    "demographics": ["applicant_ethnicity-1", "co-applicant_ethnicity-1", "applicant_ethnicity_observed",
                     "co-applicant_ethnicity_observed", "applicant_race-1", "co-applicant_race-1",
                     "applicant_race_observed", "co-applicant_race_observed", "applicant_sex", "co-applicant_sex",
                     "applicant_sex_observed", "co-applicant_sex_observed", "derived_ethnicity", "derived_race",
                     "derived_sex", "applicant_age_order", "co-applicant_age_order"],
}
_ALL = set(FEATURES) | {c for _, _, added in TRANSFORMS.values() for c in added}
for _name, _keep in BLOCKS.items():
    TRANSFORMS[f"only_{_name}"] = (None, sorted(_ALL - set(_keep)), [])
# Fair model, unawareness (fair_unaware.yaml): the 17 demographic fields out; must come after age_order.
TRANSFORMS["drop_demographics"] = (None, BLOCKS["demographics"], [])
# Pre-registered ablations A2 and A3 (report T9), lists as registered on 2026-10-18
TRANSFORMS["drop_process_fields"] = (None, ["preapproval", "submission_of_application",
                                            "initially_payable_to_institution", "aus-1"], [])
TRANSFORMS["drop_census"] = (None, [*TRACT, "ffiec_msa_md_median_family_income"], [])


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
