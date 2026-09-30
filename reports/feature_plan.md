# Feature plan (wave 1)

Every feature-engineering decision taken in "notebooks/01_eda.ipynb" (01) and "notebooks/02_eda_multivariate.ipynb" (02), with where it comes from and which model gets it. This is the spec that "src/features.py" implements. In the "Source" column, "02 (3b)" means notebook 02, part 3b.

Three models are in view:

- Trees: LightGBM, the competition model. Takes raw columns + flags; transforms don't matter to it.
- Linear: the logistic regression baseline. Needs shapes and crossings given explicitly.
- Fair: the model without protected attributes and without process artefacts. Its exact column list waits for the leakage verdicts.

How each block is accepted: it is added to the current accepted feature set, the LightGBM CV is re-run on the frozen folds, and the block is kept if the paired mean gain over the 5 folds is at least 2 x its SE. A block that fails is split and its features are tested one by one; a feature that fails is deleted from the config, not kept "just in case".

## Block 0: representation (cleaning, no new information)

| # | Change | Why | Source | Trees | Linear |
|---|---|---|---|---|---|
| 0.1 | Census block -> NaN on the location-unknown rows: the 6 tract columns (2,117 rows) and "ffiec_msa_md_median_family_income" (1,330 of them) | the 6 tract columns are zero together on exactly these rows, and MSA income is zero exactly where the county is missing too (all MSA 99999); a zero there is a missing code, not a count. The row state is kept by flag 1.3 | 01 (H1), 02 (3a) | yes | yes |
| 0.1b | "tract_to_msa_income_percentage" = 0 on a known tract -> NaN | 583 more rows (206 in test): a populated tract can't have 0% of the area income; accepted 0.80 | this plan (check below) | yes | yes |
| 0.2 | "tract_median_age_of_housing_units": -1 -> NaN (5 rows); 0 kept, flag optional | outside the location-unknown rows, 0 is 4,390 rows accepted 0.88, i.e. normal. The 0.65 of 01 mixed in the location-unknown rows | 01 (H1), this plan | yes | CV decides |
| 0.3 | "total_units" missing -> level "5+"; "conforming_loan_limit" missing -> "not applicable" | the same 437 rows are labelled Multifamily | 02 (2) | yes | yes |
| 0.4 | Drop "derived_loan_product_type", "derived_dwelling_category" | exact functions of other columns | 02 (2) | yes | no (kept as ready-made crossings) |
| 0.5 | DTI: ordered band (36 to 49 merged) + numeric midpoint + "dti_is_exact" | mixed format; 36 to 49 is flat | 01 (C3) | yes | band one-hot |
| 0.6 | Ages: ordered category. Co-applicant age: "no co-applicant" where flag 1.4 says so, missing otherwise. Applicant age blank on entity rows (1.2) = "does not apply" | co-applicant age is blank on 58,693 rows, but 1,127 of them do have a co-applicant: blank != no co-applicant. 94% of the blank applicant ages are entity rows | 02 (2c), this plan | yes | yes |
| 0.7 | "loan_term" as category: 360, 300, 240, 180, 120, other, missing | signal sits in exact values (tree 14.5 vs linear <= 7.2) | 02 (3b) | keep numeric | yes |
| 0.8 | "derived_msa-md" = 99999 kept as its own level ("outside any MSA"), not NaN | 15,200 rows over 1,669 counties, accepted 0.75: rural areas, not missing. The location-unknown rows are also 99999, flag 1.3 separates them | 01 (G4) | yes | yes |

Check behind 0.1b and 0.2 (zeros outside the location-unknown rows):

| column = 0, tract known | rows | accept rate |
|---|---|---|
| "tract_to_msa_income_percentage" | 583 | 0.80 |
| "tract_median_age_of_housing_units" | 4,390 | 0.88 |
| "tract_owner_occupied_units" / "tract_one_to_four_family_homes" / minority % | 31 / 42 / 7 | too few to read; left as values |

## Block 1: row states and missingness

| # | Feature | Why | Source | Trees | Linear |
|---|---|---|---|---|---|
| 1.1 | "is_exempt_record" + count of 1111 per row | 2,942 rows carry 1111 in all 14 columns: a row-level fact | 01 (J1) | yes | yes |
| 1.2 | "is_entity_applicant" (applicant_sex = 4) | 2,910 rows with no person-level fields; explains most missing age / income | 02 (2c) | yes | yes |
| 1.3 | "location_unknown" | 2,117 rows (929 in test) accepted 0.17; drives the census zeros, MSA 99999 and no-AUS overlap. Mostly a manufactured-home effect: 0.016 on manufactured homes (flag 3.2), 0.56 on site-built | 01 (G3), 02 (5c) | yes | yes |
| 1.4 | "has_coapplicant" = not ("co-applicant_sex" = 5) | the whole signal of the co-applicant columns (~0.55 AUC each); "no co-applicant" is the same ~57.5k rows in sex, race, ethnicity and observed; survives removal of demographics | 02 (1) | yes | yes |
| 1.5 | "income_le_0" | income 0 -> accepted 0.58, negative -> 0 of 28 | 01 (B5) | yes | yes |
| 1.6 | Missing indicators: LTV; property-block (property value, term, LTV missing together); income missing outside entity rows | missing LTV -> accepted 0.54; different kinds of missing kept apart | 01 (J2), 02 (2c) | NaN is enough | yes |

Definition of 1.3: "location_unknown" = the census block is zero ("tract_population" = 0, identical to ">= 6 census zeros" and to "all 6 tract columns zero" on every row), 2,117 train rows. 01 used "state or county missing" instead; the two disagree on 818 rows, and the data decides:

| rows | n | accept rate |
|---|---|---|
| census zero, state and county known | 781 | 0.054 |
| census zero, state or county missing | 1,336 | 0.240 |
| state or county missing, census present | 37 | 0.892 |

The signal is "the census tract is unknown", whether or not the county is known. Missing state / county alone behaves normally and stays as NaN in those columns. (With county alone instead of "state or county": 787 / 1,330 / 9 rows, same conclusion.)

## Block 2: ratios

| # | Feature | Why | Source | Trees | Linear |
|---|---|---|---|---|---|
| 2.1 | Computed LTV = loan / property | passes the 2 SE rule for gbm (+0.0023) and linear (+0.017) | 02 (4b) | yes | yes |
| 2.2 | Reported LTV - computed LTV | reported is probably combined LTV; the gap ~ other liens; disagreeing rows accepted 0.81 vs 0.90 | 01 (C1) | yes | yes |
| 2.3 | Property / income | gbm +0.0014, linear +0.010 | 02 (4b) | yes | yes |
| 2.4 | Loan / income | linear +0.011; gbm +0.0003 (fails) | 02 (4b) | no | yes |

## Block 3: segment and rule flags

| # | Feature | Why | Source | Trees | Linear |
|---|---|---|---|---|---|
| 3.1 | "manufactured_conventional" | 8,487 rows accepted 0.31 vs 0.84 to 0.86 for FHA / VA manufactured | 02 (4d) | yes | yes |
| 3.2 | "manufactured_location_unknown" | 1,516 rows accepted 0.016; also a leakage suspect | 02 (5c) | yes | yes |
| 3.3 | "dti_above_program_limit" (conventional and USDA: band "50%-60%" or ">60%"; FHA / VA: ">60%") | the DTI limit depends on the program: at 50-60%, conventional 0.33 vs FHA 0.89. USDA has too few rows above 50% to tell and is grouped with conventional. 7,211 rows, accepted 0.20 | 02 (4d, 6) | yes | yes |
| 3.4 | "ltv_at_standard_value" (80, 90, 95, 97, 100) | exact standard values accepted ~0.05 more than their neighbours | 02 (3b) | CV decides | yes |

Caveat for 3.3 and 3.4: their thresholds were read on the whole training set, so their CV gain is slightly optimistic. Better: take the thresholds from the program guides (external, fixed in advance) rather than from the data (to verify).

## Block 4: geography

| # | Feature | Why | Source | Trees | Linear |
|---|---|---|---|---|---|
| 4.1 | County frequency (train + test, label-free) + "is_unseen_county" | 84 test-only counties | data check | yes | yes |
| 4.2 | County out-of-fold target encoding, m in {5, 20, 100}, shrunk towards the MSA / state rate | county adds signal only once shrunk; county is nested in MSA and state | 02 (1b, 2) | yes (replaces raw county) | yes |
| 4.3 | Test: LightGBM without raw "county_code" / "derived_msa-md" categoricals | the baseline stopped at ~90 trees; hypothesis: overfitting on 2,898 levels | 01 (G2), runs.csv | yes | n/a |

## Linear model only

Shapes: cubic spline on log for the money trio (quartile knots) + the flags; LTV as 5-point bins; census: one size column (log) + owner-occupied share.
Crossings: DTI band x loan type, DTI band x LTV band, construction x loan type, lien x loan-amount band, construction x manufactured secured type. Then the other pairs of 02 (4c) with gain >= 1 point, one at a time.

## Coverage check: does the plan miss a segment?

02 (6) tested the flags above against the data.

- The low-acceptance flags (1.3, 1.5, 1.6, 3.1, 3.2, 3.3) cover 18.6% of the rows, accepted 0.48, and hold 65% of all denials. The other rows are accepted 0.935.
- A depth-3 tree finds 6 small segments (13% of rows, 61% of denials). All but one are covered by these flags. The exception is "initially_payable_to_institution" = 3 (1,259 rows, none accepted), a process field (see the open questions).
- On the rows without any flag, the same tree finds nothing sharp beyond that same process level: every other leaf is accepted ~0.80 or more. The remaining denials are diffuse, so they need the fine-grained features (DTI detail, geography, credit score type), not new flags.
- KMeans has only 2 to 3 real clusters (silhouette 0.56 at k = 2, 0.32 at k = 3, ~0.1 from k = 4): the exempt records and the manufactured homes, both already flags.

## Rejected, with the reason

- PCA components: signal spread over many directions, 10 components reach only 0.80 AUC (02 (5d), 01 (H3)).
- KMeans cluster id: the only real clusters are exempt records and manufactured homes, already flags 1.1 and 3.1 / 3.2 (02 (6c)).
- More segment flags: none needed, nothing sharp is left outside the plan (02 (6b)).
- Property / MSA income, income / MSA income for trees: no gain or a loss (02 (4b)).
- Dropping features with low single-feature AUC: weak alone can matter in a crossing, e.g. loan type (02 (1, 4c)).

## Open questions (their answers can change the plan)

- Data dictionary (official HMDA documentation): meaning of 1111, housing age 0, tract income % = 0, "aus-1" = 7, credit score types 11 to 15, sex = 4, the reported LTV definition, loan terms > 360 or < 12 months.
- Leakage verdicts: "preapproval", "initially_payable_to_institution", "aus-1", "location_unknown", "manufactured_location_unknown", the property-block missingness. These decide the fair model's list, not the trees' list.
  - "initially_payable_to_institution" = 3 deserves a name of its own in that list: it is the only sharp segment outside the plan's flags (1,566 rows, none accepted, 10% of all denials; 1,069 of them carry no plan flag, 02 (6b)). Trees already get it through the raw column, so it adds nothing for them. The clean / fair model must exclude it explicitly.
