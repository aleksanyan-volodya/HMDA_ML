# HMDA dictionary: codes and reporting rules used by this project

Retrieved 2026-09-30. The official pages refuse direct downloads (HTTP 403), so this file keeps the verbatim extracts and their URLs instead of a copy of the pages.

## Sources

1. FFIEC / CFPB, "HMDA LAR data fields" (public loan-level dataset): https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
2. CFPB, Regulation C (12 CFR 1003), official interpretations of § 1003.4: https://www.consumerfinance.gov/rules-policy/regulations/1003/interp-4/
3. CFPB, partial exemptions from HMDA under Regulation C (EGRRCPA, 2018): https://www.consumerfinance.gov/rules-policy/final-rules/partial-exemptions-from-requirements-of-home-mortgage-disclosure-act-under-regulation-c/
4. Bankers Compliance Group, "HMDA Data: Initially Payable to You" (secondary source quoting Regulation C): https://blog.bankerscompliance.com/en/hmda-data-initially-payable-to-you

Quotes from source 1 are verbatim. Quotes from source 2 were extracted by a summarising web tool and may be lightly paraphrased; source 4 is secondary. Both are marked where used.

## Code lists (source 1)

| Field | Codes |
|---|---|
| loan_type | 1 Conventional, 2 FHA, 3 VA, 4 USDA RHS/FSA (matches "derived_loan_product_type") |
| preapproval | 1 Preapproval requested, 2 Preapproval not requested |
| aus-1 | 1 Desktop Underwriter (DU), 2 Loan Prospector (LP) or Loan Product Advisor, 3 TOTAL Scorecard, 4 Guaranteed Underwriting System (GUS), 5 Other, 6 Not applicable, 7 Internal Proprietary System, 1111 Exempt |
| submission_of_application | 1 Submitted directly to your institution, 2 Not submitted directly, 3 Not applicable, 1111 Exempt |
| initially_payable_to_institution | 1 Initially payable to your institution, 2 Not initially payable, 3 Not applicable, 1111 Exempt |
| applicant / co-applicant_credit_score_type | 1 Equifax Beacon 5.0, 2 Experian Fair Isaac, 3 FICO Risk Score Classic 04, 4 FICO Risk Score Classic 98, 5 VantageScore 2.0, 6 VantageScore 3.0, 7 More than one credit scoring model, 8 Other credit scoring model, 9 Not applicable, 10 No co-applicant (co-applicant only), 11 FICO Score 9, 12 FICO Score 8, 13 FICO Score 10, 14 FICO Score 10T, 15 VantageScore 4.0, 1111 Exempt |
| applicant_sex | 1 Male, 2 Female, 3 Information not provided, 4 Not applicable, 6 Selected both (co-applicant adds 5 No co-applicant) |
| applicant_race-1 | 1 American Indian or Alaska Native, 2 Asian (21–27 sub-groups), 3 Black or African American, 4 Native Hawaiian or Other Pacific Islander (41–44 sub-groups), 5 White, 6 Information not provided, 7 Not applicable (co-applicant adds 8 No co-applicant) |
| applicant_ethnicity-1 | 1 Hispanic or Latino (11–14 sub-groups), 2 Not Hispanic or Latino, 3 Information not provided, 4 Not applicable (co-applicant adds 5 No co-applicant) |
| *_observed | 1 Collected on the basis of visual observation or surname, 2 Not collected on that basis, 3 Not applicable (co-applicant adds 4 No co-applicant) |
| applicant_age / co-applicant_age | <25, 25-34, 35-44, 45-54, 55-64, 65-74, >74, 8888 (not applicable); co-applicant adds 9999 (no co-applicant) |
| debt_to_income_ratio | <20%, 20%-<30%, 30%-<36%, 36% … 49% (each), 50%-60%, >60%, NA, Exempt |
| total_units | 1, 2, 3, 4, 5-24, 25-49, 50-99, 100-149, >149 |
| conforming_loan_limit | C Conforming, NC Nonconforming, U Undetermined, NA Not Applicable |
| manufactured_home_secured_property_type | 1 Manufactured home and land, 2 Manufactured home and not land, 3 Not applicable, 1111 Exempt |
| manufactured_home_land_property_interest | 1 Direct ownership, 2 Indirect ownership, 3 Paid leasehold, 4 Unpaid leasehold, 5 Not applicable, 1111 Exempt |
| yes/no loan features (reverse mortgage, balloon, …) | 1 yes, 2 no, 1111 Exempt |
| action_taken (not in our data) | 1 Loan originated, 2 Approved but not accepted, 3 Application denied, 4 Withdrawn, 5 Closed for incompleteness, 6 Purchased loan, 7 Preapproval request denied, 8 Preapproval request approved but not accepted |

Field definitions quoted from source 1:

- loan_to_value_ratio is HMDA's combined_loan_to_value_ratio: "The ratio of the total amount of debt secured by the property to the value of the property relied on in making the credit decision".
- income: "The gross annual income, in thousands of dollars, relied on in making the credit decision".
- property_value: "Rounded to the midpoint of the nearest $10,000 interval for which the reported value falls".
- loan_term: "The number of months after which the legal obligation will mature or terminate".
- derived_msa-md: "The 5 digit derived MSA (metropolitan statistical area) or MD (metropolitan division) code".
- derived_loan_product_type / derived_dwelling_category: "Derived … from Loan Type and Lien Status" / "from Construction Method and Total Units fields for easier querying".

## Resolutions (TODO 3.2 to 3.6, 3.8)

| TODO | Question | Answer | Evidence |
|---|---|---|---|
| 3.2 | What is 1111? | "Exempt": a data point the lender did not have to report under the 2018 partial exemption (EGRRCPA), available to institutions with fewer than 500 closed-end loans in each of the two preceding years. A property of the lender, not of the field. Income, race, ethnicity, sex and age are not exempt. | sources 1, 3; data: 2,942 rows carry 1111 in all 14 coded columns (01 J1) |
| 3.2 | What do blanks mean on exempt rows? | "Exempt" in a numeric field, blanked by the challenge. On the 2,942 fully exempt rows, loan term, CLTV, DTI and property value are 100% blank (89% of all missing loan terms, 62% of missing property values). | data check, this file |
| 3.3 | What is derived_msa-md = 99999? | Consistent with "outside any MSA/MD" (rural). The value list of source 1 was not retrieved for this code. | 01 G4: 15,200 rows over 1,669 counties, lower incomes |
| 3.4 | Demographic levels | real information: 1–5 and sub-groups, "selected both"; refusal: "information not provided" (sex 3, race 6, ethnicity 3); structurally absent: "not applicable" (non-natural-person applicant: sex 4, race 7, ethnicity 4, age 8888) and "no co-applicant" (sex 5, race 8, ethnicity 5, observed 4, age 9999, credit score type 10); exempt: 1111 | source 1 |
| 3.4 | Credit score types 11–15 | 11 FICO 9, 12 FICO 8, 13 FICO 10, 14 FICO 10T, 15 VantageScore 4.0. The low-acceptance types (8 other, 12, 14, 15, 6) are real scoring models, used mostly on manufactured homes (01 F1), not refusals. There is still no score in the data ([N2] confirmed). | source 1 |
| 3.5 | aus-1: system or result? | The system used (DU, LP/LPA, TOTAL, GUS, other, internal), not its recommendation. 6 = not applicable, i.e. no automated system used. ([N3] resolved.) When exactly 6 is reported was not retrieved from the commentary. | source 1 |
| 3.6 | Rounding, units, LTV | loan_amount and property_value: midpoint of the $10k interval; income in $ thousands; loan_term in months; LTV is the combined LTV (all debt secured by the property), which is why it exceeds loan / property when there are other liens (plan 2.2) | source 1; 01 B3, C1 |
| 3.6 | Blanks standing for official codes | total_units blank = 5 or more units (5-24 … >149; plan 0.3 confirmed); conforming_loan_limit blank = NA; DTI blank = NA or Exempt; ages blank = 8888 / 9999 | source 1 vs our data |
| 3.8 | Families of missing values | exempt (partial-exemption rows), not applicable (entity applicants: income, age, DTI; "no co-applicant"), other (genuinely unknown). Kept apart by flags 1.1 / 1.2 / 1.4 of the feature plan. | this file, 02 (2c) |
| open | Data collection year | Unresolved. Credit score codes 11–15 exist, so it is a recent year; the exact year was not found. | |

## Reporting rules behind the deterministic subgroups (input to TODO 4.3)

- preapproval = 1 (100% accepted, 4,713 rows). Comment 4(a)(4)-2 (source 2, via the web tool): the preapproval requirement does not apply to "applications that were denied, withdrawn, or closed for incompleteness", which are then reported as not requested. A denied application therefore can't carry preapproval = 1: the value is recorded because of the outcome.
- initially_payable_to_institution = 3 (0% accepted, 1,566 rows). Regulation C as quoted by source 4: report "not applicable" if the institution had not determined whether the loan would have been initially payable to it "when the application was withdrawn, denied, or closed for incompleteness". Among originated vs denied rows, 3 can only occur on a denial: again recorded because of the outcome.

Both fields are available in test, so a competition model may use them. Both encode the decision itself, so the fair / clean model must exclude them, and the report must say that the competition score partly rests on them.
