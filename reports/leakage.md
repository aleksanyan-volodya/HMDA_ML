# Leakage

Yes, there is leakage: 2 columns record the decision itself. 3 more record it in part. Nothing else was found.

A column leaks when its value is written because of the decision, not before it. It is still in the test set, so it helps the leaderboard, but it must stay out of the fair model.

| column | what we see | leak? | full | clean | strict |
|---|---|---|---|---|---|
| preapproval | 1 (requested): 4,713 rows, all accepted | yes | keep | drop | drop |
| initially_payable_to_institution | 3 (not applicable): 1,566 rows, none accepted | yes | keep | drop | drop |
| location unknown (census block empty) | accepted 0.17; manufactured home + unknown 0.015 | in part | keep | keep | drop |
| LTV, property value or DTI missing | missing LTV: accepted 0.54 | in part | keep | keep | drop |
| aus-1 | 6 (no system used): accepted 0.61 | no | keep | keep | drop |
| submission_of_application | all levels between 0.83 and 0.91 | no | keep | keep | keep |

Three feature lists follow from it:

- full: everything, for the leaderboard.
- clean: full minus the 2 confirmed leaks. The fair model is built on it.
- strict: clean minus the 3 partial or suspect signals. Used only to measure how much they matter (TODO 12.9), not as a model of its own.

The partial signals stay in clean because they also carry real information (manual underwriting, manufactured homes sold without land), and for a tree a missing value can't be removed without dropping the whole column for every row.

## Verdicts

Sources: Regulation C official commentary to 12 CFR 1003.4 (https://www.consumerfinance.gov/rules-policy/regulations/1003/interp-4/, read 2026-10-18 through a summarising web tool, so quotes may be lightly paraphrased), and "reports/refs/hmda_dictionary.md".

"Structural" rows below are exempt (1111), non-natural-person applicants and multifamily: they explain part of each missing value by rule, whatever the outcome. On the other rows the accept rate is 0.848. Numbers are on train; test has the same signals in the same proportions.

1. preapproval = 1: leak. Comment 4(a)(4)-2: the preapproval requirement does not apply to applications denied, withdrawn or closed for incompleteness, which are reported as not requested. A denied application can't carry 1.

2. initially_payable_to_institution = 3: leak. "Not applicable" is reported when the institution had not decided, at the time the application was withdrawn, denied or closed, whether the loan would be payable to it. Between originated and denied, 3 can only occur on a denial.

3. submission_of_application: no leak. Every level is accepted between 0.83 and 0.91.

4. aus-1 = 6: no leak. Comment 4(a)(35)-4: "not applicable" when "an AUS was not used to evaluate the application", for example manual underwriting only. The rule says nothing about the outcome, so the value describes how the lender works, chosen before the decision. It is strongly associated with denial (manufactured homes: 0.24 without an AUS, 0.85 with one; other homes 0.78 against 0.92), which is why strict drops it as a check.

5. LTV or property value missing: leak in part. Comments 4(a)(24)-4 (combined LTV), 4(a)(28)-4 (property value) and 4(a)(23)-4 (DTI): "not applicable" when the institution "makes a credit decision without relying on" the number. A lender that denies early, on other grounds, may never value the property; a secured loan that is granted almost always relied on its value. The rule allows the gap on both sides, so it is not deterministic.

| rows (not structural) | n | accepted | same group with the value |
|---|---|---|---|
| LTV missing, manufactured home | 5,952 | 0.245 | 0.562 |
| LTV missing, other home | 2,600 | 0.793 | 0.901 |
| property value missing, manufactured home | 687 | 0.221 | 0.370 |
| property value missing, other home | 871 | 0.432 | 0.903 |
| DTI missing, all homes | 2,351 | 0.675 | 0.852 |

DTI falls under the same rule, with a weaker gap; it follows LTV and property value into strict.

Part of it is real information: manufactured homes sold without land often have no combined LTV (76% missing), approved or not.

6. Location unknown: leak in part, the strongest of the three. Comments 4(a)(9)-5 and 4(a)(9)(i)-3: "not applicable" if "the site of a manufactured home has not been identified" or the address is not known; 4(a)(9)(ii)-1: the census tract is optional in counties of 30,000 people or fewer. An application that stops before a site is chosen has no location; a loan that closes has one.

| rows (not structural) | n | accepted | same group, location known |
|---|---|---|---|
| location unknown, manufactured home | 1,507 | 0.015 | 0.426 |
| location unknown, other home | 303 | 0.363 | 0.900 |

On manufactured homes it is close to deterministic (22 approvals in 1,507). The same rows also carry a missing state (794 of the 824 missing states) and county (1,330 of 1,339). It stays in clean only because it can't be separated from the census columns; the report must name it as the main remaining doubt.

## How it was checked

"src/leakage_scan.py" looks at every level of every column and reports those with at least 50 rows and an accept rate below 2% or above 98%. Out of 1,077 levels it reports 7: the two leaks above, and 5 small counties / MSAs (53 to 72 rows each).

The 5 small areas are chance, not leakage: 782 areas have 50 rows or more, and an area accepted 93% of the time has at most one denial in 55 rows with probability 9.5%.

The scan reads one column at a time. One pair of columns is also almost decided: manufactured home + location unknown, 1,516 rows accepted 0.016 (notebook 02). Verdict 6 explains it.

The tables of verdicts 5 and 6 come from a read-only check on the loaded data (split into structural / other rows, then manufactured / other homes); notebook 04_fairness will hold the code.
