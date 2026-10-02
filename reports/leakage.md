# Leakage

Yes, there is leakage: 2 columns record the decision itself. 3 more are suspected. Nothing else was found.

A column leaks when its value is written because of the decision, not before it. It is still in the test set, so it helps the leaderboard, but it must stay out of the fair model.

| column | what we see | leak? | competition model | fair model |
|---|---|---|---|---|
| preapproval | 1 (requested): 4,713 rows, all accepted | yes | keep | exclude |
| initially_payable_to_institution | 3 (not applicable): 1,566 rows, none accepted | yes | keep | exclude |
| aus-1 | 6 (no system used): accepted 0.61 | not sure | keep | to decide |
| location unknown (census block empty) | accepted 0.17 | not sure | keep | to decide |
| LTV or property value missing | missing LTV: accepted 0.54 | not sure | keep | to decide |
| submission_of_application | all levels between 0.83 and 0.91 | no | keep | keep |

Why the two "yes": the HMDA reporting rules only allow these values for certain outcomes. A denied application can't be reported with a requested preapproval, and "not applicable" is only allowed when the application was denied, withdrawn or incomplete. Source: "reports/refs/hmda_dictionary.md".

Why the three "not sure": each could be missing from the start (no leak) or missing because the application was denied early (leak). The reporting rule that decides it has not been found yet.

## How it was checked

"src/leakage_scan.py" looks at every level of every column and reports those with at least 50 rows and an accept rate below 2% or above 98%. Out of 1,077 levels it reports 7: the two above, and 5 small counties / MSAs (53 to 72 rows each).

The 5 small areas are chance, not leakage: 782 areas have 50 rows or more, and an area accepted 93% of the time has at most one denial in 55 rows with probability 9.5%.

The scan reads one column at a time. One pair of columns is also almost decided: manufactured home + location unknown, 1,516 rows accepted 0.016 (notebook 02).
