| rows with | worst FP | accepted, same scores | z FP | worst FN | denied, same scores | z FN |
|---|---|---|---|---|---|---|
| manufactured home | 0.005 | 0.002 | +1.1 | 0.580 | 0.572 | +0.2 |
| location unknown | 0.000 | 0.002 | -0.7 | 0.085 | 0.149 | -2.5 |
| LTV blank | 0.050 | 0.030 | +1.7 | 0.505 | 0.480 | +0.7 |
| no automated underwriting (aus-1 = 6) | 0.010 | 0.018 | -0.9 | 0.800 | 0.717 | +2.6 |
| DTI > 60% | 0.000 | 0.000 | -0.2 | 0.450 | 0.513 | -1.8 |
| conventional loan | 0.920 | 0.919 | +0.1 | 0.950 | 0.847 | +4.0 |
| exempt record (1111) | 0.040 | 0.017 | +2.5 | 0.000 | 0.000 | +0.0 |
| submitted directly to the lender | 0.805 | 0.869 | -2.7 | 0.835 | 0.796 | +1.4 |
| initially payable to the lender | 0.890 | 0.942 | -3.2 | 0.995 | 0.962 | +2.5 |
| single male applicant (derived_sex Male) | 0.275 | 0.161 | +4.4 | 0.330 | 0.309 | +0.6 |
| applicant age > 74 | 0.050 | 0.019 | +3.3 | 0.030 | 0.017 | +1.5 |
| H1 term off the 12-month grid | 0.005 | 0.003 | +0.4 | 0.055 | 0.022 | +3.2 |
| H2 LTV blank, property value present | 0.010 | 0.013 | -0.3 | 0.455 | 0.391 | +1.8 |
| H3 computed LTV > 100 | 0.010 | 0.006 | +0.8 | 0.125 | 0.093 | +1.6 |

Shares of rows. Worst FP: the 200 denied rows with the highest scores (0.979 to 0.992), against the 16,074 accepted rows in that range; worst FN: the 200 accepted rows with the lowest scores (0.009 to 0.185), against the 4,918 denied rows in that range. Final blend, run 1ff742f2. The rows: reports/figs/t11_worst_rows.csv.

Residual (accepted - prediction) with and without each candidate, inside the regions where the worst errors sit:

| candidate | region | rows with | residual with | residual without | difference | z |
|---|---|---|---|---|---|---|
| H1 term off the 12-month grid | low scores: manufactured or no AUS | 1,011 | +0.0299 | -0.0035 | +0.0334 | +3.3 |
| H1 term off the 12-month grid | manufactured homes only | 302 | +0.0723 | -0.0006 | +0.0729 | +3.6 |
| H1 term off the 12-month grid | high scores: prediction > 0.95 | 345 | +0.0039 | +0.0029 | +0.0010 | +0.1 |
| H2 LTV blank, property value present | low scores: manufactured or no AUS | 5,879 | +0.0037 | -0.0040 | +0.0076 | +1.6 |
| H2 LTV blank, property value present | manufactured homes only | 5,482 | +0.0068 | -0.0051 | +0.0118 | +1.9 |
| H2 LTV blank, property value present | high scores: prediction > 0.95 | 1,356 | +0.0124 | +0.0027 | +0.0097 | +2.7 |
| H3 computed LTV > 100 | low scores: manufactured or no AUS | 1,095 | +0.0054 | -0.0025 | +0.0079 | +0.9 |
| H3 computed LTV > 100 | manufactured homes only | 848 | -0.0006 | +0.0019 | -0.0025 | -0.2 |
| H3 computed LTV > 100 | high scores: prediction > 0.95 | 1,166 | +0.0110 | +0.0028 | +0.0082 | +1.9 |

Dropped before the screen:

| candidate | check |
|---|---|
| application channel (submission x payable) | largest absolute z of the residual over 5 combinations with n >= 1,000: 1.9 |
| county accept rate on exempt rows | AUC on the 3,016 exempt rows: blend 0.7729, blend + rate 0.7694, rate alone 0.5393 |
