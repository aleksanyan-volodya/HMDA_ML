| fact | value |
|---|---|
| train rows, test rows, input columns | 105,000, 45,000, 56 |
| accepted share in train | 0.850 (89,244 accepted, 15,756 denied) |
| train ids / test ids | 100,000 to 599,996 / 600,005 to 999,992, shared ids 0 |
| AUC of the row id alone (no order leak if ~0.5) | 0.4957 |
| train rows with code 1111 (Exempt): in any coded column / in all affected columns | 3,016 / 2,942 (14 columns), accepted 0.914 |
| loan amount and property value on the $10k midpoint grid (v mod 10,000 = 5,000), both files | 1.0000 of non-blank values |
| location unknown (census block all 0): train / test rows, train accept rate | 2,117 / 929, 0.171 |
| derived_msa-md = 99999 (outside any MSA): train rows, accept rate | 15,200, 0.751 |
| county levels in test but not train, test rows affected | 84, 115 (0.26%) |
| exact duplicate rows in train over the 56 inputs | 7 |
