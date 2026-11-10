Hypothesis: the DTI rank separates accepted from denied less well inside FHA and VA than inside conventional loans. AUC of the DTI rank alone (higher DTI = lower score), rows with a DTI, products with at least 1,000 rows:

| loan product | rows | denied | AUC of the DTI rank |
|---|---|---|---|
| Conventional:First Lien | 65,996 | 10,463 | 0.680 |
| Conventional:Subordinate Lien | 4,074 | 801 | 0.684 |
| FHA:First Lien | 17,555 | 2,244 | 0.668 |
| VA:First Lien | 8,652 | 753 | 0.631 |

Hypothesis: construction_method is repeated by the manufactured-home columns. Site-built rows (construction_method = 1) and the value 'not applicable' in each manufactured-home column:

| column | site-built rows | of them not applicable | of them exempt (1111) | not applicable but not site-built |
|---|---|---|---|---|
| manufactured_home_secured_property_type | 95,431 | 92,669 | 2,762 | 16 |
| manufactured_home_land_property_interest | 95,431 | 92,669 | 2,762 | 115 |
