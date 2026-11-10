| scores | AUC | log loss | Brier |
|---|---|---|---|
| as predicted, p | 0.920434 | 0.1949 | 0.0524 |
| p squared | 0.920434 | 0.2168 | 0.0578 |
| square root of p | 0.920434 | 0.2188 | 0.0590 |
| rank / (n + 1) | 0.920434 | 0.6347 | 0.2261 |
| sharpened, sigmoid(2 logit p) | 0.920434 | 0.2620 | 0.0555 |

Out-of-fold predictions of the final blend (run 1ff742f2). Every transform is strictly increasing and keeps scores in (0, 1), so log loss and Brier are defined for all of them.
