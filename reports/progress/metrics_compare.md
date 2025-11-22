# Comparação de métricas — RAW × Δ≤5 × Δ≤7 (LODO, __mean__)

## CP

| dataset   | model   |     r2 |   rmse |   mae | source     |
|:----------|:--------|-------:|-------:|------:|:-----------|
| RAW       | gb      | -1.429 |  2.514 | 2.143 | diario_raw |
| RAW       | linear  | -2.673 |  2.846 | 2.463 | diario_raw |
| RAW       | naive   | -2.314 |  2.807 | 2.376 | diario_raw |
| D5        | ridge   | -3.125 |  2.757 | 2.414 | log_d5     |
| RAW       | xgb     | -1.374 |  2.554 | 2.187 | diario_raw |

## TDN

| dataset   | model   |     r2 |   rmse |   mae | source     |
|:----------|:--------|-------:|-------:|------:|:-----------|
| RAW       | gb      | -0.27  |  4.74  | 4     | diario_raw |
| RAW       | linear  | -1.976 |  5.895 | 4.975 | diario_raw |
| RAW       | naive   | -0.532 |  5.328 | 4.471 | diario_raw |
| D5        | ridge   | -2.417 |  6.47  | 5.345 | log_d5     |
| RAW       | xgb     | -0.248 |  4.725 | 3.979 | diario_raw |

