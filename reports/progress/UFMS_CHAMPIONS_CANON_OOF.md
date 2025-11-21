# Campeões — Métrica Canônica = OOF (LODO)

Preferimos OOF (predições empilhadas) por ser mais estável do que média de folds.

## Tabela resumida (R²/ RMSE/ MAE: OOF  →  FINAL  |  Δ = FINAL-OOF)

- **D5 / clim / CP** → mlp | R² 0.684 → 0.346 (Δ -0.337) | RMSE 1.443 → 1.296 (Δ -0.148) | MAE 1.091 → 1.111 (Δ 0.020)
- **D5 / clim / TDN_based_ADF** → xgbnative | R² 0.298 → 0.235 (Δ -0.063) | RMSE 4.348 → 3.707 (Δ -0.641) | MAE 3.469 → 3.026 (Δ -0.443)
- **D5 / noclim / CP** → mlp | R² 0.554 → 0.215 (Δ -0.340) | RMSE 1.713 → 1.402 (Δ -0.311) | MAE 1.253 → 1.154 (Δ -0.099)
- **D5 / noclim / TDN_based_ADF** → xgbnative | R² 0.339 → 0.246 (Δ -0.092) | RMSE 4.221 → 3.584 (Δ -0.637) | MAE 3.357 → 2.935 (Δ -0.422)
- **D7 / clim / CP** → mlp | R² 0.541 → 0.385 (Δ -0.156) | RMSE 1.685 → 1.266 (Δ -0.420) | MAE 1.161 → 1.049 (Δ -0.111)
- **D7 / clim / TDN_based_ADF** → xgbnative | R² 0.430 → 0.366 (Δ -0.063) | RMSE 3.950 → 3.297 (Δ -0.653) | MAE 2.999 → 2.594 (Δ -0.405)
- **D7 / noclim / CP** → mlp | R² 0.582 → 0.324 (Δ -0.258) | RMSE 1.608 → 1.317 (Δ -0.291) | MAE 1.163 → 1.080 (Δ -0.083)
- **D7 / noclim / TDN_based_ADF** → xgbnative | R² 0.474 → 0.338 (Δ -0.136) | RMSE 3.794 → 3.403 (Δ -0.391) | MAE 3.022 → 2.753 (Δ -0.270)
- **RAW / clim / CP** → mlp | R² 0.570 → 0.309 (Δ -0.261) | RMSE 1.812 → 1.513 (Δ -0.298) | MAE 1.262 → 1.268 (Δ 0.006)
- **RAW / clim / TDN_based_ADF** → xgbnative | R² 0.464 → 0.388 (Δ -0.076) | RMSE 3.910 → 3.315 (Δ -0.595) | MAE 2.932 → 2.574 (Δ -0.359)
- **RAW / noclim / CP** → mlp | R² 0.545 → 0.259 (Δ -0.285) | RMSE 1.863 → 1.565 (Δ -0.298) | MAE 1.321 → 1.266 (Δ -0.056)
- **RAW / noclim / TDN_based_ADF** → xgbnative | R² 0.471 → 0.344 (Δ -0.126) | RMSE 3.886 → 3.448 (Δ -0.439) | MAE 3.063 → 2.831 (Δ -0.232)