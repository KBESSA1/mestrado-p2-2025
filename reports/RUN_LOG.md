2025-10-24 Linear LODO | CP: r2=-2.6734, rmse=2.8455, mae=2.4632 | TDN(ADF): r2=-1.9757, rmse=5.8949, mae=4.9752
## 2025-10-24 12:39 — Notas do ciclo espectral puro (LODO)
- Features: médias espaciais por **buffer** (~30 m) já fornecidas no CSV (paper). **Sem 2D-Average Pooling**.
- Clima: **não incluído** neste ciclo (ERA5/CHIRPS virá no próximo).
- TDN oficial: **TDN_based_ADF** (TDN_based_NDF fica como análise secundária).
- Validação: **LODO por data** (tira TODAS as sub-amostras da data do treino).
- Modelos rodados: **Naïve**, **Linear**, **GB (sklearn)**, **XGBoost**.
- Resultado geral: **R² médio ≤ 0** para **CP** e **TDN_based_ADF**; **GB/XGB** reduzem RMSE/MAE vs Linear/Naïve, mas não tornam R² positivo.
- Implicação: espectro puro + média espacial por buffer **não carrega sinal suficiente** para generalizar por data; próximo ciclo exige **Δ controlado** e **clima**.
## 2025-10-24 12:43 — Notas do ciclo espectral puro (LODO)
- Features: médias espaciais por **buffer** (~30 m) já fornecidas no CSV (paper). **Sem 2D-Average Pooling**.
- Clima: **não incluído** neste ciclo (ERA5/CHIRPS virá no próximo).
- TDN oficial: **TDN_based_ADF** (TDN_based_NDF fica como análise secundária).
- Validação: **LODO por data** (tira TODAS as sub-amostras da data do treino).
- Modelos rodados: **Naïve**, **Linear**, **GB (sklearn)**, **XGBoost**.
- Resultado geral: **R² médio ≤ 0** para **CP** e **TDN_based_ADF**; **GB/XGB** reduzem RMSE/MAE vs Linear/Naïve, mas não tornam R² positivo.
- Implicação: espectro puro + média espacial por buffer **não carrega sinal suficiente** para generalizar por data; próximo ciclo exige **Δ controlado** e **clima**.
