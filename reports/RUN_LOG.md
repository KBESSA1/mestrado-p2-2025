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
2025-10-25T12:58:22+00:00  | exp01 | LODO (paper-like, sem clima) | Δ<=5: N/A (ainda) | TDN_def=ADF | painel atualizado
2025-10-25T13:21:57  | exp01 | LODO (paper-like, sem clima) | Δ<=5: N/A | TDN_def=ADF | painel atualizado
2025-10-25T13:28:55  | exp01 | LODO (paper-like, sem clima) | Δ<=5: SIM | TDN_def=ADF | painel atualizado
2025-10-25T14:02:17  | exp01 | LODO (paper-like, sem clima) | Δ<=7: SIM | TDN_def=ADF | painel atualizado
2025-10-25T14:05:26  | exp01 | LODO (paper-like, sem clima) | Δ<=5: SIM | TDN_def=ADF | painel atualizado
2025-10-25T14:52:41  | exp01 | LODO (paper-like, sem clima) | Δ<=5: SIM | TDN_def=ADF | painel atualizado
2025-10-25T14:53:12  | exp01 | LODO (paper-like, sem clima) | Δ<=5: SIM | TDN_def=ADF | painel atualizado
2025-10-25T14:53:42  | exp01 | LODO (paper-like, sem clima) | Δ<=5: SIM | TDN_def=ADF | painel atualizado
2025-10-25T14:54:12  | exp01 | LODO (paper-like, sem clima) | Δ<=5: SIM | TDN_def=ADF | painel atualizado
2025-10-25T14:54:43  | exp01 | LODO (paper-like, sem clima) | Δ<=5: SIM | TDN_def=ADF | painel atualizado
2025-10-25T14:55:14  | exp01 | LODO (paper-like, sem clima) | Δ<=5: SIM | TDN_def=ADF | painel atualizado
2025-11-13T13:24:45Z | archived reports to _reports_20251113_130846.tar.gz (sha256: 2e62fd9b...f9d7)
2025-11-13T13:44:54Z | sealed DS complete_S2_allgeom_clim.csv (hash+schema) and set ACTIVE_DATASET.csv
2025-11-13T13:46:14Z | DS seal | complete_S2_allgeom_clim.csv | sha256:8d0b56d8e8106579dc0067f3228a2de793c81b69efc041f2479874dda3f2625c | nrows=313 | ncols=115
2025-11-13T13:58:44Z | DS rename | complete_S2_allgeom_clim.csv -> complete_S2_allgeom.csv | sha256:8d0b56d8e8106579dc0067f3228a2de793c81b69efc041f2479874dda3f2625c | nrows=312 | ncols=115
2025-11-13T14:08:57Z | DS create | complete_CLIMATE_only.csv (aligned to complete_S2_allgeom.csv) | sha256:b88363de9dae6cf97b5f67aa27f324048ebaa2ab83e03627be39d9a8432b16ef | nrows=312 | ncols=18
2025-11-13T14:27:59Z | DS create | complete_S2_allgeom_HLS.csv (aligned to complete_S2_allgeom.csv) | sha256:6dcdf4555095b55100f01e693457c98341f927c22a604983d013da26ca85dbbb | nrows=312 | ncols=115
2025-11-13T14:28:00Z | DS create | complete_CLIMATE_only_HLS.csv (aligned to complete_S2_allgeom.csv) | sha256:b88363de9dae6cf97b5f67aa27f324048ebaa2ab83e03627be39d9a8432b16ef | nrows=312 | ncols=18
2025-11-13T14:32:39Z | HLS skeletons optimized (concat/no-frag) | DATA_HLS=6dcdf4555095b55100f01e693457c98341f927c22a604983d013da26ca85dbbb | CLIM_HLS=b88363de9dae6cf97b5f67aa27f324048ebaa2ab83e03627be39d9a8432b16ef
2025-11-13T15:40:57Z | HLS F1 manifest | HLS_S30 | src=complete_S2_allgeom.csv -> reports/progress/HLS_S30_manifest.csv
2025-11-13T15:44:55Z | HLS F1 manifest | HLS_S30 | src=complete_S2_allgeom.csv -> reports/progress/HLS_S30_manifest.csv
2025-11-13T15:51:07Z | HLS F1 manifest | HLS_S30 | src=complete_S2_allgeom.csv -> reports/progress/HLS_S30_manifest.csv
2025-11-13T16:02:01Z | HLS F1 manifest | HLSS30 | src=complete_S2_allgeom.csv -> reports/progress/HLS_S30_manifest.csv
2025-11-13T16:07:30Z | HLS F1 manifest | HLSS30 -> reports/progress/HLS_S30_manifest.csv
2025-11-13T16:36:30Z | HLS F1 manifest | HLSS30 (±7d, pad=0.02) | 100% coverage -> reports/progress/HLS_S30_manifest.csv
2025-11-13T16:58:17Z | HLS F2 fill | complete_S2_allgeom_HLS.csv atualizado (full) a partir de HLSS30
2025-11-13T17:25:28Z | HLS F2 fill v2 | complete_S2_allgeom_HLS.csv atualizado (resume+cache) | bands=all
