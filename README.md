# Mestrado UFMS – Predição de CP e TDN

Este repositório espelha o estado real do container Docker em `/workspace` na data de 2025-11-21.

## Estrutura principal

- `Dockerfile`, `docker-compose.yml`, `environment.yml`: definição do ambiente (container `mestrado`).
- `src/`: código dos experimentos.
  - Scripts `01_*`: baselines por cenário (D5/D7, clim/noclim, CP/TDN).
  - `02_*`, `03_*`, `04_*`: Ridge, GB, XGB, feature importance e seleção de features.
  - `05_baseline_mlp.py` e `_mlp_cv_runner.py`: MLP simples (arquitetura fixa para todos os cenários).
  - `_gb_cv_runner.py`, `_hgb_cv_runner.py`, `_xgb_cv_runner.py`: runners de cross-validation.
  - KAN e XNet nos scripts `01_*_kan.py` e `01_*_xnet.py`.
- `data/`:
  - `data/data_processed/Complete_Dataset.csv`: dataset oficial UFMS (base única a partir da qual derivamos D5/D7, com/sem clima).
  - `data/data_raw/Complete_DataSet.csv`: backup do original.
  - `data/feature_selected/`: versões FS15 (top-15 features por cenário via XGBoost).
  - `data/feature_sets/`: listas de features correspondentes a cada CSV FS15.
- `reports/`:
  - `reports/progress/`: métricas consolidadas (LODO, OOF, comparações, tabelas finais).
  - `reports/baseline_experimento_UFMS/`: resultados detalhados por modelo/cenário.
  - `reports/ablations/`: ablações de KAN/XNet e KFold.

## Linha do tempo (bem resumida)

1. Baselines (Naive, Linear, Ridge, GB, XGB) sobre o `Complete_Dataset.csv` com validação LODO por data.
2. Inclusão da MLP simples, mesma arquitetura em todos os cenários.
3. Testes com KAN e XNet nos mesmos cenários (CP/TDN, com/sem clima).
4. Seleção de features via XGBoost (FS15) e re-treino de GB, XGB, Ridge, MLP, KAN e XNet.
5. Consolidação dos resultados em `reports/progress/UFMS_FINALS_best.csv` e resumos em `UFMS_CHAMPIONS_LODO.md` e `UFMS_FINAL_REPORT_FS15_LODO.md`.

