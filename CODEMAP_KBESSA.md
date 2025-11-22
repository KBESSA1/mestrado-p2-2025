# Mapa de Código do Projeto (vista Kbessa)

Visão organizada do que existe em `src/` dentro do container, **do ponto de vista de uso**  
(onde mexer para: baseline, LODO, FS, HLS, etc.).

> Regra de ouro: quando for apresentar algo pro Takashi, comece por  
> `02_baseline_*`, `01_*_cp_*`, `01_*_tdn_*`, `utils_lodo.py` e scripts de FS/XGB.

---

## 1. Baselines oficiais gerais (Linear, Ridge, GB, XGB, MLP)

Scripts “genéricos”, que não amarram cenário (D5/D7/clim/noclim) no nome:

- `02_baseline_linear.py`  
  - Regressão Linear básica (sem FS).
- `02b_baseline_ridge.py`  
  - Ridge Regression (sem FS).
- `03_baseline_gb.py`  
  - Gradient Boosting (sklearn), baseline geral.
- `04_baseline_xgb.py`  
  - XGBoost “clássico” (não nativo).
- `05_baseline_mlp.py`  
  - Baseline MLP (sklearn/PyTorch, dependendo da fase).

Uso típico:
- Servem como **baseline por cenário** (definindo CSV + target + esquema de CV via argumentos).
- Hoje, os resultados consolidados estão em `UFMS_MASTER_*`, então esses scripts são mais para **replay/pedagógico**.

---

## 2. Experimentos LODO por cenário (CP/TDN, D5/D7, clim/noclim, modelo)

Scripts 01_* amarram **CP/TDN × base (D5/D7) × clima on/off × modelo**.

### 2.1 CP — D5/D7 × clim/noclim × modelos

- `01_cp_D5_clim_gb.py`
- `01_cp_D5_clim_kan.py`
- `01_cp_D5_clim_linear.py`
- `01_cp_D5_clim_mlp.py`
- `01_cp_D5_clim_naive.py`
- `01_cp_D5_clim_ridge.py`
- `01_cp_D5_clim_xgb.py`
- `01_cp_D5_clim_xnet.py`

- `01_cp_D5_noclim_gb.py`
- `01_cp_D5_noclim_kan.py`
- `01_cp_D5_noclim_linear.py`
- `01_cp_D5_noclim_lodo.py`   ← helper específico de LODO (não é modelo)
- `01_cp_D5_noclim_mlp.py`
- `01_cp_D5_noclim_naive.py`
- `01_cp_D5_noclim_ridge.py`
- `01_cp_D5_noclim_xgb.py`
- `01_cp_D5_noclim_xnet.py`

- `01_cp_D7_clim_gb.py`
- `01_cp_D7_clim_kan.py`
- `01_cp_D7_clim_linear.py`
- `01_cp_D7_clim_mlp.py`
- `01_cp_D7_clim_naive.py`
- `01_cp_D7_clim_ridge.py`
- `01_cp_D7_clim_xgb.py`
- `01_cp_D7_clim_xnet.py`

- `01_cp_D7_noclim_gb.py`
- `01_cp_D7_noclim_kan.py`
- `01_cp_D7_noclim_lodo.py`   ← helper específico de LODO
- `01_cp_D7_noclim_mlp.py`
- `01_cp_D7_noclim_naive.py`
- `01_cp_D7_noclim_ridge.py`
- `01_cp_D7_noclim_xgb.py`
- `01_cp_D7_noclim_xnet.py`

- `01_exp_s2paper_d5_baselines.py`  
  - Script “legacy” de baseline no estilo do paper original S2 (usado pra validar o pipeline lá atrás).

### 2.2 TDN_based_ADF — D5/D7 × clim/noclim × modelos

- `01_tdn_D5_clim_gb.py`
- `01_tdn_D5_clim_kan.py`
- `01_tdn_D5_clim_linear.py`
- `01_tdn_D5_clim_mlp.py`
- `01_tdn_D5_clim_naive.py`
- `01_tdn_D5_clim_ridge.py`
- `01_tdn_D5_clim_xgb.py`
- `01_tdn_D5_clim_xnet.py`

- `01_tdn_D5_noclim_gb.py`
- `01_tdn_D5_noclim_kan.py`
- `01_tdn_D5_noclim_linear.py`
- `01_tdn_D5_noclim_mlp.py`
- `01_tdn_D5_noclim_naive.py`
- `01_tdn_D5_noclim_ridge.py`
- `01_tdn_D5_noclim_xgb.py`
- `01_tdn_D5_noclim_xnet.py`

- `01_tdn_D7_clim_gb.py`
- `01_tdn_D7_clim_kan.py`
- `01_tdn_D7_clim_linear.py`
- `01_tdn_D7_clim_mlp.py`
- `01_tdn_D7_clim_naive.py`
- `01_tdn_D7_clim_ridge.py`
- `01_tdn_D7_clim_xgb.py`
- `01_tdn_D7_clim_xnet.py`

- `01_tdn_D7_noclim_gb.py`
- `01_tdn_D7_noclim_kan.py`
- `01_tdn_D7_noclim_linear.py`
- `01_tdn_D7_noclim_lodo.py`  ← helper específico de LODO
- `01_tdn_D7_noclim_mlp.py`
- `01_tdn_D7_noclim_naive.py`
- `01_tdn_D7_noclim_ridge.py`
- `01_tdn_D7_noclim_xgb.py`
- `01_tdn_D7_noclim_xnet.py`

> **Resumo mental:**  
> 01_cp_* → CP  
> 01_tdn_* → TDN_based_ADF  
> Sufixos dizem **base (D5/D7)**, **clima (clim/noclim)** e **modelo (gb/mlp/xgb/kan/xnet/naive/linear/ridge)**.

---

## 3. Funções utilitárias / helpers gerais

- `00_utils_lodo.py`  
  - Versão antiga/auxiliar de utilitários de LODO (mantida por histórico).
- `utils_lodo.py`  
  - **Versão oficial** das funções de LODO (group por `Date`, splits, etc.).
- `sitecustomize.py`  
  - Ajustes globais de ambiente (paths, prints amigáveis, etc).
- `ping.py`  
  - Script de teste rápido (importar libs, checar ambiente).

---

## 4. Scripts Python de suporte (análises, dashboards, comparações)

- `_compare_lodo_vs_gkf.py`  
  - Compara resultados LODO vs GroupKFold (sanidade metodológica).
- `_eval_groupkfold.py`  
  - Avaliações mais focadas em GroupKFold.

- `compare_climate_gains.py`  
  - Análises de **ganho com clima** vs **sem clima**.

- `feat_picker.py`  
  - Utilitário para selecionar subconjuntos de features (auxiliar de FS).
- `feature_config.py`  
  - Configurações de features (listas, grupos, etc).

- `lit_feature_filter.py`  
  - Filtro de features relacionado ao “lit”/paper (seleções guiadas por literatura).

- `progress_dashboard.py`  
  - Dashboard textual/plots para acompanhar evolução dos experimentos (lê `UFMS_MASTER_*`).

- `quick_test.py`  
  - Script de sandbox para testar alguma ideia rápida com o dataset.

- `run_exp01.py`  
  - Orquestrador de uma bateria de experimentos “exp01” (primeiras rodadas).
- `run_with_climate_raw.py`  
  - Rodadas específicas no cenário RAW com clima.

- `05_baseline_mlp.backup.py`  
- `05_baseline_mlp.backup.preproc.py`  
  - Backups históricos da evolução do baseline MLP (não usar como fonte principal agora).

---

## 5. Runners auxiliares de cross-validation / grid search

- `_gb_cv_runner.py`  
- `_hgb_cv_runner.py`  
- `_mlp_cv_runner.py`  
- `_xgb_cv_runner.py`  
- `_xgb_cv_runner_native.py`  
- `_xgb_native_cv_runner.py`  

Esses scripts:

- Definem **grids leves** de hiperparâmetros;
- Rodam **CV sistemático** (geralmente KFold/GroupKFold);
- Alimentam os CSVs de tuning que depois entraram em  
  `UFMS_TUNED_all.csv` e `UFMS_TUNED_vs_FINALS.csv`.

> Regra prática atual: usar `_gb_cv_runner.py` e `_xgb_cv_runner_native.py` como referência de **como** foi feito tuning/FS para explicar ao Takashi.

---

## 6. Scripts de HLS / Sentinel / pré-processamento de bandas

Pasta `datasets/` e afins:

- `datasets/make_hls_s2_bands.py`  
  - Script principal para gerar bandas/índices harmonizados HLS/S2.
- `datasets/make_hls_s2_bands.backup.py`  
- `datasets/make_hls_s2_bands.procpatch.backup.py`  
- `datasets/make_hls_s2_bands.qcpatch.backup.py`  
  - Versões anteriores/patcheadas (histórico, não mexer a menos que precise auditar).

- `datasets/__pycache__/make_hls_s2_bands.cpython-310.pyc`  
  - Apenas cache (ignorar).

Scripts auxiliares de HLS:

- `hls_extract_fill_bands.py`
- `hls_extract_fill_bands_v2.py`
  - Tratamento de bandas faltantes / preenchimento.

- `hls_manifest_s2.py`
- `hls_manifest_s2_cmr.py`
  - Criação/uso de manifest de cenas S2/HLS (listagem por data/área).

- `make_hls_datasets.py`
  - Geração de datasets HLS/S2 a partir dos manifests.

---

## 7. Scripts de seleção/importância de features (FS via XGBoost / Ridge)

### 7.1 Ridge + FS por cenário

- `02_cp_D5_clim_ridge_fs.py`
- `02_cp_D5_noclim_ridge_fs.py`
- `02_cp_D7_clim_ridge_fs.py`
- `02_cp_D7_noclim_ridge_fs.py`

- `02_tdn_D5_clim_ridge_fs.py`
- `02_tdn_D5_noclim_ridge_fs.py`
- `02_tdn_D7_clim_ridge_fs.py`
- `02_tdn_D7_noclim_ridge_fs.py`

> Ridge treinado em **datasets já reduzidos por FS15** ou gerando comparações FS vs full.

### 7.2 GB + FS por cenário

- `03_cp_D5_clim_gb_fs.py`
- `03_cp_D5_noclim_gb_fs.py`
- `03_cp_D7_clim_gb_fs.py`
- `03_cp_D7_noclim_gb_fs.py`

- `03_tdn_D5_clim_gb_fs.py`
- `03_tdn_D5_noclim_gb_fs.py`
- `03_tdn_D7_clim_gb_fs.py`
- `03_tdn_D7_noclim_gb_fs.py`

> Idem Ridge, mas para GB; resultados entram na análise FS15.

### 7.3 Importâncias de features e geração de FS15

- `02_feature_importance_xgb_all.py`  
  - Roda XGBoost em todos cenários e guarda importâncias (ganho) fold-wise.
- `03_feature_importance_summary.py`  
  - Agrega importâncias por cenário (média/mediana, frequência em top-K).

- `ufms_fs_ranker.py`  
  - Gera rankings FS15 estáveis (top-15 features por cenário).
- `ufms_make_fs_csv.py`  
  - Cria CSVs reduzidos em `data/feature_selected/` e listas em `data/feature_sets/`.

Esses scripts são a base da política de FS descrita em  
`UFMS_FINAL_REPORT_FS15_LODO.md` e usada nos modelos FS.

---

## 8. Shell scripts de automação de experimentos

Na raiz de `src/`:

- `make_policy_symlinks.sh`  
  - Cria symlinks de políticas/arquivos de config (legacy/organização).

Na raiz do projeto (`/workspace`):

- `run_final_baselines.sh`  
  - Script para rodar um conjunto final de baselines (LODO) de forma automatizada.

> Ver também:
> - `/workspace/run_gkf_all.sh`
> - `/workspace/scripts/cleanup_reports.sh`

---

## 9. Arquivos auxiliares / backups / não prioritários

- `03_baseline_gb.py.bak`  
- `04_baseline_xgb.py.bak`  
  - Backups dos scripts de baseline GB/XGB quando foram mexidos (indent/argparse etc.).

- `lit_prior.yaml`  
  - Configuração/priors relacionada ao “lit”/paper (mais meta dado que código).

- `__init__.py`  
  - Marca `src` como pacote Python (importável via `PYTHONPATH=/workspace/src`).

- Arquivos `__pycache__/...`  
  - Apenas caches do Python; podem ser ignorados.

---

## 10. Como usar este mapa (rota mental)

- Precisa **reproduzir um resultado LODO**?  
  → olhar **Seção 2** (scripts `01_*`) + `utils_lodo.py`.

- Precisa explicar **baseline clássico** em aula/banca?  
  → **Seção 1** (02/03/04/05 baselines) + `UFMS_MASTER_LODO_all.csv`.

- Quer mostrar **efeito de clima**?  
  → rodadas em Seção 2 (clim vs noclim) + `compare_climate_gains.py`.

- Quer falar de **FS / importâncias de features**?  
  → **Seção 7** (XGB FS + Ridge/GB FS) + `UFMS_FINAL_REPORT_FS15_LODO.md`.

- Quer discutir **KFold vs LODO / overfitting metodológico**?  
  → **Seção 5** (runners) + scripts `_compare_lodo_vs_gkf.py`, `_eval_groupkfold.py` e os CSVs `UFMS_MASTER_KFOLD_all.csv` vs `UFMS_MASTER_LODO_all.csv`.
