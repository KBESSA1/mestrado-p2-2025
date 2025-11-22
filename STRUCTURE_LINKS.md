# STRUCTURE_LINKS (snapshot nov/2025)

## 1. Raiz do projeto

- `/workspace`  
  - Raiz do container Docker.
- Arquivos principais na raiz:
  - `README.md` — visão geral do projeto e do dataset mestre.
  - `SUMMARY_DISCOVERY.md` — resumo das descobertas e métricas-chave (Takashi).
  - `STRUCTURE_LINKS.md` — este mapa de estrutura.
  - `RUN_LOG.md` — diário de bordo das execuções.
  - `CODMAP_KBESSA.md` — mapa de código / orientação para navegação em `src/`.
  - `Dockerfile`, `docker-compose.yml`, `environment.yml` — definição do ambiente.
  - `run_gkf_all.sh` — script helper para rodar GroupKFold.
  - `start.bat`, `stop.bat` — atalhos de inicialização/parada no Windows.

---

## 2. Dados

### 2.1 Diretórios principais

- `/workspace/data`
  - Raiz de dados.
- `/workspace/data/data_raw`
  - Dados **brutos / mestre**.
- `/workspace/data/data_processed`
  - Dados processados / históricos.
- `/workspace/data/feature_sets`
  - Listas de atributos selecionados (FS15 etc.).
- `/workspace/data/feature_selected`
  - CSVs já filtrados com conjuntos de atributos (FS15 por cenário).

### 2.2 Arquivos importantes

- `/workspace/data/data_raw/Complete_DataSet.csv`
  - **Dataset mestre imutável** (campo + Sentinel-2 + clima).
  - Todas as bases (RAW, D5, D7; clim/noclim) são geradas **em memória** a partir deste arquivo.

- `/workspace/data/data_processed/Complete_Dataset.csv`
  - Versão processada **histórica** usada nas primeiras rodadas.
  - Mantida por reprodutibilidade, mas não é mais o dataset “oficial”.

- `/workspace/data/feature_sets/*.features.txt`
  - Listas de atributos selecionados por cenário (FS15 via XGBoost).

- `/workspace/data/feature_selected/*.csv`
  - Versões reduzidas dos dados (apenas features FS15), usadas em
    Ridge/GB/XGB/MLP/KAN/XNet com FS.

---

## 3. Relatórios e métricas

### 3.1 Diretórios

- `/workspace/reports/`
  - Raiz de relatórios.
- `/workspace/reports/exp01_*.csv` (e similares)
  - Resultados de experimentos individuais (rodadas antigas).
- `/workspace/reports/finals_cv/`
  - `final_*.csv` — resultados finais por cenário/modelo em fases anteriores.
- `/workspace/reports/progress/`
  - Diretório **principal** de métricas consolidadas e resumos.
- `/workspace/_reports_latest.tar.gz`
  - Snapshot compactado dos relatórios mais recentes (arquivo/symlink de conveniência).

### 3.2 Arquivos “master” de métricas

- `/workspace/reports/progress/UFMS_MASTER_metrics_all.csv`
  - Tabela **mestre** com todas as execuções (5924 linhas).

- `/workspace/reports/progress/UFMS_MASTER_LODO_all.csv`
  - Subconjunto com todas as execuções em **LODO** (validação oficial).

- `/workspace/reports/progress/UFMS_MASTER_KFOLD_all.csv`
  - Subconjunto com execuções em **KFold** (ablação/capacidade intrínseca).

- `/workspace/reports/progress/UFMS_MASTER_GKF_all.csv`
  - Subconjunto com execuções em **GroupKFold** (grupos por `Date`).

- `/workspace/reports/progress/UFMS_MASTER_LODO_champions_by_scenario.csv`
  - **Campeões em LODO** por cenário (Base × Target × Clima × FS).

### 3.3 Relatórios finais e arquivos “TAKASHI”

- `/workspace/reports/progress/UFMS_FINALS_best.csv`
  - Melhores modelos oficiais por cenário.

- `/workspace/reports/progress/UFMS_TUNED_all.csv`
  - Resultados de tuning/FS por modelo/cenário.

- `/workspace/reports/progress/UFMS_TUNED_vs_FINALS.csv`
  - Comparação tuning vs modelos finais originais.

- `/workspace/reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md`
  - Análise final considerando **feature selection (FS15)**.

- `/workspace/reports/progress/UFMS_CHAMPIONS_LODO.md`
  - Texto explicando os campeões LODO por cenário.

- `/workspace/reports/progress/UFMS_TAKASHI_LODO_champions_auto.csv`
  - Campeões LODO auto-gerados (usados em `SUMMARY_DISCOVERY.md`).

- `/workspace/reports/progress/UFMS_TAKASHI_KFOLD_champions_auto.csv`
  - Campeões KFold auto-gerados (radicalmente diferentes de LODO).

- `/workspace/reports/progress/UFMS_TAKASHI_model_stats_auto.csv`
  - Estatísticas de R² médio/máx por Modelo × Target × CV.

- `/workspace/reports/progress/R2_TABLES_FINAL.md` (se presente)
  - Tabelas “human-friendly” de R² para consulta rápida.

---

## 4. Código (`src/`)

- `/workspace/src`
  - Raiz do código-fonte.

### 4.1 Utilitários de validação

- `utils_lodo.py`
  - Funções para montar **LODO por data/campanha** sem vazamento.

### 4.2 Baselines clássicos

- `02_baseline_linear.py`
- `02b_baseline_ridge.py`
- `03_baseline_gb.py`
- `04_baseline_xgb.py`
- `05_baseline_mlp.py`
  - Baselines Linear/Ridge/GB/XGB/MLP com diferentes esquemas de validação.

### 4.3 Experimentos por cenário (CP / TDN)

- `01_cp_*.py`
  - Scripts principais para **CP** (D5/D7, clim/noclim, LODO por data).
  - Exemplos: `01_cp_D5_clim_mlp.py`, `01_cp_D7_noclim_xgb.py`.
- `01_tdn_*.py`
  - Scripts principais para **TDN_based_ADF** (D5/D7, clim/noclim, LODO por data).
  - Exemplos: `01_tdn_D5_clim_gb.py`, `01_tdn_D7_noclim_xnet.py`.

### 4.4 Redes neurais pesadas

- `01_*kan*.py`
  - Experimentos com **KAN** (CP/TDN, KFold/LODO, com/sem FS).
- `01_*xnet*.py`
  - Experimentos com **XNet** (idem).

### 4.5 Feature selection via XGBoost

- `02_*_ridge_fs.py`
- `03_*_gb_fs.py`
  - Baselines Ridge/GB usando apenas FS15.

- `02_feature_importance_xgb_all.py`
- `03_feature_importance_summary.py`
- `ufms_fs_ranker.py`
- `ufms_make_fs_csv.py`
  - Scripts para calcular importâncias, gerar rankings estáveis e criar CSVs em
    `data/feature_selected/`.

### 4.6 Runners de CV / grid

- `_gb_cv_runner.py`
- `_xgb_cv_runner.py`
- `_xgb_cv_runner_native.py`
- `_mlp_cv_runner.py`
- `_hgb_cv_runner.py`
- `_xgb_native_cv_runner.py`
  - Runners limpos para CV/tuning de GB/XGB/MLP/HGB (sem mexer nos scripts antigos 03/04).

> Detalhes adicionais de cada script estão em `CODMAP_KBESSA.md`.

---

## 5. Scripts auxiliares

- `/workspace/scripts/cleanup_reports.sh`
  - Limpa/organiza relatórios antigos (tem alias no `.bashrc`).

- `/workspace/run_gkf_all.sh`
  - Atalho na raiz para rodar todas as variações de **GroupKFold**.

- `start.bat`, `stop.bat`
  - Atalhos no Windows para subir/desligar o container `mestrado`.

*(Novos scripts devem ser adicionados aqui conforme forem criados.)*

---

## 6. Alias, PATH e ambiente

- `~/.bashrc`
  - Contém, entre outros:
    - `alias cleanup_reports='bash /workspace/scripts/cleanup_reports.sh'`
    - `export PATH="/workspace/scripts:$PATH"`

- Ambiente de pacotes:
  - Ver `environment.yml` e `Dockerfile` para versões de:
    - Python, scikit-learn, xgboost, PyTorch, etc.
  - GPU usada tipicamente: RTX 4070/4090 (dependendo da máquina host).

---

## 7. Como usar este arquivo

- Use `STRUCTURE_LINKS.md` como **mapa rápido** para achar:
  - Onde está o **dataset mestre**;
  - Onde estão as **métricas oficiais (UFMS_MASTER_*, TAKASHI_*)**;
  - Quais scripts em `src/` estão ligados a cada tipo de experimento (baseline, FS, KAN/XNet);
  - Onde ficam os **CSVs com FS15** para repetir/estender experimentos.
