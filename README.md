# Projeto Mestrado UFMS — Predição de CP e TDN com Sentinel-2 + Clima

Este repositório contém o código, dados derivados e relatórios do projeto de mestrado que usa **imagens Sentinel-2 (S2)** e **variáveis climáticas** para prever:

* **CP** (Proteína Bruta)
* **TDN_based_ADF** (Nutrientes Digestíveis Totais, estimados via ADF)

com foco em:

* **Validação temporal correta** (LODO por data / campanha)
* Comparação entre **modelos clássicos** e **modelos de rede mais pesados** (MLP, KAN, XNet)
* **Seleção de features** via XGBoost
* Ablações com/sem clima e com janelas Δ≤5/Δ≤7 dias entre campo e imagem.

---

## 1. Visão geral

### 1.1 Objetivo em 1 frase

Usar Sentinel-2 + clima para prever CP e TDN_based_ADF com validação temporal (LODO por data), comparando modelos clássicos (Linear, Ridge, GB, XGB) e profundos (MLP, KAN, XNet), incluindo seleção de features e ablações.

### 1.2 O que já está feito

* Pipeline de dados consolidado em um **arquivo mestre único**: `Complete_DataSet.csv`.
* Geração em memória dos cenários:

  * **D5** (Δ≤5 dias entre coleta de campo e imagem S2)
  * **D7** (Δ≤7 dias)
  * Versões **com clima** e **sem clima**
* Validação:

  * **LODO por data** (oficial)
  * **K-Fold** simples e **GroupKFold** (por data) como conferência
* Modelos rodados em todos os cenários (CP e TDN_based_ADF):

  * Naive (persistência)
  * Linear
  * Ridge
  * Gradient Boosting (GB)
  * XGBoost (XGB)
  * MLP
  * KAN
  * XNet
* **Seleção de features** via XGBoost:

  * Importância por ganho em LODO
  * Medidas de estabilidade por cenário
  * Conjunto FS15 (top-15 features estáveis)
* Re-treino de:

  * Ridge, GB, XGB, MLP, KAN e XNet **com FS** (FS15)
* Consolidação dos resultados em:

  * `reports/progress/UFMS_MASTER_*.csv`
  * `reports/progress/UFMS_FINALS_best.csv`
  * `reports/progress/UFMS_CHAMPIONS_LODO.md`
  * `reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md`
  * `reports/progress/SUMMARY_DISCOVERY.md`

---

## 2. Estrutura do repositório

Estrutura lógica (pode variar levemente):

```text
/workspace
├── src/                         # Código-fonte de experimentos e utilitários
│   ├── 01_*_cp_*.py             # Experimentos principais CP (D5, D7, clim/noclim)
│   ├── 01_*_tdn_*.py            # Experimentos principais TDN_based_ADF
│   ├── 02_baseline_linear.py
│   ├── 02b_baseline_ridge.py
│   ├── 03_baseline_gb.py
│   ├── 04_baseline_xgb.py
│   ├── 05_baseline_mlp.py
│   ├── 01_*_kan*.py             # Scripts KAN e KAN_FS
│   ├── 01_*_xnet*.py            # Scripts XNet e XNet_FS
│   ├── _gb_cv_runner.py         # Runners de CV/tuning (GB)
│   ├── _xgb_cv_runner.py        # Runners de CV/tuning (XGB)
│   ├── utils_lodo.py            # Utilitários para LODO por data
│   ├── feature_importance_*     # Scripts de FS via XGBoost
│   └── ...                      # Outros utilitários e scripts de apoio
│
├── data/
│   ├── data_raw/
│   │   └── Complete_DataSet.csv     # Arquivo mestre bruto / imutável
│   └── data_processed/
│       └── Complete_Dataset.csv     # Versão processada (histórica)
│
├── reports/
│   ├── exp01_*.csv                  # Resultados individuais de experimentos
│   ├── finals_cv/
│   │   └── final_*.csv              # Resultados finais por cenário/modelo
│   └── progress/
│       ├── UFMS_MASTER_metrics_all.csv
│       ├── UFMS_MASTER_LODO_all.csv
│       ├── UFMS_MASTER_KFOLD_all.csv
│       ├── UFMS_MASTER_GKF_all.csv
│       ├── UFMS_MASTER_LODO_champions_by_scenario.csv
│       ├── UFMS_FINALS_best.csv
│       ├── UFMS_TUNED_all.csv
│       ├── UFMS_TUNED_vs_FINALS.csv
│       ├── SUMMARY_DISCOVERY.md
│       ├── UFMS_CHAMPIONS_LODO.md
│       └── UFMS_FINAL_REPORT_FS15_LODO.md
│
├── README.md
└── RUN_LOG.md                     # Diário de bordo das execuções
---
````

## 3. Dados

### 3.1 Arquivo mestre oficial

* **Caminho:** `data_raw/Complete_DataSet.csv`

Este é o **arquivo mestre imutável** que representa a base UFMS consolidada (campo + Sentinel-2 + clima).

Principais colunas:

* `Date` — data/campanha (chave para LODO e GroupKFold)
* Targets:

  * `CP`
  * `TDN_based_ADF`
* Bandas/índices Sentinel-2:

  * Médias espaciais por buffer / geometria
  * ~52 bandas/índices em algumas versões (S2/HLS)
* Variáveis climáticas (quando disponíveis):

  * Ex.: precipitação, temperatura etc., agregadas em janelas temporais

> **Regra:** este arquivo **não é sobrescrito**. Ele é a **fonte única** a partir da qual todos os cenários D5/D7 com/sem clima são gerados **em memória** nos scripts.

### 3.2 Dataset processado (histórico)

* **Caminho:** `data_processed/Complete_Dataset.csv`

Versão processada gerada numa fase anterior. Foi tratada como “dataset oficial” nas primeiras rodadas de baseline.

Hoje é:

* Um **snapshot histórico** para reprodutibilidade.
* Ainda pode ser referenciado por scripts mais antigos.

### 3.3 Geração dos cenários (D5/D7, com/sem clima)

Atualmente, os cenários **não** são salvidos como CSVs separados; eles são gerados em memória a partir do arquivo mestre. Em cada script:

* **D5**: subset de amostras com Δ≤5 dias entre data de campo e data da imagem S2.
* **D7**: subset de amostras com Δ≤7 dias.
* **com clima (clim)**: mantém todas as features climáticas.
* **sem clima (noclim)**: remove colunas de clima, usando apenas S2 e variáveis auxiliares não climáticas.

Historicamente foram usados arquivos como:

* `ufms_D5_{clim|noclim}.csv`
* `ufms_D7_{clim|noclim}.csv`

Hoje essa lógica está embutida nos scripts (filtro por datas + seleção de colunas).

### 3.4 Cenário RAW

Em fases exploratórias houve cenários “RAW”, com menos restrições temporais.
Na análise final, os cenários **oficiais** são:

* **D5** e **D7**, cada um com:

  * Versão com clima
  * Versão sem clima
  * Para ambos os alvos: CP e TDN_based_ADF

---

## 4. Validação

### 4.1 LODO por data (validação oficial)

Estratégia principal:

* Cada **fold** corresponde a uma **data/campanha** (`Date`).
* Para cada fold:

  * Treino em todas as outras datas
  * Teste na data corrente
* Métricas por fold e agregadas:

  * **R²**
  * **RMSE**
  * **MAE**

Resultados consolidados:

* `reports/progress/UFMS_MASTER_LODO_all.csv`
* `reports/progress/UFMS_MASTER_LODO_champions_by_scenario.csv`
* `reports/progress/UFMS_FINALS_best.csv`

### 4.2 K-Fold e GroupKFold (sanidade)

Além do LODO, foram rodadas validações adicionais:

* **K-Fold** simples por amostra
* **GroupKFold** usando `Date` como grupo

Objetivo:

* Checar consistência das métricas
* Observar se há overfitting por cenário
* Comparar tendências entre diferentes esquemas de validação

Resultados:

* `reports/progress/UFMS_MASTER_KFOLD_all.csv`
* `reports/progress/UFMS_MASTER_GKF_all.csv`

---

## 5. Modelos

### 5.1 Modelos clássicos

* **Naive (persistência)**

  * Baseline obrigatório: previsão = último valor conhecido (por data/campanha ou estratégia equivalente).
* **Regressão Linear**
* **Ridge Regression**
* **Gradient Boosting Regressor** (sklearn)
* **XGBoost Regressor** (`xgboost`)

Scripts principais:

* `02_baseline_linear.py`
* `02b_baseline_ridge.py`
* `03_baseline_gb.py`
* `04_baseline_xgb.py`

### 5.2 Modelos neurais

* **MLP**

  * Em fases iniciais: às vezes `sklearn.MLPRegressor`
  * Em fases posteriores: implementações em PyTorch
  * Script base: `05_baseline_mlp.py`
* **KAN**

  * Scripts do tipo `01_*_kan*.py`
* **XNet**

  * Scripts do tipo `01_*_xnet*.py`

Para KAN e XNet também existem variantes com **feature selection** (FS), com sufixos como `_fs`.

---

## 6. Seleção de features (FS) via XGBoost

### 6.1 Estratégia de FS

1. Rodar **XGBoost** em cada cenário (base × alvo × clima/noclim) com validação **LODO por data**.
2. Em cada fold:

   * Extrair a importância das features por **ganho**.
3. Agregar as importâncias:

   * Média/mediana do ganho por feature.
   * Frequência com que cada feature aparece no **top-K** (ex.: top-15) em cada fold.
4. Definir um conjunto de features **estáveis** por cenário:

   * Em geral, **FS15** = top-15 features mais importantes/estáveis.

Arquivos de referência (nomes podem variar):

* `reports/progress/feature_importance_xgb_stability_CP.csv`
* `reports/progress/feature_importance_xgb_stability_TDN_based_ADF.csv`
* Arquivos com listas das features selecionadas (ex.: `UFMS_FS_selected_*.txt`)

### 6.2 Re-treino com FS (FS15)

Depois de definir os conjuntos FS, foram re-treinados, cenário a cenário:

* Ridge
* GB
* XGB
* MLP
* KAN
* XNet

Comparações “full features” vs **FS15**:

* `reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md`
* `reports/progress/UFMS_TUNED_all.csv`
* `reports/progress/UFMS_TUNED_vs_FINALS.csv`

Resumo empírico:

* **FS melhora claramente**:

  * Ridge, GB, XGB (reduz variância, melhora R² em vários cenários)
* **FS é neutra ou pior**:

  * MLP, KAN, XNet — especialmente em TDN com clima (perda de capacidade de representação em redes mais pesadas).

---

## 7. Resultados principais

Artefatos consolidados mais importantes:

* `reports/progress/UFMS_MASTER_metrics_all.csv`
  → Tabela “mestre” com todas as execuções registradas (modelos, cenários, métricas).

* `reports/progress/UFMS_MASTER_LODO_all.csv`
  → Subconjunto focado nas execuções com validação **LODO**.

* `reports/progress/UFMS_MASTER_KFOLD_all.csv`
  → Execuções com **K-Fold**.

* `reports/progress/UFMS_MASTER_GKF_all.csv`
  → Execuções com **GroupKFold** (por data).

* `reports/progress/UFMS_MASTER_LODO_champions_by_scenario.csv`
  → Campeões por cenário (melhor modelo em cada combinação base × alvo × clima).

* `reports/progress/UFMS_FINALS_best.csv`
  → **Melhores modelos oficiais** por cenário.

* `reports/progress/UFMS_CHAMPIONS_LODO.md`
  → Explicação textual dos campeões LODO.

* `reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md`
  → Análise final levando em conta a seleção de features (FS15).

* `reports/progress/SUMMARY_DISCOVERY.md`
  → Narrativa geral das principais descobertas.

### 7.1 Resumo científico (alto nível)

* **CP (Proteína Bruta)**:

  * Apresenta **sinal previsível** com S2 + clima.
  * R² em LODO geralmente na faixa **0,30–0,45** nos cenários mais difíceis.
  * Em configurações mais favoráveis (por data, agregações específicas), pode chegar a **~0,60–0,70**.

* **TDN_based_ADF**:

  * Muito mais difícil de prever com S2 + clima.
  * Mesmo com clima e FS, R² em ablações conservadoras muitas vezes próximo de **0,00–0,15**.
  * Melhores resultados típicos com **árvores (GB/XGB)**, mas com maior sensibilidade a cenário.

* **Modelos de árvore (GB/XGB)**:

  * Tendem a ser mais estáveis e competitivos em boa parte dos cenários.
  * Redes (MLP, KAN, XNet) têm ganhos pontuais, mas não superam consistentemente GB/XGB.

---

## 8. Ambiente e execução

### 8.1 Ambiente Docker

O projeto foi configurado para rodar em container com ou sem GPU:

* Base: imagem NVIDIA de deep learning (ex.: CUDA 12.x)
* Python: 3.10.x
* Bibliotecas principais:

  * `numpy`, `pandas`, `scikit-learn`
  * `xgboost`
  * `torch` (PyTorch)
  * Outras utilidades (ver `requirements.txt` / Dockerfile se presentes)

Pontos de montagem:

* `/workspace` (raiz do projeto dentro do container)

  * Código: `/workspace/src`
  * Dados: `/workspace/data`
  * Relatórios: `/workspace/reports`

### 8.2 Exemplo de execução (genérico)

Exemplo ilustrativo de chamada para MLP (os argumentos exatos podem variar):

Dentro do container em `/workspace`:

export PYTHONPATH=/workspace/src:$PYTHONPATH

python src/05_baseline_mlp.py \
--csv data_raw/Complete_DataSet.csv \
--date-col Date \
--target-col CP \
--scenario D5 \
--with-climate \
--val-scheme LODO \
--out reports/exp01_D5_CP_mlp_clim_lodo.csv

> Consulte o cabeçalho de cada script em `src/` para ver a assinatura real (nomes de flags e opções disponíveis).

---

## 9. Histórico do dataset (nota importante)

Para evitar confusão:

1. **Fase inicial**

   * `data_processed/Complete_Dataset.csv` era considerado o **dataset oficial**.
   * A partir dele, eram gerados CSVs físicos como `ufms_D5_*`, `ufms_D7_*` em disco.

2. **Fase atual (oficial)**

   * `data_raw/Complete_DataSet.csv` é o **arquivo mestre imutável**.
   * `data_processed/Complete_Dataset.csv` virou **snapshot histórico**.
   * Cenários D5/D7 e clim/noclim são derivados **em memória** a partir do mestre bruto.

Na dissertação e documentos finais, esta filosofia atual (arquivo mestre + derivação em memória) é a versão a ser considerada como “oficial”.

---

## 10. Diário de bordo (RUN_LOG)

* **Arquivo:** `RUN_LOG.md`

Usado como diário de bordo operacional, registra:

* Data e comando executado
* Script e parâmetros principais
* Caminho dos outputs gerados
* Observações rápidas de resultados (R², problemas, bugs)

Recomendação:

* Consultar o `RUN_LOG.md` em paralelo a este README para reconstruir a linha do tempo das experiências e entender a evolução dos modelos/cenários.
