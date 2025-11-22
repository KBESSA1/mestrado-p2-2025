# Projeto Mestrado UFMS — Predição de CP e TDN com Sentinel-2 + Clima

Este repositório contém o código, dados derivados e relatórios do projeto de mestrado que usa **imagens Sentinel-2** (S2) e **variáveis climáticas** para prever:

- **CP** (Proteína Bruta)
- **TDN_based_ADF** (Nutrientes Digestíveis Totais, estimados via ADF)

com foco em:

- **Validação temporal correta** (LODO por data / campanha)
- Comparação entre **modelos clássicos** e **modelos de rede mais pesados** (MLP, KAN, XNet)
- **Seleção de features** via XGBoost
- Ablações com/sem clima e com janelas Δ≤5/Δ≤7 dias entre campo e imagem.

---

## 1. Visão geral

### Objetivo (1 frase)

Usar Sentinel-2 + clima para prever CP e TDN_based_ADF com validação temporal (LODO por data), comparando modelos clássicos (Linear, Ridge, GB, XGB) e profundos (MLP, KAN, XNet), incluindo seleção de features e ablações.

### O que já está feito

- Pipeline de dados consolidado em um **arquivo mestre único** (`Complete_DataSet.csv`).
- Geração em memória dos cenários:
  - **D5** (Δ≤5 dias entre coleta de campo e imagem S2)
  - **D7** (Δ≤7 dias)
  - **com clima** e **sem clima**
- Validação:
  - **LODO por data** (oficial)
  - K-Fold simples e GroupKFold (por data) como conferência
- Modelos rodados em todos os cenários (CP e TDN_based_ADF):
  - Naive (persistência)
  - Linear
  - Ridge
  - Gradient Boosting (GB)
  - XGBoost (XGB)
  - MLP
  - KAN
  - XNet
- **Seleção de features** via XGBoost:
  - Importância por ganho em LODO
  - Estabilidade por cenário
  - Conjunto FS15 (top-15 estáveis)
- Re-treino de:
  - Ridge, GB, XGB, MLP, KAN e XNet **com FS** (FS15)
- Consolidação dos resultados em:
  - `reports/progress/UFMS_MASTER_*.csv`
  - `reports/progress/UFMS_FINALS_best.csv`
  - `reports/progress/UFMS_CHAMPIONS_LODO.md`
  - `reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md`
  - `reports/progress/SUMMARY_DISCOVERY.md`

---

## 2. Estrutura do repositório

Esqueleto simplificado (pode variar levemente):

```text
/workspace
├── src/                      # Código-fonte dos experimentos e utilitários
│   ├── 01_*_cp_*.py          # Experimentos principais CP (D5, D7, clim/noclim)
│   ├── 01_*_tdn_*.py         # Experimentos principais TDN_based_ADF
│   ├── 02_baseline_linear.py
│   ├── 02b_baseline_ridge.py
│   ├── 03_baseline_gb.py
│   ├── 04_baseline_xgb.py
│   ├── 05_baseline_mlp.py
│   ├── 01_*_kan*.py          # Scripts KAN e KAN_FS
│   ├── 01_*_xnet*.py         # Scripts XNet e XNet_FS
│   ├── _gb_cv_runner.py      # Runners de CV/tuning (GB)
│   ├── _xgb_cv_runner.py     # Runners de CV/tuning (XGB)
│   ├── utils_lodo.py         # Utilitários para LODO por data
│   ├── feature_importance_*  # Scripts de FS via XGBoost
│   └── ...                   # Outros utilitários e scripts de apoio
│
├── data/
│   ├── data_raw/
│   │   └── Complete_DataSet.csv    # Arquivo mestre bruto/imutável
│   └── data_processed/
│       └── Complete_Dataset.csv    # Versão processada (histórica)
│
├── reports/
│   ├── exp01_*.csv                 # Resultados individuais de experimentos
│   ├── finals_cv/
│   │   └── final_*.csv             # Finais por cenário/modelo
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
└── RUN_LOG.md                    # Diário de bordo das execuções
3. Dados
3.1. Arquivo mestre oficial
data_raw/Complete_DataSet.csv

Este é o arquivo mestre imutável que representa a base UFMS consolidada (campo + Sentinel-2 + clima).

Contém, entre outros:

Coluna de data: Date

Alvos:

CP

TDN_based_ADF

Bandas/índices Sentinel-2:

Médias espaciais por buffer / geometria

Múltiplas bandas e índices espectrais (≈ 52 colunas de S2/HLS no total em algumas versões)

Variáveis climáticas (quando disponíveis):

Precipitação, temperatura, etc., agregados em janelas temporais

Regra: este arquivo não é sobrescrito. Ele é a fonte única a partir da qual todos os cenários D5/D7 com/sem clima são gerados em memória nos scripts.

3.2. Dataset processado (histórico)
data_processed/Complete_Dataset.csv

Versão processada gerada numa fase anterior do projeto, que foi usada como “dataset oficial” nas primeiras rodadas de baseline.

Hoje é tratado como:

Snapshot histórico para reprodutibilidade

Alguns scripts mais antigos ainda podem referenciar este arquivo

3.3. Geração dos cenários (D5/D7, clima)
Os cenários não são mais salvos como CSVs separados; eles são gerados em memória a partir do arquivo mestre:

D5: subconjunto de amostras com Δ<=5 dias entre a data de campo e a data da imagem S2.

D7: subconjunto com Δ<=7 dias.

clim: inclui as features climáticas.

noclim: remove todas as colunas de clima, usando apenas S2 (e possíveis variáveis auxiliares não climáticas).

Historicamente foram usados arquivos como:

ufms_D5_{clim|noclim}.csv

ufms_D7_{clim|noclim}.csv

Hoje a lógica está embutida nos scripts (filtro por data + seleção de colunas).

3.4. RAW
Em fases exploratórias houve cenários “RAW”, com menos filtros temporais.
Atualmente, os cenários oficiais considerados na análise final são:

D5 e D7, com e sem clima, para CP e TDN_based_ADF.

4. Validação
4.1. LODO por data (oficial)
Estratégia principal de validação.

Cada fold corresponde a uma data/campanha de coleta.

Para cada fold:

Treino em todas as outras datas

Teste na data corrente

Métricas agregadas:

R²

RMSE

MAE

Resultados consolidados em:

reports/progress/UFMS_MASTER_LODO_all.csv

reports/progress/UFMS_MASTER_LODO_champions_by_scenario.csv

reports/progress/UFMS_FINALS_best.csv

4.2. K-Fold e GroupKFold (sanidade)
Além do LODO, foram rodados:

K-Fold simples por amostra

GroupKFold usando a Date como grupo

para checar consistência de:

Tendência das métricas

Robustez dos modelos

Os resultados estão em:

reports/progress/UFMS_MASTER_KFOLD_all.csv

reports/progress/UFMS_MASTER_GKF_all.csv

5. Modelos
5.1. Modelos clássicos
Naive (persistência)

Baseline obrigatório: usa o último valor conhecido como previsão.

Linear Regression

Ridge Regression

Gradient Boosting Regressor (sklearn)

XGBoost Regressor (xgboost)

Scripts principais:

02_baseline_linear.py

02b_baseline_ridge.py

03_baseline_gb.py

04_baseline_xgb.py

5.2. Modelos neurais
MLP (geralmente via PyTorch ou sklearn MLPRegressor nas primeiras versões)

Script base: 05_baseline_mlp.py

Arquitetura simples porém consistente entre cenários

KAN

Scripts 01_*_kan*.py

XNet

Scripts 01_*_xnet*.py

Para KAN e XNet também existem variantes com FS (ex.: *_kan_fs.py, *_xnet_fs.py).

6. Seleção de Features (FS) via XGBoost
6.1. Estratégia
Rodar XGBoost em cada cenário (base × alvo × clima/noclim) com LODO por data.

Para cada fold:

Extrair a importância das features por ganho.

Agregar as importâncias:

Média/mediana do ganho por feature

Frequência com que a feature aparece entre as top-K (ex.: top-15) em cada fold.

Definir um conjunto de features estáveis por cenário:

Geralmente FS15 = top-15 features mais importantes e estáveis.

Os resultados de FS estão em arquivos como:

reports/progress/feature_importance_xgb_stability_CP.csv

reports/progress/feature_importance_xgb_stability_TDN_based_ADF.csv

Arquivos com a lista das features selecionadas (ex.: UFMS_FS_selected_*.txt)

6.2. Re-treino com FS
Depois de definir os conjuntos FS, foram re-treinados (por cenário):

Ridge

GB

XGB

MLP

KAN

XNet

Os resultados comparando “full features” vs FS15 estão em:

reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md

reports/progress/UFMS_TUNED_all.csv

reports/progress/UFMS_TUNED_vs_FINALS.csv

Em resumo:

FS melhora claramente:

Ridge, GB, XGB (principalmente)

FS é neutra ou pior:

MLP, KAN, XNet, especialmente em TDN com clima

7. Resultados principais
Os artefatos consolidados mais importantes são:

reports/progress/UFMS_MASTER_metrics_all.csv
→ Tabela “grande” com todas as execuções registradas.

reports/progress/UFMS_MASTER_LODO_all.csv
→ Foco nas execuções com validação LODO.

reports/progress/UFMS_MASTER_KFOLD_all.csv
→ Execuções com K-Fold simples.

reports/progress/UFMS_MASTER_GKF_all.csv
→ Execuções com GroupKFold por data.

reports/progress/UFMS_MASTER_LODO_champions_by_scenario.csv
→ Campeões por cenário (modelo vencedor em cada combinação base × alvo × clima).

reports/progress/UFMS_FINALS_best.csv
→ Resumo dos melhores modelos (campeões oficiais).

reports/progress/UFMS_CHAMPIONS_LODO.md
→ Explicação textual dos campeões por cenário.

reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md
→ Análise final considerando seleção de features (FS15).

reports/progress/SUMMARY_DISCOVERY.md
→ Narrativa das descobertas principais.

Resumo científico (em linguagem de resultados):

CP é razoavelmente previsível:

R² em torno de 0,30–0,45 em LODO nos cenários mais difíceis

Chegando a ~0,60–0,70 em cenários mais favoráveis (dentro de datas / métricas agregadas específicas)

TDN_based_ADF é muito mais difícil:

Mesmo com clima e FS, R² muitas vezes próximo de 0,00–0,15 em ablações conservadoras

Melhor desempenho com árvores (GB/XGB) em alguns cenários, chegando a valores mais altos mas com maior sensibilidade

Árvores (GB/XGB) são, em geral, mais estáveis e competitivas:

Redes (MLP, KAN, XNet) brilham em cenários específicos, mas não superam consistentemente os modelos de árvore.

8. Ambiente e execução
8.1. Ambiente Docker
O projeto foi pensado para rodar em um container com GPU:

Sistema: Linux (container NVIDIA base de deep learning)

Python: 3.10.x

Principais bibliotecas:

numpy, pandas, scikit-learn

xgboost

pytorch

Outras auxiliares (ver requirements.txt / Dockerfile, se presentes)

Volume padrão:

/workspace dentro do container

Código em /workspace/src

Dados em /workspace/data

Relatórios em /workspace/reports

8.2. Exemplo de execução (ilustrativo)
Exemplo genérico para rodar um baseline MLP:

bash
Copiar código
# Dentro do container /workspace
export PYTHONPATH=/workspace/src:$PYTHONPATH

python /workspace/src/05_baseline_mlp.py \
  --csv /workspace/data/data_raw/Complete_DataSet.csv \
  --date-col Date \
  --target-col CP \
  --scenario D5 \
  --with-climate \
  --val-scheme LODO \
  --out /workspace/reports/exp01_D5_CP_mlp_clim_lodo.csv
Os argumentos exatos (nomes de flags, etc.) podem variar de script para script.
Consulte o cabeçalho de cada arquivo em src/ para a assinatura correta.

9. Histórico do dataset (nota importante)
Para evitar confusão, o histórico é o seguinte:

Fase inicial:

data_processed/Complete_Dataset.csv era tratado como dataset oficial.

A partir dele eram gerados CSVs como ufms_D5_*, ufms_D7_* no disco.

Fase atual (oficial):

data_raw/Complete_DataSet.csv é o arquivo mestre imutável.

Complete_Dataset.csv em data_processed/ é mantido como snapshot histórico (não é mais editado).

D5/D7 e clim/noclim são derivados em memória em cada script, a partir do mestre bruto.

Na dissertação / documentos finais, é esta fase atual que deve ser considerada como filosofia oficial de dados.

10. Diário de bordo (RUN_LOG)
O arquivo RUN_LOG.md registra, em formato livre, os principais passos:

Que script rodou

Com quais parâmetros

Onde foram salvos os outputs

Observações rápidas de resultados (R², problemas, bugs)

Use este arquivo como complemento ao README para entender a linha do tempo das experiências.

makefile
Copiar código
::contentReference[oaicite:0]{index=0}
