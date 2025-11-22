# Resumo das Descobertas — Projeto Mestrado UFMS

Autor: Rodrigo Luiz Campos (Kbessa)  
Data de referência: nov/2025  

---

## 1. Contexto geral e objetivos

Este projeto usa um **único dataset mestre**:

- `data/data_raw/Complete_DataSet.csv`

sobre o qual são definidos, em código, todos os cenários de modelagem para previsão de:

- **CP**
- **TDN_based_ADF**

a partir de:

- bandas e índices do **Sentinel-2**;
- variáveis climáticas agregadas em janelas temporais (quando ativadas).

Os cenários são definidos por:

- **Base temporal (`Base`)**  
  - **RAW** – sem filtro de Δdias entre coleta e imagem.  
  - **D5** – Δdias ≤ 5.  
  - **D7** – Δdias ≤ 7.
- **Uso de clima (`Clima`)**  
  - **clim** – Sentinel-2 + clima.  
  - **noclim** – apenas Sentinel-2.
- **Alvo (`Target`)**  
  - **CP**  
  - **TDN_based_ADF**
- **Família de modelo (`Modelo`)**  
  - Baselines: **naive**, **linear**, **ridge**.  
  - Árvores/ensemble: **hgb** (HistGradientBoosting), **xgb / xgbnative** (XGBoost).  
  - Redes neurais: **mlp**, **KAN**, **XNet**.
- **Validação (`CV`)**  
  - **LODO** – Leave-One-Date-Out (validação oficial).  
  - **KFold / GKF** – usados como ablações para testar capacidade intrínseca.

Todos os arquivos de métricas dos scripts em `src/` foram consolidados em:

- `reports/progress/UFMS_MASTER_metrics_all.csv` (5924 linhas)

a partir do qual derivam, entre outros:

- `UFMS_MASTER_LODO_all.csv` – todos os resultados LODO.  
- `UFMS_MASTER_KFOLD_all.csv` – ablações KFold (KAN/XNet/MLP em regime embaralhado).  
- `UFMS_MASTER_GKF_all.csv` – GroupKFold quando usado.  
- `UFMS_MASTER_LODO_champions_by_scenario.csv` – campeões LODO por cenário (Base × Target × Clima × FS).  
- `UFMS_TAKASHI_LODO_champions_auto.csv` – campeões LODO por cenário (versão auto-gerada).  
- `UFMS_TAKASHI_model_stats_auto.csv` – estatísticas de R² por modelo/alvo/CV.

O sumário abaixo se baseia **nesses arquivos consolidados**, e não em um único experimento isolado.

---

## 2. Principais descobertas científicas

### 2.1 CP e TDN_based_ADF: o que é previsível em LODO?

Usando apenas os **campeões em LODO** por cenário, a partir de  
`UFMS_TAKASHI_LODO_champions_auto.csv`:

- Para **CP**, os melhores cenários em LODO atingem:

  - **D7, CP, clim, FS, MLP**  
    - R² ≈ **0,39** (0,385)  
    - RMSE ≈ **1,27**  
    - MAE ≈ **1,05**

  - **D7, CP, noclim, FS, MLP**  
    - R² ≈ **0,32** (0,324)  
    - RMSE ≈ **1,32**  
    - MAE ≈ **1,08**

  - **D5, CP, clim, FS, MLP**  
    - R² ≈ **0,35** (0,346)  
    - RMSE ≈ **1,30**  
    - MAE ≈ **1,11**

  - **RAW, CP, clim, FS, MLP**  
    - R² ≈ **0,31** (0,309)  
    - RMSE ≈ **1,51**  
    - MAE ≈ **1,27**

  Ou seja, **CP** é moderadamente previsível em LODO, com R² por data tipicamente na faixa  
  **0,30–0,40**, dependendo de base e uso de clima.

- Para **TDN_based_ADF**, os campeões LODO são dominados por **XGBoost com FS**:

  - **RAW, TDN, clim, FS, xgbnative**  
    - R² ≈ **0,39** (0,388)  
    - RMSE ≈ **3,32**  
    - MAE ≈ **2,57**

  - **D7, TDN, clim, FS, xgbnative**  
    - R² ≈ **0,37** (0,366)  
    - RMSE ≈ **3,30**  
    - MAE ≈ **2,59**

  - **RAW, TDN, noclim, FS, xgbnative**  
    - R² ≈ **0,34** (0,344)  
    - RMSE ≈ **3,45**  
    - MAE ≈ **2,83**

  - **D7, TDN, noclim, FS, xgbnative**  
    - R² ≈ **0,34** (0,338)  
    - RMSE ≈ **3,40**  
    - MAE ≈ **2,75**

  - **D5, TDN, clim/noclim, FS, xgbnative**  
    - R² ≈ **0,24–0,25**  
    - RMSE ≈ **3,6–3,7**  
    - MAE ≈ **2,9–3,0**

Em resumo:

- **CP**: R² em LODO **~0,30–0,40**, RMSE ≈ 1,25–1,55.  
- **TDN_based_ADF**: R² em LODO **~0,25–0,39**, RMSE ≈ 3,3–3,7.

Os valores para TDN são melhores do que os da fase inicial do projeto.  
Ainda assim, os erros absolutos (RMSE ≈ 3,3–3,7) e a sensibilidade a cenário indicam que o modelo está captando **parte** da variabilidade de TDN, mas ainda longe de um “nível operacional confortável”.

---

### 2.2 Quem manda em LODO: árvores vs redes

Pelas estatísticas de `UFMS_TAKASHI_model_stats_auto.csv`:

- **Target = CP, CV = lodo**  

  - `xgbnative`  
    - R² médio ≈ **0,19**, R² máx. ≈ **0,89** (52 runs)  
  - `hgb`  
    - R² médio ≈ **-0,13**, R² máx. ≈ **0,66** (43 runs)  
  - `mlp`  
    - R² médio ≈ **-0,30**, R² máx. ≈ **0,73** (56 runs)

- **Target = TDN_based_ADF, CV = lodo**

  - `xgbnative`  
    - R² médio ≈ **0,31**, R² máx. ≈ **0,47** (51 runs)  
  - `hgb`  
    - R² médio ≈ **0,00**, R² máx. ≈ **0,20** (30 runs)  
  - `mlp`  
    - R² médio ≈ **-12,29**, R² máx. ≈ **-3,13** (30 runs)

**Leitura:**

- Nas rodadas LODO, os **melhores modelos por cenário** são em geral:
  - Para **CP**: MLP com FS (em campeões), mas XGBoost com FS também aparece forte.  
  - Para **TDN**: **XGBoost com FS** domina.

- Em termos de **R² médio por família**, quem se mantém “no azul” em LODO é:
  - **xgbnative** (árvore/ensemble) – tanto em CP quanto em TDN.  
  - `hgb` (HistGradientBoosting) fica instável, com R² médio próximo de zero.  
  - `mlp` em TDN tem desempenho muito ruim em LODO.

Ou seja, **no regime temporal correto (LODO), árvores/ensembles são os mais estáveis**.  
Redes densas (MLP) conseguem campeonatos em alguns cenários de CP, mas são **muito frágeis em TDN** e, em média, perdem para XGBoost quando se olha o conjunto completo de rodadas.

---

### 2.3 KFold vs LODO: capacidade intrínseca vs cenário real

Pelos mesmos arquivos, a diferença **KFold × LODO** é gritante:

- **Target = CP, CV = kfold**  
  - `KAN_full_KFold`  
    - R² médio ≈ **0,84**, R² máx. ≈ **0,94** (6 runs)  
  - `XNet_full_KFold`  
    - R² médio ≈ **0,79**, R² máx. ≈ **0,89** (6 runs)

- **Target = TDN_based_ADF, CV = kfold**  
  - `XNet_full_KFold`  
    - R² médio ≈ **0,69**, R² máx. ≈ **0,84** (6 runs)  
  - `KAN_full_KFold`  
    - R² médio ≈ **0,55**, R² máx. ≈ **0,69** (6 runs)

Comparando com LODO:

- **CP em LODO**:
  - `xgbnative` → R² médio ≈ **0,19**; campeões por cenário ≈ **0,30–0,39**.  
  - `mlp` → R² médio ≈ **negativo**, apesar de alguns picos ≈ **0,73**.  
  - `KAN`/`XNet` não aparecem entre campeões LODO.

- **TDN em LODO**:
  - `xgbnative` → R² médio ≈ **0,31**, máx. ≈ **0,47**.  
  - `mlp` → R² médio fortemente negativo (colapso).  
  - `KAN`/`XNet` só brilham em KFold, não em LODO.

**Conclusão:**  

- Em **KFold embaralhado**, KAN e XNet “resolvem o problema” (CP e TDN com R² ≫ 0,7).  
- Em **LODO**, que respeita a estrutura por data/campanha, essa “magia” desaparece, e quem sustenta desempenho razoável são **ensembles de árvores com FS**.

Isso ilustra um caso claro de **overfitting metodológico**:

> “Se eu treino e valido tudo embaralhado, parece que redes profundas resolvem o problema.  
> Quando respeito a estrutura temporal real (LODO), o cenário muda completamente.”

---

### 2.4 Efeito de clima

Usando os campeões LODO de `UFMS_TAKASHI_LODO_champions_auto.csv`:

- Em **CP**:
  - Os melhores cenários usam, em geral, **clima + FS + MLP**.  
  - D7/clim/FS/MLP é o campeão global de CP em LODO (R² ≈ 0,39).  
  - Cenários equivalentes sem clima (noclim) têm R² um pouco menor, mas ainda razoável.

- Em **TDN_based_ADF**:
  - Tanto **clim** quanto **noclim** têm campeões próximos (R² ≈ 0,34–0,39).  
  - O clima melhora alguns cenários, mas **não resolve** o problema de forma dramática; o ganho é menor que em CP.

**Leitura agronômica:**

- Para **CP**, o clima adiciona informação fisiológica relevante (crescimento, estresse, água).  
- Para **TDN**, o gargalo está mais em **solo, manejo, água, composição de parede celular** etc. – coisas que nem S2 nem clima conseguem capturar bem sozinhos.

---

### 2.5 Seleção de atributos (FS) via XGBoost

Os experimentos de FS estão documentados em:

- `data/feature_sets/*.features.txt`  
- `data/feature_selected/*.csv`  
- Relatórios de FS: `UFMS_FINAL_REPORT_FS15_LODO.md`, `UFMS_TUNED_all.csv`, `UFMS_TUNED_vs_FINALS.csv`.

Os resultados convergem para o seguinte padrão:

- **FS15** (top-15 features estáveis por cenário, via ganho do XGBoost em LODO):

  - Para **árvores e Ridge**:
    - Reduz variância entre folds/campanhas.  
    - Em muitos cenários, **aumenta R² em LODO**.  
    - Simplifica o espaço de entrada sem perder desempenho.

  - Para **MLP, KAN, XNet**:
    - FS costuma ser **neutra ou negativa**, principalmente em **TDN com clima**.  
    - As redes parecem preferir o espaço de atributos completo para aprender suas próprias combinações.

Em termos práticos:

- A FS via XGBoost é **uma arma muito boa para modelos estruturados** (ensembles, Ridge).  
- Para redes, a FS deve ser usada com cuidado, e possivelmente substituída por regularização / arquitetura ou deixada para um cenário com mais dados.

---

### 2.6 Gargalo científico: dados por data vs arquitetura de modelo

Mesmo com:

- múltiplas bases (RAW/D5/D7),  
- clima on/off,  
- FS via XGBoost,  
- redes sofisticadas (KAN/XNet),

os limites observados em LODO apontam fortemente para:

- **poucas amostras por data/campanha**;  
- grande variabilidade entre campanhas;  
- falta de fontes adicionais (solo, manejo, água, análises detalhadas).

A comparação **KFold vs LODO** mostra que:

- **Modelos têm capacidade de sobra** (KAN/XNet com R² > 0,8 em KFold);  
- O que falta é **densidade e diversidade de dados no eixo tempo/campanha**, além de variáveis agronômicas adicionais.

Isso aponta um caminho direto de **continuidade em doutorado**:

- Mais campanhas por área (densidade temporal).  
- Integração de múltiplos sensores (S2, HLS, SAR, hiperespectral).  
- Variáveis de manejo, solo, água, etc.  
- Exploração de redes (KAN/XNet) **no regime temporal correto** (LODO) com um dataset mais rico.

---

## 3. Contribuições concretas do trabalho

### 3.1 Padronização de LODO correto em dataset agrícola pequeno

- Implementação de utilitários em `utils_lodo.py` e scripts dedicados por cenário (`01_cp_*`, `01_tdn_*`).  
- Validação por data/campanha (`Date`) como grupo de LODO, evitando vazamento temporal.

Resultado: **pipeline de validação replicável e sem vazamento**, compatível com a pergunta prática:

> “Se eu treinar até hoje, consigo prever a próxima campanha?”

---

### 3.2 Varredura sistemática de famílias de modelos

Cobriu-se, de forma organizada, o espaço:

- Baselines: naive, linear, ridge.  
- Árvores/ensembles: hgb, xgb/xgbnative.  
- Redes: mlp, KAN, XNet (com e sem FS).  

Poucos trabalhos de mestrado em sensoriamento remoto agrícola, com dataset tão pequeno, documentam de forma tão sistemática:

- o desempenho em **LODO** por cenário;  
- a diferença radical entre LODO e KFold;  
- o comportamento conjunto de modelos clássicos e redes modernas.

---

### 3.3 Demonstração quantitativa de overfitting metodológico

Este projeto fornece um **exemplo numérico concreto** de como a escolha da validação muda a narrativa:

- Em **KFold**:
  - KAN/XNet têm R² médio **> 0,8** em CP e **> 0,55** em TDN.  

- Em **LODO**:
  - Quem sustenta desempenho razoável são **árvores/ensembles com FS** (xgbnative), com R² médio ≈ 0,19 (CP) e ≈ 0,31 (TDN).  
  - KAN/XNet desaparecem dos campeonatos em LODO.

Isso permite discutir com clareza, em aula ou banca:

> “Se eu tivesse ficado só no KFold, eu diria que redes resolvem tudo.  
> Mas em LODO, que simula uso real, a história muda: árvores com FS são os modelos mais robustos.”

---

### 3.4 Política de seleção de atributos reprodutível

A política FS via XGBoost:

1. Calcula importâncias por ganho em LODO por cenário.  
2. Agrega por estabilidade (média/mediana + frequência em top-K).  
3. Define conjuntos FS15 por cenário.  
4. Gera CSVs reduzidos (`data/feature_selected/`) e listas (`data/feature_sets/`).  
5. Re-treina modelos com/sem FS, comparando em LODO.

Esse pipeline é **reaproveitável**:

- em outros alvos (ex.: FDN, FDA, minerais);  
- em outros sensores;  
- em trabalhos futuros (doutorado).

---

### 3.5 Pipeline reprodutível de ponta a ponta

- Ambiente Docker padronizado (CUDA, Python 3.10, PyTorch, XGBoost).  
- Código organizado em `src/` por função (baselines, LODO, FS, HLS/Sentinel, ablações).  
- Métricas consolidadas em arquivos “master” (`UFMS_MASTER_*.csv` e `UFMS_TAKASHI_*.csv`).

Qualquer pessoa com acesso ao repositório consegue:

- refazer os experimentos principais;  
- adicionar novos cenários;  
- reusar os artefatos de FS e validação.

---

## 4. Panorama de métricas (onde olhar no repositório)

As métricas detalhadas não são replicadas integralmente aqui. Os arquivos-chave são:

- **LODO (validação oficial)**  
  - `reports/progress/UFMS_MASTER_LODO_all.csv`  
  - `reports/progress/UFMS_MASTER_LODO_champions_by_scenario.csv`  
  - `reports/progress/UFMS_TAKASHI_LODO_champions_auto.csv`

- **KFold / GroupKFold (ablações)**  
  - `reports/progress/UFMS_MASTER_KFOLD_all.csv`  
  - `reports/progress/UFMS_MASTER_GKF_all.csv`  
  - Rodadas específicas de KAN/XNet em `reports/ablations/`.

- **Estatísticas de modelos**  
  - `reports/progress/UFMS_TAKASHI_model_stats_auto.csv`  
    (R² médio / máximo por Modelo × Target × CV)

- **FS e comparações FS vs full**  
  - `reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md`  
  - `reports/progress/UFMS_TUNED_all.csv`  
  - `reports/progress/UFMS_TUNED_vs_FINALS.csv`

---

## 5. Guia rápido de leitura para o orientador

Sugestão de roteiro para leitura do repositório:

1. Ler este arquivo: `SUMMARY_DISCOVERY.md`  
   → visão geral do que foi feito e dos principais números.
2. Ver os **campeões em LODO** por cenário:  
   `UFMS_TAKASHI_LODO_champions_auto.csv`
3. Comparar com a **matriz completa LODO**:  
   `UFMS_MASTER_LODO_all.csv`
4. Explorar **R² médio por modelo/alvo/CV**:  
   `UFMS_TAKASHI_model_stats_auto.csv`
5. Ver **KFold vs LODO** para KAN/XNet:  
   `UFMS_MASTER_KFOLD_all.csv` e arquivos de ablações em `reports/ablations/`.
6. Examinar a política de FS e os conjuntos de atributos:  
   `data/feature_sets/` e `data/feature_selected/`.

---

## 6. Conclusão geral

Este trabalho mostra, de forma quantitativa e reprodutível, que:

- **CP** é **moderadamente previsível** em LODO com Sentinel-2 + clima, com R² ≈ 0,30–0,40 nos melhores cenários.  
- **TDN_based_ADF** apresenta **previsibilidade limitada**, com R² ≈ 0,25–0,39 e erros absolutos ainda altos, mesmo com clima e FS.

Em termos de modelos:

- **Árvores/ensembles (XGBoost com FS)** são os modelos **mais robustos em LODO**.  
- **Redes profundas (MLP, KAN, XNet)** atingem R² muito altos em KFold, mas **não transferem esse desempenho** para o regime temporal correto.  
- A diferença **KFold vs LODO** é um exemplo claro de **overfitting metodológico** em sensoriamento remoto agrícola.

Por outro lado, os resultados também mostram que:

- O gargalo principal **não é a arquitetura**, mas sim o **volume e a riqueza dos dados** (temporais, espectrais e agronômicos).  
- Existe um caminho natural de **continuidade em doutorado**, explorando:
  - maior densidade temporal e mais campanhas,  
  - múltiplos sensores e bandas derivadas,  
  - variáveis de manejo, solo e água,  
  - e uma exploração mais profunda de KAN/XNet em **regime LODO**, com mais dados por campanha.

Este repositório, com seus arquivos “master” de métricas e scripts de experimentos, documenta essa trajetória de forma que outros pesquisadores possam **replicar, criticar e estender** o trabalho.
