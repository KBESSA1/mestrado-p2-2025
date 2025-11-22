# Resumo das Descobertas — Projeto Mestrado UFMS

Autor: Rodrigo Luiz Campos (Kbessa)  
Data de referência: nov/2025  

---

## 1. Contexto geral e objetivos

Este projeto usa um **único dataset mestre**:

- `Complete_DataSet.csv`

sobre o qual são definidos, em código, todos os cenários de modelagem para previsão de:

- **CP**
- **TDN_based_ADF**

a partir de:

- bandas e índices do **Sentinel-2**;
- variáveis climáticas agregadas em janelas temporais (quando ativadas).

Os cenários são definidos por:

- **Base temporal (`base`)**  
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

a partir do qual derivam:

- `UFMS_MASTER_LODO_all.csv` – todos os resultados LODO.  
- `UFMS_MASTER_KFOLD_all.csv` – ablações KFold (KAN/XNet/MLP em regime embaralhado).  
- `UFMS_MASTER_GKF_all.csv` – GroupKFold quando usado.  
- `UFMS_MASTER_LODO_champions_by_scenario.csv` – **campeões LODO por cenário (Base × Target × Clima × FS)**.

O sumário abaixo se baseia **nesses arquivos consolidados**, e não em um único experimento isolado.

---

## 2. Principais descobertas científicas

### 2.1 CP é difícil, mas previsível; TDN_based_ADF é muito mais instável

A partir dos campeões LODO por cenário (`UFMS_MASTER_LODO_champions_by_scenario.csv`):

- Para **CP**:
  - Os melhores modelos em LODO (tipicamente **XGBoost/hgb com ou sem FS**) alcançam **R² por data na faixa ≈ 0,30–0,45**, dependendo de base (RAW/D5/D7) e uso de clima.
  - Erros típicos (RMSE) ficam em torno de **1,0–1,4** unidades, com MAE ≈ **1,0–1,2** para os cenários vencedores.

- Para **TDN_based_ADF**:
  - Mesmo nos melhores cenários (árvores bem ajustadas, com/sem FS), **R² em LODO fica baixo**, geralmente na faixa ≈ **0,00–0,15**.
  - RMSE permanece alto (≈ **3,0–3,7**) com MAE próximo de **3,0**.

**Interpretação:**  
O sinal espectral + clima carrega estrutura suficiente para **CP** atingir um R² moderado em LODO.  
Para **TDN_based_ADF**, a informação disponível (imagem + clima) é claramente insuficiente: o modelo tem dificuldade em distinguir amostras com TDN diferentes quando a validação respeita datas/campanhas.

---

### 2.2 Árvores e ensembles dominam em LODO; redes profundas perdem no regime temporal

Comparando todas as famílias em `UFMS_MASTER_LODO_all.csv`:

- Modelos baseados em **árvores/ensemble** (**hgb**, **xgb/xgbnative**) são os campeões em quase todos os cenários **LODO** (arquivo `UFMS_MASTER_LODO_champions_by_scenario.csv`).
- **Linear/Ridge** formam um degrau intermediário: piores que GB/XGB, mas frequentemente melhores que MLP/KAN/XNet em LODO.
- **Redes neurais (MLP, KAN, XNet)**:
  - Em **LODO**, raramente aparecem como campeãs: R² por data tende a ser baixo ou até negativo, sobretudo em TDN.
  - Em vários cenários, **MLP colapsa** (R² muito negativo, RMSE/MAE bem mais altos que GB/XGB).

Já em **KFold** (`UFMS_MASTER_KFOLD_all.csv`):

- **KAN_full_KFold** e **XNet_full_KFold** atingem R² **muito altos**:
  - para **CP**, há configurações com R² ≈ **0,9** em KFold, RMSE < 1,0;
  - para **TDN_based_ADF**, aparecem R² na casa de **0,6–0,7** em KFold.
- Ou seja, **as redes têm capacidade intrínseca alta**, mas essa capacidade **não se transfere** para o regime LODO.

**Conclusão:**  
A comparação entre `UFMS_MASTER_KFOLD_all.csv` e `UFMS_MASTER_LODO_all.csv` mostra um caso claro de **overfitting metodológico**:  
- em dados embaralhados (KFold), redes profundas parecem “vencer o problema”;  
- em LODO, que simula o uso real (campanhas futuras), **árvores (GB/XGB) são mais estáveis e confiáveis**.

---

### 2.3 Efeito de clima: forte em CP, fraco ou ambíguo em TDN

Da comparação `clim` vs `noclim` em `UFMS_MASTER_LODO_all.csv`:

- Para **CP**:
  - Em várias combinações Base × Modelo, os cenários **com clima** (`Clima = clim`) têm R² maiores e menores RMSE/MAE que os equivalentes **sem clima**.
  - O ganho é mais claro em D5/D7, onde o alinhamento temporal entre imagem, clima e coleta é mais rígido.

- Para **TDN_based_ADF**:
  - O efeito do clima é **inconsistente**:
    - Em alguns cenários, clim e noclim empatam dentro da variabilidade.
    - Em outros, noclim chega a ser ligeiramente melhor (clima acrescenta ruído/shift entre campanhas).

**Conclusão:**  
O clima captura variações fisiológicas relevantes para **CP**, mas não resolve o problema de TDN.  
Para TDN, o gargalo parece ser ausência de outras fontes (solo, manejo, água, composição detalhada).

---

### 2.4 Seleção de atributos (FS via XGBoost) ajuda modelos estruturados, mas não redes

Os scripts de FS (`02_*_ridge_fs.py`, `03_*_gb_fs.py`, `02_feature_importance_xgb_all.py`, `ufms_fs_ranker.py`, `ufms_make_fs_csv.py`) e os relatórios em `reports/progress` e `reports/ablations` mostram:

- A política **FS15** (top-K estáveis via XGBoost ganho) foi aplicada por cenário para gerar CSVs reduzidos.
- Em **GB/XGB/Ridge**, usar FS15:
  - reduz variância entre folds/campanhas;
  - melhora ou mantém R² em LODO;
  - simplifica o espaço de entrada.

- Em **MLP, KAN e XNet**:
  - FS15 é frequentemente **neutra ou prejudicial**, especialmente em TDN com clima;
  - as redes parecem preferir o espaço de atributos completo para extrair suas próprias representações.

**Conclusão:**  
- **Modelos estruturados** (árvores + Ridge) **ganham** com FS explícita.  
- **Redes profundas** preferem **mais dados e menos poda de atributos**, especialmente fora de LODO (KFold).

---

### 2.5 Gargalo científico: número de amostras por data e riqueza de informação

Mesmo com todos os ajustes (D5/D7, clima, FS, KAN/XNet):

- O dataset tem **~312 amostras totais**, mas **poucas por data/campanha**.
- Em **LODO**, isso aparece como:
  - folds de teste pequenos;
  - grande variabilidade entre campanhas;
  - R² modestos mesmo para os melhores modelos (principalmente em TDN).

Já nas ablações **KFold** (`UFMS_MASTER_KFOLD_all.csv`):

- KAN e XNet alcançam R² altos em CP e TDN, confirmando boa capacidade de aproximação.
- A diferença de desempenho entre KFold e LODO evidencia que **o problema não é “falta de modelo”**, mas sim:
  - densidade temporal insuficiente;
  - pouca diversidade espectral/espacial por campanha;
  - ausência de fontes complementares (solo, manejo, laboratório detalhado).

**Conclusão:**  
O limite atual é o **dataset** (tempo, espaço, espectro e contexto agronômico), não a ausência de modelos sofisticados.  
Isso abre um caminho direto para um **doutorado** focado em:

- mais datas e campanhas por área;  
- sensores adicionais e bandas derivadas;  
- integração com variáveis de solo, manejo, água e análises laboratoriais adicionais.

---

## 3. Contribuições concretas do trabalho

### 3.1 Padronização de LODO correto em dataset agrícola pequeno

- Implementação de funções utilitárias em `00_utils_lodo.py` / `utils_lodo.py`.
- Scripts dedicados por cenário (`01_cp_*`, `01_tdn_*`) que:
  - definem grupos por data/campanha;
  - evitam vazamento temporal;
  - salvam métricas e predições fold a fold.

Resultado: **validação replicável e sem vazamento**, adequada à pergunta “consigo prever uma nova campanha?”.

---

### 3.2 Varredura sistemática de famílias de modelos

Foram testadas, em múltiplos cenários RAW/D5/D7 × clim/noclim × CP/TDN:

- Baselines simples: naive, linear, ridge.
- Árvores / ensembles: hgb, xgb/xgbnative.
- Redes: mlp, KAN, XNet.

Poucos trabalhos de mestrado com dataset agrícola tão pequeno:

- cobrem esse espectro de modelos;
- documentam de forma sistemática os **limites em LODO**;
- contrapõem resultados “bonitos” em KFold com a realidade temporal.

---

### 3.3 Demonstração prática de overfitting metodológico

Comparando:

- `UFMS_MASTER_KFOLD_all.csv` (KAN/XNet/MLP em KFold), com  
- `UFMS_MASTER_LODO_all.csv` (todos os modelos em LODO),

fica evidente que:

- R² altíssimos em KFold para KAN/XNet **não se sustentam** quando o dado é particionado por data;
- a escolha da validação muda completamente a narrativa sobre “qual modelo é melhor”.

Este trabalho traz um exemplo concreto de **overfitting de metodologia** em sensoriamento remoto agrícola:  
“otimizar modelo em KFold e reportar como se fosse cenário de uso real” produz conclusões enganosas.

---

### 3.4 Política de seleção de atributos reprodutível (FS via XGBoost)

- Importâncias por cenário calculadas com `02_feature_importance_xgb_all.py` + `03_feature_importance_summary.py`.
- Rankings e conjuntos selecionados documentados em:
  - `data/feature_sets/*.features.txt`
  - `data/feature_selected/*.csv`
- Experimentos de FS acoplados a Ridge e GB em scripts dedicados (`02_*_ridge_fs.py`, `03_*_gb_fs.py`).

Isso garante que qualquer pessoa consiga:

1. Reproduzir os conjuntos FS15.  
2. Refazer os testes em outros modelos.  
3. Expandir a política (doutorado, outros sensores, outros alvos).

---

### 3.5 Pipeline reprodutível de ponta a ponta

- Ambiente descrito com Docker + configs de Python/Torch/XGBoost.
- Código organizado em `src/` por função (baselines, LODO, FS, ablações, pré-processamento Sentinel/HLS).
- Resultados agregados em arquivos “master” dentro de `reports/progress/`.

O repositório permite **replay completo** das rodadas principais, além de novas combinações de cenários.

---

## 4. Panorama de métricas (LODO e KFold)

As métricas não são replicadas integralmente neste arquivo; a fonte de verdade são os CSVs consolidados.

### 4.1 LODO – matriz completa

Matriz com **todas** as combinações rodadas em LODO:

- `reports/progress/UFMS_MASTER_LODO_all.csv`

Contém, por linha:

- Base, Target, Modelo, Clima, FS, CV_type = LODO;  
- R2, RMSE, MAE;  
- caminho do arquivo de origem.

### 4.2 LODO – campeões por cenário

Arquivo com **um campeão por combinação (Base × Target × Clima × FS)**, escolhido pelo maior R² (ou menor RMSE quando R² ausente):

- `reports/progress/UFMS_MASTER_LODO_champions_by_scenario.csv`

Se desejado, pode-se colar aqui uma tabela extraída diretamente desse CSV:

```markdown
<!-- TABELA_METRICAS_INICIO -->
(cole aqui a tabela gerada por script a partir de UFMS_MASTER_LODO_champions_by_scenario.csv)
<!-- TABELA_METRICAS_FIM -->
