# Resumo das Descobertas — Projeto Mestrado UFMS

Autor: Rodrigo Luiz Campos (Kbessa)  
Data de referência: nov/2025  

---

## 1. Contexto e objetivo

Todo o trabalho de modelagem foi feito a partir de **um único dataset mestre**:

- `Complete_DataSet.csv`

A partir desse arquivo, os cenários foram construídos **em código**, sem gerar novos CSVs derivados, variando apenas:

- **Base temporal** (campo `base`):
  - **RAW** – sem restrição de diferença entre data de coleta e data da imagem.
  - **D5** – Δdias ≤ 5.
  - **D7** – Δdias ≤ 7.
- **Uso de clima** (coluna `Clima`):
  - **clim** – Sentinel-2 + variáveis climáticas agregadas.
  - **noclim** – somente Sentinel-2 (sem features de clima).
- **Alvo (Target)**:
  - **CP**
  - **TDN_based_ADF**
- **Modelo (Modelo)**:
  - **hgb** – HistGradientBoosting (árvore/ensemble).
  - **xgbnative** – XGBoost nativo.
  - **mlp** – rede neural MLP simples.
- **Validação (CV)**:
  - nesta tabela, sempre **`lodo`** – Leave-One-Date-Out (cada data/campanha inteira fica em teste em algum fold).

As métricas mostradas aqui são:

- **R2** – coeficiente de determinação em regime LODO.
- **RMSE** – erro quadrático médio.
- **MAE** – erro absoluto médio.

Outros modelos da baseline (Naive, Linear, Ridge, KAN, XNet) e variações adicionais aparecem no CSV completo (`UFMS_ALLMODELS_metrics_LODO.csv`), mas aqui focamos nos três modelos principais de comparação (hgb, xgbnative, mlp) para manter o resumo legível.

---

## 2. Padrões principais observados na matriz de resultados

### 2.1 Comportamento geral de CP

Para **CP**, em D5, D7 e RAW:

- **xgbnative** é o único modelo que consegue **R² levemente positivos** em todos os cenários:
  - D5, CP, clim: R² ≈ 0.08  
  - D7, CP, clim: R² ≈ 0.18  
  - RAW, CP, clim/noclim: R² ≈ 0.22–0.24  
- **hgb** fica sempre com **R² levemente negativos** (entre ≈ -0.55 e -0.06), apesar de RMSE/MAE relativamente baixos.
- **mlp** apresenta **R² bem negativos**, especialmente com clima (por exemplo, D7, CP, clim: R² ≈ -3.49), indicando que a rede neural falha em generalizar em regime LODO.

Em termos de erro:

- Os menores **RMSE/MAE** para CP aparecem sistematicamente em **xgbnative** (por exemplo, D7, CP, clim: RMSE ≈ 1.37, MAE ≈ 1.15).
- **mlp** é sempre o pior em RMSE/MAE entre os três modelos.

**Leitura para CP:**  
Em LODO, CP é **difícil**, mas **xgbnative** ainda consegue extrair um pouco de sinal (R² pequeno, porém positivo) e manter os menores erros. HGB é estável mas praticamente empata com um baseline fraco em termos de R². A MLP não se adapta bem ao regime temporal LODO.

---

### 2.2 Comportamento geral de TDN_based_ADF

Para **TDN_based_ADF**, o padrão é parecido, mas com um detalhe importante:

- **xgbnative** atinge os **maiores R² da tabela**, todos ainda modestos, mas **melhores que em CP**:
  - D5, TDN, clim/noclim: R² ≈ 0.24–0.25  
  - D7, TDN, clim/noclim: R² ≈ 0.28–0.30  
  - RAW, TDN, clim/noclim: R² ≈ 0.33–0.34  
- **hgb** oscila próximo de zero (R² entre ≈ -0.11 e ≈ 0.05).
- **mlp** tem **R² extremamente negativos** para TDN, especialmente sem clima:
  - D5, TDN, noclim: R² ≈ -18.16  
  - D7, TDN, noclim: R² ≈ -23.16  
  - RAW, TDN, noclim: R² ≈ -22.67  

Em termos de erro:

- Para TDN, **xgbnative** mantém RMSE ~3.4–3.7 e MAE ~2.8–3.0 em todos os cenários.
- HGB fica um pouco acima (RMSE ~4.0–4.3).
- MLP explode para RMSE ~11–17 e MAE ~10–15.

**Leitura para TDN:**  
Mesmo sendo um alvo difícil, **xgbnative** consegue capturar mais estrutura relativa em TDN do que hgb e, principalmente, muito mais do que a MLP (que simplesmente colapsa em LODO).

---

### 2.3 Efeito de clima (clim vs noclim)

Comparando pares **clim vs noclim** na mesma Base/Target/Modelo:

- **CP**:
  - Em **D5** e **D7**, **xgbnative com clima** (clim) tende a ter **R² maior** do que sem clima:
    - D5, CP, xgbnative: 0.082 (clim) vs -0.010 (noclim).  
    - D7, CP, xgbnative: 0.176 (clim) vs 0.037 (noclim).
  - Em **RAW**, a diferença é pequena (R² ≈ 0.221–0.235), sugerindo que o clima ajuda mais quando o recorte temporal é mais restrito (D5/D7).
- **TDN_based_ADF**:
  - Para xgbnative, **noclim** geralmente é um pouco melhor:
    - D5, TDN, xgbnative: R² ≈ 0.235 (clim) vs 0.246 (noclim).  
    - D7, TDN, xgbnative: 0.277 (clim) vs 0.300 (noclim).  
    - RAW, TDN, xgbnative: 0.329 (clim) vs 0.344 (noclim).
  - Para hgb, clima vs noclim faz pouca diferença (R² bem próximo de zero).
  - Para mlp, tanto com clima quanto sem clima o comportamento é ruim, com R² muito negativos (clima tende a piorar ainda mais).

**Resumo do clima:**

- Para **CP**, clima (clim) tende a **ajudar** o xgbnative, principalmente em D5/D7.
- Para **TDN**, clima é **neutro ou ligeiramente prejudicial** para xgbnative; o melhor R² costuma aparecer em cenários **noclim**.
- Em MLP, o clima mais atrapalha do que ajuda em LODO.

---

### 2.4 Diferenças entre bases RAW, D5 e D7

Analisando **xgbnative**, que é o modelo mais consistente:

- **CP**:
  - D5: R² ≈ 0.08 (clim) / ≈ -0.01 (noclim)  
  - D7: R² ≈ 0.18 (clim) / ≈ 0.04 (noclim)  
  - RAW: R² ≈ 0.22–0.24  
  → A performance sobe levemente na direção de RAW, mas as diferenças são pequenas. D7 já concentra boa parte do sinal.

- **TDN_based_ADF**:
  - D5: R² ≈ 0.24–0.25  
  - D7: R² ≈ 0.28–0.30  
  - RAW: R² ≈ 0.33–0.34  
  → Há um ganho gradual de R² de D5 → D7 → RAW para TDN, ainda que os valores sejam baixos/modestos.

**Leitura geral:**  
As três bases (D5, D7, RAW) são todas “difíceis” em LODO. O **xgbnative** se adapta melhor em todas, com ligeira vantagem para RAW em TDN e para D7/RAW em CP. MLP é fraca em todos.

---

## 3. Tabela de métricas LODO por cenário e modelo

A tabela abaixo é a saída direta do script aplicado em  
`reports/progress/UFMS_ALLMODELS_metrics_LODO.csv`  
para os modelos **hgb**, **mlp** e **xgbnative**, nos cenários **RAW, D5 e D7**, com e sem clima (coluna `Clima`), para CP e TDN_based_ADF.

### 3.1 R2, RMSE e MAE por Base, Target, Modelo, Clima (CV = lodo)

<!-- TABELA_METRICAS_INICIO -->

| Base | Target | Modelo | Clima | FS | CV | R2 | RMSE | MAE | Arquivo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D5 | CP | hgb | clim | desconhecido | lodo | -0.304 | 1.708 | 1.509 | /workspace/reports/exp02/exp02_D5_CP_hgb_clim_lodo.csv |
| D5 | CP | mlp | clim | desconhecido | lodo | -2.995 | 2.086 | 1.826 | /workspace/reports/exp02/exp02_D5_CP_mlp_clim_lodo.csv |
| D5 | CP | xgbnative | clim | desconhecido | lodo | 0.082 | 1.441 | 1.240 | /workspace/reports/exp02/exp02_D5_CP_xgbnative_clim_lodo.csv |
| D5 | CP | hgb | noclim | desconhecido | lodo | -0.552 | 1.857 | 1.612 | /workspace/reports/exp02/exp02_D5_CP_hgb_noclim_lodo.csv |
| D5 | CP | mlp | noclim | desconhecido | lodo | -0.793 | 2.008 | 1.671 | /workspace/reports/exp02/exp02_D5_CP_mlp_noclim_lodo.csv |
| D5 | CP | xgbnative | noclim | desconhecido | lodo | -0.010 | 1.510 | 1.302 | /workspace/reports/exp02/exp02_D5_CP_xgbnative_noclim_lodo.csv |
| D5 | TDN_based_ADF | hgb | clim | desconhecido | lodo | -0.078 | 4.214 | 3.394 | /workspace/reports/exp02/exp02_D5_TDN_based_ADF_hgb_clim_lodo.csv |
| D5 | TDN_based_ADF | mlp | clim | desconhecido | lodo | -13.999 | 13.642 | 12.069 | /workspace/reports/exp02/exp02_D5_TDN_based_ADF_mlp_clim_lodo.csv |
| D5 | TDN_based_ADF | xgbnative | clim | desconhecido | lodo | 0.235 | 3.707 | 3.026 | /workspace/reports/exp02/exp02_D5_TDN_based_ADF_xgbnative_clim_lodo.csv |
| D5 | TDN_based_ADF | hgb | noclim | desconhecido | lodo | -0.112 | 4.233 | 3.417 | /workspace/reports/exp02/exp02_D5_TDN_based_ADF_hgb_noclim_lodo.csv |
| D5 | TDN_based_ADF | mlp | noclim | desconhecido | lodo | -18.160 | 16.036 | 13.710 | /workspace/reports/exp02/exp02_D5_TDN_based_ADF_mlp_noclim_lodo.csv |
| D5 | TDN_based_ADF | xgbnative | noclim | desconhecido | lodo | 0.246 | 3.584 | 2.935 | /workspace/reports/exp02/exp02_D5_TDN_based_ADF_xgbnative_noclim_lodo.csv |
| D7 | CP | hgb | clim | desconhecido | lodo | -0.310 | 1.704 | 1.453 | /workspace/reports/exp02/exp02_D7_CP_hgb_clim_lodo.csv |
| D7 | CP | mlp | clim | desconhecido | lodo | -3.494 | 1.990 | 1.743 | /workspace/reports/exp02/exp02_D7_CP_mlp_clim_lodo.csv |
| D7 | CP | xgbnative | clim | desconhecido | lodo | 0.176 | 1.365 | 1.146 | /workspace/reports/exp02/exp02_D7_CP_xgbnative_clim_lodo.csv |
| D7 | CP | hgb | noclim | desconhecido | lodo | -0.257 | 1.693 | 1.431 | /workspace/reports/exp02/exp02_D7_CP_hgb_noclim_lodo.csv |
| D7 | CP | mlp | noclim | desconhecido | lodo | -0.532 | 1.882 | 1.559 | /workspace/reports/exp02/exp02_D7_CP_mlp_noclim_lodo.csv |
| D7 | CP | xgbnative | noclim | desconhecido | lodo | 0.037 | 1.491 | 1.244 | /workspace/reports/exp02/exp02_D7_CP_xgbnative_noclim_lodo.csv |
| D7 | TDN_based_ADF | hgb | clim | desconhecido | lodo | 0.007 | 4.105 | 3.381 | /workspace/reports/exp02/exp02_D7_TDN_based_ADF_hgb_clim_lodo.csv |
| D7 | TDN_based_ADF | mlp | clim | desconhecido | lodo | -18.901 | 15.017 | 13.393 | /workspace/reports/exp02/exp02_D7_TDN_based_ADF_mlp_clim_lodo.csv |
| D7 | TDN_based_ADF | xgbnative | clim | desconhecido | lodo | 0.277 | 3.481 | 2.841 | /workspace/reports/exp02/exp02_D7_TDN_based_ADF_xgbnative_clim_lodo.csv |
| D7 | TDN_based_ADF | hgb | noclim | desconhecido | lodo | 0.052 | 3.988 | 3.273 | /workspace/reports/exp02/exp02_D7_TDN_based_ADF_hgb_noclim_lodo.csv |
| D7 | TDN_based_ADF | mlp | noclim | desconhecido | lodo | -23.160 | 17.470 | 14.509 | /workspace/reports/exp02/exp02_D7_TDN_based_ADF_mlp_noclim_lodo.csv |
| D7 | TDN_based_ADF | xgbnative | noclim | desconhecido | lodo | 0.300 | 3.509 | 2.881 | /workspace/reports/exp02/exp02_D7_TDN_based_ADF_xgbnative_noclim_lodo.csv |
| RAW | CP | hgb | clim | desconhecido | lodo | -0.064 | 1.802 | 1.547 | /workspace/reports/exp02/exp02_RAW_CP_hgb_clim_lodo.csv |
| RAW | CP | mlp | clim | desconhecido | lodo | -1.699 | 2.171 | 1.901 | /workspace/reports/exp02/exp02_RAW_CP_mlp_clim_lodo.csv |
| RAW | CP | xgbnative | clim | desconhecido | lodo | 0.221 | 1.542 | 1.309 | /workspace/reports/exp02/exp02_RAW_CP_xgbnative_clim_lodo.csv |
| RAW | CP | hgb | noclim | desconhecido | lodo | -0.127 | 1.832 | 1.541 | /workspace/reports/exp02/exp02_RAW_CP_hgb_noclim_lodo.csv |
| RAW | CP | mlp | noclim | desconhecido | lodo | -0.644 | 2.018 | 1.702 | /workspace/reports/exp02/exp02_RAW_CP_mlp_noclim_lodo.csv |
| RAW | CP | xgbnative | noclim | desconhecido | lodo | 0.235 | 1.534 | 1.273 | /workspace/reports/exp02/exp02_RAW_CP_xgbnative_noclim_lodo.csv |
| RAW | TDN_based_ADF | hgb | clim | desconhecido | lodo | 0.018 | 4.120 | 3.421 | /workspace/reports/exp02/exp02_RAW_TDN_based_ADF_hgb_clim_lodo.csv |
| RAW | TDN_based_ADF | mlp | clim | desconhecido | lodo | -8.506 | 11.682 | 10.474 | /workspace/reports/exp02/exp02_RAW_TDN_based_ADF_mlp_clim_lodo.csv |
| RAW | TDN_based_ADF | xgbnative | clim | desconhecido | lodo | 0.329 | 3.468 | 2.813 | /workspace/reports/exp02/exp02_RAW_TDN_based_ADF_xgbnative_clim_lodo.csv |
| RAW | TDN_based_ADF | hgb | noclim | desconhecido | lodo | 0.007 | 4.171 | 3.467 | /workspace/reports/exp02/exp02_RAW_TDN_based_ADF_hgb_noclim_lodo.csv |
| RAW | TDN_based_ADF | mlp | noclim | desconhecido | lodo | -22.669 | 17.069 | 15.334 | /workspace/reports/exp02/exp02_RAW_TDN_based_ADF_mlp_noclim_lodo.csv |
| RAW | TDN_based_ADF | xgbnative | noclim | desconhecido | lodo | 0.344 | 3.448 | 2.831 | /workspace/reports/exp02/exp02_RAW_TDN_based_ADF_xgbnative_noclim_lodo.csv |

<!-- TABELA_METRICAS_FIM -->

---

## 4. Onde estão os demais resultados (baseline completo)

Este arquivo é um resumo focado em **hgb, xgbnative e mlp**.  
Os demais detalhes estão em:

- **Matriz completa de modelos/cenários (incluindo Naive, Linear, Ridge, KAN, XNet)**  
  `reports/progress/UFMS_ALLMODELS_metrics_LODO.csv`
- **Tabelas organizadas de R² (baseline completo)**  
  `reports/progress/R2_TABLES_FINAL.md`
- **Campeões por cenário (melhores modelos)**  
  `reports/progress/UFMS_FINALS_best.csv`
- **Ablations (clima isolado, FS, KFold, KAN/XNet, etc.)**  
  `reports/ablations/`
- **Scripts de experimento (definição de D5/D7, clima on/off, LODO, etc.)**  
  `src/`

---

## 5. Mensagem rápida para o orientador

- Todos os resultados da tabela são **LODO por data** (coluna `CV = lodo`).  
- Em todos os cenários (RAW, D5, D7; CP e TDN; com e sem clima), **xgbnative** é o modelo que entrega o melhor compromisso entre R² e erro (RMSE/MAE).  
- **hgb** é estável, mas com R² próximo de zero.  
- **mlp** colapsa em LODO, principalmente em TDN, com R² fortemente negativos e erros altos.  
- Clima é mais útil em **CP** do que em **TDN**; para TDN, o melhor R² costuma aparecer em cenários **noclim** com XGBoost.
