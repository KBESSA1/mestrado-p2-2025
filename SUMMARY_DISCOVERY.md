# Resumo das Principais Descobertas — Projeto Mestrado UFMS

Autor: Rodrigo Luiz Campos (Kbessa)  
Data de referência: nov/2025

---

## 1. O que foi feito

A partir de um único dataset mestre (**Complete_DataSet.csv**), foram construídos cenários de modelagem para previsão de:

- **CP**  
- **TDN_based_ADF**

com base em:

- bandas e índices do **Sentinel-2**;
- variáveis climáticas agregadas por janela (quando utilizadas).

Os experimentos usam **validação LODO por data** (cada campanha/dia fica inteira em teste em algum fold) e comparam três famílias principais de modelos:

- **hgb** (HistGradientBoosting / GB de árvore);
- **xgbnative** (XGBoost nativo);
- **mlp** (rede neural simples).

Os cenários de base na tabela são:

- **RAW**  
- **D5**  
- **D7**

(que representam diferentes restrições/filtragens temporais a partir do mesmo conjunto mestre).

A tabela abaixo resume as métricas de erro (**RMSE** e **MAE**) para todos os cenários em regime **LODO por data**.

---

## 2. Leitura geral da tabela de métricas

### 2.1 CP (Proteína Bruta)

Para **CP**, nos três conjuntos (D5, D7, RAW):

- Os modelos **de árvore** (hgb e xgbnative) apresentam **menores RMSE e MAE** em todos os cenários.
- A **MLP** tem erros consistentemente maiores (RMSE ≈ 2,0–2,1; MAE ≈ 1,7–1,9), indicando pior ajuste em regime LODO.

Comparando bases:

- **D7** e **D5** (quando bem configurados) chegam a RMSE ≈ 1,36–1,71 e MAE ≈ 1,14–1,51 para os melhores modelos (normalmente xgbnative e hgb).  
- Na base **RAW**, os erros sobem ligeiramente (RMSE ≈ 1,53–1,80; MAE ≈ 1,27–1,55).

Em resumo: para CP, **árvores (xgbnative/hgb) são mais precisas que MLP**, e as variações de base (D5/D7/RAW) mudam o erro, mas sem inverter essa hierarquia.

---

### 2.2 TDN_based_ADF

Para **TDN_based_ADF**:

- **hgb** e **xgbnative** mantêm RMSE na faixa ≈ 3,4–4,2 e MAE ≈ 2,8–3,4, de forma relativamente estável entre D5, D7 e RAW.
- A **MLP** apresenta erros **muito maiores**:
  - D5: RMSE ≈ 13,64–16,04; MAE ≈ 12,07–13,71  
  - D7: RMSE ≈ 15,02–17,47; MAE ≈ 13,39–14,51  
  - RAW: RMSE ≈ 11,68–17,07; MAE ≈ 10,47–15,33

Isso indica duas coisas:

1. **TDN é um alvo mais difícil** (erros absolutos maiores do que em CP).
2. A **MLP é instável** para TDN em LODO (erros explodem), enquanto **hgb/xgbnative** continuam relativamente controlados.

---

### 2.3 Comparação geral entre bases (RAW, D5, D7)

- **D5 e D7**:
  - Para CP, oferecem os menores erros com modelos de árvore (xgbnative/hgb).
  - Para TDN, mantêm o padrão: árvores estáveis, MLP muito pior.
- **RAW**:
  - Em CP, erros um pouco maiores, mas ainda com árvores melhores que MLP.
  - Em TDN, mesma hierarquia: árvores razoáveis, MLP inconsistente.

Conclusão geral da tabela:  
Em regime LODO, **árvores (hgb, xgbnative)** são claramente superiores à **MLP** tanto em CP quanto em TDN. CP apresenta erros menores e, portanto, é mais “amigável” de modelar; TDN é mais difícil, com erros maiores e redes neurais particularmente frágeis nesse alvo.

---

## 3. Tabela de métricas (LODO por data)

A tabela abaixo é a saída direta de `reports/progress/UFMS_ALLMODELS_metrics_LODO.csv` para os modelos **hgb**, **mlp** e **xgbnative**, agregando os cenários D5, D7 e RAW para CP e TDN_based_ADF.

### 3.1 RMSE e MAE por Base, Target e Modelo

<!-- TABELA_METRICAS_INICIO -->

| Base | Target | Modelo | RMSE | MAE |
| --- | --- | --- | --- | --- |
| D5 | CP | hgb | 1.708 | 1.509 |
| D5 | CP | mlp | 2.086 | 1.826 |
| D5 | CP | xgbnative | 1.441 | 1.240 |
| D5 | CP | hgb | 1.857 | 1.612 |
| D5 | CP | mlp | 2.008 | 1.671 |
| D5 | CP | xgbnative | 1.510 | 1.302 |
| D5 | TDN_based_ADF | hgb | 4.214 | 3.394 |
| D5 | TDN_based_ADF | mlp | 13.642 | 12.069 |
| D5 | TDN_based_ADF | xgbnative | 3.707 | 3.026 |
| D5 | TDN_based_ADF | hgb | 4.233 | 3.417 |
| D5 | TDN_based_ADF | mlp | 16.036 | 13.710 |
| D5 | TDN_based_ADF | xgbnative | 3.584 | 2.935 |
| D7 | CP | hgb | 1.704 | 1.453 |
| D7 | CP | mlp | 1.990 | 1.743 |
| D7 | CP | xgbnative | 1.365 | 1.146 |
| D7 | CP | hgb | 1.693 | 1.431 |
| D7 | CP | mlp | 1.882 | 1.559 |
| D7 | CP | xgbnative | 1.491 | 1.244 |
| D7 | TDN_based_ADF | hgb | 4.105 | 3.381 |
| D7 | TDN_based_ADF | mlp | 15.017 | 13.393 |
| D7 | TDN_based_ADF | xgbnative | 3.481 | 2.841 |
| D7 | TDN_based_ADF | hgb | 3.988 | 3.273 |
| D7 | TDN_based_ADF | mlp | 17.470 | 14.509 |
| D7 | TDN_based_ADF | xgbnative | 3.509 | 2.881 |
| RAW | CP | hgb | 1.802 | 1.547 |
| RAW | CP | mlp | 2.171 | 1.901 |
| RAW | CP | xgbnative | 1.542 | 1.309 |
| RAW | CP | hgb | 1.832 | 1.541 |
| RAW | CP | mlp | 2.018 | 1.702 |
| RAW | CP | xgbnative | 1.534 | 1.273 |
| RAW | TDN_based_ADF | hgb | 4.120 | 3.421 |
| RAW | TDN_based_ADF | mlp | 11.682 | 10.474 |
| RAW | TDN_based_ADF | xgbnative | 3.468 | 2.813 |
| RAW | TDN_based_ADF | hgb | 4.171 | 3.467 |
| RAW | TDN_based_ADF | mlp | 17.069 | 15.334 |
| RAW | TDN_based_ADF | xgbnative | 3.448 | 2.831 |

<!-- TABELA_METRICAS_FIM -->

---

## 4. Onde olhar no repositório

- Métricas completas (incluindo outras famílias de modelos):  
  `reports/progress/UFMS_ALLMODELS_metrics_LODO.csv`
- Análises adicionais (R², clima, FS, KAN/XNet, etc.):  
  `reports/progress/` e `reports/ablations/`
- Código dos experimentos:  
  `src/`

