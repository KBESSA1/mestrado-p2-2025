# Resumo das Principais Descobertas — Projeto Mestrado UFMS

Autor: Rodrigo Luiz Campos (Kbessa)  
Data de referência: nov/2025

---

## 1. O que este projeto fez (em uma frase)

Usei imagens Sentinel-2 (e derivados) e variáveis climáticas para prever CP e TDN_based_ADF via aprendizado de máquina, com validação correta em regime temporal (LODO por data), avaliando modelos clássicos e profundos (MLP, KAN, XNet), incluindo seleção de atributos via XGBoost e ablações.

---

## 2. Principais descobertas científicas

### 2.1 CP é previsível; TDN não é

- CP apresentou estrutura clara em LODO por data, com valores de R² tipicamente na faixa de aproximadamente 0,30 a 0,45 nos melhores modelos.
- TDN_based_ADF, mesmo com uso de clima e seleção de atributos, permaneceu com R² baixo (por volta de 0,00 a 0,15).
- Interpretação:
  - O sinal espectral de Sentinel-2, combinado com clima, é suficiente para capturar parte relevante da variabilidade de CP.
  - Para TDN, o sinal disponível é fraco; o modelo tem pouca informação para separar amostras com TDN diferentes.

**Conclusão 1**  
Com o dataset atual, CP está relativamente próximo do limite de previsibilidade via sensoriamento remoto + clima. Já TDN exige novas fontes de informação (solo, manejo, água, dados laboratoriais adicionais) para que se obtenham ganhos significativos.

---

### 2.2 Modelos profundos não superam árvores em LODO

- KAN, XNet e MLP apresentam desempenho muito bom em K-Fold aleatório (sem respeito à estrutura temporal), o que confirma alta capacidade intrínseca.
- Em validação LODO por data:
  - O desempenho de redes profundas cai de forma acentuada.
  - Modelos baseados em árvores (GB, XGB) e modelos lineares (Ridge) se mantêm mais estáveis.
- Em cenário real de uso (previsão por data/campanha), os modelos de árvore tendem a ser superiores ou, no mínimo, mais confiáveis.

**Conclusão 2**  
Para este dataset temporal pequeno, redes profundas vencem no cenário embaralhado (K-Fold i.i.d.), mas perdem para modelos de árvore quando a validação respeita o uso real (LODO por data).

---

### 2.3 Clima melhora CP e pouco afeta TDN

- A inclusão de janelas climáticas como [t−3, t] e [t−7, t] adiciona sinal relevante para CP, com ganhos de R² da ordem de até aproximadamente +0,10 em alguns cenários.
- Para TDN, o impacto do clima é pequeno ou inconsistente; a baixa previsibilidade do alvo domina.
- Interpretação:
  - Clima captura variações fisiológicas da planta ligadas à proteína bruta (CP).
  - Para TDN, a limitação principal parece ser a natureza do alvo e a falta de informações adicionais, não apenas o clima.

**Conclusão 3**  
O clima é um componente importante para melhorar a predição de CP, mas não é suficiente para resolver a baixa previsibilidade de TDN.

---

### 2.4 Seleção de atributos (FS15 via XGBoost) ajuda modelos clássicos, mas não redes profundas

- Foi utilizada uma política de seleção de atributos baseada em XGBoost:
  - Cálculo de importâncias (por ganho) por cenário.
  - Seleção de um conjunto reduzido de aproximadamente 15 atributos por cenário (FS15).
- Efeitos observados:
  - GB, XGB e Ridge melhoram de forma clara com FS15 (menor variância e, em muitos casos, melhor R²).
  - Para MLP, KAN e XNet, FS15 é neutra ou prejudicial, especialmente em TDN com clima.
- Interpretação:
  - Modelos de árvore e lineares se beneficiam de um espaço de entrada mais enxuto e estável.
  - Redes profundas preferem um espaço de entrada mais rico, mesmo em regime de poucos dados, delegando ao próprio modelo a seleção interna de representações.

**Conclusão 4**  
Modelos estruturados (árvores e Ridge) ganham com seleção de atributos explícita. Redes profundas tendem a ganhar mais com mais dados e menos poda de atributos.

---

### 2.5 O gargalo científico é o número de amostras por data

- O dataset possui cerca de 312 amostras no total, mas poucas amostras por data/campanha.
- A validação LODO expõe essa limitação:
  - Cada fold contém poucas amostras em teste, e a variabilidade entre campanhas é alta.
- Em ablações com K-Fold aleatório, ignorando a estrutura temporal:
  - KAN e XNet atingem R² elevados (por exemplo, R² da ordem de 0,8 para CP e até cerca de 0,7 para TDN), mostrando que a capacidade do modelo não é o problema.
- O principal limitante passa a ser:
  - densidade temporal (mais datas por campanha),
  - diversidade espectral e espacial,
  - integração com outras fontes (solo, manejo, laboratório).

**Conclusão 5**  
O limite atual é a densidade temporal e espectral do dataset, não a ausência de modelos sofisticados. Isso abre caminho direto para um doutorado focado em mais dados (tempo, espaço, espectro) e integração de fontes.

---

## 3. Contribuições concretas do trabalho

### 3.1 Padronização de LODO correto para dataset agrícola pequeno

- Validação sem vazamento temporal.
- Comparação justa entre modelos.
- Pipeline reproduzível e documentado.

### 3.2 Avaliação sistemática de múltiplas famílias de modelos

- De Naive e regressões lineares até GB, XGB, MLP, KAN e XNet.
- Cobertura pouco comum em trabalhos de mestrado com dataset agrícola pequeno.

### 3.3 Demonstração prática de “overfitting metodológico”

- Embaralhar dados (K-Fold i.i.d.) gera impressão de R² muito altos para redes profundas.
- LODO por data mostra o cenário de uso real, com desempenho bem mais modesto.
- Resultado relevante para sensoriamento remoto e agronomia: a forma de validação pode distorcer conclusões sobre o “melhor” modelo.

### 3.4 FS15 via XGBoost como política reprodutível

- Critério claro de seleção de atributos.
- Arquivos de features documentados em `data/feature_sets/*.features.txt`.
- CSVs de treino compactos em `data/feature_selected/*.csv`, facilitando reuso e compartilhamento.

### 3.5 Pipeline totalmente reproduzível

- Docker, `environment.yml` e `docker-compose.yml` descrevem o ambiente.
- Scripts organizados em `src/` com nomenclatura consistente.
- Relatórios e métricas consolidados em `reports/`.

---

## 4. Tabela geral de métricas (LODO por data)

Esta seção resume, em uma única tabela, as principais métricas de todos os cenários e arquiteturas avaliados sob validação LODO por data, conforme extração atual de `UFMS_ALLMODELS_metrics_LODO.csv`.  
A tabela abaixo apresenta, para cada combinação de base, alvo e modelo, os valores de RMSE e MAE.

### 4.1 Tabela de RMSE e MAE por cenário e arquitetura

<!-- TABELA_METRICAS_INICIO -->

| Base | Target | Modelo | RMSE | MAE |
| --- | --- | --- | --- | --- |
| D5 | CP | hgb | 1.7076843802476165 | 1.509169113604525 |
| D5 | CP | hgb | 1.8573575964676663 | 1.6121471912384209 |
| D5 | CP | mlp | 2.0855283222591114 | 1.826178086766212 |
| D5 | CP | mlp | 2.0080263420406768 | 1.6712170377076514 |
| D5 | CP | xgbnative | 1.4405780742813286 | 1.239695415392295 |
| D5 | CP | xgbnative | 1.5095069452624854 | 1.301528256670904 |
| D5 | TDN_based_ADF | hgb | 4.213774305147418 | 3.394144441111956 |
| D5 | TDN_based_ADF | hgb | 4.232977820218472 | 3.4171054348769347 |
| D5 | TDN_based_ADF | mlp | 13.641937415856042 | 12.069372442597874 |
| D5 | TDN_based_ADF | mlp | 16.035791449301108 | 13.710463036138938 |
| D5 | TDN_based_ADF | xgbnative | 3.7067429605464186 | 3.0258371634075343 |
| D5 | TDN_based_ADF | xgbnative | 3.5839108210817234 | 2.935216638361731 |
| D7 | CP | hgb | 1.7044349998455677 | 1.452962141760588 |
| D7 | CP | hgb | 1.6928684977905055 | 1.4311617729581068 |
| D7 | CP | mlp | 1.9904758656405213 | 1.7432565003526674 |
| D7 | CP | mlp | 1.8820470190144811 | 1.558792400399006 |
| D7 | CP | xgbnative | 1.3648064394753845 | 1.1459560087738894 |
| D7 | CP | xgbnative | 1.4908775358431952 | 1.2443734077710915 |
| D7 | TDN_based_ADF | hgb | 4.105241059469466 | 3.3808398378777285 |
| D7 | TDN_based_ADF | hgb | 3.987981165893757 | 3.272959461576538 |
| D7 | TDN_based_ADF | mlp | 15.017380026336657 | 13.392603659026106 |
| D7 | TDN_based_ADF | mlp | 17.47027217938148 | 14.509369945451851 |
| D7 | TDN_based_ADF | xgbnative | 3.480854717050311 | 2.8414824414953848 |
| D7 | TDN_based_ADF | xgbnative | 3.5089986006918634 | 2.8808439490660525 |
| RAW | CP | hgb | 1.801987359463876 | 1.5465654460062688 |
| RAW | CP | hgb | 1.8324260556322451 | 1.5410807357849994 |
| RAW | CP | mlp | 2.170974045698828 | 1.901270310481974 |
| RAW | CP | mlp | 2.018073825715295 | 1.7018142598313435 |
| RAW | CP | xgbnative | 1.5421853259792075 | 1.3094597773727272 |
| RAW | CP | xgbnative | 1.534104432128412 | 1.2731842812875311 |
| RAW | TDN_based_ADF | hgb | 4.119794307603398 | 3.420516229970641 |
| RAW | TDN_based_ADF | hgb | 4.170787387177113 | 3.4669636528623147 |
| RAW | TDN_based_ADF | mlp | 11.681998553372011 | 10.473945786908818 |
| RAW | TDN_based_ADF | mlp | 17.06850772631298 | 15.333801223013054 |
| RAW | TDN_based_ADF | xgbnative | 3.4683987798491493 | 2.8130972372441176 |
| RAW | TDN_based_ADF | xgbnative | 3.4475272562210453 | 2.830975761222126 |

<!-- TABELA_METRICAS_FIM -->

> Nota: nesta extração estão apenas as colunas RMSE e MAE para os modelos hgb (GB/HistGB), mlp e xgbnative nos cenários RAW, D5 e D7, para CP e TDN_based_ADF. As métricas completas (incluindo R² global e por data, e demais modelos) continuam disponíveis em `UFMS_ALLMODELS_metrics_LODO.csv` e nas tabelas específicas de R².

---

## 5. Onde estão os resultados no repositório

- Relatório textual das descobertas principais (LODO):  
  `reports/progress/UFMS_CHAMPIONS_LODO.md`
- Melhores modelos por cenário (tabelas consolidadas):  
  `reports/progress/UFMS_FINALS_best.csv`
- Matriz completa de métricas por modelo e cenário (LODO):  
  `reports/progress/UFMS_ALLMODELS_metrics_LODO.csv`
- Resumo da política de seleção de atributos (FS15):  
  `reports/progress/UFMS_FS15_summary.md`
- Ablations (KAN/XNet, K-Fold, etc.):  
  `reports/ablations/*.csv`
- Tabelas finais de R² para CP e TDN:  
  `reports/progress/R2_TABLES_FINAL.md`

---

## 6. Roteiro de leitura sugerido para o orientador

1. Ler este arquivo: `SUMMARY_DISCOVERY.md`.
2. Ler `reports/progress/UFMS_CHAMPIONS_LODO.md` (interpretação dos campeões em LODO).
3. Conferir `reports/progress/R2_TABLES_FINAL.md` (tabelas-resumo de R²).
4. Ver detalhes e variações de modelos e cenários em  
   `reports/progress/UFMS_ALLMODELS_metrics_LODO.csv`.
5. Ver a política de FS15 e listas de atributos em  
   `reports/progress/UFMS_FS15_summary.md` e `data/feature_sets/`.
6. Opcional: explorar as ablações em `reports/ablations/` para entender o comportamento de KAN/XNet fora do regime LODO.

---

## 7. Conclusão geral

Este trabalho traz resultados originais principalmente em:

- Comparar, de forma sistemática, modelos clássicos e profundos sob validação temporal correta (LODO) em um dataset agrícola pequeno.
- Quantificar os limites de previsibilidade de CP e TDN_based_ADF com sensoriamento remoto Sentinel-2 combinado a clima.
- Propor e documentar uma política simples e reprodutível de seleção de atributos (FS15 via XGBoost).
- Evidenciar que o principal gargalo científico atual não é a falta de modelo sofisticado, mas sim a escassez de dados por campanha e a ausência de fontes de informação complementares.

O projeto abre um caminho claro para um futuro doutorado centrado em:

- Aumento de densidade temporal e espectral (mais datas, mais bandas e sensores).
- Integração de fontes (solo, manejo, água, análises laboratoriais adicionais).
- Exploração mais profunda de KAN e XNet em regime temporal realista, com maior volume de dados por campanha.
