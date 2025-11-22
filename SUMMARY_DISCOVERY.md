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

Conclusão 1:  
Com o dataset atual, CP está relativamente próximo do limite de previsibilidade via sensoriamento remoto + clima. Já TDN exige novas fontes de informação (solo, manejo, água, dados laboratoriais adicionais) para que se obtenham ganhos significativos.

---

### 2.2 Modelos profundos não superam árvores em LODO

- KAN, XNet e MLP apresentam desempenho muito bom em K-Fold aleatório (sem respeito à estrutura temporal), o que confirma alta capacidade intrínseca.
- Em validação LODO por data:
  - O desempenho de redes profundas cai de forma acentuada.
  - Modelos baseados em árvores (GB, XGB) e modelos lineares (Ridge) se mantêm mais estáveis.
- Em cenário real de uso (previsão por data/campanha), os modelos de árvore tendem a ser superiores ou, no mínimo, mais confiáveis.

Conclusão 2:  
Para este dataset temporal pequeno, redes profundas vencem no cenário embaralhado (K-Fold i.i.d.), mas perdem para modelos de árvore quando a validação respeita o uso real (LODO por data).

---

### 2.3 Clima melhora CP e pouco afeta TDN

- A inclusão de janelas climáticas como [t−3, t] e [t−7, t] adiciona sinal relevante para CP, com ganhos de R² da ordem de até aproximadamente +0,10 em alguns cenários.
- Para TDN, o impacto do clima é pequeno ou inconsistente; a baixa previsibilidade do alvo domina.
- Interpretação:
  - Clima captura variações fisiológicas da planta ligadas à proteína bruta (CP).
  - Para TDN, a limitação principal parece ser a natureza do alvo e a falta de informações adicionais, não apenas o clima.

Conclusão 3:  
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

Conclusão 4:  
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

Conclusão 5:  
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

Esta seção resume, em uma única tabela, as principais métricas de todos os cenários e arquiteturas avaliados sob validação LODO por data.

As informações detalhadas estão consolidadas em:

- `reports/progress/UFMS_ALLMODELS_metrics_LODO.csv`

e, em versões resumidas, em:

- `reports/progress/UFMS_FINALS_best.csv`
- `reports/progress/R2_TABLES_FINAL.md`

A tabela abaixo deve listar, para cada combinação de:

- base (RAW, D5, D7),
- janela (quando aplicável),
- uso de clima (com ou sem),
- alvo (CP, TDN_based_ADF),
- família de modelo (Naive, Linear, Ridge, GB, XGB, MLP, KAN, XNet),
- uso ou não de FS15,
- métricas principais (R² OOF global, R² médio por data, RMSE, MAE),
- referência para o arquivo de saída correspondente.

### 4.1 Tabela completa de métricas por cenário e arquitetura

> Observação: os valores numéricos devem ser gerados a partir dos arquivos em `reports/progress/`, em especial `UFMS_ALLMODELS_metrics_LODO.csv`.  
> Abaixo está o formato sugerido da tabela.

<!-- TABELA_METRICAS_INICIO -->

| Base | Janela | Clima | Target           | Modelo | FS15 | R2_OOF | R2_MEDIA_DATA | RMSE | MAE | Arquivo_relatorio |
|------|--------|-------|------------------|--------|------|--------|----------------|------|-----|--------------------|
| ...  | ...    | ...   | ...              | ...    | ...  | ...    | ...            | ...  | ... | ...                |

<!--
Preencher esta tabela com todas as linhas geradas a partir de UFMS_ALLMODELS_metrics_LODO.csv
ou do conjunto equivalente de relatórios finais.
-->

<!-- TABELA_METRICAS_FIM -->

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
