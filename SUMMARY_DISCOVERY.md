# Resumo das Principais Descobertas — Projeto Mestrado UFMS
### Rodrigo Luiz Campos (Kbessa)  
### Data: Nov/2025

## 1. O que este projeto fez (em 1 frase)
Usei Sentinel-2 + clima para prever **CP** e **TDN** via ML, validado corretamente com **LODO por data**, avaliando modelos clássicos e profundos (MLP, KAN, XNet), incluindo seleção de features e ablações.

---

## 2. Principais Descobertas Científicas

### 🔥 (1) CP é previsível — TDN não é
- **CP** apresentou estrutura clara → R² ≈ **0.30–0.45** em LODO.
- **TDN**, mesmo com clima e FS, ficou em torno de **0.00–0.15**.
- Isso mostra que o **sinal espectral + clima é suficiente para CP**, mas **fraco para TDN**.

**Conclusão 1:** TDN precisa de novas fontes de informação (laboratório, solo, manejo, água), enquanto CP está perto do limite teórico com o dataset atual.

---

### 🔥 (2) Modelos profundos NÃO superam modelos de árvore no regime correto (LODO)
- KAN, XNet e MLP vão muito bem em **KFold aleatório** (capacidade intrínseca alta).
- Em **LODO por data**:
  - O desempenho cai bastante.
  - **GB/XGB/Ridge** se mantêm mais estáveis.
- Resultado: em cenário real (previsão por data/campanha), **árvores ganham**.

**Conclusão 2:** “Redes profundas vencem no embaralhado, perdem no real” para este dataset temporal pequeno.

---

### 🔥 (3) Clima melhora CP (e pouco afeta TDN)
- Janelas climáticas [t−3, t] e [t−7, t] adicionam sinal relevante para **CP** → ganhos de até ~+0.10 em R².
- Para **TDN**, o impacto é pequeno ou inconsistente.

**Conclusão 3:** O clima explica parte da **variação fisiológica da planta (CP)**, mas não resolve a baixa previsibilidade de TDN.

---

### 🔥 (4) FS15 (XGBoost) melhora modelos clássicos, mas NÃO melhora redes profundas
- Seleção de features via XGBoost (FS15, top-15 por cenário):
  - **Melhora** GB, XGB e Ridge.
  - É **neutra ou prejudicial** para MLP, KAN e XNet.
- Redes profundas parecem preferir o espaço de features completo, mesmo em regime de poucos dados.

**Conclusão 4:** Modelos estruturados (árvores + Ridge) ganham com seleção de features; redes profundas ganham com mais dados e menos poda.

---

### 🔥 (5) O gargalo científico é o número de amostras por data
- Dataset total ~312 amostras, mas poucas amostras por data/campanha.
- LODO expõe isso de forma clara.
- Nos ablations KFold, KAN/XNet chegam a R² altos (CP ≈ 0.8, TDN até ≈ 0.7), mostrando que **capacidade do modelo não é o problema**.

**Conclusão 5:** O limite atual é **densidade temporal e espectral do dataset**, não falta de modelo sofisticado.  
Abre caminho direto para um doutorado focado em:
- Mais datas por campanha,
- Mais bandas/sensores,
- Integração com dados de solo, manejo e laboratório.

---

## 3. Contribuições Reais do Trabalho

### ✔ Padronização de LODO correto para dataset agrícola pequeno
- Validação sem vazamento,
- Comparação justa entre modelos,
- Pipeline replicável.

### ✔ Avaliação sistemática de 8 famílias de modelos
- De Naive até KAN/XNet.
- Poucos trabalhos de mestrado fazem esse espectro completo.

### ✔ Demonstração prática do “overfitting metodológico”
- Embaralhar dados (KFold) dá impressão de R² altos com redes profundas.
- LODO mostra o cenário real de uso → resultado científico importante para sensoriamento remoto e agronomia.

### ✔ FS15 via XGBoost como política reprodutível
- Critério claro,
- Arquivos de features documentados (`data/feature_sets/*.features.txt`),
- CSVs de treino compactos (`data/feature_selected/*.csv`).

### ✔ Pipeline totalmente reprodutível
- Docker + `environment.yml`,
- Scripts em `src/`,
- Relatórios consolidados em `reports/`.

---

## 4. Onde estão os resultados finais no repositório

- **Relatório textual das descobertas (LODO):**  
  `reports/progress/UFMS_CHAMPIONS_LODO.md`

- **Melhores modelos por cenário (tabelas):**  
  `reports/progress/UFMS_FINALS_best.csv`

- **Matriz completa de métricas por modelo/cenário (LODO):**  
  `reports/progress/UFMS_ALLMODELS_metrics_LODO.csv`

- **Resumo da política de seleção de features (FS15):**  
  `reports/progress/UFMS_FS15_summary.md`

- **Ablations (KAN/XNet, KFold, etc.):**  
  `reports/ablations/*.csv`

- **Tabelas finais de CP / TDN:**  
  `reports/progress/R2_TABLES_FINAL.md`

---

## 5. Roteiro de Leitura para o Orientador

1. Ler este arquivo: `SUMMARY_DISCOVERY.md`.
2. Ler `reports/progress/UFMS_CHAMPIONS_LODO.md`.
3. Conferir `reports/progress/R2_TABLES_FINAL.md`.
4. Ver detalhes em `reports/progress/UFMS_ALLMODELS_metrics_LODO.csv`.
5. Ver FS15 em `reports/progress/UFMS_FS15_summary.md`.
6. (Opcional) Ver ablations em `reports/ablations/`.

---

## 6. Conclusão Geral

- Há **ciência nova** aqui, principalmente em:
  - Comparar modelos clássicos vs profundos sob **validação temporal correta (LODO)**.
  - Entender os limites de CP vs TDN com sensoriamento remoto + clima.
  - Usar FS15 de forma estruturada e reprodutível.
- O trabalho abre caminho claro para **um doutorado** centrado em:
  - Mais dados (tempo/espectro),
  - Integração de fontes (solo, manejo, lab),
  - Exploração mais profunda de KAN/XNet em regime temporal realista.

