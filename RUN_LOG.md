# RUN_LOG — Projeto UFMS (CP / TDN_based_ADF com Sentinel-2 + Clima)

> Convenção das entradas:
>
> - **DATA:** AAAA-MM-DD
> - **FOCO:** tópico principal do dia
> - **COMANDOS / SCRIPTS:** comandos relevantes (quando lembrados)
> - **ARTEFATOS GERADOS:** arquivos criados/atualizados
> - **NOTAS:** decisões, bugs, correções, ideias

---

## 2025-10-10 — Setup inicial do container e GPU

- **FOCO:** Garantir ambiente de execução (Docker + GPU) para o projeto UFMS.
- **COMANDOS / SCRIPTS:**
  - `docker exec -it mestrado bash`
  - Teste rápido de bibliotecas e GPU:
    - `python -c "import torch, xgboost, sklearn, pandas; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"`
- **ARTEFATOS GERADOS:** N/A (somente verificação de ambiente).
- **NOTAS:**
  - Confirmado: CUDA disponível; GPU (ex.: RTX 4070) visível dentro do container.
  - Decisão de trabalho: rodar tudo no container `mestrado` com `/workspace` como raiz.
  - Combinado: avançar bem cadenciado, um passo de cada vez, com pausas para estudo.

---

## 2025-10-23/24 — Exploração do container e organização inicial

- **FOCO:** Entender estrutura de diretórios e preparar padrão de organização.
- **COMANDOS / SCRIPTS:**
  - Navegação geral:
    - `pwd`
    - `ls -la /`
    - `ls -la /workspace`
- **ARTEFATOS GERADOS:** Estrutura lógica definida (código em `src/`, dados em `data/`, relatórios em `reports/`).
- **NOTAS:**
  - Confirmado padrão:
    - `/workspace/src` → scripts de experimento.
    - `/workspace/data` → `data_raw` (original) e `data_processed` (derivados).
    - `/workspace/reports` → resultados em CSV/MD.
  - Iniciada discussão sobre padronizar scripts com interface de linha de comando (flags tipo `--csv`, `--date-col`, `--target-col`, etc.).

---

## 2025-10-25 — Primeiros baselines com MLP (D5 / CP)

- **FOCO:** Rodar MLP em cenário D5 para CP, começando a desenhar o baseline.
- **COMANDOS / SCRIPTS:**
  - Exemplo (histórico):
    ```bash
    export PYTHONPATH=/workspace/src:$PYTHONPATH

    python /workspace/src/05_baseline_mlp.py \
      --csv /workspace/data/data_processed/Complete_DataSet_d5.csv \
      --date-col Date --target-col CP \
      --out /workspace/reports/exp01_D5_CP_mlp.csv

    python /workspace/src/05_baseline_mlp.py \
      --csv /workspace/data/data_processed/Complete_DataSet_d5.csv \
      --date-col Date --target-col CP --with-climate \
      --out /workspace/reports/exp01_D5_CP_mlp_clim.csv
    ```
- **ARTEFATOS GERADOS (exemplos):**
  - `reports/exp01_D5_CP_mlp.csv`
  - `reports/exp01_D5_CP_mlp_clim.csv`
- **NOTAS:**
  - Primeira visão do comportamento da MLP em CP com/sem clima.
  - Começo da ideia de manter um painel de experimentos (CSV por rodagem).
  - Ainda não era o esquema de validação LODO final; foco era “tirar a temperatura” dos modelos.

---

## 2025-10-26 — Correção dos scripts GB/XGB e limpeza de código

- **FOCO:** Consertar `03_baseline_gb.py` e `04_baseline_xgb.py`, que estavam quebrados (indentação, argparse, monkey patches antigos).
- **COMANDOS / SCRIPTS:**
  - Backup e limpeza via script auxiliar (exemplo):
    ```bash
    python - << 'PY'
    from pathlib import Path, re, shutil

    SRC_DIR = Path("/workspace/src")
    GB = SRC_DIR/"03_baseline_gb.py"
    XGB = SRC_DIR/"04_baseline_xgb.py"
    SC = SRC_DIR/"sitecustomize.py"

    # (funções de backup e limpeza removendo blocos antigos de monkeypatch)
    PY
    ```
- **ARTEFATOS GERADOS:**
  - Backups:
    - `src/03_baseline_gb.py.bak`
    - `src/04_baseline_xgb.py.bak`
  - Versões “limpas” de:
    - `03_baseline_gb.py`
    - `04_baseline_xgb.py`
- **NOTAS:**
  - Decisão: manter scripts 03/04 o mais simples possível (sem hacks) e concentrar grids pesados nos runners `_gb_cv_runner.py` e `_xgb_cv_runner.py`.
  - Identificada necessidade de um runner de CV genérico para GB/XGB.

---

## 2025-10-27 — Rodadas extensivas de baseline + KFold/GroupKFold

- **FOCO:** Executar baselines completos (Naive, Linear, Ridge, GB, XGB, MLP) com diferentes esquemas de validação, preparando futuro “painel mestre”.
- **COMANDOS / SCRIPTS:**
  - Execução de script geral de GroupKFold:
    ```bash
    /workspace/run_gkf_all.sh
    ```
  - Vários scripts do tipo:
    ```bash
    python src/02_baseline_linear.py ...
    python src/02b_baseline_ridge.py ...
    python src/03_baseline_gb.py ...
    python src/04_baseline_xgb.py ...
    python src/05_baseline_mlp.py ...
    ```
  - Aplicados a:
    - RAW / D5 / D7
    - CP e TDN_based_ADF
    - Com e sem clima
- **ARTEFATOS GERADOS:**
  - Múltiplos `reports/exp01_*_gkf.csv` (por ex.):
    - `exp01_RAW_CP_naive_noclim_gkf.csv`
    - `exp01_RAW_CP_linear_noclim_gkf.csv`
    - (e análogos para outros cenários/modelos)
- **NOTAS:**
  - Aparição de avisos de `pkg_resources` deprecation (benignos).
  - Começo da discussão sobre diferença entre R² global OOF e R² média por data.
  - Semente para regra de bolso:
    - “OOF decide; FINAL é leitura por data.”
  - Planejamento de cálculo posterior de:
    - R² global (OOF)
    - R² médio por data
    - R² ponderado por número de amostras por data.

---

## 2025-11-05 — Snapshots de dados e redefinição do dataset mestre

- **FOCO:** Fazer snapshot de arquivos críticos em `data_processed` e repensar a construção do dataset mestre S2.
- **COMANDOS / SCRIPTS:**
  - Snapshot rápido:
    ```bash
    python - << 'PY'
    import shutil, glob, os, time, hashlib, pathlib
    ts=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dp="/workspace/data/data_processed"
    for f in ["Complete_Dataset.csv","complete_S2_allgeom.csv","complete_CLIMATE_only.csv","complete_S2_allgeom_HLS.csv"]:
        p=os.path.join(dp,f)
        if os.path.exists(p):
            dst=os.path.join(dp,f".snapshot.{ts}")
            shutil.copy2(p, dst)
            h=hashlib.sha256(open(dst,'rb').read()).hexdigest()
            print(f"snapshot -> {dst} | sha256={h[:12]}...")
        else:
            print(f"(aviso) não achei {p}")
    PY
    ```
- **ARTEFATOS GERADOS:**
  - Snapshots em:
    - `data_processed/Complete_Dataset.csv.snapshot.<timestamp>`
    - `data_processed/complete_S2_allgeom.csv.snapshot.<timestamp>`
    - `data_processed/complete_CLIMATE_only.csv.snapshot.<timestamp>`
    - `data_processed/complete_S2_allgeom_HLS.csv.snapshot.<timestamp>`
- **NOTAS:**
  - Lembrete forte: a prioridade é montar o dataset base UFMS (arquivo mestre) direito e não sair criando variações ad hoc.
  - Critério: evitar “delírio de engenharia” e focar no alinhamento com os dados reais da UFMS.
  - Decisão de médio prazo:
    - Tratar `Complete_DataSet.csv` (em `data_raw`) como fonte única/mestre.
    - Usar a versão em `data_processed` como snapshot histórico.
    - Regerar cenários D5/D7, clim/noclim sempre em memória a partir do mestre.

---

## 2025-11-10–11 — Consolidação de validação LODO como padrão oficial

- **FOCO:** Bater o martelo na validação principal e organizar a leitura de métricas.
- **COMANDOS / SCRIPTS:**
  - Refino de scripts utilitários em `src/utils_lodo.py`:
    - Funções para:
      - Criar splits LODO por `Date`
      - Agregar métricas por fold
      - Exportar CSV com colunas padrão (modelo, cenário, métrica, fold, etc.).
- **ARTEFATOS GERADOS:**
  - Versões revisadas de:
    - `src/utils_lodo.py`
    - Scripts de baseline com suporte explícito a LODO (flags e outputs).
- **NOTAS:**
  - Decisão: LODO por data é a validação oficial do projeto.
  - KFold e GroupKFold:
    - Permanecem como análises complementares/ablação.
  - Ponto anotado: comparar com K-Fold por amostra no futuro, mas sem mexer na régua oficial agora.

---

## 2025-11-13/14 — Feature selection com XGBoost e FS15

- **FOCO:** Usar XGBoost para fazer seleção de features baseada em estabilidade e ganho, e depois aplicar FS nos modelos.
- **COMANDOS / SCRIPTS:**
  - Runners de XGBoost com LODO:
    - `src/_xgb_cv_runner.py` (grid leve de parâmetros com validação por data).
  - Scripts de importância de features:
    - `src/feature_importance_xgb_*.py` (nomes aproximados).
- **ARTEFATOS GERADOS:**
  - Arquivos de importância/estabilidade:
    - `reports/progress/feature_importance_xgb_stability_CP.csv`
    - `reports/progress/feature_importance_xgb_stability_TDN_based_ADF.csv`
  - Listas de features selecionadas:
    - `reports/progress/UFMS_FS_selected_CP_*.txt`
    - `reports/progress/UFMS_FS_selected_TDN_*.txt`
  - CSVs agregando importância:
    - `reports/progress/UFMS_FS_rank_*.csv`
- **NOTAS:**
  - Estratégia:
    - Rodar XGBoost com LODO em cada cenário (base × alvo × clim/noclim).
    - Extrair importância por ganho em cada fold.
    - Agregar métricas de estabilidade (freq. em top-K, média de ganho).
  - Definido o conjunto FS15:
    - Top-15 features estáveis por cenário.
  - Ideia central: não confiar em uma única rodada; usar estabilidade em cross-validation por data.

---

## 2025-11-15–18 — Re-treino com FS e consolidação de masters

- **FOCO:** Re-rodar modelos com FS15 e produzir arquivos “master” de métricas.
- **COMANDOS / SCRIPTS:**
  - Re-treino com FS15:
    - Variantes de scripts:
      - `*_ridge_fs.py`
      - `*_gb_fs.py`
      - `*_xgb_fs.py`
      - `*_mlp_fs.py`
      - `*_kan_fs.py`
      - `*_xnet_fs.py`
  - Consolidação geral:
    - Script de agregação (por ex. `src/aggregate_results.py`) para juntar tudo em um único CSV mestre.
- **ARTEFATOS GERADOS:**
  - Arquivo mestre de métricas:
    - `reports/progress/UFMS_MASTER_metrics_all.csv`  
      (todas as rodadas registradas – modelos, cenários, com/sem FS, clima, etc.)
  - Derivados:
    - `reports/progress/UFMS_MASTER_LODO_all.csv`
    - `reports/progress/UFMS_MASTER_KFOLD_all.csv`
    - `reports/progress/UFMS_MASTER_GKF_all.csv`
    - `reports/progress/UFMS_MASTER_LODO_champions_by_scenario.csv`
  - Comparações FS vs full:
    - `reports/progress/UFMS_TUNED_all.csv`
    - `reports/progress/UFMS_TUNED_vs_FINALS.csv`
- **NOTAS:**
  - Observação forte:
    - FS ajuda muito Ridge, GB e XGB (melhor R², menor variância).
    - FS é neutra ou pior para MLP, KAN, XNet (especialmente TDN com clima).
  - Decisão: na dissertação, destacar esse comportamento diferencial FS × tipo de modelo.

---

## 2025-11-20–22 — Síntese de resultados e documentação (README + relatórios finais)

- **FOCO:** Organizar tudo em documentação clara: README, summaries, escolha de campeões por cenário.
- **COMANDOS / SCRIPTS:**
  - Geração e revisão de arquivos de narrativa:
    - `reports/progress/SUMMARY_DISCOVERY.md`
    - `reports/progress/UFMS_CHAMPIONS_LODO.md`
    - `reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md`
  - Criação/edição de:
    - `README.md`
    - `RUN_LOG.md` (este arquivo).
- **ARTEFATOS GERADOS:**
  - `reports/progress/SUMMARY_DISCOVERY.md`  
    → resumo narrativo das descobertas.
  - `reports/progress/UFMS_CHAMPIONS_LODO.md`  
    → descrição dos campeões por cenário (modelo vencedor D5/D7 × CP/TDN × clim/noclim).
  - `reports/progress/UFMS_FINAL_REPORT_FS15_LODO.md`  
    → discussão detalhada dos efeitos de FS nos modelos.
  - `reports/progress/UFMS_FINALS_best.csv`  
    → tabela oficial dos melhores modelos por cenário.
  - `README.md` (organizado e atualizado com:
    - visão geral do projeto
    - estrutura de diretórios
    - datasets
    - validação
    - modelos
    - FS
    - resultados principais).
  - `RUN_LOG.md` (este diário).
- **NOTAS:**
  - Filosofia final de dados:
    - `data_raw/Complete_DataSet.csv` = arquivo mestre imutável.
    - Versões em `data_processed/` = snapshots históricos/reprodutibilidade.
    - Cenários D5/D7, clim/noclim sempre derivados em memória nos scripts.
  - A partir daqui, o repositório está pronto para Git, com README e RUN_LOG servindo como guias oficiais do projeto.

---

## Próximos Passos (para futuras entradas do RUN_LOG)

> A serem logados quando acontecerem:
>
> - Rodadas finais de MLP/KAN/XNet com FS em cenários críticos (caso algum ainda falte).
> - Ablações adicionais:
>   - Comparação LODO × KFold por amostra para fins de discussão na dissertação.
> - Integração com HLS / Landsat-9 (branch futura).
> - Geração de figuras oficiais (R² vs. cenário, importância de features, etc.).
> - Export/push final do repositório para GitHub/GitLab.

