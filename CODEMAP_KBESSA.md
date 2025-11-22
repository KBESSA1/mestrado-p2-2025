# STRUCTURE_LINKS (snapshot UFMS)

Mapa rápido da estrutura do projeto dentro do container (`/workspace`),  
pensado como “cola” para navegação e como apoio ao RAG.

---

## 1. Raiz do projeto

- `/workspace`  
  - Raiz do repositório do mestrado.
  - Contém:
    - `src/` → código-fonte dos experimentos (ver `Mapa de Código`).
    - `data/` → dados brutos, processados e derivados.
    - `reports/` → métricas, resultados consolidados, artefatos de análise.
    - `scripts/` → utilitários de manutenção/limpeza.
    - Arquivos de documentação:  
      - `README.md`  
      - `SUMMARY_DISCOVERY.md`  
      - `UFMS_CHAMPIONS_LODO.md`  
      - `UFMS_FINAL_REPORT_FS15_LODO.md`  
      - `CODMAP_KBESSA.md` (mapa de código)  
      - `STRUCTURE_LINKS.md` (este arquivo)

---

## 2. Dados

### 2.1 Diretório principal de dados

- `/workspace/data`

Subpastas principais:

- `/workspace/data/data_raw`
  - Contém o **dataset mestre oficial**:
    - `Complete_DataSet.csv`  
      → único arquivo mestre (campo + S2 + clima).  
      → todos os cenários (RAW/D5/D7, clim/noclim) são derivados **em memória** a partir dele.

- `/workspace/data/data_processed`
  - Contém datasets de fases anteriores e derivados históricos.
  - Exemplo (pode variar):
    - `Complete_Dataset.csv`  
      → versão processada histórica usada nos primeiros baselines.
    - `ACTIVE_DATASET.csv` → **link lógico** (ou CSV ativo) apontando para o dataset S2 atual da fase em questão (ex.: `complete_S2_allgeom.csv`).
    - `_meta/`  
      → metadados, snapshots, descrições de versões intermediárias (quando existirem).

- `/workspace/data/feature_selected`
  - CSVs **reduzidos por FS** (FS15, etc.) gerados pelo pipeline XGBoost:
    - `*_FS15_*.csv`  
      → usados nos experimentos Ridge_FS, GB_FS, XGB_FS, MLP_FS, KAN_FS, XNet_FS.

- `/workspace/data/feature_sets`
  - Listas de features selecionadas por cenário:
    - `*.features.txt`  
      → nomes de colunas que compõem FS15 (e outros subconjuntos).

> Regra:  
> - Para contar a história oficial do dataset → falar em `data_raw/Complete_DataSet.csv`.  
> - Para replays históricos → mencionar `data_processed/Complete_Dataset.csv` e derivados.

---

## 3. Relatórios e métricas

### 3.1 Diretório raiz de relatórios

- `/workspace/reports/`

Subpastas mais importantes:

- `/workspace/reports/exp01_*`  
  - Resultados **individuais** de experimentos (primeiras rodadas, scripts `01_*`, `02_*`, etc.).

- `/workspace/reports/finals_cv/`
  - Resultados finais por cenário/modelo (fase de **CV/tuning por modelo**).
  - Ex.: `final_D5_CP_gb_full.csv`, `final_RAW_CP_mlp_clim.csv`.

- `/workspace/reports/progress/`
  - **Coração** dos resultados consolidados:
    - `UFMS_MASTER_metrics_all.csv`  
      → tabela mestre com **todas** as execuções (5924 linhas).
    - `UFMS_MASTER_LODO_all.csv`  
      → subset com **apenas LODO**.
    - `UFMS_MASTER_KFOLD_all.csv`  
      → subset com **KFold** (ablação, KAN/XNet/MLP).
    - `UFMS_MASTER_GKF_all.csv`  
      → subset com **GroupKFold**.
    - `UFMS_MASTER_LODO_champions_by_scenario.csv`  
      → campeões LODO por cenário (Base × Target × Clima × FS).
    - `UFMS_FINALS_best.csv`  
      → melhores modelos oficiais por cenário.
    - `UFMS_TUNED_all.csv`  
      → rodadas de tuning (GB/XGB etc.).
    - `UFMS_TUNED_vs_FINALS.csv`  
      → comparação tuning vs finais.
    - `UFMS_TAKASHI_LODO_champions_auto.csv`  
      → campeões LODO (versão “auto” para o resumo Takashi).
    - `UFMS_TAKASHI_model_stats_auto.csv`  
      → estatísticas de R² médio/máx. por Modelo × Target × CV.
    - Relatórios em texto/markdown:
      - `SUMMARY_DISCOVERY.md`  
      - `UFMS_CHAMPIONS_LODO.md`  
      - `UFMS_FINAL_REPORT_FS15_LODO.md`  
      - Outros arquivos auxiliares de análise.

- `/workspace/reports/winners/`
  - Resultados “vencedores” das fases iniciais (exp01/exp02, etc.).  
  - Mantidos por histórico, mas a leitura principal agora é via `UFMS_MASTER_*`.

### 3.2 Archive de relatórios

- `/workspace/_reports_latest.tar.gz`  
  - **Symlink** para o último archive comprimido de relatórios.  
  - Útil para backup/export rápido de todos os CSVs/MDs de `reports/`.

---

## 4. Scripts úteis (shell / manutenção)

- `/workspace/scripts/cleanup_reports.sh`  
  - Script para limpeza/organização de relatórios (arquivos antigos, cache, etc.).
  - Registrado no `PATH` via `.bashrc`.

Outros scripts de automação importantes (em `src/`):

- `/workspace/src/make_policy_symlinks.sh`  
  - Cria symlinks de políticas/configs (legacy/organização).
- `/workspace/src/run_final_baselines.sh`  
  - Roda um conjunto final de baselines LODO de forma automatizada.

Na raiz do projeto:

- `/workspace/run_gkf_all.sh`  
  - Roda todas as ablações com GroupKFold (checagem metodológica).

> Sempre que um novo script de automação for criado, idealmente:
> - colocá-lo em `/workspace/scripts` ou em `src/`,  
> - e registrar aqui com uma linha de descrição.

---

## 5. Alias & PATH (no container)

Arquivo de configuração do shell:

- `~/.bashrc`

Contém (no momento deste snapshot):

- Alias:
  - `alias cleanup_reports='cleanup_reports.sh'` (exemplo típico; ajustar para o nome real).
- PATH:
  - `export PATH="/workspace/scripts:$PATH"`

E eventualmente outros aliases/exports usados no projeto.

---

## 6. Versões e ambiente (snapshot)

Informações detalhadas de ambiente (SO, Python, pacotes, GPU, CUDA, etc.):

- Consultar o **audit** mais recente (ex.: `AUDIT_ENV_*.md` ou seção de ambiente no README/SUMMARY).
- No momento deste snapshot:
  - SO: distribuição Linux da imagem NVIDIA DL (container).  
  - Python: 3.10.x  
  - GPU: NVIDIA (4070/4090, dependendo da máquina host)  
  - Principais libs:
    - `numpy`, `pandas`, `scikit-learn`
    - `xgboost`
    - `torch` (PyTorch)
    - Demais listadas no `requirements.txt` / Dockerfile.

> Ideia: este item não congela versões exatas aqui, só aponta para o documento oficial de ambiente.

---

## 7. Como usar este arquivo na prática

- Para **explicar a estrutura** pro orientador:  
  → focar nas seções 2 (Dados) e 3 (Relatórios).

- Para **entrar no container e se achar rápido**:  
  → usar as seções 1 (Raiz), 4 (Scripts) e 5 (Alias & PATH).

- Para **futuro RAG / IA**:  
  → este arquivo serve de índice estável para localizar:
    - onde está o dataset mestre (`data_raw/Complete_DataSet.csv`);
    - onde estão as métricas consolidadas (`reports/progress/UFMS_MASTER_*`);
    - quais scripts automatizam limpezas e rodadas finais.
