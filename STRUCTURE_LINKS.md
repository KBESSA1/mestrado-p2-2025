# STRUCTURE_LINKS (snapshot)
## Raiz
- /workspace
## Dados
- /workspace/data
- /workspace/data/data_processed
- /workspace/data/data_processed/ACTIVE_DATASET.csv  (-> complete_S2_allgeom.csv)
- /workspace/data/data_processed/_meta/
## Relatórios
- /workspace/reports/
- /workspace/reports/finals_cv/
- /workspace/reports/progress/
- /workspace/reports/winners/
- /workspace/_reports_latest.tar.gz  (symlink para último archive)
## Scripts úteis
- /workspace/scripts/cleanup_reports.sh
(Adicionar outros scripts aqui conforme forem criados)
## Alias & PATH
- ~/.bashrc  (contém `alias cleanup_reports` e `export PATH="/workspace/scripts:$PATH"`)
## Versões (no momento deste snapshot)
- Ver seção [6] do audit acima (SO/Python/pacotes/GPU)
