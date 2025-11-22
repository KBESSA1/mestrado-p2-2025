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
  - Combinado: avançar **bem cadenciado**, um passo de cada vez, com pausas para estudo.

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
    - `/workspace/reports` → saídas de experimentos, tabelas de métricas, etc.
