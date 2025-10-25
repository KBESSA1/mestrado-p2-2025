# -*- coding: utf-8 -*-
"""
Runner simples para a rodada exp01:
- Descobre CSV (usa d5 se existir; caso contrário, o original)
- Roda baselines (Naïve/Linear/Ridge/GB/XGB) para CP e TDN_based_ADF
- Gera painel de progresso (CSV/MD + plots)
- Registra no RUN_LOG.md com data e flags principais
Execução:
    /usr/bin/python3 /workspace/src/run_exp01.py
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

WORKDIR = Path("/workspace")
SRC = WORKDIR / "src"
REPORTS = WORKDIR / "reports"
DATA = WORKDIR / "data"
RAW = DATA / "data_raw" / "Complete_DataSet.csv"
D5 = DATA / "data_processed" / "Complete_DataSet_d5.csv"
RUN_LOG = REPORTS / "RUN_LOG.md"

BASELINES = [
    ("01_exp_s2paper_d5_baselines.py", "CP"),             # Naïve
    ("01_exp_s2paper_d5_baselines.py", "TDN_based_ADF"),
    ("02_baseline_linear.py", "CP"),
    ("02_baseline_linear.py", "TDN_based_ADF"),
    ("02b_baseline_ridge.py", "CP"),
    ("02b_baseline_ridge.py", "TDN_based_ADF"),
    ("03_baseline_gb.py", "CP"),
    ("03_baseline_gb.py", "TDN_based_ADF"),
    ("04_baseline_xgb.py", "CP"),
    ("04_baseline_xgb.py", "TDN_based_ADF"),
]

def sh(cmd, cwd=None):
    print("→", cmd)
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr.strip())
        raise SystemExit(res.returncode)

def main():
    # 1) escolher CSV
    csv = D5 if D5.exists() else RAW
    if not csv.exists():
        raise SystemExit(f"CSV não encontrado: {csv} (nem {RAW})")
    print(f"[info] usando CSV: {csv}")

    REPORTS.mkdir(parents=True, exist_ok=True)

    # 2) rodar baselines
    for script, target in BASELINES:
        # padroniza nome de saída por modelo
        if "01_exp" in script:
            model_tag = "naive"
        elif "02_baseline_linear" in script:
            model_tag = "linear"
        elif "02b_baseline_ridge" in script:
            model_tag = "ridge"
        elif "03_baseline_gb" in script:
            model_tag = "gb"
        elif "04_baseline_xgb" in script:
            model_tag = "xgb"
        else:
            model_tag = Path(script).stem

        tgt_tag = "TDN" if target == "TDN_based_ADF" else target
        outname = f"exp01_metrics_{tgt_tag}_{model_tag}.csv"
        outpath = REPORTS / outname

        cmd = f"/usr/bin/python3 {SRC / script} --csv {csv} --date-col Date --target-col {target} --out {outpath}"
        sh(cmd)

    # 3) gerar painel
    sh(f"/usr/bin/python3 {SRC / 'progress_dashboard.py'}")

    # 4) log
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    delta_flag = "Δ<=5: SIM" if D5.exists() else "Δ<=5: N/A"
    line = f"{now}  | exp01 | LODO (paper-like, sem clima) | {delta_flag} | TDN_def=ADF | painel atualizado\n"
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    print("[ok] RUN_LOG atualizado:", RUN_LOG)

    print("\n[feito] Saídas:")
    print(" - Tabelas/plots: /workspace/reports/progress/")
    print(" - CSVs métricas: /workspace/reports/exp01_metrics_*.csv")
    print(" - Log          : /workspace/reports/RUN_LOG.md")

if __name__ == "__main__":
    main()
