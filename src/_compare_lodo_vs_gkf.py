# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Utilitário de validação Leave-One-Date-Out (LODO). Simula cenário real de prever campanhas futuras nunca vistas.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REPORTS = Path("/workspace/reports")
PROG = REPORTS/"progress"; PROG.mkdir(parents=True, exist_ok=True)

def _scan(pattern):
    return [p for p in REPORTS.rglob(pattern) if p.is_file()]

def _read_mean(fp):
    df = pd.read_csv(fp)
    if "fold" in df.columns and "__mean__" in df["fold"].astype(str).values:
        row = df[df["fold"].astype(str)=="__mean__"].iloc[0]
        return dict(r2=float(row["r2"]), rmse=float(row["rmse"]), mae=float(row["mae"]))
    return dict(r2=float(df["r2"].mean()), rmse=float(df["rmse"].mean()), mae=float(df["mae"].mean()))

def _infer(btms):
    out=[]
    rx = re.compile(r"exp01_(RAW|D5|D7)_(CP|TDN_based_ADF)_(naive|linear|ridge|gb|xgb|mlp)(?:_(clim))?(?:_gkf)?\.csv$")
    for fp in btms:
        m = rx.search(fp.name)
        if not m: continue
        base, target, model, clim = m.groups()
        source = "clim" if clim else "noclim"
        kind = "gkf" if fp.name.endswith("_gkf.csv") else "lodo"
        out.append((fp, base, target, model, source, kind))
    return out

def main():
    files = _scan("exp01_*.csv")
    items = _infer(files)
    rows=[]
    for fp, base, target, model, source, kind in items:
        met = _read_mean(fp)
        rows.append(dict(base=base, target=target, model=model, source=source, kind=kind, **met))
    if not rows:
        print("nada encontrado em /workspace/reports.")
        return
    df = pd.DataFrame(rows)

    all_summary=[]
    for (base, target, source), sub in df.groupby(["base","target","source"]):
        lodo = sub[sub["kind"]=="lodo"].set_index("model")[["r2","rmse","mae"]]
        gkf  = sub[sub["kind"]=="gkf"].set_index("model")[["r2","rmse","mae"]]
        common = sorted(set(lodo.index) & set(gkf.index))
        if len(common)<2:  # precisa de pelo menos 2 modelos em comum
            continue
        l = lodo.loc[common]
        g = gkf.loc[common]

        rank_l_r2 = l["r2"].rank(ascending=False, method="min")
        rank_g_r2 = g["r2"].rank(ascending=False, method="min")
        rho_r2, _ = spearmanr(rank_l_r2, rank_g_r2)

        rank_l_rmse = l["rmse"].rank(ascending=True, method="min")
        rank_g_rmse = g["rmse"].rank(ascending=True, method="min")
        rho_rmse, _ = spearmanr(rank_l_rmse, rank_g_rmse)

        out_fp = PROG/f"compare_LODO_vs_GKF_{base}_{target}_{source}.csv"
        pd.DataFrame({
            "model": common,
            "lodo_r2": l["r2"].values, "gkf_r2": g["r2"].values,
            "lodo_rmse": l["rmse"].values, "gkf_rmse": g["rmse"].values,
            "lodo_mae": l["mae"].values, "gkf_mae": g["mae"].values
        }).to_csv(out_fp, index=False)

        all_summary.append(dict(
            base=base, target=target, source=source,
            n_models=len(common),
            spearman_rank_r2=float(rho_r2),
            spearman_rank_rmse=float(rho_rmse),
            mean_lodo_r2=float(l["r2"].mean()), mean_gkf_r2=float(g["r2"].mean()),
            mean_lodo_rmse=float(l["rmse"].mean()), mean_gkf_rmse=float(g["rmse"].mean())
        ))

    if all_summary:
        pd.DataFrame(all_summary).sort_values(["base","target","source"]).to_csv(PROG/"LODO_vs_GKF_summary.csv", index=False)
        print(f"ok -> {PROG/'LODO_vs_GKF_summary.csv'}")
    else:
        print("nenhum par LODO×GKF com interseção de modelos >=2.")

if __name__=="__main__":
    main()
