# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import re

REPORTS  = Path("/workspace/reports")
PROGRESS = REPORTS / "progress"
PROGRESS.mkdir(parents=True, exist_ok=True)

BASES   = ["RAW","D5","D7"]
TARGETS = ["CP","TDN"]
MODELS  = ["naive","linear","ridge","gb","xgb","mlp"]

def read_metrics_file(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, comment="#")
    df.columns = [c.strip() for c in df.columns]
    # normaliza coluna de data -> heldout_date
    date_col = None
    for c in ["heldout_date","fold_date","date"]:
        if c in df.columns: date_col = c; break
    if date_col is None: raise ValueError(f"Sem coluna de data em {p.name}")
    df = df.rename(columns={date_col:"heldout_date"})
    # modelo
    if "model" not in df.columns: raise ValueError(f"Sem coluna 'model' em {p.name}")
    df["model"] = df["model"].astype(str).str.lower()
    df = df[df["model"] != "(mean)"]
    # métricas
    for k in ["r2","rmse","mae"]:
        if k not in df.columns: raise ValueError(f"Sem '{k}' em {p.name}")
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df

def mean_row(df: pd.DataFrame) -> pd.Series:
    # se tiver linha __mean__, usa; senão, média dos folds
    if "heldout_date" in df.columns and (df["heldout_date"]=="__mean__").any():
        return df.loc[df["heldout_date"]=="__mean__"].iloc[0]
    return df[["r2","rmse","mae"]].mean(numeric_only=True)

def build_manifest():
    rows = []
    for base in BASES:
        for tgt in TARGETS:
            for m in MODELS:
                wo = REPORTS/f"exp01_{base}_{tgt}_{m}.csv"
                wi = REPORTS/f"exp01_{base}_{tgt}_{m}_clim.csv"
                rows.append({"base":base,"target":tgt,"model":m,
                             "without":wo.exists(),"with":wi.exists(),
                             "path_without":str(wo), "path_with":str(wi)})
    man = pd.DataFrame(rows)
    man.to_csv(PROGRESS/"metrics_manifest.csv", index=False)
    return man

def compute_gains(man: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for _, r in man.iterrows():
        if not (r["without"] and r["with"]): 
            continue
        df_wo = read_metrics_file(Path(r["path_without"]))
        df_wi = read_metrics_file(Path(r["path_with"]))
        # média por modelo (garante que estamos comparando o mesmo m)
        m = r["model"]
        m_wo = df_wo[df_wo["model"]==m]; m_wi = df_wi[df_wi["model"]==m]
        if m_wo.empty or m_wi.empty: 
            continue
        s_wo = mean_row(m_wo); s_wi = mean_row(m_wi)
        recs.append({
            "base": r["base"], "target": r["target"], "model": m,
            "r2_without": float(s_wo["r2"]),  "r2_with": float(s_wi["r2"]),
            "rmse_without": float(s_wo["rmse"]), "rmse_with": float(s_wi["rmse"]),
            "mae_without": float(s_wo["mae"]),  "mae_with": float(s_wi["mae"]),
        })
    if not recs:
        return pd.DataFrame()
    g = pd.DataFrame(recs)
    g["dR2"]   = g["r2_with"] - g["r2_without"]
    g["dRMSE"] = g["rmse_without"] - g["rmse_with"]
    g["dMAE"]  = g["mae_without"] - g["mae_with"]
    return g

def write_diag(man: pd.DataFrame, gains: pd.DataFrame):
    miss = man[~(man["without"] & man["with"])][["base","target","model","without","with"]]
    with open(PROGRESS/"metrics_diag.md","w",encoding="utf-8") as f:
        f.write("# Diagnóstico de arquivos esperados\n\n")
        f.write("- Sem clima: `exp01_{BASE}_{TARGET}_{MODEL}.csv`\n")
        f.write("- Com clima: `exp01_{BASE}_{TARGET}_{MODEL}_clim.csv`\n\n")
        if miss.empty:
            f.write("Tudo presente ✅\n")
        else:
            f.write("## Ausências por par (with/without)\n\n")
            f.write(miss.sort_values(["base","target","model"]).to_markdown(index=False))
            f.write("\n")

def write_gains(gains: pd.DataFrame):
    gains.sort_values(["base","target","model"]).to_csv(
        PROGRESS/"metrics_gain_summary_ALL.csv", index=False
    )
    with open(PROGRESS/"metrics_gain_summary_ALL.md","w",encoding="utf-8") as f:
        f.write("# Ganho do clima — TODOS\n\n")
        f.write(gains.sort_values(["base","target","model"])[
            ["base","target","model","dR2","dRMSE","dMAE",
             "r2_without","r2_with","rmse_without","rmse_with","mae_without","mae_with"]
        ].to_markdown(index=False))
        f.write("\n")
    # também escreve por base
    for base in ["RAW","D5","D7"]:
        sub = gains[gains["base"]==base]
        if sub.empty: 
            continue
        out = PROGRESS/f"metrics_gain_summary_{base}.md"
        with open(out,"w",encoding="utf-8") as f:
            f.write(f"# Ganho do clima — {base}\n\n")
            f.write(sub.sort_values(["target","model"])[
                ["target","model","dR2","dRMSE","dMAE",
                 "r2_without","r2_with","rmse_without","rmse_with","mae_without","mae_with"]
            ].to_markdown(index=False))
            f.write("\n")

def main():
    man = build_manifest()
    gains = compute_gains(man)
    write_diag(man, gains)
    if gains.empty:
        print("[ok] Manifesto gerado; ainda não há pares com/sem clima. Veja:")
        print("     -", PROGRESS/"metrics_manifest.csv")
        print("     -", PROGRESS/"metrics_diag.md")
        return
    write_gains(gains)
    print("[ok] Consolidado.")
    print(" -", PROGRESS/"metrics_manifest.csv")
    print(" -", PROGRESS/"metrics_gain_summary_ALL.csv")
    print(" -", PROGRESS/"metrics_diag.md")

if __name__ == "__main__":
    main()
