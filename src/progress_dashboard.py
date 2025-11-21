# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

# -*- coding: utf-8 -*-
"""
Painel de progresso:
- Lê reports/exp01_metrics_*.csv (ignora linhas começadas com "# ")
- Para cada arquivo, pega a linha "__mean__"; se não existir, calcula a média de r2/rmse/mae
- Identifica target e modelo a partir do nome do arquivo de forma robusta
- Salva CSV/MD e plots por target
Requisitos: pandas, numpy, matplotlib, tabulate
"""
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPORTS_DIR = "reports"
OUT_DIR = os.path.join(REPORTS_DIR, "progress")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_TAGS = ("naive", "linear", "ridge", "gb", "xgb")

def infer_target_model_from_filename(path: str):
    """
    Extrai (target, model) do nome do arquivo.
    Exemplos:
      exp01_metrics_CP_linear.csv                 -> (CP, linear)
      exp01_metrics_TDN_based_ADF_gb.csv          -> (TDN_based_ADF, gb)
      exp01_metrics_CP_ridge_a1.csv               -> (CP, ridge)
      exp01_metrics_TDN_naive.csv                 -> (TDN, naive)
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem.startswith("exp01_metrics_"):
        stem = stem[len("exp01_metrics_"):]
    # procura o ÚLTIMO token de modelo conhecido dentro do nome
    m = re.search(rf"_(naive(?:-last)?|linear|ridge|gb|xgb)\b", stem, flags=re.IGNORECASE)
    if not m:
        return "UNKNOWN", "unknown"
    model = m.group(1).lower()
    target = stem[:m.start()]  # tudo antes do _<model>
    # normalizar naive-last -> naive
    if model.startswith("naive"):
        model = "naive"
    return target, model

def read_metrics_file(path: str) -> pd.DataFrame:
    """
    Lê CSV (aceita '# ' como comentário).
    Retorna uma linha com r2/rmse/mae e 'model'.
    Preferência: linha '__mean__' se existir; senão média das colunas.
    Tolerante a maiúsculas/minúsculas nos nomes das colunas.
    """
    try:
        df = pd.read_csv(path, comment="#")
    except Exception as e:
        print(f"[warn] não consegui ler {path}: {e}")
        return pd.DataFrame()

    # mapa case-insensitive
    colmap = {c.lower(): c for c in df.columns}
    r2c   = colmap.get("r2")
    rmsec = colmap.get("rmse")
    maec  = colmap.get("mae")
    modelc = colmap.get("model")
    heldc  = colmap.get("heldout_date")

    if not any([r2c, rmsec, maec]):
        return pd.DataFrame()

    if heldc and (df[heldc] == "__mean__").any():
        row = df.loc[df[heldc] == "__mean__"].iloc[0]
        r2 = float(row[r2c]) if r2c else np.nan
        rmse = float(row[rmsec]) if rmsec else np.nan
        mae = float(row[maec]) if maec else np.nan
        model = str(row[modelc]) if modelc else "unknown"
    else:
        # média simples das colunas disponíveis
        r2 = float(df[r2c].mean()) if r2c else np.nan
        rmse = float(df[rmsec].mean()) if rmsec else np.nan
        mae = float(df[maec].mean()) if maec else np.nan
        model = str(df[modelc].iloc[0]) if modelc and not df.empty else "unknown"

    return pd.DataFrame([{"model": model, "r2": r2, "rmse": rmse, "mae": mae}])

def tidy_model_name(m: str) -> str:
    m = (m or "").lower()
    if m in ("naive","naive-last"): return "Naïve"
    if m == "linear": return "Linear"
    if m == "ridge":  return "Ridge"
    if m == "gb":     return "GB"
    if m == "xgb":    return "XGB"
    return m.title() if m else "unknown"

def make_barplot(df_target: pd.DataFrame, metric: str, target_label: str, outpath: str):
    if metric not in df_target.columns or df_target.empty: return
    order = ["Naïve","Linear","Ridge","GB","XGB"]
    models_present = df_target["model_std"].tolist()
    models = [m for m in order if m in models_present] + [m for m in models_present if m not in order]
    data = [float(df_target.loc[df_target["model_std"] == m, metric].values[0]) if m in models_present else np.nan for m in models]
    plt.figure(figsize=(7,4))
    plt.bar(models, data)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} — {target_label} (média LODO)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(data):
        if not np.isnan(v):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def to_markdown_table(df: pd.DataFrame) -> str:
    cols = ["target","model_std","r2","rmse","mae","source_file","model_raw"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_markdown(index=False)

def main():
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, "exp01_metrics_*.csv")))
    if not files:
        print("[info] nenhum CSV encontrado em reports/exp01_metrics_*.csv")
        return

    rows = []
    for fpath in files:
        target, model_from_name = infer_target_model_from_filename(fpath)
        mean_df = read_metrics_file(fpath)
        if mean_df.empty: 
            continue
        row = mean_df.iloc[0].to_dict()
        row["target"] = target
        row["source_file"] = os.path.basename(fpath)
        # se o CSV não trouxe 'model', usa o inferido
        row["model_raw"] = row.get("model", model_from_name) or model_from_name
        # padroniza nome pra plot
        row["model_std"] = tidy_model_name(row["model_raw"])
        rows.append(row)

    if not rows:
        print("[info] não há dados para consolidar.")
        return

    summary = pd.DataFrame(rows)
    # remove targets mal inferidos
    summary = summary[summary["target"].str.upper() != "UNKNOWN"].copy()
    if summary.empty:
        print("[info] todos os arquivos ficaram como UNKNOWN; nada a consolidar.")
        return

    order = ["Naïve","Linear","Ridge","GB","XGB"]
    summary["model_order"] = summary["model_std"].apply(lambda x: order.index(x) if x in order else len(order))
    summary = summary.sort_values(["target","model_order"]).drop(columns=["model_order"])

    csv_out = os.path.join(OUT_DIR, "metrics_summary.csv")
    md_out  = os.path.join(OUT_DIR, "metrics_summary.md")
    summary.to_csv(csv_out, index=False)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Métricas LODO (linha __mean__) — Resumo por alvo e modelo\n\n")
        f.write(to_markdown_table(summary))
        f.write("\n")
    print(f"[ok] tabela-resumo salva em: {csv_out}")
    print(f"[ok] markdown salvo em:     {md_out}")

    for target in sorted(summary["target"].unique()):
        df_t = summary.loc[summary["target"] == target].copy()
        for metric in ["r2","rmse","mae"]:
            if metric in df_t.columns:
                outp = os.path.join(OUT_DIR, f"plot_{metric.upper()}_{target}.png")
                make_barplot(df_t, metric, target, outp)
                print(f"[ok] plot {metric.upper()} → {outp}")

    print("\n[feito] Veja os arquivos em reports/progress/")

if __name__ == "__main__":
    main()
