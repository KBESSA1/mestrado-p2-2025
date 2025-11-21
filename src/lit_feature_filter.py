# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

#!/usr/bin/env python3
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau

# ---------- utils ----------
def safe_div(a,b):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = a / b
        r[~np.isfinite(r)] = np.nan
        return r

def safe_nd(a,b):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (a - b) / (a + b)
        r[~np.isfinite(r)] = np.nan
        return r

def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    # garante colunas de bandas
    need = ["B3","B4","B5","B7","B8","B8A","B11","B12"]
    for n in need:
        if n not in df.columns:
            df[n] = np.nan
    B3 = pd.to_numeric(df["B3"], errors="coerce")
    B4 = pd.to_numeric(df["B4"], errors="coerce")
    B5 = pd.to_numeric(df["B5"], errors="coerce")
    B7 = pd.to_numeric(df["B7"], errors="coerce")
    B8 = pd.to_numeric(df["B8"], errors="coerce")
    B8A= pd.to_numeric(df["B8A"],errors="coerce")
    B11= pd.to_numeric(df["B11"],errors="coerce")
    B12= pd.to_numeric(df["B12"],errors="coerce")

    # --- CP-related ---
    df["NDRE"]    = safe_nd(B8A, B5)
    df["CIRE"]    = safe_div(B8A, B5) - 1
    df["MTCI"]    = safe_div((B8 - B5), (B5 - B4))
    MCARI         = ((B5 - B4) - 0.2*(B5 - B3)) * safe_div(B5, (B4 + 1e-9))
    OSAVI         = (1.16*(B8 - B4)) / (B8 + B4 + 0.16)
    df["MCARI2"]  = safe_div(MCARI, (OSAVI + 1e-9))
    df["RENDVI"]  = safe_nd(B8A, B4)
    df["CIgreen"] = safe_div(B8, B3) - 1
    df["NDVI"]    = safe_nd(B8, B4)
    df["EVI"]     = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B3 + 1e-9)
    df["SAVI"]    = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + 1e-9)
    df["GCI"]     = safe_div(B8, B3) - 1
    df["LAI"]     = np.nan  # se houver no CSV, mantemos; se não, fica NaN (não será escolhido)
    if "LAI" in df.columns: df["LAI"] = pd.to_numeric(df["LAI"], errors="coerce")
    df["PSRI"]    = safe_div((B4 - B3), B7)

    # --- TDN/ADF-related (umidade/estrutura) ---
    df["NDII"]    = safe_nd(B8, B11)
    df["NDMI"]    = safe_nd(B8, B11)          # alias
    df["NDWI_S2"] = safe_nd(B8, B11)          # NIR–SWIR
    df["GVMI"]    = safe_div((B8 + 0.1) - (B11 + 0.02), (B8 + 0.1) + (B11 + 0.02))
    df["MSI"]     = safe_div(B11, B8)
    df["NBR"]     = safe_nd(B8, B12)
    df["NBR2"]    = safe_nd(B11, B12)
    return df

def rank_in_family(df, candidates, target, expected_sign=None):
    ranks = []
    y = pd.to_numeric(df[target], errors="coerce")
    for f in candidates:
        if f not in df.columns: 
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 8:
            continue
        # Spearman (fallback Kendall se empatar/NaN)
        try:
            rho, _ = spearmanr(x[ok], y[ok])
        except Exception:
            rho = np.nan
        if not np.isfinite(rho):
            try:
                rho, _ = kendalltau(x[ok], y[ok])
            except Exception:
                rho = np.nan
        # aplica sinal esperado se definido
        if expected_sign == "+" and np.isfinite(rho) and rho < 0:
            score = -np.inf
        elif expected_sign == "-" and np.isfinite(rho) and rho > 0:
            score = -np.inf
        else:
            score = abs(rho) if np.isfinite(rho) else -np.inf
        ranks.append((f, score, rho))
    # ordena por score desc
    ranks.sort(key=lambda t: t[1], reverse=True)
    return ranks

def build_litset(df, prior, target):
    tp = prior["targets"][target]
    families = tp["families"]
    quota = tp["quota"]
    expected = tp.get("expected_sign", {})

    # calcula índices (idempotente)
    df = add_indices(df.copy())

    chosen = []
    for fam, feats in families.items():
        k = int(quota.get(fam, 0))
        if k <= 0: 
            continue
        fam_expected = {f: expected.get(f) for f in feats}
        ranks = rank_in_family(df, feats, target, None)  # vamos aplicar sinal por feature
        # re-rank aplicando sinal individual
        ranks2=[]
        for f,_,_ in ranks:
            s = fam_expected.get(f)
            rr = rank_in_family(df, [f], target, s)
            if rr:
                ranks2.append(rr[0])  # (f,score,rho)
        ranks2.sort(key=lambda t: t[1], reverse=True)
        chosen += [f for f,_,_ in ranks2[:k]]

    # remove duplicados e descarta os de score -inf
    chosen = [f for f in dict.fromkeys(chosen)]
    # bandas-core podem não ter sinal definido: já filtramos por |rho|
    cols_keep = ["Date", target] + [c for c in chosen if c in df.columns]
    # preserva IDs/coords se existirem
    for aux in ["SampleID","Sample","Sub-Sample","Latitude","Longitude","lat","lon"]:
        if aux in df.columns and aux not in cols_keep:
            cols_keep.append(aux)
    return df[cols_keep]

# ---------- cli ----------
ap = argparse.ArgumentParser()
ap.add_argument("--yaml", required=True)
ap.add_argument("--csv", required=True, help="CSV de entrada (baseline UFMS view: RAW/D5/D7)")
ap.add_argument("--target", required=True, choices=["CP","TDN_based_ADF"])
ap.add_argument("--out", required=True, help="CSV de saída com litset")
args = ap.parse_args()

with open(args.yaml,"r") as f:
    prior = yaml.safe_load(f)

df = pd.read_csv(args.csv, parse_dates=["Date"])
lit = build_litset(df, prior, args.target)
Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
lit.to_csv(args.out, index=False)
print(f"[ok] {args.target}: {args.csv} -> {args.out} ({len(lit.columns)} cols, {len(lit)} linhas)")
