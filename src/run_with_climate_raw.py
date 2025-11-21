# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

# -*- coding: utf-8 -*-
import re
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from utils_lodo import make_lodo_splits, compute_metrics

RAW = Path("/workspace/data/data_raw/Complete_DataSet.csv")
REPORTS = Path("/workspace/reports"); REPORTS.mkdir(parents=True, exist_ok=True)
PROGRESS = REPORTS / "progress"; PROGRESS.mkdir(parents=True, exist_ok=True)

# espectrais (paper-like) + detecção de clima
SPECTRAL = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12",
            "NDVI","NDWI","EVI","LAI","DVI","GCI","GEMI","SAVI"]
NOT_FEATS = {"Date","Satellite_Images_Dates","DeltaDays","lat","lon","Lat","Lon",
             "Latitude","Longitude","ID","Id","id","fid","FID","Unnamed: 0",
             "__index__","sample_id"}
TARGETS = {"CP","TDN_based_ADF","TDN_based_NDF"}
CLIMATE_RE = re.compile(r'(TEMP|T2M|TAVG|TMAX|TMIN|PREC|PRCP|RAIN|RR|CHIRPS|ERA5|RH|DEW|WIND|WS|WD|PRESS|SLP|VPD|EVAP|ET0|SOIL|SM|HUM)', re.I)

def pick_features(df: pd.DataFrame):
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    spectral = [c for c in SPECTRAL if c in df.columns]
    climate  = [c for c in numeric
                if c not in spectral and c not in TARGETS and c not in NOT_FEATS
                and CLIMATE_RE.search(c or "")]
    feats = spectral + climate
    return feats, spectral, climate

def run_one_target(df, date_col, target_col, tag_suffix):
    feats, spectral, climate = pick_features(df)
    assert target_col in df.columns, f"target {target_col} não encontrado"
    assert spectral, "não achei features espectrais"
    assert climate,  "não achei features de clima — verifique CSV"
    X = df[feats].copy()
    y = df[target_col].values

    print(f"[info] target={target_col} | feats: total={len(feats)} (spectral={len(spectral)}, climate={len(climate)})")

    models = {
        "linear": LinearRegression(),
        "ridge":  Ridge(alpha=1.0),
        "gb":     GradientBoostingRegressor(random_state=42),
        "xgb":    XGBRegressor(n_estimators=100, max_depth=3, eta=0.1,
                               subsample=1.0, colsample_bytree=1.0,
                               reg_lambda=1.0, random_state=42, n_jobs=-1),
    }

    rows = []
    for fold in make_lodo_splits(df, date_col):
        # aceita tanto objeto Fold quanto tupla
        if hasattr(fold, "train_idx") and hasattr(fold, "test_idx"):
            held_date = getattr(fold, "heldout_date", getattr(fold, "date", None))
            tr_idx, te_idx = fold.train_idx, fold.test_idx
        else:
            # fallback: (held_date, (train_idx, test_idx))
            held_date, (tr_idx, te_idx) = fold

        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        for mname, model in models.items():
            model.fit(Xtr_s, ytr)
            yhat = model.predict(Xte_s)
            r2, rmse, mae = compute_metrics(yte, yhat)
            rows.append(dict(heldout_date=held_date, model=mname, r2=r2, rmse=rmse, mae=mae))

    dfm = pd.DataFrame(rows)
    mean = dfm[["r2","rmse","mae"]].mean().to_dict()
    dfm = pd.concat([dfm, pd.DataFrame([{"heldout_date":"__mean__","model":"(mean)", **mean}])])

    out_csv = REPORTS / f"exp01_metrics_{tag_suffix}.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("# RUN=with_climate; DATASET=RAW; TDN_def=ADF\n")
        dfm.to_csv(f, index=False)
    return out_csv

def append_compare(out_csv: Path):
    cmp = PROGRESS / "metrics_compare.csv"
    if not cmp.exists():
        pd.DataFrame(columns=["dataset","target","model","r2","rmse","mae","source"]).to_csv(cmp, index=False)
    tmp = pd.read_csv(out_csv, comment="#")
    tmp = tmp[tmp["heldout_date"]=="__mean__"].copy()
    tmp = tmp.rename(columns={"model":"_model"})
    tmp["model"] = tmp["_model"].str.lower().map({"linear":"linear","ridge":"ridge","gb":"gb","xgb":"xgb"}).fillna(tmp["_model"])
    tmp["dataset"] = "RAW (com clima)"
    tmp["target"]  = "CP" if "CP_" in out_csv.name else "TDN"
    tmp["source"]  = out_csv.name
    tmp[["dataset","target","model","r2","rmse","mae","source"]].to_csv(cmp, mode="a", header=False, index=False)

def main():
    df = pd.read_csv(RAW, low_memory=False)
    out_cp  = run_one_target(df, "Date", "CP",            "CP_withclimate_RAW")
    out_tdn = run_one_target(df, "Date", "TDN_based_ADF", "TDN_withclimate_RAW")
    append_compare(out_cp)
    append_compare(out_tdn)
    print("[ok] RAW (com clima) → métricas:", out_cp.name, "e", out_tdn.name)
    print("[ok] acrescentei linhas em:", PROGRESS / "metrics_compare.csv")

if __name__ == "__main__":
    main()
