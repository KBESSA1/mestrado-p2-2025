# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

#!/usr/bin/env python3
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CLIM_COLS = ["TEMP_MAX","TEMP_MIN","RAD_SOL","RAIN","WIND_SPD","EVAPOT","PRES_ATM","HUM_REL",
             "TP_SFC_AVG","Wind_Dir","Dew_Point","Radiative_Dif_AVG","Radiative_Direct_AVG",
             "PPFD","Longwave_Rad_AVG"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    df = pd.read_csv(args.csv)

    # normaliza datas (yyyy-mm-dd) e cria coluna de data de corte (heldout_date)
    df[args.date_col] = pd.to_datetime(df[args.date_col]).dt.strftime("%Y-%m-%d")
    heldout_dates = sorted(df[args.date_col].unique())

    # seleção de features: numéricas, excluindo chaves/targets
    exclude = set([args.date_col, args.target_col, "heldout_date", "fold", "SampleID"])
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    used_clim = [c for c in feat_cols if c in CLIM_COLS]
    print(f"using {len(feat_cols)} features (clima={len(used_clim)})")

    X_all = df[feat_cols].copy()
    y_all = df[args.target_col].astype(float).values

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=True),
        MLPRegressor(hidden_layer_sizes=(64, 32),
                     activation="relu",
                     alpha=1e-4,
                     learning_rate_init=1e-3,
                     max_iter=2000,
                     early_stopping=True,
                     random_state=42)
    )

    rows = []
    for d in heldout_dates:
        te_mask = (df[args.date_col] == d)
        tr_mask = ~te_mask
        if te_mask.sum() == 0 or tr_mask.sum() == 0:
            continue

        Xtr, ytr = X_all[tr_mask].values, y_all[tr_mask]
        Xte, yte = X_all[te_mask].values, y_all[te_mask]

        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)

        r2  = r2_score(yte, yhat) if len(np.unique(yte)) > 1 else np.nan
        rmse = np.sqrt(mean_squared_error(yte, yhat))
        mae  = mean_absolute_error(yte, yhat)

        rows.append({"heldout_date": d, "model": "mlp", "r2": r2, "rmse": rmse, "mae": mae})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print(f"ok -> {args.out} | feats={len(feat_cols)} | clima={len(used_clim)} | alvo={args.target_col}")

if __name__ == "__main__":
    main()
