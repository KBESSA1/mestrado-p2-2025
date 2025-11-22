# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de experimento LODO para CP (proteína bruta), usando XGBoost (árvore), janela Δ7 dias (D7), com clima acoplado ao S2. Faz parte do painel de baselines oficiais do mestrado (UFMS).
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

def main():
    csv_path = Path("/workspace/data/data_raw/Complete_DataSet.csv")
    out_dir = Path("/workspace/reports/baseline_experimento_UFMS")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Lendo CSV de {csv_path}")
    df = pd.read_csv(csv_path)

    # --- datas e Delta_days ---
    print("[INFO] Convertendo colunas de data...")
    for col in ["Date", "Satellite_Images_Dates"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print("[INFO] Calculando Delta_days...")
    df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).dt.days.abs()

    print("[INFO] Filtrando linhas com Delta_days <= 7...")
    df = df[df["Delta_days"] <= 7].copy()
    print(f"[INFO] Nº de amostras após filtro D7: {len(df)}")

    # --- target ---
    target_col = "CP"
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no CSV.")

    # --- detectar colunas de clima (heurística) ---
    climate_cols = [
        c for c in df.columns
        if ("ERA5" in c.upper())
        or ("CHIRPS" in c.upper())
        or ("RAIN" in c.upper())
        or ("TEMP" in c.upper())
        or ("TMAX" in c.upper())
        or ("TMIN" in c.upper())
    ]
    print(f"[INFO] Colunas de clima detectadas ({len(climate_cols)}): {climate_cols}")

    # --- seleção de features: todas numéricas + clima, exceto datas/ID/target ---
    drop_cols = {
        "Date",
        "Satellite_Images_Dates",
        "Delta_days",
        target_col,
    }
    drop_cols.update({"Sample", "Sub-Sample", "lat", "lon"})

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"[INFO] Nº de features numéricas (incluindo clima se houver): {len(feature_cols)}")
    print("[INFO] Algumas features:", feature_cols[:10])

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    dates = df["Date"].values

    unique_dates = np.unique(dates)
    print(f"[INFO] Nº de datas (folds LODO): {len(unique_dates)}")

    metrics_rows = []
    oof_records = []

    print("[INFO] Iniciando LODO por Date (XGB CLIM)...")
    for i, d in enumerate(unique_dates, start=1):
        mask_test = (dates == d)
        X_train, X_test = X[~mask_test], X[mask_test]
        y_train, y_test = y[~mask_test], y[mask_test]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_hat = model.predict(X_test)

        r2 = r2_score(y_test, y_hat)
        mse = mean_squared_error(y_test, y_hat)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_test, y_hat)

        d_python = pd.to_datetime(d).date()
        print(f"[FOLD {i:02d}] Date={d_python} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

        metrics_rows.append({
            "fold": i,
            "Date": d_python,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "n_test": int(mask_test.sum()),
        })

        for yt, yp in zip(y_test, y_hat):
            oof_records.append({
                "Date": d_python,
                "y_true": yt,
                "y_pred": yp,
            })

    # --- métricas globais OOF ---
    y_true_all = np.array([r["y_true"] for r in oof_records])
    y_pred_all = np.array([r["y_pred"] for r in oof_records])

    r2_global = r2_score(y_true_all, y_pred_all)
    mse_global = mean_squared_error(y_true_all, y_pred_all)
    rmse_global = sqrt(mse_global)
    mae_global = mean_absolute_error(y_true_all, y_pred_all)

    print(f"\n[GLOBAL OOF - XGB_CLIM] R2={r2_global:.3f} | RMSE={rmse_global:.3f} | MAE={mae_global:.3f}\n")

    metrics_rows.append({
        "fold": "GLOBAL",
        "Date": "GLOBAL",
        "R2": r2_global,
        "RMSE": rmse_global,
        "MAE": mae_global,
        "n_test": len(y_true_all),
    })

    metrics_path = out_dir / "baseline_experimento_UFMS_D7_CP_clim_xgb_metrics.csv"
    oof_path = out_dir / "baseline_experimento_UFMS_D7_CP_clim_xgb_oof.csv"

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(oof_records).to_csv(oof_path, index=False)

    print(f"[OK] Métricas por fold + GLOBAL (XGB CLIM) salvas em: {metrics_path}")
    print(f"[OK] Predições OOF (XGB CLIM) salvas em: {oof_path}")

if __name__ == "__main__":
    main()
