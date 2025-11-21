# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de experimento LODO para CP (proteína bruta), usando XGBoost (árvore), janela Δ7 dias (D7), apenas bandas/índices do S2, sem clima. Faz parte do painel de baselines oficiais do mestrado (UFMS).
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

#!/usr/bin/env python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

DATA_PATH = Path("/workspace/data/data_raw/Complete_DataSet.csv")
OUT_DIR = Path("/workspace/reports/baseline_experimento_UFMS")

def main():
    print("[INFO] Lendo CSV de", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Converter colunas de data
    for col in ["Date", "Satellite_Images_Dates"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "Date" not in df.columns:
        raise SystemExit("Coluna 'Date' não encontrada no CSV.")
    if "Satellite_Images_Dates" not in df.columns:
        raise SystemExit("Coluna 'Satellite_Images_Dates' não encontrada no CSV.")

    print("[INFO] Convertendo colunas de data...")
    df = df.dropna(subset=["Date", "Satellite_Images_Dates"]).copy()

    # Delta_days
    print("[INFO] Calculando Delta_days...")
    df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).abs().dt.days

    # Filtro D7
    print("[INFO] Filtrando linhas com Delta_days <= 7...")
    df = df[df["Delta_days"] <= 7].copy()
    print(f"[INFO] Nº de amostras após filtro D7: {len(df)}")

    # Alvo CP
    if "CP" not in df.columns:
        raise SystemExit("Coluna alvo 'CP' não encontrada no CSV.")
    df = df.dropna(subset=["CP"]).copy()

    # Features numéricas (sem alvo e sem Delta_days)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["CP", "Delta_days"]:
        if col in num_cols:
            num_cols.remove(col)

    print(f"[INFO] Nº de features numéricas: {len(num_cols)}")
    print("[INFO] Algumas features:", num_cols[:10])

    X = df[num_cols].to_numpy(dtype=float)
    y = df["CP"].to_numpy(dtype=float)

    # LODO por Date
    dates = df["Date"].dt.normalize().unique()
    dates = np.sort(dates)

    print("[INFO] Iniciando LODO por Date (XGB)...")

    oof_true = []
    oof_pred = []
    oof_dates = []
    metrics_rows = []

    for i, d in enumerate(dates, start=1):
        mask_test = df["Date"].dt.normalize() == d
        X_train, X_test = X[~mask_test], X[mask_test]
        y_train, y_test = y[~mask_test], y[mask_test]

        model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        r2 = r2_score(y_test, y_hat)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
        mae = float(mean_absolute_error(y_test, y_hat))

        # converte numpy.datetime64 -> string de data legível
        d_str = str(pd.to_datetime(d).date())

        print(f"[FOLD {i:02d}] Date={d_str} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

        oof_true.append(y_test)
        oof_pred.append(y_hat)
        oof_dates.append(np.array([d_str] * len(y_test)))

        metrics_rows.append(
            {"fold": i, "Date": d_str, "R2": r2, "RMSE": rmse, "MAE": mae}
        )

    # Concat OOF
    y_true_all = np.concatenate(oof_true)
    y_pred_all = np.concatenate(oof_pred)
    dates_all = np.concatenate(oof_dates)

    r2_global = r2_score(y_true_all, y_pred_all)
    rmse_global = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    mae_global = float(mean_absolute_error(y_true_all, y_pred_all))

    print(f"\n[GLOBAL OOF - XGB] R2={r2_global:.3f} | RMSE={rmse_global:.3f} | MAE={mae_global:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.loc[len(metrics_df)] = {
        "fold": "GLOBAL",
        "Date": None,
        "R2": r2_global,
        "RMSE": rmse_global,
        "MAE": mae_global,
    }

    metrics_path = OUT_DIR / "baseline_experimento_UFMS_D7_CP_noclim_xgb_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[OK] Métricas por fold + GLOBAL (XGB) salvas em: {metrics_path}")

    oof_df = pd.DataFrame(
        {"Date": dates_all, "y_true": y_true_all, "y_pred": y_pred_all}
    )
    oof_path = OUT_DIR / "baseline_experimento_UFMS_D7_CP_noclim_xgb_oof.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[OK] Predições OOF (XGB) salvas em: {oof_path}")

if __name__ == "__main__":
    main()
