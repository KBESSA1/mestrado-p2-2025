# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de experimento LODO para TDN_based_ADF (digestibilidade estrutural), usando Regressão Linear, janela Δ7 dias (D7), com clima acoplado ao S2. Faz parte do painel de baselines oficiais do mestrado (UFMS).
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CSV_PATH = "/workspace/data/data_raw/Complete_DataSet.csv"
OUT_DIR = "/workspace/reports/baseline_experimento_UFMS"
TARGET_COL = "TDN_based_ADF"
DELTA_MAX = 7  # D7

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("[INFO] Lendo CSV de", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # --- conversão de datas ---
    for col in ["Date", "Satellite_Images_Dates"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            raise ValueError(f"Coluna de data '{col}' não encontrada no dataset.")

    # --- limpar linhas sem alvo ou datas ---
    before = len(df)
    df = df.dropna(subset=["Date", "Satellite_Images_Dates", TARGET_COL])
    after = len(df)
    print(f"[INFO] Linhas após remover nulos em datas e alvo ({TARGET_COL}): {after} (removidas {before - after})")

    # --- Delta_days ---
    df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).dt.days.abs()
    df_d7 = df[df["Delta_days"] <= DELTA_MAX].copy()
    print(f"[INFO] Nº de amostras após filtro D{DELTA_MAX}: {len(df_d7)}")

    # --- detectar colunas de clima (log) ---
    clima_cols = [c for c in df_d7.columns if c.upper() in ["TEMP_MAX", "TEMP_MIN", "RAIN"]]
    if clima_cols:
        print(f"[INFO] Colunas de clima detectadas ({len(clima_cols)}): {clima_cols}")
    else:
        print("[INFO] Nenhuma coluna de clima detectada explícita (TEMP_MAX, TEMP_MIN, RAIN).")

    # --- seleção de features numéricas ---
    numeric_cols = df_d7.select_dtypes(include=[np.number]).columns.tolist()
    # tiramos alvo e colunas que não devem ir como preditoras diretas
    drop_cols = {TARGET_COL, "Delta_days"}  # Delta_days é só para o filtro
    feature_cols = [c for c in numeric_cols if c not in drop_cols]

    print(f"[INFO] Nº de features numéricas (incluindo clima se houver): {len(feature_cols)}")
    print("[INFO] Algumas features:", feature_cols[:10])

    X_all = df_d7[feature_cols].astype(float).values
    y_all = df_d7[TARGET_COL].astype(float).values
    dates_all = df_d7["Date"].values

    unique_dates = np.unique(dates_all)
    print(f"[INFO] Nº de datas (folds LODO): {len(unique_dates)}")
    print("[INFO] Iniciando LODO por Date (Linear CLIM)...")

    rows_metrics = []
    y_true_all = []
    y_pred_all = []
    dates_oof = []

    for i, d in enumerate(sorted(unique_dates), start=1):
        mask_test = (dates_all == d)
        mask_train = ~mask_test

        X_train = X_all[mask_train]
        y_train = y_all[mask_train]
        X_test = X_all[mask_test]
        y_test = y_all[mask_test]

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"[WARN] Fold com data {d} sem treino ou teste suficiente; pulando.")
            continue

        # --- pré-processamento: padronização com parâmetros do treino ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- modelo linear ---
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_hat = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_hat)
        rmse = mean_squared_error(y_test, y_hat) ** 0.5
        mae = mean_absolute_error(y_test, y_hat)

        print(f"[FOLD {i:02d}] Date={pd.to_datetime(d).date()} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

        rows_metrics.append({
            "fold": i,
            "date": pd.to_datetime(d).date(),
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
        })

        y_true_all.append(y_test)
        y_pred_all.append(y_hat)
        dates_oof.append(dates_all[mask_test])

    if not rows_metrics:
        raise RuntimeError("Nenhum fold válido foi processado.")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    dates_oof = np.concatenate(dates_oof)

    r2_global = r2_score(y_true_all, y_pred_all)
    rmse_global = mean_squared_error(y_true_all, y_pred_all) ** 0.5
    mae_global = mean_absolute_error(y_true_all, y_pred_all)

    print(f"\n[GLOBAL OOF - LINEAR_CLIM] R2={r2_global:.3f} | RMSE={rmse_global:.3f} | MAE={mae_global:.3f}\n")

    # --- salvar métricas ---
    metrics_path = os.path.join(
        OUT_DIR,
        "baseline_experimento_UFMS_D7_TDN_clim_linear_metrics.csv",
    )
    metrics_df = pd.DataFrame(rows_metrics)
    metrics_df.loc[len(metrics_df)] = {
        "fold": "GLOBAL",
        "date": None,
        "R2": r2_global,
        "RMSE": rmse_global,
        "MAE": mae_global,
    }
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[OK] Métricas por fold + GLOBAL (Linear CLIM) salvas em: {metrics_path}")

    # --- salvar OOF ---
    oof_path = os.path.join(
        OUT_DIR,
        "baseline_experimento_UFMS_D7_TDN_clim_linear_oof.csv",
    )
    oof_df = pd.DataFrame({
        "Date": pd.to_datetime(dates_oof),
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    })
    oof_df.to_csv(oof_path, index=False)
    print(f"[OK] Predições OOF (Linear CLIM) salvas em: {oof_path}")

if __name__ == "__main__":
    main()
