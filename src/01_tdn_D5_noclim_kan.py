# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de experimento LODO para TDN_based_ADF (digestibilidade estrutural), usando Kolmogorov-Arnold Network (KAN), janela Δ5 dias (D5), apenas bandas/índices do S2, sem clima. Faz parte do painel de baselines oficiais do mestrado (UFMS).
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
from kan import KAN  # pip install pykan

CSV_PATH = "/workspace/data/data_raw/Complete_DataSet.csv"
OUT_DIR = Path("/workspace/reports/baseline_experimento_UFMS")

TARGET_COL = "TDN_based_ADF"
CLIMATE_COLS = ["TEMP_MAX", "TEMP_MIN", "RAIN"]
DELTA_MAX_DAYS = 5  # D5

# Colunas que entregam TDN e NÃO podem entrar como input
LEAK_COLS = ["ADF", "TDN_based_NDF", "CP"]


def prepare_data():
    print(f"[INFO] Lendo CSV de {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    for col in ["Date", "Satellite_Images_Dates"]:
        if col not in df.columns:
            raise ValueError(f"Coluna de data '{col}' não encontrada no CSV.")
        df[col] = pd.to_datetime(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Date", "Satellite_Images_Dates", TARGET_COL]).copy()
    after = len(df)
    print(
        f"[INFO] Linhas após remover nulos em datas e alvo ({TARGET_COL}): "
        f"{after} (removidas {before - after})"
    )

    print("[INFO] Calculando Delta_days...")
    df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).dt.days.abs()

    df = df[df["Delta_days"] <= DELTA_MAX_DAYS].copy()
    print(f"[INFO] Nº de amostras após filtro D5: {len(df)}")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [
        c
        for c in num_cols
        if c != TARGET_COL and c not in CLIMATE_COLS and c not in LEAK_COLS
    ]

    print("[INFO] Ignorando colunas de clima (noclim):", CLIMATE_COLS)
    print("[INFO] Removendo colunas 'leak' do X:", LEAK_COLS)
    print(f"[INFO] Nº de features numéricas (sem clima, sem leak): {len(feature_cols)}")
    print("[INFO] Algumas features:", feature_cols[:10])

    unique_dates = sorted(df["Date"].dropna().unique())
    print(f"[INFO] Nº de datas (folds LODO): {len(unique_dates)}")

    return df, unique_dates, feature_cols


def run_lodo_kan():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device usado pelo KAN: {device}")

    df, unique_dates, feature_cols = prepare_data()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    all_true = []
    all_pred = []

    for fold_idx, date in enumerate(unique_dates, start=1):
        print(f"[FOLD {fold_idx:02d}] Date={date.date()}")

        test_mask = df["Date"] == date
        train_mask = ~test_mask

        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        X_train = df_train[feature_cols].to_numpy(dtype=np.float64)
        y_train = df_train[TARGET_COL].to_numpy(dtype=np.float64).reshape(-1, 1)
        X_test = df_test[feature_cols].to_numpy(dtype=np.float64)
        y_test = df_test[TARGET_COL].to_numpy(dtype=np.float64).reshape(-1, 1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_t = torch.from_numpy(X_train_scaled).to(device)
        y_train_t = torch.from_numpy(y_train).to(device)
        X_test_t = torch.from_numpy(X_test_scaled).to(device)
        y_test_t = torch.from_numpy(y_test).to(device)

        dataset = {
            "train_input": X_train_t,
            "train_label": y_train_t,
            "test_input": X_test_t,
            "test_label": y_test_t,
        }

        in_dim = X_train_t.shape[1]

        model = KAN(
            width=[in_dim, 64, 32, 1],
            grid=3,
            k=3,
            seed=42,
            device=device,
        )

        model.fit(dataset, opt="LBFGS", steps=50, lamb=1e-3)

        with torch.no_grad():
            y_pred_t = model(X_test_t)
        y_pred = y_pred_t.detach().cpu().numpy().reshape(-1)
        y_true = y_test.reshape(-1)

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        print(
            f"[FOLD {fold_idx:02d}] Date={date.date()} | "
            f"R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}"
        )

        metrics_rows.append(
            {
                "fold": fold_idx,
                "date": str(date.date()),
                "n_train": len(df_train),
                "n_test": len(df_test),
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    global_r2 = r2_score(all_true, all_pred)
    global_mse = mean_squared_error(all_true, all_pred)
    global_rmse = np.sqrt(global_mse)
    global_mae = mean_absolute_error(all_true, all_pred)

    print(
        f"\n[GLOBAL OOF - KAN_D5_TDN_NOCLIM_NOADF] "
        f"R2={global_r2:.3f} | RMSE={global_rmse:.3f} | MAE={global_mae:.3f}"
    )

    metrics_rows.append(
        {
            "fold": "GLOBAL_OOF",
            "date": "ALL",
            "n_train": int(df.shape[0]),
            "n_test": int(len(all_true)),
            "R2": global_r2,
            "RMSE": global_rmse,
            "MAE": global_mae,
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)
    oof_df = pd.DataFrame({"y_true": all_true, "y_pred": all_pred})

    metrics_path = OUT_DIR / "baseline_experimento_UFMS_D5_TDN_noclim_kan_metrics.csv"
    oof_path = OUT_DIR / "baseline_experimento_UFMS_D5_TDN_noclim_kan_oof.csv"

    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    print(f"[OK] Métricas por fold + GLOBAL (KAN D5 TDN noclim, sem ADF/TDN_NDF/CP) salvas em: {metrics_path}")
    print(f"[OK] Predições OOF (KAN D5 TDN noclim, sem ADF/TDN_NDF/CP) salvas em: {oof_path}")


if __name__ == "__main__":
    run_lodo_kan()
