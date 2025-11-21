# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de experimento LODO para TDN_based_ADF (digestibilidade estrutural), usando baseline ingênuo (persistência), janela Δ5 dias (D5), com clima acoplado ao S2. Faz parte do painel de baselines oficiais do mestrado (UFMS).
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CSV_PATH = "/workspace/data/data_raw/Complete_DataSet.csv"
OUT_DIR = Path("/workspace/reports/baseline_experimento_UFMS")

TARGET_COL = "TDN_based_ADF"
CLIMATE_COLS = ["TEMP_MAX", "TEMP_MIN", "RAIN"]
DELTA_MAX_DAYS = 5  # D5


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
    print(f"[INFO] Linhas após remover nulos em datas e alvo ({TARGET_COL}): {after} (removidas {before - after})")

    print("[INFO] Calculando Delta_days...")
    df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).abs().dt.days

    df = df[df["Delta_days"] <= DELTA_MAX_DAYS].copy()
    print(f"[INFO] Nº de amostras após filtro D5: {len(df)}")

    clima_presentes = [c for c in CLIMATE_COLS if c in df.columns]
    print(f"[INFO] Colunas de clima detectadas (D5 CLIM): {clima_presentes}")

    dates = df["Date"].values
    unique_dates = np.unique(dates)
    print(f"[INFO] Nº de datas (folds LODO): {len(unique_dates)}")

    return df, dates, unique_dates


def run_lodo():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, dates, unique_dates = prepare_data()

    metrics_rows = []
    oof_rows = []

    y_all = df[TARGET_COL].to_numpy(dtype=np.float32)

    for i, d in enumerate(unique_dates, start=1):
        train_mask = dates != d
        test_mask = ~train_mask

        y_train = y_all[train_mask]
        y_test = y_all[test_mask]

        y_hat = np.full_like(y_test, fill_value=float(y_train.mean()), dtype=np.float32)

        r2 = float(r2_score(y_test, y_hat))
        mse = mean_squared_error(y_test, y_hat)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_hat))

        date_str = pd.to_datetime(str(d)).date()
        print(f"[FOLD {i:02d}] Date={date_str} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

        metrics_rows.append(
            {
                "fold": i,
                "date": date_str,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "n_test": int(test_mask.sum()),
            }
        )

        fold_dates = df.loc[test_mask, "Date"].values
        for dt, yt, yp in zip(fold_dates, y_test, y_hat):
            oof_rows.append(
                {
                    "Date": pd.to_datetime(dt),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                }
            )

    y_true_all = np.array([r["y_true"] for r in oof_rows], dtype=float)
    y_pred_all = np.array([r["y_pred"] for r in oof_rows], dtype=float)

    r2_global = float(r2_score(y_true_all, y_pred_all))
    mse_global = mean_squared_error(y_true_all, y_pred_all)
    rmse_global = float(np.sqrt(mse_global))
    mae_global = float(mean_absolute_error(y_true_all, y_pred_all))

    print(
        f"\n[GLOBAL OOF - NAIVE_D5_TDN_CLIM] "
        f"R2={r2_global:.3f} | RMSE={rmse_global:.3f} | MAE={mae_global:.3f}\n"
    )

    metrics_rows.append(
        {
            "fold": "GLOBAL",
            "date": None,
            "r2": r2_global,
            "rmse": rmse_global,
            "mae": mae_global,
            "n_test": int(len(y_true_all)),
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)
    oof_df = pd.DataFrame(oof_rows)

    metrics_path = OUT_DIR / "baseline_experimento_UFMS_D5_TDN_clim_naive_metrics.csv"
    oof_path = OUT_DIR / "baseline_experimento_UFMS_D5_TDN_clim_naive_oof.csv"

    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    print(f"[OK] Métricas por fold + GLOBAL (Naive D5 TDN CLIM) salvas em: {metrics_path}")
    print(f"[OK] Predições OOF (Naive D5 TDN CLIM) salvas em: {oof_path}")


if __name__ == "__main__":
    run_lodo()
