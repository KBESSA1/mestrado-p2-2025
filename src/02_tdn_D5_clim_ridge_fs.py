# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de Ridge Regression com seleção de features (FS) guiada por XGBoost. Usado para testar o ganho de FS em relação ao baseline linear.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from feature_config import prepare_xy, CSV_PATH_DEFAULT


OUT_DIR = Path("/workspace/reports/baseline_experimento_UFMS")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "TDN_based_ADF"
DELTA_MAX_DAYS = 5      # D5
USE_CLIMATE = True      # CLIM


def run_lodo_ridge():
    X, y, df = prepare_xy(
        csv_path=CSV_PATH_DEFAULT,
        target=TARGET_COL,
        delta_max_days=DELTA_MAX_DAYS,
        use_climate=USE_CLIMATE,
    )

    df = df.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    print(f"[INFO] Shape de X: {X.shape}, y: {y.shape}")

    unique_dates = sorted(df["Date"].unique())
    print(f"[INFO] Nº de datas (folds LODO): {len(unique_dates)}")

    metrics_rows = []
    oof_rows = []

    y_oof = np.zeros_like(y, dtype=float)
    seen_mask = np.zeros_like(y, dtype=bool)

    for i, d in enumerate(unique_dates, start=1):
        mask_test = df["Date"] == d
        mask_train = ~mask_test

        X_train = X.loc[mask_train].to_numpy()
        X_test = X.loc[mask_test].to_numpy()
        y_train = y[mask_train.to_numpy()]
        y_test = y[mask_test.to_numpy()]

        print(f"[FOLD {i:02d}] Date={d.date()} | n_train={len(y_train)} | n_test={len(y_test)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print(f"[FOLD {i:02d}] Date={d.date()} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

        metrics_rows.append(
            {
                "fold": i,
                "date": d,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

        idx_test = np.where(mask_test.to_numpy())[0]
        y_oof[idx_test] = y_pred
        seen_mask[idx_test] = True

        for idx, yt, yp in zip(idx_test, y_test, y_pred):
            oof_rows.append(
                {
                    "index": int(idx),
                    "date": df.loc[idx, "Date"],
                    "y_true": float(yt),
                    "y_pred": float(yp),
                }
            )

    if not np.all(seen_mask):
        raise RuntimeError("Nem todas as amostras receberam predição OOF.")

    r2_global = r2_score(y, y_oof)
    rmse_global = np.sqrt(mean_squared_error(y, y_oof))
    mae_global = mean_absolute_error(y, y_oof)

    print(
        f"\n[GLOBAL OOF - RIDGE_D{DELTA_MAX_DAYS}_{TARGET_COL}_CLIM_FS] "
        f"R2={r2_global:.3f} | RMSE={rmse_global:.3f} | MAE={mae_global:.3f}"
    )

    metrics_rows.append(
        {
            "fold": "GLOBAL_OOF",
            "date": None,
            "n_train": None,
            "n_test": len(y),
            "R2": r2_global,
            "RMSE": rmse_global,
            "MAE": mae_global,
        }
    )

    scenario_tag = f"D{DELTA_MAX_DAYS}_{TARGET_COL}_clim_ridge_fs"
    metrics_path = OUT_DIR / f"baseline_experimento_UFMS_{scenario_tag}_metrics.csv"
    oof_path = OUT_DIR / f"baseline_experimento_UFMS_{scenario_tag}_oof.csv"

    metrics_df = pd.DataFrame(metrics_rows)
    oof_df = pd.DataFrame(oof_rows)

    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    print(f"[OK] Métricas por fold + GLOBAL (Ridge {scenario_tag}) salvas em: {metrics_path}")
    print(f"[OK] Predições OOF (Ridge {scenario_tag}) salvas em: {oof_path}")


if __name__ == "__main__":
    run_lodo_ridge()
