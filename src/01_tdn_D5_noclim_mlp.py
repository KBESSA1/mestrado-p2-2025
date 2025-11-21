# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de experimento LODO para TDN_based_ADF (digestibilidade estrutural), usando MLP densa (rede neural), janela Δ5 dias (D5), apenas bandas/índices do S2, sem clima. Faz parte do painel de baselines oficiais do mestrado (UFMS).
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CSV_PATH = "/workspace/data/data_raw/Complete_DataSet.csv"
OUT_DIR = Path("/workspace/reports/baseline_experimento_UFMS")

TARGET_COL = "TDN_based_ADF"
CLIMATE_COLS = ["TEMP_MAX", "TEMP_MIN", "RAIN"]
DELTA_MAX_DAYS = 5  # D5

LEAK_COLS = ["ADF", "TDN_based_NDF", "CP"]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


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
    if clima_presentes:
        print(f"[INFO] Ignorando colunas de clima (noclim): {clima_presentes}")

    leak_presentes = [c for c in LEAK_COLS if c in df.columns]
    if leak_presentes:
        print(f"[INFO] Removendo colunas 'leak' do X (evitar cola de TDN): {leak_presentes}")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    drop_cols = [TARGET_COL, "Delta_days"]
    drop_cols += clima_presentes
    drop_cols += leak_presentes

    feature_cols = [c for c in num_cols if c not in drop_cols]

    print(f"[INFO] Nº de features numéricas (sem clima e sem leak): {len(feature_cols)}")
    print(f"[INFO] Algumas features: {feature_cols[:10]}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    dates = df["Date"].values
    unique_dates = np.unique(dates)
    print(f"[INFO] Nº de datas (folds LODO): {len(unique_dates)}")

    return df, X, y, dates, unique_dates, feature_cols


def run_lodo():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, dates, unique_dates, feature_cols = prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Treinando MLP em device={device}")

    metrics_rows = []
    oof_rows = []

    for i, d in enumerate(unique_dates, start=1):
        train_mask = dates != d
        test_mask = ~train_mask

        X_train_raw = X[train_mask]
        X_test_raw = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
        X_test_tensor = torch.from_numpy(X_test).float()

        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = MLP(input_dim=X_train.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        EPOCHS = 400

        model.train()
        for epoch in range(EPOCHS):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_hat_tensor = model(X_test_tensor.to(device)).cpu().squeeze(1)
        y_hat = y_hat_tensor.numpy()

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
        f"\n[GLOBAL OOF - MLP_D5_TDN_NOCLIM_NOADF] "
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

    metrics_path = OUT_DIR / "baseline_experimento_UFMS_D5_TDN_noclim_mlp_metrics.csv"
    oof_path = OUT_DIR / "baseline_experimento_UFMS_D5_TDN_noclim_mlp_oof.csv"

    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    print(f"[OK] Métricas por fold + GLOBAL (MLP D5 TDN noclim, sem ADF/TDN_NDF/CP) salvas em: {metrics_path}")
    print(f"[OK] Predições OOF (MLP D5 TDN noclim, sem ADF/TDN_NDF/CP) salvas em: {oof_path}")


if __name__ == "__main__":
    run_lodo()
